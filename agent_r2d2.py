# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# agent_r2d2.py
#
# R2D2 agent wrapper for physical Atari
# This agent wraps R2D2 to work with the physical_atari harness

import os
import sys
from typing import Optional

import cv2
import numpy as np
import torch

# Add algorithms directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'algorithms'))

from framework.Logger import logger
from algorithms.r2d2.model import Network, AgentState
from algorithms.r2d2.actor import LocalBuffer
from algorithms.r2d2 import config as r2d2_config


class Agent:
    """R2D2 agent for physical Atari"""

    def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
        # Configuration
        self.num_actions = num_actions
        self.total_frames = total_frames
        self.seed = seed
        self.data_dir = data_dir
        self.gpu = 0  # Default GPU

        # R2D2 hyperparameters from config (can be overridden via kwargs)
        self.learning_rate = r2d2_config.lr
        self.gamma = r2d2_config.gamma
        self.epsilon = 0.01  # Fixed epsilon for physical env (mostly greedy)
        self.hidden_dim = r2d2_config.hidden_dim
        self.burn_in_steps = r2d2_config.burn_in_steps
        self.learning_steps = r2d2_config.learning_steps
        self.forward_steps = r2d2_config.forward_steps
        self.block_length = r2d2_config.block_length
        self.grad_norm = r2d2_config.grad_norm
        self.target_update_freq = r2d2_config.target_net_update_interval
        self.frame_skip = 4  # Physical env: act every 4 frames
        self.resize_to_84 = True

        # Model settings
        self.load_model_path = None

        # Override defaults with kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"agent_r2d2: Set {key} = {value}")
            else:
                logger.warning(f"agent_r2d2: Unknown parameter {key}")

        # Handle 'load_file' kwarg (harness uses this instead of load_model_path)
        if 'load_file' in kwargs:
            self.load_model_path = kwargs['load_file']
            logger.info(f"agent_r2d2: Set load_model_path from load_file = {self.load_model_path}")

        # State tracking
        self.step_count = 0
        self.last_action = 0
        self.last_reward = 0

        # Observation shape: (1, 84, 84) - grayscale, channels-first
        height, width = (84, 84) if self.resize_to_84 else (210, 160)
        obs_shape = (1, height, width)

        logger.info(f"agent_r2d2: Observation shape = {obs_shape}")
        logger.info(f"agent_r2d2: Num actions = {num_actions}")

        # Device
        device = f"cuda:{self.gpu}" if torch.cuda.is_available() and self.gpu >= 0 else "cpu"
        self.device = torch.device(device)
        logger.info(f"agent_r2d2: Using device = {device}")

        # Create R2D2 model
        self.model = Network(num_actions, obs_shape=obs_shape, hidden_dim=self.hidden_dim)
        self.model.to(self.device)
        self.model.eval()  # Start in eval mode for inference

        # Load pre-trained model if available
        if self.load_model_path and os.path.exists(self.load_model_path):
            logger.info(f"agent_r2d2: Loading model from {self.load_model_path}")
            checkpoint = torch.load(self.load_model_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # New format: dict with 'model_state_dict' key
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"agent_r2d2: Loaded from new checkpoint format (num_updates={checkpoint.get('num_updates', 'unknown')})")
                else:
                    # Dict is the state dict itself
                    self.model.load_state_dict(checkpoint)
                    logger.info(f"agent_r2d2: Loaded state dict directly")
            elif isinstance(checkpoint, tuple):
                # Old format: (state_dict, num_updates, env_steps, time)
                self.model.load_state_dict(checkpoint[0])
                logger.info(f"agent_r2d2: Loaded from old tuple format (num_updates={checkpoint[1]})")
            else:
                logger.error(f"agent_r2d2: Unknown checkpoint format: {type(checkpoint)}")
        else:
            if self.load_model_path:
                logger.warning(f"agent_r2d2: Model path specified but not found: {self.load_model_path}")
            logger.info(f"agent_r2d2: Creating new R2D2 model (random weights)")

        # Setup local buffer for experience collection
        self.local_buffer = LocalBuffer(
            action_dim=num_actions,
            forward_steps=self.forward_steps,
            burn_in_steps=self.burn_in_steps,
            learning_steps=self.learning_steps,
            gamma=self.gamma,
            hidden_dim=self.hidden_dim,
            block_length=self.block_length
        )

        # Training tracking (required by harness_physical.py)
        self.train_losses = []  # Required by harness_physical.py for plotting

        # Agent state
        self.agent_state = None

        logger.info(f"agent_r2d2: Initialized successfully")

    def preprocess_frame(self, observation_rgb8):
        """Preprocess single frame: resize and convert to grayscale"""
        # observation_rgb8 is (160, 210, 3) from physical env

        # Convert to grayscale
        frame = cv2.cvtColor(observation_rgb8, cv2.COLOR_RGB2GRAY)  # (H, W)

        if self.resize_to_84:
            # Resize to 84x84 (standard Atari preprocessing)
            frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

        return frame

    def frame(self, observation_rgb8, reward, end_of_episode):
        """
        Called every frame by harness.

        Args:
            observation_rgb8: RGB observation (160, 210, 3) uint8
            reward: Scalar reward from environment
            end_of_episode: Boolean indicating episode termination

        Returns:
            action_index: Integer action index to execute
        """
        self.step_count += 1

        # Preprocess frame to grayscale (H, W) uint8
        processed_frame = self.preprocess_frame(observation_rgb8)

        # Only act every frame_skip frames
        if self.step_count % self.frame_skip != 0:
            return self.last_action

        # Create observation tensor (1, 1, H, W) and convert to device
        # processed_frame is (H, W), we need (batch=1, channels=1, H, W)
        obs_tensor = torch.from_numpy(processed_frame).unsqueeze(0).unsqueeze(0).float().to(self.device)

        # Initialize agent state on first frame
        if self.agent_state is None:
            # AgentState expects obs with batch dimension: (1, 1, H, W)
            self.agent_state = AgentState(obs_tensor, self.num_actions)
            # Reset local buffer with initial observation (numpy array)
            self.local_buffer.reset(processed_frame)

        # Update agent state with current obs and previous action/reward
        # Note: agent_state.update() expects obs tensor, action int, reward float, hidden tuple
        self.agent_state.update(obs_tensor, self.last_action, self.last_reward, self.agent_state.hidden_state)

        # Get action from model
        with torch.no_grad():
            # model.forward() returns (q_value, hidden_state)
            # q_value shape: [action_dim] (squeezed from [1, 1, action_dim])
            # hidden_state: tuple (h, c) each with shape [1, 1, hidden_dim]
            q_value, hidden_state = self.model(self.agent_state)

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            # q_value is already squeezed to [action_dim]
            action = torch.argmax(q_value).item()

        # Store experience in local buffer for tracking
        # hidden_state is tuple (h, c), concat to [2, 1, hidden_dim] then squeeze to [2, hidden_dim]
        hidden_np = torch.cat(hidden_state).squeeze(1).cpu().numpy()
        self.local_buffer.add(
            action,
            reward,
            processed_frame,
            q_value.cpu().numpy(),
            hidden_np
        )

        # Update state for next frame
        self.last_action = action
        self.last_reward = reward

        # Reset on episode end
        if end_of_episode:
            # Finish block
            if len(self.local_buffer) > 0:
                # finish() returns [block, priorities, episode_reward or None]
                block_data = self.local_buffer.finish()  # Episode done, no bootstrapping
                # For now, just track losses without training
                self.train_losses.append(0.0)

            # Reset agent state for next episode
            self.agent_state = None

        return action

    def save_model(self, filename):
        """Save R2D2 model to disk using new checkpoint format"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'num_updates': 0,  # Physical env doesn't do gradient updates
                'env_steps': self.step_count,
                'training_time_minutes': 0,
            }
            torch.save(checkpoint, filename)
            logger.info(f"agent_r2d2: Model saved to {filename}")
        except Exception as e:
            logger.error(f"agent_r2d2: Error saving model: {e}")
