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
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch

from framework.Logger import logger
from r2d2.model import Network, AgentState
from r2d2.actor import LocalBuffer, calculate_mixed_td_errors
from r2d2.learner import Learner


class Agent:
    """R2D2 agent for physical Atari"""

    def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
        # Configuration
        self.num_actions = num_actions
        self.total_frames = total_frames
        self.frame_skip = 4
        self.seed = seed
        self.data_dir = data_dir
        self.gpu = 0  # Default GPU

        # R2D2 hyperparameters (can be overridden via kwargs)
        self.learning_rate = 1e-4
        self.gamma = 0.997
        self.epsilon = 0.01  # Fixed epsilon for physical env (mostly greedy)
        self.hidden_dim = 512
        self.burn_in_steps = 40
        self.learning_steps = 80
        self.forward_steps = 5
        self.block_length = 120
        self.grad_norm = 40
        self.target_update_freq = 2500
        self.batch_size = 1  # Physical env: online learning

        # Model settings
        self.load_model_path = None
        self.resize_to_84 = True

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

        # Frame buffering (maintain history for LSTM)
        self.frame_history = deque(maxlen=self.burn_in_steps + self.learning_steps + self.forward_steps)
        self.step_count = 0
        self.last_action = 0

        # LSTM hidden state
        self.hidden_state = None

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
            state_dict = torch.load(self.load_model_path, map_location=self.device)
            # Handle different save formats
            if isinstance(state_dict, tuple):
                state_dict = state_dict[0]  # (state_dict, num_updates, env_steps, time)
            self.model.load_state_dict(state_dict)
        else:
            logger.info(f"agent_r2d2: Creating new R2D2 model")

        # Setup local buffer for experience collection
        self.local_buffer = LocalBuffer(
            num_actions,
            forward_steps=self.forward_steps,
            burn_in_steps=self.burn_in_steps,
            learning_steps=self.learning_steps,
            gamma=self.gamma,
            hidden_dim=self.hidden_dim,
            block_length=self.block_length
        )

        # Setup training (optional - for online learning)
        self.target_model = Network(num_actions, obs_shape=obs_shape, hidden_dim=self.hidden_dim)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1.5e-4)
        self.loss_fn = torch.nn.MSELoss(reduction='none')

        # Training tracking
        self.training_step = 0
        self.frames_since_train = 0
        self.train_losses = []  # Required by harness_physical.py
        self.experience_buffer = []  # Store experiences for batch training

        # Agent state
        self.agent_state = None
        self.last_reward = 0

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

        # Preprocess frame
        processed_frame = self.preprocess_frame(observation_rgb8)

        # Only act every frame_skip frames
        if self.step_count % self.frame_skip != 0:
            return self.last_action

        # Add to frame history
        self.frame_history.append(processed_frame)

        # Need at least one frame to act
        if len(self.frame_history) < 1:
            return self.last_action

        # Create observation tensor (1, 84, 84)
        obs = torch.from_numpy(processed_frame).unsqueeze(0).unsqueeze(0).float().to(self.device)

        # Initialize agent state if needed
        if self.agent_state is None:
            self.agent_state = AgentState(obs, self.num_actions)

        # Update agent state with previous action and reward
        self.agent_state.update(obs, self.last_action, self.last_reward, self.hidden_state)

        # Get action from model
        with torch.no_grad():
            q_value, hidden = self.model(self.agent_state)

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = torch.argmax(q_value, 0).item()

        # Update state
        self.last_action = action
        self.last_reward = reward
        self.hidden_state = hidden

        # Store experience in local buffer (for potential online training)
        if len(self.frame_history) > 1:  # Need previous frame
            prev_frame = self.frame_history[-2]
            self.local_buffer.add(
                self.last_action,
                reward,
                processed_frame,
                q_value.cpu().numpy(),
                torch.cat(hidden).cpu().numpy()
            )

        # Reset on episode end
        if end_of_episode:
            self.hidden_state = None
            self.agent_state = None

            # Optionally perform training update
            if len(self.local_buffer) > 0:
                # Finish block and extract training data
                block_data = self.local_buffer.finish()
                # For now, just track losses without training
                # (online training can be added later)
                self.train_losses.append(0.0)

        return action

    def save_model(self, filename):
        """Save R2D2 model to disk"""
        try:
            torch.save(
                (self.model.state_dict(), self.training_step, self.step_count, 0),
                filename
            )
            logger.info(f"agent_r2d2: Model saved to {filename}")
        except Exception as e:
            logger.error(f"agent_r2d2: Error saving model: {e}")
