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

# agent_ppo.py
#
# PPO agent wrapper for physical Atari using Stable Baselines3
# This agent wraps SB3's PPO to work with the physical_atari harness

import os
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from framework.Logger import logger


class ManualStepEnv(gym.Env):
    """Minimal gym-like environment wrapper for manual stepping by harness"""

    def __init__(self, num_actions, obs_shape=(84, 84, 4)):
        super().__init__()
        self.num_actions = num_actions
        self.obs_shape = obs_shape

        # Proper Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )
        self.action_space = spaces.Discrete(num_actions)

        # Episode tracking
        self.current_obs = None
        self.current_reward = 0
        self.current_done = False
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def reset(self):
        """Reset called by SB3 - we'll return the last observation"""
        if self.current_obs is None:
            self.current_obs = np.zeros(self.obs_shape, dtype=np.uint8)
        return self.current_obs

    def step(self, action):
        """Step called by SB3 - returns buffered state from harness"""
        obs = self.current_obs if self.current_obs is not None else np.zeros(self.obs_shape, dtype=np.uint8)
        reward = self.current_reward
        done = self.current_done
        info = {}

        # Track episode stats
        self.current_episode_reward += reward
        self.current_episode_length += 1

        if done:
            info['episode'] = {
                'r': self.current_episode_reward,
                'l': self.current_episode_length
            }
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0

        # Reset for next call
        self.current_reward = 0
        self.current_done = False

        return obs, reward, done, info

    def update_state(self, obs, reward, done):
        """Called by agent to buffer the next state from harness"""
        self.current_obs = obs
        self.current_reward = reward
        self.current_done = done


class Agent:
    """PPO agent for physical Atari using Stable Baselines3"""

    def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
        # Configuration
        self.num_actions = num_actions
        self.total_frames = total_frames
        self.frame_skip = 4
        self.seed = seed
        self.data_dir = data_dir
        self.gpu = 0  # Default GPU

        # PPO hyperparameters (can be overridden via kwargs)
        self.learning_rate = 2.5e-4
        self.n_steps = 128
        self.batch_size = 256
        self.n_epochs = 4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.1
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = None

        # Model settings
        self.load_model_path = None
        self.use_grayscale = False  # If True, convert RGB to grayscale
        self.resize_to_84 = True   # If True, resize to 84x84 (standard Atari)

        # Override defaults with kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"agent_ppo: Set {key} = {value}")
            else:
                logger.warning(f"agent_ppo: Unknown parameter {key}")

        # Frame buffering (stack 4 frames)
        self.frame_buffer = deque(maxlen=4)
        self.step_count = 0
        self.last_action = 0

        # Determine observation shape based on settings
        # Standard Atari: (4, 84, 84) - 4 grayscale frames, channels-first
        if self.resize_to_84:
            height, width = 84, 84
        else:
            height, width = 210, 160

        # SB3 expects grayscale frames stacked: (4, H, W)
        # We'll convert RGB to grayscale during preprocessing
        obs_shape = (4, height, width)  # channels-first: (frames, height, width)

        logger.info(f"agent_ppo: Observation shape = {obs_shape}")
        logger.info(f"agent_ppo: Num actions = {num_actions}")

        # Create minimal environment wrapper for SB3
        self.env = ManualStepEnv(num_actions, obs_shape)
        self.vec_env = DummyVecEnv([lambda: self.env])

        # Device
        device = f"cuda:{self.gpu}" if torch.cuda.is_available() and self.gpu >= 0 else "cpu"
        logger.info(f"agent_ppo: Using device = {device}")

        # Create or load PPO model
        if self.load_model_path and os.path.exists(self.load_model_path):
            logger.info(f"agent_ppo: Loading model from {self.load_model_path}")
            self.ppo_model = PPO.load(
                self.load_model_path,
                env=self.vec_env,
                device=device
            )
        else:
            logger.info(f"agent_ppo: Creating new PPO model")
            self.ppo_model = PPO(
                "CnnPolicy",
                self.vec_env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                target_kl=self.target_kl,
                verbose=1,
                device=device,
                seed=seed
            )

        # Setup checkpoint saving
        self.checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=os.path.join(data_dir, "checkpoints"),
            name_prefix="ppo_model"
        )

        # Training tracking
        self.training_step = 0
        self.frames_since_train = 0

        logger.info(f"agent_ppo: Initialized successfully")

    def preprocess_frame(self, observation_rgb8):
        """Preprocess single frame: resize and convert to grayscale"""
        # observation_rgb8 is (160, 210, 3) from physical env

        # Always convert to grayscale for standard Atari preprocessing
        # (SB3's CnnPolicy expects grayscale frames)
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

        # Preprocess and buffer frame
        processed_frame = self.preprocess_frame(observation_rgb8)
        self.frame_buffer.append(processed_frame)

        # Only act every frame_skip frames
        if self.step_count % self.frame_skip != 0:
            return self.last_action

        # Wait until we have 4 frames stacked
        if len(self.frame_buffer) < 4:
            return self.last_action

        # Stack 4 grayscale frames: [(H, W), ...] -> (4, H, W)
        stacked_frames = np.stack(list(self.frame_buffer), axis=0)  # (4, H, W)

        # Update environment state for PPO
        self.env.update_state(stacked_frames, reward, end_of_episode)

        # Get action from PPO model
        action, _states = self.ppo_model.predict(stacked_frames, deterministic=False)

        # Convert from numpy array to int
        # action can be a scalar or 0-d array
        action = int(np.asarray(action).item())

        self.last_action = action
        self.frames_since_train += 1

        # Periodic training step (every n_steps)
        if self.frames_since_train >= self.n_steps:
            self.training_step += 1
            self.frames_since_train = 0

            # Train for one iteration
            try:
                self.ppo_model.learn(
                    total_timesteps=self.n_steps,
                    reset_num_timesteps=False,
                    callback=self.checkpoint_callback
                )
            except Exception as e:
                logger.error(f"agent_ppo: Training error: {e}")

        return action

    def save_model(self, filename):
        """Save PPO model to disk"""
        try:
            self.ppo_model.save(filename)
            logger.info(f"agent_ppo: Model saved to {filename}")
        except Exception as e:
            logger.error(f"agent_ppo: Error saving model: {e}")
