import os

import cv2
import numpy as np
import torch

from framework.Logger import logger
from .model import Network, AgentState
from .actor import LocalBuffer
from . import config as r2d2_config


class Agent:
    def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
        self.num_actions = num_actions
        self.total_frames = total_frames
        self.seed = seed
        self.data_dir = data_dir
        self.gpu = 0

        self.learning_rate = r2d2_config.lr
        self.gamma = r2d2_config.gamma
        self.epsilon = 0.01
        self.hidden_dim = r2d2_config.hidden_dim
        self.burn_in_steps = r2d2_config.burn_in_steps
        self.learning_steps = r2d2_config.learning_steps
        self.forward_steps = r2d2_config.forward_steps
        self.block_length = r2d2_config.block_length
        self.grad_norm = r2d2_config.grad_norm
        self.target_update_freq = r2d2_config.target_net_update_interval
        self.frame_skip = 4
        self.resize_to_84 = True

        self.load_model_path = None

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"agent_r2d2: Set {key} = {value}")
            else:
                logger.warning(f"agent_r2d2: Unknown parameter {key}")

        if 'load_file' in kwargs:
            self.load_model_path = kwargs['load_file']
            logger.info(f"agent_r2d2: Set load_model_path from load_file = {self.load_model_path}")

        self.step_count = 0
        self.last_action = 0
        self.last_reward = 0

        height, width = (84, 84) if self.resize_to_84 else (210, 160)
        obs_shape = (1, height, width)

        logger.info(f"agent_r2d2: Observation shape = {obs_shape}")
        logger.info(f"agent_r2d2: Num actions = {num_actions}")

        device = f"cuda:{self.gpu}" if torch.cuda.is_available() and self.gpu >= 0 else "cpu"
        self.device = torch.device(device)
        logger.info(f"agent_r2d2: Using device = {device}")

        self.model = Network(num_actions, obs_shape=obs_shape, hidden_dim=self.hidden_dim)
        self.model.to(self.device)
        self.model.eval()

        if self.load_model_path and os.path.exists(self.load_model_path):
            logger.info(f"agent_r2d2: Loading model from {self.load_model_path}")
            checkpoint = torch.load(self.load_model_path, map_location=self.device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"agent_r2d2: Loaded checkpoint (num_updates={checkpoint.get('num_updates', 0)})")
            else:
                logger.error(f"agent_r2d2: Unsupported checkpoint format. Expected dict with 'model_state_dict' key.")
        else:
            if self.load_model_path:
                logger.warning(f"agent_r2d2: Model path specified but not found: {self.load_model_path}")
            logger.info(f"agent_r2d2: Creating new R2D2 model (random weights)")

        self.local_buffer = LocalBuffer(
            action_dim=num_actions,
            forward_steps=self.forward_steps,
            burn_in_steps=self.burn_in_steps,
            learning_steps=self.learning_steps,
            gamma=self.gamma,
            hidden_dim=self.hidden_dim,
            block_length=self.block_length
        )

        self.train_losses = []
        self.agent_state = None

        logger.info(f"agent_r2d2: Initialized successfully")

    def preprocess_frame(self, observation_rgb8):
        frame = cv2.cvtColor(observation_rgb8, cv2.COLOR_RGB2GRAY)

        if self.resize_to_84:
            frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

        return frame

    def frame(self, observation_rgb8, reward, end_of_episode):
        self.step_count += 1
        processed_frame = self.preprocess_frame(observation_rgb8)

        if self.step_count % self.frame_skip != 0:
            return self.last_action

        obs_tensor = torch.from_numpy(processed_frame).unsqueeze(0).unsqueeze(0).float().to(self.device)

        if self.agent_state is None:
            self.agent_state = AgentState(obs_tensor, self.num_actions)
            self.local_buffer.reset(processed_frame)

        self.agent_state.update(obs_tensor, self.last_action, self.last_reward, self.agent_state.hidden_state)

        with torch.no_grad():
            q_value, hidden_state = self.model(self.agent_state)

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = torch.argmax(q_value).item()

        hidden_np = torch.cat(hidden_state).squeeze(1).cpu().numpy()
        self.local_buffer.add(
            action,
            reward,
            processed_frame,
            q_value.cpu().numpy(),
            hidden_np
        )

        self.last_action = action
        self.last_reward = reward

        if end_of_episode:
            if len(self.local_buffer) > 0:
                block_data = self.local_buffer.finish()
                self.train_losses.append(0.0)
            self.agent_state = None

        return action

    def save_model(self, filename):
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'num_updates': 0,
                'env_steps': self.step_count,
                'training_time_minutes': 0,
            }
            torch.save(checkpoint, filename)
            logger.info(f"agent_r2d2: Model saved to {filename}")
        except Exception as e:
            logger.error(f"agent_r2d2: Error saving model: {e}")