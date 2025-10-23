import os
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity: int, stack_size: int, obs_shape: tuple[int, int], seed: int):
        self.capacity = capacity
        self.stack_size = stack_size
        self.obs_shape = obs_shape

        self.states = np.zeros((capacity, stack_size, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, stack_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.rng = np.random.default_rng(seed)
        self.ptr = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.capacity if self.full else self.ptr

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        assert state.shape == (self.stack_size, *self.obs_shape)
        assert next_state.shape == (self.stack_size, *self.obs_shape)

        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int, device: torch.device):
        assert self.size >= batch_size, "requested batch larger than buffer"
        indices = self.rng.integers(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[indices]).to(device=device, dtype=torch.float32) / 255.0
        next_states = torch.from_numpy(self.next_states[indices]).to(device=device, dtype=torch.float32) / 255.0
        actions = torch.from_numpy(self.actions[indices]).to(device=device, dtype=torch.int64)
        rewards = torch.from_numpy(self.rewards[indices]).to(device=device, dtype=torch.float32)
        dones = torch.from_numpy(self.dones[indices].astype(np.float32)).to(device=device, dtype=torch.float32)

        return states, actions, rewards, next_states, dones


class QNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int, height: int = 84, width: int = 84):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            conv_out = self._forward_features(dummy)
            feature_dim = conv_out.shape[1]
        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Agent:
    """
    Minimal DQN agent compatible with harness_physical.py.
    Handles observation preprocessing, replay buffering, and online updates.
    """

    def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
        # Hyperparameters (override via kwargs from harness CLI)
        self.stack_size = kwargs.get('stack_size', 4)
        self.obs_height = kwargs.get('obs_height', 84)
        self.obs_width = kwargs.get('obs_width', 84)
        self.gamma = kwargs.get('gamma', 0.99)
        self.learning_rate = kwargs.get('learning_rate', 2.5e-4)
        self.grad_clip = kwargs.get('grad_clip', 10.0)

        self.buffer_size = kwargs.get('buffer_size', 100_000)
        self.batch_size = kwargs.get('batch_size', 32)
        self.train_start = kwargs.get('train_start', 50_000)
        self.train_freq = kwargs.get('train_freq', 4)
        self.target_update_freq = kwargs.get('target_update_freq', 10_000)

        self.epsilon_start = kwargs.get('epsilon_start', 1.0)
        self.epsilon_end = kwargs.get('epsilon_end', 0.1)
        self.epsilon_decay_frames = kwargs.get('epsilon_decay_frames', 1_000_000)
        self.epsilon = self.epsilon_start

        self.frame_skip = kwargs.get('frame_skip', 1)
        self.device = self._init_device(kwargs.get('gpu', 0))

        self.num_actions = num_actions
        self.total_frames = total_frames
        self.data_dir = data_dir

        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

        self.q_network = QNetwork(self.stack_size, num_actions, self.obs_height, self.obs_width).to(self.device)
        self.q_target = QNetwork(self.stack_size, num_actions, self.obs_height, self.obs_width).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate, eps=1e-5)

        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_size,
            stack_size=self.stack_size,
            obs_shape=(self.obs_height, self.obs_width),
            seed=seed,
        )

        self.state_stack: Optional[Deque[np.ndarray]] = None
        self.last_state: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None

        self.frame_count = 0
        self.training_steps = 0
        self.loss_ema = None
        self.last_loss = 0.0
        self.last_avg_q = 0.0
        self.last_max_q = 0.0

        self.train_losses = []

        load_file = kwargs.get('load_file')
        if load_file is not None and os.path.exists(load_file):
            self.q_network.load_state_dict(torch.load(load_file, map_location=self.device))
            self.q_target.load_state_dict(self.q_network.state_dict())

    @staticmethod
    def _init_device(gpu_index: int) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f'cuda:{gpu_index}')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.obs_width, self.obs_height), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    def _ensure_state_stack(self, processed_frame: np.ndarray):
        if self.state_stack is None:
            self.state_stack = deque([processed_frame] * self.stack_size, maxlen=self.stack_size)
        else:
            self.state_stack.append(processed_frame)

    def _stack_frames(self) -> np.ndarray:
        assert self.state_stack is not None
        return np.stack(self.state_stack, axis=0)

    def _epsilon_by_frame(self, frame_idx: int) -> float:
        decay_fraction = min(frame_idx / float(self.epsilon_decay_frames), 1.0)
        epsilon = self.epsilon_start + decay_fraction * (self.epsilon_end - self.epsilon_start)
        return max(self.epsilon_end, epsilon)

    def _select_action(self, state: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))

        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
            self.last_avg_q = float(q_values.mean().item())
            self.last_max_q = float(q_values.max().item())
        return action

    def _train_step(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.q_target(next_states).max(1)[0]
            targets = rewards + self.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        self.optimizer.step()

        self.last_loss = float(loss.item())
        if self.loss_ema is None:
            self.loss_ema = self.last_loss
        else:
            self.loss_ema = 0.95 * self.loss_ema + 0.05 * self.last_loss
        self.train_losses.append(self.last_loss)

        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())

    def frame(self, observation_rgb8, reward, end_of_episode):
        processed = self._preprocess(observation_rgb8)
        self._ensure_state_stack(processed)
        current_state = self._stack_frames()

        done = end_of_episode > 0

        if self.last_state is not None and self.last_action is not None:
            self.replay_buffer.add(self.last_state, self.last_action, reward, current_state, done)
            if self.replay_buffer.size >= self.train_start and (self.frame_count % self.train_freq == 0):
                self._train_step()

        if done:
            # reset stack so the next state starts fresh
            self.state_stack = deque([processed] * self.stack_size, maxlen=self.stack_size)
            current_state = self._stack_frames()

        self.frame_count += 1
        self.epsilon = self._epsilon_by_frame(self.frame_count)

        action = self._select_action(current_state)
        self.last_state = current_state.copy()
        self.last_action = action

        return action

    def save_model(self, filename):
        torch.save(self.q_network.state_dict(), filename)
