from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from utils.agent_utils import preprocess_batch
from utils.vector_agents import VectorAgent


class QNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity: int, stack_size: int, obs_shape: tuple[int, int]):
        self.capacity = capacity
        self.stack_size = stack_size
        self.obs_shape = obs_shape
        self.states = np.zeros((capacity, stack_size, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, stack_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.int64)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.ptr = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.capacity if self.full else self.ptr

    def add(self, state, action, reward, next_state, done) -> None:
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        states = torch.from_numpy(self.states[idx]).to(device, dtype=torch.float32) / 255.0
        actions = torch.from_numpy(self.actions[idx]).to(device, dtype=torch.int64)
        rewards = torch.from_numpy(self.rewards[idx]).to(device, dtype=torch.float32)
        next_states = torch.from_numpy(self.next_states[idx]).to(device, dtype=torch.float32) / 255.0
        dones = torch.from_numpy(self.dones[idx].astype(np.float32)).to(device, dtype=torch.float32)
        return states, actions, rewards, next_states, dones


class DQNCore:
    def __init__(
        self,
        num_envs: int,
        seed: int,
        num_actions: int,
        total_frames: int,
        *,
        stack_size: int = 4,
        obs_height: int = 84,
        obs_width: int = 84,
        buffer_size: int = 100_000,
        batch_size: int = 32,
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
        train_start: int = 50_000,
        train_freq: int = 4,
        target_update_freq: int = 10_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_frames: int = 1_000_000,
        frame_skip: int = 1,
        grad_clip: Optional[float] = 10.0,
        data_dir: Optional[str] = None,
        load_file: Optional[str] = None,
        gpu: int = 0,
    ):
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.total_frames = total_frames
        self.stack_size = stack_size
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.train_start = train_start
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_frames = epsilon_decay_frames
        self.frame_skip = frame_skip
        self.grad_clip = grad_clip

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu}')
            torch.cuda.manual_seed_all(seed)
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        np.random.seed(seed)

        self.network = QNetwork(in_channels=stack_size, num_actions=num_actions).to(self.device)
        self.target_network = QNetwork(in_channels=stack_size, num_actions=num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.replay = ReplayBuffer(buffer_size, stack_size=stack_size, obs_shape=(obs_height, obs_width))

        self.state_stacks = np.zeros((num_envs, stack_size, obs_height, obs_width), dtype=np.uint8)
        self.last_states = np.zeros_like(self.state_stacks)
        self.last_actions = np.full((num_envs,), -1, dtype=np.int64)

        self.frame_count = 0
        self.training_steps = 0
        self.epsilon = epsilon_start

        self.last_loss = 0.0
        self.loss_ema = None
        self.last_avg_q = 0.0
        self.last_max_q = 0.0

        self.data_dir = data_dir or os.getcwd()

        if load_file is not None and os.path.exists(load_file):
            self.load_model(load_file)

    def update_stacks(self, processed_frames: np.ndarray) -> None:
        self.state_stacks = np.roll(self.state_stacks, shift=-1, axis=1)
        self.state_stacks[:, -1, :, :] = processed_frames

    def epsilon_scheduler(self) -> float:
        frac = min(self.frame_count / float(self.epsilon_decay_frames), 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def train_step(self):
        if self.replay.size < max(self.train_start, self.batch_size):
            return
        if self.frame_count % self.train_freq != 0:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size, self.device)
        q_values = self.network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.network(next_states)
            next_actions = torch.argmax(next_q, dim=1)
            target_q = self.target_network(next_states)
            max_next_q = target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        self.optimizer.step()

        self.last_loss = float(loss.item())
        if self.loss_ema is None:
            self.loss_ema = self.last_loss
        else:
            self.loss_ema = 0.95 * self.loss_ema + 0.05 * self.last_loss

        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def reset(self, observations: np.ndarray) -> None:
        processed = preprocess_batch(observations)
        for env in range(self.num_envs):
            self.state_stacks[env] = np.repeat(processed[env][None, ...], self.stack_size, axis=0)
            self.last_actions[env] = -1

    def act(self, observations: np.ndarray) -> np.ndarray:
        processed = preprocess_batch(observations)
        self.update_stacks(processed)

        self.epsilon = self.epsilon_scheduler()
        actions = np.empty(self.num_envs, dtype=np.int64)
        stacked = torch.from_numpy(self.state_stacks).to(self.device, dtype=torch.float32) / 255.0

        with torch.no_grad():
            q_values = self.network(stacked)
            greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy()
            self.last_avg_q = float(q_values.mean().item())
            self.last_max_q = float(q_values.max().item())

        for env in range(self.num_envs):
            if np.random.random() < self.epsilon:
                actions[env] = np.random.randint(self.num_actions)
            else:
                actions[env] = int(greedy_actions[env])
        self.last_states = self.state_stacks.copy()
        self.last_actions = actions.copy()
        return actions

    def observe(self, next_observations, rewards, terminations, truncations):
        processed = preprocess_batch(next_observations)
        next_stacks = self.state_stacks.copy()
        next_stacks = np.roll(next_stacks, shift=-1, axis=1)
        next_stacks[:, -1, :, :] = processed

        for env in range(self.num_envs):
            if self.last_actions[env] == -1:
                continue
            done = bool(terminations[env] or truncations[env])
            self.replay.add(self.last_states[env], self.last_actions[env], rewards[env], next_stacks[env], done)
            if done:
                stack = np.repeat(processed[env][None, ...], self.stack_size, axis=0)
                self.state_stacks[env] = stack
                self.last_actions[env] = -1

        self.frame_count += self.num_envs

    def save_model(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)

    def load_model(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.target_network.load_state_dict(self.network.state_dict())

    # Single-env adapter (for harness_physical + sim_latency)


class Agent:
    def __init__(self, data_dir=None, seed=0, num_actions=18, total_frames=1_000_000, **kwargs):
        self.core = DQNCore(
            num_envs=1, data_dir=data_dir, seed=seed, num_actions=num_actions, total_frames=total_frames, **kwargs
        )
        self.prev_obs: Optional[np.ndarray] = None
        self.prev_reward = 0.0
        self.prev_done = False

    def frame(self, observation_rgb8, reward, end_of_episode):
        obs_batch = observation_rgb8[None, ...]
        if self.prev_obs is None:
            self.core.reset(obs_batch)
        actions = self.core.act(obs_batch)
        if self.prev_obs is not None:
            self.core.observe(
                self.prev_obs,
                np.array([self.prev_reward]),
                np.array([self.prev_done]),
                np.array([False]),
            )
            self.core.train_step()
        self.prev_obs = observation_rgb8[None, ...]
        self.prev_reward = reward
        self.prev_done = bool(end_of_episode > 0)
        return int(actions[0])

    def save_model(self, path: str) -> None:
        self.core.save_model(path)

    def load_model(self, path: str) -> None:
        self.core.load_model(path)


# Vectorized adapter (for future vector trainer)


class VectorDQNAgent(VectorAgent):
    def __init__(
        self,
        *,
        num_envs: int,
        seed: int,
        num_actions: int,
        results_dir: Optional[str] = None,
        total_frames: int = 1_000_000,
        **kwargs,
    ):
        self.core = DQNCore(
            num_envs=num_envs,
            seed=seed,
            num_actions=num_actions,
            total_frames=total_frames,
            data_dir=results_dir,
            **kwargs,
        )
        self.num_envs = num_envs

    def reset(self, num_envs: int) -> None:
        self.num_envs = num_envs

    def act(self, observations: np.ndarray) -> np.ndarray:
        return self.core.act(observations)

    def observe(self, next_observations, rewards, terminations, truncations, infos):
        self.core.observe(next_observations, rewards, terminations, truncations)

    def train_step(self) -> None:
        return self.core.train_step()

    def save_model(self, path: str) -> None:
        return self.core.save_model(path)

    def load_model(self, path: str) -> None:
        return self.core.load_model(path)
