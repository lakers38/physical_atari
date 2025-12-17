from __future__ import annotations

import math
import os
import time
from collections import deque
from typing import Deque, Optional

from mpmath.libmp.libelefun import atan_taylor_get_cached
import numpy as np
import torch
from torch._inductor.ir import NoneAsConstantBuffer
import torch.nn as nn
import torch.nn.functional as F

from vector_agents import VectorAgent
from agent_utils import preprocess_batch

# Noisy layers / prioritized replay


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        noise = torch.randn(size, device=self.weight_mu.device)
        return noise.sign().mul_(noise.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
        self.tree = torch.zeros(2 * self.capacity, dtype=torch.float32)

    def update(self, idx: int, priority: float):
        tree_idx = idx + self.capacity
        self.tree[tree_idx] = priority
        tree_idx //= 2
        while tree_idx >= 1:
            self.tree[tree_idx] = self.tree[2 * tree_idx] + self.tree[2 * tree_idx + 1]
            tree_idx //= 2

    def total(self):
        return self.tree[1]

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if self.tree[left] >= prefixsum:
                idx = left
            else:
                prefixsum -= float(self.tree[left].item())
                idx = left + 1
        return idx - self.capacity


class PrioritizedReplay:
    def __init__(
        self,
        capacity: int,
        stack_size: int,
        obs_shape: tuple[int, int],
        alpha: float,
        beta: float,
        beta_increment: float,
        priority_eps: float,
        device: torch.device,
    ):
        self.capacity = capacity
        self.stack_size = stack_size
        self.obs_shape = obs_shape
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priority_eps = priority_eps
        self.device = device

        self.states = np.zeros((capacity, stack_size, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, stack_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.ptr = 0
        self.full = False

    @property
    def size(self):
        return self.capacity if self.full else self.ptr

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.tree.update(self.ptr, self.max_priority**self.alpha)

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int):
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx = self.tree.find_prefixsum_idx(s)
            idx = min(idx, self.size - 1)
            indices.append(idx)
            priorities.append(self.tree.tree[idx + self.tree.capacity].item())

        indices = np.array(indices, dtype=np.int64)
        priorities = np.array(priorities, dtype=np.float32)

        probs = priorities / self.tree.total().item()
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()
        weights = torch.from_numpy(weights).to(self.device, dtype=torch.float32)
        self.beta = min(1.0, self.beta + self.beta_increment)

        states = torch.from_numpy(self.states[indices]).to(self.device, dtype=torch.float32) / 255.0
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device, dtype=torch.float32) / 255.0
        actions = torch.from_numpy(self.actions[indices]).to(self.device, dtype=torch.int64)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device, dtype=torch.float32)
        dones = torch.from_numpy(self.dones[indices].astype(np.float32)).to(self.device, dtype=torch.float32)

        return indices, states, actions, rewards, next_states, dones, weights

    def update_priorities(self, indices, priorities):
        priorities = priorities.cpu().numpy()
        for idx, priority in zip(indices, priorities):
            priority = float(priority) + self.priority_eps
            self.tree.update(idx, priority**self.alpha)
            self.max_priority = max(self.max_priority, priority)


class RainbowNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int, num_atoms: int):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc_input_dim = 64 * 7 * 7

        self.value_stream = nn.Sequential(NoisyLinear(self.fc_input_dim, 512), nn.ReLU(), NoisyLinear(512, num_atoms))
        self.adv_stream = nn.Sequential(
            NoisyLinear(self.fc_input_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions * num_atoms),
        )

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)

        value = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantage = self.adv_stream(features).view(-1, self.num_actions, self.num_atoms)
        q_atoms = value + (advantage - advantage.mean(dim=1, keepdim=True))
        log_probs = F.log_softmax(q_atoms, dim=2)
        probs = torch.exp(log_probs)
        return probs, log_probs

    def q_values(self, x, support):
        probs, _ = self.forward(x)
        return torch.sum(probs * support.view(1, 1, -1), dim=2)


class RainbowCore:
    def __init__(
        self,
        *,
        num_envs: int,
        seed: int,
        num_actions: int,
        total_frames: int,
        stack_size: int = 4,
        obs_height: int = 84,
        obs_width: int = 84,
        buffer_size: int = 100_000,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        train_start: int = 50_000,
        train_freq: int = 1,
        target_update_freq: int = 2_000,
        epsilon_start: float = 0.0,
        epsilon_end: float = 0.0,
        epsilon_decay_frames: int = 1_000_000,
        frame_skip: int = 1,
        grad_clip: Optional[float] = 10.0,
        n_step: int = 3,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        priority_alpha: float = 0.5,
        priority_beta: float = 0.4,
        priority_beta_increment: float = 1e-6,
        priority_eps: float = 1e-6,
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
        self.grad_clip = grad_clip
        self.frame_skip = frame_skip

        self.n_step = n_step
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        torch.manual_seed(seed)
        np.random.seed(seed)

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu}')
            torch.cuda.manual_seed_all(seed)
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.network = RainbowNetwork(stack_size, num_actions, num_atoms).to(self.device)
        self.target_network = RainbowNetwork(stack_size, num_actions, num_atoms).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.replay = PrioritizedReplay(
            capacity=buffer_size,
            stack_size=stack_size,
            obs_shape=(obs_height, obs_width),
            alpha=priority_alpha,
            beta=priority_beta,
            beta_increment=priority_beta_increment,
            priority_eps=priority_eps,
            device=self.device,
        )

        self.state_stacks = np.zeros((num_envs, stack_size, obs_height, obs_width), dtype=np.uint8)
        self.last_states = np.zeros_like(self.state_stacks)
        self.last_actions = np.full((num_envs,), -1, dtype=np.int64)
        self.n_step_buffers: list[Deque] = [deque(maxlen=self.n_step) for _ in range(num_envs)]

        self.frame_count = 0
        self.training_steps = 0
        self.epsilon = epsilon_start

        self.last_loss = 0.0
        self.loss_ema = None
        self.last_avg_q = 0.0
        self.last_max_q = 0.0
        self.last_td_error = 0.0
        self.last_grad_norm = 0.0

        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self.device)

        self.data_dir = data_dir or os.getcwd()
        if load_file is not None and os.path.exists(load_file):
            self.load_model(load_file)

    def epsilon_scheduler(self) -> float:
        frac = min(self.frame_count / float(self.epsilon_decay_frames), 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def reset(self, observations: np.ndarray):
        processed = preprocess_batch(observations, self.obs_height, self.obs_width)
        for env in range(self.num_envs):
            self.state_stacks[env] = np.repeat(processed[env][None, ...], self.stack_size, axis=0)
            self.last_actions[env] = -1
            self.n_step_buffers[env].clear()

    def act(self, observations: np.ndarray) -> np.ndarray:
        processed = preprocess_batch(observations, self.obs_height, self.obs_width)
        self.state_stacks = np.roll(self.state_stacks, shift=-1, axis=1)
        self.state_stacks[:, -1, :, :] = processed

        self.epsilon = self.epsilon_scheduler()
        actions = np.empty(self.num_envs, dtype=np.int64)
        stacked = torch.from_numpy(self.state_stacks).to(self.device, dtype=torch.float32) / 255.0

        with torch.no_grad():
            self.network.reset_noise()
            q_values = self.network.q_values(stacked, self.support)
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

    def _flush_n_step(self, env: int, processed_next: np.ndarray, done: bool):
        buffer = self.n_step_buffers[env]
        if len(buffer) == 0:
            return
        cumulative_reward = 0.0
        for idx, (r, d, _, _) in enumerate(buffer):
            cumulative_reward += (self.gamma**idx) * r
            if d:
                done = True
                break
        state, action = buffer[0][2], buffer[0][3]
        if done:
            next_state = np.repeat(processed_next[None, ...], self.stack_size, axis=0)
        else:
            next_state = self.state_stacks[env]
        self.replay.add(state, action, cumulative_reward, next_state, done)
        buffer.clear()

    def observe(self, next_observations, rewards, terminations, truncations):
        processed = preprocess_batch(next_observations, self.obs_height, self.obs_width)
        next_stacks = self.state_stacks.copy()
        next_stacks = np.roll(next_stacks, shift=-1, axis=1)
        next_stacks[:, -1, :, :] = processed

        for env in range(self.num_envs):
            action = self.last_actions[env]
            if action == -1:
                continue

            done = bool(terminations[env] or truncations[env])
            self.n_step_buffers[env].append((rewards[env], done, self.last_states[env], action))

            if len(self.n_step_buffers[env]) == self.n_step:
                cumulative_reward = 0.0
                for idx, (r, d, _, _) in enumerate(self.n_step_buffers[env]):
                    cumulative_reward += (self.gamma**idx) * r
                    if d:
                        break
                state = self.n_step_buffers[env][0][2]
                next_state = next_stacks[env]
                final_done = done
                self.replay.add(state, action, cumulative_reward, next_state, final_done)

            if done:
                self._flush_n_step(env, processed[env], done)
                stack = np.repeat(processed[env][None, ...], self.stack_size, axis=0)
                self.state_stacks[env] = stack
                self.last_actions[env] = -1

        self.frame_count += self.num_envs

    def train_step(self):
        if self.replay.size < max(self.train_start, self.batch_size):
            return
        if self.frame_count % self.train_freq != 0:
            return

        indices, states, actions, rewards, next_states, dones, weights = self.replay.sample(self.batch_size)

        self.network.reset_noise()
        self.target_network.reset_noise()

        with torch.no_grad():
            next_probs, _ = self.network(next_states)
            q_values = torch.sum(next_probs * self.support.view(1, 1, -1), dim=2)
            next_actions = torch.argmax(q_values, dim=1)

            target_probs, _ = self.target_network(next_states)
            target_probs = target_probs[torch.arange(self.batch_size), next_actions]
            tz = rewards.unsqueeze(1) + (self.gamma**self.n_step) * (1 - dones.unsqueeze(1)) * self.support.unsqueeze(0)
            tz = tz.clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / self.delta_z
            l = b.floor().to(torch.int64)
            u = b.ceil().to(torch.int64)
            proj_dist = torch.zeros(target_probs.size(), device=self.device)
            offset = torch.arange(
                0,
                self.batch_size * self.num_atoms,
                self.num_atoms,
                device=self.device,
                dtype=torch.int64,
            ).unsqueeze(1)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (target_probs * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (target_probs * (b - l.float())).view(-1))

        self.network.reset_noise()
        probs, log_probs = self.network(states)
        log_p = log_probs[torch.arange(self.batch_size), actions]
        sample_losses = -(proj_dist * log_p).sum(dim=1)
        loss = (sample_losses * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        total_norm = 0.0
        for p in self.network.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        self.last_grad_norm = math.sqrt(total_norm)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        self.optimizer.step()

        self.replay.update_priorities(indices, sample_losses.detach())
        self.last_loss = float(loss.item())
        if self.loss_ema is None:
            self.loss_ema = self.last_loss
        else:
            self.loss_ema = 0.95 * self.loss_ema + 0.05 * self.last_loss

        self.last_td_error = float(sample_losses.detach().mean().item())

        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def save_model(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)

    def load_model(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)


# Single-env adapter
class Agent:
    def __init__(self, data_dir=None, seed=0, num_actions=18, total_frames=1_000_000, **kwargs):
        self.core = RainbowCore(
            num_envs=1,
            seed=seed,
            num_actions=num_actions,
            total_frames=total_frames,
            data_dir=data_dir,
            **kwargs,
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
        self.core.load_model(path)


# Vector Adapter


class VectorRainbowAgent(VectorAgent):
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
        self.core = RainbowCore(
            num_envs=num_envs,
            seed=seed,
            num_actions=num_actions,
            total_frames=total_frames,
            data_dir=results_dir,
            **kwargs,
        )
        self.num_envs = num_envs

    def reset(self, num_envs: int) -> None:
        if num_envs != self.num_envs:
            raise ValueError(f"VectorRainbowAgent intialised for {self.num_envs} envs; recieved {num_envs}.")
        # actual reset happens on first observe call
        pass

    def act(self, observations: np.ndarray) -> np.ndarray:
        return self.core.act(observations)

    def observe(self, next_observations, rewards, terminations, truncations, infos):
        self.core.observe(next_observations, rewards, terminations, truncations)

    def train_step(self) -> None:
        self.core.train_step()

    def save_model(self, path: str) -> None:
        self.core.save_model(path)

    def load_model(self, path: str) -> None:
        self.core.load_model(path)
