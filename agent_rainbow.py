from __future__ import annotations

import math
import os
import time
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_agents import VectorAgent
from agent_utils import preprocess_batch
import cv2

from framework.Logger import logger

# Expected observation dimensions from physical harness (matches agent_ppo)
EXPECTED_OBS_DIMS = (210, 160, 3)

# Noisy layers / prioritized replay


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.6):
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
            if float(self.tree[left].item()) >= prefixsum:
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
        obs_shape: Tuple[int, int],
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
        total = float(self.tree.total().item())
        segment = total / batch_size
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

        probs = priorities / total
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

    def save(self, path: str) -> None:
        """Save replay buffer to disk."""
        data = {
            "states": self.states[:self.size] if not self.full else self.states,
            "next_states": self.next_states[:self.size] if not self.full else self.next_states,
            "actions": self.actions[:self.size] if not self.full else self.actions,
            "rewards": self.rewards[:self.size] if not self.full else self.rewards,
            "dones": self.dones[:self.size] if not self.full else self.dones,
            "tree": self.tree.tree.numpy(),
            "max_priority": self.max_priority,
            "ptr": self.ptr,
            "full": self.full,
            "beta": self.beta,
        }
        np.savez_compressed(path, **data)
        logger.info(f"Saved replay buffer ({self.size} transitions) to {path}")

    def load(self, path: str) -> None:
        """Load replay buffer from disk."""
        data = np.load(path, allow_pickle=True)
        
        # Restore buffer data
        size = len(data["states"])
        self.states[:size] = data["states"]
        self.next_states[:size] = data["next_states"]
        self.actions[:size] = data["actions"]
        self.rewards[:size] = data["rewards"]
        self.dones[:size] = data["dones"]
        
        # Restore tree and metadata
        self.tree.tree = torch.from_numpy(data["tree"])
        self.max_priority = float(data["max_priority"])
        self.ptr = int(data["ptr"])
        self.full = bool(data["full"])
        self.beta = float(data["beta"])
        
        logger.info(f"Loaded replay buffer ({self.size} transitions) from {path}")


class RainbowNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int, num_atoms: int, obs_height: int, obs_width: int):
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

        conv_out_h, conv_out_w = self._conv_output_shape(in_channels, obs_height, obs_width)
        self.fc_input_dim = 64 * conv_out_h * conv_out_w

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

    def _conv_output_shape(self, in_channels: int, height: int, width: int) -> Tuple[int, int]:
        """Compute post-conv spatial dims for arbitrary input shapes."""
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            out = self.conv(dummy)
        return out.shape[2], out.shape[3]


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
        disable_training: bool = False,
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

        self.n_step = n_step
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # GPU handling with fallback (matches agent_ppo behavior)
        if torch.cuda.is_available() and gpu >= 0:
            self.device = torch.device(f'cuda:{gpu}')
            torch.cuda.manual_seed_all(seed)
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        logger.info(f"agent_rainbow: Using device: {self.device}")

        self.disable_training = disable_training
        self.network = RainbowNetwork(stack_size, num_actions, num_atoms, obs_height, obs_width).to(self.device)
        self.target_network = RainbowNetwork(stack_size, num_actions, num_atoms, obs_height, obs_width).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Use a slightly smaller LR for noisy parameters to slow σ drift without changing the base LR.
        noisy_params = []
        base_params = []
        for name, param in self.network.named_parameters():
            if "weight_sigma" in name or "bias_sigma" in name:
                noisy_params.append(param)
            else:
                base_params.append(param)
        self.optimizer = torch.optim.Adam(
            [
                {"params": base_params, "lr": learning_rate},
                {"params": noisy_params, "lr": learning_rate * 0.5},
            ]
        )

        # Warn about memory footprint early so large 128x128x16 configs don't OOM.
        bytes_per_transition = stack_size * obs_height * obs_width  # uint8 per pixel
        est_bytes = bytes_per_transition * buffer_size * 2  # states + next_states
        if est_bytes > 8 * (1024**3):
            est_gb = est_bytes / float(1024**3)
            print(
                f"[Rainbow] Replay buffer will allocate ~{est_gb:.1f} GiB (stack={stack_size}, "
                f"res={obs_height}x{obs_width}, capacity={buffer_size}). "
                "Consider lowering rainbow_buffer_size for larger inputs."
            )

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
        # Mirror on device to avoid per-step CPU->GPU tensor builds. Ring buffer index avoids costly rolls.
        self.state_stacks_torch = torch.zeros(
            (num_envs, stack_size, obs_height, obs_width), device=self.device, dtype=torch.uint8
        )
        self.write_idx = np.full((num_envs,), stack_size - 1, dtype=np.int64)
        self.write_idx_torch = torch.full((num_envs,), stack_size - 1, device=self.device, dtype=torch.int64)
        self.base_idx = torch.arange(self.stack_size, device=self.device, dtype=torch.int64)
        self.last_states = np.zeros_like(self.state_stacks)
        self.last_actions = np.full((num_envs,), -1, dtype=np.int64)
        self.n_step_buffers: List[Deque] = [deque(maxlen=self.n_step) for _ in range(num_envs)]

        self.frame_count = 0
        self.training_steps = 0
        self.epsilon = epsilon_start

        self.last_loss = 0.0
        self.loss_ema = None
        self.last_avg_q = 0.0
        self.last_max_q = 0.0
        self.last_td_error = 0.0
        self.last_grad_norm = 0.0
        self.train_losses: List[float] = []

        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self.device)

        self.data_dir = data_dir or os.getcwd()
        if load_file is not None and os.path.exists(load_file):
            self.load_model(load_file)

    def _prepare_obs(self, observations: np.ndarray) -> np.ndarray:
        """
        Fast-path preprocessing: if observations are already grayscale HxW at the
        configured resolution, skip cv2. Only fall back to preprocess_batch when
        we receive RGB or mismatched shapes.
        """
        # Expect (num_envs, H, W) uint8 when AtariPreprocessing is used upstream.
        if (
            observations.ndim == 3
            and observations.shape[1] == self.obs_height
            and observations.shape[2] == self.obs_width
            and observations.dtype == np.uint8
        ):
            return observations

        # Otherwise fall back to the original cv2-based preprocessing.
        if observations.ndim == 4 and observations.shape[1] == 3:
            observations = np.transpose(observations, (0, 2, 3, 1))
        if observations.ndim == 4 and observations.shape[-1] == 3:
            return preprocess_batch(observations, self.obs_height, self.obs_width)

        # If shape is unexpected (e.g., channel-first), convert via cv2 resize.
        num_envs = observations.shape[0]
        processed = np.zeros((num_envs, self.obs_height, self.obs_width), dtype=np.uint8)
        for i in range(num_envs):
            frame = observations[i]
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (self.obs_width, self.obs_height), interpolation=cv2.INTER_AREA)
            processed[i] = frame
        return processed

    def epsilon_scheduler(self) -> float:
        frac = min(self.frame_count / float(self.epsilon_decay_frames), 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def reset(self, observations: np.ndarray):
        processed = self._prepare_obs(observations)
        for env in range(self.num_envs):
            self.state_stacks[env] = np.repeat(processed[env][None, ...], self.stack_size, axis=0)
            self.last_actions[env] = -1
            self.n_step_buffers[env].clear()
        # reset ring indices
        self.write_idx.fill(self.stack_size - 1)
        self.write_idx_torch.fill_(self.stack_size - 1)
        # sync torch buffer
        frame_torch = torch.from_numpy(processed).to(self.device, non_blocking=True)
        self.state_stacks_torch = frame_torch.unsqueeze(1).repeat(1, self.stack_size, 1, 1)

    def act(self, observations: np.ndarray) -> np.ndarray:
        processed = self._prepare_obs(observations)
        # advance ring index
        self.write_idx = (self.write_idx + 1) % self.stack_size
        self.write_idx_torch = (self.write_idx_torch + 1) % self.stack_size

        # Update numpy buffer (for replay bookkeeping) at the current write slot
        self.state_stacks[np.arange(self.num_envs), self.write_idx, :, :] = processed
        # Update torch buffer on device without host/device ping-pong
        frame_torch = torch.from_numpy(processed).to(self.device, non_blocking=True)
        self.state_stacks_torch[np.arange(self.num_envs), self.write_idx, :, :] = frame_torch

        # Gather ordered stacks using ring indices (oldest->newest)
        order_idx = (self.write_idx_torch.unsqueeze(1) + 1 + self.base_idx) % self.stack_size
        gather_idx = order_idx.view(self.num_envs, self.stack_size, 1, 1).expand(
            -1, -1, self.obs_height, self.obs_width
        )
        stacked_torch = torch.gather(self.state_stacks_torch, 1, gather_idx)
        stacked = stacked_torch.float().mul_(1.0 / 255.0)

        # also keep numpy ordered stacks for replay bookkeeping
        order_idx_np = (self.write_idx[:, None] + 1 + np.arange(self.stack_size)) % self.stack_size
        self.last_states = np.take_along_axis(
            self.state_stacks, order_idx_np[:, :, None, None], axis=1
        )

        self.epsilon = self.epsilon_scheduler()
        actions = np.empty(self.num_envs, dtype=np.int64)

        with torch.no_grad():
            self.network.reset_noise()
            q_values = self.network.q_values(stacked, self.support)
            greedy_actions = torch.argmax(q_values, dim=1)
            self.last_avg_q = float(q_values.mean().item())
            self.last_max_q = float(q_values.max().item())

        for env in range(self.num_envs):
            if np.random.random() < self.epsilon:
                actions[env] = np.random.randint(self.num_actions)
            else:
                actions[env] = int(greedy_actions[env].item())

        self.last_actions = actions.copy()
        return actions

    def _flush_n_step(self, env: int, processed_next: np.ndarray, done: bool):
        buffer = self.n_step_buffers[env]
        if len(buffer) == 0:
            return
        while buffer:
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
                order_idx = (self.write_idx[env] + 1 + np.arange(self.stack_size)) % self.stack_size
                next_state = np.take_along_axis(
                    self.state_stacks[env], order_idx[:, None, None], axis=0
                )
            self.replay.add(state, action, cumulative_reward, next_state, done)
            buffer.popleft()

    def observe(self, next_observations, rewards, terminations, truncations):
        processed = self._prepare_obs(next_observations)
        next_stacks = np.empty_like(self.last_states)
        next_stacks[:, :-1, :, :] = self.last_states[:, 1:, :, :]
        next_stacks[:, -1, :, :] = processed

        for env in range(self.num_envs):
            action = self.last_actions[env]
            if action == -1:
                continue

            done = bool(terminations[env] or truncations[env])
            self.n_step_buffers[env].append((rewards[env], done, self.last_states[env], action))

            if done:
                self._flush_n_step(env, processed[env], done)
                stack = np.repeat(processed[env][None, ...], self.stack_size, axis=0)
                self.state_stacks[env] = stack
                self.state_stacks_torch[env] = torch.from_numpy(stack).to(self.device, non_blocking=True)
                self.write_idx[env] = self.stack_size - 1
                self.write_idx_torch[env] = self.stack_size - 1
                self.last_actions[env] = -1
            elif len(self.n_step_buffers[env]) == self.n_step:
                cumulative_reward = 0.0
                for idx, (r, d, _, _) in enumerate(self.n_step_buffers[env]):
                    cumulative_reward += (self.gamma**idx) * r
                    if d:
                        break
                state, oldest_action = self.n_step_buffers[env][0][2], self.n_step_buffers[env][0][3]
                next_state = next_stacks[env]
                final_done = any(d for (_, d, _, _) in self.n_step_buffers[env])
                self.replay.add(state, oldest_action, cumulative_reward, next_state, final_done)

        self.frame_count += self.num_envs

    def train_step(self):
        if self.disable_training:
            return
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
            l_idx = (l + offset).view(-1)
            u_idx = (u + offset).view(-1)
            l_weight = (u.float() - b).view(-1)
            u_weight = (b - l.float()).view(-1)
            flat_target = target_probs.view(-1)
            eq_mask = l_idx == u_idx
            if eq_mask.any():
                proj_dist.view(-1).index_add_(0, l_idx[eq_mask], flat_target[eq_mask])
            if (~eq_mask).any():
                proj_dist.view(-1).index_add_(0, l_idx[~eq_mask], (flat_target * l_weight)[~eq_mask])
                proj_dist.view(-1).index_add_(0, u_idx[~eq_mask], (flat_target * u_weight)[~eq_mask])

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
        self.train_losses.append(self.loss_ema if self.loss_ema is not None else self.last_loss)

        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def save_model(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)

    def load_model(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)

    def save_checkpoint(self, path: str) -> None:
        """Save full training checkpoint including model, optimizer, replay buffer, and training state."""
        # Save model and optimizer
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "frame_count": self.frame_count,
            "training_steps": self.training_steps,
            "epsilon": self.epsilon,
            "loss_ema": self.loss_ema,
        }
        torch.save(checkpoint, path)
        
        # Save replay buffer separately (can be large)
        replay_path = path.replace(".pt", "_replay.npz").replace(".model", "_replay.npz")
        if replay_path == path:
            replay_path = path + "_replay.npz"
        self.replay.save(replay_path)
        
        logger.info(f"Saved checkpoint to {path} (frame {self.frame_count}, {self.training_steps} training steps)")

    def load_checkpoint(self, path: str) -> None:
        """Load full training checkpoint including model, optimizer, replay buffer, and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.frame_count = checkpoint["frame_count"]
        self.training_steps = checkpoint["training_steps"]
        self.epsilon = checkpoint["epsilon"]
        self.loss_ema = checkpoint.get("loss_ema", None)
        
        # Load replay buffer
        replay_path = path.replace(".pt", "_replay.npz").replace(".model", "_replay.npz")
        if replay_path == path:
            replay_path = path + "_replay.npz"
        if os.path.exists(replay_path):
            self.replay.load(replay_path)
        else:
            logger.warning(f"Replay buffer not found at {replay_path}, starting with empty buffer")
        
        logger.info(f"Loaded checkpoint from {path} (frame {self.frame_count}, {self.training_steps} training steps)")


# Single-env adapter for physical harness
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
        self.train_losses = self.core.train_losses
        self.prev_obs: Optional[np.ndarray] = None
        self.prev_reward = 0.0
        self.prev_done = False

        # Frame skipping to match PPO behavior (act every frame_skip frames)
        self.frame_skip = 4
        self.step_count = 0
        self.last_action = 0

        # Accumulate rewards over frame_skip frames (matches SB3/PPO behavior)
        self.accumulated_reward = 0.0

        logger.info(f"agent_rainbow: Initialized with frame_skip={self.frame_skip}")

    def frame(self, observation_rgb8, reward, end_of_episode):
        """
        Called every frame by harness.

        Args:
            observation_rgb8: RGB observation (210, 160, 3) uint8
            reward: Scalar reward from environment
            end_of_episode: 0=ongoing, 1=life lost, 2=game over, 3=timeout

        Returns:
            action_index: Integer action index to execute
        """
        # Validate input shape (matches agent_ppo)
        assert observation_rgb8.shape == EXPECTED_OBS_DIMS, \
            f"Observation shape is: {observation_rgb8.shape}, but we expected: {EXPECTED_OBS_DIMS}"

        self.step_count += 1
        done = bool(end_of_episode > 0)

        # Accumulate rewards FIRST (before any reset logic)
        self.accumulated_reward += reward

        # Handle episode end logging
        if end_of_episode > 0:
            if end_of_episode == 1:
                end_reason = "life lost"
            elif end_of_episode == 2:
                end_reason = "game over"
            else:
                end_reason = "timeout"
            logger.debug(f"agent_rainbow: Episode end at step {self.step_count}: {end_reason}")

        # Only act every frame_skip frames (matches PPO behavior)
        if self.step_count % self.frame_skip != 0:
            # If episode ends mid-frameskip, flush the accumulated reward
            if end_of_episode > 0 and self.prev_obs is not None:
                obs_batch = observation_rgb8[None, ...]
                self.core.observe(
                    obs_batch,
                    np.array([self.accumulated_reward]),
                    np.array([done]),
                    np.array([False]),
                )
                self.core.train_step()
                self.accumulated_reward = 0.0
                self.prev_obs = None
            return self.last_action

        obs_batch = observation_rgb8[None, ...]

        if self.prev_obs is None:
            self.core.reset(obs_batch)
        else:
            # Use accumulated reward instead of single-frame reward
            self.core.observe(
                obs_batch,
                np.array([self.accumulated_reward]),
                np.array([done]),
                np.array([False]),
            )
            self.core.train_step()

        actions = self.core.act(obs_batch)

        self.prev_obs = obs_batch
        self.prev_reward = self.accumulated_reward
        self.prev_done = done
        self.last_action = int(actions[0])

        # Reset accumulated reward after using it
        self.accumulated_reward = 0.0

        # Handle episode end - reset state for next episode
        if end_of_episode > 0:
            self.prev_obs = None

        return self.last_action

    def save_model(self, path: str) -> None:
        self.core.save_model(path)

    def save_checkpoint(self, path: str) -> None:
        """Save full checkpoint including replay buffer for resuming training."""
        self.core.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        """Load full checkpoint including replay buffer for resuming training."""
        self.core.load_checkpoint(path)


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
        self._initialized = False

    def reset(self, num_envs: int) -> None:
        if num_envs != self.num_envs:
            raise ValueError(f"VectorRainbowAgent intialised for {self.num_envs} envs; recieved {num_envs}.")
        # actual reset happens on first act call
        self._initialized = False

    def act(self, observations: np.ndarray) -> np.ndarray:
        if not self._initialized:
            self.core.reset(observations)
            self._initialized = True
        return self.core.act(observations)

    def observe(self, next_observations, rewards, terminations, truncations, infos):
        self.core.observe(next_observations, rewards, terminations, truncations)

    def train_step(self) -> None:
        self.core.train_step()

    def save_model(self, path: str) -> None:
        self.core.save_model(path)

    def load_model(self, path: str) -> None:
        self.core.load_model(path)
