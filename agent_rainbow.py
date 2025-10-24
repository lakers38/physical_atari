import math
import os
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- NoisyNet layers ---------------------------------------------------------

class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

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


# --- Prioritized Replay Buffer -----------------------------------------------

class SumTree:
    def __init__(self, capacity: int):
        # next power of two
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
                prefixsum -= self.tree[left]
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

    def _store(self, array, value):
        array[self.ptr] = value

    def add(self, state, action, reward, next_state, done):
        self._store(self.states, state)
        self._store(self.actions, action)
        self._store(self.rewards, reward)
        self._store(self.next_states, next_state)
        self._store(self.dones, done)

        self.tree.update(self.ptr, self.max_priority ** self.alpha)

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int):
        assert self.size >= batch_size
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
        self.beta = min(1.0, self.beta + self.beta_increment)

        states = torch.from_numpy(self.states[indices]).to(self.device, dtype=torch.float32) / 255.0
        actions = torch.from_numpy(self.actions[indices]).to(self.device, dtype=torch.int64)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device, dtype=torch.float32)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device, dtype=torch.float32) / 255.0
        dones = torch.from_numpy(self.dones[indices].astype(np.float32)).to(self.device, dtype=torch.float32)
        weights = torch.from_numpy(weights).to(self.device, dtype=torch.float32)

        return indices, states, actions, rewards, next_states, dones, weights

    def update_priorities(self, indices, priorities):
        priorities = priorities.detach().cpu().numpy()
        for idx, priority in zip(indices, priorities):
            priority = (abs(priority) + self.priority_eps).item()
            self.tree.update(idx, priority ** self.alpha)
            self.max_priority = max(self.max_priority, priority)


# --- Rainbow Agent -----------------------------------------------------------


class NoisyRainbowNet(nn.Module):
    def __init__(self, in_channels: int, num_actions: int, num_atoms: int):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_actions = num_actions

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc_input_dim = 64 * 7 * 7

        self.value_stream = nn.Sequential(
            NoisyLinear(self.fc_input_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, num_atoms),
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.fc_input_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions * num_atoms),
        )

        self.reset_noise()

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)

        value = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(-1, self.num_actions, self.num_atoms)

        q_atoms = value + (advantage - advantage.mean(dim=1, keepdim=True))
        log_probs = F.log_softmax(q_atoms, dim=2)
        probs = torch.exp(log_probs)
        return probs, log_probs

    def q_values(self, x, support):
        probs, _ = self.forward(x)
        q = torch.sum(probs * support.view(1, 1, -1), dim=2)
        return q


class Agent:
    def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
        self.num_actions = num_actions
        self.total_frames = total_frames
        self.seed = seed

        self.stack_size = kwargs.get('stack_size', 4)
        self.obs_height = kwargs.get('obs_height', 84)
        self.obs_width = kwargs.get('obs_width', 84)
        self.gamma = kwargs.get('gamma', 0.99)
        self.n_step = kwargs.get('n_step', 3)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.batch_size = kwargs.get('batch_size', 32)
        self.buffer_size = kwargs.get('buffer_size', 500_000)
        self.train_start = kwargs.get('train_start', 50_000)
        self.train_freq = kwargs.get('train_freq', 1)
        self.target_update_freq = kwargs.get('target_update_freq', 2_000)
        self.grad_clip = kwargs.get('grad_clip', 10.0)

        self.alpha = kwargs.get('priority_alpha', 0.5)
        self.beta = kwargs.get('priority_beta', 0.4)
        self.beta_increment = kwargs.get('priority_beta_increment', 1e-6)
        self.priority_eps = kwargs.get('priority_eps', 1e-6)

        self.num_atoms = kwargs.get('num_atoms', 51)
        self.v_min = kwargs.get('v_min', -10.0)
        self.v_max = kwargs.get('v_max', 10.0)

        self.epsilon_start = kwargs.get('epsilon_start', 0.0)
        self.epsilon_end = kwargs.get('epsilon_end', 0.0)
        self.epsilon_decay_frames = kwargs.get('epsilon_decay_frames', 1_000_000)
        self.epsilon = self.epsilon_start

        self.frame_skip = kwargs.get('frame_skip', 1)

        self.device = self._init_device(kwargs.get('gpu', 0))

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self.device)

        self.q_network = NoisyRainbowNet(self.stack_size, num_actions, self.num_atoms).to(self.device)
        self.target_network = NoisyRainbowNet(self.stack_size, num_actions, self.num_atoms).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.replay = PrioritizedReplay(
            capacity=self.buffer_size,
            stack_size=self.stack_size,
            obs_shape=(self.obs_height, self.obs_width),
            alpha=self.alpha,
            beta=self.beta,
            beta_increment=self.beta_increment,
            priority_eps=self.priority_eps,
            device=self.device,
        )

        self.state_stack: Optional[Deque[np.ndarray]] = None
        self.last_state: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.n_step_buffer = deque(maxlen=self.n_step)

        self.frame_count = 0
        self.actions_taken = 0
        self.training_steps = 0

        self.last_loss = 0.0
        self.loss_ema = None
        self.last_avg_q = 0.0
        self.last_max_q = 0.0

        load_file = kwargs.get('load_file')
        if load_file is not None and os.path.exists(load_file):
            state_dict = torch.load(load_file, map_location=self.device)
            self.q_network.load_state_dict(state_dict)
            self.target_network.load_state_dict(self.q_network.state_dict())

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
        self.epsilon = self._epsilon_by_frame(self.frame_count)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
        self.q_network.reset_noise()
        with torch.no_grad():
            q_values = self.q_network.q_values(state_tensor, self.support)
            action = int(torch.argmax(q_values, dim=1).item())
            self.last_avg_q = float(q_values.mean().item())
            self.last_max_q = float(q_values.max().item())
        return action

    def _aggregate_n_step(self):
        reward, done = 0.0, False
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * r
            if d:
                done = True
                break
        state, action, _, _, _ = self.n_step_buffer[0]
        next_state, _, _, _, done_final = self.n_step_buffer[-1]
        return state, action, reward, next_state, done or done_final

    def _append_n_step(self, transition):
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return None
        return self._aggregate_n_step()

    def _projection_distribution(self, next_probs, rewards, dones):
        batch_size = rewards.size(0)
        support = self.support.unsqueeze(0).expand(batch_size, -1)

        T_z = rewards.unsqueeze(1) + (self.gamma ** self.n_step) * (1 - dones.unsqueeze(1)) * support
        T_z = T_z.clamp(self.v_min, self.v_max)

        b = (T_z - self.v_min) / (self.v_max - self.v_min) * (self.num_atoms - 1)
        lower = b.floor().to(torch.int64)
        upper = b.ceil().to(torch.int64)

        projection = torch.zeros_like(next_probs)
        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=self.device).unsqueeze(1)

        projection.view(-1).index_add_(
            0,
            (lower + offset).view(-1),
            (next_probs * (upper.float() - b)).view(-1),
        )
        projection.view(-1).index_add_(
            0,
            (upper + offset).view(-1),
            (next_probs * (b - lower.float())).view(-1),
        )
        return projection

    def _train_step(self):
        indices, states, actions, rewards, next_states, dones, weights = self.replay.sample(self.batch_size)

        with torch.no_grad():
            self.target_network.reset_noise()
            self.q_network.reset_noise()

            next_probs, _ = self.q_network(next_states)
            next_q = torch.sum(next_probs * self.support.view(1, 1, -1), dim=2)
            next_actions = torch.argmax(next_q, dim=1)

            target_next_probs, _ = self.target_network(next_states)
            target_next_probs = target_next_probs[torch.arange(self.batch_size), next_actions]
            target_distribution = self._projection_distribution(target_next_probs, rewards, dones)

        probs, log_probs = self.q_network(states)
        log_p = log_probs[torch.arange(self.batch_size), actions]

        per_sample_loss = -torch.sum(target_distribution * log_p, dim=1)
        loss = (per_sample_loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        self.optimizer.step()

        self.replay.update_priorities(indices, per_sample_loss.detach())

        self.last_loss = float(loss.item())
        if self.loss_ema is None:
            self.loss_ema = self.last_loss
        else:
            self.loss_ema = 0.95 * self.loss_ema + 0.05 * self.last_loss

        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def frame(self, observation_rgb8, reward, end_of_episode):
        processed = self._preprocess(observation_rgb8)
        self._ensure_state_stack(processed)
        current_state = self._stack_frames()

        done = end_of_episode > 0
        if self.last_state is not None and self.last_action is not None:
            transition = (self.last_state.copy(), self.last_action, reward, current_state.copy(), done)
            result = self._append_n_step(transition)
            if result is not None:
                self.replay.add(*result)
                if self.replay.size >= max(self.train_start, self.batch_size) and self.frame_count % self.train_freq == 0:
                    self._train_step()
                    self.q_network.reset_noise()
                    self.target_network.reset_noise()
        if done:
            if len(self.n_step_buffer) >= self.n_step:
                self.n_step_buffer.popleft()
            while len(self.n_step_buffer) > 0:
                result = self._aggregate_n_step()
                self.replay.add(*result)
                self.n_step_buffer.popleft()
            self.state_stack = deque([processed] * self.stack_size, maxlen=self.stack_size)
            current_state = self._stack_frames()
            self.n_step_buffer.clear()

        self.frame_count += 1
        action = self._select_action(current_state)
        self.last_state = current_state.copy()
        self.last_action = action

        return action

    def save_model(self, filename):
        torch.save(self.q_network.state_dict(), filename)
