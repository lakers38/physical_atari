# agent_meme.py
"""
MEME: Memory-Based Efficient Exploration (DeepMind, 2022)

New adjustments:
• Intrinsic reward clamp after episodic × RND combination.
• LSTM hidden state persisted across environment steps.
• TD σ updated with 0.99 EMA, priorities use mean |TD|.
• EMA copy of network parameters (η=0.995) for evaluation.
"""

from __future__ import annotations

import math
import os
import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_utils import preprocess_batch
from vector_agents import VectorAgent


# -----------------------------------------------------------------------------#
# Building blocks
# -----------------------------------------------------------------------------#


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

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        return noise.sign().mul_(noise.abs().sqrt_())

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        noise_matrix = torch.outer(eps_out, eps_in)
        self.weight_epsilon[:] = noise_matrix
        self.bias_epsilon[:] = eps_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class NFResidualBlock(nn.Module):
    def __init__(self, channels: int, alpha: float = 0.2, drop_rate: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.drop_rate = drop_rate
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.training and self.drop_rate > 0.0:
            mask = torch.rand_like(out[:, :1, :, :]) > self.drop_rate
            out = out * mask / (1.0 - self.drop_rate)
        return residual + self.alpha * out


class RunningMeanStd:
    def __init__(self, shape, eps: float = 1e-4):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = eps

    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.size(0)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.sqrt() + 1e-6)


# -----------------------------------------------------------------------------#
# Intrinsic reward modules
# -----------------------------------------------------------------------------#


class EpisodicMemory:
    def __init__(self, embedding_dim: int, max_memory: int = 3000, k: int = 10, cluster_distance: float = 0.008):
        self.embedding_dim = embedding_dim
        self.max_memory = max_memory
        self.k = k
        self.cluster_distance = cluster_distance
        self.memory: List[np.ndarray] = []

    def reset(self) -> None:
        self.memory.clear()

    def compute_bonus(self, embedding: np.ndarray) -> float:
        if not self.memory:
            self.memory.append(embedding)
            return 1.0
        dists = np.linalg.norm(np.asarray(self.memory) - embedding, axis=1)
        k = min(self.k, len(dists))

        if k == 0:
            nearest = dists
        else:
            nearest = np.partition(dists, k - 1)[:k]
        mean_dist = float(np.mean(nearest))
        normed = mean_dist / (self.cluster_distance + 1e-6)
        bonus = 1.0 / (normed + 1.0)

        self.memory.append(embedding)
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
        return float(np.clip(bonus, 0.0, 1.0))


class RNDModule(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], embedding_dim: int = 128):
        super().__init__()
        c, h, w = obs_shape
        self.target = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, embedding_dim),
        )
        self.predictor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-4)
        self.running_stats = RunningMeanStd(embedding_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = self.predictor(x)
        with torch.no_grad():
            target = self.target(x)
        return pred, target

    def update(self, observations: torch.Tensor) -> torch.Tensor:
        pred, target = self(observations)
        loss = F.mse_loss(pred, target, reduction="mean")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            diff = pred - target
            self.running_stats.update(diff)
        return loss.detach()

    def bonus(self, observations: torch.Tensor) -> torch.Tensor:
        pred, target = self(observations)
        with torch.no_grad():
            diff = pred - target
            bonus = torch.sum(diff**2, dim=-1)
            return bonus


# -----------------------------------------------------------------------------#
# Prioritised sequence replay
# -----------------------------------------------------------------------------#


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward_ext: float
    reward_int: float
    done: bool
    policy_idx: int
    policy_probs: np.ndarray


class PrioritisedSequenceReplay:
    def __init__(
        self,
        capacity: int,
        seq_len: int,
        burn_in: int,
        stack_size: int,
        obs_shape: Tuple[int, int],
        alpha: float,
        beta_start: float,
        beta_increment: float,
    ):
        self.capacity = capacity
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.stack_size = stack_size
        self.obs_shape = obs_shape
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        self.storage: List[List[Transition]] = []
        self.priorities: List[float] = []
        self.position = 0

    def add_episode(self, episode: List[Transition]) -> None:
        priority = max(self.priorities, default=1.0)
        if len(self.storage) < self.capacity:
            self.storage.append(episode)
            self.priorities.append(priority)
        else:
            self.storage[self.position] = episode
            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        if not self.storage:
            return [], [], []
        probs = np.array(self.priorities, dtype=np.float64) ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.storage), size=batch_size, p=probs)
        beta = min(1.0, self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (len(self.storage) * probs[indices]) ** (-beta)
        weights /= weights.max()
        sequences = []
        for idx in indices:
            episode = self.storage[idx]
            if len(episode) <= self.seq_len + self.burn_in:
                sequences.append(episode)
            else:
                start = random.randint(0, len(episode) - (self.seq_len + self.burn_in))
                sequences.append(episode[start : start + self.seq_len + self.burn_in])
        return sequences, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices: Iterable[int], priorities: torch.Tensor) -> None:
        for idx, priority in zip(indices, priorities.cpu().numpy()):
            self.priorities[idx] = float(priority)


# -----------------------------------------------------------------------------#
# MEME network with LSTM
# -----------------------------------------------------------------------------#


class MEMENetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_actions: int,
        num_policies: int,
        hidden_dim: int = 512,
        depth: int = 6,
        lstm_hidden: int = 1024,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_policies = num_policies

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([NFResidualBlock(128) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.torso = nn.Sequential(nn.Flatten(), nn.Linear(128, hidden_dim), nn.ReLU())
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True)

        self.extrinsic_head = NoisyLinear(lstm_hidden, num_policies * num_actions)
        self.intrinsic_head = NoisyLinear(lstm_hidden, num_policies * num_actions)
        self.policy_head = nn.Linear(lstm_hidden, num_policies * num_actions)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch, time = x.shape[:2]
        x = x.view(batch * time, *x.shape[2:])
        out = self.stem(x)
        for block in self.blocks:
            out = block(out)
        out = self.pool(out)
        out = self.torso(out)
        out = out.view(batch, time, -1)
        out, hidden = self.lstm(out, hidden)
        out = out.reshape(batch * time, -1)
        ext = self.extrinsic_head(out).view(batch, time, self.num_policies, self.num_actions)
        intr = self.intrinsic_head(out).view(batch, time, self.num_policies, self.num_actions)
        policy_logits = self.policy_head(out).view(batch, time, self.num_policies, self.num_actions)
        return ext, intr, policy_logits, hidden


# -----------------------------------------------------------------------------#
# Discounted-UCB meta-controller
# -----------------------------------------------------------------------------#


class DiscountedUCBBandit:
    def __init__(self, num_policies: int, gamma: float = 0.99, c: float = 1.5):
        self.num_policies = num_policies
        self.gamma = gamma
        self.c = c
        self.estimates = np.zeros(num_policies, dtype=np.float64)
        self.counts = np.zeros(num_policies, dtype=np.float64)
        self.step = 0

    def update(self, policy_idx: int, reward: float) -> None:
        self.step += 1
        self.counts *= self.gamma
        self.estimates *= self.gamma
        self.counts[policy_idx] += 1.0
        self.estimates[policy_idx] += reward

    def sample(self, rng: np.random.RandomState, epsilon: float = 0.05) -> int:
        if rng.rand() < epsilon or np.all(self.counts < 1e-6):
            return int(rng.randint(self.num_policies))
        counts = np.maximum(self.counts, 1e-3)
        mean = self.estimates / counts
        bonus = self.c * np.sqrt(np.log(self.step + 1.0) / counts)
        return int(np.argmax(mean + bonus))


# -----------------------------------------------------------------------------#
# MEME core
# -----------------------------------------------------------------------------#


class MEMECore:
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
        buffer_capacity: int = 4000,
        seq_len: int = 80,
        burn_in: int = 20,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
        adam_eps: float = 1e-5,
        num_policies: int = 16,
        beta_low: float = 0.1,
        beta_high: float = 1.5,
        gamma_low: float = 0.90,
        gamma_high: float = 0.997,
        eta: float = 0.5,
        lambda_val: float = 0.9,
        tolerance_kappa: float = 0.05,
        train_interval: int = 8,
        epsilon_start: float = 0.4,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 1_000_000,
        trust_alpha: float = 2.0,
        td_norm_eps: float = 0.01,
        policy_kl_clip: float = 0.2,
        distill_temperature: float = 0.25,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
        priority_beta_increment: float = 1e-6,
        rnd_embedding: int = 128,
        episodic_embedding: int = 32,
        meta_discount: float = 0.99,
        meta_bonus_c: float = 1.5,
        meta_epsilon: float = 0.05,
        ema_decay: float = 0.995,
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
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.eta = eta
        self.lambda_val = lambda_val
        self.tolerance_kappa = tolerance_kappa
        self.trust_alpha = trust_alpha
        self.td_norm_eps = td_norm_eps
        self.policy_kl_clip = policy_kl_clip
        self.distill_temperature = distill_temperature
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        self.priority_beta_increment = priority_beta_increment
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.meta_epsilon = meta_epsilon
        self.ema_decay = ema_decay
        self.train_interval = max(1, int(train_interval))
        self._train_call_count = 0
        self.last_metrics: Dict[str, float] = {}

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu}')
            torch.cuda.manual_seed_all(seed)
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        betas = np.geomspace(beta_low, beta_high, num_policies).astype(np.float32)
        gammas = np.linspace(gamma_low, gamma_high, num_policies).astype(np.float32)
        self.policy_betas = torch.tensor(betas, device=self.device)
        self.policy_gammas = torch.tensor(gammas, device=self.device)
        self.num_policies = num_policies

        self.network = MEMENetwork(stack_size, num_actions, num_policies).to(self.device)
        self.ema_network = MEMENetwork(stack_size, num_actions, num_policies).to(self.device)
        self.ema_network.load_state_dict(self.network.state_dict())
        for p in self.ema_network.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, eps=adam_eps)

        self.replay = PrioritisedSequenceReplay(
            capacity=buffer_capacity,
            seq_len=seq_len,
            burn_in=burn_in,
            stack_size=stack_size,
            obs_shape=(obs_height, obs_width),
            alpha=priority_alpha,
            beta_start=priority_beta,
            beta_increment=priority_beta_increment,
        )

        self.state_stacks = np.zeros((num_envs, stack_size, obs_height, obs_width), dtype=np.uint8)
        self.last_actions = np.full(num_envs, -1, dtype=np.int64)
        self.last_policy_indices = np.zeros(num_envs, dtype=np.int64)
        self.last_policy_dists = np.zeros((num_envs, num_actions), dtype=np.float32)
        self.lstm_hidden_h = torch.zeros(self.num_envs, 1024, device=self.device)
        self.lstm_hidden_c = torch.zeros(self.num_envs, 1024, device=self.device)
        self.episode_buffers: List[List[Transition]] = [[] for _ in range(num_envs)]

        self.rnd = RNDModule((stack_size, obs_height, obs_width), embedding_dim=rnd_embedding).to(self.device)
        self.embedding_head = nn.Sequential(
            nn.Conv2d(stack_size, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, episodic_embedding),
        ).to(self.device)
        self.embedding_optimizer = torch.optim.Adam(self.embedding_head.parameters(), lr=1e-4)
        self.episodic_modules = [EpisodicMemory(episodic_embedding) for _ in range(num_envs)]
        self.int_running = RunningMeanStd(1)

        self.running_td_std = torch.ones(num_policies, device=self.device)
        self.bandit = DiscountedUCBBandit(num_policies, gamma=meta_discount, c=meta_bonus_c)
        self.rng = np.random.RandomState(seed)
        self.frame_count = 0
        self.training_steps = 0
        self.data_dir = data_dir or os.getcwd()

        if load_file is not None and os.path.exists(load_file):
            self.load_model(load_file)

    # ------------------------------------------------------------------#
    # Reset / actions with persistent LSTM state
    # ------------------------------------------------------------------#

    def reset(self, observations: np.ndarray) -> None:
        processed = preprocess_batch(observations, self.obs_height, self.obs_width)
        for env in range(self.num_envs):
            stack = np.repeat(processed[env][None, ...], self.stack_size, axis=0)
            self.state_stacks[env] = stack
            self.last_actions[env] = -1
            self.episode_buffers[env].clear()
            self.episodic_modules[env].reset()
        self.lstm_hidden_h.zero_()
        self.lstm_hidden_c.zero_()

    def _epsilon(self) -> float:
        fraction = min(self.frame_count / float(self.epsilon_decay), 1.0)
        return float(self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start))

    def act(self, observations: np.ndarray) -> np.ndarray:
        processed = preprocess_batch(observations, self.obs_height, self.obs_width)
        self.state_stacks = np.roll(self.state_stacks, shift=-1, axis=1)
        self.state_stacks[:, -1, :, :] = processed

        epsilon = self._epsilon()
        obs_tensor = torch.from_numpy(self.state_stacks).to(self.device, dtype=torch.float32) / 255.0
        obs_tensor = obs_tensor.unsqueeze(1)

        hidden = (
            self.lstm_hidden_h.unsqueeze(0).contiguous(),
            self.lstm_hidden_c.unsqueeze(0).contiguous(),
        )

        with torch.no_grad():
            self.network.reset_noise()
            ext, intr, logits, hidden = self.network(obs_tensor, hidden)
            ext = ext.squeeze(1)
            intr = intr.squeeze(1)
            logits = logits.squeeze(1)
            q_combined = ext + self.policy_betas.view(1, -1, 1) * intr
            greedy_actions = torch.argmax(q_combined, dim=-1)

        self.lstm_hidden_h = hidden[0].detach().squeeze(0)
        self.lstm_hidden_c = hidden[1].detach().squeeze(0)

        actions = np.empty(self.num_envs, dtype=np.int64)
        policy_indices = np.empty(self.num_envs, dtype=np.int64)
        policy_dists = np.empty((self.num_envs, self.num_actions), dtype=np.float32)

        for env in range(self.num_envs):
            policy_idx = self.bandit.sample(self.rng, epsilon=self.meta_epsilon)
            policy_indices[env] = policy_idx
            greedy = int(greedy_actions[env, policy_idx].item())
            dist = np.ones(self.num_actions, dtype=np.float32) * (epsilon / self.num_actions)
            dist[greedy] += 1.0 - epsilon
            policy_dists[env] = dist
            if random.random() < epsilon:
                actions[env] = random.randrange(self.num_actions)
            else:
                actions[env] = greedy

        self.last_actions = actions.copy()
        self.last_policy_indices = policy_indices.copy()
        self.last_policy_dists = policy_dists.copy()
        return actions

    # ------------------------------------------------------------------#
    # Intrinsic reward
    # ------------------------------------------------------------------#

    def _compute_intrinsic(self, stack: torch.Tensor, env: int) -> float:
        with torch.no_grad():
            embedding = self.embedding_head(stack.unsqueeze(0)).cpu().numpy()[0]
        bonus_epi = self.episodic_modules[env].compute_bonus(embedding)
        rnd_bonus = self.rnd.bonus(stack.unsqueeze(0)).cpu()
        self.int_running.update(rnd_bonus.unsqueeze(-1))
        rnd_normed = float(rnd_bonus / (self.int_running.var.sqrt() + 1e-6))
        int_reward = float(bonus_epi * rnd_normed)
        return float(np.clip(int_reward, 0.0, 1.0))

    # ------------------------------------------------------------------#
    # Observe
    # ------------------------------------------------------------------#

    def observe(
        self,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
        infos: Iterable[Dict],
    ) -> None:
        processed = preprocess_batch(next_observations, self.obs_height, self.obs_width)
        next_stacks = self.state_stacks.copy()
        next_stacks = np.roll(next_stacks, shift=-1, axis=1)
        next_stacks[:, -1, :, :] = processed

        for env in range(self.num_envs):
            action = self.last_actions[env]
            if action == -1:
                continue
            policy_idx = int(self.last_policy_indices[env])
            policy_probs = self.last_policy_dists[env]
            done = bool(terminations[env] or truncations[env])

            stack_tensor = torch.from_numpy(self.state_stacks[env]).to(self.device, dtype=torch.float32) / 255.0
            intrinsic_reward = self._compute_intrinsic(stack_tensor, env)

            transition = Transition(
                obs=self.state_stacks[env].copy(),
                action=action,
                reward_ext=float(rewards[env]),
                reward_int=intrinsic_reward,
                done=done,
                policy_idx=policy_idx,
                policy_probs=policy_probs.copy(),
            )
            self.episode_buffers[env].append(transition)
            self.state_stacks[env] = next_stacks[env]

            if done:
                self.replay.add_episode(self.episode_buffers[env])
                episode_return = sum(t.reward_ext for t in self.episode_buffers[env])
                self.bandit.update(policy_idx, episode_return)
                self.episode_buffers[env] = []
                self.episodic_modules[env].reset()
                self.lstm_hidden_h[env].zero_()
                self.lstm_hidden_c[env].zero_()

        self.frame_count += self.num_envs

    # ------------------------------------------------------------------#
    # Training
    # ------------------------------------------------------------------#

    def _soft_watkins_returns(
        self,
        q_online: torch.Tensor,
        rewards_ext: torch.Tensor,
        rewards_int: torch.Tensor,
        dones: torch.Tensor,
        policy_probs: torch.Tensor,
    ) -> torch.Tensor:
        T = q_online.size(0)
        num_policies = q_online.size(1)
        running = torch.zeros(num_policies, device=self.device)
        slices: List[torch.Tensor] = []

        for t in reversed(range(T)):
            reward = rewards_ext[t] + self.policy_betas * rewards_int[t]
            done = dones[t]
            probs = policy_probs[t]

            q_t = q_online[t]  # [num_policies, num_actions]
            greedy_q, greedy_idx = q_t.max(dim=-1, keepdim=True)  # [num_policies, 1]
            tolerance = greedy_q - self.tolerance_kappa * greedy_q.abs()
            q_taken = q_t[torch.arange(num_policies), greedy_idx.squeeze(-1)]
            mask = q_taken >= tolerance.squeeze(-1)

            expectation = torch.sum(probs * q_t, dim=-1)
            boot = (
                (1 - done)
                * self.policy_gammas
                * (
                    mask.float() * ((1 - self.lambda_val) * expectation + self.lambda_val * running)
                    + (1 - mask.float()) * expectation
                )
            )
            running = (reward + boot).detach().clamp_(-1e3, 1e3)
            slices.append(running.unsqueeze(0))

        if slices:
            result = torch.cat(list(reversed(slices)), dim=0)
            return torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.zeros(T, num_policies, device=self.device)

    def _trust_region(
        self,
        q_online: torch.Tensor,
        q_target: torch.Tensor,
        targets: torch.Tensor,
        sigma: torch.Tensor,
        is_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        td = targets - q_online
        td = torch.nan_to_num(td, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-1e3, 1e3)

        sigma = torch.nan_to_num(sigma, nan=self.td_norm_eps, posinf=1.0, neginf=1.0)
        sigma = sigma.clamp(min=self.td_norm_eps, max=1e3)

        outside = (q_online - q_target).abs() > (self.trust_alpha * sigma)
        direction = torch.sign(q_online - q_target) != torch.sign(td)
        mask = ~(outside & direction)

        normalised_td = (td / sigma) * is_weights
        normalised_td = torch.nan_to_num(normalised_td, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-1e3, 1e3)
        return mask, normalised_td

    def _ema_update(self) -> None:
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)

    def train_step(self) -> None:
        self._train_call_count += 1
        if self._train_call_count % self.train_interval != 0:
            return

        self.last_metrics = {}

        sequences, indices, weights = self.replay.sample(self.batch_size)
        if not sequences:
            return

        priority_records: List[Tuple[int, torch.Tensor]] = []
        tdstd_updates: Dict[int, List[torch.Tensor]] = {}
        device_weights = weights.to(self.device)

        valid_entries: List[Tuple[List[Transition], int, torch.Tensor]] = []
        for seq, idx, w in zip(sequences, indices, device_weights):
            if len(seq) <= self.burn_in + 1:
                priority_records.append((idx, torch.tensor(1.0, device=self.device)))
                continue
            valid_entries.append((seq, idx, w))

        valid_count = len(valid_entries)
        if valid_count == 0:
            for idx, priority in priority_records:
                self.replay.update_priorities([idx], priority.detach().unsqueeze(0))
            return

        inv_count = 1.0 / float(valid_count)
        self.optimizer.zero_grad()
        self.network.reset_noise()
        total_loss_vals: List[float] = []
        behaviour_loss_vals: List[float] = []
        aux_loss_vals: List[float] = []
        policy_loss_vals: List[float] = []
        td_abs_mean_vals: List[float] = []
        ratio_abs_mean_vals: List[float] = []

        for seq, idx, w in valid_entries:

            states = torch.from_numpy(np.stack([t.obs for t in seq])).to(self.device, dtype=torch.float32) / 255.0
            states = states.unsqueeze(0)

            ext, intr, logits, hidden = self.network(states[:, : self.burn_in])
            ext, intr, logits, _ = self.network(states[:, self.burn_in :], hidden)
            ext = ext.squeeze(0)
            intr = intr.squeeze(0)
            logits = logits.squeeze(0)
            q_online = ext + self.policy_betas.view(1, -1, 1) * intr

            learn = seq[self.burn_in :]
            actions = torch.tensor([t.action for t in learn], device=self.device, dtype=torch.long)
            rewards_ext = torch.tensor([t.reward_ext for t in learn], device=self.device, dtype=torch.float32)
            rewards_int = torch.tensor([t.reward_int for t in learn], device=self.device, dtype=torch.float32)
            dones = torch.tensor([t.done for t in learn], device=self.device, dtype=torch.float32)
            policy_idx = torch.tensor([t.policy_idx for t in learn], device=self.device, dtype=torch.long)
            policy_probs_np = np.stack([t.policy_probs for t in learn], dtype=np.float32)
            policy_probs = torch.from_numpy(policy_probs_np).to(self.device)

            T = actions.size(0)
            returns = self._soft_watkins_returns(q_online, rewards_ext, rewards_int, dones, policy_probs)
            returns = torch.nan_to_num(returns, nan=0.0, posinf=1e6, neginf=-1e6)
            q_selected = q_online[torch.arange(T), policy_idx, actions]
            targets = returns[torch.arange(T), policy_idx]
            sigma = self.running_td_std[policy_idx]
            sigma = torch.nan_to_num(sigma, nan=self.td_norm_eps, posinf=1.0, neginf=1.0).clamp_(min=self.td_norm_eps, max=1e3)

            mask, normalised_td = self._trust_region(q_selected, q_selected.detach(), targets, sigma, w)
            if not torch.isfinite(normalised_td).all():
                print("⚠️ NaN in normalised_td", normalised_td.min().item(), normalised_td.max().item())
            loss_behaviour = (mask.float() * (normalised_td**2)).mean()

            actions_expanded = actions.view(T, 1, 1).expand(-1, self.num_policies, 1)
            q_taken_all = q_online.gather(dim=2, index=actions_expanded).squeeze(-1)

            # --- Stable TD ratio computation ---
            returns = torch.nan_to_num(returns, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-1e3, 1e3)
            q_taken_all = torch.nan_to_num(q_taken_all, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-1e3, 1e3)

            all_td = returns - q_taken_all
            all_td = torch.nan_to_num(all_td, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-1e3, 1e3)

            all_sigma = (
                torch.nan_to_num(self.running_td_std, nan=self.td_norm_eps, posinf=1.0, neginf=1.0)
                .clamp(min=self.td_norm_eps, max=1e3)
                .unsqueeze(0)
            )

            ratio = all_td / all_sigma
            ratio = torch.nan_to_num(ratio, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-1e3, 1e3)
            loss_all = ratio.square().mean()

            teacher = policy_probs
            log_probs = F.log_softmax(logits, dim=-1)
            teacher_expanded = teacher.unsqueeze(1).expand(-1, log_probs.size(1), -1)
            kl = torch.sum(teacher_expanded * (torch.log(teacher_expanded + 1e-8) - log_probs), dim=-1)
            kl_mask = (kl <= self.policy_kl_clip).float()
            policy_loss = -(kl_mask * torch.sum(teacher_expanded * log_probs, dim=-1)).mean()

            loss = self.eta * loss_behaviour + (1 - self.eta) * loss_all + policy_loss
            (loss * inv_count).backward()
            total_loss_vals.append(loss.detach().item())
            behaviour_loss_vals.append(loss_behaviour.detach().item())
            aux_loss_vals.append(loss_all.detach().item())
            policy_loss_vals.append(policy_loss.detach().item())
            ratio_abs_mean_vals.append(ratio.abs().mean().detach().item())

            with torch.no_grad():
                td_abs = normalised_td.detach().abs()
                priority = td_abs.mean().clamp(min=1e-3)
                td_abs_mean_vals.append(td_abs.mean().item())
                priority_records.append((idx, priority))
                unique = torch.unique(policy_idx)
                for pol in unique.tolist():
                    pol_mask = policy_idx == pol
                    if pol_mask.any():
                        std = td_abs[pol_mask].std(unbiased=False).clamp(min=self.td_norm_eps)
                        tdstd_updates.setdefault(pol, []).append(std)

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()
        self._ema_update()

        if tdstd_updates:
            with torch.no_grad():
                for pol, std_list in tdstd_updates.items():
                    avg_std = torch.stack(std_list).mean()
                    self.running_td_std[pol] = 0.99 * self.running_td_std[pol] + 0.01 * avg_std

        if total_loss_vals:
            def mean(values: List[float]) -> float:
                return float(sum(values) / max(1, len(values)))

            metrics = {
                "loss/total": mean(total_loss_vals),
                "loss/behaviour": mean(behaviour_loss_vals),
                "loss/aux_td": mean(aux_loss_vals),
                "loss/policy": mean(policy_loss_vals),
                "stats/td_abs_mean": mean(td_abs_mean_vals),
                "stats/ratio_abs_mean": mean(ratio_abs_mean_vals),
                "stats/running_td_std_mean": float(self.running_td_std.mean().item()),
                "train/epsilon": float(self._epsilon()),
                "train/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "train/interval": float(self.train_interval),
                "train/updates": float(self.training_steps),
                "train/batch_count": float(len(valid_entries)),
            }
            self.last_metrics = metrics

        for idx, priority in priority_records:
            self.replay.update_priorities([idx], priority.detach().unsqueeze(0))

        self.training_steps += 1

    # ------------------------------------------------------------------#
    # Persistence / eval
    # ------------------------------------------------------------------#

    def save_model(self, path: str) -> None:
        torch.save(self.ema_network.state_dict(), path)

    def load_model(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.ema_network.load_state_dict(state_dict)

    def pop_metrics(self) -> Dict[str, float]:
        metrics = self.last_metrics
        self.last_metrics = {}
        return metrics


# -----------------------------------------------------------------------------#
# Agent wrappers
# -----------------------------------------------------------------------------#


class Agent:
    def __init__(self, data_dir=None, seed=0, num_actions=18, total_frames=1_000_000, **kwargs):
        self.core = MEMECore(
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
                [],
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

    def get_metrics(self) -> Dict[str, float]:
        return self.core.pop_metrics()


class VectorMEMEAgent(VectorAgent):
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
        self.core = MEMECore(
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
            raise ValueError(f"VectorMEMEAgent initialised for {self.num_envs} envs; received {num_envs}.")
        return None

    def act(self, observations: np.ndarray) -> np.ndarray:
        return self.core.act(observations)

    def observe(self, next_observations, rewards, terminations, truncations, infos):
        self.core.observe(next_observations, rewards, terminations, truncations, infos)

    def train_step(self) -> None:
        self.core.train_step()

    def save_model(self, path: str) -> None:
        self.core.save_model(path)

    def load_model(self, path: str) -> None:
        self.core.load_model(path)

    def get_metrics(self) -> Dict[str, float]:
        return self.core.pop_metrics()
