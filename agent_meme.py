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
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Protocol

import contextlib
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


class ScaledStdConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gain=1.0, eps=1e-5):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.gain = gain
        self.eps = eps

    def _scaled_weight(self):
        fan_in = self.weight[0].numel()
        mean = self.weight.mean(dim=(1, 2, 3), keepdim=True)
        var = self.weight.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        scale = self.gain / math.sqrt(fan_in)
        return (self.weight - mean) * scale / torch.sqrt(var + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._scaled_weight()
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: float = 0.5):
        super().__init__()
        hidden = max(1, int(channels * reduction))
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x.mean(dim=(2, 3)))
        return x * scale.unsqueeze(-1).unsqueeze(-1)


class NFBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, alpha: float = 0.2, se_ratio: float = 0.5):
        super().__init__()
        self.stride = stride
        self.alpha = alpha
        self.act = nn.GELU()
        self.conv1 = ScaledStdConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ScaledStdConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.se = SqueezeExcite(out_channels, se_ratio) if se_ratio else None
        if stride > 1 or in_channels != out_channels:
            self.downsample = ScaledStdConv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(x)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
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
    def __init__(
        self,
        embedding_dim: int,
        max_memory: int = 3000,
        k: int = 10,
        cluster_distance: float = 0.008,
        kernel_epsilon: float = 0.0001,
        pseudo_count_c: float = 0.001,
        similarity_max: float = 8.0,
        running_mean_decay: float = 0.99,
    ):
        self.embedding_dim = embedding_dim
        self.max_memory = max_memory
        self.k = k
        self.cluster_distance = cluster_distance
        self.kernel_epsilon = kernel_epsilon
        self.pseudo_count_c = pseudo_count_c
        self.similarity_max = similarity_max
        self.running_mean_decay = running_mean_decay
        self.memory: List[np.ndarray] = []
        self.running_d2 = 1.0

    def reset(self) -> None:
        self.memory.clear()
        self.running_d2 = 1.0

    def compute_bonus(self, embedding: np.ndarray) -> float:
        embedding = embedding.astype(np.float32, copy=False)
        if not self.memory:
            self.memory.append(embedding.copy())
            return 1.0

        memory_array = np.asarray(self.memory, dtype=np.float32)
        dists = np.sum((memory_array - embedding) ** 2, axis=1)
        k = min(self.k, len(dists))
        if k <= 0:
            self.memory.append(embedding.copy())
            if len(self.memory) > self.max_memory:
                self.memory.pop(0)
            return 1.0

        idx = np.argpartition(dists, k - 1)[:k]
        dk = dists[idx]

        mean_dist = float(np.mean(dk))
        self.running_d2 = (
            self.running_mean_decay * self.running_d2 + (1.0 - self.running_mean_decay) * max(mean_dist, 1e-6)
        )
        norm = self.running_d2 if self.running_d2 > 1e-6 else 1e-6
        dn = dk / norm
        dn = np.maximum(dn - self.cluster_distance, 0.0)
        kv = self.kernel_epsilon / (dn + self.kernel_epsilon)
        similarity = math.sqrt(np.sum(kv) + self.pseudo_count_c)

        if similarity > self.similarity_max:
            bonus = 0.0
        else:
            bonus = 1.0 / similarity

        self.memory.append(embedding.copy())
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
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=6e-4)
        self.running_stats = RunningMeanStd(embedding_dim)
        self.error_stats = RunningMeanStd(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = self.predictor(x)
        with torch.no_grad():
            target = self.target(x)
        return pred, target

    def update(self, observations: torch.Tensor) -> torch.Tensor:
        pred, target = self(observations)
        diff = pred - target
        errors = torch.sum(diff**2, dim=1, keepdim=True)
        loss = errors.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.running_stats.update(diff)
            self.error_stats.update(errors.detach().cpu())
        return loss.detach()

    def bonus(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred, target = self(observations)
        with torch.no_grad():
            diff = pred - target
            errors = torch.sum(diff**2, dim=1)
            self.error_stats.update(errors.unsqueeze(1).detach().cpu())
            mean = self.error_stats.mean.to(errors.device)
            std = self.error_stats.var.sqrt().to(errors.device) + 1e-6
            normalized = (errors - mean) / std
            return normalized, errors


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
    features: np.ndarray


@dataclass
class EpisodeBatch:
    obs: np.ndarray
    actions: np.ndarray
    reward_ext: np.ndarray
    reward_int: np.ndarray
    done: np.ndarray
    policy_idx: np.ndarray
    policy_probs: np.ndarray
    features: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    def slice(self, start: int, end: int) -> "EpisodeBatch":
        return EpisodeBatch(
            obs=self.obs[start:end],
            actions=self.actions[start:end],
            reward_ext=self.reward_ext[start:end],
            reward_int=self.reward_int[start:end],
            done=self.done[start:end],
            policy_idx=self.policy_idx[start:end],
            policy_probs=self.policy_probs[start:end],
            features=self.features[start:end],
        )


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
        self.storage: List[EpisodeBatch] = []
        self.priorities: List[float] = []
        self.position = 0

    @property
    def size(self):
        return len(self.storage)

    def add_episode(self, episode: List[Transition]) -> None:
        priority = max(self.priorities, default=1.0)
        if not episode:
            return
        obs = np.stack([t.obs for t in episode]).astype(np.uint8, copy=False)
        actions = np.fromiter((t.action for t in episode), dtype=np.int64)
        reward_ext = np.fromiter((t.reward_ext for t in episode), dtype=np.float32)
        reward_int = np.fromiter((t.reward_int for t in episode), dtype=np.float32)
        done = np.fromiter((t.done for t in episode), dtype=np.bool_)
        policy_idx = np.fromiter((t.policy_idx for t in episode), dtype=np.int64)
        policy_probs = np.stack([t.policy_probs for t in episode]).astype(np.float32, copy=False)
        features = np.stack([t.features for t in episode]).astype(np.float32, copy=False)

        total_len = max(1, self.seq_len + self.burn_in)
        num_steps = actions.shape[0]
        for start in range(0, num_steps, total_len):
            end = min(start + total_len, num_steps)
            chunk = EpisodeBatch(
                obs=np.ascontiguousarray(obs[start:end]),
                actions=np.ascontiguousarray(actions[start:end]),
                reward_ext=np.ascontiguousarray(reward_ext[start:end]),
                reward_int=np.ascontiguousarray(reward_int[start:end]),
                done=np.ascontiguousarray(done[start:end]),
                policy_idx=np.ascontiguousarray(policy_idx[start:end]),
                policy_probs=np.ascontiguousarray(policy_probs[start:end]),
                features=np.ascontiguousarray(features[start:end]),
            )
            if len(self.storage) < self.capacity:
                self.storage.append(chunk)
                self.priorities.append(priority)
            else:
                self.storage[self.position] = chunk
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
            total_len = self.seq_len + self.burn_in
            if episode.length <= total_len:
                sequences.append(episode)
            else:
                start = random.randint(0, episode.length - total_len)
                sequences.append(episode.slice(start, start + total_len))
        return sequences, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices: Iterable[int], priorities: torch.Tensor) -> None:
        for idx, priority in zip(indices, priorities.cpu().numpy()):
            self.priorities[idx] = float(priority)


class ReplayBufferProtocol(Protocol):
    def add_episode(self, episode: List[Transition]) -> None: ...

    def sample(self, batch_size: int): ...

    def update_priorities(self, indices: Iterable[int], priorities: torch.Tensor) -> None: ...


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
        lstm_hidden: int = 1024,
        feature_dim: int = 68,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_policies = num_policies
        self.feature_dim = feature_dim

        self.stem = nn.Sequential(
            ScaledStdConv2d(in_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
        )
        config = [
            (2, 64, 1),
            (3, 128, 2),
            (4, 128, 2),
            (4, 64, 2),
        ]
        stages = []
        in_ch = 64
        for blocks, channels, stride in config:
            layer = []
            for i in range(blocks):
                layer.append(NFBlock(in_ch, channels, stride if i == 0 else 1))
                in_ch = channels
            stages.append(nn.Sequential(*layer))
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)

        torso_input = in_ch
        if feature_dim > 0:
            self.feature_mlp = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.GELU(),
            )
            torso_input += 256
        else:
            self.feature_mlp = None

        self.torso = nn.Sequential(
            nn.Linear(torso_input, hidden_dim),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True)

        def build_head(noisy: bool):
            layers = [
                nn.Linear(lstm_hidden, 1024),
                nn.GELU(),
                nn.Linear(1024, 1024),
                nn.GELU(),
            ]
            if noisy:
                layers.append(NoisyLinear(1024, num_policies * num_actions))
            else:
                layers.append(nn.Linear(1024, num_policies * num_actions))
            return nn.Sequential(*layers)

        self.extrinsic_head = build_head(noisy=True)
        self.intrinsic_head = build_head(noisy=True)
        self.policy_head = build_head(noisy=False)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch, time = x.shape[:2]
        out = x.view(batch * time, *x.shape[2:])
        out = self.stem(out)
        for stage in self.stages:
            out = stage(out)
        out = self.pool(out).view(batch, time, -1)
        if self.feature_mlp is not None and features is not None:
            feat = features.view(batch * time, -1)
            feat = self.feature_mlp(feat).view(batch, time, -1)
            out = torch.cat([out, feat], dim=-1)
        out = self.torso(out)
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
    @staticmethod
    def _bandit_sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _compute_beta_schedule(num_policies: int, beta_max: float) -> np.ndarray:
        if num_policies <= 0:
            return np.zeros(0, dtype=np.float32)
        if num_policies == 1:
            return np.array([beta_max], dtype=np.float32)
        betas = np.zeros(num_policies, dtype=np.float32)
        betas[-1] = beta_max
        if num_policies > 2:
            denom = max(1, num_policies - 2)
            for i in range(1, num_policies - 1):
                x = 8.0 * (2 * i - (num_policies - 2)) / denom
                betas[i] = beta_max * MEMECore._bandit_sigmoid(x)
        return betas

    @staticmethod
    def _compute_gamma_schedule(num_policies: int, gamma_min: float, gamma_max: float) -> np.ndarray:
        if num_policies <= 0:
            return np.zeros(0, dtype=np.float32)
        gamma_min = min(gamma_min, gamma_max)
        epsilon = 1e-9
        def clamp(x):
            return max(epsilon, min(1 - epsilon, x))
        log1_max = math.log(clamp(1.0 - gamma_max))
        log1_min = math.log(clamp(1.0 - gamma_min))
        if num_policies == 1:
            return np.array([gamma_max], dtype=np.float32)
        gammas = np.zeros(num_policies, dtype=np.float32)
        for i in range(num_policies):
            a = (num_policies - 1 - i) / (num_policies - 1)
            b = i / (num_policies - 1)
            inside = a * log1_max + b * log1_min
            gammas[i] = 1.0 - math.exp(inside)
        return gammas

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
        seq_len: int = 160,
        burn_in: int = 0,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        adam_eps: float = 1e-8,
        num_policies: int = 16,
        beta_low: float = 0.1,
        beta_high: float = 0.1,
        gamma_low: float = 0.97,
        gamma_high: float = 0.9997,
        eta: float = 0.5,
        lambda_val: float = 0.95,
        tolerance_kappa: float = 0.01,
        train_interval: int = 6,
        epsilon_start: float = 0.4,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 1_000_000,
        trust_alpha: float = 2.0,
        td_norm_eps: float = 0.01,
        policy_kl_clip: float = 0.5,
        distill_temperature: float = 0.25,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
        priority_beta_increment: float = 1e-6,
        rnd_embedding: int = 128,
        episodic_embedding: int = 32,
        meta_discount: float = 0.999,
        meta_bonus_c: float = 1.0,
        meta_epsilon: float = 0.5,
        ema_decay: float = 0.995,
        data_dir: Optional[str] = None,
        replay: Optional[ReplayBufferProtocol] = None,
        load_file: Optional[str] = None,
        gpu: int = 0,
        train_micro_batch: Optional[int] = None,
        use_amp: Optional[bool] = None,
        debug_probe_logging: bool = False,
        use_intrinsic_rewards: bool = True,
        use_reward_transform: bool = True,
        use_bandit_schedule: bool = True,
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
        self.train_micro_batch = train_micro_batch
        self.use_intrinsic_rewards = use_intrinsic_rewards
        self.use_reward_transform = use_reward_transform
        self.use_bandit_schedule = use_bandit_schedule
        env_flag = os.environ.get("MEME_DEBUG_PROBES", "").lower()
        env_enabled = env_flag not in ("", "0", "false", "off")
        self.debug_probe_logging = bool(debug_probe_logging or env_enabled)
        self._debug_observe_counts = np.zeros(num_envs, dtype=np.int64)
        self._debug_logged_batch = False
        self._debug_logged_returns = False
        self._debug_logged_policy = False
        self._debug_return_logs = 0
        self._debug_trust_region_calls = 0

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu}')
            torch.cuda.manual_seed_all(seed)
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        if self.use_bandit_schedule:
            betas = self._compute_beta_schedule(num_policies, beta_high)
            gammas = self._compute_gamma_schedule(num_policies, gamma_low, gamma_high)
        else:
            if beta_low <= 0.0 or beta_high <= 0.0:
                betas = np.linspace(beta_low, beta_high, num_policies).astype(np.float32)
            else:
                betas = np.geomspace(beta_low, beta_high, num_policies).astype(np.float32)
            gammas = np.linspace(gamma_low, gamma_high, num_policies).astype(np.float32)
        self.policy_betas = torch.tensor(betas, device=self.device)
        self.policy_gammas = torch.tensor(gammas, device=self.device)
        self.num_policies = num_policies

        self.state_stacks = np.zeros((num_envs, stack_size, obs_height, obs_width), dtype=np.uint8)
        self.last_actions = np.full(num_envs, -1, dtype=np.int64)
        self.last_policy_indices = np.zeros(num_envs, dtype=np.int64)
        self.last_policy_dists = np.zeros((num_envs, num_actions), dtype=np.float32)
        self.lstm_hidden_h = torch.zeros(self.num_envs, 1024, device=self.device)
        self.lstm_hidden_c = torch.zeros(self.num_envs, 1024, device=self.device)
        self.episode_buffers: List[List[Transition]] = [[] for _ in range(num_envs)]
        action_embed_dim = 32
        self.action_embedding_table = np.zeros((num_actions, action_embed_dim), dtype=np.float32)
        eye_dim = min(num_actions, action_embed_dim)
        self.action_embedding_table[:, :eye_dim] = np.eye(num_actions, eye_dim, dtype=np.float32)
        self.prev_action_ids = np.zeros(num_envs, dtype=np.int64)
        self.prev_ext_reward = np.zeros(num_envs, dtype=np.float32)
        self.prev_int_reward = np.zeros(num_envs, dtype=np.float32)
        self.prev_rnd_component = np.zeros(num_envs, dtype=np.float32)
        self.prev_epi_component = np.zeros(num_envs, dtype=np.float32)
        self.prev_ap_embedding = np.zeros((num_envs, 32), dtype=np.float32)
        self.feature_dim = action_embed_dim + 4 + self.prev_ap_embedding.shape[1]
        self.last_features = np.zeros((num_envs, self.feature_dim), dtype=np.float32)

        self.network = MEMENetwork(stack_size, num_actions, num_policies, feature_dim=self.feature_dim).to(self.device)
        self.ema_network = MEMENetwork(stack_size, num_actions, num_policies, feature_dim=self.feature_dim).to(self.device)
        self.ema_network.load_state_dict(self.network.state_dict())
        for p in self.ema_network.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            eps=adam_eps,
            weight_decay=0.05,
        )

        if replay is not None:
            self.replay = replay
        else:
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
        self.use_amp = bool(use_amp) if use_amp is not None else self.device.type == "cuda"
        if self.use_amp and self.device.type != "cuda":
            self.use_amp = False
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.last_td_error = 0.0

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
        self.prev_action_ids.fill(0)
        self.prev_ext_reward.fill(0.0)
        self.prev_int_reward.fill(0.0)
        self.prev_rnd_component.fill(0.0)
        self.prev_epi_component.fill(0.0)
        self.prev_ap_embedding.fill(0.0)
        self.last_features.fill(0.0)

    def _epsilon(self) -> float:
        fraction = min(self.frame_count / float(self.epsilon_decay), 1.0)
        return float(self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start))

    def _current_feature_array(self) -> np.ndarray:
        action_feat = self.action_embedding_table[self.prev_action_ids]
        scalar_feats = np.stack(
            [
                self.prev_ext_reward,
                self.prev_int_reward,
                self.prev_rnd_component,
                self.prev_epi_component,
            ],
            axis=1,
        )
        return np.concatenate([action_feat, scalar_feats, self.prev_ap_embedding], axis=1).astype(np.float32)

    def act(self, observations: np.ndarray) -> np.ndarray:
        processed = preprocess_batch(observations, self.obs_height, self.obs_width)
        self.state_stacks = np.roll(self.state_stacks, shift=-1, axis=1)
        self.state_stacks[:, -1, :, :] = processed

        epsilon = self._epsilon()
        obs_tensor = torch.from_numpy(self.state_stacks).to(self.device, dtype=torch.float32) / 255.0
        obs_tensor = obs_tensor.unsqueeze(1)
        feature_np = self._current_feature_array()
        self.last_features = feature_np
        feature_tensor = torch.from_numpy(feature_np).to(self.device, dtype=torch.float32).unsqueeze(1)

        hidden = (
            self.lstm_hidden_h.unsqueeze(0).contiguous(),
            self.lstm_hidden_c.unsqueeze(0).contiguous(),
        )

        with torch.no_grad():
            self.network.reset_noise()
            ext, intr, logits, hidden = self.network(obs_tensor, feature_tensor, hidden)
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
        self.prev_action_ids = actions.copy()
        return actions

    # ------------------------------------------------------------------#
    # Intrinsic reward
    # ------------------------------------------------------------------#

    def _compute_intrinsic(self, stack: torch.Tensor, env: int) -> Tuple[float, float, float]:
        if not self.use_intrinsic_rewards:
            return 0.0, 0.0, 0.0
        with torch.no_grad():
            embedding = self.embedding_head(stack.unsqueeze(0)).cpu().numpy()[0]
        bonus_epi = self.episodic_modules[env].compute_bonus(embedding)
        rnd_norm, rnd_raw = self.rnd.bonus(stack.unsqueeze(0))
        rnd_norm = float(rnd_norm.squeeze().cpu().numpy())
        rnd_raw = float(rnd_raw.squeeze().cpu().numpy())
        alpha = np.clip(max(rnd_norm, 1.0), 1.0, 5.0)
        intrinsic = float(np.clip(bonus_epi * alpha, 0.0, 1.0))
        return intrinsic, rnd_norm, float(bonus_epi)

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
            intrinsic_reward, rnd_component, epi_component = self._compute_intrinsic(stack_tensor, env)

            transition = Transition(
                obs=self.state_stacks[env].copy(),
                action=action,
                reward_ext=float(rewards[env]),
                reward_int=intrinsic_reward,
                done=done,
                policy_idx=policy_idx,
                policy_probs=policy_probs.copy(),
                features=self.last_features[env].copy(),
            )
            self.episode_buffers[env].append(transition)
            self.state_stacks[env] = next_stacks[env]
            self.prev_ext_reward[env] = float(rewards[env])
            self.prev_int_reward[env] = intrinsic_reward
            self.prev_rnd_component[env] = rnd_component
            self.prev_epi_component[env] = epi_component
            if self.debug_probe_logging and self._debug_observe_counts[env] < 5:
                self._debug_observe_counts[env] += 1
                probs_preview = [round(float(x), 3) for x in policy_probs[: min(4, self.num_actions)]]
                self._debug(
                    f"observe env={env} action={action} reward_ext={float(rewards[env]):.3f} "
                    f"reward_int={intrinsic_reward:.3f} done={done} policy_idx={policy_idx} "
                    f"policy_probs={probs_preview} buffer_len={len(self.episode_buffers[env])}"
                )

            if done:
                episode_len = len(self.episode_buffers[env])
                episode_return = sum(t.reward_ext for t in self.episode_buffers[env])
                if self.debug_probe_logging:
                    self._debug(
                        f"episode_done env={env} len={episode_len} return={episode_return:.3f} policy_idx={policy_idx}"
                    )
                self.replay.add_episode(self.episode_buffers[env])
                self.bandit.update(policy_idx, episode_return)
                self.episode_buffers[env] = []
                self.episodic_modules[env].reset()
                self.lstm_hidden_h[env].zero_()
                self.lstm_hidden_c[env].zero_()
                self.prev_action_ids[env] = 0
                self.prev_ext_reward[env] = 0.0
                self.prev_int_reward[env] = 0.0
                self.prev_rnd_component[env] = 0.0
                self.prev_epi_component[env] = 0.0
                self.prev_ap_embedding[env].fill(0.0)

        self.frame_count += self.num_envs

    # ------------------------------------------------------------------#
    # Training
    # ------------------------------------------------------------------#

    def _transform_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if not self.use_reward_transform:
            return rewards
        return torch.sign(rewards) * (torch.sqrt(rewards * rewards + 1.0) - 1.0) + 0.001 * rewards

    def _soft_watkins_returns(
        self,
        q_online: torch.Tensor,
        rewards_ext: torch.Tensor,
        rewards_int: torch.Tensor,
        dones: torch.Tensor,
        policy_probs: torch.Tensor,
    ) -> torch.Tensor:
        batched = q_online.dim() == 4
        if not batched:
            q_online = q_online.unsqueeze(0)
            rewards_ext = rewards_ext.unsqueeze(0)
            rewards_int = rewards_int.unsqueeze(0)
            dones = dones.unsqueeze(0)
            policy_probs = policy_probs.unsqueeze(0)

        B, T, num_policies, _ = q_online.shape
        running = torch.zeros(B, num_policies, device=self.device, dtype=q_online.dtype)
        slices: List[torch.Tensor] = []
        betas = self.policy_betas.view(1, num_policies)
        gammas = self.policy_gammas.view(1, num_policies)

        for t in reversed(range(T)):
            reward = rewards_ext[:, t].unsqueeze(-1) + betas * rewards_int[:, t].unsqueeze(-1)
            done = dones[:, t].unsqueeze(-1).expand(-1, num_policies)
            probs = policy_probs[:, t]  # [B, num_actions]

            q_t = q_online[:, t]  # [B, P, A]
            greedy_q, greedy_idx = q_t.max(dim=-1, keepdim=True)  # [B, P, 1]
            tolerance = greedy_q - self.tolerance_kappa * greedy_q.abs()
            q_taken = torch.gather(q_t, dim=-1, index=greedy_idx).squeeze(-1)  # [B, P]
            mask = q_taken >= tolerance.squeeze(-1)
            mask_f = mask.float()

            expectation = torch.sum(probs.unsqueeze(1) * q_t, dim=-1)  # [B, P]
            boot = (
                (1 - done)
                * gammas
                * (mask_f * ((1 - self.lambda_val) * expectation + self.lambda_val * running) + (1 - mask_f) * expectation)
            )
            running = (reward + boot).detach().clamp_(-1e3, 1e3)
            slices.append(running.unsqueeze(1))

        if slices:
            result = torch.cat(list(reversed(slices)), dim=1)
            result = torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            result = torch.zeros(B, T, num_policies, device=self.device, dtype=q_online.dtype)

        if self.debug_probe_logging and self._debug_return_logs < 5:
            self._debug_return_logs += 1
            self._debug(
                "soft_watkins: "
                f"ext={self._tensor_stats(rewards_ext)} "
                f"int={self._tensor_stats(rewards_int)} "
                f"result={self._tensor_stats(result)}"
            )

        return result.squeeze(0) if not batched else result

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
        if self.debug_probe_logging and self._debug_trust_region_calls < 5:
            self._debug_trust_region_calls += 1
            self._debug(
                "trust_region: "
                f"keep_ratio={float(mask.float().mean().item()):.3f} "
                f"td={self._tensor_stats(td)} "
                f"sigma={self._tensor_stats(sigma)} "
                f"norm_td={self._tensor_stats(normalised_td)}"
            )
        return mask, normalised_td

    def _ema_update(self) -> None:
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)

    # ------------------------------------------------------------------#
    # Debug helpers
    # ------------------------------------------------------------------#

    def _debug(self, msg: str) -> None:
        if self.debug_probe_logging:
            print(f"[MEME DEBUG] {msg}")

    def _tensor_stats(self, tensor: torch.Tensor) -> str:
        if tensor.numel() == 0:
            return "empty"
        data = tensor.detach()
        return (
            f"shape={tuple(data.shape)} "
            f"min={float(data.min().item()):.4f} "
            f"mean={float(data.mean().item()):.4f} "
            f"max={float(data.max().item()):.4f}"
        )

    def _format_vector(self, tensor: torch.Tensor, limit: int = 6) -> List[float]:
        if tensor.numel() == 0:
            return []
        data = tensor.detach().cpu().flatten()
        limit = min(limit, data.numel())
        return [round(float(x), 4) for x in data[:limit].tolist()]

    def _log_debug_batch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rew_ext: torch.Tensor,
        rew_int: torch.Tensor,
        dones: torch.Tensor,
        p_idx: torch.Tensor,
        p_probs: torch.Tensor,
    ) -> None:
        if obs.size(0) == 0:
            return
        sample = 0
        obs_sample = obs[sample].float()
        frame_means = obs_sample[:, -1].mean(dim=(1, 2))
        summary = {
            "obs_shape": tuple(obs.shape),
            "actions": actions[sample, self.burn_in : self.burn_in + 5].tolist(),
            "rew_ext": self._format_vector(rew_ext[sample, self.burn_in : self.burn_in + 5]),
            "rew_int": self._format_vector(rew_int[sample, self.burn_in : self.burn_in + 5]),
            "dones": dones[sample, self.burn_in : self.burn_in + 5].tolist(),
            "policy_idx": p_idx[sample, self.burn_in : self.burn_in + 5].tolist(),
            "frame_means": [round(float(x), 2) for x in frame_means[-5:].tolist()],
            "policy_probs": self._format_vector(p_probs[sample, self.burn_in], limit=4),
        }
        self._debug(f"batch snapshot: {summary}")

    def _log_debug_returns(
        self,
        rew_ext: torch.Tensor,
        rew_int: torch.Tensor,
        returns: torch.Tensor,
    ) -> None:
        if returns.numel() == 0:
            return
        ret_mean = returns.mean(dim=2)
        summary = {
            "rew_ext_stats": self._tensor_stats(rew_ext[:, self.burn_in :]),
            "rew_int_stats": self._tensor_stats(rew_int[:, self.burn_in :]),
            "returns_stats": self._tensor_stats(returns),
            "returns_sample": self._format_vector(ret_mean[0, : min(3, ret_mean.size(1))]),
        }
        self._debug(f"return targets: {summary}")

    def _log_debug_policy(self, teacher: torch.Tensor, log_probs: torch.Tensor) -> None:
        if teacher.numel() == 0 or log_probs.numel() == 0:
            return
        if log_probs.dim() == 4:
            policy_slice = log_probs[0, 0, 0]
        else:
            policy_slice = log_probs[0, 0]
        dist_teacher = self._format_vector(teacher[0, 0], limit=6)
        policy_probs = self._format_vector(policy_slice.exp(), limit=6)
        self._debug(
            f"policy sample: teacher={dist_teacher} policy_probs={policy_probs} "
            f"log_probs_stats={self._tensor_stats(log_probs)}"
        )

    def train_step(self) -> None:
        self._train_call_count += 1
        if self._train_call_count % self.train_interval != 0:
            return

        self.last_metrics = {}
        if self.debug_probe_logging:
            self._debug_logged_batch = False
            self._debug_logged_returns = False
            self._debug_logged_policy = False
            self._debug_trust_region_calls = 0

        sequences, indices, weights = self.replay.sample(self.batch_size)
        if not sequences:
            return

        total_len = self.burn_in + self.seq_len
        invalid_records: List[Tuple[int, torch.Tensor]] = []
        valid_entries: List[Tuple[EpisodeBatch, int, torch.Tensor]] = []
        device_weights = weights.to(self.device)

        for seq, idx, w in zip(sequences, indices, device_weights):
            if seq.length < total_len:
                invalid_records.append((idx, torch.tensor(1.0, device=self.device)))
                continue
            valid_entries.append((seq, idx, w))

        if not valid_entries:
            for idx, priority in invalid_records:
                self.replay.update_priorities([idx], priority.detach().unsqueeze(0).cpu())
            return

        self.optimizer.zero_grad(set_to_none=True)
        self.network.reset_noise()

        B = len(valid_entries)
        obs = torch.from_numpy(
            np.stack([np.ascontiguousarray(seq.obs) for seq, _, _ in valid_entries])
        )
        actions = torch.from_numpy(
            np.stack([np.ascontiguousarray(seq.actions) for seq, _, _ in valid_entries])
        )
        rew_ext = torch.from_numpy(
            np.stack([np.ascontiguousarray(seq.reward_ext) for seq, _, _ in valid_entries]).astype(np.float32, copy=False)
        )
        rew_int = torch.from_numpy(
            np.stack([np.ascontiguousarray(seq.reward_int) for seq, _, _ in valid_entries]).astype(np.float32, copy=False)
        )
        dones = torch.from_numpy(
            np.stack([np.ascontiguousarray(seq.done).astype(np.float32) for seq, _, _ in valid_entries])
        )
        p_idx = torch.from_numpy(
            np.stack([np.ascontiguousarray(seq.policy_idx) for seq, _, _ in valid_entries])
        )
        p_probs = torch.from_numpy(
            np.stack([np.ascontiguousarray(seq.policy_probs) for seq, _, _ in valid_entries]).astype(np.float32, copy=False)
        )
        features = torch.from_numpy(
            np.stack([np.ascontiguousarray(seq.features) for seq, _, _ in valid_entries]).astype(np.float32, copy=False)
        )
        sample_weights = torch.stack([w for _, _, w in valid_entries]).to(self.device).view(B, 1)
        replay_indices = [int(idx) for _, idx, _ in valid_entries]
        if self.debug_probe_logging and not self._debug_logged_batch:
            self._log_debug_batch(obs, actions, rew_ext, rew_int, dones, p_idx, p_probs)
            self._debug_logged_batch = True

        micro = self.train_micro_batch or B
        micro = max(1, min(micro, B))

        tdstd_updates: Dict[int, List[torch.Tensor]] = {}
        priority_indices: List[int] = []
        priority_values: List[torch.Tensor] = []

        loss_total_sum = 0.0
        loss_behaviour_sum = 0.0
        loss_aux_sum = 0.0
        loss_policy_sum = 0.0
        td_abs_sum = 0.0
        ratio_abs_sum = 0.0
        total_samples = 0

        for start in range(0, B, micro):
            end = min(start + micro, B)
            chunk_size = end - start
            total_samples += chunk_size
            weight = chunk_size / float(B)

            obs_mb = obs[start:end].to(self.device, dtype=torch.float32, non_blocking=True) / 255.0
            actions_mb = actions[start:end].to(self.device, dtype=torch.long, non_blocking=True)
            rew_ext_mb = rew_ext[start:end].to(self.device, dtype=torch.float32, non_blocking=True)
            rew_int_mb = rew_int[start:end].to(self.device, dtype=torch.float32, non_blocking=True)
            dones_mb = dones[start:end].to(self.device, dtype=torch.float32, non_blocking=True)
            p_idx_mb = p_idx[start:end].to(self.device, dtype=torch.long, non_blocking=True)
            p_probs_mb = p_probs[start:end].to(self.device, dtype=torch.float32, non_blocking=True)
            sample_w_mb = sample_weights[start:end]
            features_mb = features[start:end].to(self.device, dtype=torch.float32, non_blocking=True)

            if self.use_amp:
                if hasattr(torch, "amp"):
                    autocast_ctx = torch.amp.autocast("cuda")
                else:
                    autocast_ctx = torch.cuda.amp.autocast()
            else:
                autocast_ctx = contextlib.nullcontext()
            with autocast_ctx:
                ext, intr, logits, _ = self.network(obs_mb, features_mb)
                ext, intr, logits = [x[:, self.burn_in :] for x in (ext, intr, logits)]

                q_online = ext + self.policy_betas.view(1, 1, -1, 1) * intr

                rew_ext_train = self._transform_rewards(rew_ext_mb[:, self.burn_in :])
                returns = self._soft_watkins_returns(
                    q_online,
                    rew_ext_train,
                    rew_int_mb[:, self.burn_in :],
                    dones_mb[:, self.burn_in :],
                    p_probs_mb[:, self.burn_in :],
                )
                returns = torch.nan_to_num(returns, nan=0.0, posinf=1e6, neginf=-1e6).clamp_(-1e3, 1e3)
                if self.debug_probe_logging and not self._debug_logged_returns:
                    self._log_debug_returns(rew_ext_mb[:, self.burn_in :], rew_int_mb[:, self.burn_in :], returns)
                    self._debug_logged_returns = True

                act = actions_mb[:, self.burn_in :].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_policies, 1)
                q_taken_all = q_online.gather(3, act).squeeze(3)

                idx_mb = p_idx_mb[:, self.burn_in :].unsqueeze(-1)
                q_selected = q_taken_all.gather(2, idx_mb).squeeze(2)
                targets = returns.gather(2, idx_mb).squeeze(2)

                sigma = torch.clamp(
                    torch.nan_to_num(self.running_td_std[p_idx_mb[:, self.burn_in :]], nan=self.td_norm_eps),
                    min=self.td_norm_eps,
                    max=1e3,
                )

                mask, td = self._trust_region(q_selected, q_selected.detach(), targets, sigma, sample_w_mb)
                loss_behaviour = (mask.float() * (td**2)).mean()

                sigma_all = torch.clamp(
                    torch.nan_to_num(self.running_td_std, nan=self.td_norm_eps), min=self.td_norm_eps, max=1e3
                ).view(1, 1, -1)
                all_td = torch.nan_to_num((returns - q_taken_all) / sigma_all, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(
                    -1e3, 1e3
                )
                loss_aux = all_td.square().mean()

                teacher = p_probs_mb[:, self.burn_in :]
                log_probs = F.log_softmax(logits, dim=-1)
                teacher_expanded = teacher.unsqueeze(2)
                kl = torch.sum(teacher_expanded * (torch.log(teacher_expanded + 1e-8) - log_probs), dim=-1)
                kl_mask = (kl <= self.policy_kl_clip).float()
                if self.debug_probe_logging and not self._debug_logged_policy:
                    self._log_debug_policy(teacher, log_probs[:, :, 0, :])
                    self._debug_logged_policy = True
                loss_policy = -(kl_mask * (teacher_expanded * log_probs).sum(-1)).mean()

                loss = self.eta * loss_behaviour + (1 - self.eta) * loss_aux + loss_policy

            if self.use_amp:
                self.grad_scaler.scale(loss * weight).backward()
            else:
                (loss * weight).backward()

            loss_total_sum += loss.detach().item() * chunk_size
            loss_behaviour_sum += loss_behaviour.detach().item() * chunk_size
            loss_aux_sum += loss_aux.detach().item() * chunk_size
            loss_policy_sum += loss_policy.detach().item() * chunk_size

            td_fp32 = td.detach().float()
            all_td_fp32 = all_td.detach().float()

            td_abs_chunk = td_fp32.abs().mean(dim=1).clamp_min(1e-3).cpu()
            priority_indices.extend(replay_indices[start:end])
            priority_values.append(td_abs_chunk)

            td_abs_sum += td_abs_chunk.mean().item() * chunk_size
            ratio_abs_sum += all_td_fp32.abs().mean().item() * chunk_size

            with torch.no_grad():
                p_idx_slice = p_idx_mb[:, self.burn_in :]
                for pol in torch.unique(p_idx_slice).tolist():
                    pol_mask = p_idx_slice == pol
                    if pol_mask.any():
                        std = td_fp32[pol_mask].std(unbiased=False).clamp(min=self.td_norm_eps)
                        tdstd_updates.setdefault(int(pol), []).append(std)

        if self.use_amp:
            self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        if self.use_amp:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        self._ema_update()

        if priority_values:
            priority_tensor = torch.cat(priority_values, dim=0)
            self.replay.update_priorities(priority_indices, priority_tensor.cpu())

        if tdstd_updates:
            with torch.no_grad():
                for pol, std_list in tdstd_updates.items():
                    avg_std = torch.stack(std_list).mean()
                    self.running_td_std[pol] = 0.99 * self.running_td_std[pol] + 0.01 * avg_std

        denom = float(total_samples)
        self.last_td_error = float(td_abs_sum / denom) if denom > 0 else 0.0
        self.last_metrics = {
            "loss/total": float(loss_total_sum / denom),
            "loss/behaviour": float(loss_behaviour_sum / denom),
            "loss/aux_td": float(loss_aux_sum / denom),
            "loss/policy": float(loss_policy_sum / denom),
            "stats/td_abs_mean": float(td_abs_sum / denom),
            "stats/ratio_abs_mean": float(ratio_abs_sum / denom),
            "stats/running_td_std_mean": float(self.running_td_std.mean().item()),
            "train/epsilon": float(self._epsilon()),
            "train/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "train/updates": float(self.training_steps),
            "train/batch_count": float(B),
            "train/micro_batch": float(micro),
        }

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
