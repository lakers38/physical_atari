"""
Soft Actor-Critic (SAC) for discrete action spaces.

Implements the entropy-regularized RL algorithm with:
- Twin Q-networks to reduce overestimation bias
- Target networks for stable learning
- Entropy maximization in the objective
- Discrete action space formulation

Reference: Soft Actor-Critic for Discrete Action Settings (Christodoulou, 2019)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional
import os


class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor based on Nature DQN architecture.

    Input: (batch, n_stack, H, W) - stacked grayscale frames
    Output: (batch, feature_dim) - feature vector

    Supports 84x84 and 128x128 input sizes.
    """

    def __init__(self, n_stack=4, feature_dim=512, input_size=128):
        """
        Args:
            n_stack: Number of stacked frames (default: 4)
            feature_dim: Output feature dimension (default: 512)
            input_size: Input image size (84 or 128, default: 128)
        """
        super().__init__()
        self.n_stack = n_stack
        self.feature_dim = feature_dim
        self.input_size = input_size

        # Nature DQN convolutional layers
        # Same architecture works for both 84x84 and 128x128
        self.conv = nn.Sequential(
            nn.Conv2d(n_stack, 32, kernel_size=8, stride=4),  # 84→20, 128→31
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 20→9, 31→14
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 9→7, 14→12
            nn.ReLU(True),
            nn.Flatten(),
        )

        # Calculate conv output size
        conv_out_size = self._get_conv_output_size(input_size)

        # Fully connected layer
        self.fc = nn.Sequential(nn.Linear(conv_out_size, feature_dim), nn.ReLU(True))

        # Initialize weights
        self._initialize_weights()

    def _get_conv_output_size(self, input_size):
        """Calculate the output size of conv layers."""
        # Conv1: kernel=8, stride=4, padding=0
        size = (input_size - 8) // 4 + 1
        # Conv2: kernel=4, stride=2, padding=0
        size = (size - 4) // 2 + 1
        # Conv3: kernel=3, stride=1, padding=0
        size = (size - 3) // 1 + 1
        # Output: 64 channels × size × size
        return 64 * size * size

    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, n_stack, H, W) where H=W=input_size

        Returns:
            features: Tensor of shape (batch, feature_dim)
        """
        # Normalize pixel values to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = self.conv(x)
        features = self.fc(x)
        return features

    def extract_features_numpy(self, obs):
        """
        Extract features from numpy observation.

        Args:
            obs: Numpy array of shape (n_stack, 84, 84) or (batch, n_stack, 84, 84)

        Returns:
            features: Numpy array of shape (feature_dim,) or (batch, feature_dim)
        """
        # Convert to tensor
        if obs.ndim == 3:
            obs = obs[np.newaxis, ...]  # Add batch dimension
            squeeze = True
        else:
            squeeze = False

        obs_tensor = torch.from_numpy(obs).float()

        # Move to same device as model
        device = next(self.parameters()).device
        obs_tensor = obs_tensor.to(device)

        # Extract features
        with torch.no_grad():
            features = self.forward(obs_tensor)

        # Convert back to numpy
        features_np = features.cpu().numpy()

        if squeeze:
            features_np = features_np[0]  # Remove batch dimension

        return features_np


class QNetwork(nn.Module):
    """
    Q-network for discrete actions.

    Outputs Q-values for all actions: Q(s, ·) ∈ ℝ^|A|
    """

    def __init__(self, feature_dim=512, hidden_dim=256, num_actions=18):
        """
        Args:
            feature_dim: Input feature dimension from CNN
            hidden_dim: Hidden layer dimension
            num_actions: Number of discrete actions
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, num_actions)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Forward pass.

        Args:
            features: Tensor of shape (batch, feature_dim)

        Returns:
            q_values: Tensor of shape (batch, num_actions)
        """
        return self.network(features)


class PolicyNetwork(nn.Module):
    """
    Policy network (actor) for discrete action selection.

    Outputs action logits for sampling via Categorical distribution.
    """

    def __init__(self, feature_dim=512, hidden_dim=256, num_actions=18):
        """
        Args:
            feature_dim: Input feature dimension from CNN
            hidden_dim: Hidden layer dimension
            num_actions: Number of discrete actions
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, num_actions)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Forward pass.

        Args:
            features: Tensor of shape (batch, feature_dim)

        Returns:
            logits: Tensor of shape (batch, num_actions)
        """
        return self.network(features)


class SACAgent:
    """
    Soft Actor-Critic agent for discrete action spaces.

    Key features:
    - Twin Q-networks (Q1, Q2) with target networks
    - Entropy-regularized policy optimization
    - Maximum entropy RL objective
    """

    def __init__(
        self,
        num_actions: int,
        feature_dim: int = 512,
        actor_hidden_dim: int = 256,
        value_hidden_dim: int = 256,
        n_stack: int = 4,
        input_size: int = 128,
        device: str = "cuda",
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        entropy_coef: float = 0.01,
        tau: float = 0.005,  # Polyak averaging coefficient for target networks
        fail_on_nonfinite: bool = True,
        auto_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
    ):
        self.num_actions = num_actions
        self.device = torch.device(device)
        print("THE DEVICE IS : ", self.device)
        self.gamma = gamma
        self.tau = tau
        self.fail_on_nonfinite = fail_on_nonfinite
        self.n_stack = n_stack
        self.input_size = input_size
        self.auto_entropy_tuning = auto_entropy_tuning

        # Automatic entropy tuning
        if self.auto_entropy_tuning:
            # Target entropy: -log(1/|A|) * 0.5 (50% of maximum entropy)
            # Lower target for online learning stability
            if target_entropy is None:
                self.target_entropy = -np.log(1.0 / num_actions) * 0.5
                print(f"SETTING TARGET ENTROPY TO: {self.target_entropy}")
            else:
                self.target_entropy = target_entropy

            # Use log_alpha for numerical stability (ensures alpha > 0)
            # Initialize to log(0.2) to start with reasonable alpha
            self.log_alpha = torch.tensor([np.log(0.2)], requires_grad=True, device=self.device, dtype=torch.float32)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
            # Keep alpha in a sane range to avoid exploding value targets
            self.min_alpha = 1e-4
            self.max_alpha = 10.0
            self.log_alpha_min = np.log(self.min_alpha)
            self.log_alpha_max = np.log(self.max_alpha)
        else:
            # Fixed entropy coefficient
            self.entropy_coef = entropy_coef
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optimizer = None

        # Shared CNN feature extractor
        self.cnn = CNNFeatureExtractor(n_stack=n_stack, feature_dim=feature_dim, input_size=input_size).to(self.device)

        # Policy network (actor)
        self.actor = PolicyNetwork(feature_dim=feature_dim, hidden_dim=actor_hidden_dim, num_actions=num_actions).to(
            self.device
        )

        # Twin Q-networks (critics)
        self.q1 = QNetwork(feature_dim=feature_dim, hidden_dim=value_hidden_dim, num_actions=num_actions).to(
            self.device
        )
        self.q2 = QNetwork(feature_dim=feature_dim, hidden_dim=value_hidden_dim, num_actions=num_actions).to(
            self.device
        )

        # Target Q-networks
        self.q1_target = QNetwork(feature_dim=feature_dim, hidden_dim=value_hidden_dim, num_actions=num_actions).to(
            self.device
        )
        self.q2_target = QNetwork(feature_dim=feature_dim, hidden_dim=value_hidden_dim, num_actions=num_actions).to(
            self.device
        )

        # Initialize target networks with same weights
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Freeze target networks (no gradient computation)
        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False

        # Optimizers
        self.cnn_optimizer = torch.optim.Adam(self.cnn.parameters(), lr=learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=learning_rate)

    def _check_finite(self, tensor: torch.Tensor, ctx: str) -> torch.Tensor:
        if self.fail_on_nonfinite and not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite tensor in {ctx}: min={tensor.min().item()}, max={tensor.max().item()}")
        return tensor

    def _polyak_update(self, source: nn.Module, target: nn.Module):
        """Update target network using Polyak averaging."""
        with torch.no_grad():
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)

    @property
    def alpha(self):
        """Get current entropy coefficient (alpha)."""
        if self.auto_entropy_tuning:
            return float(self.log_alpha.clamp(self.log_alpha_min, self.log_alpha_max).exp().item())
        else:
            return self.entropy_coef

    def start_episodes(self, obs_batch: np.ndarray):
        # No bootstrapping needed; keep signature for compatibility
        return self._extract_features(obs_batch)

    def reset_done(self, done: int, obs_batch: np.ndarray):
        # Stateless reset (maintained for API compatibility)
        _ = obs_batch

    def _extract_features(self, obs_batch: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        assert (
            obs_batch.ndim == 4 and obs_batch.shape[-1] == self.n_stack
        ), f"obs_batch expected shape (*, {self.input_size}, {self.input_size}, {self.n_stack}), got {obs_batch.shape}"
        obs_batch = np.transpose(obs_batch, (0, 3, 1, 2))
        obs_t = torch.as_tensor(obs_batch, device=self.device, dtype=torch.float32) / 255.0
        feats = self.cnn(obs_t)
        feats = self._check_finite(feats, ctx="features")
        return feats, feats.detach().cpu().numpy()

    def select_actions(self, obs_batch: np.ndarray):
        """
        Select actions using the current policy.

        Returns:
            actions: Sampled actions
            log_probs: Log probabilities of sampled actions
            entropy: Entropy of the policy distribution
            feats_torch: Feature tensor (for later use in update)
        """
        feats_torch, feats_np = self._extract_features(obs_batch)
        logits = self._check_finite(self.actor(feats_torch), ctx="logits")
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions.cpu().numpy(), log_probs, entropy, feats_torch

    def update(
        self,
        obs_batch: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs_batch: np.ndarray,
        dones: np.ndarray,
    ):
        """
        SAC update step.

        Updates:
        1. Q-networks using TD target with entropy regularization
        2. Policy network to maximize Q - α·log(π)
        3. Target networks via Polyak averaging
        """
        # Extract current features
        current_feats, _ = self._extract_features(obs_batch)

        # Convert inputs to tensors
        actions_t = torch.as_tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        dones_t = torch.as_tensor(dones, device=self.device, dtype=torch.float32)

        # Get current alpha (entropy coefficient)
        if self.auto_entropy_tuning:
            alpha = self.log_alpha.clamp(self.log_alpha_min, self.log_alpha_max).exp().detach()
        else:
            alpha = self.entropy_coef

        # ===== Compute Q-targets =====
        with torch.no_grad():
            # Extract features for next state
            next_feats, _ = self._extract_features(next_obs_batch)

            # Get next policy distribution
            next_logits = self.actor(next_feats)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)

            # Get target Q-values for all actions
            next_q1_all = self.q1_target(next_feats)
            next_q2_all = self.q2_target(next_feats)

            # Take minimum to reduce overestimation (double Q-learning)
            next_q_all = torch.min(next_q1_all, next_q2_all)

            # Compute V(s') = E_π[Q(s',a) - α·log π(a|s')]
            # = Σ_a π(a|s') * [Q(s',a) - α·log π(a|s')]
            next_v = (next_probs * (next_q_all - alpha * next_log_probs)).sum(dim=-1)

            # Q-target: r + γ * (1 - done) * V(s')
            q_target = rewards_t + self.gamma * (1 - dones_t) * next_v

        # ===== Update Q-networks =====
        # Get current Q-values for taken actions
        q1_values = self.q1(current_feats).gather(1, actions_t).squeeze(1)
        q2_values = self.q2(current_feats).gather(1, actions_t).squeeze(1)

        # Q-loss (MSE between Q(s,a) and target)
        q1_loss = F.mse_loss(q1_values, q_target)
        q2_loss = F.mse_loss(q2_values, q_target)
        q_loss = q1_loss + q2_loss

        # Update Q-networks
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.cnn_optimizer.zero_grad()
        q_loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.cnn.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=10.0)

        self.cnn_optimizer.step()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # ===== Update Policy =====
        # Re-extract features after Q-update (CNN weights changed)
        current_feats_new = self.cnn(
            torch.as_tensor(np.transpose(obs_batch, (0, 3, 1, 2)), device=self.device, dtype=torch.float32) / 255.0
        )

        # Get current policy distribution
        logits = self.actor(current_feats_new.detach())  # Detach to avoid backprop through Q
        probs = F.softmax(logits, dim=-1)
        log_probs_all = F.log_softmax(logits, dim=-1)

        # Get Q-values for all actions (use minimum of Q1 and Q2)
        with torch.no_grad():
            q1_all = self.q1(current_feats_new)
            q2_all = self.q2(current_feats_new)
            q_all = torch.min(q1_all, q2_all)

        # Policy loss: maximize E_π[Q(s,a) - α·log π(a|s)]
        # = minimize E_π[α·log π(a|s) - Q(s,a)]
        policy_loss = (probs * (alpha * log_probs_all - q_all)).sum(dim=-1).mean()

        # Update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        # ===== Update entropy coefficient (alpha) =====
        if self.auto_entropy_tuning:
            # Compute current entropy
            with torch.no_grad():
                current_entropy = -(probs * log_probs_all).sum(dim=-1).mean()

            # Alpha loss: minimize α * (entropy - target_entropy)
            # This increases alpha when entropy is too low, decreases when too high
            target_entropy_t = torch.as_tensor(self.target_entropy, device=self.device, dtype=torch.float32)
            alpha_loss = self.log_alpha.clamp(self.log_alpha_min, self.log_alpha_max).exp() * (
                current_entropy.detach() - target_entropy_t
            )

            # Update alpha
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # Explicitly clamp log_alpha to keep alpha bounded
            self.log_alpha.data.clamp_(self.log_alpha_min, self.log_alpha_max)
        else:
            alpha_loss = torch.tensor(0.0)

        # ===== Update target networks =====
        self._polyak_update(self.q1, self.q1_target)
        self._polyak_update(self.q2, self.q2_target)

        # ===== Compute metrics =====
        with torch.no_grad():
            # Compute policy entropy
            policy_entropy = -(probs * log_probs_all).sum(dim=-1).mean()

            # Compute advantages for logging
            q_taken = q_all.gather(1, actions_t).squeeze(1)
            advantages = q_target - q_taken

        return {
            "actor_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(q_loss.detach().cpu().item()),
            "total_loss": float((policy_loss + q_loss).detach().cpu().item()),
            "q1_loss": float(q1_loss.detach().cpu().item()),
            "q2_loss": float(q2_loss.detach().cpu().item()),
            "advantage": float(advantages.mean().item()),
            "value_pred": float(q1_values.mean().item()),
            "value_next": float(next_v.mean().item()),
            "value_target": float(q_target.mean().item()),
            "policy_entropy": float(policy_entropy.item()),
            "log_prob_mean": float(log_probs_all.mean().item()),
            "alpha": self.alpha,  # Current entropy coefficient
            "alpha_loss": float(alpha_loss.detach().cpu().item()) if self.auto_entropy_tuning else 0.0,
            "target_entropy": self.target_entropy if self.auto_entropy_tuning else None,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "cnn": self.cnn.state_dict(),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "cnn_optimizer": self.cnn_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q1_optimizer": self.q1_optimizer.state_dict(),
            "q2_optimizer": self.q2_optimizer.state_dict(),
        }
        if self.auto_entropy_tuning:
            save_dict["log_alpha"] = self.log_alpha
            save_dict["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(save_dict, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.cnn.load_state_dict(ckpt["cnn"])
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.cnn_optimizer.load_state_dict(ckpt["cnn_optimizer"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.q1_optimizer.load_state_dict(ckpt["q1_optimizer"])
        self.q2_optimizer.load_state_dict(ckpt["q2_optimizer"])
        if self.auto_entropy_tuning and "log_alpha" in ckpt:
            self.log_alpha = ckpt["log_alpha"].to(self.device)
            self.log_alpha.requires_grad = True
            if "alpha_optimizer" in ckpt:
                self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])


class ReplayBuffer:
    """Simple FIFO replay buffer storing preprocessed (grayscale, stacked) frames."""

    def __init__(self, capacity: int, obs_shape: tuple[int, int, int]):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.ptr = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.capacity if self.full else self.ptr

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx].astype(np.float32),
        )
