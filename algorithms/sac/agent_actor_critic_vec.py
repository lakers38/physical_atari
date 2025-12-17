"""
Vectorized Actor-Critic for Atari using a shared CNN encoder and two-layer MLP
heads for both policy and value prediction. Critic trains with TD targets and
both heads are optimized via backprop.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
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


class PolicyHead(nn.Module):
    """
    Policy network (actor) for action selection.

    Takes features from CNN and outputs action probabilities.
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
    Shared actor (CNN + policy head) with a learned value function.
    Vectorized over n_envs environments.
    """

    def __init__(
        self,
        num_actions: int,
        num_envs: int = 1,
        feature_dim: int = 512,
        actor_hidden_dim: int = 256,
        value_hidden_dim: int = 256,
        n_stack: int = 4,
        input_size: int = 128,
        device: str = "cuda",
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        fail_on_nonfinite: bool = True,
    ):
        self.num_actions = num_actions
        self.num_envs = num_envs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.fail_on_nonfinite = fail_on_nonfinite
        self.n_stack = n_stack
        self.input_size = input_size

        self.cnn = CNNFeatureExtractor(n_stack=n_stack, feature_dim=feature_dim, input_size=input_size).to(self.device)
        self.actor = PolicyHead(feature_dim=feature_dim, hidden_dim=actor_hidden_dim, num_actions=num_actions).to(
            self.device
        )
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, value_hidden_dim),
            nn.ReLU(True),
            nn.Linear(value_hidden_dim, 1),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.cnn.parameters()) + list(self.actor.parameters()) + list(self.value_head.parameters()),
            lr=learning_rate,
        )

    def _check_finite(self, tensor: torch.Tensor, ctx: str) -> torch.Tensor:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite tensor in {ctx}: min={tensor.min().item()}, max={tensor.max().item()}")
        return tensor

    def start_episodes(self, obs_batch: np.ndarray):
        return self._extract_features(obs_batch)

    def reset_done(self, dones: np.ndarray, obs_batch: np.ndarray):
        _ = obs_batch

    def _extract_features(self, obs_batch: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        # obs_batch: (batch, H, W, n_stack)
        assert obs_batch.ndim == 4, f"obs_batch should be 4D (batch,H,W,C), got {obs_batch.shape}"
        obs_batch = np.transpose(obs_batch, (0, 3, 1, 2))
        obs_t = torch.as_tensor(obs_batch, device=self.device, dtype=torch.float32) / 255.0
        feats = self.cnn(obs_t)
        feats = self._check_finite(feats, ctx="features")
        feats_np = feats.detach().cpu().numpy()
        return feats, feats_np

    def select_actions(self, obs_batch: np.ndarray):
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
        log_probs: torch.Tensor,
        entropy: torch.Tensor,
        current_feats: torch.Tensor,
    ):
        # Compute current and next state values
        value_pred = self._check_finite(self.value_head(current_feats).squeeze(-1), ctx="value_pred")

        with torch.no_grad():
            next_feats, _ = self._extract_features(next_obs_batch)
            value_next = self.value_head(next_feats).squeeze(-1)
            target = torch.as_tensor(rewards, device=self.device, dtype=torch.float32) + (
                torch.as_tensor(1 - dones.astype(np.int32), device=self.device, dtype=torch.float32)
                * self.gamma
                * value_next
            )

        value_loss = F.mse_loss(value_pred, target)
        advantages = (target - value_pred).detach()
        actor_loss = (
            -(log_probs.to(self.device) * advantages).mean() - self.entropy_coef * entropy.to(self.device).mean()
        )
        total_loss = actor_loss + self.value_coef * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cnn.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), max_norm=10.0)
        self.optimizer.step()

        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
            "advantage_mean": float(advantages.mean().item()),
            "value_pred_mean": float(value_pred.mean().item()),
            "value_next_mean": float(value_next.mean().item()),
            "value_target_mean": float(target.mean().item()),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "cnn": self.cnn.state_dict(),
                "actor": self.actor.state_dict(),
                "value_head": self.value_head.state_dict(),
                "optim": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.cnn.load_state_dict(ckpt["cnn"])
        self.actor.load_state_dict(ckpt["actor"])
        if "value_head" in ckpt:
            self.value_head.load_state_dict(ckpt["value_head"])
        self.optimizer.load_state_dict(ckpt["optim"])
