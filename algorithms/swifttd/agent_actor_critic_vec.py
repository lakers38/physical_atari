"""
Vectorized Actor-Critic with SwiftTD critic for Atari.

Shared CNN + policy head; per-environment SwiftTD critics with independent traces.
"""

from math import isfinite
import numpy as np
import torch
import torch.nn as nn
import swifttd
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
            nn.Conv2d(32, 64, kernel_size=4, stride=2),       # 20→9, 31→14
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),       # 9→7, 14→12
            nn.ReLU(True),
            nn.Flatten()
        )

        # Calculate conv output size
        conv_out_size = self._get_conv_output_size(input_size)

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, feature_dim),
            nn.ReLU(True)
        )

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
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, num_actions)
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

class SwiftTDAgent:
    """
    Shared actor (CNN + policy head) with per-environment SwiftTD critics.
    Vectorized over n_envs environments.
    """

    def __init__(
        self,
        num_actions: int,
        num_envs: int = 1,
        feature_dim: int = 512,
        actor_hidden_dim: int = 256,
        n_stack: int = 4,
        input_size: int = 128,
        device: str = "cuda",
        # SwiftTD hyperparameters
        lambda_: float = 0.95,
        initial_alpha: float = 1e-4,
        gamma: float = 0.99,
        eps: float = 1e-5,
        max_step_size: float = 0.01,
        step_size_decay: float = 0.99,
        meta_step_size: float = 1e-4,
        eta_min: float = 1e-6,
        # Optimization
        learning_rate: float = 1e-4,
        entropy_coef: float = 0.01,
        fail_on_nonfinite: bool = True,
        critic_feature_scale: float = 0.01,
    ):
        self.num_actions = num_actions
        self.num_envs = num_envs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.fail_on_nonfinite = fail_on_nonfinite
        self._signal_eps = 1e-6
        self.critic_feature_scale = critic_feature_scale
        self.n_stack = n_stack
        self.input_size = input_size

        self.cnn = CNNFeatureExtractor(n_stack=n_stack, feature_dim=feature_dim, input_size=input_size).to(
            self.device
        )
        self.actor = PolicyHead(feature_dim=feature_dim, hidden_dim=actor_hidden_dim, num_actions=num_actions).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(list(self.cnn.parameters()) + list(self.actor.parameters()), lr=learning_rate)

        # Per-env SwiftTD critics + caches
        self._critic_kwargs = dict(
            number_of_features=feature_dim,
            lambda_init=lambda_,
            alpha_init=initial_alpha,
            gamma_init=gamma,
            epsilon_init=eps,
            eta_init=max_step_size,
            decay_init=step_size_decay,
            meta_step_size_init=meta_step_size,
            eta_min_init=eta_min,
        )
        self.critics: List[swifttd.SwiftTDNonSparse] = [self._build_critic() for _ in range(self.num_envs)]
        self.prev_value = np.zeros(self.num_envs, dtype=np.float32)
        self.prev_feature = np.zeros((self.num_envs, feature_dim), dtype=np.float32)

    def _build_critic(self) -> swifttd.SwiftTDNonSparse:
        return swifttd.SwiftTDNonSparse(
            self._critic_kwargs["number_of_features"],
            self._critic_kwargs["lambda_init"],
            self._critic_kwargs["alpha_init"],
            self._critic_kwargs["gamma_init"],
            self._critic_kwargs["epsilon_init"],
            self._critic_kwargs["eta_init"],
            self._critic_kwargs["decay_init"],
            self._critic_kwargs["meta_step_size_init"],
            self._critic_kwargs["eta_min_init"],
        )

    def _check_finite(self, tensor: torch.Tensor, ctx: str) -> torch.Tensor:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite tensor in {ctx}: min={tensor.min().item()}, max={tensor.max().item()}")
        return tensor

    def start_episodes(self, obs_batch: np.ndarray):
        feats_torch, feats_np = self._extract_features(obs_batch)
        for i in range(self.num_envs):
            v = self.critics[i].step(feats_np[i].tolist(), 0.0)  # bootstrap V(s)
            self.prev_value[i] = v
            self.prev_feature[i] = feats_np[i]
        return feats_torch, feats_np

    def reset_done(self, dones: np.ndarray, obs_batch: np.ndarray):
        if not np.any(dones):
            return
        _, feats_np = self._extract_features(obs_batch)
        for idx, done in enumerate(dones):
            if done:
                v = self.critics[idx].step(feats_np[idx].tolist(), 0.0)
                self.prev_value[idx] = v
                self.prev_feature[idx] = feats_np[idx]

    def _extract_features(self, obs_batch: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        # obs_batch: (batch, H, W, n_stack)
        assert obs_batch.ndim == 4, f"obs_batch should be 4D (batch,H,W,C), got {obs_batch.shape}"
        obs_batch = np.transpose(obs_batch, (0, 3, 1, 2))
        obs_t = torch.as_tensor(obs_batch, device=self.device, dtype=torch.float32) / 255.0
        feats = self.cnn(obs_t)  # shared encoder φ(s_t)
        feats = self._check_finite(feats, ctx="features")
        feats_np = feats.detach().cpu().numpy() * self.critic_feature_scale  # scaled φ(s) for SwiftTD
        return feats, feats_np

    def select_actions(self, obs_batch: np.ndarray):
        feats_torch, feats_np = self._extract_features(obs_batch)
        logits = self._check_finite(self.actor(feats_torch), ctx="logits")
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions.cpu().numpy(), log_probs, entropy, feats_np

    def update(
        self,
        obs_batch: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs_batch: np.ndarray,
        dones: np.ndarray,
        log_probs: torch.Tensor,
        entropy: torch.Tensor,
        current_feats_np: np.ndarray,
        step_idx: int = -1,
    ):
        # Compute next features
        _, next_feats_np = self._extract_features(next_obs_batch)

        advantages = []
        value_targets = []
        return_errors = []
        value_preds = []
        value_nexts = []

        # Per-env SwiftTD update
        for i in range(self.num_envs):
            V_current = float(self.prev_value[i])
            reward = float(rewards[i])
            assert isfinite(reward), f"Reward is not finite: {reward}"
            assert isfinite(V_current), f"V_current is not finite: {V_current}"
            done = bool(dones[i])

            target_feats = np.zeros_like(next_feats_np[i]) if done else next_feats_np[i]
            V_next = self.critics[i].step(target_feats.tolist(), reward)
            if not isfinite(V_next):
                feats_min = float(np.min(target_feats))
                feats_max = float(np.max(target_feats))
                feats_norm = float(np.linalg.norm(target_feats))
                raise ValueError(
                    f"V_next is not finite: {V_next} "
                    f"(env={i}, reward={reward}, done={done}, V_current={V_current}, "
                    f"feat_min={feats_min}, feat_max={feats_max}, feat_norm={feats_norm})"
                )
            advantage = reward + (0.0 if done else self.gamma * V_next) - V_current
            value_target = reward + (0.0 if done else self.gamma * V_next)
            return_error = (value_target - V_current) ** 2

            # Update caches
            if done:
                self.prev_value[i] = 0.0
                self.prev_feature[i] = 0.0
            else:
                self.prev_value[i] = V_next
                self.prev_feature[i] = target_feats

            advantages.append(advantage)
            value_targets.append(value_target)
            return_errors.append(return_error)
            value_preds.append(V_current)
            value_nexts.append(V_next)

        # Actor loss (batched)
        adv_tensor = torch.as_tensor(advantages, device=self.device, dtype=torch.float32)
        adv_tensor = self._check_finite(adv_tensor, ctx="advantage")
        log_probs = log_probs.to(self.device)
        entropy = entropy.to(self.device)

        actor_loss = -(log_probs * adv_tensor).mean() - self.entropy_coef * entropy.mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cnn.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.optimizer.step()

        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "advantage_mean": float(np.mean(advantages)),
            "return_error_mean": float(np.mean(return_errors)),
            "value_pred_mean": float(np.mean(value_preds)),
            "value_next_mean": float(np.mean(value_nexts)),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {"cnn": self.cnn.state_dict(), "actor": self.actor.state_dict(), "optim": self.optimizer.state_dict()}, path
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.cnn.load_state_dict(ckpt["cnn"])
        self.actor.load_state_dict(ckpt["actor"])
        self.optimizer.load_state_dict(ckpt["optim"])
        self.prev_value = np.zeros(self.num_envs, dtype=np.float32)
        self.prev_feature = np.zeros((self.num_envs, self._critic_kwargs["number_of_features"]), dtype=np.float32)
