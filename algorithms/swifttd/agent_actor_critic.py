"""
Actor-Critic architecture with SwiftTD critic for Atari.

Implements A2C with:
- Shared CNN feature extractor (trained with TD backprop)
- Policy head (actor) for action selection
- SwiftTD critic for value function V(s)

Following the approach from SwiftTD paper Section 7:
- SwiftTD applied ONLY to the last layer (critic)
- CNN trained with standard TD(λ) backprop
- Separate learning rates for CNN and SwiftTD
"""

from math import isfinite
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import swifttd

from typing import Tuple
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
    """

    def __init__(
        self,
        num_actions: int,
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
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.fail_on_nonfinite = fail_on_nonfinite
        self._signal_eps = 1e-6
        self.critic_feature_scale = critic_feature_scale
        self.n_stack = n_stack
        self.input_size = input_size

        self.cnn = CNNFeatureExtractor(n_stack=n_stack, feature_dim=feature_dim, input_size=input_size).to(self.device)
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
        self.critic = self._build_critic()
        self.prev_value = 0.0
        self.prev_feature = None

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
        v = self.critic.step(feats_np[0].tolist(), 0.0)  # CRITICAL_LINE bootstrap V(s)
        self.prev_value = v
        self.prev_feature = feats_np[0]
        return feats_torch, feats_np

    def reset_done(self, done: int, obs_batch: np.ndarray):
        if not done:
            return
        _, feats_np = self._extract_features(obs_batch)
        v = self.critic.step(feats_np[0].tolist(), 0.0)  # CRITICAL_LINE reset V(s) after terminal
        self.prev_value = v
        self.prev_feature = feats_np[0]

    def _extract_features(self, obs_batch: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        assert obs_batch.shape == (1, self.input_size, self.input_size, self.n_stack), f"obs_batch.shape is expected to be (1, {self.input_size}, {self.input_size}, {self.n_stack}), but got {obs_batch.shape}"
        obs_batch = np.transpose(obs_batch, (0, 3, 1, 2))
        obs_t = torch.as_tensor(
            obs_batch, device=self.device, dtype=torch.float32
        ) / 255.0
        feats = self.cnn(obs_t)  # CRITICAL_LINE shared encoder φ(s_t) for all envs
        feats = self._check_finite(feats, ctx="features")
        return feats, feats.detach().cpu().numpy() * self.critic_feature_scale  # CRITICAL_LINE scaled φ(s) fed to SwiftTD

    def select_actions(self, obs_batch: np.ndarray):
        feats_torch, feats_np = self._extract_features(obs_batch)
        logits = self._check_finite(self.actor(feats_torch), ctx="logits")  # CRITICAL_LINE policy logits π(a|s)
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
        next_feats_torch, next_feats_np = self._extract_features(next_obs_batch)  # CRITICAL_LINE compute φ(s_{t+1})

        # Single environment processing
        V_current = float(self.prev_value)
        reward = float(rewards[0])
        assert isfinite(reward), f"Reward is not finite: {reward}"
        assert isfinite(V_current), f"V_current is not finite: {V_current}"
        done = bool(dones[0])

        target_feats = np.zeros_like(next_feats_np[0]) if done else next_feats_np[0]  # CRITICAL_LINE absorbing-state handling
        V_next = self.critic.step(target_feats.tolist(), reward)  # CRITICAL_LINE SwiftTD update/query
        if not isfinite(V_next):
            feats_min = float(np.min(target_feats))
            feats_max = float(np.max(target_feats))
            feats_norm = float(np.linalg.norm(target_feats))
            raise ValueError(
                f"V_next is not finite: {V_next} "
                f"(reward={reward}, done={done}, V_current={V_current}, "
                f"feat_min={feats_min}, feat_max={feats_max}, feat_norm={feats_norm})"
            )
        advantage = reward + (0.0 if done else self.gamma * V_next) - V_current  # CRITICAL_LINE TD error / advantage
        value_target = reward + (0.0 if done else self.gamma * V_next)
        return_error = (value_target - V_current) ** 2  # Squared TD target error

        # Update caches
        if done:
            self.prev_value = 0.0
            self.prev_feature = None
        else:
            self.prev_value = V_next
            self.prev_feature = target_feats

        # Actor loss
        adv_tensor = torch.as_tensor([advantage], device=self.device, dtype=torch.float32)
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
            "advantage": float(advantage),
            "value_pred": float(V_current),
            "value_next": float(V_next),
            "value_target": float(value_target),
            "return_error": float(return_error),
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
        self.prev_value = 0.0
        self.prev_feature = None
