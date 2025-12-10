# Copyright 2025
# Swift-Sarsa Agent for Atari
#
# Uses the official Swift-Sarsa C++ implementation from swiftsarsa package
# Based on "Swift-Sarsa: Fast and Robust Linear Control" (Javed & Sutton, 2025)

"""
Swift-Sarsa agent for Atari games.

This agent uses the official C++ SwiftSarsaBinaryFeatures implementation which:
- Uses efficient sparse eligibility trace tracking
- Has a simple API: learn(features, reward, gamma, action) and get_action_values(features)

TEMPORAL ALIGNMENT (CRITICAL):
The C++ learn() method computes:
    δ = r + γ * Q(s_t, a_t) - v_old
    v_old = Q(s_t, a_t)  (for next call)

So the intended call pattern is:
    learn(φ(s_t), r_{t+1}, γ, a_t)
    
where v_old already holds Q(s_{t-1}, a_{t-1}) from the previous call.

This means we must pass the features of the state we ACTED ON, not the next state.
We store (features_t, action_t) at act() time and use them in observe().
"""

from __future__ import annotations

import os
import numpy as np
from typing import Dict, Iterable, Optional, List, Sequence, Tuple
from collections import deque
import logging
import pickle

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_agents import VectorAgent

# Import the official Swift-Sarsa C++ implementation
# Package name is 'swiftsarsa' (one word, no underscore)
from swiftsarsa import SwiftSarsaBinaryFeatures, SwiftSarsa

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AtariFeatureExtractor (unchanged)
# ---------------------------------------------------------------------------

class AtariFeatureExtractor:
    """
    Binary feature extractor matching the SwiftTD paper (Section 4.1),
    plus generic memory features inspired by Javed's thesis (Fig. 9.2).
    
    SwiftTD pixel pipeline:
    1. Raw ALE frame: 210×160×3 (H×W×RGB), uint8 [0,255]
    2. Resize to 105×80×3
    3. Per-channel 8-bin one-hot: bin = p // 32 ∈ {0,...,7}
       - Each pixel becomes 8 binary features (one-hot per channel)
       - Stacked: 105×80×24 (3 RGB channels × 8 bins)
    4. Flatten: 201,600 binary features
    
    Extra features:
    - Previous action one-hot (num_actions)
    - Action history for last K_actions steps (num_actions × K_actions)
    - Reward sign history for last K_rewards steps (2 × K_rewards)
      (pos_k and neg_k bits)
    - Time-since-last-positive-reward bucket (gap_bins one-hot)
    - Current cumulant one-hot for {-1, 0, +1} (3 bits)
    
    All features are binary indices for SwiftSarsaBinaryFeatures.

    NOTE: The full 9.2 imprinting buffer is implemented as a *separate*
    ImprintingFeatureBuffer class above. You can:
      - Construct one with a chosen subset of features/cumulants
      - Concatenate its active indices onto the indices returned here.
    """
    
    def __init__(
        self,
        height: int = 105,      # SwiftTD paper: 105
        width: int = 80,        # SwiftTD paper: 80
        num_bins: int = 8,      # SwiftTD paper: 8 (p // 32)
        num_actions: int = 18,
        use_grayscale: bool = False,   # SwiftTD uses RGB
        use_frame_diff: bool = True,  # SwiftTD doesn't use frame diff
        use_cumulant: bool = True,     # One-hot encode {-1,0,+1} cumulant
        # --- generic memory features (cross-game) ---
        K_actions: int = 3,            # how many past actions to encode
        K_rewards: int = 3,            # how many past reward signs to encode
        use_reward_gap: bool = True,   # time-since-last-positive-reward bucket
        gap_bins: int = 6,             # number of buckets for reward gap
    ):
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.num_actions = num_actions
        self.use_grayscale = use_grayscale
        self.use_frame_diff = use_frame_diff
        self.use_cumulant = use_cumulant
        
        # Memory config
        self.K_actions = K_actions
        self.K_rewards = K_rewards
        self.use_reward_gap = use_reward_gap
        self.gap_bins = gap_bins
        
        # --- Pixel feature dimensions (SwiftTD) ---
        num_channels = 1 if use_grayscale else 3
        if use_frame_diff:
            num_channels *= 2  # current + diff
        self.num_channels = num_channels
        
        # Pixel features: H × W × C × bins (one-hot per channel per pixel)
        self.pixel_features = height * width * num_channels * num_bins
        
        # --- Layout of extra features ---
        # [pixels |
        #  prev_action (num_actions) |
        #  action_history (num_actions * K_actions) |
        #  reward_history (2 * K_rewards) |
        #  reward_gap (gap_bins, optional) |
        #  cumulant (3, optional; one-hot {-1,0,+1})
        # ]
        
        # Offsets
        self.prev_action_offset = self.pixel_features
        self.action_hist_offset = self.prev_action_offset + num_actions
        self.reward_hist_offset = self.action_hist_offset + num_actions * self.K_actions
        
        if self.use_reward_gap:
            self.reward_gap_offset = self.reward_hist_offset + 2 * self.K_rewards
            cumulant_base = self.reward_gap_offset + self.gap_bins
        else:
            self.reward_gap_offset = None
            cumulant_base = self.reward_hist_offset + 2 * self.K_rewards
        
        self.cumulant_base = cumulant_base if self.use_cumulant else None
        self.cumulant_features = 3 if self.use_cumulant else 0
        
        extra_features = (
            num_actions +                       # prev action
            num_actions * self.K_actions +      # action history
            2 * self.K_rewards +                # reward history
            (self.gap_bins if self.use_reward_gap else 0) +
            self.cumulant_features              # cumulant one-hot (-1,0,+1)
        )
        self.total_features = self.pixel_features + extra_features
        
        # State for memory
        self.prev_action = 0
        self.prev_frame = None
        self.action_hist = deque(
            [0] * self.K_actions, maxlen=self.K_actions
        )  # store last K actions (ints)
        self.reward_hist = deque(
            [0] * self.K_rewards, maxlen=self.K_rewards
        )  # store last K reward signs {-1,0,1}
        self.steps_since_pos = 10**9  # large initial value
        self.current_cumulant = 0.0  # {-1, 0, 1}
        
        # Pre-compute base indices for vectorized pixel feature extraction
        y_idx = np.arange(height).reshape(-1, 1, 1)
        x_idx = np.arange(width).reshape(1, -1, 1)
        c_idx = np.arange(num_channels).reshape(1, 1, -1)
        
        self._base_indices = (
            y_idx * (width * num_channels * num_bins) +
            x_idx * (num_channels * num_bins) +
            c_idx * num_bins
        ).astype(np.int32)
    
        # Approx expected active features per frame
        active_per_frame = height * width * num_channels  # one bin per pixel per channel
        active_per_frame += 1  # prev action
        active_per_frame += self.K_actions  # one action per lag
        active_per_frame += self.K_rewards  # one sign bit per lag (pos or neg)
        if self.use_reward_gap:
            active_per_frame += 1           # one gap bucket
        if self.use_cumulant:
            active_per_frame += 1           # exactly one cumulant bit is active
        
        self._active_per_frame = active_per_frame
        
        logger.info(
            f"AtariFeatureExtractor (SwiftTD-style + memory): "
            f"{height}x{width}x{num_channels}, {num_bins} bins, "
            f"K_actions={K_actions}, K_rewards={K_rewards}, gap_bins={gap_bins}, "
            f"total={self.total_features} features, "
            f"active_per_frame≈{active_per_frame}"
        )
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target resolution. Keep as uint8 for binning."""
        # Resize to target resolution
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale if needed
        if self.use_grayscale:
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            elif frame.ndim == 3 and frame.shape[2] == 1:
                frame = frame[:, :, 0]
            # Ensure 3D for consistency
            if frame.ndim == 2:
                frame = frame[:, :, np.newaxis]
        else:
            # Keep RGB - handle 2D grayscale input
            if frame.ndim == 2:
                frame = np.stack([frame, frame, frame], axis=-1)
            elif frame.ndim == 3 and frame.shape[2] != 3:
                frame = frame[:, :, :3]
        
        return frame
    
    def _reward_gap_bin(self) -> int:
        """Bucket steps_since_pos into one of gap_bins."""
        # Simple hand-crafted buckets similar to:
        # 0, 1–5, 6–20, 21–50, 51–100, >100 (for gap_bins=6)
        s = self.steps_since_pos
        if self.gap_bins == 0:
            return 0
        if self.gap_bins == 6:
            if s == 0:
                return 0
            elif s <= 5:
                return 1
            elif s <= 20:
                return 2
            elif s <= 50:
                return 3
            elif s <= 100:
                return 4
            else:
                return 5
        # Fallback: uniform split
        # clamp s to [0, max_s] and linearly bucket
        max_s = 200
        s = max(0, min(s, max_s))
        bucket_width = (max_s + 1) / self.gap_bins
        return int(s // bucket_width)
    

    
    def update_reward(self, reward: float):
        """Update reward history and cumulant."""
        # Store reward sign in history
        if reward > 0:
            self.reward_hist.append(1)
            self.steps_since_pos = 0
        elif reward < 0:
            self.reward_hist.append(-1)
            self.steps_since_pos += 1
        else:
            self.reward_hist.append(0)
            self.steps_since_pos += 1
            
        # Store current cumulant value (reward sign)
        if reward > 0:
            self.current_cumulant = 1.0
        elif reward < 0:
            self.current_cumulant = -1.0
        else:
            self.current_cumulant = 0.0

    def extract(self, frame: np.ndarray) -> List[int]:
        """
        Extract sparse binary feature indices from frame.
        Returns list[int] (all binary features have value 1.0).
        """
        # --- Pixel part ---
        current = self._preprocess_frame(frame)
        
        if self.use_frame_diff:
            if self.prev_frame is None:
                self.prev_frame = current.copy()
            diff = np.abs(current.astype(np.int32) - self.prev_frame.astype(np.int32))
            diff = np.clip(diff, 0, 255).astype(np.uint8)
            combined = np.concatenate([current, diff], axis=-1)
            self.prev_frame = current.copy()
        else:
            combined = current
        
        if self.num_bins == 8:
            # Exact SwiftTD formula: p // 32
            binned = combined.astype(np.int32) // 32
        else:
            binned = np.minimum(
                combined.astype(np.int32) * self.num_bins // 256,
                self.num_bins - 1
            )
        
        # Each (y,x,channel) maps to base_index + bin_value
        indices = (self._base_indices + binned).ravel()
        
        # --- Previous action one-hot ---
        indices = np.append(indices, self.prev_action_offset + self.prev_action)
        
        # --- Action history (last K_actions) ---
        base = self.action_hist_offset
        for k, act in enumerate(self.action_hist):
            # one-hot for action taken k steps ago
            indices = np.append(indices, base + k * self.num_actions + act)
        
        # --- Reward history (last K_rewards signs) ---
        # For each lag k, two possible bits: pos_k or neg_k
        base = self.reward_hist_offset
        for k, s in enumerate(self.reward_hist):
            if s > 0:
                indices = np.append(indices, base + 2 * k)       # pos_k
            elif s < 0:
                indices = np.append(indices, base + 2 * k + 1)   # neg_k
        
        # --- Reward-gap bucket ---
        if self.use_reward_gap and self.reward_gap_offset is not None:
            gap_bin = self._reward_gap_bin()
            indices = np.append(indices, self.reward_gap_offset + gap_bin)
        
        # --- Cumulant one-hot (-1, 0, +1) ---
        if self.use_cumulant and self.cumulant_base is not None:
            cumulant_val = getattr(self, "current_cumulant", 0.0)
            if cumulant_val > 0:
                indices = np.append(indices, self.cumulant_base + 2)
            elif cumulant_val < 0:
                indices = np.append(indices, self.cumulant_base)
            else:
                indices = np.append(indices, self.cumulant_base + 1)
        
        return [int(idx) for idx in indices]
    
    def set_cumulant(self, value: float):
        """Manually set cumulant value (one of -1, 0, +1) for next extraction."""
        if value > 0:
            self.current_cumulant = 1.0
        elif value < 0:
            self.current_cumulant = -1.0
        else:
            self.current_cumulant = 0.0
    
    def set_prev_action(self, action: int):
        """Store previous action and push into action history."""
        self.prev_action = action
        if self.K_actions > 0:
            self.action_hist.appendleft(action)
    
    def reset(self):
        """Reset frame history and memory on episode end."""
        self.prev_frame = None
        self.prev_action = 0
        self.action_hist = deque([0] * self.K_actions, maxlen=self.K_actions)
        self.reward_hist = deque([0] * self.K_rewards, maxlen=self.K_rewards)
        self.steps_since_pos = 10**9
        self.current_cumulant = 0.0
    
    def get_active_feature_count(self) -> int:
        """Return approximate number of active features per frame."""
        return self._active_per_frame


# ---------------------------------------------------------------------------
# Neural Baseline Helpers (CNN + Normalizer + Dense-to-Sparse)
# ---------------------------------------------------------------------------

class SwiftTDNeuralBody(nn.Module):
    """
    CNN body matching the SwiftTD-style neural baseline:
    - Input: (B, 24, 105, 80) binary tensor (binned RGB, 8 bins/channel).
    - Single conv layer: 24 -> 25, kernel=3, stride=2, ReLU, flatten.

    Output: (B, feat_dim) feature vector. feat_dim is inferred at runtime.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.conv = nn.Conv2d(24, 25, kernel_size=3, stride=2)
        # Small uniform init (safer with Swift)
        nn.init.uniform_(self.conv.weight, -0.01, 0.01)
        nn.init.zeros_(self.conv.bias)
        self.device = torch.device(device)

        # Infer feat_dim by running a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 24, 105, 80)
            out = self.conv(dummy)
            self.feat_dim = int(out.numel())
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 24, 105, 80)
        returns: (B, feat_dim)
        """
        x = x.to(self.device)
        out = self.conv(x)
        out = F.relu(out)
        return out.reshape(out.shape[0], -1)


class NeuralAtariExtractor:
    """
    Neural feature extractor that:
    - Preprocesses frame to 105x80x3 (RGB).
    - Bins pixels to 8 levels and builds a (24, 105, 80) binary tensor.
    - Runs the SwiftTDNeuralBody CNN.
    - Concatenates prev-action 1-hot (num_actions) + reward sign 1-hot (3).
    Returns a dense NumPy feature vector (phi_full, shape (D,)).
    """

    def __init__(self, num_actions: int, device: str = "cpu"):
        self.height = 105
        self.width = 80
        self.num_bins = 8
        self.num_actions = num_actions
        self.device = device

        self.cnn = SwiftTDNeuralBody(device=device)

        # Extra features: prev-action (num_actions) + reward sign (-1,0,+1)
        self.extra_dim = num_actions + 3
        self.feat_dim = self.cnn.feat_dim + self.extra_dim

        # State
        self.prev_action = 0
        self.prev_reward_sign = 0  # -1, 0, 1

    def reset(self):
        self.prev_action = 0
        self.prev_reward_sign = 0

    def set_prev_action(self, action: int):
        self.prev_action = int(action)

    def update_reward(self, reward: float):
        """Store just the sign of reward for extra features."""
        if reward > 0:
            self.prev_reward_sign = 1
        elif reward < 0:
            self.prev_reward_sign = -1
        else:
            self.prev_reward_sign = 0

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize to (105,80), keep RGB, uint8.
        Returns: (105,80,3) uint8.
        """
        # Resize
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        # Ensure RGB
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        elif frame.ndim == 3 and frame.shape[2] != 3:
            frame = frame[:, :, :3]

        return frame.astype(np.uint8)

    def _frame_to_binned_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert (105,80,3) uint8 frame to (1,24,105,80) binary tensor.
        Each pixel/channel gets 8 one-hot bins.
        """
        frame = self._preprocess_frame(frame)
        # binned in {0,...,7}
        binned = frame // 32  # (105,80,3), int

        H, W, C = binned.shape
        assert C == 3

        # Create one-hot over bins -> (H,W,3,8)
        one_hot = np.eye(self.num_bins, dtype=np.float32)[binned]  # (H,W,3,8)

        # Reshape to (24,H,W): 3 channels * 8 bins
        one_hot = one_hot.reshape(H, W, C * self.num_bins)
        one_hot = np.transpose(one_hot, (2, 0, 1))  # (24,H,W)

        x = torch.from_numpy(one_hot).unsqueeze(0)  # (1,24,H,W)
        return x

    def extract_dense(self, frame: np.ndarray) -> np.ndarray:
        """
        Main API: frame -> dense feature vector phi_full (np.float32, shape (feat_dim,)).
        """
        with torch.no_grad():
            x = self._frame_to_binned_tensor(frame)  # (1,24,105,80)
            cnn_feats = self.cnn(x).cpu().numpy().reshape(-1)  # (feat_dim_cnn,)

        # Extra: prev-action one-hot
        extra = np.zeros(self.extra_dim, dtype=np.float32)
        if 0 <= self.prev_action < self.num_actions:
            extra[self.prev_action] = 1.0

        # Reward sign one-hot in positions [num_actions : num_actions+3]
        sign_offset = self.num_actions
        if self.prev_reward_sign < 0:
            extra[sign_offset + 0] = 1.0
        elif self.prev_reward_sign == 0:
            extra[sign_offset + 1] = 1.0
        else:
            extra[sign_offset + 2] = 1.0

        phi_full = np.concatenate([cnn_feats.astype(np.float32), extra], axis=0)
        assert phi_full.shape[0] == self.feat_dim
        return phi_full


class FeatureNormalizer:
    """
    Online feature normalizer:
    - Maintains running mean and variance per feature.
    - Normalizes to approx N(0,1) and clips to [-clip, clip].
    """

    def __init__(self, dim: int, beta: float = 0.9999, eps: float = 1e-3, clip: float = 5.0):
        self.dim = dim
        self.beta = beta
        self.eps = eps
        self.clip = clip

        self.mu = np.zeros(dim, dtype=np.float32)
        self.var = np.ones(dim, dtype=np.float32)
        self.initialized = False

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        x: (dim,) dense feature vector.
        returns: normalized/clipped vector (dim,).
        """
        x = x.astype(np.float32, copy=False)

        if not self.initialized:
            # Initialize mean to first batch, var to 1
            self.mu = x.copy()
            self.var.fill(1.0)
            self.initialized = True

        # Update mean and variance (exponential moving)
        diff = x - self.mu
        self.mu = self.beta * self.mu + (1.0 - self.beta) * x
        self.var = self.beta * self.var + (1.0 - self.beta) * (diff * diff)

        std = np.sqrt(self.var) + self.eps
        x_norm = (x - self.mu) / std
        # Clip to keep τ,b,α stable
        x_norm = np.clip(x_norm, -self.clip, self.clip)
        return x_norm


def dense_to_index_value_pairs(phi: np.ndarray, threshold: float = 1e-6) -> List[Tuple[int, float]]:
    """
    Convert dense feature vector into (index, value) pairs.
    Small values (|v| <= threshold) are dropped for efficiency.
    """
    phi = phi.astype(np.float32, copy=False)
    idx = np.nonzero(np.abs(phi) > threshold)[0]
    return [(int(i), float(phi[i])) for i in idx]


# ---------------------------------------------------------------------------
# SwiftSarsaCore, VectorSwiftSarsaAgent, Agent
# ---------------------------------------------------------------------------


class SwiftSarsaCore:
    """
    Swift-Sarsa core using the official C++ implementation.
    
    TEMPORAL ALIGNMENT:
    The C++ learn(features, reward, gamma, action) computes:
        δ = r + γ * Q(s, a) - v_old
    where Q(s, a) is computed from the features passed in, and v_old is
    stored from the previous call.
    
    The intended call pattern is:
        At step t, call: learn(φ(s_t), r_{t+1}, γ, a_t)
    where v_old holds Q(s_{t-1}, a_{t-1}).
    
    So we must store (features_t, action_t) at act() time and use them
    when observe() is called with r_{t+1}.
    """
    
    def __init__(
        self,
        num_envs: int,
        num_actions: int,
        seed: int = 0,
        # Pixel feature params - SwiftTD paper defaults
        screen_height: int = 105,     # SwiftTD paper
        screen_width: int = 80,       # SwiftTD paper
        num_bins: int = 8,            # SwiftTD paper (p // 32)
        use_grayscale: bool = False,  # SwiftTD uses RGB
        use_frame_diff: bool = True,  # Motion channel (default True)
        use_cumulant: bool = True,    # One-hot cumulant {-1,0,+1}
        # Imprinting features (Section 9.2)
        K_actions: int = 3,
        K_rewards: int = 3,
        use_reward_gap: bool = True,
        gap_bins: int = 6,
        # Swift-Sarsa hyperparameters - Algorithm 1 defaults
        gamma: float = 0.99,
        lambda_: float = 0.9,
        alpha_init: float = 1e-7,     # Algorithm 1 default
        theta: float = 1e-3,          # Meta step-size (sweep 1e-7 to 1e-1)
        eta: float = 1.0,             # As in paper experiments (or 0.1 from alg header)
        decay: float = 0.999,         # Step-size decay (or 0.9999 as in Fig. 1)
        epsilon: float = 1e-6,        # Trace clearing threshold
        eta_min: float = 3.06e-7,     # exp(-15) ≈ 3.06e-7
        # Policy
        temperature: float = 0.1,
        exploration_epsilon: float = 0.1,
        # Reward
        reward_scale: float = 1.0,
        reward_clip: Optional[float] = 1.0,
        # Neural Baseline
        use_neural_features: bool = False,
        neural_device: str = "cpu",
    ):
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.gamma = gamma
        self.temperature = temperature
        self.exploration_epsilon = exploration_epsilon
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        
        np.random.seed(seed)
        
        self.use_neural_features = use_neural_features
        self.extractors = []
        self.neural_extractors = []
        self.normalizers = []
        self.use_continuous = False

        if self.use_neural_features:
            # One neural extractor + normalizer per env
            for i in range(num_envs):
                ne = NeuralAtariExtractor(num_actions=num_actions, device=neural_device)
                self.neural_extractors.append(ne)
                self.normalizers.append(FeatureNormalizer(dim=ne.feat_dim))

            num_features = self.neural_extractors[0].feat_dim
            active_per_frame = num_features  # dense; used only for logging
        else:
            # Original binary AtariFeatureExtractor path
            for i in range(num_envs):
                extractor = AtariFeatureExtractor(
                    height=screen_height,
                    width=screen_width,
                    num_bins=num_bins,
                    num_actions=num_actions,
                    use_grayscale=use_grayscale,
                    use_frame_diff=use_frame_diff,
                    use_cumulant=use_cumulant,
                    K_actions=K_actions,
                    K_rewards=K_rewards,
                    use_reward_gap=use_reward_gap,
                    gap_bins=gap_bins,
                )
                self.extractors.append(extractor)
            
            num_features = self.extractors[0].total_features
            active_per_frame = self.extractors[0].get_active_feature_count()
        
        # Auto-scale alpha if using default and feature count is very different from pixels
        # Pixel features: ~25k active. SwiftTD default alpha=1e-7 is for ~25k active features.
        if alpha_init == 1e-7:  # User didn't override default
            pass
        
        expected_tau = active_per_frame * alpha_init
        
        logger.info(
            f"Swift-Sarsa (C++): {num_features} total features, ~{active_per_frame} active/frame, "
            f"{num_actions} actions, {num_envs} envs"
        )
        logger.info(
            f"Hyperparams: alpha={alpha_init}, theta={theta}, eta={eta}, "
            f"tau≈{expected_tau:.6f} (should be < eta={eta})"
        )
        
        # Warn if tau is extremely small (likely won't learn)
        if expected_tau < 1e-4:
            logger.warning(
                f"WARNING: tau={expected_tau:.2e} is very small! "
                f"Consider increasing alpha_init. Recommended: alpha_init={0.01/active_per_frame:.2e}"
            )
        
        # Create official C++ Swift-Sarsa learners (one per env)
        # Select backend based on feature type
        if self.use_neural_features:
            AgentClass = SwiftSarsa  # Expects (index, value) pairs
        else:
            AgentClass = SwiftSarsaBinaryFeatures  # Expects indices only

        self.agents = [
            AgentClass(
                num_features,
                num_actions,
                lambda_,       # lambda (trace decay)
                alpha_init,    # alpha (initial step-size)
                theta,         # meta_step_size
                eta,           # eta (max step-size bound)
                decay,         # decay (step-size decay factor)
                epsilon,       # epsilon (trace clearing threshold)
                eta_min,       # eta_min (min step-size)
            )
            for _ in range(num_envs)
        ]
        
        # CORRECT TEMPORAL ALIGNMENT for Sarsa:
        # C++ learns with: δ = r_{t+1} + γ * Q(s_t, a_t) - v_old
        # where v_old = Q(s_{t-1}, a_{t-1}).
        # We therefore call: learn(φ(s_t), r_{t+1}, γ, a_t)
        # with v_old already holding Q(s_{t-1}, a_{t-1}).
        self.last_features: List[Optional[List[Tuple[int, float]]]] = [None] * num_envs
        self.last_actions: List[Optional[int]] = [None] * num_envs
        
        # Track v_old per env for TD error computation in Python
        # (C++ doesn't expose TD error, so we compute it ourselves)
        self.v_old: List[float] = [0.0] * num_envs
        
        # Statistics
        self.frame_count = 0
        self.last_loss = 0.0
        self.loss_ema = None
        self.last_avg_q = 0.0
        self.last_max_q = 0.0
        self.last_td_error = 0.0
        self.td_error_ema = None
        
        # Exploration decay - MUCH LONGER for sparse reward games
        self.initial_epsilon = exploration_epsilon
        self.epsilon_decay_frames = 1_000_000  # 10x longer decay
        self.min_epsilon = 0.05  # Never go below 5% exploration
        self.epsilon = exploration_epsilon
    
    def reset(self) -> None:
        """Reset agent state (traces, v_old, extractors) at start of training or between episodes."""
        for env_idx in range(self.num_envs):
            if self.use_neural_features:
                self.neural_extractors[env_idx].reset()
            else:
                self.extractors[env_idx].reset()
            self.last_features[env_idx] = None
            self.last_actions[env_idx] = None
            self.v_old[env_idx] = 0.0
            # Reset C++ agent traces
            if hasattr(self.agents[env_idx], "reset_episode"):
                self.agents[env_idx].reset_episode()
    
    def _select_action(self, q_values: List[float]) -> int:
        """Select action using ε-greedy with softmax."""
        q_arr = np.array(q_values)
        
        # Handle NaN/Inf Q-values
        if np.any(~np.isfinite(q_arr)):
            logger.warning("Non-finite Q-values detected")
            q_arr = np.nan_to_num(q_arr, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # ε-greedy exploration
        if self.epsilon > 0 and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        # Numerically stable softmax
        q_shifted = q_arr - np.max(q_arr)
        q_scaled = np.clip(q_shifted / max(self.temperature, 1e-8), -50, 50)
        exp_q = np.exp(q_scaled)
        probs = exp_q / (np.sum(exp_q) + 1e-10)
        
        # Safety check
        if np.any(~np.isfinite(probs)) or np.abs(probs.sum() - 1.0) > 0.01:
            return np.random.randint(self.num_actions)
        
        return np.random.choice(self.num_actions, p=probs)
    
    def act(self, observations: np.ndarray) -> np.ndarray:
        """
        Return actions for current observations.
        
        We compute actions directly from the current observation, and stash
        (features_t, action_t) so observe() can perform the TD update once
        r_{t+1} arrives.
        """
        actions = np.zeros(self.num_envs, dtype=np.int64)
        q_values_all = []
        
        # Decay exploration over time (but never below min_epsilon)
        decay_factor = max(0, 1 - self.frame_count / self.epsilon_decay_frames)
        self.epsilon = max(self.min_epsilon, self.initial_epsilon * decay_factor)
        
        for env_idx in range(self.num_envs):
            if self.use_neural_features:
                # Dense neural features
                ne = self.neural_extractors[env_idx]
                phi_dense = ne.extract_dense(observations[env_idx])
                phi_norm = self.normalizers[env_idx].normalize(phi_dense)
                features = dense_to_index_value_pairs(phi_norm)
            else:
                # Original sparse indices
                features = self.extractors[env_idx].extract(observations[env_idx])

            q_values = self.agents[env_idx].get_action_values(features)
            q_values_all.append(q_values)
            
            action = self._select_action(q_values)
            actions[env_idx] = action
            
            # Store for learning once reward arrives
            self.last_features[env_idx] = features
            self.last_actions[env_idx] = action
            
            # Update prev action for feature extraction
            if self.use_neural_features:
                self.neural_extractors[env_idx].set_prev_action(action)
            else:
                self.extractors[env_idx].set_prev_action(action)
        
        # Stats
        if q_values_all:
            q_arr = np.array(q_values_all)
            self.last_avg_q = float(q_arr.mean())
            self.last_max_q = float(q_arr.max())
        
        return actions



    def observe(
        self,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
    ) -> None:
        """
        Observe transition and do Sarsa learning update.
        
        CORRECT TEMPORAL ALIGNMENT for Sarsa:
        The C++ learn() computes: δ = r + γ * Q(s, a) - v_old
        For Sarsa TD: δ = r_{t+1} + γ * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
        
        We store (features_t, action_t) when act() is called on s_t and
        feed them here when r_{t+1} arrives. This matches the intended
        temporal alignment of the C++ backend (v_old holds the previous Q).
        """
        td_errors = []
        
        for env_idx in range(self.num_envs):
            # Process reward (r_{t+1})
            reward = float(rewards[env_idx]) * self.reward_scale
            if self.reward_clip is not None:
                reward = np.clip(reward, -self.reward_clip, self.reward_clip)

            # Update reward state for features
            if self.use_neural_features:
                self.neural_extractors[env_idx].update_reward(reward)
            else:
                if hasattr(self.extractors[env_idx], "update_reward"):
                    self.extractors[env_idx].update_reward(reward)

            done = bool(terminations[env_idx] or truncations[env_idx])
            gamma = 0.0 if done else self.gamma
            
            prev_features = self.last_features[env_idx]
            prev_action = self.last_actions[env_idx]
            if prev_features is None or prev_action is None:
                # No action taken yet (shouldn't happen in steady state)
                continue
            
            # SARSA update on the state-action we acted on
            # learn(φ(s_t), r_{t+1}, γ, a_t) where v_old is Q(s_{t-1}, a_{t-1})
            q_next = self.agents[env_idx].learn(prev_features, reward, gamma, prev_action)
            
            # Compute TD error for logging
            td_error = reward + gamma * q_next - self.v_old[env_idx]
            td_errors.append(abs(td_error))
            
            # Update our v_old tracker
            self.v_old[env_idx] = q_next
            
            if done:
                # Reset for new episode
                if hasattr(self.agents[env_idx], "reset_episode"):
                    self.agents[env_idx].reset_episode()  # Reset C++ v_old and traces
                
                if self.use_neural_features:
                    self.neural_extractors[env_idx].reset()
                else:
                    self.extractors[env_idx].reset()
                
                self.v_old[env_idx] = 0.0  # Reset Python v_old tracker too
                self.last_features[env_idx] = None
                self.last_actions[env_idx] = None
        
        # Update TD error statistics
        if td_errors:
            mean_td = float(np.mean(td_errors))
            self.last_td_error = mean_td
            self.last_loss = mean_td ** 2
            if self.td_error_ema is None:
                self.td_error_ema = mean_td
            else:
                self.td_error_ema = 0.99 * self.td_error_ema + 0.01 * mean_td
            if self.loss_ema is None:
                self.loss_ema = self.last_loss
            else:
                self.loss_ema = 0.99 * self.loss_ema + 0.01 * self.last_loss
        
        self.frame_count += self.num_envs
    
    def train_step(self) -> None:
        """No-op - updates happen in observe()."""
        pass
    
    def save_model(self, path: str) -> None:
        """Save model - not implemented for C++ backend."""
        logger.warning("save_model not implemented for C++ Swift-Sarsa")
    
    def load_model(self, path: str) -> None:
        """Load model - not implemented for C++ backend."""
        logger.warning("load_model not implemented for C++ Swift-Sarsa")


class VectorSwiftSarsaAgent:
    """
    Vectorized Swift-Sarsa agent implementing VectorAgent protocol.
    
    Uses the official C++ SwiftSarsaBinaryFeatures implementation.
    
    Usage:
        python sim_latency_vec.py --agent agent_swift_sarsa:VectorSwiftSarsaAgent \
            --reduce_action_set --rom MsPacman --num_envs 4 --total_frames 500000
    
    Hyperparameter tuning via --agent_arg:
        --agent_arg alpha_init=1e-3 --agent_arg theta=1e-2 --agent_arg temperature=0.05
    """
    
    def __init__(
        self,
        *,
        num_envs: int,
        seed: int,
        num_actions: int,
        results_dir: Optional[str] = None,
        total_frames: int = 1_000_000,
        # Pixel feature params - SwiftTD paper defaults (Section 4.1)
        screen_height: int = 105,     # SwiftTD paper
        screen_width: int = 80,       # SwiftTD paper
        num_bins: int = 8,            # SwiftTD paper (p // 32)
        use_grayscale: bool = False,  # SwiftTD uses RGB
        use_frame_diff: bool = True,  # Motion channel
        # Swift-Sarsa hyperparameters - Algorithm 1 defaults
        gamma: float = 0.99,
        lambda_: float = 0.9,
        alpha_init: float = 1e-7,     # Algorithm 1 default
        theta: float = 1e-3,          # Meta step-size (sweep 1e-7 to 1e-1)
        eta: float = 1.0,             # As in paper experiments
        decay: float = 0.999,         # Step-size decay
        epsilon: float = 1e-6,        # Trace clearing threshold
        eta_min: float = 3.06e-7,     # exp(-15)
        # Policy
        temperature: float = 0.5,     # Softmax temperature (was 0.1)
        exploration_epsilon: float = 0.2,  # Epsilon-greedy (was 0.1)
        # Reward
        reward_scale: float = 1.0,
        reward_clip: float = 1.0,
        # Neural Baseline
        use_neural_features: bool = False,
        neural_device: str = "cpu",
        **kwargs,
    ):
        self.results_dir = results_dir
        self.core = SwiftSarsaCore(
            num_envs=num_envs,
            num_actions=num_actions,
            seed=seed,
            screen_height=screen_height,
            screen_width=screen_width,
            num_bins=num_bins,
            use_grayscale=use_grayscale,
            use_frame_diff=use_frame_diff,
            gamma=gamma,
            lambda_=lambda_,
            alpha_init=alpha_init,
            theta=theta,
            eta=eta,
            decay=decay,
            epsilon=epsilon,
            eta_min=eta_min,
            temperature=temperature,
            exploration_epsilon=exploration_epsilon,
            reward_scale=reward_scale,
            reward_clip=reward_clip,
            use_neural_features=use_neural_features,
            neural_device=neural_device,
        )
        
        logger.info(
            f"VectorSwiftSarsaAgent: {num_envs} envs, {num_actions} actions, "
            f"alpha={alpha_init}, theta={theta}, eta={eta}, lambda={lambda_}"
        )
    
    def reset(self, num_envs: int = None) -> None:
        """Reset agent state (traces, v_old, extractors)."""
        self.core.reset()
    
    def act(self, observations: np.ndarray) -> np.ndarray:
        return self.core.act(observations)
    
    def observe(
        self,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
        infos: Iterable[Dict] = (),
    ) -> None:
        self.core.observe(next_observations, rewards, terminations, truncations)
    
    def train_step(self) -> None:
        self.core.train_step()
    
    def save_model(self, path: str) -> None:
        self.core.save_model(path)
    
    def load_model(self, path: str) -> None:
        self.core.load_model(path)


# Single-env adapter for harness_physical.py
class Agent:
    """
    Single-environment Swift-Sarsa for harness_physical.py.
    
    The harness calls frame(obs, reward, end_of_episode) which provides:
    - obs: current observation (s_t)
    - reward: reward from PREVIOUS action (r_t, resulting from s_{t-1}, a_{t-1})
    - end_of_episode: whether this is a terminal state
    
    Temporal alignment:
    - When we receive (obs_t, r_t), we should learn from (s_{t-1}, a_{t-1}, r_t)
    - The core stores (features_{t-1}, action_{t-1}) which observe() uses
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        seed: int = 0,
        num_actions: int = 18,
        total_frames: int = 1_000_000,
        **kwargs,
    ):
        self.core = SwiftSarsaCore(
            num_envs=1,
            num_actions=num_actions,
            seed=seed,
            **kwargs,
        )
        self.initialized = False
    
    def frame(self, observation_rgb8: np.ndarray, reward: float, end_of_episode: int) -> int:
        """
        Process one frame and return action.
        
        Flow:
        1. If not first frame: observe(reward) triggers learn() with stored (s_{t-1}, a_{t-1})
        2. act(obs) picks action and stores (s_t, a_t) for next learn()
        """
        obs_batch = observation_rgb8[None, ...]
        done = bool(end_of_episode > 0)
        
        if not self.initialized:
            self.core.reset()
            self.initialized = True
            # First frame: just pick an action (no reward to learn from yet)
            actions = self.core.act(obs_batch)
            return int(actions[0])
        
        # Observe the transition
        self.core.observe(
            obs_batch,
            np.array([reward]),
            np.array([done]),
            np.array([False]),
        )
        
        if done:
            # Episode ended - reset will happen on next frame
            self.initialized = False
            return 0
        
        # Act on current observation
        actions = self.core.act(obs_batch)
        return int(actions[0])
    
    def save_model(self, path: str) -> None:
        self.core.save_model(path)
    
    def load_model(self, path: str) -> None:
        self.core.load_model(path)
