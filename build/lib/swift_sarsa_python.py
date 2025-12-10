# Copyright 2025
# Swift-Sarsa: Fast and Robust Linear Control
#
# Based on:
# - "SwiftTD: A Fast and Robust Algorithm for TD Learning" (Javed et al., RLC 2024)
# - "Swift-Sarsa: Fast and Robust Linear Control" (Javed & Sutton, 2025)
#
# Extends SwiftTD to on-policy control using True Online Sarsa(λ).

"""
Swift-Sarsa: True Online Sarsa(λ) with:
1. Per-feature step-size optimization (meta-learning)
2. Overshoot bound to prevent divergence
3. Step-size decay when bound triggers

This implementation supports sparse binary features for efficiency.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Set, Optional
import pickle


@dataclass
class SwiftSarsaConfig:
    """Configuration for Swift-Sarsa algorithm."""
    num_features: int           # n: Total number of features
    num_actions: int            # m: Number of discrete actions
    gamma: float = 0.99         # Discount factor
    lambda_: float = 0.95       # Eligibility trace decay (λ)
    alpha_init: float = 1e-7    # Initial step-size parameter
    theta: float = 1e-3         # Meta step-size for step-size optimization
    eta: float = 0.1            # Max correction ratio (overshoot bound)
    epsilon: float = 0.999      # Step-size decay factor (when tau > eta)
    eta_min: float = 1e-15      # Minimum step-size (e^{-15})
    trace_threshold: float = 1e-10  # Threshold for clearing small traces (separate from epsilon!)


class SwiftSarsaBinary:
    """
    Swift-Sarsa with sparse binary features.
    
    Implements Algorithm 1 from "Swift-Sarsa: Fast and Robust Linear Control".
    
    Features are binary (0 or 1). Only indices of active features (=1) are provided
    for efficiency with high-dimensional sparse representations.
    """
    
    def __init__(self, config: SwiftSarsaConfig):
        self.cfg = config
        n = config.num_features
        m = config.num_actions
        
        # Weights: w^j for each action j (Equation 2 in paper)
        self.w: List[np.ndarray] = [np.zeros(n, dtype=np.float64) for _ in range(m)]
        
        # Step-size parameters β (actual step-size α = e^β)
        log_alpha_init = np.log(max(config.alpha_init, 1e-300))
        self.beta: List[np.ndarray] = [np.full(n, log_alpha_init, dtype=np.float64) for _ in range(m)]
        
        # Eligibility traces per action
        self.z: List[np.ndarray] = [np.zeros(n, dtype=np.float64) for _ in range(m)]
        self.z_delta: List[np.ndarray] = [np.zeros(n, dtype=np.float64) for _ in range(m)]
        self.z_bar: List[np.ndarray] = [np.zeros(n, dtype=np.float64) for _ in range(m)]
        
        # Meta-gradient traces per action
        self.p: List[np.ndarray] = [np.zeros(n, dtype=np.float64) for _ in range(m)]
        self.h: List[np.ndarray] = [np.zeros(n, dtype=np.float64) for _ in range(m)]
        self.h_old: List[np.ndarray] = [np.zeros(n, dtype=np.float64) for _ in range(m)]
        self.h_temp: List[np.ndarray] = [np.zeros(n, dtype=np.float64) for _ in range(m)]
        
        # Weight changes
        self.delta_w: List[np.ndarray] = [np.zeros(n, dtype=np.float64) for _ in range(m)]
        
        # Track last step-size used (for trace clearing optimization)
        self.last_alpha: List[np.ndarray] = [np.zeros(n, dtype=np.float64) for _ in range(m)]
        
        # Sets of features with non-zero eligibility (for sparse efficiency)
        self.eligible: List[Set[int]] = [set() for _ in range(m)]
        
        # Bookkeeping
        self.v_old = 0.0      # Q(s_{t-1}, a_{t-1}) from previous step
        self.v_delta = 0.0    # Accumulated delta for correction
        
        # Statistics for logging
        self.last_delta = 0.0
        self.last_tau = 0.0
        self.update_count = 0
        
    def get_q_values(self, feature_indices) -> np.ndarray:
        """
        Compute Q-values for all actions: v[j] = Σ_{φ[i]≠0} w^j[i] * φ[i]
        
        For binary features where φ[i]=1, this is just summing weights at active indices.
        
        Args:
            feature_indices: Indices where φ[i] = 1 (list or numpy array)
            
        Returns:
            Q-values array of shape (num_actions,)
        """
        # Convert to numpy array for vectorized indexing
        indices = np.asarray(feature_indices, dtype=np.int32)
        
        q = np.zeros(self.cfg.num_actions, dtype=np.float64)
        for a in range(self.cfg.num_actions):
            q[a] = np.sum(self.w[a][indices])
        return q
    
    def step(
        self,
        feature_indices: List[int],
        reward: float,
        action: int,
        done: bool = False
    ) -> np.ndarray:
        """
        Perform one Swift-Sarsa update step (Algorithm 1).
        
        Args:
            feature_indices: List of indices where φ[i] = 1
            reward: Reward r_t received
            action: Action k chosen at current state
            done: Whether episode terminated
            
        Returns:
            Q-values for all actions at current state
        """
        cfg = self.cfg
        m = cfg.num_actions
        gamma = cfg.gamma
        lambda_ = cfg.lambda_
        theta = cfg.theta
        eta = cfg.eta
        epsilon = cfg.epsilon
        log_eta = np.log(eta)
        log_eta_min = np.log(cfg.eta_min)
        
        # Compute Q-values for all actions: v[i] = Σ w^i[j]φ[j]
        q_values = self.get_q_values(feature_indices)
        
        # TD error (Equation 4): δ' = r + γ*v[k] - v_old
        if done:
            delta_prime = reward - self.v_old
        else:
            delta_prime = reward + gamma * q_values[action] - self.v_old
        
        self.last_delta = delta_prime
        
        # === Update ALL actions with non-zero eligibility traces ===
        for j in range(m):
            # Process eligible features for action j
            to_remove = []
            for i in list(self.eligible[j]):
                if self.z[j][i] == 0:
                    to_remove.append(i)
                    continue
                
                # Weight update: δw^j[i] = δ' * z^j[i] - z_delta^j[i] * v_delta
                self.delta_w[j][i] = delta_prime * self.z[j][i] - self.z_delta[j][i] * self.v_delta
                self.w[j][i] += self.delta_w[j][i]
                
                # Step-size optimization (meta-gradient update)
                # β[i] += θ/e^{β[i]} * (δ' - v_delta) * p[i]
                alpha_j_i = np.exp(self.beta[j][i])
                if alpha_j_i > 1e-300:  # Numerical safety
                    self.beta[j][i] += (theta / alpha_j_i) * (delta_prime - self.v_delta) * self.p[j][i]
                
                # Clip β to [ln(η_min), ln(η)]
                if np.exp(self.beta[j][i]) > eta or np.isinf(np.exp(self.beta[j][i])):
                    self.beta[j][i] = log_eta
                if np.exp(self.beta[j][i]) < cfg.eta_min:
                    self.beta[j][i] = log_eta_min
                
                # Meta-gradient trace updates (h vectors)
                self.h_old[j][i] = self.h[j][i]
                self.h[j][i] = self.h_temp[j][i] + delta_prime * self.z_bar[j][i] - self.z_delta[j][i] * self.v_delta
                self.h_temp[j][i] = self.h[j][i]
                
                # Reset z_delta
                self.z_delta[j][i] = 0
                
                # Decay traces: z, p, z_bar *= γλ
                self.z[j][i] *= gamma * lambda_
                self.p[j][i] *= gamma * lambda_
                self.z_bar[j][i] *= gamma * lambda_
                
                # Remove from eligible set if trace too small
                # Use trace_threshold, NOT epsilon (epsilon is for step-size decay)
                if self.z[j][i] <= cfg.trace_threshold:
                    self.z[j][i] = 0
                    self.p[j][i] = 0
                    self.z_bar[j][i] = 0
                    self.delta_w[j][i] = 0
                    to_remove.append(i)
            
            for i in to_remove:
                self.eligible[j].discard(i)
        
        # Reset v_delta for accumulation
        self.v_delta = 0.0
        
        # === Only increment traces for CHOSEN action k ===
        k = action
        
        # Compute τ = Σ_{φ[i]≠0} e^{β_k[i]} * φ[i]² (for binary features, φ[i]²=1)
        tau = 0.0
        for i in feature_indices:
            tau += np.exp(self.beta[k][i])
        
        self.last_tau = tau
        
        # E = max(η, τ) for the bound
        E = max(eta, tau)
        
        # Compute b = Σ_{φ[i]≠0} z_k[i] * φ[i]
        b = 0.0
        for i in feature_indices:
            b += self.z[k][i]
        
        # Update traces for active features of chosen action
        for i in feature_indices:
            # Add to eligible set if not already
            if self.z[k][i] == 0:
                self.eligible[k].add(i)
            
            # Accumulate v_delta (φ[i]=1 for binary)
            self.v_delta += self.delta_w[k][i]
            
            # η-bound: z_delta_k[i] = (η/E) * e^{β_k[i]} * φ[i]
            self.z_delta[k][i] = (eta / E) * np.exp(self.beta[k][i])
            self.last_alpha[k][i] = self.z_delta[k][i]
            
            # Step-size decay if τ > η
            if (eta / E) < 1:
                self.h_temp[k][i] = 0
                self.h[k][i] = 0
                self.h_old[k][i] = 0
                self.z_bar[k][i] = 0
                self.beta[k][i] += np.log(epsilon)  # φ[i]²=1
            
            # Update eligibility trace: z_k[i] += z_delta_k[i] * (1 - b)
            self.z[k][i] += self.z_delta[k][i] * (1 - b)
            
            # Update meta-gradient traces
            self.p[k][i] += self.h_old[k][i]  # φ[i]=1
            self.z_bar[k][i] += self.z_delta[k][i] * (1 - b - self.z_bar[k][i])  # φ[i]=1
            
            # h_temp update
            self.h_temp[k][i] = (
                self.h[k][i] 
                - self.z_delta[k][i] * self.h[k][i]  # φ[i]=1
                - self.h_old[k][i] * (self.z[k][i] - self.z_delta[k][i])  # φ[i]=1
            )
        
        # Store v_old for next step
        self.v_old = q_values[action]
        self.update_count += 1
        
        # Reset traces on episode end
        if done:
            self._reset_traces()
        
        return q_values
    
    def _reset_traces(self):
        """Reset all traces at episode end."""
        m = self.cfg.num_actions
        
        for j in range(m):
            # Only reset eligible features (sparse)
            for i in self.eligible[j]:
                self.z[j][i] = 0
                self.z_delta[j][i] = 0
                self.z_bar[j][i] = 0
                self.p[j][i] = 0
                self.h[j][i] = 0
                self.h_old[j][i] = 0
                self.h_temp[j][i] = 0
                self.delta_w[j][i] = 0
            self.eligible[j].clear()
        
        self.v_old = 0.0
        self.v_delta = 0.0
    
    def get_step_size_stats(self) -> dict:
        """Get statistics about learned step-sizes for logging."""
        all_alphas = []
        for j in range(self.cfg.num_actions):
            for i in self.eligible[j]:
                all_alphas.append(np.exp(self.beta[j][i]))
        
        if not all_alphas:
            return {"alpha_mean": 0.0, "alpha_max": 0.0, "alpha_min": 0.0}
        
        return {
            "alpha_mean": float(np.mean(all_alphas)),
            "alpha_max": float(np.max(all_alphas)),
            "alpha_min": float(np.min(all_alphas)),
            "num_eligible": len(all_alphas),
        }
    
    def save(self, path: str):
        """Save model weights and step-sizes."""
        state = {
            "config": self.cfg,
            "w": [w.copy() for w in self.w],
            "beta": [b.copy() for b in self.beta],
            "update_count": self.update_count,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    def load(self, path: str):
        """Load model weights and step-sizes."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.w = state["w"]
        self.beta = state["beta"]
        self.update_count = state.get("update_count", 0)


class SwiftSarsaDense:
    """
    Swift-Sarsa with dense (non-sparse) features.
    
    Use this when features are not sparse. Less efficient but works
    with any feature representation.
    """
    
    def __init__(self, config: SwiftSarsaConfig):
        self.cfg = config
        n = config.num_features
        m = config.num_actions
        
        # Weights per action
        self.w = np.zeros((m, n), dtype=np.float64)
        
        # Step-size parameters β
        log_alpha_init = np.log(max(config.alpha_init, 1e-300))
        self.beta = np.full((m, n), log_alpha_init, dtype=np.float64)
        
        # Traces per action
        self.z = np.zeros((m, n), dtype=np.float64)
        self.z_delta = np.zeros((m, n), dtype=np.float64)
        self.z_bar = np.zeros((m, n), dtype=np.float64)
        self.p = np.zeros((m, n), dtype=np.float64)
        self.h = np.zeros((m, n), dtype=np.float64)
        self.h_old = np.zeros((m, n), dtype=np.float64)
        self.h_temp = np.zeros((m, n), dtype=np.float64)
        self.delta_w = np.zeros((m, n), dtype=np.float64)
        
        self.v_old = 0.0
        self.v_delta = 0.0
        self.update_count = 0
        
    def get_q_values(self, features: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions."""
        return self.w @ features
    
    def step(
        self,
        features: np.ndarray,
        reward: float,
        action: int,
        done: bool = False
    ) -> np.ndarray:
        """Perform one Swift-Sarsa update with dense features."""
        cfg = self.cfg
        m = cfg.num_actions
        gamma = cfg.gamma
        lambda_ = cfg.lambda_
        theta = cfg.theta
        eta = cfg.eta
        epsilon = cfg.epsilon
        
        features = np.asarray(features, dtype=np.float64)
        phi_sq = features * features
        
        # Q-values
        q_values = self.get_q_values(features)
        
        # TD error
        if done:
            delta_prime = reward - self.v_old
        else:
            delta_prime = reward + gamma * q_values[action] - self.v_old
        
        # Update all actions
        for j in range(m):
            # Weight updates
            self.delta_w[j] = delta_prime * self.z[j] - self.z_delta[j] * self.v_delta
            self.w[j] += self.delta_w[j]
            
            # Step-size optimization
            alpha = np.exp(self.beta[j])
            alpha = np.maximum(alpha, 1e-300)
            self.beta[j] += (theta / alpha) * (delta_prime - self.v_delta) * self.p[j]
            self.beta[j] = np.clip(self.beta[j], np.log(cfg.eta_min), np.log(eta))
            
            # Meta traces
            self.h_old[j] = self.h[j].copy()
            self.h[j] = self.h_temp[j] + delta_prime * self.z_bar[j] - self.z_delta[j] * self.v_delta
            self.h_temp[j] = self.h[j].copy()
            
            self.z_delta[j].fill(0)
            self.z[j] *= gamma * lambda_
            self.p[j] *= gamma * lambda_
            self.z_bar[j] *= gamma * lambda_
        
        self.v_delta = 0.0
        
        # Increment traces for chosen action only
        k = action
        
        # τ = Σ e^{β_k[i]} * φ[i]²
        tau = np.sum(np.exp(self.beta[k]) * phi_sq)
        E = max(eta, tau)
        
        # b = Σ z_k[i] * φ[i]
        b = np.dot(self.z[k], features)
        
        # v_delta accumulation
        self.v_delta = np.dot(self.delta_w[k], features)
        
        # η-bound
        self.z_delta[k] = (eta / E) * np.exp(self.beta[k]) * features
        
        # Step-size decay if τ > η
        if tau > eta:
            self.beta[k] += np.log(epsilon) * phi_sq
            self.h_temp[k].fill(0)
            self.h[k].fill(0)
            self.z_bar[k].fill(0)
        
        # Update traces
        self.z[k] += self.z_delta[k] * (1 - b)
        self.p[k] += self.h_old[k] * features
        self.z_bar[k] += self.z_delta[k] * (1 - b - self.z_bar[k] * features)
        
        self.h_temp[k] = (
            self.h[k]
            - self.z_delta[k] * features * self.h[k]
            - self.h_old[k] * features * (self.z[k] - self.z_delta[k])
        )
        
        self.v_old = q_values[action]
        self.update_count += 1
        
        if done:
            self._reset_traces()
        
        return q_values
    
    def _reset_traces(self):
        """Reset traces at episode end."""
        self.z.fill(0)
        self.z_delta.fill(0)
        self.z_bar.fill(0)
        self.p.fill(0)
        self.h.fill(0)
        self.h_old.fill(0)
        self.h_temp.fill(0)
        self.delta_w.fill(0)
        self.v_old = 0.0
        self.v_delta = 0.0
    
    def save(self, path: str):
        """Save model."""
        state = {
            "config": self.cfg,
            "w": self.w.copy(),
            "beta": self.beta.copy(),
            "update_count": self.update_count,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    def load(self, path: str):
        """Load model."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.w = state["w"]
        self.beta = state["beta"]
        self.update_count = state.get("update_count", 0)


class AtariBinaryFeatureExtractor:
    """
    Convert Atari frames to sparse binary features.
    
    Based on SwiftTD paper Section 4.1:
    1. Resize 210×160×3 → 105×80×3
    2. Bin each pixel value into 8 bins (lossy one-hot)
    3. Result: 105×80×3×8 = 201,600 binary features
    4. Add previous action one-hot (18 max)
    
    Total: ~201,618 features (can vary based on action space)
    
    Uses vectorized NumPy operations for speed.
    """
    
    def __init__(
        self,
        height: int = 105,
        width: int = 80,
        num_bins: int = 8,
        num_actions: int = 18,
        include_prev_action: bool = True,
    ):
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.num_actions = num_actions
        self.include_prev_action = include_prev_action
        
        # Feature dimensions
        self.pixel_features = height * width * 3 * num_bins  # 201,600 for default
        if include_prev_action:
            self.total_features = self.pixel_features + num_actions
        else:
            self.total_features = self.pixel_features
        
        self.prev_action = 0
        
        # Pre-compute index offsets for vectorized extraction (FAST)
        # Shape: (height, width, 3) containing base indices
        y_indices = np.arange(height).reshape(-1, 1, 1)
        x_indices = np.arange(width).reshape(1, -1, 1)
        c_indices = np.arange(3).reshape(1, 1, -1)
        
        # Base index = y * (width * 3 * num_bins) + x * (3 * num_bins) + c * num_bins
        self._base_indices = (
            y_indices * (width * 3 * num_bins) +
            x_indices * (3 * num_bins) +
            c_indices * num_bins
        ).astype(np.int32)
        
    def extract(self, frame: np.ndarray) -> List[int]:
        """
        Extract sparse binary feature indices from frame (VECTORIZED).
        
        Args:
            frame: Grayscale or RGB frame. If grayscale (H, W), treats as single channel.
                   If RGB (H, W, 3), uses all channels.
                   
        Returns:
            List of indices where feature = 1
        """
        # Handle grayscale input
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        
        # Resize if needed
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            import cv2
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        # Ensure 3 channels
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        elif frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        
        # Bin pixel values: 0-255 → 0-7 (8 bins) - VECTORIZED
        binned = np.minimum((frame.astype(np.int32) * self.num_bins) >> 8, self.num_bins - 1)
        
        # Compute feature indices - VECTORIZED
        # feature_idx = base_index + bin_value
        feature_indices = (self._base_indices + binned).ravel()
        
        # Add previous action one-hot
        if self.include_prev_action:
            action_idx = self.pixel_features + self.prev_action
            feature_indices = np.append(feature_indices, action_idx)
        
        return feature_indices.tolist()
    
    def extract_numpy(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features returning numpy array (even faster for batch ops).
        """
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            import cv2
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        elif frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        
        binned = np.minimum((frame.astype(np.int32) * self.num_bins) >> 8, self.num_bins - 1)
        feature_indices = (self._base_indices + binned).ravel()
        
        if self.include_prev_action:
            action_idx = self.pixel_features + self.prev_action
            feature_indices = np.append(feature_indices, action_idx)
        
        return feature_indices
    
    def set_prev_action(self, action: int):
        """Update previous action for next extraction."""
        self.prev_action = action
