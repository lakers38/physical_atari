# Copyright 2025
# Swift-Sarsa: Fast and Robust Linear Control - OPTIMIZED VERSION
#
# This is a NumPy-vectorized version for better performance.
# Uses array operations instead of Python loops where possible.

"""
Swift-Sarsa optimized with NumPy vectorization.
Significantly faster than the loop-based implementation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import pickle


@dataclass
class SwiftSarsaConfig:
    """Configuration for Swift-Sarsa algorithm."""
    num_features: int
    num_actions: int
    gamma: float = 0.99
    lambda_: float = 0.95
    alpha_init: float = 1e-7
    theta: float = 1e-3
    eta: float = 0.1
    epsilon: float = 0.999  # Step-size decay factor (when tau > eta)
    eta_min: float = 1e-15
    trace_threshold: float = 1e-10  # Threshold for clearing small traces (separate from epsilon!)


class SwiftSarsaBinaryFast:
    """
    Swift-Sarsa with sparse binary features - OPTIMIZED.
    
    Key optimizations:
    - Vectorized numpy operations on eligible feature arrays
    - Reduced Python loop overhead
    - Efficient sparse index handling
    """
    
    def __init__(self, config: SwiftSarsaConfig):
        self.cfg = config
        n = config.num_features
        m = config.num_actions
        
        # Weights per action
        self.w = np.zeros((m, n), dtype=np.float64)
        
        # Step-size parameters β (α = e^β)
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
        self.last_alpha = np.zeros((m, n), dtype=np.float64)
        
        # Track which features have non-zero traces (for sparse efficiency)
        # Use a single array of all eligible indices across all actions
        self.eligible_mask = np.zeros((m, n), dtype=bool)
        
        # Bookkeeping
        self.v_old = 0.0
        self.v_delta = 0.0
        self.last_delta = 0.0
        self.last_tau = 0.0
        self.update_count = 0
        
        # Pre-compute constants
        self.log_eta = np.log(config.eta)
        self.log_eta_min = np.log(config.eta_min)
        self.gamma_lambda = config.gamma * config.lambda_
        self.log_epsilon = np.log(config.epsilon)
        self.trace_threshold = config.trace_threshold  # For clearing small traces
    
    def get_q_values(self, feature_indices: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions (vectorized)."""
        # Sum weights at active indices for each action
        return self.w[:, feature_indices].sum(axis=1)
    
    def step(
        self,
        feature_indices: np.ndarray,
        reward: float,
        action: int,
        done: bool = False
    ) -> np.ndarray:
        """
        Perform one Swift-Sarsa update step (vectorized).
        """
        cfg = self.cfg
        m = cfg.num_actions
        gamma = cfg.gamma
        theta = cfg.theta
        eta = cfg.eta
        
        # Ensure numpy array
        feature_indices = np.asarray(feature_indices, dtype=np.int32)
        
        # Compute Q-values
        q_values = self.get_q_values(feature_indices)
        
        # TD error
        if done:
            delta_prime = reward - self.v_old
        else:
            delta_prime = reward + gamma * q_values[action] - self.v_old
        
        self.last_delta = delta_prime
        
        # === Phase 1: Update ALL actions with non-zero eligibility traces ===
        # Find all eligible features across all actions
        eligible_any = self.eligible_mask.any(axis=0)
        if eligible_any.any():
            eligible_indices = np.where(eligible_any)[0]
            
            for j in range(m):
                # Get eligible indices for this action
                j_mask = self.eligible_mask[j, eligible_indices]
                if not j_mask.any():
                    continue
                
                j_indices = eligible_indices[j_mask]
                
                # Vectorized weight update
                self.delta_w[j, j_indices] = (
                    delta_prime * self.z[j, j_indices] 
                    - self.z_delta[j, j_indices] * self.v_delta
                )
                self.w[j, j_indices] += self.delta_w[j, j_indices]
                
                # Step-size optimization (vectorized)
                alpha_j = np.exp(self.beta[j, j_indices])
                valid = alpha_j > 1e-300
                if valid.any():
                    valid_idx = j_indices[valid]
                    self.beta[j, valid_idx] += (
                        (theta / alpha_j[valid]) 
                        * (delta_prime - self.v_delta) 
                        * self.p[j, valid_idx]
                    )
                
                # Clip β (vectorized)
                self.beta[j, j_indices] = np.clip(
                    self.beta[j, j_indices], 
                    self.log_eta_min, 
                    self.log_eta
                )
                
                # Meta-gradient trace updates (vectorized)
                self.h_old[j, j_indices] = self.h[j, j_indices]
                self.h[j, j_indices] = (
                    self.h_temp[j, j_indices] 
                    + delta_prime * self.z_bar[j, j_indices] 
                    - self.z_delta[j, j_indices] * self.v_delta
                )
                self.h_temp[j, j_indices] = self.h[j, j_indices]
                
                # Reset z_delta
                self.z_delta[j, j_indices] = 0
                
                # Decay traces (vectorized)
                self.z[j, j_indices] *= self.gamma_lambda
                self.p[j, j_indices] *= self.gamma_lambda
                self.z_bar[j, j_indices] *= self.gamma_lambda
                
                # Clear small traces (use trace_threshold, NOT epsilon!)
                # epsilon is for step-size decay, trace_threshold is for clearing negligible traces
                small = self.z[j, j_indices] <= self.trace_threshold
                if small.any():
                    clear_idx = j_indices[small]
                    self.z[j, clear_idx] = 0
                    self.p[j, clear_idx] = 0
                    self.z_bar[j, clear_idx] = 0
                    self.delta_w[j, clear_idx] = 0
                    self.eligible_mask[j, clear_idx] = False
        
        # Reset v_delta
        self.v_delta = 0.0
        
        # === Phase 2: Only increment traces for CHOSEN action k ===
        k = action
        
        # Compute τ (vectorized)
        tau = np.exp(self.beta[k, feature_indices]).sum()
        self.last_tau = tau
        
        E = max(eta, tau)
        
        # Compute b (vectorized)
        b = self.z[k, feature_indices].sum()
        
        # Accumulate v_delta (vectorized)
        self.v_delta = self.delta_w[k, feature_indices].sum()
        
        # η-bound multiplier
        bound_multiplier = eta / E
        
        # Update traces for active features (vectorized)
        alpha_k = np.exp(self.beta[k, feature_indices])
        self.z_delta[k, feature_indices] = bound_multiplier * alpha_k
        self.last_alpha[k, feature_indices] = self.z_delta[k, feature_indices]
        
        # Step-size decay if τ > η
        if tau > eta:
            self.beta[k, feature_indices] += self.log_epsilon
            self.h_temp[k, feature_indices] = 0
            self.h[k, feature_indices] = 0
            self.z_bar[k, feature_indices] = 0
        
        # Update eligibility traces (vectorized)
        self.z[k, feature_indices] += self.z_delta[k, feature_indices] * (1 - b)
        self.p[k, feature_indices] += self.h_old[k, feature_indices]
        self.z_bar[k, feature_indices] += self.z_delta[k, feature_indices] * (
            1 - b - self.z_bar[k, feature_indices]
        )
        
        # h_temp update (vectorized)
        self.h_temp[k, feature_indices] = (
            self.h[k, feature_indices]
            - self.z_delta[k, feature_indices] * self.h[k, feature_indices]
            - self.h_old[k, feature_indices] * (
                self.z[k, feature_indices] - self.z_delta[k, feature_indices]
            )
        )
        
        # Mark as eligible
        self.eligible_mask[k, feature_indices] = True
        
        # Store v_old
        self.v_old = q_values[action]
        self.update_count += 1
        
        # Reset on episode end
        if done:
            self._reset_traces()
        
        return q_values
    
    def _reset_traces(self):
        """Reset all traces at episode end."""
        # Only reset eligible features (sparse optimization)
        if self.eligible_mask.any():
            self.z[self.eligible_mask] = 0
            self.z_delta[self.eligible_mask] = 0
            self.z_bar[self.eligible_mask] = 0
            self.p[self.eligible_mask] = 0
            self.h[self.eligible_mask] = 0
            self.h_old[self.eligible_mask] = 0
            self.h_temp[self.eligible_mask] = 0
            self.delta_w[self.eligible_mask] = 0
            self.eligible_mask.fill(False)
        
        self.v_old = 0.0
        self.v_delta = 0.0
    
    def save(self, path: str):
        """Save model weights and step-sizes."""
        state = {
            "config": self.cfg,
            "w": self.w.copy(),
            "beta": self.beta.copy(),
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

