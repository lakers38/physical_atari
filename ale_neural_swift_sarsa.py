#!/usr/bin/env python3
"""
Neural Swift-SARSA Implementation for ALE.
Uses a CNN to extract dense features, normalizes them, and feeds them
into the dense Swift-SARSA C++ backend.
"""

import argparse
import numpy as np
import gymnasium as gym
import cv2
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from swiftsarsa import SwiftSarsa as SwiftSarsaDense
from shimmy.registration import register_gymnasium_envs

# ------------------------------------------------------------------------------
# 1. Neural Network Body
# ------------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    Single conv layer on 105x80x24 binned input (stride=2, kernel=3x3).
    Outputs feature maps (25, 52, 40) flattened to ~52k features.
    """
    def __init__(self, input_channels, num_kernels=25):
        super().__init__()
        # Padding (0,1) to match ~52x40 output (text: 52x40x25)
        self.conv = nn.Conv2d(input_channels, num_kernels, kernel_size=3, stride=2, padding=(0, 1))
        with torch.no_grad():
            nn.init.uniform_(self.conv.weight, -1.0, 1.0)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = F.relu(self.conv(x))
        return x.reshape(x.size(0), -1)

# ------------------------------------------------------------------------------
# 2. Feature Normalizer
# ------------------------------------------------------------------------------

class FeatureNormalizer:
    def __init__(self, num_features, alpha=0.001):
        self.num_features = num_features
        self.alpha = alpha
        self.mean = np.zeros(num_features, dtype=np.float32)
        self.var = np.ones(num_features, dtype=np.float32)
        self.count = 0

    def normalize(self, features):
        """
        Update stats and normalize features.
        features: numpy array (num_features,)
        """
        # Online update of mean and variance (exponential moving average for non-stationary)
        # Or standard Welford's if we assume stationarity, but here features shift as CNN learns.
        # Let's use simple EMA for tracking.
        
        if self.count == 0:
            self.mean = features.copy()
            self.var = np.zeros_like(features) + 1.0
        else:
            delta = features - self.mean
            self.mean += self.alpha * delta
            # var_new = (1-alpha)*var + alpha * delta^2
            # Actually for standardization we want E[x^2] - E[x]^2 or similar.
            # Let's stick to a simple EMA of variance.
            self.var = (1 - self.alpha) * self.var + self.alpha * (delta ** 2)
            
        self.count += 1
        
        std = np.sqrt(self.var + 1e-6)
        return (features - self.mean) / std

# ------------------------------------------------------------------------------
# 3. Preprocessor
# ------------------------------------------------------------------------------

class SwiftNeuralPreprocessor:
    def __init__(self, num_actions, device="cpu"):
        self.height = 105
        self.width = 80
        self.channels = 3
        self.bins = 8
        self.num_actions = num_actions
        self.device = device
        self.bin_eye = torch.eye(self.bins, device=self.device, dtype=torch.float32)
        
        # 3 channels * 8 bins = 24 input channels for CNN
        self.input_channels = self.channels * self.bins

    def reset(self):
        pass

    def extract(self, frame, prev_action, prev_reward):
        """
        Returns:
            image_tensor: (1, 24, 105, 80) torch tensor
            extra_vec: (num_actions + 3,) numpy array
        """
        # 1. Image processing
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        binned = torch.from_numpy(frame // 32).to(self.device, non_blocking=True)  # (H, W, 3)
        one_hot = self.bin_eye[binned.long()]  # (H, W, 3, 8)
        image_tensor = one_hot.permute(2, 0, 1, 3).reshape(
            self.channels * self.bins, self.height, self.width
        ).unsqueeze(0)  # (1, 24, 105, 80)

        # 2. Extra features (prev action, reward)
        extra_vec = np.zeros(self.num_actions + 3, dtype=np.float32)
        
        if 0 <= prev_action < self.num_actions:
            extra_vec[prev_action] = 1.0
            
        r_clipped = int(np.clip(prev_reward, -1, 1))
        reward_idx = r_clipped + 1 # 0, 1, 2
        extra_vec[self.num_actions + reward_idx] = 1.0
        
        return image_tensor, extra_vec

# ------------------------------------------------------------------------------
# 4. Main Agent Loop
# ------------------------------------------------------------------------------

def select_action(q_values, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(len(q_values))
    return int(np.argmax(q_values))

def current_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    if step < decay_steps:
        return start - (step / decay_steps) * (start - end)
    return end

def resolve_action_set(env, action_set_choice: str):
    try:
        action_meanings = env.unwrapped.get_action_meanings()
    except Exception:
        action_meanings = []

    if action_set_choice == "full":
        desc = "Action set: full ALE action space"
        if action_meanings:
            desc += f" ({len(action_meanings)} actions)"
        return None, env.action_space.n, desc, action_meanings

    try:
        minimal_actions = list(env.unwrapped.ale.getMinimalActionSet())
        num_actions = len(minimal_actions)
        minimal_meanings = []
        if action_meanings:
            minimal_meanings = [
                action_meanings[a] if a < len(action_meanings) else str(a) for a in minimal_actions
            ]
        desc = f"Action set: ALE minimal ({num_actions} actions)"
        if minimal_meanings:
            desc += f" meanings={minimal_meanings}"
        return minimal_actions, num_actions, desc, action_meanings
    except Exception:
        fallback_desc = "Action set: env action space (minimal set unavailable)"
        if action_meanings:
            fallback_desc += f" ({len(action_meanings)} actions)"
        return None, env.action_space.n, fallback_desc, action_meanings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", type=str, default="Pong")
    parser.add_argument("--decisions", type=int, default=1_000_000)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--meta_step", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=0.03)
    parser.add_argument("--lambda_val", type=float, default=0.99)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay_steps", type=int, default=100000)
    parser.add_argument("--swift_decay", type=float, default=0.999)
    parser.add_argument("--swift_trace_epsilon", type=float, default=1e-6)
    parser.add_argument("--swift_eta_min", type=float, default=3e-7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--cnn_lr", type=float, default=1e-4)
    parser.add_argument("--freeze_cnn", action="store_true", help="Freeze CNN weights (random features)")
    parser.add_argument("--no_cnn", action="store_true", help="Bypass CNN and use flattened binned features directly.")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--action_set", choices=["minimal", "full"], default="minimal")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--debug_print_every", type=int, default=0,
                        help="If >0, print debug stats every N env steps.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)

    def _fmt_val(v):
        return str(v).replace(".", "p")

    if args.log_file:
        log_filename = args.log_file
    else:
        log_filename = (
            f"neural_{args.rom}_seed{args.seed}"
            f"_dec{args.decisions}"
            f"_alpha{_fmt_val(args.alpha)}"
            f"_ts{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            ".txt"
        )
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, log_filename)
    log_file = open(log_path, "w", buffering=1)

    def log(msg: str):
        print(msg)
        log_file.write(msg + "\n")

    try:
        register_gymnasium_envs()
    except Exception:
        pass
        
    env = gym.make(f"ALE/{args.rom}-v5", obs_type="rgb", full_action_space=True)
    action_map, num_actions, action_desc, action_meanings = resolve_action_set(env, args.action_set)
    log(action_desc)

    # Initialize components
    preprocessor = SwiftNeuralPreprocessor(num_actions, device=args.device)
    
    use_conv = not args.no_cnn
    if use_conv:
        # CNN Output + Extra Features (25 kernels, stride-2, 3x3 -> 52x40 maps)
        conv_feature_dim = 25 * 52 * 40
        cnn = SimpleCNN(preprocessor.input_channels).to(device)
        optimizer = torch.optim.Adam(cnn.parameters(), lr=args.cnn_lr) if not args.freeze_cnn else None
    else:
        # Flattened binned one-hot features: 24 * 105 * 80
        conv_feature_dim = preprocessor.input_channels * preprocessor.height * preprocessor.width
        cnn = None
        optimizer = None
    total_features = conv_feature_dim + num_actions + 3
    
    normalizer = FeatureNormalizer(total_features)
    
    learner = SwiftSarsaDense(
        total_features,
        num_actions,
        args.lambda_val,
        args.alpha,
        args.meta_step,
        args.eta,
        args.swift_decay,
        args.swift_trace_epsilon,
        args.swift_eta_min
    )

    log(f"Logging to {log_path}")
    log(f"--- Configuration ---")
    log(f"ROM: {args.rom}")
    log(f"Total Features: {total_features} (conv={conv_feature_dim}, extra={num_actions + 3}, use_conv={use_conv})")
    log(f"Initial Energy:  {total_features * args.alpha:.4f}")
    if (total_features * args.alpha) >= args.eta:
        log("WARNING: Unsafe initialization! C++ Kill Switch will trigger.")
    else:
        log("STATUS: Safe. Meta-learning enabled.")
    log(f"---------------------")

    step = 0
    episode_idx = 0
    returns = []

    while step < args.decisions:
        frame, _ = env.reset(seed=args.seed + episode_idx)
        preprocessor.reset()
        learner.reset_episode()

        # ---- Initial state (s0) ----
        img_t, extra = preprocessor.extract(frame, 0, 0.0)

        if use_conv:
            if not args.freeze_cnn:
                img_t = img_t.to(device)
                img_t.requires_grad_(True)
                phi_img_var = cnn(img_t)  # (1, F)
                phi_img = phi_img_var.detach().cpu().numpy().flatten()
            else:
                with torch.no_grad():
                    phi_img_var = cnn(img_t.to(device))
                    phi_img = phi_img_var.cpu().numpy().flatten()
        else:
            phi_img_var = None
            phi_img = img_t.cpu().numpy().reshape(-1)

        phi_full = np.concatenate([phi_img, extra])
        phi_norm = normalizer.normalize(phi_full)
        mask = np.abs(phi_norm) > 1e-6
        idx = np.flatnonzero(mask)
        vals = phi_norm[mask]
        feature_indices = list(zip(idx, vals))
        if args.debug_print_every and (step % args.debug_print_every == 0):
            log(f"[debug] step={step} feat_mean={phi_norm.mean():.4f} feat_std={phi_norm.std():.4f} "
                f"raw_mean={phi_img.mean():.4f} raw_std={phi_img.std():.4f} "
                f"num_active={len(feature_indices)}")

        eps = current_epsilon(step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
        q_vals = learner.get_action_values(feature_indices)
        action = select_action(q_vals, eps)

        # Seed v_old without a real transition (sets v_old = Q(s0, a0) in C++)
        learner.learn(feature_indices, 0.0, 0.0, action)

        episode_reward = 0
        done = False

        prev_phi_img_var = phi_img_var
        prev_phi_norm = phi_norm
        prev_action = action
        prev_feature_indices = feature_indices

        while not done and step < args.decisions:
            # ---- Env step with frame skipping ----
            ale_action = action if action_map is None else action_map[action]
            total_clipped_r = 0.0
            frames_used = 0

            for _ in range(max(1, args.frame_skip)):
                next_frame, r, term, trunc, _ = env.step(ale_action)
                frames_used += 1
                done = term or trunc
                total_clipped_r += np.clip(r, -1, 1)
                episode_reward += r
                if done:
                    break

            # ---- Next state features (s_{t+1}) ----
            img_t_next, extra_next = preprocessor.extract(next_frame, action, total_clipped_r)

            if use_conv:
                if not args.freeze_cnn:
                    with torch.no_grad():
                        phi_img_next_var = cnn(img_t_next.to(device))
                        phi_img_next = phi_img_next_var.cpu().numpy().flatten()
                else:
                    with torch.no_grad():
                        phi_img_next_var = cnn(img_t_next.to(device))
                        phi_img_next = phi_img_next_var.cpu().numpy().flatten()
            else:
                phi_img_next_var = None
                phi_img_next = img_t_next.cpu().numpy().reshape(-1)

            phi_full_next = np.concatenate([phi_img_next, extra_next])
            phi_norm_next = normalizer.normalize(phi_full_next)
            mask_next = np.abs(phi_norm_next) > 1e-6
            idx_next = np.flatnonzero(mask_next)
            vals_next = phi_norm_next[mask_next]
            feature_indices_next = list(zip(idx_next, vals_next))

            eps = current_epsilon(step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
            q_vals_next = learner.get_action_values(feature_indices_next)
            action_next = select_action(q_vals_next, eps)

            gamma = 0.0 if done else args.discount

            # ---- CNN semi-gradient update ----
            if use_conv and not args.freeze_cnn:
                all_weights = np.array(learner.get_weights(), dtype=np.float32)
                w_start = prev_action * total_features
                w_end = w_start + total_features
                w_a = all_weights[w_start:w_end]

                q_s = float(np.dot(w_a, prev_phi_norm))
                q_next = float(q_vals_next[action_next])
                delta = total_clipped_r + gamma * q_next - q_s

                w_a_cnn = w_a[:conv_feature_dim]
                std_cnn = np.sqrt(normalizer.var[:conv_feature_dim] + 1e-6)

                grad_phi_norm = -delta * w_a_cnn
                grad_phi_img = grad_phi_norm / std_cnn

                if optimizer is not None:
                    optimizer.zero_grad()
                    grad_tensor = torch.from_numpy(grad_phi_img).unsqueeze(0).to(device)
                    prev_phi_img_var.backward(grad_tensor)
                    optimizer.step()
                    if args.debug_print_every and (step % args.debug_print_every == 0):
                        log(f"[debug] step={step} delta={float(delta):.4f} q_s={float(q_s):.4f} "
                            f"q_next={float(q_next):.4f} active_prev={len(prev_feature_indices)}")

            # ---- C++ learner update ----
            learner.learn(prev_feature_indices, total_clipped_r, gamma, prev_action)
            if args.debug_print_every and (step % args.debug_print_every == 0):
                w_dbg = np.array(learner.get_weights(), dtype=np.float32)
                log(f"[debug] step={step} weight_mean={w_dbg.mean():.6f} weight_std={w_dbg.std():.6f}")

            # ---- Shift for next step ----
            if use_conv and not args.freeze_cnn:
                img_for_grad = img_t_next.to(device)
                img_for_grad.requires_grad_(True)
                prev_phi_img_var = cnn(img_for_grad)
            else:
                prev_phi_img_var = None
            prev_phi_norm = phi_norm_next
            prev_action = action_next
            prev_feature_indices = feature_indices_next
            action = action_next
            step += frames_used

            if step >= args.decisions:
                break

        returns.append(episode_reward)
        mean_100 = float(np.mean(returns[-100:])) if returns else 0.0
        log(f"Ep {episode_idx}: reward={episode_reward:.1f} mean_last_100={mean_100:.2f} step={step}")
        episode_idx += 1

    env.close()
    log_file.close()

if __name__ == "__main__":
    main()
