#!/usr/bin/env python3
"""
Proper Swift-Sarsa Implementation for Atari Pong.
Matches the SwiftTD Paper's exact feature construction:
- 105x80x3 RGB Input (Binned 8) -> Dense Representation (~25k active features)
- No Frame Differencing
- Includes Cumulant (Previous Reward) in the Observation Vector (One-Hot Encoded)
- Physics-Compliant Initialization (alpha=1e-5) to avoid C++ Kill Switch
"""

import argparse
import numpy as np
import gymnasium as gym
import cv2
import os
import random

from swiftsarsa import SwiftSarsaBinaryFeatures
from shimmy.registration import register_gymnasium_envs

# ---------------------------------------------------------------------------
# 1. Exact Preprocessing (SwiftTD Paper)
# ---------------------------------------------------------------------------
class SwiftTDPaperPreprocessor:
    """
    Implements the preprocessing described in the SwiftTD paper.
    Observation Vector Components:
    1. Image: 105x80x3, 8 bins per pixel (One-hot).
    2. Context: Previous Action (One-hot).
    3. Cumulant: Previous Reward (One-hot: -1, 0, 1).
    """
    def __init__(self, num_actions):
        self.height = 105
        self.width = 80
        self.channels = 3
        self.bins = 8
        self.num_actions = num_actions
        
        # 1. Pixel Features Dimensions
        # Structure: [Channel 0][Channel 1][Channel 2]...
        self.pixels_per_channel = self.height * self.width
        self.features_per_channel = self.pixels_per_channel * self.bins
        self.total_pixel_features = self.features_per_channel * self.channels
        
        # 2. Context Features Offsets
        self.action_offset = self.total_pixel_features
        self.reward_offset = self.action_offset + self.num_actions
        
        # Total Dimension
        self.total_features = self.reward_offset + 3
        
        print(f"Feature Vector Dimension: {self.total_features}")
        print(f"  - Pixels: {self.total_pixel_features}")
        print(f"  - Action: {self.num_actions}")
        print(f"  - Reward: 3")

    def reset(self):
        pass

    def extract(self, frame, prev_action, prev_reward):
        """
        Constructs the binary feature vector indices.
        """
        active_indices = []

        # --- A. Image Features (No Frame Diff) ---
        # 1. Resize to 105x80
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        # 2. Binning (0-255 -> 0-7)
        binned = frame // 32 
        
        # 3. Calculate Indices
        stride_pixel = self.bins
        stride_channel = self.features_per_channel

        for c in range(self.channels):
            # Flatten channel to 1D
            channel_bins = binned[:, :, c].flatten()
            
            # Base offsets for each pixel in this channel
            pixel_offsets = np.arange(self.pixels_per_channel) * stride_pixel
            
            # Channel offset
            channel_start = c * stride_channel
            
            # Final Index = Channel_Start + Pixel_Base + Bin_Value
            indices = channel_start + pixel_offsets + channel_bins
            active_indices.extend(indices)
            
        # --- B. Previous Action (One-Hot) ---
        if 0 <= prev_action < self.num_actions:
            active_indices.append(self.action_offset + prev_action)
            
        # --- C. Cumulant / Previous Reward (One-Hot) ---
        # Map reward to index: -1 -> 0, 0 -> 1, 1 -> 2
        r_clipped = int(np.clip(prev_reward, -1, 1))
        reward_idx = r_clipped + 1  # [-1,0,1] -> [0,1,2]
        active_indices.append(self.reward_offset + reward_idx)
        
        return active_indices

# ---------------------------------------------------------------------------
# 2. Training Loop
# ---------------------------------------------------------------------------
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", type=str, default="Pong")
    parser.add_argument("--decisions", type=int, default=1_000_000)
    
    # --- PHYSICS PARAMETERS ---
    # Active Features per step: ~25,203
    # Safety Limit (Eta): 1.0
    #
    # Constraint: Total Energy (Active * Alpha) MUST BE < Eta
    parser.add_argument("--alpha", type=float, default=1e-5)
    
    # Meta-step
    parser.add_argument("--meta_step", type=float, default=1e-2)
    
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--lambda_val", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay_steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--betas_dir", type=str, default="runs/pong_paper")
    parser.add_argument("--heatmap_inverse_beta", action="store_true", default=True,
                        help="Use exp(-beta) for heatmaps (safer magnitude) instead of exp(beta) (default: inverse on)")
    parser.add_argument("--heatmap_raw_beta", dest="heatmap_inverse_beta", action="store_false",
                        help="Use exp(beta) for heatmaps (turn off inverse view)")
    parser.add_argument("--video_dir", type=str, default=None, help="Optional directory to record gameplay MP4s")
    args = parser.parse_args()

    # Environment
    try: register_gymnasium_envs()
    except: pass
    render_mode = "rgb_array" if args.video_dir else None
    env = gym.make(f"ALE/{args.rom}-v5", obs_type="rgb", full_action_space=True, render_mode=render_mode)
    
    # Full action space
    action_map = None
    num_actions = env.action_space.n

    np.random.seed(args.seed)
    
    extractor = SwiftTDPaperPreprocessor(num_actions)
    
    learner = SwiftSarsaBinaryFeatures(
        extractor.total_features,
        num_actions,
        args.lambda_val,
        args.alpha,
        args.meta_step,
        args.eta,
        0.9999,      # Decay
        1e-6,       # Trace Epsilon
        3e-7        # Eta Min
    )

    print(f"--- Configuration ---")
    print(f"Active Features: ~25,200")
    print(f"Initial Energy:  {25200 * args.alpha:.4f}")
    if (25200 * args.alpha) >= args.eta:
        print("WARNING: Unsafe initialization! C++ Kill Switch will trigger.")
    else:
        print("STATUS: Safe. Meta-learning enabled.")
    print(f"---------------------")

    step = 0
    episode_idx = 0
    
    os.makedirs(args.betas_dir, exist_ok=True)
    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)

    while step < args.decisions:
        frame, _ = env.reset(seed=args.seed + episode_idx)
        extractor.reset()
        learner.reset_episode()

        writer = None

        def close_writer():
            nonlocal writer
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
                writer = None
        
        # --- PRIME THE LEARNER ---
        # Initial State: S0, PrevAction=0, PrevReward=0
        features = extractor.extract(frame, 0, 0.0)
        
        # Prime v_old with Q(S0)
        # We pass reward=0, gamma=0 just to set the internal state
        learner.learn(features, 0.0, 0.0, 0)
        
        eps = current_epsilon(step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
        q_vals = learner.get_action_values(features)
        action = select_action(q_vals, eps)
        
        episode_reward = 0
        done = False
        
        # Record first frame if recording
        if args.video_dir:
            try:
                import imageio
                if writer is None:
                    path = os.path.join(args.video_dir, f"episode_{episode_idx:05d}.mp4")
                    writer = imageio.get_writer(path, fps=60)
                frame_rgb = env.render()
                if frame_rgb is not None:
                    writer.append_data(frame_rgb)
            except Exception:
                pass

        while not done and step < args.decisions:
            # 1. Act
            ale_action = action if action_map is None else action_map[action]
            next_frame, r, term, trunc, _ = env.step(ale_action)
            done = term or trunc
            reward = np.clip(r, -1, 1)
            episode_reward += r
            
            # 2. Next State (S')
            # Note: We pass 'action' (A) and 'reward' (R) as features for S'
            # This aligns with the "Cumulant is part of observation" logic
            features_next = extractor.extract(next_frame, action, reward)
            
            # 3. Next Action (A')
            eps = current_epsilon(step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
            q_vals_next = learner.get_action_values(features_next)
            action_next = select_action(q_vals_next, eps)
            
            # 4. Learn
            # SwiftSarsa C++: delta = r + gamma * Q(S') - v_old
            g = 0.0 if term else 0.99
            learner.learn(features_next, reward, g, action_next)
            
            # 5. Advance
            action = action_next
            step += 1

            # Record frame
            if args.video_dir and writer is not None:
                try:
                    frame_rgb = env.render()
                    if frame_rgb is not None:
                        writer.append_data(frame_rgb)
                except Exception:
                    pass
            
            # Visualization
            if step % 200 == 0:
                try:
                    import imageio
                    betas = np.array(learner.get_feature_betas_max_over_actions())
                    # Getter returns |beta|; optionally invert to avoid blowout.
                    if args.heatmap_inverse_beta:
                        alphas = np.exp(-betas)
                    else:
                        alphas = np.exp(betas)
                    
                    # Extract only pixel features for visualization
                    pixel_alphas = alphas[:extractor.total_pixel_features]
                    pixel_alphas = pixel_alphas.reshape(3, 105, 80, 8)
                    
                    # Max projection over channels and bins
                    heatmap = np.max(pixel_alphas, axis=(0, 3))
                    
                    # Normalize
                    h_min, h_max = heatmap.min(), heatmap.max()
                    span = h_max - h_min
                    if span > 1e-12:
                        norm_img = (heatmap - h_min) / span
                    else:
                        # Constant map fallback (all zeros)
                        norm_img = np.zeros_like(heatmap)
                    imageio.imwrite(f"{args.betas_dir}/step_{step:07d}.png", (norm_img * 255).astype(np.uint8))
                except Exception:
                    pass

        episode_idx += 1
        print(f"Ep {episode_idx}: Reward {episode_reward} (Step {step})")
        close_writer()

if __name__ == "__main__":
    main()
