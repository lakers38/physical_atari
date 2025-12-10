#!/usr/bin/env python3
"""
Generic Swift-Sarsa Implementation for any ALE game using the SwiftTD paper
feature construction (no frame differencing, 105x80x3 RGB binned to 8 levels).
Switch ROMs via --rom without changing the code; action set can be minimal or
full.
"""

import argparse
import numpy as np
import gymnasium as gym
import cv2
import os
import random
from datetime import datetime

from swiftsarsa import SwiftSarsaBinaryFeatures
from shimmy.registration import register_gymnasium_envs


class SwiftTDPaperPreprocessor:
    """Preprocessing described in the SwiftTD paper."""

    def __init__(self, num_actions):
        self.height = 105
        self.width = 80
        self.channels = 3
        self.bins = 8
        self.num_actions = num_actions

        self.pixels_per_channel = self.height * self.width
        self.features_per_channel = self.pixels_per_channel * self.bins
        self.total_pixel_features = self.features_per_channel * self.channels

        self.action_offset = self.total_pixel_features
        self.reward_offset = self.action_offset + self.num_actions
        self.total_features = self.reward_offset + 3

        print(f"Feature Vector Dimension: {self.total_features}")
        print(f"  - Pixels: {self.total_pixel_features}")
        print(f"  - Action: {self.num_actions}")
        print(f"  - Reward: 3")

    def reset(self):
        pass

    def extract(self, frame, prev_action, prev_reward):
        active_indices = []

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        binned = frame // 32

        stride_pixel = self.bins
        stride_channel = self.features_per_channel

        for c in range(self.channels):
            channel_bins = binned[:, :, c].flatten()
            pixel_offsets = np.arange(self.pixels_per_channel) * stride_pixel
            channel_start = c * stride_channel
            indices = channel_start + pixel_offsets + channel_bins
            active_indices.extend(indices)

        if 0 <= prev_action < self.num_actions:
            active_indices.append(self.action_offset + prev_action)

        r_clipped = int(np.clip(prev_reward, -1, 1))
        reward_idx = r_clipped + 1
        active_indices.append(self.reward_offset + reward_idx)

        return active_indices


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
    """
    Returns (action_map, num_actions, description).
    action_map is None when using the environment's action space directly.
    """
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
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--meta_step", type=float, default=1e-2)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--lambda_val", type=float, default=0.99)
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor gamma.")
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay_steps", type=int, default=100000)
    parser.add_argument("--swift_decay", type=float, default=0.9999,
                        help="Per-feature step-size decay factor (decay_init).")
    parser.add_argument("--swift_trace_epsilon", type=float, default=1e-6,
                        help="Trace culling threshold epsilon (epsilon_init).")
    parser.add_argument("--swift_eta_min", type=float, default=3e-7,
                        help="Minimum per-feature step-size (eta_min_init).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame_skip", type=int, default=4,
                        help="Repeat each action for this many frames, accumulating reward.")
    parser.add_argument("--betas_dir", type=str, default=None,
                        help="Directory for beta/alpha heatmaps (default: runs/<rom>_paper).")
    parser.add_argument("--heatmap_inverse_beta", action="store_true", default=True,
                        help="Use exp(-beta) for heatmaps (safer magnitude) instead of exp(beta) (default: inverse on)")
    parser.add_argument("--heatmap_raw_beta", dest="heatmap_inverse_beta", action="store_false",
                        help="Use exp(beta) for heatmaps (turn off inverse view)")
    parser.add_argument("--heatmap_every_steps", type=int, default=200,
                        help="Save beta/alpha heatmap every N steps (default: 200). Set <=0 to disable.")
    parser.add_argument("--credit_heatmap", action="store_true",
                        help="If set, accumulate credit-style heatmaps using active-bin alphas.")
    parser.add_argument("--video_dir", type=str, default=None, help="Optional directory to record gameplay MP4s")
    parser.add_argument("--video_every_episodes", type=int, default=1,
                        help="Record every N-th episode when video_dir is set (default: 1 = every episode).")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to write training logs.")
    parser.add_argument("--log_file", type=str, default=None, help="Optional explicit log file name.")
    parser.add_argument("--action_set", choices=["minimal", "full"], default="minimal",
                        help="Use ALE minimal action set (default) or the full action space.")
    args = parser.parse_args()

    betas_dir = args.betas_dir or os.path.join("runs", f"{args.rom.lower()}_paper")

    np.random.seed(args.seed)
    random.seed(args.seed)

    def _fmt_val(v):
        return str(v).replace(".", "p")

    if args.log_file:
        log_filename = args.log_file
    else:
        log_filename = (
            f"ale_{args.rom}_seed{args.seed}"
            f"_dec{args.decisions}"
            f"_alpha{_fmt_val(args.alpha)}"
            f"_meta{_fmt_val(args.meta_step)}"
            f"_eta{_fmt_val(args.eta)}"
            f"_lambda{_fmt_val(args.lambda_val)}"
            f"_swiftdecay{_fmt_val(args.swift_decay)}"
            f"_eps{_fmt_val(args.epsilon_end)}"
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
    render_mode = "rgb_array" if args.video_dir else None
    env = gym.make(f"ALE/{args.rom}-v5", obs_type="rgb", full_action_space=True, render_mode=render_mode)

    action_map, num_actions, action_desc, action_meanings = resolve_action_set(env, args.action_set)
    log(action_desc)
    if action_map is None and action_meanings:
        log(f"Action meanings: {action_meanings}")

    extractor = SwiftTDPaperPreprocessor(num_actions)

    learner = SwiftSarsaBinaryFeatures(
        extractor.total_features,
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
    log(f"Active Features: ~25,200")
    log(f"Initial Energy:  {25200 * args.alpha:.4f}")
    if (25200 * args.alpha) >= args.eta:
        log("WARNING: Unsafe initialization! C++ Kill Switch will trigger.")
    else:
        log("STATUS: Safe. Meta-learning enabled.")
    log(f"---------------------")

    step = 0
    episode_idx = 0
    returns = []
    next_heatmap_at = args.heatmap_every_steps if args.heatmap_every_steps > 0 else None
    credit_accum = np.zeros((105, 80), dtype=np.float32) if args.credit_heatmap else None

    os.makedirs(betas_dir, exist_ok=True)
    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)

    while step < args.decisions:
        frame, _ = env.reset(seed=args.seed + episode_idx)
        extractor.reset()
        learner.reset_episode()

        features = extractor.extract(frame, 0, 0.0)

        eps = current_epsilon(step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
        q_vals = learner.get_action_values(features)
        action = select_action(q_vals, eps)

        learner.learn(features, 0.0, 0.0, action)

        writer = None
        record_this_episode = bool(args.video_dir) and (
            args.video_every_episodes <= 0 or (episode_idx + 1) % args.video_every_episodes == 0
        )

        def close_writer():
            nonlocal writer
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
                writer = None

        episode_reward = 0
        done = False

        if record_this_episode:
            try:
                import imageio
                if writer is None:
                    path = os.path.join(args.video_dir, f"episode_{episode_idx + 1:05d}.mp4")
                    writer = imageio.get_writer(path, fps=60)
                frame_rgb = env.render()
                if frame_rgb is not None:
                    writer.append_data(frame_rgb)
            except Exception:
                pass

        while not done and step < args.decisions:
            ale_action = action if action_map is None else action_map[action]
            total_clipped_r = 0.0
            frames_used = 0
            for _ in range(max(1, args.frame_skip)):
                next_frame, r, term, trunc, _ = env.step(ale_action)
                frames_used += 1
                done = term or trunc
                clipped_r = np.clip(r, -1, 1)
                total_clipped_r += clipped_r
                episode_reward += r

                if args.video_dir and writer is not None:
                    try:
                        frame_rgb = env.render()
                        if frame_rgb is not None:
                            writer.append_data(frame_rgb)
                    except Exception:
                        pass

                if done:
                    break

            next_features = extractor.extract(next_frame, action, total_clipped_r)

            eps = current_epsilon(step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
            q_vals_next = learner.get_action_values(next_features)
            action_next = select_action(q_vals_next, eps)

            gamma = 0.0 if term else args.discount
            learner.learn(next_features, total_clipped_r, gamma, action_next)

            features = next_features
            action = action_next
            step += frames_used

            if next_heatmap_at is not None and step >= next_heatmap_at:
                try:
                    import imageio
                    betas = np.array(learner.get_feature_betas_max_over_actions())
                    alphas = np.exp(-betas) if args.heatmap_inverse_beta else np.exp(betas)

                    frame_resized = cv2.resize(next_frame, (extractor.width, extractor.height), interpolation=cv2.INTER_AREA)
                    binned = frame_resized // 32

                    pixels_per_channel = extractor.pixels_per_channel
                    bins = extractor.bins
                    features_per_channel = extractor.features_per_channel

                    pixel_offsets = np.arange(pixels_per_channel) * bins
                    heatmaps = []
                    for c in range(extractor.channels):
                        bin_vals = binned[:, :, c].reshape(-1).astype(np.int64)
                        channel_start = c * features_per_channel
                        indices = channel_start + pixel_offsets + bin_vals
                        heatmaps.append(alphas[indices].reshape(extractor.height, extractor.width))

                    active_heatmap = np.max(np.stack(heatmaps, axis=0), axis=0)

                    heatmap = active_heatmap
                    if args.credit_heatmap and credit_accum is not None:
                        credit_accum += active_heatmap
                        heatmap = credit_accum

                    h_min, h_max = heatmap.min(), heatmap.max()
                    span = h_max - h_min
                    if span > 1e-12:
                        norm_img = (heatmap - h_min) / span
                    else:
                        norm_img = np.zeros_like(heatmap)
                    imageio.imwrite(f"{betas_dir}/step_{step:07d}.png", (norm_img * 255).astype(np.uint8))
                except Exception:
                    pass
                next_heatmap_at += args.heatmap_every_steps

        episode_idx += 1
        returns.append(episode_reward)
        window_mean = float(np.mean(returns[-100:])) if returns else 0.0
        log(f"Ep {episode_idx}: reward={episode_reward:.1f} mean_last_100={window_mean:.2f} step={step}")
        close_writer()

    env.close()
    log_file.close()


if __name__ == "__main__":
    main()
