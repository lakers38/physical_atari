#!/usr/bin/env python3
"""
Generic Swift-Sarsa implementation for any ALE game using the SwiftTD-style
AtariFeatureExtractor (105x80 RGB with frame differencing, 8-bin per-channel).
Switch ROMs via --rom without changing the code; action set can be minimal or
full.

Includes simple memory-feature imprinting:
- Extra binary features that remember when a parent feature fired k2 steps ago
  and stay active for k1 steps.
- Parents are active observation features with |w| >= tenure_threshold.
- Generation is gated by tau_t = sum_{i active obs} exp(beta[i]) < eta.
"""

import argparse
import numpy as np
import gymnasium as gym
import cv2
import os
import random
from datetime import datetime
from typing import List, Optional

from swiftsarsa import SwiftSarsaBinaryFeatures
from shimmy.registration import register_gymnasium_envs
from agent_swift_sarsa import AtariFeatureExtractor


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

    if action_set_choice == "pong_reduced":
        # Paper setup: NOOP, RIGHT, LEFT
        reduced_actions = [0, 3, 4]
        num_actions = len(reduced_actions)
        labels = []
        if action_meanings:
            labels = [
                action_meanings[a] if a < len(action_meanings) else str(a) for a in reduced_actions
            ]
        desc = f"Action set: Pong reduced {reduced_actions} (NOOP, RIGHT, LEFT)"
        if labels:
            desc += f" meanings={labels}"
        return reduced_actions, num_actions, desc, action_meanings

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


# ----------------------------------------------------------------------
# Memory feature imprinting
# ----------------------------------------------------------------------


class MemoryFeature:
    """
    Memory feature using a bitwise shift register to handle overlapping activations.
    Acts like a delay line: if the parent fires again while waiting/active, the window extends.
    """

    def __init__(self, parent_idx: int, feature_idx: int, k1: int, k2: int):
        self.parent_idx = parent_idx
        self.feature_idx = feature_idx
        self.k1 = k1
        self.k2 = k2

        # Bit 0 is the parent's status at time T, bit 1 at T-1, etc. Seed with 1 to mark creation.
        self.history = 1
        # Mask selects bits in [k2, k2 + k1)
        self.window_mask = ((1 << self.k1) - 1) << self.k2

    def update(self, parent_active: bool) -> bool:
        """Shift history, insert new parent state, and report if the memory feature is active."""
        self.history = ((self.history << 1) | int(parent_active)) & ((1 << 64) - 1)
        return (self.history & self.window_mask) > 0


class ImprintingFeatureManager:
    """
    Generates and manages delayed memory features with pruning and slot reuse.
    - Parents are active obs features with |w| >= tenure_threshold.
    - tau_t computed over all active features (obs + memory).
    - Prunes idle memory features to recycle slots.
    - Resets learner parameters when recycling to avoid stale weights/betas.
    """

    def __init__(
        self,
        learner: SwiftSarsaBinaryFeatures,
        obs_features: int,
        max_memory_features: int,
        k_per_step: int,
        k1_values: List[int],
        k2_values: List[int],
        tenure_threshold: float,
        eta: float,
        alpha_init: float,
        prune_interval: int = 200,
        idle_threshold: float = 0.001,
    ):
        self.learner = learner
        self.obs_features = obs_features
        self.max_memory_features = max_memory_features
        self.k_per_step = k_per_step
        self.k1_values = k1_values
        self.k2_values = k2_values
        self.tenure_threshold = tenure_threshold
        self.eta = eta
        self.alpha_init = alpha_init
        self.prune_interval = prune_interval
        self.idle_threshold = idle_threshold

        self.active_slots: dict[int, MemoryFeature] = {}
        self.free_slots = list(range(max_memory_features))
        self.step_counter = 0
        self.active_sum = 0
        self.active_steps = 0

    def build_feature_vector(self, obs_indices: List[int]) -> List[int]:
        """
        Given active observation feature indices, update memory features and
        return combined obs + memory feature indices.
        """
        self.step_counter += 1
        obs_set = set(obs_indices)

        # Update existing memory features
        active_mem_indices = []
        for slot, mem in self.active_slots.items():
            parent_active = mem.parent_idx in obs_set
            if mem.update(parent_active):
                active_mem_indices.append(mem.feature_idx)

        full_indices = obs_indices + active_mem_indices
        self.active_sum += len(active_mem_indices)
        self.active_steps += 1

        if self.prune_interval > 0 and self.step_counter % self.prune_interval == 0:
            self._prune_features()

        self._imprint_new_memory_features(obs_indices, full_indices)
        return full_indices

    def _imprint_new_memory_features(self, obs_indices: List[int], full_indices: List[int]) -> None:
        """Possibly generate up to k_per_step new memory features this step."""
        if not self.free_slots or self.k_per_step <= 0:
            return

        # tau_t over all active features (obs + memory)
        betas = np.array(self.learner.get_feature_betas_max_over_actions())
        tau = float(np.sum(np.exp(betas[full_indices])))

        # Gate by eta
        if tau >= self.eta:
            return

        # Tenured parents: active obs features with large |w|
        weights = np.array(self.learner.get_feature_weights_max_over_actions())
        parent_candidates = [i for i in obs_indices if abs(weights[i]) >= self.tenure_threshold]
        if not parent_candidates:
            return

        generated = 0
        while generated < self.k_per_step and self.free_slots:
            if tau >= self.eta:
                break

            parent_idx = random.choice(parent_candidates)
            k1 = random.choice(self.k1_values)
            k2 = random.choice(self.k2_values)

            slot = self.free_slots.pop()
            feature_idx = self.obs_features + slot

            if hasattr(self.learner, "reset_feature"):
                try:
                    self.learner.reset_feature(feature_idx, self.alpha_init)
                except Exception:
                    pass

            self.active_slots[slot] = MemoryFeature(parent_idx, feature_idx, k1, k2)
            tau += self.alpha_init  # heuristic increment to avoid overshooting eta badly
            generated += 1

    def _prune_features(self):
        """Remove idle memory features (low |w|) to recycle slots."""
        if not self.active_slots:
            return

        weights = np.array(self.learner.get_feature_weights_max_over_actions())
        slots_to_remove = []

        for slot, mem in self.active_slots.items():
            if abs(weights[mem.feature_idx]) < self.idle_threshold:
                slots_to_remove.append(slot)

        max_remove = self.k_per_step * 5 if self.k_per_step > 0 else len(slots_to_remove)
        for slot in slots_to_remove[:max_remove]:
            del self.active_slots[slot]
            self.free_slots.append(slot)

    def pop_active_stats(self) -> float:
        """Return avg active memory features per step since last call and reset counters."""
        if self.active_steps == 0:
            return 0.0
        avg = float(self.active_sum) / float(self.active_steps)
        self.active_sum = 0
        self.active_steps = 0
        return avg


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


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
    parser.add_argument(
        "--action_set",
        choices=["minimal", "full", "pong_reduced"],
        default=None,
        help="Action set: minimal/full generic, or pong_reduced=[0,3,4] paper setup (default: pong_reduced for Pong, minimal otherwise).",
    )
    # Imprinting args
    parser.add_argument("--use_imprinting", action="store_true",
                        help="Enable memory feature generation via imprinting.")
    parser.add_argument("--max_memory_features", type=int, default=1000,
                        help="Maximum number of memory features to allocate.")
    parser.add_argument("--imprint_k_per_step", type=int, default=2,
                        help="Maximum new memory features to generate per step.")
    parser.add_argument("--imprint_k1s", type=int, nargs="+", default=[2, 3, 4],
        help="Durations k1 (active steps) to sample for new memory features.")
    parser.add_argument("--imprint_k2s", type=int, nargs="+", default=[1, 2, 4],
                        help="Delays k2 (wait steps) to sample for new memory features.")
    parser.add_argument("--tenure_threshold", type=float, default=0.1,
                        help="Absolute weight magnitude to treat a feature as tenured.")
    parser.add_argument("--imprint_idle_threshold", type=float, default=0.001,
                    help="Weight magnitude below which a feature is pruned (default: 0.001).")
    parser.add_argument("--imprint_prune_interval", type=int, default=200,
                    help="How often (steps) to check for pruning (default: 200).")
    args = parser.parse_args()
    if args.action_set is None:
        args.action_set = "pong_reduced" if args.rom.lower() == "pong" else "minimal"

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

    extractor = AtariFeatureExtractor(
        height=105,
        width=80,
        num_bins=8,
        num_actions=num_actions,
        use_grayscale=False,
        use_frame_diff=False,  # Match SwiftTD paper: no frame differencing
        K_actions=0,
        K_rewards=0,
        use_reward_gap=False,
    )

    obs_feature_count = extractor.total_features
    max_mem = args.max_memory_features if args.use_imprinting else 0
    total_feature_count = obs_feature_count + max_mem

    learner = SwiftSarsaBinaryFeatures(
        total_feature_count,
        num_actions,
        args.lambda_val,
        args.alpha,
        args.meta_step,
        args.eta,
        args.swift_decay,
        args.swift_trace_epsilon,
        args.swift_eta_min
    )

    imprint_mgr: Optional[ImprintingFeatureManager] = None
    if args.use_imprinting:
        imprint_mgr = ImprintingFeatureManager(
            learner=learner,
            obs_features=obs_feature_count,
            max_memory_features=args.max_memory_features,
            k_per_step=args.imprint_k_per_step,
            k1_values=args.imprint_k1s,
            k2_values=args.imprint_k2s,
            tenure_threshold=args.tenure_threshold,
            eta=args.eta,
            alpha_init=args.alpha,
            idle_threshold=args.imprint_idle_threshold,
            prune_interval=args.imprint_prune_interval,

        )

    log(f"Logging to {log_path}")
    log(f"--- Configuration ---")
    log(f"ROM: {args.rom}")
    log(f"Observation Features: {obs_feature_count}")
    log(f"Total Features (with memory cap): {total_feature_count}")
    log(f"Initial Energy:  {obs_feature_count * args.alpha:.4f}")
    if args.use_imprinting:
        log(
            f"Imprinting: max_mem={args.max_memory_features} k_per_step={args.imprint_k_per_step} "
            f"k1s={args.imprint_k1s} k2s={args.imprint_k2s} tenure={args.tenure_threshold}"
        )
        log("Imprinting enabled.")
    log(f"---------------------")

    step = 0
    episode_idx = 0
    returns = []
    next_heatmap_at = args.heatmap_every_steps if args.heatmap_every_steps > 0 else None
    credit_accum = np.zeros((extractor.height, extractor.width), dtype=np.float32) if args.credit_heatmap else None

    os.makedirs(betas_dir, exist_ok=True)
    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)

    while step < args.decisions:
        frame, _ = env.reset(seed=args.seed + episode_idx)
        extractor.reset()
        extractor.update_reward(0.0)
        learner.reset_episode()

        obs_indices = extractor.extract(frame)
        features = (
            imprint_mgr.build_feature_vector(obs_indices)
            if imprint_mgr is not None else obs_indices
        )

        eps = current_epsilon(step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
        q_vals = learner.get_action_values(features)
        serve_action = None
        if args.rom.lower() == "pong":
            if action_map is None:
                # Full action space: 1 is FIRE
                serve_action = 1 if num_actions > 1 else None
            else:
                # Find FIRE within the chosen mapping if available
                am_list = list(action_map)
                fire_ale_idx = None
                if action_meanings:
                    try:
                        fire_ale_idx = action_meanings.index("FIRE")
                    except ValueError:
                        fire_ale_idx = None
                if fire_ale_idx is not None and fire_ale_idx in am_list:
                    serve_action = am_list.index(fire_ale_idx)
                elif 1 in am_list:
                    serve_action = am_list.index(1)
        action = serve_action if serve_action is not None else select_action(q_vals, eps)

        learner.learn(features, 0.0, 0.0, action)
        extractor.set_prev_action(action)

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

            extractor.update_reward(total_clipped_r)
            obs_next = extractor.extract(next_frame)
            next_features = (
                imprint_mgr.build_feature_vector(obs_next)
                if imprint_mgr is not None else obs_next
            )

            eps = current_epsilon(step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
            q_vals_next = learner.get_action_values(next_features)
            action_next = select_action(q_vals_next, eps)
            extractor.set_prev_action(action_next)

            gamma = 0.0 if term else args.discount
            learner.learn(next_features, total_clipped_r, gamma, action_next)

            features = next_features
            action = action_next
            step += frames_used

            if next_heatmap_at is not None and step >= next_heatmap_at:
                if extractor.num_channels != 3:
                    next_heatmap_at += args.heatmap_every_steps
                    continue
                try:
                    import imageio
                    betas = np.array(learner.get_feature_betas_max_over_actions())
                    alphas = np.exp(-betas) if args.heatmap_inverse_beta else np.exp(betas)

                    frame_resized = cv2.resize(next_frame, (extractor.width, extractor.height), interpolation=cv2.INTER_AREA)
                    binned = frame_resized // 32

                    pixels_per_channel = extractor.height * extractor.width
                    bins = extractor.num_bins
                    features_per_channel = pixels_per_channel * bins

                    pixel_offsets = np.arange(pixels_per_channel) * bins
                    heatmaps = []
                    for c in range(3):
                        bin_vals = binned[:, :, c].reshape(-1).astype(np.int64)
                        channel_start = c * features_per_channel
                        indices = channel_start + pixel_offsets + bin_vals
                        # Only observation features contribute to these indices,
                        # so this ignores memory features (which is fine for now).
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
        if imprint_mgr is not None and episode_idx % 10 == 0:
            avg_active = imprint_mgr.pop_active_stats() if imprint_mgr is not None else 0.0
            log(
                f"[Imprint] mem_features={len(imprint_mgr.active_slots)} "
                f"free_slots={len(imprint_mgr.free_slots)} "
                f"avg_active_per_step={avg_active:.3f}"
            )
        close_writer()

    env.close()
    log_file.close()


if __name__ == "__main__":
    main()
