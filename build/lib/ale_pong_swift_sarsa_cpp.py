#!/usr/bin/env python3
"""
Swift-Sarsa trainer with Section 9.2 imprinting on Atari Pong.

Usage:
    python ale_pong_swift_sarsa.py --rom Pong --reduce_action_set
    python ale_pong_swift_sarsa.py --rom Pong --reduce_action_set --use_imprinting
"""

from __future__ import annotations

import argparse
import math
import random
import numpy as np
import gymnasium as gym
from collections import deque
from typing import List, Sequence, Tuple
from shimmy.registration import register_gymnasium_envs

from swiftsarsa import SwiftSarsa, MemoryFeatures, MemoryImprinter, MemoryReaper
from agent_swift_sarsa import AtariFeatureExtractor


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def parse_int_list(arg: str, default: List[int]) -> List[int]:
    """Parse comma-separated ints with a fallback default."""
    if not arg:
        return default
    try:
        vals = [int(x.strip()) for x in arg.split(",") if x.strip() != ""]
        return vals if vals else default
    except Exception:
        return default


def select_action(q_values: np.ndarray, eps: float, temp: float, use_softmax: bool = True) -> int:
    """Epsilon-greedy with optional softmax exploration."""
    # Guard against NaNs/Infs from upstream; fall back to zeros.
    if not np.all(np.isfinite(q_values)):
        q_values = np.nan_to_num(q_values, nan=0.0, posinf=0.0, neginf=0.0)

    if np.random.random() < eps:
        return np.random.randint(len(q_values))
    if use_softmax:
        q_shifted = q_values - np.max(q_values)
        exp_q = np.exp(np.clip(q_shifted / temp, -50, 50))
        probs = exp_q / (exp_q.sum() + 1e-10)
        # If numerical issues still yield invalid probabilities, choose uniformly.
        if not np.all(np.isfinite(probs)) or np.any(probs < 0) or probs.sum() <= 0:
            return int(np.random.randint(len(q_values)))
        return int(np.random.choice(len(q_values), p=probs))
    return int(np.argmax(q_values))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decisions", type=int, default=500_000)
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--reduce_action_set", action="store_true")
    # Memory features (Figure 9.2 style)
    parser.add_argument("--use_memory", action="store_true",
                        help="Enable memory features (Figure 9.2)")
    parser.add_argument("--ema_decay", type=float, default=0.95,
                        help="EMA decay for return tracking")
    parser.add_argument("--plot_path", type=str, default=None,
                        help="If set, save a PNG plot of return and EMA to this path")
    parser.add_argument("--alpha", type=float, default=2e-6,
                        help="Initial alpha. Constraint: alpha * active_feats <= eta. For 50k active and eta=0.1, use ~2e-6.")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Epsilon for exploration")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Softmax temperature")
    parser.add_argument("--pure_epsilon", action="store_true",
                        help="Disable softmax; use pure epsilon-greedy")
    parser.add_argument("--debug_features_every", type=int, default=0,
                        help="If >0, print base feature stats every N global steps")
    parser.add_argument("--max_memory_features", type=int, default=5000,
                        help="Max number of memory features to allocate capacity for")
    parser.add_argument("--max_new_memory_per_step", type=int, default=5,
                        help="Upper bound on new memory features created each step")
    parser.add_argument("--tenure_threshold", type=float, default=0.05,
                        help="|w| threshold to consider a feature tenured")
    parser.add_argument("--memory_durations", type=str, default="1,4,16",
                        help="Comma-separated durations (k1) to sample for memory features")
    parser.add_argument("--memory_delays", type=str, default="0,4,16,32,64,128,256,512",
                        help="Comma-separated delays (k2) to sample for memory features")
    parser.add_argument("--idle_threshold", type=float, default=0.005,
                        help="|w| threshold to consider a memory feature idle (eligible for removal)")
    parser.add_argument("--reaper_patience", type=int, default=5000,
                        help="Steps of inactivity before a low-weight memory feature can be removed")
    parser.add_argument("--reaper_every", type=int, default=500,
                        help="Check for memory feature removal every N decisions")
    args = parser.parse_args()

    # Environment
    try:
        register_gymnasium_envs()
    except Exception:
        pass

    env = gym.make(f"ALE/{args.rom}-v5", obs_type="rgb", full_action_space=True)
    frame, _ = env.reset(seed=args.seed)

    # Action mapping for Pong
    if args.reduce_action_set and args.rom.lower() == "pong":
        # Minimal, sensible Pong set: NOOP, UP, DOWN
        action_map = np.array([0, 2, 3], dtype=np.int64)
        num_actions = len(action_map)
    else:
        action_map = None
        num_actions = env.action_space.n

    try:
        meanings = env.unwrapped.get_action_meanings()
        print(f"Action meanings: {meanings}")
    except Exception:
        pass

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Hyperparams
    gamma = 0.98  # From text
    temperature = args.temperature
    exploration_eps = args.epsilon
    reward_clip = 1.0
    learning_enabled = args.alpha > 0
    alpha = max(args.alpha, 1e-12)  # avoid log(0) NaNs in backend
    prev_feature_set = None  # for debug change tracking

    # Feature extractor (SwiftTD style)
    extractor = AtariFeatureExtractor(
        height=105,
        width=80,
        num_bins=8,
        num_actions=num_actions,
        use_grayscale=False,
        use_frame_diff=True,    # Enable frame differencing for motion perception
        K_actions=0,            # disable action history for paper alignment
        K_rewards=0,            # disable reward history for paper alignment
        use_reward_gap=False,   # no gap buckets in paper setup
    )
    base_features = extractor.total_features
    active_per_frame = getattr(extractor, "_active_per_frame", None)
    if active_per_frame is not None:
        tau = alpha * active_per_frame
        print(
            f"Feature stats: total={base_features}, active_per_frame≈{active_per_frame}, "
            f"alpha={alpha}, alpha*active≈{tau:.3e}"
        )

    # Memory timing patterns (discrete grid)
    durations = parse_int_list(args.memory_durations, [1, 4, 16])
    delays = parse_int_list(args.memory_delays, [0, 4, 16, 32, 64, 128, 256, 512])
    patterns = [(k1, k2) for k1 in durations for k2 in delays]

    # Memory features (Figure 9.2 style) – allocate capacity, fill online
    if args.use_memory:
        max_mem = args.max_memory_features
        total_features = base_features + max_mem
        free_mem_ids = list(range(base_features, base_features + max_mem))
        memory = MemoryFeatures(num_tenured=base_features)
        print(f"Memory capacity: {max_mem} (patterns={len(patterns)})")
    else:
        memory = None
        free_mem_ids = []
        total_features = base_features

    # Swift-Sarsa learner
    learner = SwiftSarsa(
        total_features,
        num_actions,
        0.95,     # lambda (from snippet)
        alpha,    # alpha
        1e-3,     # theta/meta_step_size
        0.1,      # eta (from snippet)
        0.999,    # decay
        1e-5,     # epsilon (from snippet)
        1e-10,    # eta_min (from snippet)
    )

    if args.use_memory:
        imprinter = MemoryImprinter(
            base_features=base_features,
            memory=memory,
            max_memory_features=max_mem,
            alpha_init=learner.get_alpha_init(),
            eta=learner.get_eta(),
            tenure_thresh=args.tenure_threshold,
            patterns=patterns,
            max_new_per_step=args.max_new_memory_per_step,
        )
        imprinter.set_free_ids(free_mem_ids)
        
        reaper = MemoryReaper(
            base_features=base_features,
            memory=memory,
            imprinter=imprinter,
            idle_thresh=args.idle_threshold,
            patience=args.reaper_patience,
            check_every=args.reaper_every,
        )
    else:
        imprinter = None
        reaper = None

    print(f"Base features: {base_features}, Total: {total_features}")

    # -------------------------------------------------------------------------
    # Initialize for first state s_0
    # -------------------------------------------------------------------------
    extractor.reset()
    if memory is not None:
        memory.reset()

    base_pairs = extractor.extract(frame)
    # For memory logic, we need indices of active features (non-zero value)
    base_indices_only = [idx for idx, val in base_pairs if val != 0]
    
    if memory:
        memory_indices = memory.step(base_indices_only)
        # Combine base pairs with memory pairs (value 1.0)
        memory_pairs = [(idx, 1.0) for idx in memory_indices]
        
        memory_indices = memory.step(base_indices_only)
        memory_only_indices = [idx for idx in memory_indices if idx >= base_features]
        features = base_pairs + [(idx, 1.0) for idx in memory_only_indices]
    else:
        features = base_pairs
        
    base_indices_prev = base_indices_only # Store indices for next step's imprinting logic

    q_values = np.array(learner.get_action_values(features))
    action = select_action(q_values, exploration_eps, temperature, use_softmax=not args.pure_epsilon)

    # Store prev state/action for learn() call
    prev_features = features
    prev_action = action

    extractor.set_prev_action(action)

    # Stats
    episode_reward = 0.0
    reward_ema = None
    episode_idx = 0
    global_step = 0
    returns = []
    ema_series = []
    td_ema = None
    last_avg_q = 0.0
    last_max_q = 0.0
    action_counts = np.zeros(num_actions, dtype=np.int64)
    td_ema = None
    last_avg_q = 0.0
    last_max_q = 0.0

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    while global_step < args.decisions:
        # Map to ALE action and step environment
        ale_action = int(action_map[prev_action]) if action_map is not None else prev_action

        total_r = 0.0
        done = trunc = False
        for _ in range(args.frame_skip):
            next_frame, r, terminated, truncated, _ = env.step(ale_action)
            total_r += float(r)
            frame = next_frame
            done = bool(terminated)
            trunc = bool(truncated)
            if done or trunc:
                break

        reward = np.clip(total_r, -reward_clip, reward_clip)
        g = 0.0 if (done or trunc) else gamma

        # Learn on (prev_features, prev_action) with reward from this transition
        # Update reward in extractor for NEXT step's extraction
        extractor.update_reward(reward)
        
        q_curr = np.array(learner.get_action_values(prev_features))
        if learning_enabled:
            learner.learn(prev_features, reward, g, prev_action)

        # Online imprinting (uses base features from time t)
        if imprinter is not None:
            w_feat = learner.get_feature_weights_max_over_actions()
            beta_feat = learner.get_feature_betas_max_over_actions()
            imprinter.step(base_indices_prev, w_feat, beta_feat)

        # Extract features for next state
        base_pairs_next = extractor.extract(frame)
        base_indices_next_only = [idx for idx, val in base_pairs_next if val != 0]
        
        if memory:
            memory_indices_next = memory.step(base_indices_next_only)
            memory_only_indices_next = [idx for idx in memory_indices_next if idx >= base_features]
            features_next = base_pairs_next + [(idx, 1.0) for idx in memory_only_indices_next]
        else:
            features_next = base_pairs_next

        # Online removal / recycling for memory features
        if reaper is not None:
            # reaper expects list of indices, not pairs
            features_next_indices = [idx for idx, _ in features_next]
            reaper.step(features_next_indices, w_feat)

        if args.debug_features_every and (global_step % args.debug_features_every == 0):
            cur_set = set(base_indices_next_only)
            change_frac = None
            if prev_feature_set is not None and len(cur_set) > 0:
                inter = len(cur_set & prev_feature_set)
                change_frac = 1.0 - inter / len(cur_set)
            prev_feature_set = cur_set

            sample = list(cur_set)
            random.shuffle(sample)
            sample = sample[:5]

            print(
                f"debug step {global_step}: "
                f"active_base={len(cur_set)} "
                f"change_frac={change_frac} "
                f"sample={sample}"
            )

        # Get Q-values and select next action
        q_values_next = np.array(learner.get_action_values(features_next))
        last_avg_q = float(q_values_next.mean())
        last_max_q = float(q_values_next.max())
        delta_hat = reward + g * last_max_q - float(q_curr.max())
        if td_ema is None:
            td_ema = delta_hat
        else:
            td_ema = args.ema_decay * td_ema + (1 - args.ema_decay) * delta_hat
        action_next = select_action(q_values_next, exploration_eps, temperature, use_softmax=not args.pure_epsilon)
        action_counts[action_next] += 1

        # Update extractor state
        extractor.set_prev_action(action_next)

        # Shift for next iteration
        base_indices_prev = base_indices_next_only
        prev_features = features_next
        prev_action = action_next

        # Bookkeeping
        episode_reward += total_r
        global_step += 1
        
        if global_step % 1000 == 0:
            print(
                f"step {global_step}  "
                f"ep_rew {episode_reward:.1f}  "
                f"avg_q {last_avg_q:.3f}  "
                f"max_q {last_max_q:.3f}  "
                f"mem_feats={memory.num_memory_features() if memory else 0}"
            )

        # Episode end
        if done or trunc:
            if reward_ema is None:
                reward_ema = episode_reward
            else:
                reward_ema = args.ema_decay * reward_ema + (1 - args.ema_decay) * episode_reward

            returns.append(episode_reward)
            ema_series.append(reward_ema)

            mem_str = f"  mem_feats={memory.num_memory_features()}" if memory else ""
            td_str = f"{td_ema:.3f}" if td_ema is not None else "nan"
            action_str = f"actions {action_counts.tolist()}"
            learn_str = "frozen" if not learning_enabled else "learn"
            print(
                f"ep {episode_idx}  "
                f"reward {episode_reward:.1f}  "
                f"ema {reward_ema:.2f}  "
                f"step {global_step}  "
                f"avg_q {last_avg_q:.3f}  "
                f"max_q {last_max_q:.3f}  "
                f"td_ema {td_str}  "
                f"{action_str}  "
                f"{learn_str}"
                f"{mem_str}"
            )

            episode_idx += 1
            episode_reward = 0.0

            # Reset for new episode
            frame, _ = env.reset()
            extractor.reset()
            if memory is not None:
                memory.reset()
            learner.reset_episode()

            # Initialize for new s_0
            base_pairs = extractor.extract(frame)
            base_indices_only = [idx for idx, val in base_pairs if val != 0]
            
            if memory:
                memory_indices = memory.step(base_indices_only)
                memory_only_indices = [idx for idx in memory_indices if idx >= base_features]
                features = base_pairs + [(idx, 1.0) for idx in memory_only_indices]
            else:
                features = base_pairs
                base_indices_prev = base_indices_only
            q_values = np.array(learner.get_action_values(features))
            action = select_action(q_values, exploration_eps, temperature, use_softmax=not args.pure_epsilon)

            prev_features = features
            prev_action = action
            extractor.set_prev_action(action)

    print(f"\nTraining finished. Episodes: {episode_idx}")

    if args.plot_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 4))
            plt.plot(returns, label="return")
            plt.plot(ema_series, label=f"ema (decay={args.ema_decay})")
            plt.xlabel("episode")
            plt.ylabel("return")
            plt.title(f"Pong Swift-Sarsa ({'with' if memory else 'no'} memory)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.plot_path)
            print(f"Saved plot to {args.plot_path}")
        except Exception as e:
            print(f"Could not save plot: {e}")


if __name__ == "__main__":
    main()
