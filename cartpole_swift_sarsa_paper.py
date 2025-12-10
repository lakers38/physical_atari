#!/usr/bin/env python3
"""
Swift-Sarsa runner for CartPole with tile-coded state features.

Features (binary indices):
- Tile-coded 4D state using overlapping tilings
- Previous action one-hot
- Previous reward one-hot {-1, 0, +1}
"""

import argparse
import os
import random
import numpy as np
import gymnasium as gym
from datetime import datetime

from swiftsarsa import SwiftSarsaBinaryFeatures


class TileCoder:
    """
    Simple n-dimensional tile coder for continuous states.

    - num_tilings: how many overlapping tilings
    - tiles_per_dim: number of tiles per dimension per tiling
    - ranges: list of (low, high) for each dimension
    """

    def __init__(self, num_tilings: int, tiles_per_dim: int, ranges):
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.ranges = ranges
        self.dims = len(ranges)

        # Tiles per tiling and total tiles
        self.tiles_per_tiling = tiles_per_dim ** self.dims
        self.total_tiles = self.num_tilings * self.tiles_per_tiling

        # Scale from continuous range → tile coordinates
        self.scales = []
        for low, high in self.ranges:
            span = high - low
            self.scales.append(self.tiles_per_dim / (span + 1e-9))

        # Per-tiling offsets (stagger each dim a bit differently)
        # Offsets are in "tile units" (added before floor)
        self.offsets = []
        for m in range(self.num_tilings):
            frac = (m + 1) / float(self.num_tilings + 1)
            self.offsets.append([frac * (i + 1) / self.tiles_per_dim for i in range(self.dims)])

    def encode(self, obs: np.ndarray):
        """
        Return active tile indices (list[int]) for a given continuous observation.
        """
        active = []
        obs = np.asarray(obs, dtype=np.float32)

        for m in range(self.num_tilings):
            idxs = []
            for i, (low, high) in enumerate(self.ranges):
                x = float(obs[i])
                if x < low:
                    x = low
                elif x > high:
                    x = high

                scaled = (x - low) * self.scales[i] + self.offsets[m][i]
                tile = int(np.floor(scaled))
                if tile < 0:
                    tile = 0
                elif tile >= self.tiles_per_dim:
                    tile = self.tiles_per_dim - 1
                idxs.append(tile)

            # Flatten multi-d index into single index within this tiling
            flat_index = 0
            for i in range(self.dims):
                flat_index *= self.tiles_per_dim
                flat_index += idxs[i]

            # Global index across all tilings
            global_index = m * self.tiles_per_tiling + flat_index
            active.append(global_index)

        return active


def select_action(q_values, epsilon: float) -> int:
    """Standard epsilon-greedy exploration (Python side, NOT Swift)."""
    if np.random.random() < epsilon:
        return np.random.randint(len(q_values))
    return int(np.argmax(q_values))


def current_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """Linearly anneal exploration epsilon from start→end over decay_steps."""
    if decay_steps <= 0:
        return end
    if step < decay_steps:
        return start - (step / decay_steps) * (start - end)
    return end


class CartPolePreprocessor:
    def __init__(
        self,
        num_actions: int,
        num_tilings: int = 8,
        tiles_per_dim: int = 8,
    ):
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim

        # Typical CartPole ranges (clipped)
        self.ranges = [
            (-4.8, 4.8),        # cart position
            (-5.0, 5.0),        # cart velocity
            (-0.418, 0.418),    # pole angle (~24 deg)
            (-5.0, 5.0),        # pole angular velocity
        ]

        # Tile coder over all 4 dims jointly
        self.tilecoder = TileCoder(
            num_tilings=num_tilings,
            tiles_per_dim=tiles_per_dim,
            ranges=self.ranges,
        )

        self.total_state_features = self.tilecoder.total_tiles

        # Offsets for action + reward features
        self.action_offset = self.total_state_features
        self.reward_offset = self.action_offset + self.num_actions
        self.total_features = self.reward_offset + 3  # reward one-hot {-1,0,+1}

        print(f"Feature Vector Dimension: {self.total_features}")
        print(f"  - State (tile-coded): {self.total_state_features} "
              f"(tilings={num_tilings}, tiles_per_dim={tiles_per_dim})")
        print(f"  - Action: {self.num_actions}")
        print(f"  - Reward: 3")

    def reset(self):
        # No internal state needed for now.
        pass

    def extract(self, obs: np.ndarray, prev_action: int, prev_reward: float):
        """
        Build sparse binary features:
        - Tile-coded state (multiple tilings over 4D state)
        - Prev action one-hot
        - Prev reward one-hot in {-1, 0, +1}
        """
        active = []

        # State tiles
        active.extend(self.tilecoder.encode(obs))

        # Prev action one-hot
        if 0 <= prev_action < self.num_actions:
            active.append(self.action_offset + prev_action)

        # Prev reward one-hot
        r_clipped = int(np.clip(prev_reward, -1, 1))
        reward_idx = r_clipped + 1  # [-1,0,1] -> [0,1,2]
        active.append(self.reward_offset + reward_idx)

        return active


def main():
    parser = argparse.ArgumentParser()

    # Training horizon
    parser.add_argument("--decisions", type=int, default=200_000)

    # Swift-Sarsa base hyperparameters
    parser.add_argument("--alpha", type=float, default=1e-4,
                        help="Initial per-feature step-size (alpha_init).")
    parser.add_argument("--meta_step", type=float, default=1e-2,
                        help="Meta step-size for adapting step-sizes (meta_step_size_init).")
    parser.add_argument("--eta", type=float, default=1.0,
                        help="Target total step-size budget (eta_init).")
    parser.add_argument("--lambda_val", type=float, default=0.99,
                        help="Trace decay parameter lambda.")

    # Discount factor (Python-side)
    parser.add_argument("--discount", type=float, default=0.99,
                        help="Discount factor gamma for non-terminal steps.")

    # Exploration epsilon (Python-side, for epsilon-greedy)
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="Initial exploration epsilon for epsilon-greedy.")
    parser.add_argument("--epsilon_end", type=float, default=0.01,
                        help="Final exploration epsilon for epsilon-greedy.")
    parser.add_argument("--epsilon_decay_steps", type=int, default=50_000,
                        help="Steps over which to anneal exploration epsilon.")

    # Swift-Sarsa specific knobs (C++ side)
    parser.add_argument("--swift_decay", type=float, default=0.999,
                        help="Per-feature step-size decay factor (decay_init).")
    parser.add_argument("--swift_trace_epsilon", type=float, default=0.999,
                        help="Trace culling threshold epsilon (epsilon_init). "
                             "Traces are dropped when z <= last_alpha * epsilon.")
    parser.add_argument("--swift_eta_min", type=float, default=1e-8,
                        help="Minimum per-feature step-size (eta_min_init).")

    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--betas_dir", type=str, default="runs/cartpole_paper")
    parser.add_argument("--ema_decay", type=float, default=0.95,
                        help="EMA decay for logging returns.")
    parser.add_argument("--heatmap_inverse_beta", action="store_true", default=True,
                        help="Use exp(-beta) for heatmaps instead of exp(beta).")
    parser.add_argument("--heatmap_raw_beta", dest="heatmap_inverse_beta", action="store_false",
                        help="Use exp(beta) for heatmaps.")
    parser.add_argument("--save_betas_every", type=int, default=500,
                        help="Save beta heatmap every N steps (0 to disable).")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to write training logs.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional explicit log file name. If unset, a name is generated from args.")
    parser.add_argument("--eval_every_episodes", type=int, default=0,
                        help="If >0, run periodic eval every N episodes (greedy, no learning).")
    parser.add_argument("--eval_episodes", type=int, default=0,
                        help="Number of eval episodes to run each eval pass.")
    parser.add_argument("--eval_seed_offset", type=int, default=10_000,
                        help="Offset added to seed for eval env to avoid replaying training RNG.")

    args = parser.parse_args()

    # Deterministic RNG for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ----------------------------------------------------------------------
    # Logging setup
    # ----------------------------------------------------------------------
    def _fmt_val(v):
        # Replace dots to keep filenames safe; keep negatives if ever needed.
        return str(v).replace(".", "p")

    if args.log_file:
        log_filename = args.log_file
    else:
        log_filename = (
            f"cartpole_seed{args.seed}"
            f"_dec{args.decisions}"
            f"_alpha{_fmt_val(args.alpha)}"
            f"_meta{_fmt_val(args.meta_step)}"
            f"_eta{_fmt_val(args.eta)}"
            f"_lambda{_fmt_val(args.lambda_val)}"
            f"_eps{_fmt_val(args.swift_trace_epsilon)}"
            f"_ts{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            ".txt"
        )

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, log_filename)
    log_file = open(log_path, "w", buffering=1)

    def log(msg: str):
        print(msg)
        log_file.write(msg + "\n")

    # Environment
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)
    num_actions = env.action_space.n
    eval_env = None
    if args.eval_every_episodes > 0 and args.eval_episodes > 0:
        eval_env = gym.make("CartPole-v1")

    extractor = CartPolePreprocessor(num_actions)

    # ----------------------------------------------------------------------
    # Swift-SarsaBinaryFeatures wiring
    # ----------------------------------------------------------------------
    # Constructor signature (C++):
    # SwiftSarsaBinaryFeatures(int num_of_features,
    #                          int num_of_actions,
    #                          float lambda_init,
    #                          float alpha_init,
    #                          float meta_step_size_init,
    #                          float eta_init,
    #                          float decay_init,
    #                          float epsilon_init,
    #                          float eta_min_init)
    #
    # - decay_init        (args.swift_decay)        → controls decay of per-feature step-sizes alpha.
    # - epsilon_init      (args.swift_trace_epsilon)→ controls when traces z are culled.
    # - eta_min_init      (args.swift_eta_min)      → lower bound on alpha.
    learner = SwiftSarsaBinaryFeatures(
        extractor.total_features,
        num_actions,
        args.lambda_val,
        args.alpha,
        args.meta_step,
        args.eta,
        args.swift_decay,
        args.swift_trace_epsilon,
        args.swift_eta_min,
    )

    os.makedirs(args.betas_dir, exist_ok=True)

    step = 0
    episode_idx = 0
    prev_action = 0
    prev_reward = 0.0
    done = False
    episode_return = 0.0
    returns = []

    # Log run header
    log(
        f"Starting CartPole: seed={args.seed} decisions={args.decisions} "
        f"alpha={args.alpha} meta_step={args.meta_step} eta={args.eta} lambda={args.lambda_val} "
        f"swift_decay={args.swift_decay} trace_eps={args.swift_trace_epsilon} "
        f"discount={args.discount}"
    )
    log(
        f"Features: total={extractor.total_features} "
        f"(state={extractor.total_state_features}, action={num_actions}, reward=3); "
        f"tilings={extractor.num_tilings}, tiles_per_dim={extractor.tiles_per_dim}"
    )
    log(f"Logging to {log_path}")

    while step < args.decisions:
        if done:
            episode_idx += 1
            returns.append(episode_return)
            window_mean = float(np.mean(returns[-100:])) if returns else 0.0
            log(
                f"Episode {episode_idx}: "
                f"return={episode_return:.1f}, "
                f"mean_last_100={window_mean:.2f}, "
                f"steps_total={step}"
            )
            obs, _ = env.reset()
            extractor.reset()
            learner.reset_episode()
            prev_action = 0
            prev_reward = 0.0
            done = False
            episode_return = 0.0

            # Periodic evaluation (greedy, no learning)
            if eval_env and args.eval_every_episodes > 0 and episode_idx % args.eval_every_episodes == 0:
                eval_returns = []
                for eval_ep in range(args.eval_episodes):
                    eval_seed = args.seed + args.eval_seed_offset + eval_ep
                    eval_obs, _ = eval_env.reset(seed=eval_seed)
                    eval_prev_action = 0
                    eval_prev_reward = 0.0
                    eval_done = False
                    eval_ret = 0.0
                    eval_step = 0
                    while not eval_done:
                        eval_features = extractor.extract(eval_obs, eval_prev_action, eval_prev_reward)
                        eval_q = learner.get_action_values(eval_features)
                        eval_action = int(np.argmax(eval_q))  # greedy
                        eval_obs, eval_r, eval_term, eval_trunc, _ = eval_env.step(eval_action)
                        eval_done = eval_term or eval_trunc
                        eval_ret += float(eval_r)
                        eval_prev_action = eval_action
                        eval_prev_reward = float(eval_r)
                        eval_step += 1
                    eval_returns.append(eval_ret)
                log(
                    f"[Eval @ episode {episode_idx}] "
                    f"mean_return={np.mean(eval_returns):.2f}, "
                    f"max_return={np.max(eval_returns):.2f}, "
                    f"episodes={args.eval_episodes}"
                )

        # Python-side exploration epsilon
        eps = current_epsilon(
            step,
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay_steps,
        )

        # Build binary features using current obs + previous (action, reward)
        features = extractor.extract(obs, prev_action, prev_reward)

        # Swift-Sarsa value function over actions
        q_vals = learner.get_action_values(features)

        # Epsilon-greedy action selection (Python-side)
        action = select_action(q_vals, eps)

        # Step the env
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Discount factor (Python-side)
        gamma = 0.0 if done else args.discount

        # Core Swift-Sarsa update:
        # learn(phi_t, r_{t+1}, gamma, a_t)
        learner.learn(features, reward, gamma, action)
        episode_return += reward

        # Save heatmap periodically
        if args.save_betas_every > 0 and step % args.save_betas_every == 0:
            try:
                import imageio
                betas = np.array(learner.get_feature_betas_max_over_actions())
                alphas = np.exp(-betas) if args.heatmap_inverse_beta else np.exp(betas)
                # Only state features; reshape as [tiling, flat_tiles]
                state_alphas = alphas[:extractor.total_state_features]
                heatmap = state_alphas.reshape(extractor.num_tilings, -1)
                h_min, h_max = heatmap.min(), heatmap.max()
                span = h_max - h_min
                if span > 1e-12:
                    norm_img = (heatmap - h_min) / span
                else:
                    norm_img = np.zeros_like(heatmap)
                imageio.imwrite(
                    os.path.join(args.betas_dir, f"betas_step_{step:07d}.png"),
                    (norm_img * 255).astype(np.uint8),
                )
            except Exception:
                # Heatmaps are purely diagnostic; don't crash training on failure.
                pass

        prev_action = action
        prev_reward = reward
        obs = next_obs
        step += 1

    env.close()
    log_file.close()


if __name__ == "__main__":
    main()
