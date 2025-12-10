#!/usr/bin/env python3
"""
SwiftTD (scalar) trainer on Atari Pong using the existing preprocessing pipeline.

Note: SwiftTD's Python API exposes a scalar value learner. To get action-values,
we instantiate one SwiftTD agent per discrete action and bootstrap with the
next state's predicted value (max over actions). This is a pragmatic wiring for
experimentation; it is not a drop-in replacement for the Swift-Sarsa control
algorithm.
"""

from __future__ import annotations

import argparse
import os
import random
import numpy as np
import gymnasium as gym
from collections import deque
from typing import List, Optional
from shimmy.registration import register_gymnasium_envs

from agent_swift_sarsa import AtariFeatureExtractor


def select_action(q_values: np.ndarray, eps: float, temp: float) -> int:
    """
    Greedy by default. If eps>0, epsilon-greedy. If temp>0 and eps==0, softmax.
    """
    if eps > 0 and np.random.random() < eps:
        return int(np.random.randint(len(q_values)))
    if temp <= 0 or not np.all(np.isfinite(q_values)):
        return int(np.argmax(q_values))

    q_shifted = q_values - np.max(q_values)
    exp_q = np.exp(np.clip(q_shifted / max(temp, 1e-6), -50, 50))
    probs = exp_q / (exp_q.sum() + 1e-10)
    if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
        return int(np.argmax(q_values))
    return int(np.random.choice(len(q_values), p=probs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decisions", type=int, default=100_000)
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--reduce_action_set", action="store_true")
    parser.add_argument("--alpha", type=float, default=1e-7, help="Initial step-size (paper: 1e-7)")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Set >0 only if you want exploration")
    parser.add_argument("--temperature", type=float, default=0.0, help="Softmax temp; 0 means greedy")
    parser.add_argument("--lambda_", type=float, default=0.9)
    parser.add_argument("--eta", type=float, default=0.501, help="Learning rate bound (eta)")
    parser.add_argument("--theta", type=float, default=1e-2, help="Meta step-size")
    parser.add_argument("--decay", type=float, default=0.9, help="Step-size decay")
    parser.add_argument("--eta_min", type=float, default=3.06e-7)
    parser.add_argument("--ema_decay", type=float, default=0.95, help="EMA decay for return tracking")
    parser.add_argument("--video_dir", type=str, default=None, help="If set, save episode videos to this folder")
    parser.add_argument("--log_actions_every", type=int, default=0,
                        help="If >0, log chosen policy_action->ale_action every N steps")
    args = parser.parse_args()

    try:
        import swifttd
    except Exception as e:
        raise SystemExit(f"SwiftTD not available: {e}")

    try:
        register_gymnasium_envs()
    except Exception:
        pass

    # Custom ffmpeg writer (avoids moviepy)
    writer_factory = None
    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)
        try:
            import imageio
            import imageio_ffmpeg  # noqa: F401

            def make_writer(path: str, fps: int = 15):
                return imageio.get_writer(path, fps=fps, codec="libx264", quality=8)

            writer_factory = make_writer
        except Exception as e:
            print(f"Warning: imageio/imageio-ffmpeg not available ({e}); disabling video recording.")
            writer_factory = None

    render_mode = "rgb_array" if writer_factory else None
    env = gym.make(f"ALE/{args.rom}-v5", obs_type="rgb", full_action_space=True, render_mode=render_mode)
    frame, _ = env.reset(seed=args.seed)

    try:
        meanings = env.unwrapped.get_action_meanings()
        print(f"Action meanings: {meanings}")
    except Exception:
        meanings = None

    if args.reduce_action_set and args.rom.lower() == "pong":
        fire_idx = None
        up_idx = None
        down_idx = None
        if meanings:
            for i, m in enumerate(meanings):
                mu = m.upper()
                if fire_idx is None and mu == "FIRE":
                    fire_idx = i
                if up_idx is None and mu == "UP":
                    up_idx = i
                if down_idx is None and mu == "DOWN":
                    down_idx = i
            if up_idx is None:
                for i, m in enumerate(meanings):
                    if "UP" in m.upper():
                        up_idx = i
                        break
            if down_idx is None:
                for i, m in enumerate(meanings):
                    if "DOWN" in m.upper():
                        down_idx = i
                        break
        fire_idx = 1 if fire_idx is None else fire_idx
        up_idx = 2 if up_idx is None else up_idx
        down_idx = 5 if down_idx is None else down_idx
        action_map = np.array([fire_idx, up_idx, down_idx], dtype=np.int64)
        num_actions = len(action_map)
        print(f"Using reduced Pong action set (FIRE/UP/DOWN): {action_map.tolist()}")
    else:
        action_map = None
        num_actions = env.action_space.n
        print(f"Using full action space: {num_actions} actions")

    np.random.seed(args.seed)
    random.seed(args.seed)

    extractor = AtariFeatureExtractor(
        height=105,
        width=80,
        num_bins=8,
        num_actions=num_actions,
        use_grayscale=False,
        use_frame_diff=False,
        K_actions=0,
        K_rewards=0,
        use_reward_gap=False,
    )
    base_features = extractor.total_features
    active_per_frame = extractor.get_active_feature_count()
    tau = args.alpha * active_per_frame
    print(
        f"Feature stats: total={base_features}, active_per_frame≈{active_per_frame}, "
        f"alpha={args.alpha}, alpha*active≈{tau:.3e}"
    )

    # One SwiftTD agent per action
    agents: List = []
    for _ in range(num_actions):
        agents.append(
            swifttd.SwiftTDBinaryFeatures(
                num_of_features=base_features,
                lambda_=args.lambda_,
                alpha=args.alpha,
                gamma=0.99,
                epsilon=1e-5,
                eta=args.eta,
                decay=args.decay,
                meta_step_size=args.theta,
                eta_min=args.eta_min,
            )
        )

    extractor.reset()
    writer = None  # type: Optional[any]
    episode_idx = 0

    def close_writer():
        nonlocal writer
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
            writer = None

    def maybe_write_frame():
        nonlocal writer
        if writer_factory is None:
            return
        frame_rgb = env.render()  # returns HxWx3
        if frame_rgb is None:
            return
        if writer is None:
            video_path = os.path.join(args.video_dir, f"episode_{episode_idx:05d}.mp4")
            writer = writer_factory(video_path, fps=int(60 / args.frame_skip))
        writer.append_data(frame_rgb)

    maybe_write_frame()

    features = extractor.extract(frame)
    q_values = np.array([agent.predict(features) for agent in agents])
    if action_map is not None:
        action = 0  # FIRE to serve ball at episode start
    else:
        action = select_action(q_values, args.epsilon, args.temperature)
    extractor.set_prev_action(action)

    episode_reward = 0.0
    reward_ema = None
    global_step = 0

    returns = []
    ema_series = []

    while global_step < args.decisions:
        ale_action = int(action_map[action]) if action_map is not None else action

        if args.log_actions_every and (global_step % args.log_actions_every == 0):
            if meanings and 0 <= ale_action < len(meanings):
                meaning_str = meanings[ale_action]
            else:
                meaning_str = "n/a"
            print(f"act_log step {global_step}: policy_action={action} -> ale_action={ale_action} ({meaning_str})")

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
        maybe_write_frame()

        reward = np.clip(total_r, -1.0, 1.0)
        gamma = 0.0 if (done or trunc) else 0.99

        extractor.update_reward(reward)

        # Bootstrap target: max predicted value of next state
        next_features = extractor.extract(frame)
        next_q = np.array([agent.predict(next_features) for agent in agents])
        bootstrap = 0.0 if gamma == 0.0 else float(next_q.max())

        # Set v_old to bootstrap target and step with reward
        agents[action].set_v_old(bootstrap)
        agents[action].step(features, reward)

        # Select next action
        action_next = select_action(next_q, args.epsilon, args.temperature)
        extractor.set_prev_action(action_next)

        features = next_features
        action = action_next

        episode_reward += total_r
        global_step += 1

        if done or trunc:
            close_writer()
            reward_ema = episode_reward if reward_ema is None else args.ema_decay * reward_ema + (1 - args.ema_decay) * episode_reward
            returns.append(episode_reward)
            ema_series.append(reward_ema)
            print(f"ep {episode_idx} reward {episode_reward:.1f} ema {reward_ema:.2f} step {global_step}")

            episode_idx += 1
            episode_reward = 0.0

            frame, _ = env.reset()
            maybe_write_frame()
            extractor.reset()
            features = extractor.extract(frame)
            q_values = np.array([agent.predict(features) for agent in agents])
            action = select_action(q_values, args.epsilon, args.temperature)
            extractor.set_prev_action(action)

    print(f"\nTraining finished. Episodes: {episode_idx}")
    close_writer()

    if returns:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 4))
            plt.plot(returns, label="return")
            plt.plot(ema_series, label=f"ema (decay={args.ema_decay})")
            plt.xlabel("episode")
            plt.ylabel("return")
            plt.title("Pong SwiftTD (binary)")
            plt.legend()
            plt.tight_layout()
            plt.savefig("pong_swifttd_run.png")
            print("Saved plot to pong_swifttd_run.png")
        except Exception as e:
            print(f"Could not save plot: {e}")


if __name__ == "__main__":
    main()
