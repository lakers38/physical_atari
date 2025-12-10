#!/usr/bin/env python3
"""
Simulation script for Neural Swift-SARSA agent.
Uses the Agent class from agent_swift_sarsa.py to verify the neural pipeline.
"""

import argparse
import numpy as np
import gymnasium as gym
import cv2
import os
import random
from datetime import datetime
import torch

from shimmy.registration import register_gymnasium_envs
from agent_swift_sarsa import Agent


def resolve_action_set(env, action_set_choice: str):
    """
    Returns (action_map, num_actions, description, action_meanings).
    action_map is None when using the environment's native action space.
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
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--alpha", type=float, default=1e-4, help="Alpha for neural (usually higher than binary)")
    parser.add_argument("--meta_step", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=0.03)
    parser.add_argument("--lambda_val", type=float, default=0.95)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_steps", type=int, default=1_000_000)
    parser.add_argument("--swift_decay", type=float, default=0.999)
    parser.add_argument("--swift_trace_epsilon", type=float, default=1e-6)
    parser.add_argument("--swift_eta_min", type=float, default=1e-7)
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--use_neural_features", action="store_true", default=True)
    parser.add_argument("--no_neural", dest="use_neural_features", action="store_false")
    parser.add_argument("--neural_device", type=str, default="cpu")
    
    # Logging / playback
    parser.add_argument("--log_dir", type=str, default="logs_neural")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--video_every", type=int, default=10)
    parser.add_argument("--action_set", choices=["minimal", "full"], default="minimal")
    parser.add_argument("--disable_frame_diff", action="store_true",
                        help="Turn off frame differencing in binary extractor (paper-style).")
    parser.add_argument("--imprinting", action="store_true",
                        help="Enable action/reward history (disabled by default).")
    
    args = parser.parse_args()

    def _fmt_val(v):
        return str(v).replace(".", "p")

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
    
    # Generate log filename similar to paper script
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_filename = (
        f"neural_{args.rom}_seed{args.seed}"
        f"_dec{args.decisions}"
        f"_alpha{_fmt_val(args.alpha)}"
        f"_gamma{_fmt_val(args.discount)}"
        f"_lambda{_fmt_val(args.lambda_val)}"
        f"_eps{_fmt_val(args.epsilon_end)}"
        f"_{timestamp}.txt"
    )
    log_path = os.path.join(args.log_dir, log_filename)
    
    # Open file in buffered mode
    log_file = open(log_path, "w", buffering=1)

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    log(f"Logging to {log_path}")
    log(f"--- Configuration ---")
    log(f"ROM: {args.rom}")
    log(f"Device: {args.neural_device}")
    log(f"Neural Features: {args.use_neural_features}")
    log(f"Seed: {args.seed}")
    log(f"Alpha: {args.alpha}")
    log(f"Meta-step: {args.meta_step}")
    log(f"Eta: {args.eta}")
    log(f"Gamma: {args.discount}")
    log(f"Lambda: {args.lambda_val}")
    log(f"Epsilon: start={args.epsilon_start} end={args.epsilon_end} decay_steps={args.epsilon_decay_steps}")
    log(f"Frame skip: {args.frame_skip}")
    log(f"Imprinting: {args.imprinting}")
    log(f"---------------------")
    
    # Register envs
    try:
        register_gymnasium_envs()
    except Exception:
        pass

    # Create Env
    render_mode = "rgb_array" if args.video_dir else None
    env = gym.make(f"ALE/{args.rom}-v5", obs_type="rgb", full_action_space=True, render_mode=render_mode)
    action_map, num_actions, action_desc, action_meanings = resolve_action_set(env, args.action_set)
    log(action_desc)
    if action_map is None and action_meanings:
        log(f"Action meanings: {action_meanings}")

    # Create Agent
    # Disable imprinting-style extras by default (action/reward history + gap)
    K_actions = 3 if args.imprinting else 0
    K_rewards = 3 if args.imprinting else 0
    use_reward_gap = bool(args.imprinting)
    gap_bins = 6 if args.imprinting else 0

    agent = Agent(
        num_actions=num_actions,
        seed=args.seed,
        alpha_init=args.alpha,
        gamma=args.discount,
        lambda_=args.lambda_val,
        exploration_epsilon=args.epsilon_start,
        theta=args.meta_step,
        eta=args.eta,
        decay=args.swift_decay,
        epsilon=args.swift_trace_epsilon,
        eta_min=args.swift_eta_min,
        use_neural_features=args.use_neural_features,
        neural_device=args.neural_device,
        use_frame_diff=not args.disable_frame_diff,
        K_actions=K_actions,
        K_rewards=K_rewards,
        use_reward_gap=use_reward_gap,
        gap_bins=gap_bins,
    )

    # Align epsilon schedule with CLI
    agent.core.initial_epsilon = args.epsilon_start
    agent.core.epsilon = args.epsilon_start
    agent.core.min_epsilon = args.epsilon_end
    agent.core.epsilon_decay_frames = args.epsilon_decay_steps

    if agent.core.use_neural_features:
        total_features = agent.core.neural_extractors[0].feat_dim
        active_features = total_features
    else:
        total_features = agent.core.extractors[0].total_features
        active_features = agent.core.extractors[0].get_active_feature_count()
    log(f"Feature dim: total={total_features}, active≈{active_features}")

    step = 0
    episode_idx = 0
    returns = []

    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)

    while step < args.decisions:
        obs, _ = env.reset(seed=args.seed + episode_idx)
        done = False
        truncated = False
        episode_reward = 0.0
        
        writer = None
        if args.video_dir and (episode_idx % args.video_every == 0):
            import imageio
            video_path = os.path.join(args.video_dir, f"ep_{episode_idx:04d}.mp4")
            writer = imageio.get_writer(video_path, fps=60)
        
        # First action (no reward yet)
        action = agent.frame(obs, 0.0, 0)
        ale_action = action if action_map is None else action_map[action]
        
        while not (done or truncated):
            total_reward = 0.0
            frames_used = 0

            for _ in range(max(1, args.frame_skip)):
                if writer:
                    frame_rgb = env.render()
                    if frame_rgb is not None:
                        writer.append_data(frame_rgb)
                
                next_obs, reward, done, truncated, _ = env.step(ale_action)
                frames_used += 1
                episode_reward += reward
                total_reward += np.clip(reward, -1, 1)

                if done or truncated:
                    break
            
            # Agent step on accumulated reward
            is_terminal = 1 if (done or truncated) else 0
            action = agent.frame(next_obs, total_reward, is_terminal)
            ale_action = action if action_map is None else action_map[action]
            
            obs = next_obs
            step += frames_used
            
            if step >= args.decisions:
                break
        
        if writer:
            writer.close()
            
        returns.append(episode_reward)
        mean_100 = np.mean(returns[-100:]) if returns else 0.0
        log(f"Ep {episode_idx}: Reward={episode_reward:.1f}, Mean100={mean_100:.1f}, Steps={step}")
        
        episode_idx += 1

    env.close()
    log_file.close()
    log(f"Finished {args.decisions} steps.")

if __name__ == "__main__":
    main()
