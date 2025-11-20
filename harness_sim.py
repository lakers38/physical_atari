#!/usr/bin/env python3
"""
Simulation harness for custom agent_ppo.py

Runs your custom PPO agent in Gymnasium Atari environments for fast iteration.

Usage:
    python harness_sim.py --agent_type=agent_ppo --game=MsPacman --total_frames=500000
"""

import argparse
import datetime
import logging
import os
import time

import gymnasium as gym
import numpy as np
import ale_py

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from framework.Logger import logger

# Register ALE environments
gym.register_envs(ale_py)


def make_env(game_name, reduce_action_set=2, render_mode=None):
    """Create Atari environment with action set reduction"""
    # Create base environment
    env = gym.make(f"ALE/{game_name}-v5", render_mode=render_mode, frameskip=1)

    # Apply action set reduction (matching physical setup)
    if reduce_action_set == 2:
        # 4 directional actions: UP, DOWN, LEFT, RIGHT
        # ALE action indices: UP=2, DOWN=5, LEFT=4, RIGHT=3
        action_mapping = [2, 5, 4, 3]
        env = ActionSetWrapper(env, action_mapping)
        logger.info(f"Action space reduced to 4 directional actions: {action_mapping}")

    return env


class ActionSetWrapper(gym.Wrapper):
    """Wrapper to restrict action space to a subset"""
    def __init__(self, env, action_mapping):
        super().__init__(env)
        self.action_mapping = action_mapping
        self.action_space = gym.spaces.Discrete(len(action_mapping))

    def step(self, action):
        # Map restricted action to full action space
        full_action = self.action_mapping[action]
        return self.env.step(full_action)


def main(args):
    logger.setLevel(getattr(logging, args.log_level))

    # Setup experiment directory
    experiment_name = f"{args.game}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_dir = os.path.join(args.results_dir, experiment_name)
    os.makedirs(data_dir, exist_ok=True)

    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Results dir: {data_dir}")

    # Import agent
    logger.info(f"Importing agent: {args.agent_type}")
    if args.agent_type == 'agent_ppo':
        from algorithms.ppo.agent_ppo import Agent
    else:
        raise ValueError(f"Invalid agent type={args.agent_type}")

    # Create environment
    env = make_env(args.game, reduce_action_set=args.reduce_action_set)
    num_actions = env.action_space.n
    logger.info(f"Game: {args.game}, Actions: {num_actions}")

    # Initialize wandb if requested (MUST happen before agent creation)
    if args.use_wandb and WANDB_AVAILABLE:
        config = vars(args).copy()
        config.update({
            "agent_type": args.agent_type,
            "game": args.game,
            "total_frames": args.total_frames,
            "algorithm": "PPO",
            "training_mode": "sim",
        })
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=experiment_name,
            config=config,
            sync_tensorboard=False,
            save_code=True,
        )
        logger.info(f"Wandb initialized: {wandb_run.name}")
        logger.info(f"View at: {wandb_run.url}")

    # Create agent
    agent_kwargs = {
        'use_wandb': args.use_wandb and WANDB_AVAILABLE,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'ent_coef_initial': args.ent_coef_initial,
        'ent_coef_final': args.ent_coef_final,
        'ent_coef_decay_steps': args.ent_coef_decay_steps,
    }

    agent = Agent(
        data_dir=data_dir,
        seed=args.seed,
        num_actions=num_actions,
        total_frames=args.total_frames,
        **agent_kwargs
    )

    logger.info("Starting training...")

    # Training loop
    total_steps = 0
    episode_num = 0
    episode_reward = 0.0
    episode_length = 0
    start_time = time.time()

    # Track rolling statistics
    episode_rewards = []
    episode_lengths = []

    obs, info = env.reset(seed=args.seed)
    reward = 0.0
    end_of_episode = False

    while total_steps < args.total_frames:
        time.sleep(.016)
        # Get action from agent
        action = agent.frame(obs, reward, end_of_episode)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_steps += 1
        episode_reward += reward
        episode_length += 1

        # Handle episode end
        if done:
            episode_num += 1
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            elapsed_time = time.time() - start_time
            fps = total_steps / elapsed_time if elapsed_time > 0 else 0

            # Calculate rolling averages (last 100 episodes)
            recent_rewards = episode_rewards[-100:]
            recent_lengths = episode_lengths[-100:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_length = sum(recent_lengths) / len(recent_lengths)

            logger.info(
                f"Episode {episode_num}: "
                f"reward={episode_reward:.1f}, "
                f"length={episode_length}, "
                f"steps={total_steps}/{args.total_frames}, "
                f"fps={fps:.1f}, "
                f"avg_reward={avg_reward:.1f}"
            )

            # Log to wandb
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "rollout/ep_rew_mean": avg_reward,
                    "rollout/ep_len_mean": avg_length,
                    "episode/reward": episode_reward,
                    "episode/length": episode_length,
                    "episode/number": episode_num,
                    "time/fps": fps,
                    "time/total_timesteps": total_steps,
                }, step=total_steps)

            # Reset for next episode
            obs, info = env.reset()
            end_of_episode = True
            episode_reward = 0
            episode_length = 0
        else:
            end_of_episode = False

    # Save final model
    if args.save_model:
        model_path = os.path.join(data_dir, "final_model")
        agent.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

    env.close()

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train agent in Atari simulation")

    # Environment args
    parser.add_argument("--game", type=str, default="MsPacman", help="Atari game name (e.g., MsPacman, Qbert)")
    parser.add_argument("--reduce_action_set", type=int, default=2, help="0=full, 1=minimal, 2=4 directional")

    # Agent args
    parser.add_argument("--agent_type", type=str, default="agent_ppo", help="Agent type to use")
    parser.add_argument("--total_frames", type=int, default=500000, help="Total training frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # PPO hyperparameters
    parser.add_argument("--n_steps", type=int, default=128, help="Rollout buffer size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--n_epochs", type=int, default=4, help="Number of training epochs per update")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ent_coef_initial", type=float, default=0.005, help="Initial entropy coefficient")
    parser.add_argument("--ent_coef_final", type=float, default=0.001, help="Final entropy coefficient")
    parser.add_argument("--ent_coef_decay_steps", type=int, default=50000, help="Steps to decay entropy coefficient")

    # Logging args
    parser.add_argument("--results_dir", type=str, default="results_sim", help="Directory for results")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--use_wandb", type=int, default=1, help="Use wandb logging (0=no, 1=yes)")
    parser.add_argument("--wandb_project", type=str, default="physical_atari_sim", help="Wandb project name")

    # Model saving
    parser.add_argument("--save_model", type=int, default=1, help="Save model at end (0=no, 1=yes)")

    args = parser.parse_args()
    main(args)
