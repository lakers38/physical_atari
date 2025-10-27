#!/usr/bin/env python3
"""
Train SwiftTD agent in Gymnasium with optional hardware latency simulation

This script trains a SwiftTD Q-learning agent on Atari games with two modes:
- sim: Pure simulation (no latency) - fast baseline training
- sim_lat: Simulation with LatencyModel - simulates real hardware delays

SwiftTD uses 18 separate learners (one per action) for Q-learning with
adaptive step sizes and eligibility traces.
"""

import argparse
import os
import sys
from datetime import datetime
import time

import gymnasium as gym
import ale_py
import numpy as np
from coolname import generate_slug

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from agent import SwiftTDAgent
from environment import create_swifttd_env, create_eval_env
import config as swifttd_config

# WandB integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Register ALE environments
gym.register_envs(ale_py)


def linear_schedule(initial_value, final_value, decay_steps):
    """
    Create a linear schedule for epsilon decay.

    Args:
        initial_value: Initial value
        final_value: Final value
        decay_steps: Number of steps to decay from initial to final

    Returns:
        Function that takes current step and returns scheduled value
    """
    def schedule(step):
        if step >= decay_steps:
            return final_value
        else:
            return initial_value + (final_value - initial_value) * (step / decay_steps)
    return schedule


def evaluate_agent(agent, eval_env, num_episodes=10, epsilon=0.001):
    """
    Evaluate agent for a fixed number of episodes.

    Args:
        agent: SwiftTDAgent
        eval_env: Evaluation environment
        num_episodes: Number of episodes to evaluate
        epsilon: Exploration rate during evaluation

    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []

    for _ in range(num_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action = agent.select_action(obs, epsilon=epsilon)
            obs, reward, done, info = eval_env.step([action])
            episode_reward += reward[0]
            episode_length += 1

            if done[0]:
                done = True

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
    }


def train_agent(
    env_name="ALE/MsPacman-v5",
    total_timesteps=1000000,
    simulate_latency=False,
    latency_model_dir=None,
    experiment_dir=None,
    device="cuda",
    use_wandb=False,
    wandb_project="physical-atari",
    wandb_entity=None,
    wandb_run_name=None,
    seed=0,
    record_videos=True,
    video_freq=50000,
    video_length=500
):
    """
    Train SwiftTD agent with optional latency simulation.

    Args:
        env_name: Atari environment name
        total_timesteps: Total training timesteps
        simulate_latency: If True, apply LatencyModel
        latency_model_dir: Directory with LatencyModel weights
        experiment_dir: Directory to save results
        device: "cuda" or "cpu"
        use_wandb: If True, log to Weights & Biases
        wandb_project: WandB project name
        wandb_entity: WandB entity/team name
        wandb_run_name: WandB run name
        seed: Random seed
        record_videos: If True, record gameplay videos
        video_freq: Record video every N steps
        video_length: Number of frames per video

    Returns:
        Trained agent and save path
    """
    # Initialize WandB if requested
    wandb_run = None
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: WandB not installed. Install with: pip install wandb")
            print("Continuing without WandB logging...")
        else:
            # Create config dict for WandB
            config = {
                "env_name": env_name,
                "total_timesteps": total_timesteps,
                "simulate_latency": simulate_latency,
                "device": device,
                "algorithm": "SwiftTD",
                "training_mode": "sim_lat" if simulate_latency else "sim",
                "latency_enabled": simulate_latency,
                # SwiftTD hyperparameters
                "num_features": swifttd_config.num_features,
                "lambda": swifttd_config.lambda_,
                "initial_alpha": swifttd_config.initial_alpha,
                "gamma": swifttd_config.gamma,
                "max_step_size": swifttd_config.max_step_size,
                "step_size_decay": swifttd_config.step_size_decay,
                "meta_step_size": swifttd_config.meta_step_size,
                "epsilon_start": swifttd_config.epsilon_start,
                "epsilon_end": swifttd_config.epsilon_end,
                "epsilon_decay_steps": swifttd_config.epsilon_decay_steps,
                "cnn_learning_rate": swifttd_config.cnn_learning_rate,
                "n_stack": swifttd_config.n_stack,
                "seed": seed,
            }

            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config=config,
                sync_tensorboard=True,  # Auto-upload TensorBoard metrics
                monitor_gym=True,       # Auto-upload videos
                save_code=True,
            )
            print(f"[WandB] Initialized run: {wandb_run.name}")
            print(f"[WandB] View at: {wandb_run.url}")

    # Set random seeds
    np.random.seed(seed)

    # Create environments
    monitor_path = os.path.join(experiment_dir, "logs", "monitor") if experiment_dir else None
    video_path = os.path.join(experiment_dir, "videos") if experiment_dir and record_videos else None

    print("\nCreating training environment...")
    env = create_swifttd_env(
        env_name,
        n_stack=swifttd_config.n_stack,
        seed=seed,
        simulate_latency=simulate_latency,
        latency_model_dir=latency_model_dir,
        monitor_path=monitor_path,
        video_path=video_path,
        record_video=record_videos,
        video_freq=video_freq,
        video_length=video_length
    )

    print("\nCreating evaluation environment...")
    eval_env = create_eval_env(
        env_name,
        n_stack=swifttd_config.n_stack,
        seed=seed + 100,
        simulate_latency=simulate_latency,
        latency_model_dir=latency_model_dir
    )

    # Create agent
    print("\nInitializing SwiftTD agent...")
    agent = SwiftTDAgent(
        num_actions=env.action_space.n,
        num_features=swifttd_config.num_features,
        n_stack=swifttd_config.n_stack,
        device=device,
        lambda_=swifttd_config.lambda_,
        initial_alpha=swifttd_config.initial_alpha,
        gamma=swifttd_config.gamma,
        eps=swifttd_config.eps,
        max_step_size=swifttd_config.max_step_size,
        step_size_decay=swifttd_config.step_size_decay,
        meta_step_size=swifttd_config.meta_step_size,
        eta_min=swifttd_config.eta_min,
        cnn_learning_rate=swifttd_config.cnn_learning_rate,
        cnn_update_frequency=swifttd_config.cnn_update_frequency,
        gradient_clip=swifttd_config.gradient_clip,
        epsilon=swifttd_config.epsilon_start
    )

    # Epsilon schedule
    epsilon_schedule = linear_schedule(
        swifttd_config.epsilon_start,
        swifttd_config.epsilon_end,
        swifttd_config.epsilon_decay_steps
    )

    # Training loop
    mode_name = "sim_lat (with LatencyModel)" if simulate_latency else "sim (no latency)"
    print(f"\n{'='*60}")
    print(f"Starting Training: {mode_name}")
    print(f"{'='*60}")
    print(f"Environment: {env_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Latency simulation: {simulate_latency}")
    print(f"Device: {device}")
    if use_wandb and wandb_run:
        print(f"WandB: {wandb_run.url}")
    print(f"{'='*60}\n")

    obs = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    start_time = time.time()

    for step in range(total_timesteps):
        # Update epsilon
        current_epsilon = epsilon_schedule(step)
        agent.set_epsilon(current_epsilon)

        # Select action
        action = agent.select_action(obs, epsilon=current_epsilon)

        # Take step
        next_obs, reward, done, info = env.step([action])

        # Update agent
        agent.update(obs, action, reward[0], next_obs, done[0])

        episode_reward += reward[0]
        episode_length += 1

        # Handle episode end
        if done[0]:
            episode_count += 1

            # Log episode metrics
            if step % swifttd_config.log_frequency == 0 or True:
                elapsed_time = time.time() - start_time
                fps = (step + 1) / elapsed_time

                print(f"Step {step+1}/{total_timesteps} | "
                      f"Episode {episode_count} | "
                      f"Reward: {episode_reward:.1f} | "
                      f"Length: {episode_length} | "
                      f"Epsilon: {current_epsilon:.3f} | "
                      f"FPS: {fps:.1f}")

                if use_wandb and wandb_run:
                    wandb.log({
                        "episode/reward": episode_reward,
                        "episode/length": episode_length,
                        "episode/count": episode_count,
                        "train/epsilon": current_epsilon,
                        "train/fps": fps,
                        "train/step": step + 1,
                    })

            # Reset for next episode
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
        else:
            obs = next_obs

        # Evaluation
        if (step + 1) % swifttd_config.eval_frequency == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at step {step+1}")
            print(f"{'='*60}")

            eval_metrics = evaluate_agent(
                agent,
                eval_env,
                num_episodes=swifttd_config.eval_episodes,
                epsilon=swifttd_config.eval_epsilon
            )

            print(f"Mean reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            print(f"Min/Max: {eval_metrics['min_reward']:.1f} / {eval_metrics['max_reward']:.1f}")
            print(f"Mean length: {eval_metrics['mean_length']:.1f}")
            print(f"{'='*60}\n")

            if use_wandb and wandb_run:
                wandb.log({
                    "eval/mean_reward": eval_metrics['mean_reward'],
                    "eval/std_reward": eval_metrics['std_reward'],
                    "eval/min_reward": eval_metrics['min_reward'],
                    "eval/max_reward": eval_metrics['max_reward'],
                    "eval/mean_length": eval_metrics['mean_length'],
                    "train/step": step + 1,
                })

        # Save checkpoint
        if (step + 1) % swifttd_config.save_frequency == 0:
            if experiment_dir:
                checkpoint_path = os.path.join(
                    experiment_dir,
                    "checkpoints",
                    f"checkpoint_{step+1}"
                )
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                agent.save(checkpoint_path)

    # Save final model
    if experiment_dir:
        final_model_path = os.path.join(experiment_dir, "final_model")
        agent.save(final_model_path)
        print(f"\nTraining complete! Model saved to: {final_model_path}")
    else:
        final_model_path = None
        print(f"\nTraining complete!")

    # Close environments
    env.close()
    eval_env.close()

    # Finish WandB run
    if use_wandb and wandb_run is not None:
        wandb_run.finish()
        print(f"[WandB] Run finished and uploaded")

    return agent, final_model_path


def main():
    parser = argparse.ArgumentParser(
        description="Train SwiftTD agent in Gymnasium simulation with optional latency"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="ALE/MsPacman-v5",
        help="Atari environment name (default: ALE/MsPacman-v5)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000000,
        help="Total training timesteps (default: 1M)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sim",
        choices=["sim", "sim_lat"],
        help="Training mode: sim (no latency) or sim_lat (with LatencyModel)"
    )
    parser.add_argument(
        "--latency-model-dir",
        type=str,
        default="./latency_wrap",
        help="Directory containing LatencyModel weights"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/swifttd/",
        help="Base directory for outputs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Disable video recording (default: videos enabled)"
    )
    parser.add_argument(
        "--video-freq",
        type=int,
        default=50000,
        help="Record video every N steps (default: 50000)"
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=500,
        help="Number of frames per video (default: 500)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="physical-atari",
        help="WandB project name (default: physical-atari)"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity/team name (default: your username)"
    )
    args = parser.parse_args()

    # Generate run name (mode-name-timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}-swifttd-{args.mode}-{generate_slug(2)}"

    # Create experiment directory using run_name
    env_dir_name = args.env.replace('/', '_')
    experiment_dir = os.path.join(args.output_dir, args.mode, env_dir_name, run_name)

    # Create directory structure
    os.makedirs(os.path.join(experiment_dir, "logs", "monitor"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    if not args.no_videos:
        os.makedirs(os.path.join(experiment_dir, "videos"), exist_ok=True)

    # Save configuration
    config_path = os.path.join(experiment_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"Run name: {run_name}\n")
        f.write(f"Algorithm: SwiftTD Q-Learning\n")
        f.write(f"Training mode: {args.mode}\n")
        f.write(f"Environment: {args.env}\n")
        f.write(f"Total timesteps: {args.timesteps}\n")
        f.write(f"Latency simulation: {args.mode == 'sim_lat'}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"\nSwiftTD Hyperparameters:\n")
        f.write(f"Lambda: {swifttd_config.lambda_}\n")
        f.write(f"Initial alpha: {swifttd_config.initial_alpha}\n")
        f.write(f"Gamma: {swifttd_config.gamma}\n")
        f.write(f"Max step size: {swifttd_config.max_step_size}\n")
        f.write(f"Step size decay: {swifttd_config.step_size_decay}\n")
        f.write(f"Meta step size: {swifttd_config.meta_step_size}\n")
        f.write(f"Epsilon start: {swifttd_config.epsilon_start}\n")
        f.write(f"Epsilon end: {swifttd_config.epsilon_end}\n")
        f.write(f"Epsilon decay steps: {swifttd_config.epsilon_decay_steps}\n")
        f.write(f"Record videos: {not args.no_videos}\n")
        if not args.no_videos:
            f.write(f"Video frequency: {args.video_freq} steps\n")
            f.write(f"Video length: {args.video_length} frames\n")

    # Train agent
    agent, model_path = train_agent(
        env_name=args.env,
        total_timesteps=args.timesteps,
        simulate_latency=(args.mode == "sim_lat"),
        latency_model_dir=args.latency_model_dir,
        experiment_dir=experiment_dir,
        device=args.device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=run_name,
        seed=args.seed,
        record_videos=not args.no_videos,
        video_freq=args.video_freq,
        video_length=args.video_length
    )

    print(f"\n{'='*60}")
    print(f"Training completed successfully!")
    print(f"{'='*60}")
    if model_path:
        print(f"Model saved at: {model_path}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"\nNext step: Transfer to physical hardware")
    print(f"  python harness_physical.py \\")
    print(f"    --agent_type=agent_swifttd \\")
    if model_path:
        print(f"    --load_model={model_path} \\")
    print(f"    --total_frames=500000")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
