#!/usr/bin/env python3
"""
Train R2D2 agent using distributed multi-actor architecture

This script trains an R2D2 agent on Atari games using torch.multiprocessing
with multiple parallel actors for efficient data collection and training.
"""

import argparse
import os
import random
import torch.multiprocessing as mp
from datetime import datetime

import gymnasium as gym
import ale_py
import numpy as np
import torch
from coolname import generate_slug


# Import R2D2 components
# Note: Using absolute imports since this is run as a script
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from model import Network
import config as r2d2_config
from actor import Actor
from learner import Learner
from replay_buffer import ReplayBuffer
from environment import create_env

# Register ALE environments
gym.register_envs(ale_py)

# WandB integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def load_checkpoint(checkpoint_path, model, device='cpu'):
    """
    Load checkpoint from file.

    Args:
        checkpoint_path: Path to checkpoint .pth file
        model: Model to load state dict into
        device: Device to load checkpoint on

    Returns:
        Dictionary containing checkpoint data (num_updates, env_steps, etc.)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both old tuple format and new dict format
    if isinstance(checkpoint, tuple):
        # Old format: (state_dict, num_updates, env_steps, training_time)
        state_dict, num_updates, env_steps, training_time = checkpoint
        model.load_state_dict(state_dict)
        print(f"✓ Loaded checkpoint (old format)")
        print(f"  - num_updates: {num_updates}")
        print(f"  - env_steps: {env_steps}")
        print(f"  - training_time: {training_time:.2f} minutes")
        return {
            'num_updates': num_updates,
            'env_steps': env_steps,
            'training_time_minutes': training_time
        }
    elif isinstance(checkpoint, dict):
        # New format: dictionary with keys
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint (new format)")
        print(f"  - num_updates: {checkpoint.get('num_updates', 0)}")
        print(f"  - env_steps: {checkpoint.get('env_steps', 0)}")
        if 'training_time_minutes' in checkpoint:
            print(f"  - training_time: {checkpoint['training_time_minutes']:.2f} minutes")
        return checkpoint
    else:
        # Just a state dict
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded checkpoint (state dict only)")
        return {'num_updates': 0, 'env_steps': 0}


def train_agent_distributed(
    env_name,
    total_timesteps,
    num_actors,
    experiment_dir,
    device,
    use_wandb,
    wandb_project,
    wandb_entity,
    wandb_run_name,
    seed,
    simulate_latency=False,
    latency_model_dir="./latency_wrap",
    load_model_path=None
):
    """
    Train R2D2 agent using distributed multi-actor architecture

    Args:
        env_name: Atari environment name
        total_timesteps: Total training timesteps
        num_actors: Number of parallel actor threads
        experiment_dir: Directory to save results
        device: "cuda" or "cpu"
        use_wandb: If True, log to Weights & Biases
        wandb_project: WandB project name
        wandb_entity: WandB entity/team name
        wandb_run_name: WandB run name
        seed: Random seed
        simulate_latency: If True, apply LatencyModel wrapper
        latency_model_dir: Directory containing LatencyModel weights
        load_model_path: Path to checkpoint to resume training

    Returns:
        Trained model and save path
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)

    mode_name = "sim_lat (with LatencyModel)" if simulate_latency else "sim (no latency)"
    print(f"\n{'='*60}")
    print(f"Starting Distributed R2D2 Training: {mode_name}")
    print(f"{'='*60}")
    print(f"Environment: {env_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Number of actors: {num_actors}")
    print(f"Device: {device}")
    print(f"Latency simulation: {simulate_latency}")
    if load_model_path:
        print(f"Loading checkpoint: {load_model_path}")
    print(f"{'='*60}\n")

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config={
                'algorithm': 'R2D2',
                'env_name': env_name,
                'total_timesteps': total_timesteps,
                'num_actors': num_actors,
                'device': device,
                'seed': seed,
                'training_mode': 'sim_lat' if simulate_latency else 'sim',
                'latency_enabled': simulate_latency,
                'latency_model_dir': latency_model_dir if simulate_latency else None,
                'resume_from_checkpoint': load_model_path is not None,
                # R2D2 hyperparameters
                'learning_rate': r2d2_config.lr,
                'gamma': r2d2_config.gamma,
                'grad_norm': r2d2_config.grad_norm,
                'batch_size': r2d2_config.batch_size,
                'buffer_capacity': r2d2_config.buffer_capacity,
                'learning_starts': r2d2_config.learning_starts,
                'block_length': r2d2_config.block_length,
                'burn_in_steps': r2d2_config.burn_in_steps,
                'learning_steps': r2d2_config.learning_steps,
                'forward_steps': r2d2_config.forward_steps,
                'prio_exponent': r2d2_config.prio_exponent,
                'importance_sampling_exponent': r2d2_config.importance_sampling_exponent,
                'base_explore_eps': r2d2_config.base_explore_eps,
                'alpha': r2d2_config.alpha,
                'hidden_dim': r2d2_config.hidden_dim,
                'target_net_update_interval': r2d2_config.target_net_update_interval,
            }
        )
        print(f"✓ WandB initialized (project: {wandb_project}, run: {wandb_run_name})\n")

    # Create a test environment to get action dimension
    test_env = create_env(env_name=env_name, noop_start=True)
    action_dim = test_env.action_space.n
    test_env.close()

    # Create shared model
    shared_model = Network(action_dim)
    shared_model.share_memory()  # Enable weight sharing across processes

    # Load checkpoint if provided
    initial_num_updates = 0
    if load_model_path:
        checkpoint_data = load_checkpoint(load_model_path, shared_model, device=device)
        initial_num_updates = checkpoint_data.get('num_updates', 0)
        print(f"✓ Resuming training from update {initial_num_updates}\n")

    # Create communication queues
    sample_queue_list = [mp.Queue() for _ in range(num_actors)]
    batch_queue = mp.Queue(8)
    priority_queue = mp.Queue(8)
    stats_queue = mp.Queue(8)  # For ReplayBuffer stats → Learner

    # Create epsilon schedule for actors (diverse exploration)
    # Formula: eps = base_eps ** (1 + (i / (num_actors - 1)) * alpha)
    base_eps = r2d2_config.base_explore_eps
    alpha = r2d2_config.alpha
    epsilons = [
        base_eps ** (
            1 + (i / (num_actors - 1) * alpha if num_actors > 1 else 0)
        )
        for i in range(num_actors)
    ]

    print(f"Actor eps (exploration) values: {[f'{eps:.3f}' for eps in epsilons]}")

    # Create environment factory function
    def env_factory():
        return create_env(
            env_name=env_name,
            noop_start=True,
            simulate_latency=simulate_latency,
            latency_model_dir=latency_model_dir
        )

    # Create ReplayBuffer
    replay_buffer = ReplayBuffer(
        sample_queue_list=sample_queue_list,
        batch_queue=batch_queue,
        priority_queue=priority_queue,
        stats_queue=stats_queue,
        buffer_capacity=r2d2_config.buffer_capacity,
        batch_size=r2d2_config.batch_size
    )

    # Create Learner
    video_dir = os.path.join(experiment_dir, "videos") if experiment_dir else None
    learner = Learner(
        batch_queue=batch_queue,
        priority_queue=priority_queue,
        stats_queue=stats_queue,
        model=shared_model,
        game_name=env_name,
        models_dir=os.path.join(experiment_dir, "models"),
        use_wandb=(use_wandb and WANDB_AVAILABLE),
        env_name=env_name,
        video_dir=video_dir,
        initial_num_updates=initial_num_updates
    )

    # Create Actors
    actors = []
    for i in range(num_actors):
        actor = Actor(
            epsilon=epsilons[i],
            model=shared_model,
            sample_queue=sample_queue_list[i],
            env_fn=env_factory
        )
        actors.append(actor)

    # Start all processes
    print("\nStarting processes...")

    # Start actor processes
    actor_procs = []
    for i, actor in enumerate(actors):
        proc = mp.Process(target=actor.run)
        proc.start()
        actor_procs.append(proc)
    print(f"  ✓ {num_actors} Actor processes started")

    # Start replay buffer process
    buffer_proc = mp.Process(target=replay_buffer.run)
    buffer_proc.start()
    print("  ✓ ReplayBuffer process started")

    print("\nTraining in progress...")

    # Learner runs in main process
    learner.run()

    # Wait for buffer to complete
    buffer_proc.join()

    # Terminate actor processes
    for proc in actor_procs:
        proc.terminate()

    # Save final model
    if experiment_dir:
        model_save_path = os.path.join(experiment_dir, "models", "final_model.pth")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(shared_model.state_dict(), model_save_path)
        print(f"\nTraining complete! Model saved to: {model_save_path}")
    else:
        model_save_path = None
        print(f"\nTraining complete!")

    # Cleanup wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("✓ WandB run finished")

    return shared_model, model_save_path


def main():
    parser = argparse.ArgumentParser(
        description="Train R2D2 agent using distributed multi-actor architecture"
    )
    parser.add_argument(
        "--env",
        type=str,
        default=r2d2_config.game_name,
        help="Atari environment name (default: ALE/MsPacman-v5)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=r2d2_config.training_steps,
        help="Total training timesteps (default: 1M)"
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        default=r2d2_config.num_actors,
        help="Number of parallel actors (default: 8)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/r2d2",
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
        help="Directory containing LatencyModel weights (default: ./latency_wrap)"
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to pre-trained model checkpoint to resume training"
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

    # Generate run name (r2d2-name-timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}-r2d2-{args.mode}-{generate_slug(2)}"

    # Create experiment directory using run_name
    env_dir_name = args.env.replace('/', '_')
    experiment_dir = os.path.join(args.output_dir, args.mode, env_dir_name, run_name)

    # Create directory structure
    os.makedirs(os.path.join(experiment_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "videos"), exist_ok=True)

    # Save configuration
    config_path = os.path.join(experiment_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"Run name: {run_name}\n")
        f.write(f"Algorithm: R2D2 (distributed multi-actor)\n")
        f.write(f"Training mode: {args.mode}\n")
        f.write(f"Environment: {args.env}\n")
        f.write(f"Total timesteps: {args.timesteps}\n")
        f.write(f"Number of actors: {args.num_actors}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Latency simulation: {args.mode == 'sim_lat'}\n")
        if args.mode == 'sim_lat':
            f.write(f"Latency model directory: {args.latency_model_dir}\n")
        if args.load_model:
            f.write(f"Loaded from checkpoint: {args.load_model}\n")
        f.write(f"Learning rate: {r2d2_config.lr}\n")
        f.write(f"Gamma: {r2d2_config.gamma}\n")
        f.write(f"Batch size: {r2d2_config.batch_size}\n")
        f.write(f"Buffer capacity: {r2d2_config.buffer_capacity}\n")
        f.write(f"Learning starts: {r2d2_config.learning_starts}\n")
        f.write(f"Base epsilon: {r2d2_config.base_explore_eps}\n")
        f.write(f"Alpha (epsilon schedule): {r2d2_config.alpha}\n")

    # Train agent using distributed multi-actor architecture
    model, model_path = train_agent_distributed(
        env_name=args.env,
        total_timesteps=args.timesteps,
        num_actors=args.num_actors,
        experiment_dir=experiment_dir,
        device=args.device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=run_name,
        seed=args.seed,
        simulate_latency=(args.mode == "sim_lat"),
        latency_model_dir=args.latency_model_dir,
        load_model_path=args.load_model
    )

    print(f"\n{'='*60}")
    print(f"Training completed successfully!")
    print(f"{'='*60}")
    if model_path:
        print(f"Model saved at: {model_path}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"\nNext step: Transfer to physical hardware")
    print(f"  python harness_physical.py \\")
    print(f"    --agent_type=agent_r2d2 \\")
    if model_path:
        print(f"    --load_model={model_path} \\")
    print(f"    --total_frames=500000")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Use fork method for multiprocessing (required for queue/lock sharing)
    # Note: fork is Linux/Unix only. R2D2 architecture requires fork.
    mp.set_start_method('fork', force=True)
    main()
