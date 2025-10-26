#!/usr/bin/env python3
"""
Phase 2: Train PPO agent in Gymnasium with simulated hardware latency

This script trains a PPO agent on Atari games using the LatencyModel to simulate
the real-world action delay that occurs with the physical Robotroller hardware.

Flow:
    obs = env.step()
    action = PPO.predict(obs)
    delayed_action = LatencyModel.act(action)  # Simulates hardware latency
    obs, reward, done = env.step(delayed_action)
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

import gymnasium as gym
import ale_py
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from gymnasium import spaces

# WandB integration (optional)
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    WandbCallback = None

# Import the latency model
sys.path.append(os.path.join(os.path.dirname(__file__), 'latency_wrap'))
from wrapper_v0_2 import LatencyModel

# Register ALE environments
gym.register_envs(ale_py)


class LatencyWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that applies the LatencyModel to simulate hardware latency.

    The LatencyModel maintains a history of the past 30 actions and uses a neural
    network to predict which action should actually be executed, simulating the
    real-world delay between action selection and execution.
    """

    def __init__(self, env, latency_model_dir="./latency_wrap"):
        """
        Args:
            env: The Gymnasium environment to wrap
            latency_model_dir: Directory containing the LatencyModel weights
        """
        super().__init__(env)
        self.latency_model = LatencyModel(directory_with_weights=latency_model_dir)
        print(f"[LatencyWrapper] Initialized with weights from {latency_model_dir}")

    def step(self, action):
        """
        Step function with latency simulation.

        Args:
            action: The action selected by the agent

        Returns:
            obs, reward, terminated, truncated, info
        """
        # Convert action to ALE action format and apply latency model
        ale_action = ale_py.Action(int(action))
        delayed_action = self.latency_model.act(ale_action)

        # Execute the delayed action in the environment
        return self.env.step(int(delayed_action))

    def reset(self, **kwargs):
        """Reset the environment and latency model state"""
        # Reset the latency model's action queue to NOOPs
        self.latency_model.action_queue = []
        for _ in range(30):
            self.latency_model.action_queue.append(
                self.latency_model._LatencyModel__one_hot_encode(0, 0, 36)
            )
        self.latency_model.last_action = 0

        return self.env.reset(**kwargs)


class VecLatencyWrapper:
    """
    Vectorized environment wrapper for latency simulation.
    Works with VecEnv from stable-baselines3.
    """

    def __init__(self, venv, latency_model_dir="./latency_wrap"):
        """
        Args:
            venv: VecEnv to wrap
            latency_model_dir: Directory containing LatencyModel weights
        """
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space

        # Create separate latency model for each parallel environment
        self.latency_models = [
            LatencyModel(directory_with_weights=latency_model_dir)
            for _ in range(self.num_envs)
        ]
        print(f"[VecLatencyWrapper] Initialized {self.num_envs} latency models")

    def step_async(self, actions):
        """Apply latency model to actions before sending to environments"""
        delayed_actions = []
        for i, action in enumerate(actions):
            ale_action = ale_py.Action(int(action))
            delayed_action = self.latency_models[i].act(ale_action)
            delayed_actions.append(int(delayed_action))

        self.venv.step_async(np.array(delayed_actions))

    def step_wait(self):
        """Wait for step to complete"""
        return self.venv.step_wait()

    def step(self, actions):
        """Synchronous step"""
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        """Reset all environments and latency models"""
        for model in self.latency_models:
            model.action_queue = []
            for _ in range(30):
                model.action_queue.append(model._LatencyModel__one_hot_encode(0, 0, 36))
            model.last_action = 0

        return self.venv.reset()

    def __getattr__(self, name):
        """Forward attribute access to wrapped venv"""
        return getattr(self.venv, name)


def create_atari_env_with_latency(
    env_name,
    n_envs=4,
    seed=0,
    simulate_latency=True,
    latency_model_dir="./latency_wrap",
    monitor_path=None,
    video_path=None,
    record_video=False,
    video_freq=10000,
    video_length=500
):
    """
    Create Atari environment with preprocessing and optional latency simulation.

    Args:
        env_name: Name of the Atari environment (e.g., "ALE/MsPacman-v5")
        n_envs: Number of parallel environments
        seed: Random seed
        simulate_latency: If True, apply LatencyModel wrapper
        latency_model_dir: Directory containing LatencyModel weights
        monitor_path: Path for monitoring logs
        video_path: Path for video recordings
        record_video: If True, record videos during training
        video_freq: Record video every N steps
        video_length: Number of frames per video

    Returns:
        Vectorized environment with frame stacking and optional latency simulation
    """
    # Create base Atari environment with standard preprocessing
    # IMPORTANT: Use full_action_space=True because LatencyModel expects 18 actions
    env = make_atari_env(env_name, n_envs=n_envs, seed=seed, env_kwargs={'full_action_space': True})

    # Apply latency wrapper BEFORE frame stacking
    # This ensures the latency affects the raw actions
    if simulate_latency:
        print(f"[Phase 2] Applying latency simulation to {env_name}")
        env = VecLatencyWrapper(env, latency_model_dir=latency_model_dir)

    # Apply frame stacking (4 frames)
    env = VecFrameStack(env, n_stack=4)

    # Add monitoring
    if monitor_path:
        env = VecMonitor(
            env,
            filename=os.path.join(monitor_path, f"{env_name.replace('/', '_')}_monitor.csv")
        )

    # Add video recording
    if record_video and video_path:
        print(f"[Video] Recording videos every {video_freq} steps to {video_path}")
        env = VecVideoRecorder(
            env,
            video_path,
            record_video_trigger=lambda x: x % video_freq == 0,
            video_length=video_length,
        )

    return env


def train_agent(
    env_name="ALE/MsPacman-v5",
    total_timesteps=1000000,
    simulate_latency=True,
    latency_model_dir="./latency_wrap",
    experiment_dir=None,
    device="cuda",
    learning_rate=2.5e-4,
    n_steps=128,
    batch_size=128,
    n_epochs=4,
    load_model_path=None,
    record_videos=True,
    video_freq=10000,
    video_length=500,
    use_wandb=False,
    wandb_project="physical-atari-phase2",
    wandb_entity=None,
    wandb_run_name=None
):
    """
    Train PPO agent with optional latency simulation.

    Args:
        env_name: Atari environment name
        total_timesteps: Total training timesteps
        simulate_latency: If True, apply LatencyModel
        latency_model_dir: Directory with LatencyModel weights
        experiment_dir: Directory to save results
        device: "cuda", "cpu", or "mps"
        learning_rate: PPO learning rate
        n_steps: Steps per PPO update
        batch_size: Minibatch size
        n_epochs: PPO epochs per update
        load_model_path: Path to pre-trained model to continue training
        record_videos: If True, record gameplay videos
        video_freq: Record video every N steps
        video_length: Number of frames per video
        use_wandb: If True, log to Weights & Biases
        wandb_project: WandB project name
        wandb_entity: WandB entity/team name
        wandb_run_name: WandB run name

    Returns:
        Trained model and save path
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
                "learning_rate": learning_rate,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.1,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "device": device,
                "algorithm": "PPO",
                "phase": "Phase 2 - Latency" if simulate_latency else "Phase 1 - No Latency",
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

    # Create environments
    monitor_path = os.path.join(experiment_dir, "logs", "monitor") if experiment_dir else None
    video_path = os.path.join(experiment_dir, "videos") if experiment_dir and record_videos else None

    env = create_atari_env_with_latency(
        env_name,
        n_envs=4,
        simulate_latency=simulate_latency,
        latency_model_dir=latency_model_dir,
        monitor_path=monitor_path,
        video_path=video_path,
        record_video=record_videos,
        video_freq=video_freq,
        video_length=video_length
    )

    eval_env = create_atari_env_with_latency(
        env_name,
        n_envs=1,
        simulate_latency=simulate_latency,
        latency_model_dir=latency_model_dir,
        monitor_path=monitor_path
    )

    # Create or load PPO model
    tensorboard_log_dir = os.path.join(experiment_dir, "logs", "tensorboard") if experiment_dir else "./logs/"

    if load_model_path and os.path.exists(load_model_path):
        print(f"Loading pre-trained model from {load_model_path}")
        model = PPO.load(
            load_model_path,
            env=env,
            device=device
        )
        # Update learning rate if specified
        if learning_rate:
            model.learning_rate = learning_rate
    else:
        print(f"Creating new PPO model")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_log_dir,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=device
        )

    # Setup callbacks
    timestamp = os.path.basename(experiment_dir) if experiment_dir else datetime.now().strftime("%Y%m%d_%H%M%S")
    phase = "Phase2_Latency" if simulate_latency else "Phase1_NoLatency"
    model_name = f"PPO_{phase}_{env_name.replace('/', '_')}_{timestamp}"

    model_save_path = os.path.join(experiment_dir, "models", "final_model") if experiment_dir else f"./models/{model_name}"

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(experiment_dir, "models", "checkpoints") if experiment_dir else "./models/checkpoints/",
        name_prefix=model_name
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(experiment_dir, "models", "best_model") if experiment_dir else f"./models/best_{model_name}/",
        log_path=os.path.join(experiment_dir, "logs", "eval") if experiment_dir else "./logs/eval/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Setup callbacks list
    callbacks = [checkpoint_callback, eval_callback]

    # Add WandB callback if enabled
    if use_wandb and WANDB_AVAILABLE and wandb_run is not None:
        wandb_callback = WandbCallback(
            model_save_path=os.path.join(experiment_dir, "models", f"wandb_{wandb_run.id}") if experiment_dir else f"./models/wandb_{wandb_run.id}",
            verbose=2,
        )
        callbacks.append(wandb_callback)
        print(f"[WandB] Callback added - models will be uploaded")

    # Train the model
    print(f"\n{'='*60}")
    print(f"Starting Phase 2 Training: PPO with Latency Simulation")
    print(f"{'='*60}")
    print(f"Environment: {env_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Latency simulation: {simulate_latency}")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    if use_wandb and wandb_run:
        print(f"WandB: {wandb_run.url}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=model_name,
        reset_num_timesteps=False if load_model_path else True
    )

    # Save final model
    model.save(model_save_path)
    print(f"\nTraining complete! Model saved to: {model_save_path}")

    # Finish WandB run
    if use_wandb and wandb_run is not None:
        wandb_run.finish()
        print(f"[WandB] Run finished and uploaded")

    return model, model_save_path


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Train PPO agent with simulated hardware latency"
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
        "--no-latency",
        action="store_true",
        help="Train WITHOUT latency simulation (Phase 1)"
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
        default="outputs_sim",
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
        "--load-model",
        type=str,
        default=None,
        help="Path to pre-trained model to continue training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="Learning rate (default: 2.5e-4)"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=128,
        help="Steps per PPO update (default: 128)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Minibatch size (default: 128)"
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Disable video recording (default: videos enabled)"
    )
    parser.add_argument(
        "--video-freq",
        type=int,
        default=10000,
        help="Record video every N steps (default: 10000)"
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
        default="physical-atari-phase2",
        help="WandB project name (default: physical-atari-phase2)"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity/team name (default: your username)"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="WandB run name (default: auto-generated)"
    )

    args = parser.parse_args()

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    phase = "Phase1" if args.no_latency else "Phase2_Latency"
    env_dir_name = args.env.replace('/', '_')
    experiment_dir = os.path.join(args.output_dir, phase, env_dir_name, timestamp)

    # Create directory structure
    os.makedirs(os.path.join(experiment_dir, "logs", "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs", "eval"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs", "monitor"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "models", "checkpoints"), exist_ok=True)
    if not args.no_videos:
        os.makedirs(os.path.join(experiment_dir, "videos"), exist_ok=True)

    # Save configuration
    config_path = os.path.join(experiment_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"Phase: {phase}\n")
        f.write(f"Environment: {args.env}\n")
        f.write(f"Total timesteps: {args.timesteps}\n")
        f.write(f"Latency simulation: {not args.no_latency}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"N steps: {args.n_steps}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Record videos: {not args.no_videos}\n")
        if not args.no_videos:
            f.write(f"Video frequency: {args.video_freq} steps\n")
            f.write(f"Video length: {args.video_length} frames\n")
        if args.load_model:
            f.write(f"Loaded from: {args.load_model}\n")

    # Train agent
    model, model_path = train_agent(
        env_name=args.env,
        total_timesteps=args.timesteps,
        simulate_latency=not args.no_latency,
        latency_model_dir=args.latency_model_dir,
        experiment_dir=experiment_dir,
        device=args.device,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        load_model_path=args.load_model,
        record_videos=not args.no_videos,
        video_freq=args.video_freq,
        video_length=args.video_length,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name
    )

    print(f"\n{'='*60}")
    print(f"Training completed successfully!")
    print(f"{'='*60}")
    print(f"Model saved at: {model_path}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"View training progress:")
    print(f"  tensorboard --logdir {os.path.join(experiment_dir, 'logs', 'tensorboard')}")
    print(f"\nNext step: Transfer to physical hardware")
    print(f"  python harness_physical.py \\")
    print(f"    --agent_type=agent_ppo \\")
    print(f"    --load_model={model_path}.zip \\")
    print(f"    --total_frames=500000")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
