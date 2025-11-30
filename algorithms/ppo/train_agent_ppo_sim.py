#!/usr/bin/env python3
"""
Train PPO agent in Gymnasium with optional hardware latency simulation

This script trains a PPO agent on Atari games with two modes:
- sim: Pure simulation (no latency) - fast baseline training
- sim_lat: Simulation with LatencyModel - simulates real hardware delays

Flow (sim_lat mode):
    obs = env.step()
    action = PPO.predict(obs)
    delayed_action = LatencyModel.act(action)  # Simulates hardware latency
    obs, reward, done = env.step(delayed_action)
"""

import argparse
import os
import sys
from datetime import datetime

import ale_py
import gymnasium as gym
import numpy as np
from coolname import generate_slug
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper, VecFrameStack, VecMonitor, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'latency_wrap'))
from wrapper_v0_2 import LatencyModel, BatchedLatencyModel

# Register ALE environments
gym.register_envs(ale_py)


class ActionSetWrapper(gym.Wrapper):
    """
    Wrapper to restrict the action space similar to agent_delay_target.py logic.

    Supports three modes:
    - reduce_action_set=0: Full legal action set (18 actions)
    - reduce_action_set=1: ALE minimal action set per game
    - reduce_action_set=2: Further restricted for ms_pacman/qbert (4 directional actions: UP, DOWN, LEFT, RIGHT)
    """

    def __init__(self, env, reduce_action_set=1, game_name=""):
        super().__init__(env)
        self.reduce_action_set = reduce_action_set
        self.game_name = game_name.lower()

        if reduce_action_set == 0:
            self.action_mapping = None
        elif reduce_action_set == 2:
            # ALE action indices: UP=2, DOWN=5, LEFT=4, RIGHT=3
            self.action_mapping = [2, 5, 4, 3]  # UP, DOWN, LEFT, RIGHT
            print(f"[ActionSetWrapper] Restricting {game_name} to 4 directional actions only")
        else:
            self.action_mapping = None

        # Update action space if we have a custom mapping
        if self.action_mapping is not None:
            self.action_space = spaces.Discrete(len(self.action_mapping))
            print(f"[ActionSetWrapper] Action space reduced to {len(self.action_mapping)} actions: {self.action_mapping}")

    def step(self, action):
        # Map the restricted action to the full action space if needed
        if self.action_mapping is not None:
            action = self.action_mapping[action]
        return self.env.step(action)

class VecActionSetWrapper(VecEnvWrapper):
    """Vectorized wrapper for action set restriction"""
    def __init__(self, venv, action_mapping):
        super().__init__(venv)
        self.action_mapping = action_mapping
        # Update action space
        self.action_space = spaces.Discrete(len(action_mapping))

    def step_async(self, actions):
        # Map restricted actions to full action space
        mapped_actions = np.array([self.action_mapping[a] for a in actions])
        self.venv.step_async(mapped_actions)

    def step_wait(self):
        return self.venv.step_wait()

    def reset(self):
        return self.venv.reset()


class LatencyWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that applies the LatencyModel to simulate hardware latency.

    The LatencyModel maintains a history of the past 30 actions and uses a neural
    network to predict which action should actually be executed, simulating the
    real-world delay between action selection and execution.
    """

    def __init__(self, env, latency_model_dir):
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
            self.latency_model.action_queue.append(self.latency_model._LatencyModel__one_hot_encode(0, 0, 36))
        self.latency_model.last_action = 0

        return self.env.reset(**kwargs)


class VecLatencyWrapper:
    """
    Vectorized environment wrapper for latency simulation using batched processing.
    Processes all environments in a single forward pass for efficiency.
    """

    def __init__(self, venv, latency_model_dir, action_mapping=None):
        """
        Args:
            venv: VecEnv to wrap
            latency_model_dir: Directory containing LatencyModel weights
            action_mapping: Optional list mapping reduced action space to full 18-action space
                          e.g., [2, 5, 4, 3] maps 4 actions (UP, DOWN, LEFT, RIGHT) to their 18-action equivalents
        """
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_mapping = action_mapping

        # If using action mapping, update action space to reduced size
        if action_mapping is not None:
            self.action_space = spaces.Discrete(len(action_mapping))
            print(f"[VecLatencyWrapper] Using action mapping: {action_mapping}")
        else:
            self.action_space = venv.action_space

        # Single batched model instead of n_envs separate models
        self.latency_model = BatchedLatencyModel(latency_model_dir, self.num_envs)
        print(f"[VecLatencyWrapper] Initialized batched latency model for {self.num_envs} environments")

    def step_async(self, actions):
        """Apply latency model to all actions in one batched forward pass"""
        # Map from reduced action space to full 18-action space if needed
        if self.action_mapping is not None:
            actions = np.array([self.action_mapping[int(a)] for a in actions])
        else:
            actions = np.array(actions)

        # Single batched forward pass for all envs
        delayed_actions = self.latency_model.act_batch(actions)

        self.venv.step_async(delayed_actions)

    def step_wait(self):
        """Wait for step to complete"""
        return self.venv.step_wait()

    def step(self, actions):
        """Synchronous step"""
        self.step_async(actions)
        return self.step_wait()

    def reset(self, **kwargs):
        """Reset all environments and latency model"""
        self.latency_model.reset_all()
        return self.venv.reset(**kwargs)

    def __getattr__(self, name):
        """Forward attribute access to wrapped venv"""
        return getattr(self.venv, name)


def create_atari_env_with_latency(
    env_name,
    n_envs,
    seed,
    simulate_latency,
    latency_model_dir,
    monitor_path,
    video_path,
    record_video,
    video_freq,
    video_length,
    reduce_action_set,
    n_stack,
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
        reduce_action_set: 0=full 18 actions, 1=minimal per game, 2=restricted 4-dir for ms_pacman/qbert

    Returns:
        Vectorized environment with frame stacking and optional latency simulation
    """
    # LatencyModel always needs the environment to have full 18-action space
    # because it outputs actions in the 18-action space
    use_full_action_space = simulate_latency or (reduce_action_set == 0)

    # Create base Atari environment with standard preprocessing
    env = make_atari_env(
        env_name, n_envs=n_envs, seed=seed,
        env_kwargs={'full_action_space': use_full_action_space},
        wrapper_kwargs={"screen_size": 128},
        # vec_env_cls=SubprocVecEnv
    )

    # Determine if we need action mapping for reduced action set
    action_mapping = None
    if reduce_action_set == 2:
            # Action mapping: agent uses indices 0-3, which map to ALE actions [2, 5, 4, 3]
            # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT (matching agent_delay_target.py)
            action_mapping = [2, 5, 4, 3]
            print(f"[ActionRestriction] Will use reduced action space")

    # Apply latency wrapper BEFORE frame stacking
    # The latency wrapper now handles action mapping internally
    if simulate_latency:
        print(f"[sim_lat] Applying latency simulation to {env_name}")
        env = VecLatencyWrapper(env, latency_model_dir=latency_model_dir, action_mapping=action_mapping)
    elif action_mapping is not None:
        # If not using latency but still want reduced action set, use VecActionSetWrapper
        env = VecActionSetWrapper(env, action_mapping)
        print(f"[ActionRestriction] Applied action space reduction to {len(action_mapping)} actions: {action_mapping}")

    # Apply frame stacking
    env = VecFrameStack(env, n_stack=n_stack)

    # Add monitoring
    if monitor_path:
        env = VecMonitor(env, filename=os.path.join(monitor_path, f"{env_name.replace('/', '_')}_monitor.csv"))

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
    env_name,
    total_timesteps,
    simulate_latency,
    latency_model_dir,
    experiment_dir,
    device,
    learning_rate,
    n_steps,
    batch_size,
    n_epochs,
    n_envs,
    seed,
    load_model_path,
    record_videos,
    video_freq,
    video_length,
    use_wandb,
    wandb_project,
    wandb_entity,
    wandb_run_name,
    reduce_action_set,
    n_stack,
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
        reduce_action_set: 0=full 18 actions, 1=minimal per game, 2=restricted 4-dir for ms_pacman/qbert

    Returns:
        Trained model and save path
    """
    assert experiment_dir is not None
    # Initialize WandB if requested
    wandb_run = None
    if use_wandb:
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
            "training_mode": "sim_lat" if simulate_latency else "sim",
            "latency_enabled": simulate_latency,
        }

        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=config,
            sync_tensorboard=True,  # Auto-upload TensorBoard metrics
            monitor_gym=True,  # Auto-upload videos
            save_code=True,
        )
        print(f"[WandB] Initialized run: {wandb_run.name}")
        print(f"[WandB] View at: {wandb_run.url}")

    # Create environments
    monitor_path = os.path.join(experiment_dir, "logs", "monitor") if experiment_dir else None
    video_path = os.path.join(experiment_dir, "videos") if experiment_dir and record_videos else None

    env = create_atari_env_with_latency(
        env_name=env_name,
        n_envs=n_envs,
        seed=seed,
        simulate_latency=simulate_latency,
        latency_model_dir=latency_model_dir,
        monitor_path=monitor_path,
        video_path=video_path,
        record_video=record_videos,
        video_freq=video_freq,
        video_length=video_length,
        reduce_action_set=reduce_action_set,
        n_stack=n_stack,
    )

    eval_env = create_atari_env_with_latency(
        env_name=env_name,
        n_envs=1,
        seed=seed,
        simulate_latency=simulate_latency,
        latency_model_dir=latency_model_dir,
        monitor_path=monitor_path,
        video_path=None,
        record_video=False,
        video_freq=video_freq,
        video_length=video_length,
        reduce_action_set=reduce_action_set,
        n_stack=n_stack,
    )

    # Create or load PPO model
    tensorboard_log_dir = os.path.join(experiment_dir, "logs", "tensorboard") if experiment_dir else "./logs/"

    if load_model_path and os.path.exists(load_model_path):
        print(f"Loading pre-trained model from {load_model_path}")
        model = PPO.load(load_model_path, env=env, device=device)
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
            device=device,
        )

    # Setup callbacks
    timestamp = os.path.basename(experiment_dir) if experiment_dir else datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "sim_lat" if simulate_latency else "sim"
    model_name = f"PPO_{mode}_{env_name.replace('/', '_')}_{timestamp}"

    model_save_path = (
        os.path.join(experiment_dir, "models", "final_model") if experiment_dir else f"./models/{model_name}"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(experiment_dir, "models", "checkpoints") if experiment_dir else "./models/checkpoints/",
        name_prefix=model_name,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(experiment_dir, "models", "best_model")
        if experiment_dir
        else f"./models/best_{model_name}/",
        log_path=os.path.join(experiment_dir, "logs", "eval") if experiment_dir else "./logs/eval/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # Setup callbacks list
    callbacks = [checkpoint_callback, eval_callback]

    # Add WandB callback if enabled
    if use_wandb:
        wandb_callback = WandbCallback(
            model_save_path=os.path.join(experiment_dir, "models", f"wandb_{wandb_run.id}")
            if experiment_dir
            else f"./models/wandb_{wandb_run.id}",
            verbose=2,
        )
        callbacks.append(wandb_callback)
        print(f"[WandB] Callback added - models will be uploaded")

    # Train the model
    mode_name = "sim_lat (with LatencyModel)" if simulate_latency else "sim (no latency)"
    print(f"\n{'=' * 60}")
    print(f"Starting Training: {mode_name}")
    print(f"{'=' * 60}")
    print(f"Environment: {env_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Latency simulation: {simulate_latency}")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    if use_wandb and wandb_run:
        print(f"WandB: {wandb_run.url}")
    print(f"{'=' * 60}\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=model_name,
        reset_num_timesteps=False if load_model_path else True,
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
    parser = argparse.ArgumentParser(description="Train PPO agent in Gymnasium simulation with optional latency")
    parser.add_argument(
        "--env", type=str, default="ALE/MsPacman-v5", help="Atari environment name (default: ALE/MsPacman-v5)"
    )
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps (default: 1M)")
    parser.add_argument(
        "--mode",
        type=str,
        default="sim_lat",
        choices=["sim", "sim_lat"],
        help="Training mode: sim (no latency) or sim_lat (with LatencyModel, default)",
    )
    parser.add_argument(
        "--latency-model-dir", type=str, default="./latency_wrap", help="Directory containing LatencyModel weights"
    )
    parser.add_argument("--output-dir", type=str, default="outputs/ppo/", help="Base directory for outputs")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Device to use for training"
    )
    parser.add_argument("--load-model", type=str, default=None, help="Path to pre-trained model to continue training")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Learning rate (default: 2.5e-4)")
    parser.add_argument("--n-steps", type=int, default=128, help="Steps per PPO update (default: 128)")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size (default: 128)")
    parser.add_argument("--n-epochs", type=int, default=4, help="PPO epochs per update (default: 4)")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel environments (default: 16)")
    parser.add_argument("--n-stack", type=int, default=4, help="Number of frames to stack (default: 4)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--no-videos", action="store_true", help="Disable video recording (default: videos enabled)")
    parser.add_argument("--video-freq", type=int, default=10000, help="Record video every N steps (default: 10000)")
    parser.add_argument("--video-length", type=int, default=500, help="Number of frames per video (default: 500)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb-project", type=str, default="physical-atari", help="WandB project name (default: physical-atari)"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="WandB entity/team name (default: your username)"
    )
    parser.add_argument(
        "--reduce-action-set",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Action set mode: 0=full 18 actions, 1=minimal per game (default), 2=restricted 4-dir for ms_pacman/qbert",
    )
    args = parser.parse_args()

    # Generate run name (mode-name-timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}-ppo-{args.mode}-{generate_slug(2)}"

    # Create experiment directory using run_name
    env_dir_name = args.env.replace('/', '_')
    experiment_dir = os.path.join(args.output_dir, args.mode, env_dir_name, run_name)

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
        f.write(f"Run name: {run_name}\n")
        f.write(f"Training mode: {args.mode}\n")
        f.write(f"Environment: {args.env}\n")
        f.write(f"Total timesteps: {args.timesteps}\n")
        f.write(f"Latency simulation: {args.mode == 'sim_lat'}\n")
        f.write(f"Reduce action set: {args.reduce_action_set}\n")
        f.write(f"\n# Training Hyperparameters\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"N steps: {args.n_steps}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"N epochs: {args.n_epochs}\n")
        f.write(f"N envs: {args.n_envs}\n")
        f.write(f"N stack: {args.n_stack}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"\n# Video Recording\n")
        f.write(f"Record videos: {not args.no_videos}\n")
        if not args.no_videos:
            f.write(f"Video frequency: {args.video_freq} steps\n")
            f.write(f"Video length: {args.video_length} frames\n")
        if args.load_model:
            f.write(f"\n# Model Loading\n")
            f.write(f"Loaded from: {args.load_model}\n")

    # Train agent
    model, model_path = train_agent(
        env_name=args.env,
        total_timesteps=args.timesteps,
        simulate_latency=(args.mode == "sim_lat"),
        latency_model_dir=args.latency_model_dir,
        experiment_dir=experiment_dir,
        device=args.device,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        n_envs=args.n_envs,
        seed=args.seed,
        load_model_path=args.load_model,
        record_videos=not args.no_videos,
        video_freq=args.video_freq,
        video_length=args.video_length,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=run_name,
        reduce_action_set=args.reduce_action_set,
        n_stack=args.n_stack,
    )

    print(f"\n{'=' * 60}")
    print(f"Training completed successfully!")
    print(f"{'=' * 60}")
    print(f"Model saved at: {model_path}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"View training progress:")
    print(f"  tensorboard --logdir {os.path.join(experiment_dir, 'logs', 'tensorboard')}")
    print(f"\nNext step: Transfer to physical hardware")
    print(f"  python harness_physical.py \\")
    print(f"    --agent_type=agent_ppo \\")
    print(f"    --load_model={model_path}.zip \\")
    print(f"    --total_frames=500000")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
