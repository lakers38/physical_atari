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
from datetime import datetime

import ale_py
import gymnasium as gym
import numpy as np
from coolname import generate_slug
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecEnvWrapper, VecFrameStack, VecMonitor, VecVideoRecorder

import wandb
from wandb.integration.sb3 import WandbCallback

from utils.latency_wrap.wrapper_v0_2 import BatchedLatencyModel
from framework.Logger import logger

gym.register_envs(ale_py)


class VecActionSetWrapper(VecEnvWrapper):
    """Vectorized wrapper for action set restriction"""

    def __init__(self, venv, action_mapping):
        super().__init__(venv)
        self.action_mapping = action_mapping
        self.action_space = spaces.Discrete(len(action_mapping))

    def step_async(self, actions):
        mapped_actions = np.array([self.action_mapping[a] for a in actions])
        self.venv.step_async(mapped_actions)

    def step_wait(self):
        return self.venv.step_wait()

    def reset(self):
        return self.venv.reset()


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

        if action_mapping is not None:
            self.action_space = spaces.Discrete(len(action_mapping))
            logger.info("ppo: VecLatencyWrapper using action mapping: %s", action_mapping)
        else:
            self.action_space = venv.action_space

        self.latency_model = BatchedLatencyModel(latency_model_dir, self.num_envs)
        logger.info("ppo: VecLatencyWrapper initialized batched latency model for %s environments", self.num_envs)

    def step_async(self, actions):
        """Apply latency model to all actions in one batched forward pass"""
        if self.action_mapping is not None:
            actions = np.array([self.action_mapping[int(a)] for a in actions])
        else:
            actions = np.array(actions)

        allowed_actions = self.action_mapping if self.action_mapping is not None else None
        delayed_actions = self.latency_model.act_batch(actions, allowed_actions=allowed_actions)

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
    use_full_action_space = simulate_latency or (reduce_action_set == 0)

    env = make_atari_env(
        env_name,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={'full_action_space': use_full_action_space},
        wrapper_kwargs={"screen_size": 128},
    )

    action_mapping = None
    if reduce_action_set == 2:
        action_mapping = [2, 5, 4, 3]
        logger.info("ppo: ActionRestriction using reduced action space")

    if simulate_latency:
        logger.info("ppo: Applying latency simulation to %s", env_name)
        env = VecLatencyWrapper(env, latency_model_dir=latency_model_dir, action_mapping=action_mapping)
    elif action_mapping is not None:
        env = VecActionSetWrapper(env, action_mapping)
        logger.info("ppo: ActionRestriction applied mapping (%s actions): %s", len(action_mapping), action_mapping)

    env = VecFrameStack(env, n_stack=n_stack)

    if monitor_path:
        env = VecMonitor(env, filename=os.path.join(monitor_path, f"{env_name.replace('/', '_')}_monitor.csv"))

    if record_video and video_path:
        logger.info("ppo: Recording videos every %s steps to %s", video_freq, video_path)
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
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        logger.info("ppo: WandB initialized run=%s url=%s", wandb_run.name, wandb_run.url)

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

    tensorboard_log_dir = os.path.join(experiment_dir, "logs", "tensorboard") if experiment_dir else "./logs/"

    if load_model_path and os.path.exists(load_model_path):
        logger.info("ppo: Loading pre-trained model from %s", load_model_path)
        model = PPO.load(load_model_path, env=env, device=device)
        if learning_rate:
            model.learning_rate = learning_rate
    else:
        logger.info("ppo: Creating new PPO model")
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

    callbacks = [checkpoint_callback, eval_callback]

    if use_wandb:
        wandb_callback = WandbCallback(
            model_save_path=os.path.join(experiment_dir, "models", f"wandb_{wandb_run.id}")
            if experiment_dir
            else f"./models/wandb_{wandb_run.id}",
            verbose=2,
        )
        callbacks.append(wandb_callback)
        logger.info("ppo: WandB callback added - models will be uploaded")

    mode_name = "sim_lat (with LatencyModel)" if simulate_latency else "sim (no latency)"
    logger.info("Starting Training: %s", mode_name)
    logger.info("Environment: %s", env_name)
    logger.info("Total timesteps: %s", f"{total_timesteps:,}")
    logger.info("Latency simulation: %s", simulate_latency)
    logger.info("Device: %s", device)
    logger.info("Learning rate: %s", learning_rate)
    if use_wandb and wandb_run:
        logger.info("WandB: %s", wandb_run.url)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=model_name,
        reset_num_timesteps=False if load_model_path else True,
    )

    model.save(model_save_path)
    logger.info("ppo: Training complete! Model saved to: %s", model_save_path)

    if use_wandb and wandb_run is not None:
        wandb_run.finish()
        logger.info("ppo: WandB run finished and uploaded")

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
        help="Training mode: sim (no latency) or sim_lat (with LatencyModel) (default: sim_lat)",
    )
    parser.add_argument(
        "--latency-model-dir", type=str, default="./utils/latency_wrap", help="Directory containing LatencyModel weights"
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}-ppo-{args.mode}-{generate_slug(2)}"

    env_dir_name = args.env.replace('/', '_')
    experiment_dir = os.path.join(args.output_dir, args.mode, env_dir_name, run_name)

    os.makedirs(os.path.join(experiment_dir, "logs", "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs", "eval"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs", "monitor"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "models", "checkpoints"), exist_ok=True)
    if not args.no_videos:
        os.makedirs(os.path.join(experiment_dir, "videos"), exist_ok=True)

    config_path = os.path.join(experiment_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"Run name: {run_name}\n")
        f.write(f"Training mode: {args.mode}\n")
        f.write(f"Environment: {args.env}\n")
        f.write(f"Total timesteps: {args.timesteps}\n")
        f.write(f"Latency simulation: {args.mode == 'sim_lat'}\n")
        f.write(f"Reduce action set: {args.reduce_action_set}\n")
        f.write("\n# Training Hyperparameters\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"N steps: {args.n_steps}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"N epochs: {args.n_epochs}\n")
        f.write(f"N envs: {args.n_envs}\n")
        f.write(f"N stack: {args.n_stack}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write("\n# Video Recording\n")
        f.write(f"Record videos: {not args.no_videos}\n")
        if not args.no_videos:
            f.write(f"Video frequency: {args.video_freq} steps\n")
            f.write(f"Video length: {args.video_length} frames\n")
        if args.load_model:
            f.write("\n# Model Loading\n")
            f.write(f"Loaded from: {args.load_model}\n")

    _model, model_path = train_agent(
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

    logger.info("Training completed successfully!")
    logger.info("Model saved at: %s", model_path)
    logger.info("Experiment directory: %s", experiment_dir)
    logger.info("View training progress:")
    logger.info("  tensorboard --logdir %s", os.path.join(experiment_dir, 'logs', 'tensorboard'))
    logger.info("Next step: Transfer to physical hardware")
    logger.info("  python harness_physical.py \\")
    logger.info("    --agent_type=agent_ppo \\")
    logger.info("    --load_model=%s.zip \\", model_path)
    logger.info("    --total_frames=500000")


if __name__ == "__main__":
    main()
