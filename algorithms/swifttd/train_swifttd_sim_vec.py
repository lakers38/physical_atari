#!/usr/bin/env python3
"""
Train SwiftTD agent in Gymnasium with optional hardware latency simulation

This script trains a SwiftTD agent on Atari games with two modes:
- sim: Pure simulation (no latency) - fast baseline training
- sim_lat: Simulation with LatencyModel - simulates real hardware delays

Flow (sim_lat mode):
    obs = env.step()
    action = SwiftTDAgent.select_actions(obs)
    delayed_action = LatencyModel.act(action)  # Simulates hardware latency
    obs, reward, done = env.step(delayed_action)
"""

import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Optional
import math

import ale_py
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import numpy as np
from coolname import generate_slug
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from scipy.ndimage import zoom
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv
from torch.utils.tensorboard import SummaryWriter

import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'latency_wrap'))
from wrapper_v0_2 import LatencyModel

sys.path.append(os.path.dirname(__file__))
from agent_actor_critic_vec import SwiftTDAgent

# Register ALE environments
gym.register_envs(ale_py)


class PreprocessWrapper(gym.Wrapper):
    """Grayscale + resize to square, channel-last."""

    def __init__(self, env, frame_size: int):
        super().__init__(env)
        self.frame_size = frame_size
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(frame_size, frame_size, 1),
            dtype=np.uint8,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
        zoom_factors = (self.frame_size / gray.shape[0], self.frame_size / gray.shape[1])
        resized = zoom(gray, zoom_factors, order=1)
        return resized.astype(np.uint8)[..., None]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._preprocess(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._preprocess(obs), reward, terminated, truncated, info


class StackFrames(gym.Wrapper):
    """Simple frame stack along the channel-last dimension."""

    def __init__(self, env, num_stack: int):
        super().__init__(env)
        self.num_stack = num_stack
        h, w, c = env.observation_space.shape
        self.frames: deque = deque(maxlen=num_stack)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(h, w, c * num_stack),
            dtype=np.uint8,
        )

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info


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


class LatencyWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that applies the LatencyModel to simulate hardware latency.

    The LatencyModel maintains a history of the past 30 actions and uses a neural
    network to predict which action should actually be executed, simulating the
    real-world delay between action selection and execution.
    """

    def __init__(self, env, latency_model_dir, allowed_actions=None):
        """
        Args:
            env: The Gymnasium environment to wrap
            latency_model_dir: Directory containing the LatencyModel weights
            allowed_actions: Optional list of action indices to restrict latency outputs
        """
        super().__init__(env)
        self.latency_model = LatencyModel(directory_with_weights=latency_model_dir)
        self.allowed_actions = None if allowed_actions is None else list(allowed_actions)
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
        delayed_action = self.latency_model.act(ale_action, allowed_actions=self.allowed_actions)

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


def make_atari_env(
    env_name: str,
    seed: int,
    simulate_latency: bool,
    latency_model_dir: str,
    reduce_action_set: int,
    n_stack: int,
    input_size: int,
    video_path: Optional[str] = None,
    video_freq: int = 50,
    record_video: bool = False,
):
    """Factory for a single Atari env with preprocessing and optional latency."""
    def thunk():
        use_full_action_space = reduce_action_set in (0, 2)
        env = gym.make(
            env_name,
            obs_type="rgb",
            render_mode="rgb_array" if record_video else None,
            full_action_space=use_full_action_space,
        )

        # Allowed actions for latency masking
        allowed_actions = None
        if reduce_action_set == 1:
            allowed_actions = list(range(env.action_space.n))
        elif reduce_action_set == 2:
            allowed_actions = [2, 5, 4, 3]

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if simulate_latency:
            env = LatencyWrapper(env, latency_model_dir, allowed_actions=allowed_actions)
        if reduce_action_set == 2:
            env = ActionSetWrapper(env, reduce_action_set, env_name)
        env = PreprocessWrapper(env, frame_size=input_size)
        env = StackFrames(env, num_stack=n_stack)
        env = RecordEpisodeStatistics(env)
        if record_video and video_path:
            env = RecordVideo(
                env,
                video_folder=video_path,
                episode_trigger=lambda ep: ep % video_freq == 0,
                name_prefix="training",
                video_length=500,
            )
        env.reset(seed=seed)
        return env
    return thunk


def create_vector_atari_envs(
    num_envs: int,
    env_name: str,
    seed: int,
    simulate_latency: bool,
    latency_model_dir: str,
    reduce_action_set: int,
    n_stack: int,
    input_size: int,
    video_path: Optional[str],
    video_freq: int,
):
    """Create vectorized Atari envs; record video only from env 0 if requested."""
    env_fns = []
    for idx in range(num_envs):
        record_video = (idx == 0) and (video_path is not None)
        env_fns.append(
            make_atari_env(
                env_name=env_name,
                seed=seed + idx,
                simulate_latency=simulate_latency,
                latency_model_dir=latency_model_dir,
                reduce_action_set=reduce_action_set,
                n_stack=n_stack,
                input_size=input_size,
                video_path=video_path,
                video_freq=video_freq,
                record_video=record_video,
            )
        )
    return SyncVectorEnv(env_fns)


def evaluate_agent(agent: SwiftTDAgent, eval_env: gym.Env, n_episodes: int = 10) -> float:
    """Evaluate agent for n_episodes and return mean reward."""
    episode_rewards = []

    for ep in range(n_episodes):
        obs, info = eval_env.reset()
        # Use only the first env slot of the vector agent
        obs_batch = np.repeat(np.expand_dims(obs, axis=0), agent.num_envs, axis=0)
        agent.start_episodes(obs_batch)

        episode_reward = 0
        done = False

        while not done:
            obs_batch = np.repeat(np.expand_dims(obs, axis=0), agent.num_envs, axis=0)
            actions, log_probs, entropy, feats = agent.select_actions(obs_batch)
            next_obs, reward, terminated, truncated, info = eval_env.step(actions[0])
            done = terminated or truncated
            episode_reward += reward

            if done:
                done_mask = np.zeros(agent.num_envs, dtype=bool)
                done_mask[0] = done
                agent.reset_done(done_mask, np.repeat(np.expand_dims(obs, axis=0), agent.num_envs, axis=0))
            else:
                obs = next_obs

        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)


def train_loop(
    agent: SwiftTDAgent,
    env: gym.Env,
    eval_env: gym.Env,
    total_timesteps: int,
    tensorboard_writer: SummaryWriter,
    checkpoint_dir: str,
    model_name: str,
    wandb_run=None,
):
    """Custom training loop for vectorized SwiftTDAgent."""
    num_envs = env.num_envs

    # Episode tracking per env
    episode_rewards = [[] for _ in range(num_envs)]
    episode_lengths = [[] for _ in range(num_envs)]
    current_episode_reward = np.zeros(num_envs, dtype=np.float32)
    current_episode_length = np.zeros(num_envs, dtype=np.int32)
    # For lifetime-error style metric: store per-step (value_pred, reward) per env
    episode_values = [[] for _ in range(num_envs)]
    episode_rewards_stream = [[] for _ in range(num_envs)]
    episode_count = np.zeros(num_envs, dtype=np.int64)

    # Metrics tracking (for logging window)
    recent_advantages = []
    recent_entropies = []
    recent_actor_losses = []
    recent_log_probs = []
    recent_return_errors = []
    recent_value_preds = []
    recent_value_next = []

    # Initialize
    obs, info = env.reset()
    agent.start_episodes(obs)

    start_time = time.time()
    frame_count = 0  # global frames = env_steps * num_envs
    max_env_steps = math.ceil(total_timesteps / num_envs)
    last_log_frame = 0

    for step in range(max_env_steps):
        # Select action
        actions, log_probs, entropy, feats = agent.select_actions(obs)

        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(actions)
        done = np.logical_or(terminated, truncated)

        # Clip reward for stable learning
        reward_clipped = np.clip(reward, -1.0, 1.0)

        # Update agent
        metrics = agent.update(
            obs, actions, reward_clipped,
            next_obs, done,
            log_probs, entropy, feats
        )

        # Track metrics for logging window
        recent_advantages.append(metrics["advantage_mean"])
        recent_entropies.append(float(entropy.mean().item()))
        recent_actor_losses.append(metrics["actor_loss"])
        recent_log_probs.append(float(log_probs.mean().item()))
        recent_return_errors.append(metrics["return_error_mean"])
        recent_value_preds.append(metrics["value_pred_mean"])
        recent_value_next.append(metrics["value_next_mean"])

        # Track episode stats
        current_episode_reward += reward
        current_episode_length += 1
        for i in range(num_envs):
            episode_values[i].append(metrics["value_pred_mean"])
            episode_rewards_stream[i].append(reward[i])

        # Handle episode end
        if np.any(done):
            done_indices = np.where(done)[0]
            for idx in done_indices:
                episode_rewards[idx].append(current_episode_reward[idx])
                episode_lengths[idx].append(current_episode_length[idx])

                returns = []
                G = 0.0
                for r in reversed(episode_rewards_stream[idx]):
                    G = r + agent.gamma * G
                    returns.append(G)
                returns = list(reversed(returns))
                if len(returns) == len(episode_values[idx]) and len(returns) > 0:
                    squared_errors = [(episode_values[idx][j] - returns[j]) ** 2 for j in range(len(returns))]
                    lifetime_err_ep = float(np.mean(squared_errors))
                    tensorboard_writer.add_scalar("train/episode_lifetime_error", lifetime_err_ep, frame_count)
                    if wandb_run is not None:
                        wandb.log({"train/episode_lifetime_error": lifetime_err_ep}, step=frame_count)

                episode_count[idx] += 1
                tensorboard_writer.add_scalar("train/episode_reward", current_episode_reward[idx], frame_count)
                tensorboard_writer.add_scalar("train/episode_length", current_episode_length[idx], frame_count)

                current_episode_reward[idx] = 0
                current_episode_length[idx] = 0
                episode_values[idx] = []
                episode_rewards_stream[idx] = []

            agent.reset_done(done, next_obs)

        obs = next_obs

        frame_count += num_envs

        # Periodic logging every 1000 global frames
        if frame_count >= last_log_frame + 1000:
            flat_rewards = [r for env_rews in episode_rewards for r in env_rews]
            flat_lengths = [l for env_len in episode_lengths for l in env_len]
            mean_reward = np.mean(flat_rewards[-100:]) if flat_rewards else 0
            mean_length = np.mean(flat_lengths[-100:]) if flat_lengths else 0
            max_reward = np.max(flat_rewards[-100:]) if flat_rewards else 0
            min_reward = np.min(flat_rewards[-100:]) if flat_rewards else 0
            fps = frame_count / (time.time() - start_time + 1e-8)
            last_log_frame = frame_count

            # Compute metrics over recent window (last 1000 steps)
            mean_advantage = np.mean(recent_advantages[-1000:])
            std_advantage = np.std(recent_advantages[-1000:])
            mean_entropy = np.mean(recent_entropies[-1000:])
            mean_actor_loss = np.mean(recent_actor_losses[-1000:])
            mean_log_prob = np.mean(recent_log_probs[-1000:])
            mean_return_error = np.mean(recent_return_errors[-1000:])
            mean_value_pred = np.mean(recent_value_preds[-1000:])
            mean_value_next = np.mean(recent_value_next[-1000:])

            # Console output (global frame count first)
            print(f"Frame {frame_count:,} | Step {step:,} | Ep: {int(np.sum(episode_count))} | "
                  f"Reward: {mean_reward:6.2f} (max:{max_reward:5.1f} min:{min_reward:5.1f}) | "
                  f"Len: {mean_length:5.1f} | "
                  f"Adv: {mean_advantage:6.3f}±{std_advantage:.3f} | "
                  f"Ent: {mean_entropy:.3f} | "
                  f"RetErr: {mean_return_error:.4f} | "
                  f"Loss: {mean_actor_loss:.4f} | "
                  f"FPS: {fps:5.1f}")

            # TensorBoard - Episode metrics
            tensorboard_writer.add_scalar("train/mean_reward_100ep", mean_reward, frame_count)
            tensorboard_writer.add_scalar("train/max_reward_100ep", max_reward, frame_count)
            tensorboard_writer.add_scalar("train/min_reward_100ep", min_reward, frame_count)
            tensorboard_writer.add_scalar("train/mean_length_100ep", mean_length, frame_count)
            tensorboard_writer.add_scalar("train/fps", fps, frame_count)

            # TensorBoard - Training metrics
            tensorboard_writer.add_scalar("train/actor_loss", mean_actor_loss, frame_count)
            tensorboard_writer.add_scalar("train/advantage_mean", mean_advantage, frame_count)
            tensorboard_writer.add_scalar("train/advantage_std", std_advantage, frame_count)
            tensorboard_writer.add_scalar("train/entropy", mean_entropy, frame_count)
            tensorboard_writer.add_scalar("train/log_prob", mean_log_prob, frame_count)
            tensorboard_writer.add_scalar("train/return_error_mean", mean_return_error, frame_count)
            tensorboard_writer.add_scalar("train/value_pred_mean", mean_value_pred, frame_count)
            tensorboard_writer.add_scalar("train/value_next_mean", mean_value_next, frame_count)

            # WandB logging mirrors TB when enabled
            if wandb_run is not None:
                wandb.log(
                    {
                        "train/mean_reward_100ep": mean_reward,
                        "train/max_reward_100ep": max_reward,
                        "train/min_reward_100ep": min_reward,
                        "train/mean_length_100ep": mean_length,
                        "train/fps": fps,
                        "train/actor_loss": mean_actor_loss,
                        "train/advantage_mean": mean_advantage,
                        "train/advantage_std": std_advantage,
                        "train/entropy": mean_entropy,
                        "train/log_prob": mean_log_prob,
                        "train/return_error_mean": mean_return_error,
                        "train/value_pred_mean": mean_value_pred,
                        "train/value_next_mean": mean_value_next,
                    },
                    step=frame_count,
                )

        # Evaluation
        if frame_count % 10000 == 0 and frame_count > 0:
            eval_reward = evaluate_agent(agent, eval_env, n_episodes=10)
            tensorboard_writer.add_scalar("eval/mean_reward", eval_reward, frame_count)
            print(f"  Eval @ {step:,}: {eval_reward:.2f}")

        # Checkpointing
        if frame_count % 50000 == 0 and frame_count > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_{frame_count}")
            agent.save(checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    return agent


def train_agent(
    env_name,
    total_timesteps,
    simulate_latency,
    latency_model_dir,
    experiment_dir,
    device,
    learning_rate,
    entropy_coef,
    gamma,
    lambda_,
    critic_initial_alpha,
    critic_eps,
    critic_max_step_size,
    critic_step_decay,
    critic_meta_step_size,
    critic_eta_min,
    critic_feature_scale,
    fail_on_nonfinite,
    num_envs,
    seed,
    load_model_path,
    record_videos,
    video_freq,
    use_wandb,
    wandb_project,
    wandb_entity,
    wandb_run_name,
    reduce_action_set,
    n_stack,
    input_size=128,
):
    """
    Train SwiftTD agent with optional latency simulation.

    Args:
        env_name: Atari environment name
        total_timesteps: Total training timesteps
        simulate_latency: If True, apply LatencyModel
        latency_model_dir: Directory with LatencyModel weights
        experiment_dir: Directory to save results
        device: "cuda", "cpu", or "mps"
        learning_rate: Learning rate for actor/CNN
        entropy_coef: Entropy bonus coefficient
        gamma: Discount factor
        lambda_: TD(lambda) trace parameter
        critic_initial_alpha: SwiftTD initial step size alpha
        critic_eps: SwiftTD epsilon threshold
        critic_max_step_size: SwiftTD max step size eta
        critic_step_decay: SwiftTD step size decay factor
        critic_meta_step_size: SwiftTD meta step size
        critic_eta_min: SwiftTD minimum eta
        critic_feature_scale: Scale factor applied to features before SwiftTD
        fail_on_nonfinite: If True, raise on non-finite critic outputs
        seed: Random seed
        load_model_path: Path to pre-trained model to continue training
        record_videos: If True, record gameplay videos
        video_freq: Record video every N episodes
        use_wandb: If True, log to Weights & Biases
        wandb_project: WandB project name
        wandb_entity: WandB entity/team name
        wandb_run_name: WandB run name
        reduce_action_set: 0=full 18 actions, 1=minimal per game, 2=restricted 4-dir for ms_pacman/qbert
        n_stack: Number of frames to stack
        input_size: Input image size (default 128)

    Returns:
        Trained agent and save path
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
            "gamma": gamma,
            "lambda_": lambda_,
            "ent_coef": entropy_coef,
            "device": device,
            "algorithm": "SwiftTD",
            "training_mode": "sim_lat" if simulate_latency else "sim",
            "latency_enabled": simulate_latency,
            "input_size": input_size,
            "feature_dim": 512,
            "n_stack": n_stack,
            "reduce_action_set": reduce_action_set,
            "num_envs": num_envs,
            "critic_initial_alpha": critic_initial_alpha,
            "critic_eps": critic_eps,
            "critic_max_step_size": critic_max_step_size,
            "critic_step_decay": critic_step_decay,
            "critic_meta_step_size": critic_meta_step_size,
            "critic_eta_min": critic_eta_min,
            "critic_feature_scale": critic_feature_scale,
            "fail_on_nonfinite": fail_on_nonfinite,
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
    video_path = os.path.join(experiment_dir, "videos") if record_videos else None

    env = create_vector_atari_envs(
        num_envs=num_envs,
        env_name=env_name,
        seed=seed,
        simulate_latency=simulate_latency,
        latency_model_dir=latency_model_dir,
        reduce_action_set=reduce_action_set,
        n_stack=n_stack,
        input_size=input_size,
        video_path=video_path,
        video_freq=video_freq,
    )

    eval_env = make_atari_env(
        env_name=env_name,
        seed=seed + 1000,
        simulate_latency=simulate_latency,
        latency_model_dir=latency_model_dir,
        reduce_action_set=reduce_action_set,
        n_stack=n_stack,
        input_size=input_size,
        video_path=None,
        video_freq=0,
        record_video=False,
    )()

    # Create SwiftTDAgent
    agent = SwiftTDAgent(
        num_actions=env.single_action_space.n if hasattr(env, "single_action_space") else env.action_space.n,
        num_envs=num_envs,
        feature_dim=512,
        actor_hidden_dim=256,
        n_stack=n_stack,
        input_size=input_size,
        device=device,
        lambda_=lambda_,
        gamma=gamma,
        initial_alpha=critic_initial_alpha,
        learning_rate=learning_rate,
        entropy_coef=entropy_coef,
        eps=critic_eps,
        max_step_size=critic_max_step_size,
        step_size_decay=critic_step_decay,
        meta_step_size=critic_meta_step_size,
        eta_min=critic_eta_min,
        critic_feature_scale=critic_feature_scale,
        fail_on_nonfinite=fail_on_nonfinite,
    )

    if load_model_path and os.path.exists(load_model_path):
        print(f"Loading pre-trained model from {load_model_path}")
        agent.load(load_model_path)

    # TensorBoard
    tensorboard_log_dir = os.path.join(experiment_dir, "logs", "tensorboard")
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # Training
    checkpoint_dir = os.path.join(experiment_dir, "models", "checkpoints")
    model_name = f"SwiftTD_{env_name.replace('/', '_')}"

    mode_name = "sim_lat (with LatencyModel)" if simulate_latency else "sim (no latency)"
    print(f"\n{'=' * 60}")
    print(f"Starting Training: {mode_name}")
    print(f"{'=' * 60}")
    print(f"Environment: {env_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Device: {device}")
    print(f"Learning rate (actor/CNN): {learning_rate}")
    print(f"Entropy coef: {entropy_coef}")
    print(f"Gamma: {gamma}")
    print(f"Lambda: {lambda_}")
    print(f"Critic initial alpha: {critic_initial_alpha}")
    print(f"Critic eps: {critic_eps}")
    print(f"Critic max step size: {critic_max_step_size}")
    print(f"Critic step decay: {critic_step_decay}")
    print(f"Critic meta step size: {critic_meta_step_size}")
    print(f"Critic eta min: {critic_eta_min}")
    print(f"Critic feature scale: {critic_feature_scale}")
    print(f"Num envs: {num_envs}")
    if use_wandb and wandb_run:
        print(f"WandB: {wandb_run.url}")
    print(f"{'=' * 60}\n")

    train_loop(
        agent=agent,
        env=env,
        eval_env=eval_env,
        total_timesteps=total_timesteps,
        tensorboard_writer=writer,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        wandb_run=wandb_run,
    )

    # Save final model
    final_path = os.path.join(experiment_dir, "models", "final_model")
    agent.save(final_path)
    print(f"\nTraining complete! Model saved to: {final_path}")

    writer.close()
    env.close()
    eval_env.close()

    # Finish WandB run
    if use_wandb and wandb_run is not None:
        wandb_run.finish()
        print(f"[WandB] Run finished and uploaded")

    return agent, final_path


def main():
    parser = argparse.ArgumentParser(description="Train SwiftTD agent in Gymnasium simulation with optional latency")
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
    parser.add_argument("--output-dir", type=str, default="outputs/swifttd/", help="Base directory for outputs")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Device to use for training"
    )
    parser.add_argument("--load-model", type=str, default=None, help="Path to pre-trained model to continue training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for actor/CNN (default: 1e-4)")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient (default: 0.01)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for critic (default: 0.99)")
    parser.add_argument("--lambda_", type=float, default=0.95, help="TD(lambda) trace parameter for critic (default: 0.95)")
    parser.add_argument("--critic-initial-alpha", type=float, default=1e-4, help="Initial SwiftTD step size alpha (default: 1e-4)")
    parser.add_argument("--critic-eps", type=float, default=1e-8, help="SwiftTD epsilon threshold (default: 1e-5)")
    parser.add_argument("--critic-max-step-size", type=float, default=0.01, help="SwiftTD max step size eta (default: 0.01)")
    parser.add_argument("--critic-step-decay", type=float, default=0.9, help="SwiftTD step size decay factor (default: 0.9)")
    parser.add_argument("--critic-meta-step-size", type=float, default=1e-4, help="SwiftTD meta step size (default: 1e-4)")
    parser.add_argument("--critic-eta-min", type=float, default=1e-6, help="SwiftTD minimum eta (default: 1e-6)")
    parser.add_argument("--critic-feature-scale", type=float, default=0.01, help="Scale factor applied to features before SwiftTD (default: 0.01)")
    parser.add_argument("--fail-on-nonfinite", action="store_true", help="Raise on non-finite critic outputs instead of continuing")
    parser.add_argument("--n-stack", type=int, default=4, help="Number of frames to stack (default: 4)")
    parser.add_argument("--input-size", type=int, default=128, help="Input image size (default: 128)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--no-videos", action="store_true", help="Disable video recording (default: videos enabled)")
    parser.add_argument("--video-freq", type=int, default=30, help="Record video every N episodes (default: 50)")
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
    parser.add_argument("--num-envs", type=int, default=4, help="Number of vectorized environments (default: 4)")
    args = parser.parse_args()

    # Generate run name (mode-name-timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}-swifttd-{args.mode}-{generate_slug(2)}"

    # Create experiment directory using run_name
    env_dir_name = args.env.replace('/', '_')
    experiment_dir = os.path.join(args.output_dir, args.mode, env_dir_name, run_name)

    # Create directory structure
    os.makedirs(os.path.join(experiment_dir, "logs", "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "models", "checkpoints"), exist_ok=True)
    if not args.no_videos:
        os.makedirs(os.path.join(experiment_dir, "videos"), exist_ok=True)

    # Save configuration
    config_path = os.path.join(experiment_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"Run name: {run_name}\n")
        f.write(f"Algorithm: SwiftTD\n")
        f.write(f"Training mode: {args.mode}\n")
        f.write(f"Environment: {args.env}\n")
        f.write(f"Total timesteps: {args.timesteps}\n")
        f.write(f"Latency simulation: {args.mode == 'sim_lat'}\n")
        f.write(f"Reduce action set: {args.reduce_action_set}\n")
        f.write(f"Num envs: {args.num_envs}\n")
        f.write(f"\n# Training Hyperparameters\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Entropy coef: {args.entropy_coef}\n")
        f.write(f"Gamma: {args.gamma}\n")
        f.write(f"Lambda: {args.lambda_}\n")
        f.write(f"Critic initial alpha: {args.critic_initial_alpha}\n")
        f.write(f"Critic eps: {args.critic_eps}\n")
        f.write(f"Critic max step size: {args.critic_max_step_size}\n")
        f.write(f"Critic step decay: {args.critic_step_decay}\n")
        f.write(f"Critic meta step size: {args.critic_meta_step_size}\n")
        f.write(f"Critic eta min: {args.critic_eta_min}\n")
        f.write(f"Critic feature scale: {args.critic_feature_scale}\n")
        f.write(f"Fail on non-finite: {args.fail_on_nonfinite}\n")
        f.write(f"N stack: {args.n_stack}\n")
        f.write(f"Input size: {args.input_size}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"\n# Video Recording\n")
        f.write(f"Record videos: {not args.no_videos}\n")
        if not args.no_videos:
            f.write(f"Video frequency: {args.video_freq} episodes\n")
        if args.load_model:
            f.write(f"\n# Model Loading\n")
            f.write(f"Loaded from: {args.load_model}\n")

    # Train agent
    agent, model_path = train_agent(
        env_name=args.env,
        total_timesteps=args.timesteps,
        simulate_latency=(args.mode == "sim_lat"),
        latency_model_dir=args.latency_model_dir,
        experiment_dir=experiment_dir,
        device=args.device,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        gamma=args.gamma,
        lambda_=args.lambda_,
        critic_initial_alpha=args.critic_initial_alpha,
        critic_eps=args.critic_eps,
        critic_max_step_size=args.critic_max_step_size,
        critic_step_decay=args.critic_step_decay,
        critic_meta_step_size=args.critic_meta_step_size,
        critic_eta_min=args.critic_eta_min,
        critic_feature_scale=args.critic_feature_scale,
        fail_on_nonfinite=args.fail_on_nonfinite,
        num_envs=args.num_envs,
        seed=args.seed,
        load_model_path=args.load_model,
        record_videos=not args.no_videos,
        video_freq=args.video_freq,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=run_name,
        reduce_action_set=args.reduce_action_set,
        n_stack=args.n_stack,
        input_size=args.input_size,
    )

    print(f"\n{'=' * 60}")
    print(f"Training completed successfully!")
    print(f"{'=' * 60}")
    print(f"Model saved at: {model_path}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"View training progress:")
    print(f"  tensorboard --logdir {os.path.join(experiment_dir, 'logs', 'tensorboard')}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
