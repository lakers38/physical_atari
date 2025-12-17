#!/usr/bin/env python3
"""
Train a simple Soft Actor-Critic style agent in Gymnasium with optional hardware latency simulation.

This script trains the agent on Atari games with two modes:
- sim: Pure simulation (no latency) - fast baseline training
- sim_lat: Simulation with LatencyModel - simulates real hardware delays

Flow (sim_lat mode):
    obs = env.step()
    action = SACAgent.select_actions(obs)
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

import ale_py
import gymnasium as gym
import numpy as np
from coolname import generate_slug
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from scipy.ndimage import zoom
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, NoopResetEnv
from torch.utils.tensorboard import SummaryWriter

import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'latency_wrap'))
from wrapper_v0_2 import LatencyModel

sys.path.append(os.path.dirname(__file__))
from sac import ReplayBuffer, SACAgent

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
            self.action_mapping = [2, 5, 4, 3]
            print(f"[ActionSetWrapper] Restricting {game_name} to 4 directional actions only")
        else:
            self.action_mapping = None

        if self.action_mapping is not None:
            self.action_space = spaces.Discrete(len(self.action_mapping))
            print(
                f"[ActionSetWrapper] Action space reduced to {len(self.action_mapping)} actions: {self.action_mapping}"
            )

    def step(self, action):
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
        ale_action = ale_py.Action(int(action))
        delayed_action = self.latency_model.act(ale_action, allowed_actions=self.allowed_actions)

        return self.env.step(int(delayed_action))

    def reset(self, **kwargs):
        """Reset the environment and latency model state"""
        self.latency_model.action_queue = []
        for _ in range(30):
            self.latency_model.action_queue.append(self.latency_model._LatencyModel__one_hot_encode(0, 0, 36))
        self.latency_model.last_action = 0

        return self.env.reset(**kwargs)


def create_single_atari_env(
    env_name: str,
    seed: int,
    simulate_latency: bool,
    latency_model_dir: str,
    reduce_action_set: int,
    n_stack: int = 4,
    input_size: int = 128,
    video_path: Optional[str] = None,
    video_freq: int = 50,
):
    """Create single Atari environment with preprocessing and optional latency."""
    # Never override reduce_action_set because of latency; instead, mask latency outputs.
    use_full_action_space = reduce_action_set in (0, 2)

    env = gym.make(
        env_name,
        obs_type="rgb",
        render_mode="rgb_array" if video_path else None,
        full_action_space=use_full_action_space,
    )

    allowed_actions = None
    if reduce_action_set == 1:
        allowed_actions = list(range(env.action_space.n))
    elif reduce_action_set == 2:
        allowed_actions = [2, 5, 4, 3]

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    if simulate_latency:
        print(f"[sim_lat] Applying latency simulation to {env_name}")
        env = LatencyWrapper(env, latency_model_dir, allowed_actions=allowed_actions)

    if reduce_action_set == 2:
        env = ActionSetWrapper(env, reduce_action_set, env_name)

    env = PreprocessWrapper(env, frame_size=input_size)
    env = StackFrames(env, num_stack=n_stack)

    env = RecordEpisodeStatistics(env)

    if video_path:
        print(f"[Video] Recording videos every {video_freq} episodes to {video_path}")
        env = RecordVideo(
            env,
            video_folder=video_path,
            episode_trigger=lambda ep: ep % video_freq == 0,
            name_prefix="training",
            video_length=500,
        )

    env.reset(seed=seed)
    return env


def evaluate_agent(agent: SACAgent, eval_env: gym.Env, n_episodes: int = 10) -> float:
    """Evaluate agent for n_episodes and return mean reward."""
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _info = eval_env.reset()
        agent.start_episodes(obs[np.newaxis])

        episode_reward = 0
        done = False

        while not done:
            actions, _log_probs, _entropy, _feats = agent.select_actions(obs[np.newaxis])
            next_obs, reward, terminated, truncated, _info = eval_env.step(actions[0])
            done = terminated or truncated
            episode_reward += reward

            if done:
                agent.reset_done(done, obs[np.newaxis])
            else:
                obs = next_obs

        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)


def train_loop(
    agent: SACAgent,
    env: gym.Env,
    eval_env: gym.Env,
    total_timesteps: int,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    learning_starts: int,
    train_freq: int,
    gradient_steps: int,
    tensorboard_writer: SummaryWriter,
    checkpoint_dir: str,
    model_name: str,
    wandb_run=None,
):
    """Custom training loop for SACAgent."""
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    episode_values = []
    episode_rewards_stream = []
    episode_count = 0

    recent_advantages = []
    recent_entropies = []
    recent_actor_losses = []
    recent_log_probs = []
    recent_value_losses = []
    recent_total_losses = []
    recent_value_preds = []
    recent_value_next = []
    recent_alphas = []

    obs, _info = env.reset()
    agent.start_episodes(obs[np.newaxis])

    start_time = time.time()

    for step in range(total_timesteps):
        actions, _, _, _ = agent.select_actions(obs[np.newaxis])
        action = actions[0]

        next_obs, reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated

        reward_clipped = np.clip(reward, -1.0, 1.0)

        replay_buffer.add(obs, action, reward_clipped, next_obs, done)

        metrics = None
        if replay_buffer.size >= batch_size and step >= learning_starts and step % train_freq == 0:
            for _ in range(gradient_steps):
                batch = replay_buffer.sample(batch_size)
                metrics = agent.update(*batch)

                recent_advantages.append(metrics["advantage"])
                recent_entropies.append(metrics["policy_entropy"])
                recent_actor_losses.append(metrics["actor_loss"])
                recent_log_probs.append(metrics["log_prob_mean"])
                recent_value_losses.append(metrics["value_loss"])
                recent_total_losses.append(metrics["total_loss"])
                recent_value_preds.append(metrics["value_pred"])
                recent_value_next.append(metrics["value_next"])
                recent_alphas.append(metrics["alpha"])

        current_episode_reward += reward
        current_episode_length += 1
        if metrics is not None:
            episode_values.append(metrics["value_pred"])
        episode_rewards_stream.append(reward)

        if done:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)

            returns = []
            G = 0.0
            for r in reversed(episode_rewards_stream):
                G = r + agent.gamma * G
                returns.append(G)
            returns = list(reversed(returns))
            if len(returns) == len(episode_values):
                squared_errors = [(v - g) ** 2 for v, g in zip(episode_values, returns)]
                lifetime_err_ep = float(np.mean(squared_errors)) if squared_errors else 0.0
                tensorboard_writer.add_scalar("train/episode_lifetime_error", lifetime_err_ep, step)
                if wandb_run is not None:
                    wandb.log({"train/episode_lifetime_error": lifetime_err_ep}, step=step)

            episode_count += 1

            tensorboard_writer.add_scalar("train/episode_reward", current_episode_reward, step)
            tensorboard_writer.add_scalar("train/episode_length", current_episode_length, step)

            current_episode_reward = 0
            current_episode_length = 0
            obs, _info = env.reset()
            agent.reset_done(done, obs[np.newaxis])
            episode_values = []
            episode_rewards_stream = []
        else:
            obs = next_obs

        if step % 1000 == 0 and len(episode_rewards) > 0 and len(recent_advantages) > 0:
            mean_reward = np.mean(episode_rewards[-100:])
            mean_length = np.mean(episode_lengths[-100:])
            max_reward = np.max(episode_rewards[-100:]) if len(episode_rewards) > 0 else 0
            min_reward = np.min(episode_rewards[-100:]) if len(episode_rewards) > 0 else 0
            fps = step / (time.time() - start_time)

            mean_advantage = np.mean(recent_advantages[-1000:])
            std_advantage = np.std(recent_advantages[-1000:])
            mean_entropy = np.mean(recent_entropies[-1000:])
            mean_actor_loss = np.mean(recent_actor_losses[-1000:])
            mean_log_prob = np.mean(recent_log_probs[-1000:])
            mean_value_loss = np.mean(recent_value_losses[-1000:])
            mean_total_loss = np.mean(recent_total_losses[-1000:])
            mean_value_pred = np.mean(recent_value_preds[-1000:])
            mean_value_next = np.mean(recent_value_next[-1000:])
            mean_alpha = np.mean(recent_alphas[-1000:])

            print(
                f"Step {step:,} | Ep: {episode_count} | "
                f"Reward: {mean_reward:6.2f} (max:{max_reward:5.1f} min:{min_reward:5.1f}) | "
                f"Len: {mean_length:5.1f} | "
                f"Adv: {mean_advantage:6.3f}±{std_advantage:.3f} | "
                f"Ent: {mean_entropy:.3f} | "
                f"Alpha: {mean_alpha:.4f} | "
                f"ValLoss: {mean_value_loss:.4f} | "
                f"TotLoss: {mean_total_loss:.4f} | "
                f"ActLoss: {mean_actor_loss:.4f} | "
                f"FPS: {fps:5.1f}"
            )

            tensorboard_writer.add_scalar("train/mean_reward_100ep", mean_reward, step)
            tensorboard_writer.add_scalar("train/max_reward_100ep", max_reward, step)
            tensorboard_writer.add_scalar("train/min_reward_100ep", min_reward, step)
            tensorboard_writer.add_scalar("train/mean_length_100ep", mean_length, step)
            tensorboard_writer.add_scalar("train/fps", fps, step)

            tensorboard_writer.add_scalar("train/actor_loss", mean_actor_loss, step)
            tensorboard_writer.add_scalar("train/value_loss", mean_value_loss, step)
            tensorboard_writer.add_scalar("train/total_loss", mean_total_loss, step)
            tensorboard_writer.add_scalar("train/advantage_mean", mean_advantage, step)
            tensorboard_writer.add_scalar("train/advantage_std", std_advantage, step)
            tensorboard_writer.add_scalar("train/entropy", mean_entropy, step)
            tensorboard_writer.add_scalar("train/alpha", mean_alpha, step)
            tensorboard_writer.add_scalar("train/log_prob", mean_log_prob, step)
            tensorboard_writer.add_scalar("train/value_pred_mean", mean_value_pred, step)
            tensorboard_writer.add_scalar("train/value_next_mean", mean_value_next, step)

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
                        "train/alpha": mean_alpha,
                        "train/log_prob": mean_log_prob,
                        "train/value_loss": mean_value_loss,
                        "train/total_loss": mean_total_loss,
                        "train/value_pred_mean": mean_value_pred,
                        "train/value_next_mean": mean_value_next,
                    },
                    step=step,
                )

        if step % 10000 == 0 and step > 0:
            eval_reward = evaluate_agent(agent, eval_env, n_episodes=10)
            tensorboard_writer.add_scalar("eval/mean_reward", eval_reward, step)
            print(f"  Eval @ {step:,}: {eval_reward:.2f}")

        if step % 50000 == 0 and step > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_{step}")
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
    seed,
    load_model_path,
    record_videos,
    video_freq,
    buffer_size,
    batch_size,
    learning_starts,
    train_freq,
    gradient_steps,
    use_wandb,
    wandb_project,
    wandb_entity,
    wandb_run_name,
    reduce_action_set,
    n_stack,
    input_size=128,
    auto_entropy_tuning=True,
):
    """
    Train soft actor-critic style agent with optional latency simulation.

    Args:
        env_name: Atari environment name
        total_timesteps: Total training timesteps
        simulate_latency: If True, apply LatencyModel
        latency_model_dir: Directory with LatencyModel weights
        experiment_dir: Directory to save results
        device: "cuda", "cpu", or "mps"
        learning_rate: Learning rate
        entropy_coef: Entropy bonus coefficient
        gamma: Discount factor
        seed: Random seed
        load_model_path: Path to pre-trained model to continue training
        record_videos: If True, record gameplay videos
        video_freq: Record video every N episodes
        buffer_size: Replay buffer capacity
        batch_size: Minibatch size for gradient steps
        learning_starts: Number of steps to collect transitions before updates
        train_freq: Environment steps per training phase
        gradient_steps: Number of gradient updates per training phase
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

    wandb_run = None
    if use_wandb:
        config = {
            "env_name": env_name,
            "total_timesteps": total_timesteps,
            "simulate_latency": simulate_latency,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "ent_coef": entropy_coef,
            "device": device,
            "algorithm": "SoftActorCritic",
            "training_mode": "sim_lat" if simulate_latency else "sim",
            "latency_enabled": simulate_latency,
            "input_size": input_size,
            "feature_dim": 512,
            "n_stack": n_stack,
            "reduce_action_set": reduce_action_set,
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
        print(f"[WandB] Initialized run: {wandb_run.name}")
        print(f"[WandB] View at: {wandb_run.url}")

    video_path = os.path.join(experiment_dir, "videos") if record_videos else None

    env = create_single_atari_env(
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

    eval_env = create_single_atari_env(
        env_name=env_name,
        seed=seed + 1000,
        simulate_latency=simulate_latency,
        latency_model_dir=latency_model_dir,
        reduce_action_set=reduce_action_set,
        n_stack=n_stack,
        input_size=input_size,
        video_path=None,
        video_freq=0,
    )

    agent = SACAgent(
        num_actions=env.action_space.n,
        feature_dim=512,
        actor_hidden_dim=256,
        value_hidden_dim=256,
        n_stack=n_stack,
        input_size=input_size,
        device=device,
        gamma=gamma,
        learning_rate=learning_rate,
        entropy_coef=entropy_coef,
        auto_entropy_tuning=auto_entropy_tuning,
    )

    if load_model_path and os.path.exists(load_model_path):
        print(f"Loading pre-trained model from {load_model_path}")
        agent.load(load_model_path)

    replay_buffer = ReplayBuffer(
        capacity=buffer_size,
        obs_shape=(input_size, input_size, n_stack),
    )

    tensorboard_log_dir = os.path.join(experiment_dir, "logs", "tensorboard")
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    checkpoint_dir = os.path.join(experiment_dir, "models", "checkpoints")
    model_name = f"SAC_{env_name.replace('/', '_')}"

    mode_name = "sim_lat (with LatencyModel)" if simulate_latency else "sim (no latency)"
    print(f"\n{'=' * 60}")
    print(f"Starting Training: {mode_name}")
    print(f"{'=' * 60}")
    print(f"Environment: {env_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Device: {device}")
    print(f"Learning rate (actor/CNN): {learning_rate}")
    if auto_entropy_tuning:
        print(f"Auto entropy tuning: ENABLED (target entropy: {agent.target_entropy:.4f})")
    else:
        print(f"Entropy coef (fixed): {entropy_coef}")
    print(f"Replay buffer size: {buffer_size} | Batch size: {batch_size}")
    print(
        f"Learning starts after: {learning_starts} steps | Train freq: {train_freq} | Gradient steps: {gradient_steps}"
    )
    print(f"Gamma: {gamma}")
    if use_wandb and wandb_run:
        print(f"WandB: {wandb_run.url}")
    print(f"{'=' * 60}\n")

    train_loop(
        agent=agent,
        env=env,
        eval_env=eval_env,
        total_timesteps=total_timesteps,
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        learning_starts=learning_starts,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        tensorboard_writer=writer,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        wandb_run=wandb_run,
    )

    final_path = os.path.join(experiment_dir, "models", "final_model")
    agent.save(final_path)
    print(f"\nTraining complete! Model saved to: {final_path}")

    writer.close()
    env.close()
    eval_env.close()

    if use_wandb and wandb_run is not None:
        wandb_run.finish()
        print("[WandB] Run finished and uploaded")

    return agent, final_path


def main():
    parser = argparse.ArgumentParser(
        description="Train a soft actor-critic style agent in Gymnasium simulation with optional latency"
    )
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
    parser.add_argument("--output-dir", type=str, default="outputs/sac/", help="Base directory for outputs")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Device to use for training"
    )
    parser.add_argument("--load-model", type=str, default=None, help="Path to pre-trained model to continue training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for actor/CNN (default: 1e-4)")
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient (default: 0.01, only used if --no-auto-entropy-tuning)",
    )
    parser.add_argument(
        "--no-auto-entropy-tuning",
        action="store_true",
        help="Disable automatic entropy tuning (use fixed entropy_coef instead)",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for critic (default: 0.99)")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer size (default: 100k)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for updates (default: 256)")
    parser.add_argument(
        "--learning-starts", type=int, default=1_000, help="Steps to collect before starting updates (default: 1,000)"
    )
    parser.add_argument(
        "--train-freq", type=int, default=1, help="Environment steps between training phases (default: 1)"
    )
    parser.add_argument("--gradient-steps", type=int, default=1, help="Gradient steps per training phase (default: 1)")
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
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}-sac-{args.mode}-{generate_slug(2)}"

    env_dir_name = args.env.replace('/', '_')
    experiment_dir = os.path.join(args.output_dir, args.mode, env_dir_name, run_name)

    os.makedirs(os.path.join(experiment_dir, "logs", "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "models", "checkpoints"), exist_ok=True)
    if not args.no_videos:
        os.makedirs(os.path.join(experiment_dir, "videos"), exist_ok=True)

    config_path = os.path.join(experiment_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"Run name: {run_name}\n")
        f.write("Algorithm: SoftActorCritic\n")
        f.write(f"Training mode: {args.mode}\n")
        f.write(f"Environment: {args.env}\n")
        f.write(f"Total timesteps: {args.timesteps}\n")
        f.write(f"Latency simulation: {args.mode == 'sim_lat'}\n")
        f.write(f"Reduce action set: {args.reduce_action_set}\n")
        f.write("\n# Training Hyperparameters\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Entropy coef: {args.entropy_coef}\n")
        f.write(f"Gamma: {args.gamma}\n")
        f.write(f"Buffer size: {args.buffer_size}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning starts: {args.learning_starts}\n")
        f.write(f"Train freq: {args.train_freq}\n")
        f.write(f"Gradient steps: {args.gradient_steps}\n")
        f.write(f"N stack: {args.n_stack}\n")
        f.write(f"Input size: {args.input_size}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write("\n# Video Recording\n")
        f.write(f"Record videos: {not args.no_videos}\n")
        if not args.no_videos:
            f.write(f"Video frequency: {args.video_freq} episodes\n")
        if args.load_model:
            f.write("\n# Model Loading\n")
            f.write(f"Loaded from: {args.load_model}\n")

    _agent, model_path = train_agent(
        env_name=args.env,
        total_timesteps=args.timesteps,
        simulate_latency=(args.mode == "sim_lat"),
        latency_model_dir=args.latency_model_dir,
        experiment_dir=experiment_dir,
        device=args.device,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        gamma=args.gamma,
        seed=args.seed,
        load_model_path=args.load_model,
        record_videos=not args.no_videos,
        video_freq=args.video_freq,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=run_name,
        reduce_action_set=args.reduce_action_set,
        n_stack=args.n_stack,
        input_size=args.input_size,
        auto_entropy_tuning=not args.no_auto_entropy_tuning,
    )

    print(f"\n{'=' * 60}")
    print("Training completed successfully!")
    print(f"{'=' * 60}")
    print(f"Model saved at: {model_path}")
    print(f"Experiment directory: {experiment_dir}")
    print("View training progress:")
    print(f"  tensorboard --logdir {os.path.join(experiment_dir, 'logs', 'tensorboard')}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
