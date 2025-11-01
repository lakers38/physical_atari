#!/usr/bin/env python3
"""
Train R2D2 agent in Gymnasium with optional hardware latency simulation

This script trains an R2D2 agent on Atari games with two modes:
- sim: Pure simulation (no latency) - fast baseline training
- sim_lat: Simulation with LatencyModel - simulates real hardware delays

Single-process implementation for simplicity and cross-platform compatibility.
"""

import argparse
import os
import sys
import random
from datetime import datetime
from collections import deque

import gymnasium as gym
import ale_py
import numpy as np
import torch
from coolname import generate_slug
from gymnasium.wrappers import RecordVideo

# Import the latency model
sys.path.append(os.path.join(os.path.dirname(__file__), 'latency_wrap'))
from wrapper_v0_2 import LatencyModel

# Import R2D2 components
from r2d2.model import Network, AgentState
from r2d2 import config as r2d2_config

# Register ALE environments
gym.register_envs(ale_py)

# WandB integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class SimpleReplayBuffer:
    """Simple replay buffer for single-process R2D2 training"""

    def __init__(self, capacity=100000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, last_action, last_reward, action, reward, next_obs, done, hidden_state, q_values):
        """Add a transition to the buffer"""
        self.buffer.append({
            'obs': obs,
            'last_action': last_action,
            'last_reward': last_reward,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
            'hidden_state': hidden_state,
            'q_values': q_values
        })

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch

    def __len__(self):
        return len(self.buffer)


class AtariWrapper(gym.Wrapper):
    """Wrapper to preprocess Atari observations for R2D2"""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._preprocess(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._preprocess(obs), reward, terminated, truncated, info

    def _preprocess(self, obs):
        """Convert to grayscale and resize to 84x84"""
        import cv2
        # obs is (210, 160, 3) RGB
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized


class LatencyWrapper(gym.Wrapper):
    """Wrapper that applies LatencyModel to simulate hardware latency"""

    def __init__(self, env, latency_model_dir="./latency_wrap"):
        super().__init__(env)
        self.latency_model = LatencyModel(directory_with_weights=latency_model_dir)
        print(f"[LatencyWrapper] Initialized with weights from {latency_model_dir}")

    def step(self, action):
        """Apply latency model to action"""
        ale_action = ale_py.Action(int(action))
        delayed_action = self.latency_model.act(ale_action)
        return self.env.step(int(delayed_action))

    def reset(self, **kwargs):
        """Reset latency model state"""
        self.latency_model.action_queue = []
        for _ in range(30):
            self.latency_model.action_queue.append(
                self.latency_model._LatencyModel__one_hot_encode(0, 0, 36)
            )
        self.latency_model.last_action = 0
        return self.env.reset(**kwargs)


def create_env(env_name, simulate_latency=False, latency_model_dir="./latency_wrap",
               record_video=False, video_dir=None, video_freq=10, video_length=1000):
    """Create Atari environment with preprocessing and optional latency"""
    env = gym.make(env_name, full_action_space=True, render_mode="rgb_array" if record_video else None)

    # Apply latency wrapper before preprocessing
    if simulate_latency:
        env = LatencyWrapper(env, latency_model_dir)

    # Apply Atari preprocessing
    env = AtariWrapper(env)

    # Apply video recording (after preprocessing)
    if record_video and video_dir:
        env = RecordVideo(env, video_dir, episode_trigger=lambda x: x % video_freq == 0)

    return env


def train_agent(
    env_name,
    total_timesteps,
    simulate_latency,
    latency_model_dir,
    experiment_dir,
    device,
    learning_rate,
    gamma,
    base_explore_eps,
    batch_size,
    buffer_capacity,
    learning_starts,
    train_freq,
    target_update_freq,
    save_freq,
    record_videos,
    video_freq,
    video_length,
    use_wandb,
    wandb_project,
    wandb_entity,
    wandb_run_name,
    seed
):
    """
    Train R2D2 agent in single process

    Args:
        env_name: Atari environment name
        total_timesteps: Total training timesteps
        simulate_latency: If True, apply LatencyModel
        latency_model_dir: Directory with LatencyModel weights
        experiment_dir: Directory to save results
        device: "cuda" or "cpu"
        learning_rate: Learning rate
        gamma: Discount factor
        base_explore_eps: Fixed epsilon for epsilon-greedy exploration
        batch_size: Batch size for training
        buffer_capacity: Replay buffer capacity
        learning_starts: Start training after N steps
        train_freq: Train every N steps
        target_update_freq: Update target network every N steps
        save_freq: Save model every N steps
        record_videos: If True, record gameplay videos
        video_freq: Record video every N episodes
        video_length: Maximum frames per video
        use_wandb: If True, log to Weights & Biases
        wandb_project: WandB project name
        wandb_entity: WandB entity/team name
        wandb_run_name: WandB run name
        seed: Random seed

    Returns:
        Trained model and save path
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize WandB if requested
    wandb_run = None
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: WandB not installed. Install with: pip install wandb")
            print("Continuing without WandB logging...")
        else:
            config_dict = {
                "env_name": env_name,
                "total_timesteps": total_timesteps,
                "simulate_latency": simulate_latency,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "batch_size": batch_size,
                "buffer_capacity": buffer_capacity,
                "learning_starts": learning_starts,
                "algorithm": "R2D2",
                "training_mode": "sim_lat" if simulate_latency else "sim",
                "seed": seed,
            }

            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config=config_dict,
                save_code=True,
            )
            print(f"[WandB] Initialized run: {wandb_run.name}")
            print(f"[WandB] View at: {wandb_run.url}")

    # Create environment
    video_dir = os.path.join(experiment_dir, "videos") if experiment_dir and record_videos else None
    env = create_env(
        env_name,
        simulate_latency=simulate_latency,
        latency_model_dir=latency_model_dir,
        record_video=record_videos,
        video_dir=video_dir,
        video_freq=video_freq,
        video_length=video_length
    )
    action_dim = env.action_space.n

    # Create networks
    device_obj = torch.device(device)
    online_net = Network(action_dim).to(device_obj)
    target_net = Network(action_dim).to(device_obj)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=learning_rate, eps=r2d2_config.eps_adam)
    loss_fn = torch.nn.MSELoss()

    # Create replay buffer
    replay_buffer = SimpleReplayBuffer(capacity=buffer_capacity)

    # Training state
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    losses = []

    mode_name = "sim_lat (with LatencyModel)" if simulate_latency else "sim (no latency)"
    print(f"\n{'='*60}")
    print(f"Starting Training: {mode_name}")
    print(f"{'='*60}")
    print(f"Environment: {env_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Latency simulation: {simulate_latency}")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Buffer capacity: {buffer_capacity}")
    if use_wandb and wandb_run:
        print(f"WandB: {wandb_run.url}")
    print(f"{'='*60}\n")

    # Reset environment
    obs, _ = env.reset()
    agent_state = AgentState(torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).float(), action_dim)
    hidden_state = None
    last_action = 0
    last_reward = 0.0

    # Training loop
    for step in range(total_timesteps):
        # Select action
        agent_state.update(obs, last_action, last_reward, hidden_state)

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).float().to(device_obj) / 255.0
            agent_state.obs = obs_tensor
            agent_state.last_action = torch.zeros((1, action_dim)).to(device_obj)
            agent_state.last_action[0, last_action] = 1.0
            agent_state.last_reward = torch.tensor([[last_reward]]).float().to(device_obj)
            agent_state.hidden_state = hidden_state

            q_values, new_hidden = online_net(agent_state)

        # Epsilon-greedy exploration with fixed epsilon
        if random.random() < base_explore_eps:
            action = env.action_space.sample()
        else:
            action = torch.argmax(q_values).item()  # Flatten and take argmax

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        replay_buffer.add(
            obs, last_action, last_reward, action, reward,
            next_obs, done,
            (new_hidden[0].cpu().numpy(), new_hidden[1].cpu().numpy()) if new_hidden else None,
            q_values.cpu().numpy()
        )

        # Update state
        obs = next_obs
        last_action = action
        last_reward = reward
        hidden_state = new_hidden
        current_episode_reward += float(reward)
        current_episode_length += 1

        # Reset on done
        if done:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)

            if wandb_run:
                wandb_run.log({
                    "episode/reward": current_episode_reward,
                    "episode/length": current_episode_length,
                    "train/epsilon": base_explore_eps,
                }, step=step)

            obs, _ = env.reset()
            hidden_state = None
            last_action = 0
            last_reward = 0.0
            current_episode_reward = 0
            current_episode_length = 0

        # Training
        if step >= learning_starts and step % train_freq == 0 and len(replay_buffer) >= batch_size:
            online_net.train()

            # Sample batch
            batch = replay_buffer.sample(batch_size)

            # Prepare batch tensors (simplified - just use current Q-values)
            obs_batch = torch.from_numpy(np.stack([t['obs'] for t in batch])).unsqueeze(1).float().to(device_obj) / 255.0
            action_batch = torch.tensor([t['action'] for t in batch]).long().to(device_obj)
            reward_batch = torch.tensor([t['reward'] for t in batch]).float().to(device_obj)
            next_obs_batch = torch.from_numpy(np.stack([t['next_obs'] for t in batch])).unsqueeze(1).float().to(device_obj) / 255.0
            done_batch = torch.tensor([t['done'] for t in batch]).float().to(device_obj)

            # Create simple agent states for batch (no LSTM for now - simplified)
            with torch.no_grad():
                # Target Q-values for next states
                next_q_values = []
                for i in range(batch_size):
                    next_state = AgentState(next_obs_batch[i:i+1], action_dim)
                    next_q, _ = target_net(next_state)
                    next_q_values.append(next_q.max().item())
                next_q_batch = torch.tensor(next_q_values).to(device_obj)

            # Current Q-values
            current_q_values = []
            for i in range(batch_size):
                state = AgentState(obs_batch[i:i+1], action_dim)
                q, _ = online_net(state)
                current_q_values.append(q[0, action_batch[i]])
            current_q_batch = torch.stack(current_q_values)

            # Compute target Q-values
            target_q_batch = reward_batch + gamma * next_q_batch * (1 - done_batch)

            # Compute loss
            loss = loss_fn(current_q_batch, target_q_batch)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), 40)
            optimizer.step()

            losses.append(loss.item())

            if wandb_run:
                wandb_run.log({
                    "train/loss": loss.item(),
                    "train/q_value_mean": current_q_batch.mean().item(),
                    "train/target_q_mean": target_q_batch.mean().item(),
                }, step=step)

            online_net.eval()

        # Update target network
        if step % target_update_freq == 0 and step > 0:
            target_net.load_state_dict(online_net.state_dict())
            print(f"Step {step}: Updated target network")

        # Save model
        if step % save_freq == 0 and step > 0:
            if experiment_dir:
                save_path = os.path.join(experiment_dir, "models", f"checkpoint_{step}.pth")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(online_net.state_dict(), save_path)
                print(f"Step {step}: Saved model to {save_path}")

        # Logging
        if step % 1000 == 0 and step > 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Step {step}/{total_timesteps} | Eps: {base_explore_eps:.3f} | "
                  f"Avg Reward: {avg_reward:.1f} | Avg Loss: {avg_loss:.4f} | "
                  f"Episodes: {len(episode_rewards)}")

            if wandb_run:
                wandb_run.log({
                    "metrics/avg_reward_100ep": avg_reward,
                    "metrics/avg_loss_100steps": avg_loss,
                    "metrics/total_episodes": len(episode_rewards),
                    "metrics/buffer_size": len(replay_buffer),
                }, step=step)

    # Save final model
    if experiment_dir:
        model_save_path = os.path.join(experiment_dir, "models", "final_model.pth")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(online_net.state_dict(), model_save_path)
        print(f"\nTraining complete! Model saved to: {model_save_path}")
    else:
        model_save_path = None
        print(f"\nTraining complete!")

    # Finish WandB run
    if use_wandb and wandb_run is not None:
        wandb_run.finish()
        print(f"[WandB] Run finished and uploaded")

    env.close()
    return online_net, model_save_path

def main():
    parser = argparse.ArgumentParser(
        description="Train R2D2 agent in Gymnasium simulation with optional latency"
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
        "--learning-rate",
        type=float,
        default=r2d2_config.lr,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=r2d2_config.gamma,
        help="Discount factor (default: 0.997)"
    )
    parser.add_argument(
        "--base-explore-eps",
        type=float,
        default=r2d2_config.base_explore_eps,
        help="Fixed epsilon for epsilon-greedy exploration (default: 0.4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=100000,
        help="Replay buffer capacity (default: 100k)"
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=10000,
        help="Start training after N steps (default: 10k)"
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=4,
        help="Train every N steps (default: 4)"
    )
    parser.add_argument(
        "--target-update-freq",
        type=int,
        default=2500,
        help="Update target network every N steps (default: 2500)"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50000,
        help="Save model every N steps (default: 50k)"
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Disable video recording (default: videos enabled)"
    )
    parser.add_argument(
        "--video-freq",
        type=int,
        default=100,
        help="Record video every N episodes (default: 10)"
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=1000,
        help="Maximum frames per video (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
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
    run_name = f"{args.mode}-r2d2-{generate_slug(2)}-{timestamp}"

    # Create experiment directory using run_name
    env_dir_name = args.env.replace('/', '_')
    experiment_dir = os.path.join(args.output_dir, args.mode, env_dir_name, run_name)

    # Create directory structure
    os.makedirs(os.path.join(experiment_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    if not args.no_videos:
        os.makedirs(os.path.join(experiment_dir, "videos"), exist_ok=True)

    # Save configuration
    config_path = os.path.join(experiment_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"Run name: {run_name}\n")
        f.write(f"Algorithm: R2D2 (single-process)\n")
        f.write(f"Training mode: {args.mode}\n")
        f.write(f"Environment: {args.env}\n")
        f.write(f"Total timesteps: {args.timesteps}\n")
        f.write(f"Latency simulation: {args.mode == 'sim_lat'}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Adam epsilon: {r2d2_config.eps_adam}\n")
        f.write(f"Gamma: {args.gamma}\n")
        f.write(f"Exploration epsilon: {args.base_explore_eps}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Buffer capacity: {args.buffer_capacity}\n")
        f.write(f"Learning starts: {args.learning_starts}\n")
        f.write(f"Record videos: {not args.no_videos}\n")
        if not args.no_videos:
            f.write(f"Video frequency: {args.video_freq} episodes\n")
            f.write(f"Video length: {args.video_length} frames\n")
        f.write(f"Seed: {args.seed}\n")

    # Train agent
    model, model_path = train_agent(
        env_name=args.env,
        total_timesteps=args.timesteps,
        simulate_latency=(args.mode == "sim_lat"),
        latency_model_dir=args.latency_model_dir,
        experiment_dir=experiment_dir,
        device=args.device,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        base_explore_eps=args.base_explore_eps,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_freq=args.target_update_freq,
        save_freq=args.save_freq,
        record_videos=not args.no_videos,
        video_freq=args.video_freq,
        video_length=args.video_length,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=run_name,
        seed=args.seed
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
    main()
