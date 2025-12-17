"""Learner for R2D2: performs gradient updates using prioritized replay"""

import os
import time
import threading
import glob
from copy import deepcopy
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import wandb
from .model import Network
from .actor import calculate_mixed_td_errors
from . import config


class Learner:
    """
    Learner process: samples from replay buffer and trains the network

    Uses Double DQN with value rescaling and recurrent network
    """

    def __init__(self, batch_queue, priority_queue, stats_queue, model,
                 grad_norm: int = config.grad_norm,
                 lr: float = config.lr,
                 eps: float = config.eps_adam,
                 game_name: str = config.game_name,
                 target_net_update_interval: int = config.target_net_update_interval,
                 save_interval: int = config.save_interval,
                 models_dir: str = 'models',
                 use_wandb: bool = False,
                 env_name: Optional[str] = None,
                 video_dir: Optional[str] = None,
                 initial_num_updates: int = 0):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.online_net = deepcopy(model)
        self.online_net.to(self.device)
        self.online_net.train()
        self.target_net = deepcopy(self.online_net)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr, eps=eps)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.grad_norm = grad_norm
        self.batch_queue = batch_queue
        self.priority_queue = priority_queue
        self.stats_queue = stats_queue
        self.num_updates = initial_num_updates
        self.done = False

        self.target_net_update_interval = target_net_update_interval
        self.save_interval = save_interval

        self.batched_data = []

        self.shared_model = model

        self.game_name = game_name
        self.models_dir = models_dir
        self.use_wandb = use_wandb
        self.env_name = env_name
        self.video_dir = video_dir
        self.action_dim = model.action_dim

    def store_weights(self):
        """Store current weights to shared model for actors"""
        self.shared_model.load_state_dict(self.online_net.state_dict())

    def record_video_episode(self, video_folder: str, name_prefix: str = "eval"):
        """
        Record one evaluation episode with epsilon=0 (greedy policy)

        Args:
            video_folder: Directory to save video
            name_prefix: Prefix for video filename

        Returns:
            episode_reward: Total reward for the episode
        """
        if not self.env_name or not self.video_dir:
            return None

        try:
            from gymnasium.wrappers import RecordVideo
            from environment import create_env
            from model import AgentState
            import numpy as np

            # Create environment with render_mode for video recording
            env = create_env(env_name=self.env_name, noop_start=False, render_mode="rgb_array")
            env = RecordVideo(
                env,
                video_folder=video_folder,
                name_prefix=name_prefix,
                episode_trigger=lambda x: True  # Record this episode
            )

            # Reset environment
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result

            # Create agent state
            agent_state = AgentState(torch.from_numpy(obs).unsqueeze(0), self.action_dim)

            done = False
            episode_reward = 0.0
            steps = 0
            max_steps = config.max_episode_steps

            # Run episode with greedy policy (epsilon=0)
            while not done and steps < max_steps:
                with torch.no_grad():
                    # Move agent_state to device
                    agent_state.obs = agent_state.obs.to(self.device)
                    agent_state.last_action = agent_state.last_action.to(self.device)
                    agent_state.last_reward = agent_state.last_reward.to(self.device)
                    if agent_state.hidden_state is not None:
                        agent_state.hidden_state = (
                            agent_state.hidden_state[0].to(self.device),
                            agent_state.hidden_state[1].to(self.device)
                        )

                    # Get Q-values
                    q_values, hidden = self.online_net(agent_state)

                    # Greedy action selection (q_values shape: [1, action_dim])
                    action = q_values.squeeze(0).argmax().item()

                # Take action
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, _ = step_result

                # Update state
                agent_state.update(next_obs, action, reward, hidden)

                episode_reward += float(reward)
                steps += 1

            env.close()
            return episode_reward

        except Exception as e:
            print(f"Warning: Failed to record video: {e}")
            return None

    def prepare_data(self):
        """Background thread to prepare batched data"""
        while True:
            if not self.batch_queue.empty() and len(self.batched_data) < 4:
                data = self.batch_queue.get_nowait()
                self.batched_data.append(data)
            else:
                time.sleep(0.1)

    def run(self):
        """Main learner loop: train network on sampled batches"""
        background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        background_thread.start()
        time.sleep(2)

        start_time = time.time()
        while self.num_updates < config.training_steps:
            if self.num_updates % 1000 == 0:
                print(f"{self.num_updates} / {config.training_steps} (learner: num_updates/total training steps)")

            while not self.batched_data:
                time.sleep(1)
            data = self.batched_data.pop(0)

            (batch_obs, batch_last_action, batch_last_reward, batch_hidden,
             batch_action, batch_n_step_reward, batch_n_step_gamma,
             burn_in_steps, learning_steps, forward_steps,
             idxes, is_weights, old_ptr, env_steps) = data

            batch_obs = batch_obs.to(self.device)
            batch_last_action = batch_last_action.to(self.device)
            batch_last_reward = batch_last_reward.to(self.device)
            batch_hidden = batch_hidden.to(self.device)
            batch_action = batch_action.to(self.device)
            batch_n_step_reward = batch_n_step_reward.to(self.device)
            batch_n_step_gamma = batch_n_step_gamma.to(self.device)
            is_weights = is_weights.to(self.device)

            batch_obs = batch_obs.float()
            batch_last_action = batch_last_action.float()
            batch_action = batch_action.long()

            batch_hidden = (batch_hidden[:1], batch_hidden[1:])

            batch_obs = batch_obs / 255

            # Double Q-learning: use online net to select actions, target net to evaluate
            with torch.no_grad():
                batch_action_ = self.online_net.calculate_q_(
                    batch_obs, batch_last_action, batch_last_reward, batch_hidden,
                    burn_in_steps, learning_steps, forward_steps
                ).argmax(1).unsqueeze(1)

                batch_q_ = self.target_net.calculate_q_(
                    batch_obs, batch_last_action, batch_last_reward, batch_hidden,
                    burn_in_steps, learning_steps, forward_steps
                ).gather(1, batch_action_).squeeze(1)

            # Value rescaling for stability
            target_q = self.value_rescale(
                batch_n_step_reward + batch_n_step_gamma * self.inverse_value_rescale(batch_q_)
            )

            batch_q = self.online_net.calculate_q(
                batch_obs, batch_last_action, batch_last_reward, batch_hidden,
                burn_in_steps, learning_steps
            ).gather(1, batch_action).squeeze(1)

            loss = (is_weights * self.loss_fn(batch_q, target_q)).mean()

            td_errors = (target_q - batch_q).detach().clone().squeeze().abs().cpu().float().numpy()

            priorities = calculate_mixed_td_errors(td_errors, learning_steps.numpy())

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()

            # Compute grad norm before clipping for logging
            total_norm = 0.0
            for p in self.online_net.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_norm)
            self.optimizer.step()

            self.num_updates += 1

            # print(f"[DEBUG learner] Putting loss {loss.item()} into priority_queue")
            self.priority_queue.put((idxes, priorities, old_ptr, loss.item()))

            # Collect metrics for wandb logging
            if self.use_wandb:
                metrics = {
                    'train/loss': loss.item(),
                    'train/q_value_mean': batch_q.mean().item(),
                    'train/q_value_max': batch_q.max().item(),
                    'train/q_value_std': batch_q.std().item(),
                    'train/target_q_mean': target_q.mean().item(),
                    'train/td_error_mean': td_errors.mean(),
                    'train/td_error_max': td_errors.max(),
                    'train/grad_norm': total_norm,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                }

                # Read stats from replay buffer (non-blocking)
                while not self.stats_queue.empty():
                    try:
                        buffer_stats = self.stats_queue.get_nowait()
                        metrics.update(buffer_stats)
                    except:
                        break

                wandb.log(metrics, step=self.num_updates)

            # Store new weights in shared memory
            if self.num_updates % 4 == 0:
                self.store_weights()

            # Update target network
            if self.num_updates % self.target_net_update_interval == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            # Save model and record video
            if self.num_updates % self.save_interval == 0:
                save_path = os.path.join(self.models_dir, f'{self.num_updates}.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                checkpoint = {
                    'model_state_dict': self.online_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'num_updates': self.num_updates,
                    'env_steps': env_steps,
                    'training_time_minutes': (time.time() - start_time) / 60,
                    'target_net_state_dict': self.target_net.state_dict()
                }
                torch.save(checkpoint, save_path)
                print(f"Model saved to {save_path}")

                # Record evaluation video
                if self.video_dir:
                    print(f"Recording evaluation video (step {self.num_updates})...")
                    os.makedirs(self.video_dir, exist_ok=True)

                    eval_reward = self.record_video_episode(
                        video_folder=self.video_dir,
                        name_prefix=f"eval_step_{self.num_updates}"
                    )

                    if eval_reward is not None:
                        print(f"Evaluation episode reward: {eval_reward:.2f}")

                        if self.use_wandb:
                            try:
                                # Find the recorded video file with the specific prefix
                                prefix = f"eval_step_{self.num_updates}"
                                video_files = glob.glob(os.path.join(self.video_dir, f"{prefix}*.mp4"))
                                if video_files:
                                    video_path = video_files[0]
                                    wandb.log({
                                        'eval/episode_reward': eval_reward,
                                        'eval/video': wandb.Video(video_path, fps=30, format="mp4")
                                    }, step=self.num_updates)
                                    print(f"Video logged to wandb: {video_path}")
                            except Exception as e:
                                print(f"Warning: Failed to log video to wandb: {e}")

    @staticmethod
    def value_rescale(value, eps=1e-3):
        """Value rescaling transformation for stability"""
        return value.sign() * ((value.abs() + 1).sqrt() - 1) + eps * value

    @staticmethod
    def inverse_value_rescale(value, eps=1e-3):
        """Inverse value rescaling transformation"""
        temp = ((1 + 4 * eps * (value.abs() + 1 + eps)).sqrt() - 1) / (2 * eps)
        return value.sign() * (temp.square() - 1)
