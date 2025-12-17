import os
import threading
from collections import deque
import time

import cv2
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import RolloutBuffer

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from framework.Logger import logger

EXPECTED_OBS_DIMS = (210, 160, 3)


class ManualStepEnv(gym.Env):
    """Minimal gym-like environment wrapper for manual stepping by harness"""

    def __init__(self, num_actions, obs_shape=(84, 84, 4)):
        super().__init__()
        self.num_actions = num_actions
        self.obs_shape = obs_shape

        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(num_actions)

        self.current_obs = None
        self.current_reward = 0
        self.current_done = False
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def reset(self, seed=None, options=None):
        """Reset called by SB3 - we'll return the last observation"""
        if seed is not None:
            np.random.seed(seed)
        if self.current_obs is None:
            self.current_obs = np.zeros(self.obs_shape, dtype=np.uint8)
        return self.current_obs, {}

    def step(self, action):
        """Step called by SB3 - returns buffered state from harness"""
        obs = self.current_obs if self.current_obs is not None else np.zeros(self.obs_shape, dtype=np.uint8)
        reward = self.current_reward
        terminated = self.current_done
        truncated = False
        info = {}

        self.current_episode_reward += reward
        self.current_episode_length += 1

        if terminated:
            info['episode'] = {'r': self.current_episode_reward, 'l': self.current_episode_length}
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0

        self.current_reward = 0
        self.current_done = False

        return obs, reward, terminated, truncated, info

    def update_state(self, obs, reward, done):
        """Called by agent to buffer the next state from harness"""
        self.current_obs = obs
        self.current_reward = reward
        self.current_done = done


class Agent:
    """PPO agent for physical Atari using Stable Baselines3"""

    def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
        logger.info(f"{'-'*8} INITIALIZING NEW PPO AGENT {'-'*8}")
        self.num_actions = num_actions
        self.total_frames = total_frames
        self.frame_skip = 4
        self.seed = seed
        self.data_dir = data_dir
        self.gpu = 0

        self.learning_rate = 2.5e-4
        self.n_steps = 128
        self.batch_size = 32
        self.n_epochs = 4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.1
        self.ent_coef_initial = 0.01
        self.ent_coef_final = 0.001
        self.ent_coef = self.ent_coef_initial
        self.ent_coef_decay_steps = 100_000
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = None

        self.load_file = None
        self.eval_mode = False
        self.use_grayscale = True
        self.n_stack = 4
        self.obs_height = 84
        self.obs_width = 84
        self.reduce_action_set = 2
        self.use_wandb = True

        for key, value in kwargs.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                logger.info("agent_ppo: Overwriting %s from %r to %r", key, old_value, value)
            else:
                logger.warning(f"agent_ppo: Unknown parameter {key}")

        if self.use_wandb and WANDB_AVAILABLE:
            if wandb.run is not None:
                logger.info(f"agent_ppo: Using existing wandb run from harness (run name: {wandb.run.name})")
                logger.info(f"agent_ppo: Will log train/ metrics every {self.n_steps} steps")
            else:
                logger.warning("agent_ppo: Wandb requested but no run initialized. Please initialize in harness.")
                self.use_wandb = False
        elif self.use_wandb and not WANDB_AVAILABLE:
            logger.warning("agent_ppo: Wandb requested but not installed. Run: pip install wandb")
            self.use_wandb = False

        self.policy_num_actions = num_actions

        self.frame_buffer = deque(maxlen=self.n_stack)
        self.step_count = 0
        self.last_action = 0

        self.last_reward = 0
        self.last_episode_start = True
        self.prev_done = True

        self.accumulated_reward = 0

        self.actor_inference_count = 0
        self.actor_inference_start = time.time()
        self.actor_inference_log_interval = 500

        height, width = self.obs_height, self.obs_width
        obs_shape = (self.n_stack, height, width)

        logger.info(f"agent_ppo: Observation shape = {obs_shape}")
        logger.info(f"agent_ppo: Policy num actions = {self.policy_num_actions}")

        self.env = ManualStepEnv(self.policy_num_actions, obs_shape)
        self.vec_env = DummyVecEnv([lambda: self.env])

        device = f"cuda:{self.gpu}" if torch.cuda.is_available() and self.gpu >= 0 else "cpu"
        logger.info(f"agent_ppo: Using device = {device}")

        # Separate actor/learner models keep action selection responsive during training.
        model_exists = self.load_file and (os.path.exists(self.load_file) or os.path.exists(self.load_file + '.zip'))
        if model_exists:
            logger.info(f"agent_ppo: Loading model from {self.load_file}")
            self.learner_model = PPO.load(self.load_file, env=self.vec_env, device=device)
            self.actor_model = PPO.load(self.load_file, env=self.vec_env, device=device)
        else:
            logger.info(f"agent_ppo: Creating new PPO model")
            self.learner_model = PPO(
                "CnnPolicy",
                self.vec_env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                target_kl=self.target_kl,
                verbose=1,
                device=device,
                seed=seed,
            )
            self.actor_model = PPO(
                "CnnPolicy",
                self.vec_env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                target_kl=self.target_kl,
                verbose=1,
                device=device,
                seed=seed,
            )

        self.actor_model.policy.eval()
        self.learner_model.policy.train()
        logger.info(f"agent_ppo: Dual model architecture initialized (actor + learner)")

        # Keep reference to ppo_model for backward compatibility
        self.ppo_model = self.learner_model

        self.checkpoint_callback = CheckpointCallback(
            save_freq=50000, save_path=os.path.join(data_dir, "checkpoints"), name_prefix="ppo_model"
        )

        self.training_step = 0
        self.frames_since_train = 0
        self.train_losses = []  # Required by harness_physical.py

        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=1,
        )

        self.last_obs = None
        self.last_value = None
        self.last_log_prob = None
        self.is_training = False

        self.weight_lock = threading.Lock()
        self.training_thread = None
        self.episode_start_idx = 0

        logger.info(f"agent_ppo: Initialized successfully")
        if self.eval_mode:
            logger.info(f"agent_ppo: Running in EVALUATION MODE (training disabled)")
        logger.info(f"agent_ppo: Rollout buffer size = {self.n_steps}")
        logger.info(f"agent_ppo: Entropy coefficient schedule: {self.ent_coef_initial} -> {self.ent_coef_final}")

    def preprocess_frame(self, observation_rgb8):
        """Preprocess single frame: resize and convert to grayscale"""
        assert (
            observation_rgb8.shape == EXPECTED_OBS_DIMS
        ), f"Observation Shape is: {observation_rgb8.shape}, but we expected: {EXPECTED_OBS_DIMS}"
        frame = cv2.cvtColor(observation_rgb8, cv2.COLOR_BGR2GRAY)  # (H, W)
        frame = cv2.resize(frame, (self.obs_width, self.obs_height), interpolation=cv2.INTER_AREA)
        return frame

    def update_entropy_coefficient(self):
        """Update entropy coefficient based on training progress (linear decay)"""
        if self.ent_coef_decay_steps is None:
            max_steps = self.total_frames // self.frame_skip
        else:
            max_steps = self.ent_coef_decay_steps

        progress = min(1.0, self.step_count / max_steps)
        self.ent_coef = self.ent_coef_initial + progress * (self.ent_coef_final - self.ent_coef_initial)

        return self.ent_coef

    def frame(self, observation_rgb8, reward, end_of_episode):
        """
        Called every frame by harness.

        Args:
            observation_rgb8: RGB observation (210, 160, 3) uint8
            reward: Scalar reward from environment
            end_of_episode: 0=ongoing, 1=life lost, 2=game over, 3=timeout

        Returns:
            action_index: Integer action index to execute
        """
        self.step_count += 1
        reward = np.clip(reward, -1, 1)

        processed_frame = self.preprocess_frame(observation_rgb8)

        if end_of_episode > 0:
            if self.use_wandb:
                wandb.log(
                    {
                        "episode/end_reason": end_of_episode,
                        "episode/total_frames": self.step_count,
                    },
                    step=self.step_count,
                )

            self.frame_buffer.clear()
            self.last_obs = None
            self.last_reward = 0
            self.accumulated_reward = 0

        self.frame_buffer.append(processed_frame)

        self.accumulated_reward += reward

        if self.step_count % self.frame_skip != 0:
            return self.last_action

        if len(self.frame_buffer) < self.n_stack:
            return self.last_action

        stacked_frames = np.stack(list(self.frame_buffer), axis=0)
        # Treat loss of life as terminal.
        done = end_of_episode >= 1

        if (
            not self.eval_mode
            and self.last_obs is not None
            and not self.is_training
            and self.last_value is not None
            and self.last_log_prob is not None
        ):
            self.rollout_buffer.add(
                obs=self.last_obs,
                action=np.array([self.last_policy_action]),
                reward=np.array([self.accumulated_reward]),
                episode_start=np.array([self.last_episode_start]),
                value=self.last_value,
                log_prob=self.last_log_prob,
            )
            self.frames_since_train += 1

            if self.use_wandb and self.frames_since_train % 10 == 0:
                wandb.log(
                    {
                        "collection/frames_collected": self.step_count,
                        "collection/buffer_fill": self.frames_since_train / self.n_steps,
                    },
                    step=self.step_count,
                )

        # Action selection uses actor_model; learner_model trains in the background.
        with torch.no_grad():
            obs_tensor = torch.as_tensor(stacked_frames).unsqueeze(0).to(self.actor_model.device)

            with self.weight_lock:
                actions, values, log_probs = self.actor_model.policy.forward(obs_tensor)

            self.actor_inference_count += 1
            elapsed = time.time() - self.actor_inference_start
            if elapsed > 0 and self.actor_inference_count >= self.actor_inference_log_interval:
                actor_fps = self.actor_inference_count / elapsed
                if self.use_wandb:
                    wandb.log({"performance/actor_inference_fps": actor_fps}, step=self.step_count)
                else:
                    logger.debug(f"agent_ppo: actor inference rate {actor_fps:.1f} fps")
                self.actor_inference_count = 0
                self.actor_inference_start = time.time()

            policy_action = int(actions.cpu().numpy()[0])

        env_action = policy_action

        self.last_obs = stacked_frames
        self.last_policy_action = policy_action
        self.last_value = values
        self.last_log_prob = log_probs
        self.last_action = env_action

        self.last_episode_start = self.prev_done

        self.accumulated_reward = 0
        self.prev_done = done

        if not self.eval_mode and self.frames_since_train >= self.n_steps and not self.is_training:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(stacked_frames).unsqueeze(0).to(self.learner_model.device)
                last_values = self.learner_model.policy.predict_values(obs_tensor)

            if not self.rollout_buffer.full:
                raise RuntimeError("agent_ppo: rollout_buffer expected full before training but is not full")

            self.rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=np.array([done]))

            self.is_training = True
            self.frames_since_train = 0

            self.training_thread = threading.Thread(target=self._train_async, daemon=True)
            self.training_thread.start()

        return env_action

    def _train_async(self):
        """
        Train the learner model in a background thread.
        This allows the actor to continue selecting actions without blocking.
        """
        logger.info("agent_ppo: Starting training step %s", self.training_step)
        train_start = time.time()
        self.training_step += 1

        current_ent_coef = self.update_entropy_coefficient()

        logger.info(f"agent_ppo: Training step {self.training_step} started (ent_coef={current_ent_coef:.6f})")

        epoch_losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []
        raw_entropies = []
        grad_norms = []

        policy_loss_raw = []
        entropy_loss_raw = []
        value_loss_raw = []

        action_counts = np.zeros(self.policy_num_actions)

        all_advantages = []
        all_raw_advantages = []

        final_loss = 0.0
        for _ in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(actions, torch.Tensor):
                    actions = actions.long().flatten()

                for action_idx in actions.cpu().numpy():
                    action_counts[action_idx] += 1

                values, log_prob, entropy = self.learner_model.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                advantages = rollout_data.advantages

                all_raw_advantages.extend(advantages.detach().cpu().numpy().tolist())

                if self.learner_model.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                all_advantages.extend(advantages.detach().cpu().numpy().tolist())

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                values_pred = values
                if self.learner_model.clip_range_vf is not None:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -self.learner_model.clip_range_vf,
                        self.learner_model.clip_range_vf,
                    )
                value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values_pred)

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                    raw_entropy = torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                    raw_entropy = torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                self.learner_model.policy.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.learner_model.policy.parameters(), self.max_grad_norm)
                self.learner_model.policy.optimizer.step()

                final_loss = loss.detach().cpu().item()
                epoch_losses.append(final_loss)

                policy_losses.append(policy_loss.detach().cpu().item())
                value_losses.append(value_loss.detach().cpu().item())
                entropy_losses.append(entropy_loss.detach().cpu().item())
                raw_entropies.append(raw_entropy.detach().cpu().item())

                policy_loss_raw.append(policy_loss.detach().cpu().item())
                entropy_loss_raw.append(entropy_loss.detach().cpu().item())
                value_loss_raw.append(value_loss.detach().cpu().item())

                grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

        # Sync learner weights to actor under lock.
        with self.weight_lock:
            self.actor_model.policy.load_state_dict(self.learner_model.policy.state_dict())
            self.actor_model.policy.eval()

        train_duration = time.time() - train_start

        if self.use_wandb:
            mean_policy_raw = np.mean(policy_loss_raw)
            mean_entropy_raw = abs(np.mean(entropy_loss_raw))
            loss_ratio = mean_policy_raw / mean_entropy_raw if mean_entropy_raw > 0 else 0

            total_actions = action_counts.sum()
            action_percentages = (action_counts / total_actions * 100) if total_actions > 0 else action_counts

            returns_array = self.rollout_buffer.returns.flatten()
            rewards_array = self.rollout_buffer.rewards.flatten()
            values_array = self.rollout_buffer.values.flatten()

            advantages_array = np.array(all_advantages)
            raw_advantages_array = np.array(all_raw_advantages)

            metrics = {
                "train/loss": np.mean(epoch_losses),
                "train/policy_loss": np.mean(policy_losses),
                "train/value_loss": np.mean(value_losses),
                "train/entropy_loss": np.mean(entropy_losses),
                "train/entropy": np.mean(raw_entropies),
                "train/grad_norm": np.mean(grad_norms),
                "train/learning_rate": self.learning_rate,
                "train/ent_coef": self.ent_coef,
                "train/training_time": train_duration,
                "train/training_step": self.training_step,
                "debug/policy_loss_raw": mean_policy_raw,
                "debug/entropy_loss_raw": mean_entropy_raw,
                "debug/value_loss_raw": np.mean(value_loss_raw),
                "debug/policy_to_entropy_ratio": loss_ratio,
                "debug/returns_mean": returns_array.mean(),
                "debug/returns_std": returns_array.std(),
                "debug/returns_min": returns_array.min(),
                "debug/returns_max": returns_array.max(),
                "debug/rewards_mean": rewards_array.mean(),
                "debug/rewards_std": rewards_array.std(),
                "debug/values_mean": values_array.mean(),
                "debug/values_std": values_array.std(),
                "debug/advantages_mean": advantages_array.mean(),
                "debug/advantages_std": advantages_array.std(),
                "debug/advantages_min": advantages_array.min(),
                "debug/advantages_max": advantages_array.max(),
                "debug/raw_advantages_mean": raw_advantages_array.mean(),
                "debug/raw_advantages_std": raw_advantages_array.std(),
                "debug/raw_advantages_min": raw_advantages_array.min(),
                "debug/raw_advantages_max": raw_advantages_array.max(),
                "time/training_iterations": self.training_step,
                "time/total_timesteps": self.step_count,
            }

            for action_idx in range(self.policy_num_actions):
                metrics[f"debug/action_{action_idx}_pct"] = action_percentages[action_idx]

            wandb.log(metrics, step=self.step_count)
            logger.info(f"agent_ppo: Logged train/ metrics to wandb at step {self.step_count}")
            logger.info(f"agent_ppo: Action distribution: {action_percentages}")

        logger.info(f"agent_ppo: Training step {self.training_step} completed in {train_duration:.2f}s")

        self.train_losses.append(float(final_loss))

        self.rollout_buffer.reset()

        self.is_training = False

    def save_model(self, filename):
        """Save PPO model to disk (saves learner model with latest weights)"""
        self.learner_model.save(filename)
        logger.info(f"agent_ppo: Learner model saved to {filename}")
