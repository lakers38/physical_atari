# agent_sac.py
#
# Soft Actor-Critic agent wrapper for physical Atari
# Wraps algorithms/sac/agent_actor_critic.py to work with harness_physical.py

import os
import cv2
import numpy as np
import torch
from collections import deque
from typing import Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from framework.Logger import logger
from algorithms.sac.sac import SACAgent, ReplayBuffer

EXPECTED_OBS_DIMS = (210, 160, 3)


class Agent:
    """SAC agent wrapper for physical Atari harness"""

    def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
        logger.info(f"{'-'*8} INITIALIZING NEW SAC AGENT {'-'*8}")

        # Configuration
        self.num_actions = num_actions
        self.total_frames = total_frames
        self.seed = seed
        self.data_dir = data_dir
        self.gpu = kwargs.get('gpu', 0)

        # SAC hyperparameters (can be overridden via kwargs)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.gamma = kwargs.get('gamma', 0.99)
        self.entropy_coef = kwargs.get('entropy_coef', 0.01)
        self.feature_dim = kwargs.get('feature_dim', 512)
        self.actor_hidden_dim = kwargs.get('actor_hidden_dim', 256)
        self.value_hidden_dim = kwargs.get('value_hidden_dim', 256)

        # Frame processing settings
        self.frame_skip = kwargs.get('frame_skip', 4)
        self.n_stack = kwargs.get('n_stack', 4)
        self.obs_height = kwargs.get('obs_height', 128)
        self.obs_width = kwargs.get('obs_width', 128)
        # SAC implementation expects channel-last with depth == n_stack, so force grayscale stacking
        self.use_grayscale = kwargs.get('use_grayscale', True)

        # Training settings
        self.update_freq = kwargs.get('update_freq', 1)  # Gradient step cadence (every frame_skip * update_freq env frames)
        self.batch_size = kwargs.get('batch_size', 32)
        self.buffer_size = kwargs.get('buffer_size', 100_000)
        self.learning_starts = kwargs.get('learning_starts', 10_000)
        self.gradient_steps = kwargs.get('gradient_steps', 1)
        self.train_freq = kwargs.get('train_freq', 1)
        self.eval_mode = kwargs.get('eval_mode', False)  # Evaluation mode (no training)

        # Model loading
        self.load_file = kwargs.get('load_file', None)

        # Wandb logging
        self.use_wandb = kwargs.get('use_wandb', False)

        # Validate wandb setup
        if self.use_wandb and WANDB_AVAILABLE:
            if wandb.run is not None:
                logger.info(f"agent_sac: Using existing wandb run from harness (run name: {wandb.run.name})")
            else:
                logger.warning("agent_sac: Wandb requested but no run initialized. Please initialize in harness.")
                self.use_wandb = False
        elif self.use_wandb and not WANDB_AVAILABLE:
            logger.warning("agent_sac: Wandb requested but not installed. Run: pip install wandb")
            self.use_wandb = False

        logger.info(f"agent_sac: Configuration:")
        logger.info(f"  learning_rate={self.learning_rate}")
        logger.info(f"  gamma={self.gamma}")
        logger.info(f"  entropy_coef={self.entropy_coef}")
        logger.info(f"  frame_skip={self.frame_skip}")
        logger.info(f"  n_stack={self.n_stack}")
        logger.info(f"  obs_size={self.obs_height}x{self.obs_width}")
        logger.info(f"  update_freq={self.update_freq}")
        logger.info(f"  eval_mode={self.eval_mode}")

        # Device
        if torch.cuda.is_available() and self.gpu >= 0:
            device = f"cuda:{self.gpu}"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"agent_sac: Using device = {device}")

        # Initialize SAC agent
        self.sac_agent = SACAgent(
            num_actions=num_actions,
            feature_dim=self.feature_dim,
            actor_hidden_dim=self.actor_hidden_dim,
            value_hidden_dim=self.value_hidden_dim,
            n_stack=self.n_stack,
            input_size=self.obs_height,
            device=device,
            gamma=self.gamma,
            learning_rate=self.learning_rate,
            entropy_coef=self.entropy_coef,
            fail_on_nonfinite=False,  # Don't crash on NaN during physical runs
        )

        # Load model if specified
        if self.load_file and os.path.exists(self.load_file):
            logger.info(f"agent_sac: Loading model from {self.load_file}")
            self.sac_agent.load(self.load_file)

        # Set to eval mode if requested
        if self.eval_mode:
            self.sac_agent.cnn.eval()
            self.sac_agent.actor.eval()
            logger.info(f"agent_sac: Running in EVALUATION MODE (training disabled)")
        else:
            self.sac_agent.cnn.train()
            self.sac_agent.actor.train()

        # Frame buffering (stack n_stack frames)
        self.frame_buffer = deque(maxlen=self.n_stack)
        self.step_count = 0
        self.last_action = 0

        # Training tracking
        self.train_losses = []  # Required by harness_physical.py
        self.training_step = 0
        self.frames_since_update = 0

        # Transition buffering for SAC updates
        self.last_obs = None
        self.last_action = 0
        self.last_done = False
        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_size,
            obs_shape=(self.obs_height, self.obs_width, self.n_stack),
        )

        # Accumulate rewards over frame_skip frames (for action repeat)
        self.accumulated_reward = 0.0

        # Episode tracking
        self.episode_reward = 0
        self.episode_length = 0

        logger.info(f"agent_sac: Initialized successfully")
        logger.info(f"agent_sac: Observation shape = ({self.n_stack}, {self.obs_height}, {self.obs_width})")
        logger.info(f"agent_sac: Policy num actions = {self.num_actions}")

    def preprocess_frame(self, observation_rgb8):
        """Preprocess single frame: resize and optionally convert to grayscale"""
        assert observation_rgb8.shape == EXPECTED_OBS_DIMS, \
            f"Observation Shape is: {observation_rgb8.shape}, but we expected: {EXPECTED_OBS_DIMS}"

        if self.use_grayscale:
            # Convert to grayscale
            frame = cv2.cvtColor(observation_rgb8, cv2.COLOR_BGR2GRAY)  # (H, W)
            frame = cv2.resize(frame, (self.obs_width, self.obs_height), interpolation=cv2.INTER_AREA)
        else:
            # Keep RGB
            frame = cv2.cvtColor(observation_rgb8, cv2.COLOR_BGR2RGB)  # (H, W, 3)
            frame = cv2.resize(frame, (self.obs_width, self.obs_height), interpolation=cv2.INTER_AREA)

        return frame

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

        # Preprocess and buffer frame
        processed_frame = self.preprocess_frame(observation_rgb8)

        # Handle episode end
        if end_of_episode > 0:
            if self.use_wandb:
                wandb.log({
                    "episode/end_reason": end_of_episode,
                    "episode/total_frames": self.step_count,
                    "episode/reward": self.episode_reward,
                    "episode/length": self.episode_length,
                }, step=self.step_count)

            self.frame_buffer.clear()
            self.last_obs = None
            self.last_action = 0
            self.accumulated_reward = 0
            self.episode_reward = 0
            self.episode_length = 0

        self.frame_buffer.append(processed_frame)

        # Accumulate rewards over frame_skip frames
        self.accumulated_reward += reward
        self.episode_reward += reward
        self.episode_length += 1

        # Only act every frame_skip frames
        if self.step_count % self.frame_skip != 0:
            return self.last_action

        # Wait until we have n_stack frames stacked
        if len(self.frame_buffer) < self.n_stack:
            return self.last_action

        # Stack frames based on format
        if self.use_grayscale:
            # Grayscale: [(H, W), ...] -> (H, W, n_stack)
            stacked_frames = np.stack(list(self.frame_buffer), axis=-1)  # (H, W, n_stack)
        else:
            # RGB stacking would mismatch SAC input (expects depth = n_stack), so keep grayscale by default
            stacked_frames = np.stack(list(self.frame_buffer), axis=-1)

        # Reshape to (1, H, W, n_stack) or (1, H, W, n_stack*3) for SAC
        obs_batch = stacked_frames[np.newaxis, ...]  # (1, H, W, C)

        # Convert end_of_episode to done flag
        done = end_of_episode >= 1

        # Inform agent when a new episode starts
        if self.last_obs is None:
            self.sac_agent.start_episodes(obs_batch)

        # If we have a previous transition, push to replay and maybe update (unless in eval mode)
        if self.last_obs is not None:
            self.replay_buffer.add(
                self.last_obs[0],
                self.last_action,
                self.accumulated_reward,
                obs_batch[0],
                done,
            )

            if not self.eval_mode:
                try:
                    should_train = (
                        self.replay_buffer.size >= self.batch_size
                        and self.step_count >= self.learning_starts
                        and (self.step_count // self.frame_skip) % self.train_freq == 0
                    )
                    if should_train:
                        for _ in range(self.gradient_steps):
                            batch = self.replay_buffer.sample(self.batch_size)
                            metrics = self.sac_agent.update(*batch)
                            self.frames_since_update += 1
                            self.training_step += 1
                            self.train_losses.append(metrics['total_loss'])

                            if self.use_wandb and self.training_step % 10 == 0:
                                wandb.log({
                                    "train/total_loss": metrics['total_loss'],
                                    "train/actor_loss": metrics['actor_loss'],
                                    "train/value_loss": metrics['value_loss'],
                                    "train/advantage": metrics['advantage'],
                                    "train/value_pred": metrics['value_pred'],
                                    "train/value_next": metrics['value_next'],
                                    "train/value_target": metrics['value_target'],
                                    "train/training_step": self.training_step,
                                }, step=self.step_count)
                except Exception as e:
                    logger.error(f"agent_sac: Training error at step {self.step_count}: {e}")
                    # Continue with action selection even if training fails

        # Get action from SAC policy
        with torch.no_grad():
            actions, log_probs, entropy, features = self.sac_agent.select_actions(obs_batch)
            action = int(actions[0])

        # Store for next update
        self.last_obs = obs_batch
        self.last_action = action
        if done:
            # Inform agent of episode boundary for compatibility (no-op currently)
            self.sac_agent.reset_done(done, obs_batch)

        # Reset accumulator after using it
        self.accumulated_reward = 0

        return action

    def save_model(self, filename):
        """Save SAC model to disk"""
        # Ensure .pt extension
        if not filename.endswith('.pt'):
            filename = filename + '.pt'

        self.sac_agent.save(filename)
        logger.info(f"agent_sac: Model saved to {filename}")
