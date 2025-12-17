"""Actor for R2D2: collects experience by interacting with environment"""

import math
import random
from typing import Any, Optional, Tuple

import numpy as np
import torch

from . import config
from .model import AgentState, Network
from .replay_buffer import Block


def calculate_mixed_td_errors(td_error, learning_steps):
    """
    Calculate mixed TD errors (90% max + 10% mean) for prioritization

    Args:
        td_error: Array of TD errors
        learning_steps: Length of each sequence

    Returns:
        Mixed TD errors per sequence
    """
    start_idx = 0
    mixed_td_errors = np.empty(learning_steps.shape, dtype=td_error.dtype)
    for i, steps in enumerate(learning_steps):
        mixed_td_errors[i] = (
            0.9 * td_error[start_idx : start_idx + steps].max() + 0.1 * td_error[start_idx : start_idx + steps].mean()
        )
        start_idx += steps

    return mixed_td_errors


class LocalBuffer:
    """Store transitions of one episode locally before sending to replay buffer"""

    def __init__(
        self,
        action_dim: int,
        forward_steps: int = config.forward_steps,
        burn_in_steps=config.burn_in_steps,
        learning_steps: int = config.learning_steps,
        gamma: float = config.gamma,
        hidden_dim: int = config.hidden_dim,
        block_length: int = config.block_length,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.forward_steps = forward_steps
        self.learning_steps = learning_steps
        self.burn_in_steps = burn_in_steps
        self.block_length = block_length
        self.curr_burn_in_steps = 0

    def __len__(self):
        return self.size

    def reset(self, init_obs: np.ndarray):
        """Reset buffer for new episode"""
        self.obs_buffer = [init_obs]
        self.last_action_buffer = [np.array([1 if i == 0 else 0 for i in range(self.action_dim)], dtype=bool)]
        self.last_reward_buffer = [0]
        self.hidden_buffer = [np.zeros((2, self.hidden_dim), dtype=np.float32)]
        self.action_buffer = []
        self.reward_buffer = []
        self.qval_buffer = []
        self.curr_burn_in_steps = 0
        self.size = 0
        self.sum_reward = 0
        self.done = False

    def add(self, action: int, reward: float, next_obs: np.ndarray, q_value: np.ndarray, hidden_state: np.ndarray):
        """Add one transition to buffer"""
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.hidden_buffer.append(hidden_state)
        self.obs_buffer.append(next_obs)
        self.last_action_buffer.append(np.array([1 if i == action else 0 for i in range(self.action_dim)], dtype=bool))
        self.last_reward_buffer.append(reward)
        self.qval_buffer.append(q_value)
        self.sum_reward += reward
        self.size += 1

    def finish(self, last_qval: Optional[np.ndarray] = None) -> Tuple:
        """
        Finish current block and prepare for replay buffer

        Args:
            last_qval: Q-values for bootstrapping (None if episode ended)

        Returns:
            [block, priorities, episode_reward]
        """
        assert self.size <= self.block_length

        num_sequences = math.ceil(self.size / self.learning_steps)

        max_forward_steps = min(self.size, self.forward_steps)
        n_step_gamma = [self.gamma**self.forward_steps] * (self.size - max_forward_steps)

        # last_qval is None means episode done
        if last_qval is not None:
            self.qval_buffer.append(last_qval)
            n_step_gamma.extend([self.gamma**i for i in reversed(range(1, max_forward_steps + 1))])
        else:
            self.done = True
            self.qval_buffer.append(np.zeros_like(self.qval_buffer[0]))
            n_step_gamma.extend([0 for _ in range(max_forward_steps)])  # Set gamma to 0 (no bootstrap)

        n_step_gamma = np.array(n_step_gamma, dtype=np.float32)

        obs = np.stack(self.obs_buffer)
        last_action = np.stack(self.last_action_buffer)
        last_reward = np.array(self.last_reward_buffer, dtype=np.float32)

        hiddens = np.stack(self.hidden_buffer[slice(0, self.size, self.learning_steps)])

        actions = np.array(self.action_buffer, dtype=np.uint8)

        # Stack q_values: list of [action_dim] -> [size, action_dim]
        qval_buffer = np.stack(self.qval_buffer)
        reward_buffer = self.reward_buffer + [0 for _ in range(self.forward_steps - 1)]
        n_step_reward = np.convolve(
            reward_buffer, [self.gamma ** (self.forward_steps - 1 - i) for i in range(self.forward_steps)], 'valid'
        ).astype(np.float32)

        burn_in_steps = np.array(
            [min(i * self.learning_steps + self.curr_burn_in_steps, self.burn_in_steps) for i in range(num_sequences)],
            dtype=np.uint8,
        )
        learning_steps = np.array(
            [min(self.learning_steps, self.size - i * self.learning_steps) for i in range(num_sequences)],
            dtype=np.uint8,
        )
        forward_steps = np.array(
            [min(self.forward_steps, self.size + 1 - np.sum(learning_steps[: i + 1])) for i in range(num_sequences)],
            dtype=np.uint8,
        )

        assert forward_steps[-1] == 1 and burn_in_steps[0] == self.curr_burn_in_steps

        max_qval = np.max(qval_buffer[max_forward_steps : self.size + 1], axis=1)
        max_qval = np.pad(max_qval, (0, max_forward_steps - 1), 'edge')
        target_qval = qval_buffer[np.arange(self.size), actions]

        td_errors = np.abs(n_step_reward + n_step_gamma * max_qval - target_qval, dtype=np.float32)
        # Use ceil to handle partial sequences: ceil(120/80) = 2, not floor(120/80) = 1
        max_num_sequences = math.ceil(self.block_length / self.learning_steps)
        priorities = np.zeros(max_num_sequences, dtype=np.float32)
        priorities[:num_sequences] = calculate_mixed_td_errors(td_errors, learning_steps)

        # Save burn-in information for next block
        self.obs_buffer = self.obs_buffer[-self.burn_in_steps - 1 :]
        self.last_action_buffer = self.last_action_buffer[-self.burn_in_steps - 1 :]
        self.last_reward_buffer = self.last_reward_buffer[-self.burn_in_steps - 1 :]
        self.hidden_buffer = self.hidden_buffer[-self.burn_in_steps - 1 :]
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.qval_buffer.clear()
        self.curr_burn_in_steps = len(self.obs_buffer) - 1
        self.size = 0

        block = Block(
            obs,
            last_action,
            last_reward,
            actions,
            n_step_reward,
            n_step_gamma,
            hiddens,
            num_sequences,
            burn_in_steps,
            learning_steps,
            forward_steps,
        )
        return [block, priorities, self.sum_reward if self.done else None]


class Actor:
    """
    Actor process: interacts with environment and collects experience

    Uses epsilon-greedy policy with actor-specific epsilon value
    """

    def __init__(
        self,
        epsilon: float,
        model,
        sample_queue,
        env_fn,
        obs_shape: np.ndarray = config.obs_shape,
        max_episode_steps: int = config.max_episode_steps,
        block_length: int = config.block_length,
    ):
        self.env = env_fn()
        self.action_dim = self.env.action_space.n
        self.model = Network(self.action_dim)
        self.model.eval()
        self.local_buffer = LocalBuffer(self.action_dim)

        self.epsilon = epsilon
        self.shared_model = model
        self.sample_queue = sample_queue
        self.max_episode_steps = max_episode_steps
        self.block_length = block_length

    def run(self):
        """Main actor loop: collect experience and send to replay buffer"""
        actor_steps = 0

        while True:
            done = False
            agent_state = self.reset()
            episode_steps = 0

            while not done and episode_steps < self.max_episode_steps:
                with torch.no_grad():
                    q_value, hidden = self.model(agent_state)

                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    # q_value shape: [action_dim] = [9]
                    # Select action with highest Q-value
                    action = q_value.argmax().item()

                # Apply action in env
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    # Gymnasium API (obs, reward, terminated, truncated, info)
                    next_obs, reward, done, truncated, _ = step_result
                    done = done or truncated
                else:
                    # Old gym API (obs, reward, done, info)
                    next_obs, reward, done, _ = step_result

                agent_state.update(next_obs, action, reward, hidden)

                episode_steps += 1
                actor_steps += 1

                # hidden is tuple (h, c) with shapes [num_layers, batch_size, hidden_dim]
                # Concatenate and squeeze batch dimension: [2, 1, hidden_dim] -> [2, hidden_dim]
                hidden_np = torch.cat(hidden).squeeze(1).numpy()
                self.local_buffer.add(action, reward, next_obs, q_value.numpy(), hidden_np)

                if done:
                    block = self.local_buffer.finish()
                    self.sample_queue.put(block)

                elif len(self.local_buffer) == self.block_length or episode_steps == self.max_episode_steps:
                    with torch.no_grad():
                        q_value, hidden = self.model(agent_state)

                    block = self.local_buffer.finish(q_value.numpy())

                    if self.epsilon > 0.01:
                        block[2] = None
                    self.sample_queue.put(block)

                if actor_steps % 400 == 0:
                    self.update_weights()

    def update_weights(self):
        """Load the latest weights from shared model"""
        self.model.load_state_dict(self.shared_model.state_dict())

    def reset(self):
        """Reset environment and local buffer"""
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            # Gymnasium API returns (obs, info)
            obs, _ = reset_result
        else:
            # Old gym API returns just obs
            obs = reset_result
        self.local_buffer.reset(obs)

        # obs shape: [C, H, W] = [1, 84, 84]
        # unsqueeze(0) adds batch dimension: [B, C, H, W] = [1, 1, 84, 84]
        state = AgentState(torch.from_numpy(obs).unsqueeze(0), self.action_dim)

        return state
