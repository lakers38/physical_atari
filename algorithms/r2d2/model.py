"""Neural network model for R2D2"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from r2d2 import config


@dataclass
class AgentState:
    """State of the agent including observation, last action, last reward, and LSTM hidden state"""
    obs: torch.Tensor
    action_dim: int
    last_action: torch.Tensor = field(init=False)
    last_reward: torch.Tensor = torch.zeros((1, 1), dtype=torch.float32)
    hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def __post_init__(self):
        self.last_action = torch.zeros((1, self.action_dim), dtype=torch.float32)

    def update(self, obs, last_action, last_reward, hidden):
        """Update state with new observation and action"""
        if isinstance(obs, torch.Tensor):
            self.obs = obs.unsqueeze(0) if obs.dim() == 3 else obs
        else:
            self.obs = torch.from_numpy(obs).unsqueeze(0)

        self.last_action = torch.FloatTensor([[1 if i == last_action else 0 for i in range(self.action_dim)]])
        self.last_reward = torch.FloatTensor([[last_reward]])
        self.hidden_state = hidden


class Network(nn.Module):
    """R2D2 network with CNN feature extractor + LSTM + Dueling DQN head"""

    def __init__(self, action_dim, obs_shape=config.obs_shape, hidden_dim=config.hidden_dim):
        super().__init__()

        # 84 x 84 grayscale input
        self.action_dim = action_dim
        self.obs_shape = obs_shape
        self.hidden_dim = hidden_dim

        self.max_forward_steps = config.forward_steps

        # CNN feature extractor (Nature DQN architecture)
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(True),
        )

        # LSTM takes [features, last_action, last_reward] as input
        self.recurrent = nn.LSTM(512 + self.action_dim + 1, self.hidden_dim, batch_first=True)

        # Dueling DQN head
        self.advantage = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, state: AgentState):
        """
        Single-step forward pass (for inference)

        Args:
            state: AgentState with obs, last_action, last_reward, hidden_state

        Returns:
            q_value: Q-values for each action [action_dim]
            recurrent_output: New hidden state tuple
        """
        latent = self.feature(state.obs / 255)

        recurrent_input = torch.cat((latent, state.last_action, state.last_reward), dim=1)

        _, recurrent_output = self.recurrent(recurrent_input.unsqueeze(1), state.hidden_state)

        hidden = recurrent_output[0]

        adv = self.advantage(hidden)
        val = self.value(hidden)
        q_value = val + adv - adv.mean(1, keepdim=True)

        # Squeeze all singleton dimensions: [1, 1, action_dim] -> [action_dim]
        return q_value.squeeze(), recurrent_output

    def calculate_q_(self, obs, last_action, last_reward, hidden_state, burn_in_steps, learning_steps, forward_steps):
        """
        Batch forward pass with burn-in and forward steps for n-step Q-learning

        Args:
            obs: Batched observations (batch_size, seq_len, C, H, W)
            last_action: Batched one-hot actions (batch_size, seq_len, action_dim)
            last_reward: Batched rewards (batch_size, seq_len, 1)
            hidden_state: Initial LSTM hidden state
            burn_in_steps: Steps for LSTM warm-up per sequence
            learning_steps: Steps used for gradient computation per sequence
            forward_steps: Steps for n-step return bootstrapping per sequence

        Returns:
            q_value: Q-values for learning steps (sum(learning_steps), action_dim)
        """
        batch_size, max_seq_len, *_ = obs.size()

        obs = obs.reshape(-1, *self.obs_shape)
        last_action = last_action.view(-1, self.action_dim)
        last_reward = last_reward.view(-1, 1)
        latent = self.feature(obs)

        seq_len = burn_in_steps + learning_steps + forward_steps

        recurrent_input = torch.cat((latent, last_action, last_reward), dim=1)
        recurrent_input = recurrent_input.view(batch_size, max_seq_len, -1)

        recurrent_input = pack_padded_sequence(recurrent_input, seq_len.cpu(), batch_first=True, enforce_sorted=False)

        self.recurrent.flatten_parameters()
        recurrent_output, _ = self.recurrent(recurrent_input, hidden_state)

        recurrent_output, _ = pad_packed_sequence(recurrent_output, batch_first=True)

        seq_start_idx = burn_in_steps + self.max_forward_steps
        forward_pad_steps = torch.minimum(self.max_forward_steps - forward_steps, learning_steps)

        hidden = []
        for hidden_seq, start_idx, end_idx, padding_length in zip(recurrent_output, seq_start_idx, seq_len, forward_pad_steps):
            hidden.append(hidden_seq[start_idx:end_idx])
            if padding_length > 0:
                hidden.append(hidden_seq[end_idx-1:end_idx].repeat(padding_length, 1))

        hidden = torch.cat(hidden)

        assert hidden.size(0) == torch.sum(learning_steps)

        adv = self.advantage(hidden)
        val = self.value(hidden)
        q_value = val + adv - adv.mean(1, keepdim=True)

        return q_value

    def calculate_q(self, obs, last_action, last_reward, hidden_state, burn_in_steps, learning_steps):
        """
        Batch forward pass for Q-value computation (without forward steps)

        Args:
            obs: Batched observations (batch_size, seq_len, C, H, W)
            last_action: Batched one-hot actions (batch_size, seq_len, action_dim)
            last_reward: Batched rewards (batch_size, seq_len, 1)
            hidden_state: Initial LSTM hidden state
            burn_in_steps: Steps for LSTM warm-up per sequence
            learning_steps: Steps used for gradient computation per sequence

        Returns:
            q_value: Q-values for learning steps (sum(learning_steps), action_dim)
        """
        batch_size, max_seq_len, *_ = obs.size()

        obs = obs.reshape(-1, *self.obs_shape)
        last_action = last_action.view(-1, self.action_dim)
        last_reward = last_reward.view(-1, 1)

        latent = self.feature(obs)

        seq_len = burn_in_steps + learning_steps

        recurrent_input = torch.cat((latent, last_action, last_reward), dim=1)
        recurrent_input = recurrent_input.view(batch_size, max_seq_len, -1)
        recurrent_input = pack_padded_sequence(recurrent_input, seq_len.cpu(), batch_first=True, enforce_sorted=False)

        recurrent_output, _ = self.recurrent(recurrent_input, hidden_state)

        recurrent_output, _ = pad_packed_sequence(recurrent_output, batch_first=True)

        hidden = torch.cat([output[burn_in:burn_in+learning] for output, burn_in, learning in zip(recurrent_output, burn_in_steps, learning_steps)], dim=0)

        adv = self.advantage(hidden)
        val = self.value(hidden)

        q_value = val + adv - adv.mean(1, keepdim=True)

        return q_value
