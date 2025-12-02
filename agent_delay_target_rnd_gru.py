# Copyright 2025 Keen Technologies, Inc.
# Modified to add RND (Random Network Distillation) + GRU memory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# agent_delay_target_rnd_gru.py
#
# agent_delay_target + RND + GRU memory for improved temporal reasoning
# GRU helps remember power pellet timing and other temporal dependencies

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ale_py import Action, ALEInterface, LoggerMode, roms
from pynvml import *

try:
    import wandb
except ImportError:
    wandb = None

from tqdm import tqdm


class RNDNetwork(nn.Module):
    """
    Small CNN for RND target/predictor networks.
    Takes normalized observation stacks and outputs a feature vector.
    Pre-computes FC layer size for CUDA graph compatibility.
    """
    def __init__(self, input_channels, feature_dim=512, input_height=128, input_width=128):
        super().__init__()
        # Simple CNN that progressively downsamples
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(),
        )
        self.feature_dim = feature_dim
        
        # Pre-compute conv output size for CUDA graph compatibility
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_height, input_width)
            conv_out = self.conv(dummy)
            self._conv_out_size = conv_out.view(1, -1).size(1)
        
        # Initialize FC layer upfront (required for CUDA graphs)
        self.fc = nn.Linear(self._conv_out_size, feature_dim)
        nn.init.orthogonal_(self.fc.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self.fc(conv_out)


class OddPooledGRU(nn.Module):
    """
    CNN feature extractor + GRU for temporal memory.
    
    Two modes:
    - Online inference: Uses persistent hidden state for action selection
    - Training: Uses fresh (zero) hidden states for replay samples
    """
    def __init__(
        self,
        input_shape,
        base_channels,
        output_channels,
        gru_hidden_size=128,
        kernel_size=3,
        pool_size=3,
        dirac=1,
        use_biases=0,
        norm=0,
        weighting=1,
    ):
        super().__init__()
        self.cnn = nn.ModuleList()
        self.pool_size = pool_size
        self.norm = norm
        self.weighting = weighting
        self.gru_hidden_size = gru_hidden_size
        self.base_channels = base_channels
        
        out_channels = base_channels
        in_channels = input_shape[1]
        img_height = input_shape[2]
        img_width = input_shape[3]
        
        while img_width > 3 or img_height > 3:
            c = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=use_biases)
            self.cnn.append(c)
            if dirac == 1:
                w = c.weight.data
                torch.nn.init.dirac_(w[: w.size(1)])
            in_channels = out_channels
            out_channels *= 2
            img_width = img_width // 2 + 1
            img_height = img_height // 2 + 1
        
        # CNN output features (before pooling to single values)
        self.cnn_out_channels = in_channels
        
        # Intermediate conv to reduce to features
        self.feature_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=use_biases)
        
        # GRU takes pooled CNN features as input
        self.gru = nn.GRU(
            input_size=base_channels,  # Pooled features from CNN
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        
        # Output head from GRU hidden state
        self.output_head = nn.Linear(gru_hidden_size, output_channels, bias=use_biases)
        
        # Persistent hidden state for ONLINE inference only (batch_size=1)
        self.online_hidden = None
        
    def init_online_hidden(self, device):
        """Initialize persistent hidden state for online inference."""
        self.online_hidden = torch.zeros(1, 1, self.gru_hidden_size, device=device)
        
    def reset_online_hidden(self):
        """Reset online hidden state at episode end. CUDA-graph compatible."""
        if self.online_hidden is not None:
            self.online_hidden.zero_()

    def _cnn_forward(self, x):
        """Shared CNN forward pass."""
        x = F.pad(x, (1, 0, 1, 0))
        for c in self.cnn:
            x = c(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            x = F.relu(x)
        x = self.feature_conv(x)
        x = F.relu(x)
        # Global average pooling
        return x.mean(dim=(2, 3))  # (batch_size, base_channels)

    def forward(self, x):
        """
        Training forward pass - uses FRESH (zero) hidden states.
        This is appropriate for replay buffer samples that don't have
        continuous temporal context.
        """
        batch_size = x.size(0)
        features = self._cnn_forward(x)
        
        # Fresh hidden state for each sample
        hidden = torch.zeros(1, batch_size, self.gru_hidden_size, 
                            device=x.device, dtype=features.dtype)
        
        # GRU forward
        gru_input = features.unsqueeze(1)
        gru_out, _ = self.gru(gru_input, hidden)
        
        # Output
        return self.output_head(gru_out.squeeze(1))
    
    def forward_online(self, x):
        """
        Online inference forward pass - uses PERSISTENT hidden state.
        Call this for action selection during rollout.
        Updates the internal online_hidden state.
        """
        assert x.size(0) == 1, "Online forward expects batch_size=1"
        
        if self.online_hidden is None:
            self.init_online_hidden(x.device)
        
        # Ensure dtype matches
        if self.online_hidden.dtype != x.dtype:
            self.online_hidden = self.online_hidden.to(dtype=x.dtype)
        
        features = self._cnn_forward(x)
        
        # GRU with persistent hidden
        gru_input = features.unsqueeze(1)
        gru_out, self.online_hidden = self.gru(gru_input, self.online_hidden)
        
        # Output
        return self.output_head(gru_out.squeeze(1))

    def forward_with_burnin(self, observation_ring, buffer_indexes, input_stack, episode_buffer, burn_in_length=8):
        """
        Training forward with burn-in to reconstruct hidden states.
        
        For each sample, processes preceding frames WITHOUT gradients to 
        build up the hidden state, then processes target frame WITH gradients.
        
        This matches the hidden state distribution seen during online inference.
        Resets hidden state at episode boundaries.
        """
        batch_size = buffer_indexes.size(0)
        ring_buffer_size = observation_ring.size(0)
        obs_channels = observation_ring.size(1)
        obs_height = observation_ring.size(2)
        obs_width = observation_ring.size(3)
        input_channels = input_stack * obs_channels
        dtype = next(self.parameters()).dtype
        device = observation_ring.device
        
        # Get target episode for each sample (to detect episode boundaries)
        target_episodes = episode_buffer[buffer_indexes]
        
        # Initialize hidden states for all samples
        hidden = torch.zeros(1, batch_size, self.gru_hidden_size, device=device, dtype=dtype)
        
        # Burn-in: process preceding frames without gradients
        with torch.no_grad():
            for step in range(-burn_in_length, 0):
                # Get observation stacks for this burn-in step
                step_indexes = (buffer_indexes + step) % ring_buffer_size
                
                # Check if this step is in the same episode as target
                step_episodes = episode_buffer[step_indexes]
                same_episode = (step_episodes == target_episodes).float().view(1, -1, 1)
                
                # Reset hidden for samples that crossed episode boundary
                hidden = hidden * same_episode
                
                # Stack frames
                final_stack_indexes = step_indexes.unsqueeze(dim=1).expand(batch_size, input_stack)
                offsets = torch.arange(-input_stack + 1, 1, device=device).unsqueeze(dim=0).expand(batch_size, input_stack)
                ring_indexes = (offsets + final_stack_indexes) % ring_buffer_size
                obs_stacks = observation_ring[ring_indexes]
                obs_stacks = obs_stacks.view(batch_size, input_channels, obs_height, obs_width)
                obs_stacks = obs_stacks.to(dtype=dtype) / 255.0
                
                # CNN forward
                features = self._cnn_forward(obs_stacks)
                
                # GRU step (update hidden)
                gru_input = features.unsqueeze(1)
                _, hidden = self.gru(gru_input, hidden)
        
        # Now get the target observation stacks
        final_stack_indexes = buffer_indexes.unsqueeze(dim=1).expand(batch_size, input_stack)
        offsets = torch.arange(-input_stack + 1, 1, device=device).unsqueeze(dim=0).expand(batch_size, input_stack)
        ring_indexes = (offsets + final_stack_indexes) % ring_buffer_size
        observation_stacks = observation_ring[ring_indexes]
        observation_stacks = observation_stacks.view(batch_size, input_channels, obs_height, obs_width)
        observation_stacks = observation_stacks.to(dtype=dtype) / 255.0
        
        # Forward with gradients using burned-in hidden
        features = self._cnn_forward(observation_stacks)
        gru_input = features.unsqueeze(1)
        
        # Detach hidden so gradients don't flow through burn-in
        gru_out, _ = self.gru(gru_input, hidden.detach())
        
        return self.output_head(gru_out.squeeze(1)), observation_stacks

    def forward_with_stored_hidden(self, observation_ring, buffer_indexes, input_stack, hidden_state_buffer):
        """
        Training forward using STORED hidden states from the replay buffer.
        
        This is faster than burn-in and still correct because we use the exact
        hidden state that was active when the observation was collected.
        """
        batch_size = buffer_indexes.size(0)
        ring_buffer_size = observation_ring.size(0)
        obs_channels = observation_ring.size(1)
        obs_height = observation_ring.size(2)
        obs_width = observation_ring.size(3)
        input_channels = input_stack * obs_channels
        dtype = next(self.parameters()).dtype
        device = observation_ring.device
        
        # Load stored hidden states for each sample
        # hidden_state_buffer shape: (ring_buffer_size, hidden_size)
        # We need shape: (1, batch_size, hidden_size) for GRU
        stored_hidden = hidden_state_buffer[buffer_indexes]  # (batch, hidden_size)
        hidden = stored_hidden.unsqueeze(0).to(dtype=dtype)  # (1, batch, hidden_size)
        
        # Build observation stacks
        final_stack_indexes = buffer_indexes.unsqueeze(dim=1).expand(batch_size, input_stack)
        offsets = torch.arange(-input_stack + 1, 1, device=device).unsqueeze(dim=0).expand(batch_size, input_stack)
        ring_indexes = (offsets + final_stack_indexes) % ring_buffer_size
        observation_stacks = observation_ring[ring_indexes]
        observation_stacks = observation_stacks.view(batch_size, input_channels, obs_height, obs_width)
        observation_stacks = observation_stacks.to(dtype=dtype) / 255.0
        
        # Forward with stored hidden (detached so no grad through stored states)
        features = self._cnn_forward(observation_stacks)
        gru_input = features.unsqueeze(1)
        gru_out, _ = self.gru(gru_input, hidden.detach())
        
        return self.output_head(gru_out.squeeze(1)), observation_stacks


class OddPooled(nn.Module):
    """Original OddPooled without GRU for comparison."""
    def __init__(
        self,
        input_shape,
        base_channels,
        output_channels,
        kernel_size=3,
        pool_size=3,
        dirac=1,
        use_biases=0,
        norm=0,
        weighting=1,
    ):
        super().__init__()
        self.cnn = nn.ModuleList()
        self.pool_size = pool_size
        self.norm = norm
        self.weighting = weighting
        out_channels = base_channels
        in_channels = input_shape[1]
        img_height = input_shape[2]
        img_width = input_shape[3]
        while img_width > 3 or img_height > 3:
            c = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=use_biases)
            self.cnn.append(c)
            if dirac == 1:
                w = c.weight.data
                torch.nn.init.dirac_(w[: w.size(1)])
            in_channels = out_channels
            out_channels *= 2
            img_width = img_width // 2 + 1
            img_height = img_height // 2 + 1
        self.final = nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1, bias=use_biases)
        self.weight_tensor = torch.tensor(
            [[[[4 / 49, 6 / 49, 4 / 49], [6 / 49, 9 / 49, 6 / 49], [4 / 49, 6 / 49, 4 / 49]]]]
        )

    def forward(self, x):
        x = F.pad(x, (1, 0, 1, 0))
        for c in self.cnn:
            x = c(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            x = F.relu(x)
        conv_out = self.final(x)
        if self.weighting == 0:
            return conv_out[:, :, 1, 1]
        elif self.weighting == 1:
            return conv_out.mean(dim=(2, 3))
        elif self.weighting == 2:
            weighted = conv_out * self.weight_tensor
            return weighted.sum(dim=(2, 3))
        assert False, 'bad weighting'


def model_parameter_count(model):
    count = 0
    for p in model.parameters():
        count += p.numel()
    return count


# Wraps a function and allows it to be turned into a cuda graph
class cuda_graph_wrapper:
    def __init__(self, func, stream, use_cuda_graphs, args):
        super().__init__()
        self.graph_warmups = 3 if use_cuda_graphs else -1
        self.cuda_graph = None
        self.func = func
        self.stream = stream
        self.args = args
        return

    def __call__(self):
        if self.cuda_graph:
            with torch.cuda.stream(self.stream):
                self.cuda_graph.replay()
            return
        if self.graph_warmups == -1:
            self.func(*self.args)
            return
        
        if self.graph_warmups > 0:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self.func(*self.args)
            torch.cuda.current_stream().wait_stream(self.stream)
        else:
            print('capture start')
            torch.cuda.synchronize()
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graph, stream=self.stream):
                self.func(*self.args)
            print('capture stop')
            self.stream.wait_stream(torch.cuda.current_stream())
            self.cuda_graph.replay()

        self.graph_warmups -= 1


def build_downsample(src, dst):
    mat = torch.zeros(src, dst)
    scale = dst / src
    for j in range(dst):
        low = j / dst * src
        low_i = math.floor(low)
        low_frac = low_i + 1 - low
        high = (j + 1) / dst * src
        high_i = math.floor(high)
        high_frac = high - high_i
        mat[low_i][j] = low_frac * scale
        for i in range(low_i + 1, high_i):
            mat[i][j] = scale
        if high_frac > 0.0:
            mat[high_i][j] = high_frac * scale
    return mat


def train_function_rnd_gru(
    # constants
    input_stack,
    train_batch,
    train_steps,
    train_indexes,
    min_train_frames,
    resample_width,
    resample_height,
    temperature_log2,
    multisteps_max,
    ema_log2,
    lr_tensor,
    weight_decay,
    online_batch,
    online_loss_scale,
    use_weight_norm,
    reward_discounts,
    value_discounts,
    # RND constants
    intrinsic_reward_scale,
    rnd_update_proportion,
    # GRU burn-in
    burn_in_length,
    # variable inputs
    new_observations,
    tensor_u,
    observation_ring,
    # GRU hidden state management
    episode_end_flags,  # For resetting online hidden state
    # policy output
    selected_action_index,
    # GRU hidden state storage
    hidden_state_buffer,
    # output
    state_value_buffer,
    distribution_factor_buffer,
    policy_actions_buffer,
    episode_buffer,
    reward_buffer,
    loss_buffer,
    target_ema,
    train_loss,
    train_loss_ema,
    avg_error_ema,
    max_error_ema,
    # RND outputs
    intrinsic_reward_ema,
    rnd_loss_ema,
    # updated models
    optimizer,
    linear_optimizer,
    training_model,
    anchor_model,
    # RND models
    rnd_target,
    rnd_predictor,
    rnd_optimizer,
):
    with torch.no_grad():
        # pull some constants from tensor dimensions
        ring_buffer_size, obs_channels, obs_height, obs_width = observation_ring.shape
        frame_skip, input_height, input_width, _ = new_observations.shape
        input_channels = input_stack * obs_channels

        # The rewards and episodes have been added to the buffers for these indexes
        torch.add(tensor_u, frame_skip, out=tensor_u)

        # resample the new images to a different resolution and store them in the ring buffer
        atari_resampled = new_observations.permute(0, 3, 1, 2)
        rows = frame_skip * obs_channels
        atari_resampled = atari_resampled.reshape(rows * input_height, input_width).float() @ resample_width
        atari_resampled = atari_resampled.view(rows, input_height, obs_width)
        atari_resampled = atari_resampled.permute(0, 2, 1) @ resample_height
        atari_resampled = atari_resampled.view(rows, obs_width, obs_height).permute(0, 2, 1)
        atari_resampled = atari_resampled.reshape(frame_skip, obs_channels, obs_height, obs_width)
        atari_resampled = atari_resampled.clamp(min=0.0, max=255.0).to(dtype=torch.uint8)
        ring = (tensor_u - 3) % observation_ring.shape[0] // frame_skip
        observation_ring.view(
            ring_buffer_size // frame_skip, frame_skip, obs_channels, obs_height, obs_width
        ).index_put_((ring,), atari_resampled)

        # Reset online GRU hidden state if episode ended
        # episode_end_flags is (1,) tensor: 1 if episode ended, 0 otherwise
        # CUDA-graph compatible: multiply by (1 - flag) to zero out on episode end
        if training_model.online_hidden is not None:
            training_model.online_hidden = training_model.online_hidden * (1.0 - episode_end_flags.view(1, 1, 1))

        # see which samples from the replay buffer are being evaluated
        train_step = tensor_u // train_steps
        index_indexes = torch.arange(train_batch) + train_step * train_batch
        buffer_indexes = train_indexes[index_indexes]

        # Build online observation stack for action selection
        online_index = buffer_indexes[:1]
        final_stack_indexes = online_index.unsqueeze(dim=1).expand(1, input_stack)
        offsets = torch.arange(-input_stack + 1, 1).unsqueeze(dim=0).expand(1, input_stack)
        ring_indexes = (offsets + final_stack_indexes) % ring_buffer_size
        online_obs = observation_ring[ring_indexes]
        online_obs = online_obs.view(1, input_channels, obs_height, obs_width)
        online_obs = online_obs.to(next(training_model.parameters()).dtype) / 255.0
        
        # === Online action selection with PERSISTENT hidden state ===
        # Store the hidden state BEFORE processing (this is the INPUT hidden for this observation)
        # Use the actual ring buffer index where current observation is stored, not buffer_indexes[0]
        # This matches how reward_buffer and episode_buffer are indexed
        current_ring_idx = (tensor_u - 3) % ring_buffer_size
        if training_model.online_hidden is not None:
            hidden_state_buffer[current_ring_idx] = training_model.online_hidden.squeeze()
        
        # This forward pass updates the online_hidden state for temporal continuity
        online_values = training_model.forward_online(online_obs)
        
        # Action selection from online values
        online_q = online_values[:, :-1]
        online_probs = F.softmax(online_q / (avg_error_ema * 2**temperature_log2), dim=1)
        sample = torch.multinomial(online_probs[0], num_samples=1)
        selected_action_index.copy_(sample[0])

        # the next frame_skip frames will use this selected_action_index
        num_distributions = distribution_factor_buffer.shape[1]
        dist = F.one_hot(selected_action_index, num_classes=num_distributions).float()
        online_indexes = (torch.arange(frame_skip) + (tensor_u + 1)).clamp(max=distribution_factor_buffer.shape[0] - 1)
        distribution_factor_buffer[online_indexes] = dist.unsqueeze(dim=0).expand(frame_skip, num_distributions)
        policy_actions_buffer[online_indexes] = selected_action_index

    # === Training forward pass ===
    if burn_in_length > 0:
        # Burn-in: reconstruct hidden states by processing preceding frames (slower, most accurate)
        train_values, observation_stacks = training_model.forward_with_burnin(
            observation_ring, buffer_indexes, input_stack, episode_buffer, burn_in_length=burn_in_length
        )
    elif burn_in_length == 0:
        # Stored hidden: use hidden states saved during collection (fast, correct)
        train_values, observation_stacks = training_model.forward_with_stored_hidden(
            observation_ring, buffer_indexes, input_stack, hidden_state_buffer
        )
    else:
        # Zero hidden: fastest but train/inference mismatch (burn_in_length < 0)
        # Build observation stacks manually
        final_stack_indexes = buffer_indexes.unsqueeze(dim=1).expand(train_batch, input_stack)
        offsets = torch.arange(-input_stack + 1, 1).unsqueeze(dim=0).expand(train_batch, input_stack)
        ring_indexes = (offsets + final_stack_indexes) % ring_buffer_size
        observation_stacks = observation_ring[ring_indexes]
        observation_stacks = observation_stacks.view(train_batch, input_channels, obs_height, obs_width)
        observation_stacks = observation_stacks.to(next(training_model.parameters()).dtype) / 255.0
        train_values = training_model(observation_stacks)
    num_model_distributions = train_values.shape[1]

    # ===== RND: Compute intrinsic rewards =====
    with torch.no_grad():
        rnd_target_features = rnd_target(observation_stacks)
    
    rnd_predictor_features = rnd_predictor(observation_stacks.detach())
    
    with torch.no_grad():
        intrinsic_rewards = (rnd_target_features - rnd_predictor_features.detach()).pow(2).mean(dim=1)
        current_intrinsic_mean = intrinsic_rewards.mean()
        torch.lerp(intrinsic_reward_ema, current_intrinsic_mean, 2**ema_log2, out=intrinsic_reward_ema)
        normalized_intrinsic = intrinsic_rewards / (intrinsic_reward_ema + 1e-8)

    with torch.no_grad():
        # build target values for training
        all_q = train_values[:, :-1].detach()
        probs = F.softmax(all_q / (avg_error_ema * 2**temperature_log2), dim=1)
        all_v = (all_q * probs).sum(dim=1)

        state_value_buffer[buffer_indexes] = all_v

        # get the observed rewards up to the max bootstrap point
        reward_indexes = buffer_indexes.unsqueeze(dim=1) + torch.arange(multisteps_max).unsqueeze(dim=0)
        reward_indexes %= reward_buffer.shape[0]
        state_values = state_value_buffer[reward_indexes.flatten()].view(train_batch, multisteps_max)
        observed_rewards = reward_buffer[reward_indexes.flatten()].view(train_batch, multisteps_max)

        # mask all rewards off that cross an episode boundary
        initial_episode = episode_buffer[buffer_indexes]
        episodes = episode_buffer[reward_indexes.flatten()].view(train_batch, multisteps_max)
        episodes_match = episodes.eq(initial_episode.unsqueeze(dim=1)).float()
        state_values *= episodes_match
        observed_rewards *= episodes_match

        # build targets out of the observed rewards and the bootstrap values
        blended_rewards = observed_rewards @ reward_discounts
        blended_states = state_values @ value_discounts

        # Add intrinsic rewards to return targets
        return_targets = blended_rewards + blended_states + intrinsic_reward_scale * normalized_intrinsic.unsqueeze(1)

        torch.lerp(target_ema, return_targets[online_batch:].mean(), 2**ema_log2, out=target_ema)

        distribution_factors = distribution_factor_buffer[buffer_indexes]
        distribution_factors[:online_batch].zero_()

        training_targets = return_targets.view(train_batch, 1)
        training_targets = training_targets.expand(train_batch, num_model_distributions)

    loss_individual = F.mse_loss(train_values, training_targets, reduction='none')
    distribution_factors[online_batch : online_batch * 2] *= online_loss_scale
    loss_individual = loss_individual * distribution_factors
    loss_individual = loss_individual * (tensor_u > min_train_frames)

    loss_buffer[buffer_indexes] = loss_individual.sum(dim=1)

    loss = loss_individual.sum() / (train_batch - online_batch)

    linear_optimizer.zero_grad()
    optimizer.zero_grad()
    loss.backward()

    avg_error = loss_individual.detach()[online_batch:].sqrt().sum() / (train_batch - online_batch)
    max_error = loss_individual.detach().sqrt().max()

    torch.lerp(train_loss_ema, loss.detach(), 2**ema_log2, out=train_loss_ema)
    torch.lerp(avg_error_ema, avg_error.detach(), 2**ema_log2, out=avg_error_ema)
    torch.lerp(max_error_ema, max_error.detach(), 2**ema_log2, out=max_error_ema)

    training = list(training_model.parameters())
    if weight_decay != 0.0:
        with torch.no_grad():
            anchor_network_alpha = lr_tensor * abs(weight_decay)
            init = list(anchor_model.parameters())
            for p in range(len(training)):
                if init[p].dim() > 1:
                    torch.lerp(training[p], init[p], anchor_network_alpha, out=training[p])

    optimizer.step()
    linear_optimizer.step()

    if use_weight_norm:
        with torch.no_grad():
            # Only normalize CNN weights (4D tensors), skip GRU and linear layers
            for name, p in training_model.named_parameters():
                if 'cnn' in name and p.dim() == 4:
                    norms = torch.norm(p.flatten(start_dim=1), dim=1)
                    p /= norms.view(-1, 1, 1, 1)

    train_loss.copy_(loss.detach())

    # ===== RND: Train the predictor network =====
    rnd_loss = F.mse_loss(rnd_predictor_features, rnd_target_features.detach())
    rnd_loss = rnd_loss * rnd_update_proportion * (tensor_u > min_train_frames)
    
    rnd_optimizer.zero_grad()
    rnd_loss.backward()
    rnd_optimizer.step()
    
    torch.lerp(rnd_loss_ema, rnd_loss.detach(), 2**ema_log2, out=rnd_loss_ema)


class Agent:
    def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
        # defaults that might be overridden by explicit experiment runs
        self.gpu = 0

        # Value / reward
        self.target_network_alpha_log2 = -7
        self.ema_log2 = -10
        self.reward_discount = 0.9975
        self.multisteps_max = 64
        self.td_lambda = 0.95
        self.death_punishment = 0

        # The observation
        self.frame_skip = 4
        self.input_width = 160
        self.input_height = 210
        self.input_stack = 16
        self.obs_width = 128
        self.obs_height = 128
        self.obs_channels = 3

        # exploration
        self.greedy_max = 0.99
        self.greedy_ramp = 100_000
        self.temperature_log2 = -7

        # The model
        self.load_file = None
        self.seed = seed
        self.num_actions = num_actions
        self.use_model = 3
        self.kernel_size = 3
        self.base_width = 80
        self.use_biases = 0
        self.use_dirac = 1

        self.use_precision = 0

        # GRU settings
        self.gru_hidden_size = 128  # GRU hidden dimension

        # training
        self.use_softv = 1
        self.use_weight_norm = 1
        self.repeat_train = 1
        self.min_train_frames = 256

        self.base_lr_log2 = -16
        self.lr_log2 = -18

        self.train_batch = 32
        self.online_batch = 4
        self.online_loss_scale = 2
        self.train_steps = 4

        self.ring_buffer_size = 200 * 1024

        self.weight_decay = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.momentum = 0.9

        self.total_frames = total_frames

        # CUDA graphs - disabled by default for GRU (can be enabled with care)
        self.use_cuda_graphs = False  # GRU hidden state updates are tricky with graphs

        # GRU burn-in length (0 = no burn-in, faster but less accurate hidden states)
        self.burn_in_length = 8  # Default: 8 steps of burn-in for proper RNN training

        # RND hyperparameters
        self.intrinsic_reward_scale = 0.1
        self.rnd_feature_dim = 512
        self.rnd_lr = 1e-4
        self.rnd_update_proportion = 0.25

        # dynamically override configuration
        for key, value in kwargs.items():
            assert hasattr(self, key), f"Unknown parameter: {key}"
            setattr(self, key, value)

        self.dev = f'cuda:{self.gpu}'

        self.ring_buffer_size -= self.ring_buffer_size % self.frame_skip

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        print('torch version: ', torch.__version__)
        print('cuda version : ', torch.version.cuda)
        print('dev          : ', self.dev)
        print('GRU hidden   : ', self.gru_hidden_size)

        torch.set_num_threads(1)
        torch.set_default_device(self.dev)
        torch.cuda.set_device(self.dev)
        torch.set_printoptions(linewidth=160)

        self.input_channels = self.input_stack * self.obs_channels

        self.observation_ring = torch.zeros(
            self.ring_buffer_size, self.obs_channels, self.obs_height, self.obs_width, dtype=torch.uint8
        )

        self.new_observations = torch.zeros(self.frame_skip, self.input_height, self.input_width, self.obs_channels)

        self.resample_width = build_downsample(self.input_width, self.obs_width).to(self.dev)
        self.resample_height = build_downsample(self.input_height, self.obs_height).to(self.dev)

        self.train_loss = torch.tensor(0.0)
        self.train_loss_ema = torch.tensor(0.0)
        self.avg_error_ema = torch.tensor(10.0)
        self.max_error_ema = torch.tensor(0.0)
        self.target_ema = torch.tensor(1.0)

        # RND EMAs
        self.intrinsic_reward_ema = torch.tensor(1.0)
        self.rnd_loss_ema = torch.tensor(0.0)

        self.u = self.frame_skip - 1
        self.tensor_u = torch.tensor(3, dtype=torch.int64)

        self.frame_count = 0
        self.observation_rgb8 = np.zeros(
            (self.frame_skip, self.input_height, self.input_width, self.obs_channels), dtype=np.uint8
        )
        self.rewards = np.zeros(self.frame_skip)
        self.end_of_episodes = np.zeros(self.frame_skip)

        # Episode end flag for GRU reset (CUDA graph compatible)
        self.episode_end_flags = torch.zeros(1, dtype=torch.float32)

        self.lr_tensor = torch.tensor(2**self.lr_log2)
        self.base_lr_tensor = torch.tensor(2**self.base_lr_log2)
        self.lr_warmup_start = torch.tensor(0, dtype=torch.int64)

        self.selected_action_index = torch.tensor(0, dtype=torch.int64)

        self.punishment_tensor = torch.tensor(self.death_punishment)

        fmt = torch.float32
        if self.use_precision == 1:
            fmt = torch.bfloat16

        total_model_outputs = self.num_actions + 1

        # train_indexes[] will be the training location, which must have multisteps_max valid after it.
        total_train = self.total_frames // self.train_steps

        highs = torch.arange(total_train) * self.train_steps - self.multisteps_max + self.frame_skip - 1
        lows = torch.clamp(highs - (self.ring_buffer_size - (self.multisteps_max * 2)), self.input_stack)
        index_fraction = torch.rand(total_train, self.train_batch)
        self.train_indexes = (
            (highs - lows).unsqueeze(dim=1).expand(total_train, self.train_batch) * index_fraction
        ).long() + lows.unsqueeze(dim=1)

        # force the online indexes to use the most recent frames
        assert self.online_batch > 0 and self.online_batch <= self.train_batch // 2
        for i in range(self.online_batch):
            self.train_indexes[:, i] = highs + self.multisteps_max - i
            self.train_indexes[:, self.online_batch + i] = highs - i

        self.train_indexes = self.train_indexes.flatten()

        self.state_value_buffer = torch.zeros(self.ring_buffer_size)
        self.distribution_factor_buffer = torch.zeros(self.ring_buffer_size, total_model_outputs)
        self.policy_actions_buffer = torch.zeros(self.ring_buffer_size, dtype=torch.int64)
        self.episode_buffer = torch.zeros(self.ring_buffer_size, dtype=torch.int64)
        self.reward_buffer = torch.zeros(self.ring_buffer_size)
        self.loss_buffer = torch.zeros(self.ring_buffer_size)
        
        # Store GRU hidden states for each observation (for training without burn-in)
        self.hidden_state_buffer = torch.zeros(self.ring_buffer_size, self.gru_hidden_size)

        self.episode_number = 0
        self.train_losses = []

        torch.random.manual_seed(self.seed)

        # Use OddPooledGRU instead of OddPooled
        print("Creating OddPooledGRU model with GRU memory...")
        self.training_model = OddPooledGRU(
            (1, self.input_channels, self.obs_height, self.obs_width),
            self.base_width,
            total_model_outputs,
            gru_hidden_size=self.gru_hidden_size,
            use_biases=self.use_biases,
            dirac=self.use_dirac,
            kernel_size=self.kernel_size,
            weighting=(self.use_model - 1) if self.use_model >= 1 else 1,
        )
        print(self.training_model)
        print('parameters: ', model_parameter_count(self.training_model))

        if self.load_file is not None:
            checkpoint = torch.load(self.load_file, weights_only=True)
            if isinstance(checkpoint, dict) and 'training_model' in checkpoint:
                self.training_model.load_state_dict(checkpoint['training_model'])
            else:
                self.training_model.load_state_dict(checkpoint)

        self.training_model.to(dtype=fmt)
        self.training_model.train()
        
        # Initialize persistent online hidden state for action selection
        self.training_model.init_online_hidden(self.dev)

        self.train_values = torch.zeros(self.train_batch, total_model_outputs)

        # td-lambda combination
        self.reward_discounts = torch.zeros(self.multisteps_max)
        self.value_discounts = torch.zeros(self.multisteps_max)
        total = 0.0
        for step in range(1, self.multisteps_max):
            factor = self.td_lambda ** (step - 1)
            total += factor
            self.value_discounts[step] = factor * (self.reward_discount**step)
            for n in range(step):
                self.reward_discounts[n] += factor * (self.reward_discount**n)

        self.reward_discounts = (self.reward_discounts / total).unsqueeze(dim=1)
        self.value_discounts = (self.value_discounts / total).unsqueeze(dim=1)

        self.anchor_model = copy.deepcopy(self.training_model)
        if self.weight_decay > 0:
            with torch.no_grad():
                for p in self.anchor_model.parameters():
                    p.zero_()
        adamwd = 0

        parms = list(self.training_model.parameters())
        # GRU has more parameters, need to handle carefully
        # Exclude GRU and output head from CNN optimizer
        cnn_parms = []
        other_parms = []
        for name, param in self.training_model.named_parameters():
            if 'gru' in name or 'output_head' in name:
                other_parms.append(param)
            else:
                cnn_parms.append(param)

        self.optimizer = torch.optim.AdamW(
            cnn_parms,
            lr=self.base_lr_tensor,
            fused=True,
            capturable=self.use_cuda_graphs,
            weight_decay=adamwd,
            betas=(self.beta1, self.beta2),
        )
        self.linear_optimizer = torch.optim.SGD(other_parms, lr=2**self.lr_log2, momentum=self.momentum)

        # RND Networks
        print("Initializing RND networks...")
        self.rnd_target = RNDNetwork(
            self.input_channels, self.rnd_feature_dim,
            input_height=self.obs_height, input_width=self.obs_width
        )
        self.rnd_target.to(dtype=fmt)  # Match main model dtype
        self.rnd_target.eval()
        for param in self.rnd_target.parameters():
            param.requires_grad = False
        
        self.rnd_predictor = RNDNetwork(
            self.input_channels, self.rnd_feature_dim,
            input_height=self.obs_height, input_width=self.obs_width
        )
        self.rnd_predictor.to(dtype=fmt)  # Match main model dtype
        self.rnd_predictor.train()
        
        print(f'RND target parameters: {model_parameter_count(self.rnd_target)}')
        print(f'RND predictor parameters: {model_parameter_count(self.rnd_predictor)}')
        
        self.rnd_optimizer = torch.optim.AdamW(
            self.rnd_predictor.parameters(),
            lr=self.rnd_lr,
            capturable=self.use_cuda_graphs,
        )

        self.intrinsic_reward_scale_tensor = torch.tensor(self.intrinsic_reward_scale)
        self.rnd_update_proportion_tensor = torch.tensor(self.rnd_update_proportion)
        self.burn_in_length_value = self.burn_in_length  # For passing to train function

        self.spin_stream = torch.cuda.Stream(priority=0)
        self.train_stream = torch.cuda.Stream(priority=0)
        
        self.train_graph = cuda_graph_wrapper(
            train_function_rnd_gru,
            self.train_stream,
            self.use_cuda_graphs,
            [
                # constants
                self.input_stack,
                self.train_batch,
                self.train_steps,
                self.train_indexes,
                self.min_train_frames,
                self.resample_width,
                self.resample_height,
                self.temperature_log2,
                self.multisteps_max,
                self.ema_log2,
                self.lr_tensor,
                self.weight_decay,
                self.online_batch,
                self.online_loss_scale,
                self.use_weight_norm,
                self.reward_discounts,
                self.value_discounts,
                # RND constants
                self.intrinsic_reward_scale_tensor,
                self.rnd_update_proportion_tensor,
                # GRU burn-in
                self.burn_in_length_value,
                # variable state
                self.new_observations,
                self.tensor_u,
                self.observation_ring,
                # GRU hidden state management
                self.episode_end_flags,
                # policy output
                self.selected_action_index,
                # GRU hidden state storage
                self.hidden_state_buffer,
                # output
                self.state_value_buffer,
                self.distribution_factor_buffer,
                self.policy_actions_buffer,
                self.episode_buffer,
                self.reward_buffer,
                self.loss_buffer,
                self.target_ema,
                self.train_loss,
                self.train_loss_ema,
                self.avg_error_ema,
                self.max_error_ema,
                # RND outputs
                self.intrinsic_reward_ema,
                self.rnd_loss_ema,
                # updated models
                self.optimizer,
                self.linear_optimizer,
                self.training_model,
                self.anchor_model,
                # RND models
                self.rnd_target,
                self.rnd_predictor,
                self.rnd_optimizer,
            ],
        )

    def frame(self, observation_rgb8, reward, end_of_episode):
        assert observation_rgb8.shape == (self.input_height, self.input_width, self.obs_channels)

        i = self.frame_count % self.frame_skip
        self.observation_rgb8[i] = observation_rgb8
        self.rewards[i] = reward
        self.end_of_episodes[i] = end_of_episode
        self.frame_count += 1

        if i != (self.frame_skip - 1):
            return self.selected_action_index.item()

        if self.u > self.total_frames - self.frame_skip:
            return 0

        with torch.cuda.stream(self.spin_stream):
            self.new_observations.copy_(torch.from_numpy(self.observation_rgb8))
            # Handle ring buffer wraparound for reward assignment
            ring_size = self.reward_buffer.shape[0]
            for i in range(self.frame_skip):
                self.reward_buffer[(self.u + i) % ring_size] = self.rewards[i]

            # Track episode ends for GRU reset
            episode_ended = 0.0
            for i in range(self.frame_skip):
                self.episode_number += int(self.end_of_episodes[i] > 0)
                self.episode_buffer[(self.u + 1 + i) % ring_size] = self.episode_number
                if self.end_of_episodes[i] > 0:
                    episode_ended = 1.0
            
            # Update episode end flag for GRU (CUDA graph compatible)
            self.episode_end_flags.fill_(episode_ended)

            self.u += self.frame_skip
            self.selected_action_index.fill_(-1)

        torch.cuda.nvtx.range_push("train")

        with torch.cuda.stream(self.spin_stream):
            self.train_losses.append(self.train_loss_ema.item())
            self.train_losses.append(self.avg_error_ema.item())
            self.train_losses.append(self.max_error_ema.item())
            self.train_losses.append(self.target_ema.item())
            self.train_losses.append(self.intrinsic_reward_ema.item())
            self.train_losses.append(self.rnd_loss_ema.item())

        self.train_graph()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("spin")
        with torch.cuda.stream(self.spin_stream):
            spins = 0
            while self.selected_action_index.item() == -1:
                time.sleep(0.0001)
                spins += 1
        torch.cuda.nvtx.range_pop()

        return self.selected_action_index.item()

    def save_model(self, filename):
        torch.save({
            'training_model': self.training_model.state_dict(),
            'rnd_predictor': self.rnd_predictor.state_dict(),
            'rnd_target': self.rnd_target.state_dict(),
            'online_hidden': self.training_model.online_hidden,
        }, filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename, weights_only=False)
        self.training_model.load_state_dict(checkpoint['training_model'])
        if 'rnd_predictor' in checkpoint:
            self.rnd_predictor.load_state_dict(checkpoint['rnd_predictor'])
        if 'rnd_target' in checkpoint:
            self.rnd_target.load_state_dict(checkpoint['rnd_target'])
        if 'online_hidden' in checkpoint and checkpoint['online_hidden'] is not None:
            self.training_model.online_hidden = checkpoint['online_hidden']


def main():
    parser = argparse.ArgumentParser(description='Delay Target RND GRU Agent')
    parser.add_argument('rank', type=int, nargs='?', default=0, help='GPU rank / process ID')
    parser.add_argument('--mode', type=str, default='', choices=['', 'atari', 'physical'], help='Training mode')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='physical-atari-gru', help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Wandb entity/team name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--game', type=str, default=None, help='Game to play (overrides mode default)')
    parser.add_argument('--total-frames', type=int, default=None, help='Total frames to train')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--load', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--gru-hidden', type=int, default=128, help='GRU hidden size')
    parser.add_argument('--burn-in', type=int, default=8, help='GRU training mode: >0=burn-in steps, 0=stored hidden, -1=zero hidden (fastest)')
    args = parser.parse_args()

    rank = args.rank

    parms = {'gpu': rank % 8}
    
    # Add GRU settings
    parms['gru_hidden_size'] = args.gru_hidden
    parms['burn_in_length'] = args.burn_in

    lives_as_episodes = False

    ale = ALEInterface()
    ale.setLoggerMode(LoggerMode.Error)

    data_dir = args.results_dir
    os.makedirs(data_dir, exist_ok=True)

    save_model = True
    save_incremental_models = True
    last_model_save = -1
    max_frames_without_reward = 18_000

    if args.mode == 'atari':
        atari_list = ['krull', 'defender', 'battle_zone', 'ms_pacman', 'seaquest', 'yars_revenge', 'name_this_game', 'beam_rider']
        game = atari_list[rank % 8]
        seed = rank // 24
        total_frames = 1_000_000
        ale.setFloat('repeat_action_probability', 0.0)
        reduce_action_set = 1
        delay_frames = 0
    elif args.mode == 'physical':
        physical_list = ['centipede', 'up_n_down', 'qbert', 'battle_zone', 'krull', 'defender', 'ms_pacman', 'atlantis']
        game = physical_list[rank % 8]
        total_frames = 20_000_000
        parms['ring_buffer_size'] = 1_500_000
        parms['multisteps_max'] = 64
        parms['td_lambda'] = 0.95
        parms['online_loss_scale'] = 2
        parms['train_batch'] = 32
        parms['lr_log2'] = -18
        parms['base_lr_log2'] = -16
        seed = (rank // 8) % 4
        reduce_action_set = 2
        delay_frames = 6
    else:
        reduce_action_set = 2
        total_frames = 2_000_000
        parms['lr_log2'] = -17
        parms['base_lr_log2'] = -15
        seed = 0
        game = 'ms_pacman'
        delay_frames = 6
        if game == 'breakout':
            delay_frames = 0

    if args.game:
        game = args.game
    if args.total_frames:
        total_frames = args.total_frames

    rom_path = roms.get_rom_path(game)
    ale.loadROM(rom_path)
    ale.reset_game()

    if reduce_action_set == 0:
        action_set = ale.getLegalActionSet()
    else:
        if reduce_action_set == 2 and (game == 'ms_pacman' or game == 'qbert'):
            action_set = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        else:
            action_set = ale.getMinimalActionSet()
    num_actions = len(action_set)
    print(f'{num_actions} actions: {action_set}')

    name = f'delay_rnd_gru_{game}{delay_frames}'
    for k, v in parms.items():
        if k != 'gpu':
            name += '_'
            name += str(v)
    print(name)

    agent = Agent(data_dir, seed, num_actions, total_frames, **parms)

    if args.load:
        print(f"Loading checkpoint from {args.load}")
        agent.load_model(args.load)

    run = None
    if args.wandb and wandb is not None:
        run_name = args.wandb_run_name or f"{name}_seed{seed}"
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "game": game,
                "seed": seed,
                "total_frames": total_frames,
                "delay_frames": delay_frames,
                "num_actions": num_actions,
                "gru_hidden_size": args.gru_hidden,
                **parms,
            },
            reinit=True,
        )
        print(f"Wandb run: {run.url}")

    avg = 0
    episode_scores = []
    episode_end = []
    environment_start = 0
    running_episode_score = 0
    environment_start_time = time.time()
    score_ema = None

    episode_graph = torch.zeros(1000, device='cpu')
    parms_graph = torch.zeros(1000, len(list(agent.training_model.parameters())))

    episode_number = 0
    frames_without_reward = 0
    previous_lives = ale.lives()
    delayed_actions = [0] * delay_frames

    taken_action = 0
    average_frames = 100_000

    with tqdm(total=agent.total_frames, desc="Training (GRU)", unit="frame", dynamic_ncols=True) as pbar:
      for u in range(agent.total_frames):
        if save_incremental_models and (u + 1) // 500_000 != last_model_save:
            last_model_save = (u + 1) // 500_000
            filename = f'{data_dir}/{name}_{u + 1}.model'
            print('writing ' + filename)
            agent.save_model(filename)

        if u * episode_graph.shape[0] // agent.total_frames != (u + 1) * episode_graph.shape[0] // agent.total_frames:
            torch.cuda.synchronize()
            i = u * episode_graph.shape[0] // agent.total_frames
            count = 0
            total = 0
            for j in range(len(episode_scores) - 1, -1, -1):
                if episode_end[j] < u - average_frames:
                    break
                count += 1
                total += episode_scores[j]
            if count == 0:
                avg = -999
            else:
                avg = total / count
                for j in range(i - 1, -1, -1):
                    if episode_graph[j] != -999:
                        break
                    episode_graph[j] = avg
            episode_graph[i] = avg

            filename = data_dir + '/' + name + '.score'
            episode_graph.cpu().numpy().tofile(filename)

            for j, p in enumerate(agent.training_model.parameters()):
                parms_graph[i, j] = torch.norm(p.flatten()).item()

        delayed_actions.append(taken_action)

        torch.cuda.nvtx.range_push("act")
        cmd = delayed_actions.pop(0)
        reward = ale.act(int(action_set[cmd]))
        running_episode_score += reward
        torch.cuda.nvtx.range_pop()
        if reward != 0:
            frames_without_reward = 0
        else:
            frames_without_reward += 1

        end_of_episode = 0

        if lives_as_episodes and ale.lives() < previous_lives:
            previous_lives = ale.lives()
            episode_number += 1
            end_of_episode = 1
        if ale.game_over() or frames_without_reward == max_frames_without_reward:
            torch.cuda.synchronize()
            if frames_without_reward == max_frames_without_reward:
                print(f'terminated at {frames_without_reward} frames without reward')
            episode_number = ((episode_number // 100) + 1) * 100
            end_of_episode = 1
            torch.cuda.nvtx.range_push("reset")
            ale.reset_game()
            previous_lives = ale.lives()
            frames_without_reward = 0

            frames = u - environment_start
            episode_end.append(u)
            environment_start = u
            episode_scores.append(running_episode_score)
            episode_score = running_episode_score
            running_episode_score = 0

            if score_ema is None:
                score_ema = float(episode_score)
            else:
                score_ema = 0.95 * score_ema + 0.05 * float(episode_score)

            now = time.time()
            frames_per_second = frames / (now - environment_start_time)
            environment_start_time = now

            print(
                f'{rank}:{name} frame:{u:7} {frames_per_second:4.0f}/s eps {len(episode_scores) - 1:3},{frames:5}={int(episode_score):5} '
                f'ema {score_ema:.1f} err {agent.avg_error_ema:.1f} {agent.max_error_ema:.1f} loss {agent.train_loss_ema:.1f} '
                f'targ {agent.target_ema:.1f} rnd {agent.intrinsic_reward_ema:.3f} avg {avg:4.1f}'
            )

            if run is not None:
                run.log({
                    "score": episode_score,
                    "score_ema": score_ema,
                    "episode_length": frames,
                    "episode_reward_avg": avg if avg != -999 else None,
                    "sps": frames_per_second,
                    "train/loss": agent.train_loss_ema.item(),
                    "train/avg_error": agent.avg_error_ema.item(),
                    "train/max_error": agent.max_error_ema.item(),
                    "train/target": agent.target_ema.item(),
                    "rnd/intrinsic_reward": agent.intrinsic_reward_ema.item(),
                    "rnd/loss": agent.rnd_loss_ema.item(),
                }, step=u)

            torch.cuda.nvtx.range_pop()
            pbar.set_postfix(sps=int(frames_per_second), score=int(episode_score), ema=f"{score_ema:.0f}")

        taken_action = agent.frame(ale.getScreenRGB(), reward, end_of_episode)
        pbar.update(1)

    filename = data_dir + '/' + name + '.policy_actions'
    print('writing ' + filename)
    agent.policy_actions_buffer.cpu().numpy().tofile(filename)

    filename = data_dir + '/' + name + '.score'
    print('writing ' + filename)
    episode_graph.cpu().numpy().tofile(filename)

    filename = data_dir + '/' + name + '.parms'
    print('writing ' + filename)
    parms_graph.cpu().numpy().tofile(filename)

    plots = torch.zeros(len(episode_scores), 2)
    for i in range(len(episode_scores)):
        plots[i][0] = episode_end[i]
        plots[i][1] = episode_scores[i]
    filename = data_dir + '/' + name + '.scatter'
    print('writing ' + filename)
    plots.cpu().numpy().tofile(filename)

    filename = data_dir + '/' + name + '.loss'
    print('writing ' + filename)
    torch.tensor(agent.train_losses).cpu().numpy().tofile(filename)

    if save_model:
        filename = f'{data_dir}/{name}.model'
        print('writing ' + filename)
        agent.save_model(filename)

    if run is not None:
        run.finish()

    print('done')


if __name__ == '__main__':
    main()

