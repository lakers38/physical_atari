# Copyright 2025 Keen Technologies, Inc.
# Modified to add RND (Random Network Distillation) intrinsic motivation
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

# agent_delay_target_rnd.py
#
# agent_delay_target + RND (Random Network Distillation) for intrinsic motivation
# RND provides curiosity-driven exploration bonuses for novel states

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ale_py import Action, ALEInterface, LoggerMode, roms
from pynvml import *

from vector_agents import VectorAgent

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


def train_function_rnd(
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
    # variable inputs
    new_observations,
    tensor_u,
    observation_ring,
    # policy output
    selected_action_index,
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
        # In a cuda graph, you can't just do the obvious tensor[index] = value
        # because pytorch will do index.item(), which is not allowed in a graph.
        observation_ring.view(
            ring_buffer_size // frame_skip, frame_skip, obs_channels, obs_height, obs_width
        ).index_put_((ring,), atari_resampled)

        # see which samples from the replay buffer are being evaluated
        train_step = tensor_u // train_steps
        index_indexes = torch.arange(train_batch) + train_step * train_batch
        buffer_indexes = train_indexes[index_indexes]

        # stack sets of input_stack frames together to make each observation
        final_stack_indexes = buffer_indexes.unsqueeze(dim=1).expand(train_batch, input_stack)
        offsets = torch.arange(-input_stack + 1, 1).unsqueeze(dim=0).expand(train_batch, input_stack)
        ring_indexes = offsets + final_stack_indexes

        # The policy will be evaluated based on the first bootstrap target; make sure it is the most recent frame!
        # assert( ring_indexes[0,-1] == tensor_u )

        ring_indexes = ring_indexes % ring_buffer_size
        observation_stacks = observation_ring[ring_indexes]
        observation_stacks = observation_stacks.view(train_batch, input_channels, obs_height, obs_width)

        # Using the target network, calculate state values for the bootstrap position
        observation_stacks = observation_stacks.to(next(training_model.parameters()).dtype) / 255.0

    # evaluate the model with gradients
    train_values = training_model(observation_stacks)
    num_model_distributions = train_values.shape[1]

    with torch.no_grad():
        # build target values for training
        all_q = train_values[:, :-1].detach()
        probs = F.softmax(all_q / (avg_error_ema * 2**temperature_log2), dim=1)
        all_v = (all_q * probs).sum(dim=1)

        state_value_buffer[buffer_indexes] = all_v

        # set selected_action_index IMMEDIATELY after model forward (before RND)
        # softmax-greedy policy
        sample = torch.multinomial(probs[0], num_samples=1)
        selected_action_index.copy_(sample[0])

        # The main code can now return the action while training goes on in the background

    # ===== RND: Compute intrinsic rewards (AFTER action selection) =====
    # Get target features (frozen, no grad needed)
    with torch.no_grad():
        rnd_target_features = rnd_target(observation_stacks)
    
    # Get predictor features WITH gradients (will be used for training later)
    rnd_predictor_features = rnd_predictor(observation_stacks.detach())
    
    with torch.no_grad():
        # Intrinsic reward = prediction error (MSE per sample)
        intrinsic_rewards = (rnd_target_features - rnd_predictor_features.detach()).pow(2).mean(dim=1)
        
        # Normalize intrinsic rewards by running mean (helps stability)
        current_intrinsic_mean = intrinsic_rewards.mean()
        torch.lerp(intrinsic_reward_ema, current_intrinsic_mean, 2**ema_log2, out=intrinsic_reward_ema)
        
        # Normalize by EMA (avoid division by zero)
        normalized_intrinsic = intrinsic_rewards / (intrinsic_reward_ema + 1e-8)

        # the next frame_skip frames will use this selected_action_index
        num_distributions = distribution_factor_buffer.shape[1]
        dist = F.one_hot(selected_action_index, num_classes=num_distributions).float()
        online_indexes = (torch.arange(frame_skip) + (tensor_u + 1)).clamp(max=distribution_factor_buffer.shape[0] - 1)
        distribution_factor_buffer[online_indexes] = dist.unsqueeze(dim=0).expand(frame_skip, num_distributions)
        policy_actions_buffer[online_indexes] = selected_action_index

        # get the observed rewards up to the max bootstrap point
        reward_indexes = buffer_indexes.unsqueeze(dim=1) + torch.arange(multisteps_max).unsqueeze(dim=0)
        reward_indexes %= reward_buffer.shape[
            0
        ]  # necessary because a short multistep near the end of training would cause fetching multisteps_max to overrun
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

        # ===== RND: Add intrinsic rewards to return targets =====
        # The intrinsic reward is for the sampled observations, add it to their targets
        # normalized_intrinsic is (train_batch,), unsqueeze to (train_batch, 1) to match blended_rewards shape
        return_targets = blended_rewards + blended_states + intrinsic_reward_scale * normalized_intrinsic.unsqueeze(1)

        # collect statistics on average targets for the IID samples
        torch.lerp(target_ema, return_targets[online_batch:].mean(), 2**ema_log2, out=target_ema)

        # Actions that weren't taken at all will have the loss scaled to 0
        distribution_factors = distribution_factor_buffer[buffer_indexes]
        # the online samples aren't trained
        distribution_factors[:online_batch].zero_()

        # assert(distribution_factors.min() >= 0.0)
        # all QV will use the same target, so let it broadcast
        training_targets = return_targets.view(train_batch, 1)
        # pytorch gives a warning if we just let this broadcast
        training_targets = training_targets.expand(train_batch, num_model_distributions)

    loss_individual = F.mse_loss(train_values, training_targets, reduction='none')
    distribution_factors[online_batch : online_batch * 2] *= online_loss_scale
    loss_individual = loss_individual * distribution_factors

    # We don't want to actually modify the model or stats until there is a reasonable number of samples in the buffer
    loss_individual = loss_individual * (tensor_u > min_train_frames)

    loss_buffer[buffer_indexes] = loss_individual.sum(dim=1)

    loss = loss_individual.sum() / (train_batch - online_batch)

    linear_optimizer.zero_grad()
    optimizer.zero_grad()
    loss.backward()

    # avg_error and value ignores the online part of the batch, max_error looks at everything
    avg_error = loss_individual.detach()[online_batch:].sqrt().sum() / (train_batch - online_batch)
    max_error = loss_individual.detach().sqrt().max()

    torch.lerp(train_loss_ema, loss.detach(), 2**ema_log2, out=train_loss_ema)
    torch.lerp(avg_error_ema, avg_error.detach(), 2**ema_log2, out=avg_error_ema)
    torch.lerp(max_error_ema, max_error.detach(), 2**ema_log2, out=max_error_ema)

    # if we are doing weight anchoring to the init model, blend that in now, before the step, just as conventional weight decay would
    training = list(training_model.parameters())
    if weight_decay != 0.0:
        with torch.no_grad():
            anchor_network_alpha = lr_tensor * abs(weight_decay)
            init = list(anchor_model.parameters())
            for p in range(len(training)):
                if init[p].dim() > 1:  # don't decay biases
                    torch.lerp(training[p], init[p], anchor_network_alpha, out=training[p])

    optimizer.step()
    linear_optimizer.step()

    if use_weight_norm:
        # norm the CNN weights, but not the final linear layer
        with torch.no_grad():
            plist = list(training_model.parameters())
            for i in range(len(plist) - 2):
                p = plist[i]
                if p.dim() > 1:  # don't change biases
                    norms = torch.norm(p.flatten(start_dim=1), dim=1)
                    p /= norms.view(-1, 1, 1, 1)

    train_loss.copy_(loss.detach())

    # ===== RND: Train the predictor network =====
    # Reuse predictor features computed earlier (already has gradients)
    # RND loss: predict target features
    rnd_loss = F.mse_loss(rnd_predictor_features, rnd_target_features.detach())
    
    # Scale by update proportion and zero out before min_train_frames (CUDA graph compatible)
    rnd_loss = rnd_loss * rnd_update_proportion * (tensor_u > min_train_frames)
    
    rnd_optimizer.zero_grad()
    rnd_loss.backward()
    rnd_optimizer.step()
    
    # Track RND loss
    torch.lerp(rnd_loss_ema, rnd_loss.detach(), 2**ema_log2, out=rnd_loss_ema)


class OddPooled(nn.Module):
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


class Pooled(nn.Module):
    def __init__(
        self, input_shape, base_channels, output_channels, kernel_size=3, pool_size=3, dirac=1, use_biases=0, norm=0
    ):
        super().__init__()
        self.cnn = nn.ModuleList()
        self.pool_size = pool_size
        self.norm = norm
        out_channels = base_channels
        in_channels = input_shape[1]
        img_height = input_shape[2]
        img_width = input_shape[3]
        while img_width > 3 or img_height > 3:
            c = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', bias=use_biases)
            self.cnn.append(c)
            if dirac:
                w = c.weight.data
                torch.nn.init.dirac_(w[: w.size(1)])
            in_channels = out_channels
            out_channels *= 2

            # with full size, non-power-of-two sizes, reducing the dimensions quicker is a benefit
            pad = (pool_size - 1) // 2
            img_width = (img_width + pad * 2 - (pool_size - 1) - 1) // 2 + 1
            img_height = (img_height + pad * 2 - (pool_size - 1) - 1) // 2 + 1
        in_channels = in_channels * img_width * img_height
        self.final = nn.Linear(in_channels, output_channels, bias=use_biases)

    def forward(self, x):
        for c in self.cnn:
            x = c(x)
            x = F.max_pool2d(x, kernel_size=self.pool_size, stride=2, padding=1)
            x = F.relu(x)
        conv_out = x.flatten(start_dim=1)
        return self.final(conv_out)


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
            # self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self.cuda_graph.replay()
            return
        if self.graph_warmups == -1:
            self.func(*self.args)
            return
        # Warmup, build, or use the CUDA graph for a training step

        if self.graph_warmups > 0:
            # Warmup before graph capture to make sure all memory is allocated and known.
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self.func(*self.args)
            torch.cuda.current_stream().wait_stream(self.stream)
        else:
            # capture the graph -- doesn't actually execute it
            print('capture start')
            torch.cuda.synchronize()  # EVERYTHING must be synchronized before graph capture
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graph, stream=self.stream):
                self.func(*self.args)
            print('capture stop')
            # use the graph capture
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


class Agent:
    def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
        # defaults that might be overridden by explicit experiment runs
        self.gpu = 0

        # Value / reward
        self.target_network_alpha_log2 = (
            -7
        )  # a fraction of the training network is blended into the target each training step
        self.ema_log2 = -10
        self.reward_discount = 0.9975  # discount per 60 fps frame
        self.multisteps_max = 64  # inclusive
        self.td_lambda = 0.95
        self.death_punishment = 0

        # The observation
        self.frame_skip = 4  # the number of observations, rewards, and end_of_episodes processed each call
        self.input_width = 160  # image dimensions provided to the agent
        self.input_height = 210
        self.input_stack = 16  # number of previous 60 fps frames to stack for the input
        self.obs_width = 128  # image dimensions provided to the model
        self.obs_height = 128
        self.obs_channels = 3  # 1 for grey, 3 for RGB

        # exploration
        self.greedy_max = 0.99
        self.greedy_ramp = 100_000
        self.temperature_log2 = -7

        # The model
        self.load_file = None
        self.seed = seed
        self.num_actions = num_actions  # many games can use a reduced action set for faster learning
        self.use_model = 3
        self.kernel_size = 3
        self.base_width = 80
        self.use_biases = 0
        self.use_dirac = 1  # CNN weight initialization

        self.use_precision = 0  # 1 = bfloat16 (doesn't work well)

        # training
        self.use_softv = 1  # v from softmax q
        self.use_weight_norm = 1
        self.repeat_train = 1  # repeat the training multiple times with the same target
        self.min_train_frames = 256  # minimum is input_stack + multisteps, but waiting a little longer may avoid overtraining the first few frames

        self.base_lr_log2 = -16
        self.lr_log2 = -18

        self.train_batch = (
            32  # One will be the most current data, the rest will be randomly sampled from the ring buffer
        )
        self.online_batch = 4  # samples in train_batch that will be forced to most recent, must be >= 1 for policy
        self.online_loss_scale = 2
        self.train_steps = 4  # run training after this many 60 fps frames

        self.ring_buffer_size = 200 * 1024

        self.weight_decay = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.momentum = 0.9

        self.total_frames = total_frames

        # should be strictly a performance optimization, with no behavior change
        self.use_cuda_graphs = True  # faster with graphs, but you can't debug them

        # ===== RND hyperparameters =====
        self.intrinsic_reward_scale = 0.1  # Scale factor for intrinsic rewards
        self.rnd_feature_dim = 512  # Output dimension of RND networks
        self.rnd_lr = 1e-4  # Learning rate for RND predictor
        self.rnd_update_proportion = 0.25  # Proportion of batches to train RND on (for stability)

        # dynamically override configuration
        for key, value in kwargs.items():
            assert hasattr(self, key), f"Unknown parameter: {key}"
            setattr(self, key, value)

        self.dev = f'cuda:{self.gpu}'

        # force the ring buffer to be an exact multiple of frame_skip
        self.ring_buffer_size -= self.ring_buffer_size % self.frame_skip

        # set CuBLAS environment variable so matmuls can be deterministic
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # helps debugging cuda issues
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # Make more deterministic
        # must be combined with os.environ['CUBLAS_WORKSPACE_CONFIG']= ':4096:8' before loading torch
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        print('torch version: ', torch.__version__)
        print('cuda version : ', torch.version.cuda)
        print('dev          : ', self.dev)

        # if this isn't done, pytorch seems to use 3-9 cores worth of time per process
        # just to busy wait, such that trying to run 8 processes was getting CPU bound.
        torch.set_num_threads(1)

        # Which GPU everything will be done on.
        torch.set_default_device(self.dev)

        # This is necessary for graph compilation to work on devices other than 0
        torch.cuda.set_device(self.dev)

        # let pytorch print wider tensor dumps
        torch.set_printoptions(linewidth=160)

        self.input_channels = self.input_stack * self.obs_channels

        # the environment returns a current observation and episode number, and the reward from the just-executed action,
        # which may have been the end of a different episode.
        self.observation_ring = torch.zeros(
            self.ring_buffer_size, self.obs_channels, self.obs_height, self.obs_width, dtype=torch.uint8
        )

        # The numpy arrays will be staged here so the cuda graph can process it.
        self.new_observations = torch.zeros(self.frame_skip, self.input_height, self.input_width, self.obs_channels)

        self.resample_width = build_downsample(self.input_width, self.obs_width).to(self.dev)
        self.resample_height = build_downsample(self.input_height, self.obs_height).to(self.dev)

        # when training, observation_ring[tensor_u%ring_buffer_size] and episode_buffer[tensor_u] are valid
        self.train_loss = torch.tensor(0.0)
        self.train_loss_ema = torch.tensor(0.0)

        # this gets divided as a temperature, so don't start at 0, and starting higher
        # forces more random exploration at the start
        self.avg_error_ema = torch.tensor(10.0)
        self.max_error_ema = torch.tensor(0.0)

        # the model values are what comes directly out of the model final layer
        self.target_ema = torch.tensor(1.0)

        # ===== RND statistics =====
        self.intrinsic_reward_ema = torch.tensor(1.0)  # Running mean of intrinsic rewards
        self.rnd_loss_ema = torch.tensor(0.0)  # Running mean of RND loss

        # start at 3 so the previous four steps can be referenced without going negative
        self.u = self.frame_skip - 1
        self.tensor_u = torch.tensor(3, dtype=torch.int64)

        # buffer frame_skip frames for train
        self.frame_count = 0
        self.observation_rgb8 = np.zeros(
            (self.frame_skip, self.input_height, self.input_width, self.obs_channels), dtype=np.uint8
        )
        self.rewards = np.zeros(self.frame_skip)
        self.end_of_episodes = np.zeros(self.frame_skip)

        self.lr_tensor = torch.tensor(2**self.lr_log2)  # may be modified by ramp
        self.base_lr_tensor = torch.tensor(2**self.base_lr_log2)  # may be modified by ramp
        self.lr_warmup_start = torch.tensor(0, dtype=torch.int64)

        # set by policy for environment
        self.selected_action_index = torch.tensor(0, dtype=torch.int64)

        self.punishment_tensor = torch.tensor(self.death_punishment)

        fmt = torch.bfloat16 if self.use_precision == 1 else torch.float32

        self.episode_buffer = torch.full((self.total_frames,), -999, dtype=torch.int64)
        self.reward_buffer = torch.full((self.total_frames,), -999.0)
        self.state_value_buffer = torch.full((self.total_frames,), 0.0)

        # Every frame is assigned the most recent selected_action_index.
        # With the standard frame_skip 4, each selected_action_index will be repeated 4 times.
        self.policy_actions_buffer = torch.full((self.total_frames,), -999, dtype=torch.int64)

        self.loss_buffer = torch.full((self.total_frames,), -999.0)

        # Random seed for policy action selection and training index selection
        torch.random.manual_seed(self.seed)

        # train_indexes[] will be the training location, which must have multisteps_max valid after it.
        # The first 4 entries in the buffer are not valid, so make sure they are never referenced.
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

        # epsilon-greedy random action exploration, -1 = take best from policy, otherwise use this random index
        self.rand_action_indexes = torch.randint(self.num_actions, (self.total_frames,))
        self.take_policy_action = torch.rand(self.total_frames) < (
            torch.arange(self.total_frames) / self.greedy_ramp
        ).clamp(max=self.greedy_max)
        self.rand_action_indexes = torch.where(self.take_policy_action, torch.full((1,), -1), self.rand_action_indexes)

        self.num_model_distributions = self.num_actions + 1
        total_model_outputs = self.num_model_distributions

        self.distribution_factor_buffer = torch.full((self.total_frames, total_model_outputs), -999.0)

        self.episode_number = 0
        self.train_losses = []

        torch.random.manual_seed(self.seed)

        if self.use_model >= 1:
            self.training_model = OddPooled(
                (1, self.input_channels, self.obs_height, self.obs_width),
                self.base_width,
                total_model_outputs,
                use_biases=self.use_biases,
                dirac=self.use_dirac,
                kernel_size=self.kernel_size,
                weighting=(self.use_model - 1),
            )
        else:
            self.training_model = Pooled(
                (1, self.input_channels, self.obs_height, self.obs_width),
                self.base_width,
                total_model_outputs,
                use_biases=self.use_biases,
                dirac=self.use_dirac,
                kernel_size=self.kernel_size,
            )
        print(self.training_model)
        print('parameters: ', model_parameter_count(self.training_model))

        if self.load_file is not None:
            self.training_model.load_state_dict(torch.load(self.load_file, weights_only=True))

        self.training_model.to(dtype=fmt)
        self.training_model.train()

        self.train_values = torch.zeros(self.train_batch, total_model_outputs)

        # td-lambda combination of rewards and values
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

        # negative wd values do weight anchoring to the init values instead of weight decay to 0
        self.anchor_model = copy.deepcopy(self.training_model)
        if self.weight_decay > 0:
            with torch.no_grad():
                for p in self.anchor_model.parameters():
                    p.zero_()
        adamwd = 0

        parms = list(self.training_model.parameters())
        if self.use_biases:
            final_parms = parms[-2:]
            initial_parms = parms[:-2]
        else:
            final_parms = parms[-1:]
            initial_parms = parms[:-1]

        self.optimizer = torch.optim.AdamW(
            initial_parms,
            lr=self.base_lr_tensor,
            fused=True,
            capturable=True,
            weight_decay=adamwd,
            betas=(self.beta1, self.beta2),
        )
        self.linear_optimizer = torch.optim.SGD(final_parms, lr=2**self.lr_log2, momentum=self.momentum)

        # ===== RND Networks =====
        print("Initializing RND networks...")
        
        # Target network: random, frozen
        self.rnd_target = RNDNetwork(
            self.input_channels, self.rnd_feature_dim,
            input_height=self.obs_height, input_width=self.obs_width
        )
        self.rnd_target.to(dtype=fmt)  # Match training model dtype
        self.rnd_target.eval()
        for param in self.rnd_target.parameters():
            param.requires_grad = False
        
        # Predictor network: trained to match target
        self.rnd_predictor = RNDNetwork(
            self.input_channels, self.rnd_feature_dim,
            input_height=self.obs_height, input_width=self.obs_width
        )
        self.rnd_predictor.to(dtype=fmt)  # Match training model dtype
        self.rnd_predictor.train()
        
        print(f'RND target parameters: {model_parameter_count(self.rnd_target)}')
        print(f'RND predictor parameters: {model_parameter_count(self.rnd_predictor)}')
        
        # RND optimizer (capturable for CUDA graphs)
        self.rnd_optimizer = torch.optim.AdamW(
            self.rnd_predictor.parameters(),
            lr=self.rnd_lr,
            capturable=True,
        )

        # Convert RND constants to tensors for CUDA graph
        self.intrinsic_reward_scale_tensor = torch.tensor(self.intrinsic_reward_scale)
        self.rnd_update_proportion_tensor = torch.tensor(self.rnd_update_proportion)

        self.spin_stream = torch.cuda.Stream(priority=0)

        self.train_stream = torch.cuda.Stream(priority=0)
        
        self.train_graph = cuda_graph_wrapper(
            train_function_rnd,
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
                # variable state
                self.new_observations,
                self.tensor_u,
                self.observation_ring,
                # policy output
                self.selected_action_index,
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

    # --------------------------------
    # Returns the selected action index
    # --------------------------------
    def frame(self, observation_rgb8, reward, end_of_episode):  # [height,width,3]
        assert observation_rgb8.shape == (self.input_height, self.input_width, self.obs_channels)

        i = self.frame_count % self.frame_skip
        self.observation_rgb8[i] = observation_rgb8
        self.rewards[i] = reward
        self.end_of_episodes[i] = end_of_episode
        self.frame_count += 1

        if i != (self.frame_skip - 1):
            return self.selected_action_index.item()

        if self.u > self.total_frames - self.frame_skip:
            # don't overflow any of the buffers
            return 0

        with torch.cuda.stream(self.spin_stream):
            self.new_observations.copy_(torch.from_numpy(self.observation_rgb8))

            self.reward_buffer[self.u : self.u + self.frame_skip] = torch.from_numpy(self.rewards)

            for i in range(self.frame_skip):
                self.episode_number += int(self.end_of_episodes[i] > 0)
                self.episode_buffer[self.u + 1 + i] = self.episode_number

            self.u += self.frame_skip

            # we will wait for the training graph to update this
            self.selected_action_index.fill_(-1)

        # run the policy every four frames
        # sets selected_action_index, which will be returned to the environment,
        # and various things for the training to use
        torch.cuda.nvtx.range_push("train")
        # make sure the new data transfers have completed
        #        torch.cuda.synchronize()

        # results from last training run
        with torch.cuda.stream(self.spin_stream):
            self.train_losses.append(self.train_loss_ema.item())
            self.train_losses.append(self.avg_error_ema.item())
            self.train_losses.append(self.max_error_ema.item())
            self.train_losses.append(self.target_ema.item())
            # Add RND stats
            self.train_losses.append(self.intrinsic_reward_ema.item())
            self.train_losses.append(self.rnd_loss_ema.item())

        self.train_graph()
        torch.cuda.nvtx.range_pop()

        # wait for the policy to write the selected action and get it
        # Can't block on an event from inside a CUDA graph, so busy waiting it is...
        torch.cuda.nvtx.range_push("spin")
        with torch.cuda.stream(self.spin_stream):
            spins = 0
            while self.selected_action_index.item() == -1:
                time.sleep(0.0001)
                spins += 1
            # Debug: print spin count every 1000 frames to check timing
            if self.u % 4000 == 0:
                print(f"[RND] frame={self.u}, spins={spins}, spin_time={spins * 0.1:.1f}ms")
        torch.cuda.nvtx.range_pop()
        # the rest of training will continue in the background

        return self.selected_action_index.item()

    def save_model(self, filename):
        torch.save({
            'training_model': self.training_model.state_dict(),
            'rnd_predictor': self.rnd_predictor.state_dict(),
            'rnd_target': self.rnd_target.state_dict(),
        }, filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename, weights_only=True)
        self.training_model.load_state_dict(checkpoint['training_model'])
        if 'rnd_predictor' in checkpoint:
            self.rnd_predictor.load_state_dict(checkpoint['rnd_predictor'])
        if 'rnd_target' in checkpoint:
            self.rnd_target.load_state_dict(checkpoint['rnd_target'])


# --------------------------------
# Vectorized Core and Agent for multi-environment training
# --------------------------------


def preprocess_batch(obs_batch: np.ndarray, height: int = 84, width: int = 84) -> np.ndarray:
    """Preprocess a batch of observations to grayscale and resize."""
    num_envs = obs_batch.shape[0]
    # Fast path: already grayscale and correctly sized
    if obs_batch.ndim == 3 and obs_batch.shape[1] == height and obs_batch.shape[2] == width:
        return obs_batch.astype(np.uint8, copy=False)

    processed = np.zeros((num_envs, height, width), dtype=np.uint8)
    for i in range(num_envs):
        frame = obs_batch[i]
        if frame.ndim == 3 and frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif frame.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected observation shape: {frame.shape}")

        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        processed[i] = frame
    return processed


class RingBuffer:
    """
    Ring buffer matching original agent_delay_target structure.
    Stores observations, rewards, episodes, actions, and computed state values.
    """
    def __init__(self, capacity: int, obs_channels: int, obs_height: int, obs_width: int, num_actions: int):
        self.capacity = capacity
        self.obs_channels = obs_channels
        self.obs_height = obs_height
        self.obs_width = obs_width
        
        # Observation ring buffer (like original)
        self.observations = np.zeros((capacity, obs_channels, obs_height, obs_width), dtype=np.uint8)
        # Per-frame buffers
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.episodes = np.zeros(capacity, dtype=np.int64)
        self.state_values = np.zeros(capacity, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        # Distribution factors: one-hot of action taken (num_actions + 1 for state value output)
        self.distribution_factors = np.zeros((capacity, num_actions + 1), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, obs: np.ndarray, reward: float, episode_id: int, action: int, num_distributions: int) -> int:
        """Add a frame and return its index."""
        idx = self.ptr
        self.observations[idx] = obs
        self.rewards[idx] = reward
        self.episodes[idx] = episode_id
        self.actions[idx] = action
        # One-hot encode the action
        self.distribution_factors[idx] = 0.0
        self.distribution_factors[idx, action] = 1.0
        self.state_values[idx] = 0.0
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx


class DelayTargetRNDCore:
    """
    Vectorized core for Delay Target + RND agent.
    Exact match to original algorithm, just vectorized with RND added.
    """
    def __init__(
        self,
        *,
        num_envs: int,
        seed: int,
        num_actions: int,
        total_frames: int,
        # Observation params
        stack_size: int = 4,
        obs_height: int = 84,
        obs_width: int = 84,
        # Buffer/training params
        buffer_size: int = 200_000,
        batch_size: int = 32,
        train_start: int = 1000,
        train_freq: int = 4,
        # Original algorithm params
        gamma: float = 0.9975,  # reward_discount in original
        td_lambda: float = 0.95,
        multisteps_max: int = 64,
        temperature_log2: float = -7,
        ema_log2: float = -10,  # ~0.001
        base_lr_log2: float = -16,
        lr_log2: float = -18,
        # Network params
        base_width: int = 80,
        use_biases: int = 0,
        use_dirac: int = 1,
        kernel_size: int = 3,
        use_model: int = 2,
        # RND params
        intrinsic_reward_scale: float = 0.1,
        rnd_feature_dim: int = 512,
        rnd_lr: float = 1e-4,
        rnd_update_proportion: float = 0.25,
        # Misc
        data_dir: Optional[str] = None,
        load_file: Optional[str] = None,
        gpu: int = 0,
    ):
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.total_frames = total_frames
        self.stack_size = stack_size
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.train_start = train_start
        self.train_freq = train_freq
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.multisteps_max = multisteps_max
        self.temperature_log2 = temperature_log2
        self.ema_log2 = ema_log2
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.rnd_update_proportion = rnd_update_proportion

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Device
        if torch.cuda.is_available() and gpu >= 0:
            self.device = torch.device(f'cuda:{gpu}')
            torch.cuda.manual_seed_all(seed)
        else:
            self.device = torch.device('cpu')
        print(f"DelayTargetRNDCore: device={self.device}")

        self.num_distributions = num_actions + 1

        # Ring buffer (matching original structure)
        self.ring = RingBuffer(buffer_size, 1, obs_height, obs_width, num_actions)

        # Per-frame state stacks for each env
        self.state_stacks = np.zeros((num_envs, stack_size, obs_height, obs_width), dtype=np.uint8)
        self.last_actions = np.full(num_envs, 0, dtype=np.int64)
        self.episode_ids = np.zeros(num_envs, dtype=np.int64)

        # Precompute TD-lambda weights (matching original)
        reward_discounts = np.zeros(multisteps_max, dtype=np.float32)
        value_discounts = np.zeros(multisteps_max, dtype=np.float32)
        total = 0.0
        for step in range(1, multisteps_max):
            factor = td_lambda ** (step - 1)
            total += factor
            value_discounts[step] = factor * (gamma ** step)
            for n in range(step):
                reward_discounts[n] += factor * (gamma ** n)
        self.reward_discounts = torch.from_numpy(reward_discounts / total).to(self.device).unsqueeze(1)
        self.value_discounts = torch.from_numpy(value_discounts / total).to(self.device).unsqueeze(1)

        # Network
        if use_model >= 1:
            self.network = OddPooled(
                (1, stack_size, obs_height, obs_width),
                base_width, self.num_distributions,
                use_biases=use_biases, dirac=use_dirac,
                kernel_size=kernel_size, weighting=(use_model - 1),
            ).to(self.device)
        else:
            self.network = Pooled(
                (1, stack_size, obs_height, obs_width),
                base_width, self.num_distributions,
                use_biases=use_biases, dirac=use_dirac, kernel_size=kernel_size,
            ).to(self.device)
        print(f"Network params: {model_parameter_count(self.network)}")

        # Optimizers (matching original: AdamW for conv, SGD for linear)
        parms = list(self.network.parameters())
        final_parms = parms[-1:] if not use_biases else parms[-2:]
        initial_parms = parms[:-1] if not use_biases else parms[:-2]
        self.optimizer = torch.optim.AdamW(initial_parms, lr=2**base_lr_log2)
        self.linear_optimizer = torch.optim.SGD(final_parms, lr=2**lr_log2, momentum=0.9)

        # RND networks
        self.rnd_target = RNDNetwork(stack_size, rnd_feature_dim).to(self.device)
        self.rnd_target.eval()
        for p in self.rnd_target.parameters():
            p.requires_grad = False
        self.rnd_predictor = RNDNetwork(stack_size, rnd_feature_dim).to(self.device)
        # Init RND
        with torch.no_grad():
            dummy = torch.zeros(1, stack_size, obs_height, obs_width, device=self.device)
            self.rnd_target(dummy)
            self.rnd_predictor(dummy)
        self.rnd_optimizer = torch.optim.Adam(self.rnd_predictor.parameters(), lr=rnd_lr)
        print(f"RND params: target={model_parameter_count(self.rnd_target)}, predictor={model_parameter_count(self.rnd_predictor)}")

        # Stats (matching original)
        self.frame_count = 0
        self.avg_error_ema = 10.0
        self.max_error_ema = 0.0
        self.loss_ema = 0.0
        self.target_ema = 1.0
        self.intrinsic_reward_ema = 1.0
        self.rnd_loss_ema = 0.0
        self.last_loss = 0.0
        self.last_avg_q = 0.0
        self.last_max_q = 0.0

        self.data_dir = data_dir or os.getcwd()
        if load_file and os.path.exists(load_file):
            self.load_model(load_file)

    def reset(self, observations: np.ndarray) -> None:
        processed = preprocess_batch(observations, self.obs_height, self.obs_width)
        for env in range(self.num_envs):
            self.state_stacks[env] = np.repeat(processed[env][None, ...], self.stack_size, axis=0)

    def act(self, observations: np.ndarray) -> np.ndarray:
        processed = preprocess_batch(observations, self.obs_height, self.obs_width)
        # Roll and add new frame
        self.state_stacks = np.roll(self.state_stacks, -1, axis=1)
        self.state_stacks[:, -1] = processed

        stacked = torch.from_numpy(self.state_stacks).to(self.device, dtype=torch.float32) / 255.0
        with torch.no_grad():
            out = self.network(stacked)
            q = out[:, :-1]
            # Softmax policy (matching original)
            temp = self.avg_error_ema * (2 ** self.temperature_log2)
            probs = F.softmax(q / max(temp, 1e-8), dim=1)
            self.last_avg_q = q.mean().item()
            self.last_max_q = q.max().item()

        actions = torch.multinomial(probs, 1).squeeze(1).cpu().numpy()
        self.last_actions = actions.copy()
        return actions

    def observe(self, next_obs: np.ndarray, rewards: np.ndarray,
                terminations: np.ndarray, truncations: np.ndarray) -> None:
        processed = preprocess_batch(next_obs, self.obs_height, self.obs_width)
        for env in range(self.num_envs):
            # Store frame in ring buffer
            self.ring.add(
                self.state_stacks[env, -1:],  # Just the latest frame
                rewards[env],
                self.episode_ids[env],
                self.last_actions[env],
                self.num_distributions,
            )
            done = terminations[env] or truncations[env]
            if done:
                self.episode_ids[env] += 1
                self.state_stacks[env] = np.repeat(processed[env][None, ...], self.stack_size, axis=0)
        self.frame_count += self.num_envs

    def train_step(self) -> None:
        if self.ring.size < max(self.train_start, self.batch_size + self.multisteps_max + self.stack_size):
            return
        if self.frame_count % self.train_freq != 0:
            return

        # Sample indices (leave room for stack and multistep)
        max_idx = self.ring.size - self.multisteps_max
        min_idx = self.stack_size
        if max_idx <= min_idx:
            return
        indices = np.random.randint(min_idx, max_idx, size=self.batch_size)

        # Build observation stacks from ring buffer
        stack_indices = indices[:, None] + np.arange(-self.stack_size + 1, 1)[None, :]
        stack_indices = stack_indices % self.ring.capacity
        obs_stacks = self.ring.observations[stack_indices].squeeze(2)  # (batch, stack, H, W)
        states = torch.from_numpy(obs_stacks).to(self.device, dtype=torch.float32) / 255.0

        # Forward pass
        outputs = self.network(states)
        q_values = outputs[:, :-1]

        # Compute V = sum(softmax(Q) * Q)
        with torch.no_grad():
            temp = self.avg_error_ema * (2 ** self.temperature_log2)
            probs = F.softmax(q_values.detach() / max(temp, 1e-8), dim=1)
            state_values = (q_values.detach() * probs).sum(dim=1)
            # Store state values back
            self.ring.state_values[indices] = state_values.cpu().numpy()

            # RND intrinsic reward
            rnd_target_feat = self.rnd_target(states)
        rnd_pred_feat = self.rnd_predictor(states)

        with torch.no_grad():
            intrinsic = (rnd_target_feat - rnd_pred_feat.detach()).pow(2).mean(dim=1)
            self.intrinsic_reward_ema += (2 ** self.ema_log2) * (intrinsic.mean().item() - self.intrinsic_reward_ema)
            norm_intrinsic = intrinsic / (self.intrinsic_reward_ema + 1e-8)

            # TD-lambda targets
            reward_idx = indices[:, None] + np.arange(self.multisteps_max)[None, :]
            reward_idx = reward_idx % self.ring.capacity
            obs_rewards = torch.from_numpy(self.ring.rewards[reward_idx]).to(self.device)
            obs_values = torch.from_numpy(self.ring.state_values[reward_idx]).to(self.device)

            # Mask by episode
            init_eps = self.ring.episodes[indices]
            eps = self.ring.episodes[reward_idx]
            mask = torch.from_numpy((eps == init_eps[:, None]).astype(np.float32)).to(self.device)
            obs_rewards = obs_rewards * mask
            obs_values = obs_values * mask

            blended_r = obs_rewards @ self.reward_discounts
            blended_v = obs_values @ self.value_discounts
            targets = blended_r.squeeze(-1) + blended_v.squeeze(-1) + self.intrinsic_reward_scale * norm_intrinsic

            # Action factors
            actions = self.ring.actions[indices]
            factors = torch.zeros(self.batch_size, self.num_distributions, device=self.device)
            factors[np.arange(self.batch_size), actions] = 1.0

            targets_expanded = targets.unsqueeze(1).expand(-1, self.num_distributions)

        # Loss
        loss_ind = F.mse_loss(outputs, targets_expanded, reduction='none') * factors
        loss = loss_ind.sum() / max(factors.sum(), 1.0)

        self.optimizer.zero_grad()
        self.linear_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.linear_optimizer.step()

        # RND loss
        rnd_loss = F.mse_loss(rnd_pred_feat, rnd_target_feat.detach()) * self.rnd_update_proportion
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()

        # Stats
        ema = 2 ** self.ema_log2
        avg_err = loss_ind.sqrt().mean().item()
        max_err = loss_ind.sqrt().max().item()
        self.loss_ema += ema * (loss.item() - self.loss_ema)
        self.avg_error_ema += ema * (avg_err - self.avg_error_ema)
        self.max_error_ema += ema * (max_err - self.max_error_ema)
        self.target_ema += ema * (targets.mean().item() - self.target_ema)
        self.rnd_loss_ema += ema * (rnd_loss.item() - self.rnd_loss_ema)
        self.last_loss = loss.item()

    def save_model(self, path: str) -> None:
        torch.save({
            'network': self.network.state_dict(),
            'rnd_predictor': self.rnd_predictor.state_dict(),
            'rnd_target': self.rnd_target.state_dict(),
            'frame_count': self.frame_count,
            'avg_error_ema': self.avg_error_ema,
            'intrinsic_reward_ema': self.intrinsic_reward_ema,
        }, path)

    def load_model(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(ckpt.get('network', ckpt.get('training_model')))
        if 'rnd_predictor' in ckpt:
            self.rnd_predictor.load_state_dict(ckpt['rnd_predictor'])
        if 'rnd_target' in ckpt:
            self.rnd_target.load_state_dict(ckpt['rnd_target'])
        self.frame_count = ckpt.get('frame_count', 0)
        self.avg_error_ema = ckpt.get('avg_error_ema', 10.0)
        self.intrinsic_reward_ema = ckpt.get('intrinsic_reward_ema', 1.0)


class VectorDelayTargetRNDAgent(VectorAgent):
    """
    Vectorized Delay Target + RND agent implementing the VectorAgent protocol.
    Wraps DelayTargetRNDCore for multi-environment training.
    """
    def __init__(
        self,
        *,
        num_envs: int,
        seed: int,
        num_actions: int,
        results_dir: Optional[str] = None,
        total_frames: int = 1_000_000,
        **kwargs,
    ):
        self.core = DelayTargetRNDCore(
            num_envs=num_envs,
            seed=seed,
            num_actions=num_actions,
            total_frames=total_frames,
            data_dir=results_dir,
            **kwargs,
        )
        self.num_envs = num_envs
        self._initialized = False

    def reset(self, num_envs: int) -> None:
        if num_envs != self.num_envs:
            raise ValueError(f"VectorDelayTargetRNDAgent initialized for {self.num_envs} envs; received {num_envs}.")
        self._initialized = False

    def act(self, observations: np.ndarray) -> np.ndarray:
        if not self._initialized:
            self.core.reset(observations)
            self._initialized = True
        return self.core.act(observations)

    def observe(
        self,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
        infos: Iterable[Dict],
    ) -> None:
        self.core.observe(next_observations, rewards, terminations, truncations)

    def train_step(self) -> None:
        self.core.train_step()

    def save_model(self, path: str) -> None:
        self.core.save_model(path)

    def load_model(self, path: str) -> None:
        self.core.load_model(path)


class SingleEnvDelayTargetRNDAgent(VectorAgent):
    """
    Single-environment wrapper using the original CUDA-graph Agent.
    Much faster than vectorized version (~600+ SPS vs ~100 SPS).
    Use with num_envs=1 in sim_latency_vec.py.
    """
    def __init__(
        self,
        *,
        num_envs: int,
        seed: int,
        num_actions: int,
        results_dir: Optional[str] = None,
        total_frames: int = 1_000_000,
        **kwargs,
    ):
        if num_envs != 1:
            raise ValueError(f"SingleEnvDelayTargetRNDAgent only supports num_envs=1, got {num_envs}")
        
        self.agent = Agent(
            data_dir=results_dir or './results',
            seed=seed,
            num_actions=num_actions,
            total_frames=total_frames,
            **kwargs,
        )
        self.num_envs = 1
        self._last_reward = 0.0
        self._last_done = False

    def reset(self, num_envs: int) -> None:
        if num_envs != 1:
            raise ValueError(f"SingleEnvDelayTargetRNDAgent only supports num_envs=1, got {num_envs}")

    def act(self, observations: np.ndarray) -> np.ndarray:
        # observations is (1, H, W) or (1, H, W, C)
        obs = observations[0]
        if obs.ndim == 2:
            # Grayscale, need to expand to RGB for original agent
            obs = np.stack([obs, obs, obs], axis=-1)
        action = self.agent.frame(obs, self._last_reward, 1 if self._last_done else 0)
        return np.array([action], dtype=np.int64)

    def observe(
        self,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
        infos: Iterable[Dict],
    ) -> None:
        self._last_reward = float(rewards[0])
        self._last_done = bool(terminations[0] or truncations[0])

    def train_step(self) -> None:
        # Training happens inside agent.frame() via CUDA graph
        pass

    def save_model(self, path: str) -> None:
        self.agent.save_model(path)

    def load_model(self, path: str) -> None:
        self.agent.load_model(path)


# --------------------------------
# standalone simulator work using the same model interface for physical atari
#
# This file can be run directly to experiment in simulator, or imported by the physical harness.
# --------------------------------
def main():
    parser = argparse.ArgumentParser(description="Delay Target RND Agent")
    parser.add_argument("rank", type=int, nargs="?", default=0, help="GPU rank")
    parser.add_argument("mode", type=str, nargs="?", default="default", help="atari100k, physical, or default")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="delay-target-rnd", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--game", type=str, default=None, help="Override game (e.g. ms_pacman)")
    parser.add_argument("--total_frames", type=int, default=None, help="Override total frames")
    parser.add_argument("--results_dir", type=str, default="./results", help="Results directory")
    parser.add_argument("--load", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    data_dir = args.results_dir
    os.makedirs(data_dir, exist_ok=True)

    save_model = True
    save_incremental_models = True
    last_model_save = -1
    max_frames_without_reward = 18_000

    ale = ALEInterface()
    ale.setLoggerMode(LoggerMode.Error)
    ale.setInt('random_seed', 0)

    lives_as_episodes = 1
    rank = args.rank

    parms = {}
    parms['gpu'] = rank % 8

    frame_skip = 4
    parms['frame_skip'] = frame_skip

    if args.mode == 'atari100k':
        atari100k_list = [
            'assault', 'asterix', 'bank_heist', 'battle_zone', 'boxing', 'breakout',
            'chopper_command', 'crazy_climber', 'demon_attack', 'freeway', 'frostbite',
            'gopher', 'hero', 'jamesbond', 'kangaroo', 'krull', 'kung_fu_master',
            'ms_pacman', 'pong', 'private_eye', 'qbert', 'road_runner', 'seaquest', 'up_n_down',
        ]
        game = atari100k_list[rank % 24]
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

    # Override with command line args
    if args.game:
        game = args.game
    if args.total_frames:
        total_frames = args.total_frames

    # use the ale_py installation path
    rom_path = roms.get_rom_path(game)
    ale.loadROM(rom_path)
    ale.reset_game()

    if reduce_action_set == 0:
        action_set = ale.getLegalActionSet()
    else:
        # optionally apply more restrictions to the action set, since the ALE minimal action set isn't really minimal
        if reduce_action_set == 2 and (game == 'ms_pacman' or game == 'qbert'):
            action_set = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        else:
            action_set = ale.getMinimalActionSet()
    num_actions = len(action_set)
    print(f'{num_actions} actions: {action_set}')

    name = f'delay_rnd_{game}{delay_frames}'
    for k, v in parms.items():
        if k != 'gpu':
            name += '_'
            name += str(v)
    print(name)

    agent = Agent(data_dir, seed, num_actions, total_frames, **parms)

    # Load checkpoint if specified
    if args.load:
        print(f"Loading checkpoint from {args.load}")
        agent.load_model(args.load)

    # Initialize wandb
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
    
    # Score EMA for smoothed tracking (like rainbow agent)
    score_ema = None  # Will be initialized on first episode

    # put the average of 100 episodes in each slot, evenly divided by the total number of learning steps
    episode_graph = torch.zeros(1000, device='cpu')
    parms_graph = torch.zeros(1000, len(list(agent.training_model.parameters())))

    episode_number = 0
    frames_without_reward = 0
    previous_lives = ale.lives()
    delayed_actions = [0] * delay_frames  # allow the commands to be delayed by this many 60 fps frames

    taken_action = 0  # until a policy can be evaluated on observations

    # note that atlantis can learn to play indefinitely, so there may be no completed episodes in the window
    average_frames = 100_000  # frames to average episode scores over for episode_graph

    with tqdm(total=agent.total_frames, desc="Training", unit="frame", dynamic_ncols=True) as pbar:
      for u in range(agent.total_frames):
        if save_incremental_models and (u + 1) // 500_000 != last_model_save:
            last_model_save = (u + 1) // 500_000
            filename = f'{data_dir}/{name}_{u + 1}.model'
            print('writing ' + filename)
            agent.save_model(filename)

        # fill in our average score graph so we get exactly 1000 points on it
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
                # if no episodes were completed in the previous window, backfill with the current value
                for j in range(i - 1, -1, -1):
                    if episode_graph[j] != -999:
                        break
                    episode_graph[j] = avg
            episode_graph[i] = avg

            # write the graph out so it can be viewed incrementally
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

            # Update score EMA (like rainbow agent)
            if score_ema is None:
                score_ema = float(episode_score)
            else:
                score_ema = 0.95 * score_ema + 0.05 * float(episode_score)

            # calculate step speed
            now = time.time()
            frames_per_second = frames / (now - environment_start_time)
            environment_start_time = now

            # Include RND stats in output
            print(
                f'{rank}:{name} frame:{u:7} {frames_per_second:4.0f}/s eps {len(episode_scores) - 1:3},{frames:5}={int(episode_score):5} '
                f'ema {score_ema:.1f} err {agent.avg_error_ema:.1f} {agent.max_error_ema:.1f} loss {agent.train_loss_ema:.1f} '
                f'targ {agent.target_ema:.1f} rnd {agent.intrinsic_reward_ema:.3f} avg {avg:4.1f}'
            )

            # Wandb logging
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
            
            # Update progress bar
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

