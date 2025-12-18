# Copyright 2025 Keen Technologies, Inc.
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

# agent_delay_target.py
#
# Use the last evaluations for target calculation instead of a target model evaluation
import copy
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ale_py import Action, ALEInterface, LoggerMode, roms
from pynvml import *

from framework.Logger import logger


def train_function(
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
    training_enabled,
    reward_discounts,
    value_discounts,
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
    # updated models
    optimizer,
    linear_optimizer,
    training_model,
    anchor_model,
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
    if training_enabled:
        train_values = training_model(observation_stacks)
    else:
        with torch.no_grad():
            train_values = training_model(observation_stacks)
    num_model_distributions = train_values.shape[1]

    with torch.no_grad():
        # build target values for training
        all_q = train_values[:, :-1].detach()
        probs = F.softmax(all_q / (avg_error_ema * 2**temperature_log2), dim=1)
        all_v = (all_q * probs).sum(dim=1)

        state_value_buffer[buffer_indexes] = all_v

        # set selected_action_index
        # softmax-greedy policy
        sample = torch.multinomial(probs[0], num_samples=1)
        selected_action_index.copy_(sample[0])

        # The main code can now return the action while training goes on in the background

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

        return_targets = blended_rewards + blended_states

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

    if not training_enabled:
        return

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
            #            self.stream.wait_stream(torch.cuda.current_stream())
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
            logger.info("delay_target: cuda graph capture start")
            torch.cuda.synchronize()  # EVERYTHING must be synchronized before graph capture
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graph, stream=self.stream):
                self.func(*self.args)
            logger.info("delay_target: cuda graph capture stop")
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
        self.training_enabled = False  # hardcoded eval toggle; set True to train
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

        # dynamically override configuration
        for key, value in kwargs.items():
            assert hasattr(self, key)
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

        logger.info("delay_target: torch version: %s", torch.__version__)
        logger.info("delay_target: cuda version: %s", torch.version.cuda)
        logger.info("delay_target: dev: %s", self.dev)

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

        self.resample_from = (0, 0, 0)
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
        logger.info("delay_target: training_model=%s", self.training_model)
        logger.info("delay_target: parameters=%s", model_parameter_count(self.training_model))

        if self.load_file is not None:
            logger.info("delay_target: Loading checkpoint from: %s", self.load_file)
            checkpoint = torch.load(self.load_file, weights_only=True)
            self.training_model.load_state_dict(checkpoint)
            logger.info("delay_target: Checkpoint loaded successfully (state dict entries=%s)", len(checkpoint))
            # Print a sample of weights to verify they're not random/zero
            first_param_name = list(checkpoint.keys())[0]
            first_param = checkpoint[first_param_name]
            logger.info(
                "delay_target: Sample weights %s mean=%.6f std=%.6f",
                first_param_name,
                first_param.float().mean().item(),
                first_param.float().std().item(),
            )

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

        self.spin_stream = torch.cuda.Stream(priority=0)

        self.train_stream = torch.cuda.Stream(priority=0)
        self.train_graph = cuda_graph_wrapper(
            train_function,
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
                self.training_enabled,
                self.reward_discounts,
                self.value_discounts,
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
                # updated models
                self.optimizer,
                self.linear_optimizer,
                self.training_model,
                self.anchor_model,
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
        #            print(spins)
        torch.cuda.nvtx.range_pop()
        # the rest of training will continue in the background

        return self.selected_action_index.item()

    def save_model(self, filename):
        torch.save(self.training_model.state_dict(), filename)


# --------------------------------
# standalone simulator work using the same model interface for physical atari
#
# This file can be run directly to experiment in simulator, or imported by the physical harness.
# --------------------------------
def main():
    data_dir = './results'
    os.makedirs(data_dir, exist_ok=True)

    save_model = True
    save_incremental_models = True
    last_model_save = -1

    # phoenix in particular has the opportunity to hide in a corner from the boss fight and never collect any rewards, so
    # restarting after a certain number of steps keeps learning going.
    #
    # "Revisiting the ALE" recommends a max episode frames (60fps) of 18_000, which is only five minutes, which would cut short many
    # valid high performing games.
    # "Is Deep Reinforcement Learning Really Superhuman on Atari?" https://arxiv.org/pdf/1908.04683 recommends 18k limit without a reward.
    max_frames_without_reward = 18_000

    ale = ALEInterface()
    ale.setLoggerMode(LoggerMode.Error)
    ale.setInt('random_seed', 0)

    lives_as_episodes = 1

    # if there is a command line parameter, take it as the cuda gpu
    # The rank can be used to try different hyperparameters on each gpu
    if len(sys.argv) >= 2:
        rank = int(sys.argv[1])
    else:
        rank = 0

    parms = {}
    parms['gpu'] = rank % 8

    frame_skip = 4  # number of times an action is repeated in the environment
    parms['frame_skip'] = frame_skip

    if sys.argv[-1] == 'atari100k':
        atari100k_list = [
            'assault',
            'asterix',
            'bank_heist',
            'battle_zone',
            'boxing',
            'breakout',
            'chopper_command',
            'crazy_climber',
            'demon_attack',
            'freeway',
            'frostbite',
            'gopher',
            'hero',
            'jamesbond',
            'kangaroo',
            'krull',
            'kung_fu_master',
            'ms_pacman',
            'pong',
            'private_eye',
            'qbert',
            'road_runner',
            'seaquest',
            'up_n_down',
        ]
        game = atari100k_list[rank % 24]
        seed = rank // 24
        total_frames = 1_000_000  # real atari100k is only 400_000, but training longer is helpful

        # run without sticky actions, which should give slightly better scores, because the models don't need to deal with any randomness
        ale.setFloat('repeat_action_probability', 0.0)
        reduce_action_set = 1
        delay_frames = 0
    elif sys.argv[-1] == 'physical':
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

        reduce_action_set = (
            2  # 0 = always 18, 1 = ALE minimum action set, 2 = restricted even more for ms_pacman and qbert
        )
        delay_frames = 6  # 60 fps frames to delay commands to simulate real world latency
    else:
        reduce_action_set = (
            2  # 0 = always 18, 1 = ALE minimum action set, 2 = restricted even more for ms_pacman and qbert
        )
        total_frames = 2_000_000
        parms['lr_log2'] = -17  # -18 + rank//4
        parms['base_lr_log2'] = -15  # -18 + rank%4
        seed = 0
        game = 'ms_pacman'
        #        game = 'up_n_down'
        #        game = 'atlantis'
        #        game = 'qbert'
        #        game = 'centipede'
        #        game = 'battle_zone'

        delay_frames = 6  # 60 fps frames to delay commands to simulate real world latency
        if game == 'breakout':
            delay_frames = 0

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
    logger.info(f'{num_actions} actions: {action_set}')

    name = f'delay_{game}{delay_frames}'
    for k, v in parms.items():
        if k != 'gpu':
            name += '_'
            name += str(v)
    logger.info(name)

    agent = Agent(data_dir, seed, num_actions, total_frames, **parms)

    avg = 0

    episode_scores = []
    episode_end = []
    environment_start = 0
    running_episode_score = 0
    environment_start_time = time.time()

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

    for u in range(agent.total_frames):
        if save_incremental_models and (u + 1) // 500_000 != last_model_save:
            last_model_save = (u + 1) // 500_000
            filename = f'{data_dir}/{name}_{u + 1}.model'
            logger.info('writing ' + filename)
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
                logger.warning(f'terminated at {frames_without_reward} frames without reward')
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
            running_episode_score = 0

            # calculate step speed
            now = time.time()
            frames_per_second = frames / (now - environment_start_time)
            environment_start_time = now

            logger.info(
                f'{rank}:{name} frame:{u:7} {frames_per_second:4.0f}/s eps {len(episode_scores) - 1:3},{frames:5}={int(episode_scores[-1]):5} err {agent.avg_error_ema:.1f} {agent.max_error_ema:.1f} loss {agent.train_loss_ema:.1f} targ {agent.target_ema:.1f} avg {avg:4.1f}'
            )

            torch.cuda.nvtx.range_pop()

        taken_action = agent.frame(ale.getScreenRGB(), reward, end_of_episode)

    filename = data_dir + '/' + name + '.policy_actions'
    logger.info('writing ' + filename)
    agent.policy_actions_buffer.cpu().numpy().tofile(filename)

    filename = data_dir + '/' + name + '.score'
    logger.info('writing ' + filename)
    episode_graph.cpu().numpy().tofile(filename)

    filename = data_dir + '/' + name + '.parms'
    logger.info('writing ' + filename)
    parms_graph.cpu().numpy().tofile(filename)

    plots = torch.zeros(len(episode_scores), 2)
    for i in range(len(episode_scores)):
        plots[i][0] = episode_end[i]
        plots[i][1] = episode_scores[i]
    filename = data_dir + '/' + name + '.scatter'
    logger.info('writing ' + filename)
    plots.cpu().numpy().tofile(filename)

    filename = data_dir + '/' + name + '.loss'
    logger.info('writing ' + filename)
    torch.tensor(agent.train_losses).cpu().numpy().tofile(filename)

    if save_model:
        filename = f'{data_dir}/{name}.model'
        logger.info('writing ' + filename)
        agent.save_model(filename)

    logger.info('done')


if __name__ == '__main__':
    main()
