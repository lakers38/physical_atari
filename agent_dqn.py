from __future__ import annotations

import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Deque, List, Optional
from collections import deque

from agent_utils import preprocess_batch
from vector_agents import VectorAgent

# Defines QNetwork class that inherits from nn.Module
class QNetwork(nn.Module):
    # Initializes class
    # Need in_channels for the CNN to know how many input nodes to have
    # Need num_actions for the output fully connected layer to know the output size (Q-value for each action) 
    def __init__(self, in_channels: int, num_actions: int):
        # runs base class constructor to initialize different book-keeping done inside of nn.Module (OrderedDict for
        # parameters, buffers, hooks)
        super().__init__()
        # defines first convolutional layer in the network
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # defines second convolutional layer in the network
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # defines third convolutional layer in the network
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # first fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        # second fully connected layer
        self.fc2 = nn.Linear(512, num_actions)

    # Forward pass for the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pass in input to first convolutional layer
        x = F.relu(self.conv1(x))
        # apply relu to output of second CNN
        x = F.relu(self.conv2(x))
        # apply relu to output of third CNN
        x = F.relu(self.conv3(x))
        # takes the input with size (batch, channels, H, W) and resizes into (batch, channels*height*width)
        x = x.view(x.size(0), -1)
        # passes resized input into fully connected layer and applies ReLU
        x = F.relu(self.fc1(x))
        # final output given by last fully connected layer
        return self.fc2(x)

# Replay buffer class that stores experience data to sample from
class ReplayBuffer:
    # Initializes replay buffer with specified capactiy, stack size (put together multiple frames as a replay experience to capture dynamics), 
    # obs_shape defines the size of the experiences stored in the buffer
    def __init__(self, capacity: int, stack_size: int, obs_shape: tuple[int, int]):
        # sets capacity for the buffer
        self.capacity = capacity
        # sets stack_size for the buffer
        self.stack_size = stack_size
        # sets expected obs_shape for the experiences
        self.obs_shape = obs_shape
        # create numpy state array to store states
        self.states = np.zeros((capacity, stack_size, *obs_shape), dtype=np.uint8)
        # create numpy next_state array to store the next state reached from the current state
        self.next_states = np.zeros((capacity, stack_size, *obs_shape), dtype=np.uint8)
        # create numpy action array, stores action taken at each state for each experience
        self.actions = np.zeros(capacity, dtype=np.int64)
        # the assigned reward for each experience 
        self.rewards = np.zeros(capacity, dtype=np.int64)
        # numpy array specifying if each state for experiences 
        self.dones = np.zeros(capacity, dtype=np.bool_)
        # ptr keeps track of where we are in the buffer
        self.ptr = 0
        # tracks if the buffer is full, if it is then we return capacity for the size of the buffer
        # as opposed to the ptr
        self.full = False

    # allows this method to be an attribute on self (you can call buffer.size)
    @property
    def size(self) -> int:
        return self.capacity if self.full else self.ptr

    # adds sample (state, action, reward, next_state, done) to the buffer 
    def add(self, state, action, reward, next_state, done) -> None:
        # set the specific point in buffer to be the state that was just observed
        self.states[self.ptr] = state
        # set the next_state that was reached in this observation
        self.next_states[self.ptr] = next_state
        # set the action that was taken for this experience
        self.actions[self.ptr] = action
        # set the reward that was gained for this experience
        self.rewards[self.ptr] = reward
        # set the done state for this experience (whether the state observed was terminal or not)
        self.dones[self.ptr] = done
        # update pointer, we do mod capacity because we want the ptr value 
        # to be constrained to be at most the capacity value
        self.ptr = (self.ptr + 1) % self.capacity
        # once ptr has gotten to zero it means that it has reached capacity
        # so we set the buffer being full to true
        if self.ptr == 0:
            self.full = True
    # sample experience data from the buffer by batch and also put it on the same device used for training
    def sample(self, batch_size: int, device: torch.device):
        # set random index to use for sampling
        idx = np.random.randint(0, self.size, size=batch_size)
        # sample experience state, change from numpy array to normalized torch tensor on device used for
        # training, with the appropriate dtype
        states = torch.from_numpy(self.states[idx]).to(device, dtype=torch.float32) / 255.0
        # sample experience action, change from numpy array to torch tensor on device used for
        # training, with the appropriate dtype
        actions = torch.from_numpy(self.actions[idx]).to(device, dtype=torch.int64)
        # sample experience reward, change from numpy array to torch tensor on device used for
        # training, with the appropriate dtype
        rewards = torch.from_numpy(self.rewards[idx]).to(device, dtype=torch.float32)
        # sample experience next state, change from numpy array to normalized torch tensor on device used for
        # training, with the appropriate dtype
        next_states = torch.from_numpy(self.next_states[idx]).to(device, dtype=torch.float32) / 255.0
        # sample experience done state, change from numpy array to torch tensor
        # on device used for training, with appropriate dtype
        dones = torch.from_numpy(self.dones[idx].astype(np.float32)).to(device, dtype=torch.float32)
        # return experience data
        return states, actions, rewards, next_states, dones

# Defines DQNCoreClass that instatiates buffer, Q-Networks, and ReplayBuffer
class DQNCore:
    # initialization method for the class
    def __init__(
        self,
        num_envs: int, # number of environments to be training at one time
        seed: int, # seed to use for random number generators
        num_actions: int, # the number of actions available in the action space
        total_frames: int, # total_frames to train for 
        *, # in python this separate positional args from keyword args
        stack_size: int = 4, # stack_size argument for how many frames we stack together
        obs_height: int = 84, # height of the observation frames
        obs_width: int = 84, # width of the observation frames 
        buffer_size: int = 100_000, # how big the buffer will be (max number of experiences it'll hold)
        batch_size: int = 32, # batch size used for training
        learning_rate: float = 2.5e-4, # learning rate used for training
        gamma: float = 0.99, # discount factor for returns
        train_start: int = 50_000, # training starts after this many experiences are in the buffer
        train_freq: int = 4, # we train the network every train_freq number of steps
        target_update_freq: int = 10_000, # we copy over the network's weights every target_update_freq number of steps
        epsilon_start: float = 1.0, # this is the start value for epsilon used for epsilon-greedy
        epsilon_end: float = 0.1, # this is the end value epsilon is decayed to
        epsilon_decay_frames: int = 1_000_000, # this is how many frames over which we decay (why can't this just be total_frames?)
        grad_clip: Optional[float] = 10.0, # this is the value we clip the gradients to
        load_file: Optional[str] = None, # if we want resume training from a previous model
        gpu: int = 0, # which gpu we use to train
    ):
        self.num_envs = num_envs # sets the number of environments we use to train
        self.num_actions = num_actions # the number of actions in the game's action space
        self.total_frames = total_frames # total number of frames for training
        self.stack_size = stack_size #  the number of frames we stack together for each observation
        self.obs_height = obs_height # observation height
        self.obs_width = obs_width # observation width
        self.buffer_size = buffer_size # buffer size for replay buffer
        self.batch_size = batch_size # batch size for training 
        self.learning_rate = learning_rate # learning rate for training
        self.gamma = gamma # gamma used as discount factor
        self.train_start = train_start # at what step we start training 
        self.train_freq = train_freq # how often training happens
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_frames = epsilon_decay_frames
        self.grad_clip = grad_clip

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu}')
            torch.cuda.manual_seed_all(seed)
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        np.random.seed(seed)

        self.network = QNetwork(in_channels=stack_size, num_actions=num_actions).to(self.device)
        self.target_network = QNetwork(in_channels=stack_size, num_actions=num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.replay = ReplayBuffer(buffer_size, stack_size=stack_size, obs_shape=(obs_height, obs_width))

        self.state_stacks = np.zeros((num_envs, stack_size, obs_height, obs_width), dtype=np.uint8)
        self.last_states = np.zeros_like(self.state_stacks)
        self.last_actions = np.full((num_envs,), -1, dtype=np.int64)

        self.frame_count = 0
        self.training_steps = 0
        self.epsilon = epsilon_start

        self.last_loss = 0.0
        self.loss_ema = None
        self.last_avg_q = 0.0
        self.last_max_q = 0.0


        if load_file is not None and os.path.exists(load_file):
            self.load_model(load_file)

    def update_stacks(self, processed_frames: np.ndarray) -> None:
        self.state_stacks = np.roll(self.state_stacks, shift=-1, axis=1)
        self.state_stacks[:, -1, :, :] = processed_frames

    def epsilon_scheduler(self) -> float:
        frac = min(self.frame_count / float(self.epsilon_decay_frames), 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def train_step(self):
        if self.replay.size < max(self.train_start, self.batch_size):
            return
        if self.frame_count % self.train_freq != 0:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size, self.device)
        q_values = self.network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.network(next_states)
            next_actions = torch.argmax(next_q, dim=1)
            target_q = self.target_network(next_states)
            max_next_q = target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        self.optimizer.step()

        self.last_loss = float(loss.item())
        if self.loss_ema is None:
            self.loss_ema = self.last_loss
        else:
            self.loss_ema = 0.95 * self.loss_ema + 0.05 * self.last_loss

        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def reset(self, observations: np.ndarray) -> None:
        processed = preprocess_batch(observations)
        for env in range(self.num_envs):
            self.state_stacks[env] = np.repeat(processed[env][None, ...], self.stack_size, axis=0)
            self.last_actions[env] = -1

    def act(self, observations: np.ndarray) -> np.ndarray:
        processed = preprocess_batch(observations)
        self.update_stacks(processed)

        self.epsilon = self.epsilon_scheduler()
        actions = np.empty(self.num_envs, dtype=np.int64)
        stacked = torch.from_numpy(self.state_stacks).to(self.device, dtype=torch.float32) / 255.0

        with torch.no_grad():
            q_values = self.network(stacked)
            greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy()
            self.last_avg_q = float(q_values.mean().item())
            self.last_max_q = float(q_values.max().item())

        for env in range(self.num_envs):
            if np.random.random() < self.epsilon:
                actions[env] = np.random.randint(self.num_actions)
            else:
                actions[env] = int(greedy_actions[env])
        self.last_states = self.state_stacks.copy()
        self.last_actions = actions.copy()
        return actions

    def observe(self, next_observations, rewards, terminations, truncations):
        processed = preprocess_batch(next_observations)
        next_stacks = self.state_stacks.copy()
        next_stacks = np.roll(next_stacks, shift=-1, axis=1)
        next_stacks[:, -1, :, :] = processed

        for env in range(self.num_envs):
            if self.last_actions[env] == -1:
                continue
            done = bool(terminations[env] or truncations[env])
            self.replay.add(self.last_states[env], self.last_actions[env], rewards[env], next_stacks[env], done)
            if done:
                stack = np.repeat(processed[env][None, ...], self.stack_size, axis=0)
                self.state_stacks[env] = stack
                self.last_actions[env] = -1

        self.frame_count += self.num_envs

    def save_model(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)

    def load_model(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.target_network.load_state_dict(self.network.state_dict())

    # Single-env adapter (for harness_physical + sim_latency)


class Agent:
    def __init__(self, seed=0, num_actions=18, total_frames=1_000_000, **kwargs):
        self.core = DQNCore(
            num_envs=1, seed=seed, num_actions=num_actions, total_frames=total_frames, **kwargs
        )
        self.prev_obs: Optional[np.ndarray] = None
        self.prev_reward = 0.0
        self.prev_done = False

    def frame(self, observation_rgb8, reward, end_of_episode):
        obs_batch = observation_rgb8[None, ...]
        if self.prev_obs is None:
            self.core.reset(obs_batch)
        actions = self.core.act(obs_batch)
        if self.prev_obs is not None:
            self.core.observe(
                self.prev_obs,
                np.array([self.prev_reward]),
                np.array([self.prev_done]),
                np.array([False]),
            )
            self.core.train_step()
        self.prev_obs = observation_rgb8[None, ...]
        self.prev_reward = reward
        self.prev_done = bool(end_of_episode > 0)
        return int(actions[0])

    def save_model(self, path: str) -> None:
        self.core.save_model(path)

    def load_model(self, path: str) -> None:
        self.core.load_model(path)


# Vectorized adapter (for future vector trainer)


class VectorDQNAgent(VectorAgent):
    def __init__(
        self,
        *,
        num_envs: int,
        seed: int,
        num_actions: int,
        total_frames: int = 1_000_000,
        **kwargs,
    ):
        self.core = DQNCore(
            num_envs=num_envs,
            seed=seed,
            num_actions=num_actions,
            total_frames=total_frames,
            **kwargs,
        )
        self.num_envs = num_envs

    def reset(self, num_envs: int) -> None:
        self.num_envs = num_envs

    def act(self, observations: np.ndarray) -> np.ndarray:
        return self.core.act(observations)

    def observe(self, next_observations, rewards, terminations, truncations, infos):
        self.core.observe(next_observations, rewards, terminations, truncations)

    def train_step(self) -> None:
        return self.core.train_step()

    def save_model(self, path: str) -> None:
        return self.core.save_model(path)

    def load_model(self, path: str) -> None:
        return self.core.load_model(path)
