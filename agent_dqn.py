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
    # Need in_channels for the CNN to know how many input nodes to have
    # Need num_actions for the output fully connected layer to know the output size (Q-value for each action)
    def __init__(self, in_channels: int, num_actions: int, obs_height: int, obs_width: int):
        super().__init__()
        # convolutional stem matches the classic DQN layout
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # compute the flattened size dynamically for arbitrary obs sizes
        conv_out = self._forward_conv(torch.zeros(1, in_channels, obs_height, obs_width))
        flat_size = conv_out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    # Forward pass for the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Replay buffer class that stores experience data to sample from
class ReplayBuffer:
    # Initializes replay buffer with specified capactiy, stack size (put together multiple frames as a replay experience to capture dynamics), 
    # obs_shape defines the size of the experiences stored in the buffer
    def __init__(self, capacity: int, stack_size: int, obs_shape: tuple[int, int, int]):
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
        # reshape to (B, stack_size * channels, H, W) for the conv net
        b, s, c, h, w = states.shape
        states = states.view(b, s * c, h, w)
        next_states = next_states.view(b, s * c, h, w)
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
        epsilon_decay_frames: int = 1_000_000, # this is how many frames over which we decay
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
        self.num_channels = 3  # RGB observations
        self.buffer_size = buffer_size # buffer size for replay buffer
        self.batch_size = batch_size # batch size for training 
        self.learning_rate = learning_rate # learning rate for training
        self.gamma = gamma # gamma used as discount factor
        self.train_start = train_start # at what step we start training 
        self.train_freq = train_freq # how often training happens
        self.target_update_freq = target_update_freq # how often we update the weights for the target network
        self.epsilon_start = epsilon_start # the value of epsilon in the beginning
        self.epsilon_end = epsilon_end  # the value of epsilon in the end
        self.epsilon_decay_frames = epsilon_decay_frames # over how many frames we decay epsilon
        self.grad_clip = grad_clip # value used for gradient clipping

        if torch.cuda.is_available(): # check if we have CUDA
            self.device = torch.device(f'cuda:{gpu}') # use GPU if yes
            torch.cuda.manual_seed_all(seed) # sets seed value for CUDA RNG that's on GPU
        elif torch.backends.mps.is_available(): # check if we have MPS
            self.device = torch.device('mps') # ues MPS as device if yes
        else:
            self.device = torch.device('cpu') # if no CUDA or MPS use CPU
        np.random.seed(seed) # use same seed with all np.random operations

        # instatiates Q Network, input will be (B, stack_size * channels, H, W)
        # num_actions is for the action space
        self.network = QNetwork(
            in_channels=stack_size * self.num_channels, num_actions=num_actions, obs_height=obs_height, obs_width=obs_width
        ).to(self.device)

        # we instantiate the target_network similarly
        self.target_network = QNetwork(
            in_channels=stack_size * self.num_channels, num_actions=num_actions, obs_height=obs_height, obs_width=obs_width
        ).to(self.device)
        
        # copy over regular Q network to target
        self.target_network.load_state_dict(self.network.state_dict())
        
        # optimizer used during training, pass in the network params to be updated and learning rate
        # need to understand why Adam vs regular SGD
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        # instantiate replay buffer
        self.replay = ReplayBuffer(buffer_size, stack_size=stack_size, obs_shape=(self.num_channels, obs_height, obs_width))

        #array to hold stack of frames for each env
        self.state_stacks = np.zeros((num_envs, stack_size, self.num_channels, obs_height, obs_width), dtype=np.uint8)
        # array to hold last stack of frames (assuming this is current state and state_stack is next?) for each env
        self.last_states = np.zeros_like(self.state_stacks)
        # array holding the last action for each environment, np.full populates an array of shape 
        # (numenvs, ) with -1's (no actions)
        self.last_actions = np.full((num_envs,), -1, dtype=np.int64)
        # number of frames trained
        self.frame_count = 0
        # number of steps trained
        self.training_steps = 0
        # where we start epsilon from
        self.epsilon = epsilon_start

        # what the last calcuated loss was
        self.last_loss = 0.0
        # exponential moving average of loss initialized
        self.loss_ema = None
        # the average over  q values calculated for optimal action each env
        self.last_avg_q = 0.0
        # the maximum q value calculated over each env
        self.last_max_q = 0.0

        # if loading from previous model and path exists then load model from file
        if load_file is not None and os.path.exists(load_file):
            self.load_model(load_file)

    # update each stack by replacing the oldest frame with the newest one
    def update_stacks(self, processed_frames: np.ndarray) -> None:
        # move all states in each stack one to the left (oldest frame wraps to end)
        self.state_stacks = np.roll(self.state_stacks, shift=-1, axis=1)
        # replace the last frame in each stack (the oldest frame) with the new processed frame
        # for each env's stack of frames
        self.state_stacks[:, -1, :, :] = processed_frames
    
    # scheduler for decaying epsilon over 'epsilon_decay_frames' frames
    # for epsilon greedy
    def epsilon_scheduler(self) -> float:
        # the fraction of epsilon_decay_frames that we have trained on so far
        # max of fraction can only ever be 1
        frac = min(self.frame_count / float(self.epsilon_decay_frames), 1.0)
        # add the fraction times distance from epsilon_end to epsilon_start to epsilon_start
        # to give appropriate epsilon
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # function for how we will handle a step of training
    def train_step(self):
        # we don't train till we reach self.train_start
        if self.replay.size < self.train_start:
            return
        # since we don't train every step, we return if
        # we haven't yet forgone enough frames to kick off
        # the next training step
        if self.frame_count % self.train_freq != 0:
            return

        # we sample experiences to train on from the replay_buffer
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size, self.device)
        # calculate q_values for each action from the network
        # (network is learning q function)
        q_values = self.network(states)
        # actions.unsqueeze(1) gives us an array of shape (B, 1) telling us what action was taken
        # for each batch
        # q_values is of shape (B, num_actions) where each entry is the 
        # q_value for each action for each batch
        # .gather with dimension 1 goes across the column dimension (goes through each row)
        # and uses the actions.unsqueeze(1) as the indices to index the q_value for the corresponding actions taken
        # .squeeze(1) get's rid of the 1 in (B, 1) because we need only a scalar per sample
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
       
        # we turn off gradient tracking for the target network (don't want to train it)
        with torch.no_grad():
            # calculate the Q values for the next_states we go to
            next_q = self.network(next_states)
            # find the optimal actions to take based on action that corresponds
            # with optimal q value
            next_actions = torch.argmax(next_q, dim=1)
            # calculate the q_values using the target_network 
            target_q = self.target_network(next_states)
            # get the max q_values for each of the actions from target_q
            max_next_q = target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # use max_next_q to estimate future return 
            target = rewards + self.gamma * max_next_q * (1.0 - dones)

        # compute loss between estimated q_values and target estimated q_values (calculated with one-step bootstrap)
        # the smooth_l1_loss dampens small losses while being linear for larger errors 
        # this means that gradients don't grow with error (for large errors the gradient magnitude will be a constant)
        # how smooth_l1_loss works :
        # err = |x - y|
        # if err > beta:
        #   loss = 0.5 * err ^2
        # else:
        #   loss = err - 0.5
        loss = F.smooth_l1_loss(q_values, target)
        # zero out all gradients
        self.optimizer.zero_grad()
        # calculate gradients for all parameters in respect to the loss
        loss.backward()
        # normalize gradients by computing L2 norm and then multiply by scaling factor of 
        # max_norm/norm
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        # update parameters
        self.optimizer.step()

        # store loss for logging in driver files
        self.last_loss = float(loss.item())
        # determine if we use the exponential moving average or not for smoother loss calculation
        if self.loss_ema is None:
            # keep the loss the same
            self.loss_ema = self.last_loss
        else:
            # use exponential moving average
            # smooths over spikes because most recent loss over time matters more 
            # (existing loss contribution decays by 0.95 each train_step)
            self.loss_ema = 0.95 * self.loss_ema + 0.05 * self.last_loss
        # increase training step count
        self.training_steps += 1
        # check if we hit target_update_freq in training
        if self.training_steps % self.target_update_freq == 0:
            # copy online network over to target_network if we've reached
            # the frequency for copying over
            self.target_network.load_state_dict(self.network.state_dict())

    # we reset everthing for a new episode
    def reset(self, observations: np.ndarray) -> None:
        # preprocess the new batch of observations
        processed = preprocess_batch(observations, height=self.obs_height, width=self.obs_width)
        # go through each env
        for env in range(self.num_envs):
            # update state_stacks by env by setting the initial stack as just the initial frame passed
            # in until we populate it with calling update
            self.state_stacks[env] = np.repeat(processed[env][None, ...], self.stack_size, axis=0)
            # there is no_last action at the start so set to -1
            self.last_actions[env] = -1
    # how the agent acts
    def act(self, observations: np.ndarray) -> np.ndarray:
        # process the observation batch the agent acts on
        processed = preprocess_batch(observations, height=self.obs_height, width=self.obs_width)
        # update the state stack with the processed batch
        self.update_stacks(processed)
        # use the epsilon scheduler to get the epsilon value for epsilon-greedy
        self.epsilon = self.epsilon_scheduler()
        # create array for actions for each env
        actions = np.empty(self.num_envs, dtype=np.int64)
        # normalize the frames and convert them to torch tensors
        stacked = (
            torch.from_numpy(self.state_stacks)
            .view(self.num_envs, self.stack_size * self.num_channels, self.obs_height, self.obs_width)
            .to(self.device, dtype=torch.float32)
            / 255.0
        )
        # turn off gradient tracking
        with torch.no_grad():
            # calculate q values from the online network
            q_values = self.network(stacked)
            # pick actions with highest q_values put them to cpu and np to avoid device
            # mismatch for action picking logic
            greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy()
            # storing this for logging 
            self.last_avg_q = float(q_values.mean().item())
            # storing this for logging 
            self.last_max_q = float(q_values.max().item())

        # go through each environment 
        for env in range(self.num_envs):
            # probability check
            if np.random.random() < self.epsilon:
                # randomly select action
                actions[env] = np.random.randint(self.num_actions)
            else:  
                # pick policy optimal action
                actions[env] = int(greedy_actions[env])
        # last_states are now the states we just observed
        self.last_states = self.state_stacks.copy()
        # last_actions are the actions we just took
        self.last_actions = actions.copy()
        # return the selected actions
        return actions

    # adds observations to replay buffer
    def observe(self, next_observations, rewards, terminations, truncations):
        # preprocess next batch of observations
        processed = preprocess_batch(next_observations, height=self.obs_height, width=self.obs_width)
        # make a copy of state_stacks
        next_stacks = self.state_stacks.copy()
        # roll the stack over, oldest observation is now the last entry
        next_stacks = np.roll(next_stacks, shift=-1, axis=1)
        # add the new observation frame in
        next_stacks[:, -1, :, :] = processed

        # go through the envs
        for env in range(self.num_envs):
            # if the last action was a no action, we don't have any next state
            if self.last_actions[env] == -1:
                continue
            # this lets us know if the next_observation was a terminal state
            done = bool(terminations[env] or truncations[env])
            # we add the experience to the replay buffer
            self.replay.add(self.last_states[env], self.last_actions[env], rewards[env], next_stacks[env], done)
            # if we reached a terminal state then we do the following
            if done:
                # we repeat the new frame stack_size times 
                stack = np.repeat(processed[env][None, ...], self.stack_size, axis=0)
                # set the stack as the new stack for the env
                self.state_stacks[env] = stack
                # set the last action as -1 since the env is starting anew
                self.last_actions[env] = -1
        # increase frame_count by the number of frames (one frame for each env)
        self.frame_count += self.num_envs

    # save the model at the certain path
    def save_model(self, path: str) -> None:
        # save the model's state_dict (learned weights)
        torch.save(self.network.state_dict(), path)
    # load a pre-existing model from given path
    def load_model(self, path: str) -> None:
        # load the state dict from given path
        state_dict = torch.load(path, map_location=self.device)
        # load the weights into the online network
        self.network.load_state_dict(state_dict)
        # loads same weights into target network
        self.target_network.load_state_dict(self.network.state_dict())

# Vectorized adapter 
class VectorDQNAgent(VectorAgent): 
    def __init__(  # construct the vector agent wrapper
        self,  # instance reference
        *,  # force keyword-only args
        num_envs: int,  # number of parallel envs
        seed: int,  # RNG seed
        num_actions: int,  # size of action space
        total_frames: int = 1_000_000,  # training horizon
        results_dir: Optional[str] = None,  # ignored for compatibility with vector harness
        **kwargs,  # extra args passed to DQNCore
    ):
        # sim_latency_vec passes results_dir; strip it so DQNCore doesn't error on unexpected kwarg
        kwargs.pop("results_dir", None)
        self.core = DQNCore(  # instantiate shared DQN core
            num_envs=num_envs,  # store env count in core
            seed=seed,  # seed core RNG
            num_actions=num_actions,  # pass action count
            total_frames=total_frames,  # pass training horizon
            **kwargs,  # forward remaining kwargs
        )
        self.num_envs = num_envs  # track current env batch size

     # update env count on reset
    def reset(self, num_envs: int) -> None: 
        self.num_envs = num_envs

    # pick actions for env batch
    def act(self, observations: np.ndarray) -> np.ndarray:  
        return self.core.act(observations)

    # log experiences into replay buffer
    def observe(self, next_observations, rewards, terminations, truncations, infos):  
        self.core.observe(next_observations, rewards, terminations, truncations)

     # trigger a training update
    def train_step(self) -> None:
        return self.core.train_step() 

    # save network weights
    def save_model(self, path: str) -> None:  
        return self.core.save_model(path)

    # load network weights 
    def load_model(self, path: str) -> None:  
        return self.core.load_model(path)
