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

# agent_random.py
#
# Use the last evaluations for target calculation instead of a target model evaluation
import time

import numpy as np

from framework.Logger import logger


class Agent:
    def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
        # defaults that might be overridden by explicit experiment runs

        self.num_actions = num_actions  # many games can use a reduced action set for faster learning
        self.gpu = -1
        self.total_frames = total_frames
        self.frame_skip = 4
        self.seed = seed
        self.training_model = None
        self.ring_buffer_size = 0
        self.train_losses = 0
        self.use_model = 0

        # dynamically override configuration
        for key, value in kwargs.items():
            try:
                assert hasattr(self, key)
            except AssertionError:
                logger.error(f"agent_random: Request to set unknown property: {key}")
                continue
            setattr(self, key, value)

        # variables used by policy
        self.step = 0
        self.rng = np.random.default_rng(self.seed)
        self.selected_action_index = 0

    # --------------------------------
    # Returns the selected action index
    # --------------------------------
    def frame(self, observation_rgb8, reward, end_of_episode):
        if 0 == self.step % self.frame_skip:
            self.selected_action_index = self.rng.integers(self.num_actions)
        self.step += 1
        return self.selected_action_index

    def save_model(self, filename):
        pass
