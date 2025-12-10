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


class BaseEnv:
    def close(self):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError

    def get_action_set(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def act(self, action):
        raise NotImplementedError

    def game_over(self):
        raise NotImplementedError

    def lives(self):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError
