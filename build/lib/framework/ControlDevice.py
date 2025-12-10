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

from abc import ABC, abstractmethod

from framework.Actions import Action


class ControlDevice(ABC):
    @abstractmethod
    def apply_action(self, action: Action, state: int):
        # apply an action such as UP, FIRE, etc., optionally with a press/release state.
        pass

    @abstractmethod
    def shutdown(self):
        # shutdown or clean up the control device.
        pass

    def get_pins(self) -> list[int]:
        # optional: return list of active GPIO-like pins used by the device.
        return []
