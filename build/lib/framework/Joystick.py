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

import queue
import threading
import time

from framework.Actions import Action
from framework.ControlDevice import ControlDevice
from framework.Logger import logger


class Joystick:
    def __init__(self, device: ControlDevice, threaded: bool = False):
        self.device = device
        assert self.device is not None
        self.threaded = threaded
        self.running = True
        self.action_queue = queue.Queue(maxsize=2)
        self.thread = None

        if self.threaded:
            self.thread = threading.Thread(target=self._process_action_queue, daemon=True)
            self.thread.start()

    def shutdown(self):
        self.running = False
        if self.threaded and self.thread is not None:
            self.thread.join()
        if self.device:
            self.device.shutdown()
            self.device = None
        with self.action_queue.mutex:
            self.action_queue.queue.clear()

    def _process_action_queue(self):
        while self.running:
            try:
                action = self.action_queue.get(timeout=0.1)
                self.device.apply_action(action, 1)
            except queue.Empty:
                pass
            time.sleep(0.001)

    def apply_action(self, action: Action) -> None:
        if self.threaded:
            if self.action_queue.full():
                try:
                    _ = self.action_queue.get_nowait()
                except queue.Empty:
                    pass
            self.action_queue.put(action)
        else:
            self.device.apply_action(action, 1)

    def __repr__(self):
        return f"<Joystick threaded={self.threaded} thread_running={self.running} device={type(self.device).__name__}>"
