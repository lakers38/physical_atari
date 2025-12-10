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
from enum import Enum
from typing import Optional

from pynput.keyboard import Key, Listener

from framework.Actions import Action
from framework.ControlDevice import ControlDevice
from framework.Logger import logger

"""
Use a keyboard to send commands to control device.
"""


class Keys(Enum):
    UP = "w"
    LEFT = "a"
    DOWN = "s"
    RIGHT = "d"
    FIRE = " "
    NOOP = "e"


def get_keys_to_action() -> dict[tuple[int, ...], Action]:
    mapping = {
        Action.NOOP: (Keys.NOOP.value,),
        Action.UP: (Keys.UP.value,),
        Action.FIRE: (Keys.FIRE.value,),
        Action.DOWN: (Keys.DOWN.value,),
        Action.LEFT: (Keys.LEFT.value,),
        Action.RIGHT: (Keys.RIGHT.value,),
        Action.UPFIRE: (Keys.UP.value, Keys.FIRE.value),
        Action.DOWNFIRE: (Keys.DOWN.value, Keys.FIRE.value),
        Action.LEFTFIRE: (Keys.LEFT.value, Keys.FIRE.value),
        Action.RIGHTFIRE: (Keys.RIGHT.value, Keys.FIRE.value),
        Action.UPLEFT: (Keys.UP.value, Keys.LEFT.value),
        Action.UPRIGHT: (Keys.UP.value, Keys.RIGHT.value),
        Action.DOWNLEFT: (Keys.DOWN.value, Keys.LEFT.value),
        Action.DOWNRIGHT: (Keys.DOWN.value, Keys.RIGHT.value),
        Action.UPLEFTFIRE: (Keys.UP.value, Keys.LEFT.value, Keys.FIRE.value),
        Action.UPRIGHTFIRE: (Keys.UP.value, Keys.RIGHT.value, Keys.FIRE.value),
        Action.DOWNLEFTFIRE: (Keys.DOWN.value, Keys.LEFT.value, Keys.FIRE.value),
        Action.DOWNRIGHTFIRE: (Keys.DOWN.value, Keys.RIGHT.value, Keys.FIRE.value),
    }

    full_action_set = [act for act in Action]

    return {tuple(sorted(mapping[act_idx])): act_idx for act_idx in full_action_set}


class Keyboard:
    def __init__(self, device: ControlDevice, threaded: bool = False):
        self.device = device
        assert self.device is not None

        self.keys_to_action = get_keys_to_action()
        self.relevant_keys = {k for combo in self.keys_to_action for k in combo}
        self.pressed_keys = set()
        self.input_focus = True
        self.exit_requested = False

        self.threaded = threaded
        self.running = False
        self.action_queue = queue.Queue()
        self.thread = None

        self.start()

        help_text = (
            ", ".join(f"{key.name}:{'space' if key.value == ' ' else key.value}" for key in Keys) + ", QUIT: esc"
        )
        logger.info(f"Keyboard: {help_text}")

    def shutdown(self):
        self.stop()

        self.device.shutdown()
        self.device = None

    def _parse_key(self, key) -> Optional[str]: # type: ignore
        if key == Key.space:
            return ' '
        try:
            return key.char
        except AttributeError:
            return None

    def on_press(self, key):
        if not self.input_focus:
            return
        keycode = self._parse_key(key)
        if keycode and keycode in self.relevant_keys and keycode not in self.pressed_keys:
            self.pressed_keys.add(keycode)
            new_action = self._get_action_from_keys(self.pressed_keys)
            self.action_queue.put(('press', new_action))

    def on_release(self, key):
        keycode = self._parse_key(key)
        if keycode and keycode in self.relevant_keys and keycode in self.pressed_keys:
            self.pressed_keys.remove(keycode)
            new_action = self._get_action_from_keys(self.pressed_keys)
            self.action_queue.put(('release', new_action))

        if key == Key.esc:
            self.exit_requested = True

    def start(self):
        if self.threaded:
            self.running = True
            self.thread = threading.Thread(target=self._process_actions, daemon=True)
            self.thread.start()

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def stop(self):
        # stop listening for events, this should join the listener thread
        self.listener.stop()

        self.running = False
        if self.threaded and self.thread is not None:
            self.thread.join()

    def set_input_focus(self, focus: bool):
        logger.info(f"Keyboard: input_focus={focus}")
        self.input_focus = focus

    def _get_action_from_keys(self, keys):
        return self.keys_to_action.get(tuple(sorted(keys)), Action.NOOP)

    # when running non-threaded, expects the calling program
    # to update at a regular frequency
    def update(self):
        if not self.input_focus:
            return False, Action.NOOP

        try:
            state, action = self.action_queue.get_nowait()
            # logger.debug(action)
            signal_state = 1 if state == "press" else 0
            # print(f"action={action} state={signal_state}")
            self.device.apply_action(action, signal_state)
            return self.exit_requested, action
        except queue.Empty:
            return self.exit_requested, Action.NOOP

    def _process_actions(self):
        while self.running:
            self.update()
            time.sleep(0.01)

    def __repr__(self):
        return f"<Keyboard threaded={self.threaded} thread_running={self.running} device={type(self.device).__name__}>"
