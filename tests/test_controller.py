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

import json
import time

from framework.ControlDeviceCfg import create_control_device_from_cfg
from framework.Keyboard import Keyboard

# Use the keyboard to send command to the device communicating actions to the Atari.


def main(args):
    with open(args.device_config) as kf:
        device_data = kf.read()

    device_data = json.loads(device_data)
    device = create_control_device_from_cfg(**device_data)
    keyboard = Keyboard(device)

    try:
        while True:
            should_quit, _ = keyboard.update()
            if should_quit:
                print("Exiting...")
                break
            time.sleep(0.001)

    except KeyboardInterrupt:
        pass

    finally:
        keyboard.shutdown()


def get_argument_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="keyboard_test.py arguments")
    # parser.add_argument('--device_config', type=str, default="configs/controllers/io_controller.json")
    parser.add_argument('--device_config', type=str, default="configs/controllers/robotroller.json")
    return parser


if __name__ == '__main__':
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    main(args)
