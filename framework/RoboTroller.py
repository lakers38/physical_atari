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

import os

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

from framework.Actions import Action
from framework.ControlDevice import ControlDevice
from framework.Logger import logger

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_OPERATING_MODE = 11
ADDR_GOAL_CURRENT = 102
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

BUTTON_SERVO_DEFAULT = 2000  # center position
DPAD_SERVO_DEFAULT = 2048  # unpressed position


# tuple: (left_right_servo, up_down_servo, button_servo)
def get_positions_from_action_mapping() -> dict[Action, tuple[int, ...]]:
    DPAD_SERVO_STRENGTH = 300  # movement strength from reference
    BUTTON_DEFLECTION = 2048 - 200
    DPAD_SERVO_UP = DPAD_SERVO_DEFAULT + DPAD_SERVO_STRENGTH
    DPAD_SERVO_RIGHT = DPAD_SERVO_DEFAULT + DPAD_SERVO_STRENGTH
    DPAD_SERVO_DOWN = DPAD_SERVO_DEFAULT - DPAD_SERVO_STRENGTH
    DPAD_SERVO_LEFT = DPAD_SERVO_DEFAULT - DPAD_SERVO_STRENGTH
    mapping = {
        Action.NOOP: (
            DPAD_SERVO_DEFAULT,
            DPAD_SERVO_DEFAULT,
            BUTTON_SERVO_DEFAULT,
        ),
        Action.UP: (
            DPAD_SERVO_DEFAULT,
            DPAD_SERVO_UP,
            BUTTON_SERVO_DEFAULT,
        ),
        Action.FIRE: (
            DPAD_SERVO_DEFAULT,
            DPAD_SERVO_DEFAULT,
            BUTTON_DEFLECTION,
        ),
        Action.DOWN: (
            DPAD_SERVO_DEFAULT,
            DPAD_SERVO_DOWN,
            BUTTON_SERVO_DEFAULT,
        ),
        Action.LEFT: (
            DPAD_SERVO_LEFT,
            DPAD_SERVO_DEFAULT,
            BUTTON_SERVO_DEFAULT,
        ),
        Action.RIGHT: (
            DPAD_SERVO_RIGHT,
            DPAD_SERVO_DEFAULT,
            BUTTON_SERVO_DEFAULT,
        ),
        Action.UPFIRE: (DPAD_SERVO_DEFAULT, DPAD_SERVO_UP, BUTTON_DEFLECTION),
        Action.DOWNFIRE: (DPAD_SERVO_DEFAULT, DPAD_SERVO_DOWN, BUTTON_DEFLECTION),
        Action.LEFTFIRE: (DPAD_SERVO_LEFT, DPAD_SERVO_DEFAULT, BUTTON_DEFLECTION),
        Action.RIGHTFIRE: (DPAD_SERVO_RIGHT, DPAD_SERVO_DEFAULT, BUTTON_DEFLECTION),
        Action.UPLEFT: (
            DPAD_SERVO_LEFT,
            DPAD_SERVO_UP,
            BUTTON_SERVO_DEFAULT,
        ),
        Action.UPRIGHT: (
            DPAD_SERVO_RIGHT,
            DPAD_SERVO_UP,
            BUTTON_SERVO_DEFAULT,
        ),
        Action.DOWNLEFT: (
            DPAD_SERVO_LEFT,
            DPAD_SERVO_DOWN,
            BUTTON_SERVO_DEFAULT,
        ),
        Action.DOWNRIGHT: (
            DPAD_SERVO_RIGHT,
            DPAD_SERVO_DOWN,
            BUTTON_SERVO_DEFAULT,
        ),
        Action.UPLEFTFIRE: (DPAD_SERVO_LEFT, DPAD_SERVO_UP, BUTTON_DEFLECTION),
        Action.UPRIGHTFIRE: (DPAD_SERVO_RIGHT, DPAD_SERVO_UP, BUTTON_DEFLECTION),
        Action.DOWNLEFTFIRE: (DPAD_SERVO_LEFT, DPAD_SERVO_DOWN, BUTTON_DEFLECTION),
        Action.DOWNRIGHTFIRE: (DPAD_SERVO_RIGHT, DPAD_SERVO_DOWN, BUTTON_DEFLECTION),
    }
    return mapping


class RoboTroller(ControlDevice):
    def __init__(self, model_name, vendor_id, product_id, port_name, baud_rate=15200, current_limit=400):
        super().__init__()
        self.vendor_id = vendor_id
        self.product_id = product_id
        # TODO: auto-discover port_name via pyudev or serial attributes
        if not os.path.exists(port_name):
            raise ValueError(f"RoboTroller: {port_name} does not exist. Is Robotroller connected?")

        self.portHandler = PortHandler(port_name)
        self.packetHandler = PacketHandler(2.0)

        if not self.portHandler.openPort():
            raise ValueError(f"RoboTroller: Failed to open port {port_name}.")

        if not self.portHandler.setBaudRate(baud_rate):
            raise ValueError("RoboTroller: Failed to set baudrate.")

        self.position_from_action_mapping = get_positions_from_action_mapping()
        self.current_limit = current_limit

        self.left_right_servo_id = 51
        self.up_down_servo_id = 52
        self.button_servo_id = 50
        self.servo_ids = [self.left_right_servo_id, self.up_down_servo_id, self.button_servo_id]

        self.prev_positions = (DPAD_SERVO_DEFAULT, DPAD_SERVO_DEFAULT, BUTTON_SERVO_DEFAULT)

        for servo_id in self.servo_ids:
            self._initialize_servo(servo_id)

        # set to default positions
        self.update_positions(self.prev_positions, force=True)

    def _initialize_servo(self, servo_id):
        # Disable torque. Mode can only be changed when torque is disabled.
        self._write_byte(servo_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "disable torque")
        # Change mode to current based position control. This mode allows us to limit the maximum current draw.
        self._write_byte(servo_id, ADDR_OPERATING_MODE, 5, "set mode")
        # The ADDR_GOAL_CURRENT is treated as a limit on the current draw in mA.
        # In current based position control mode, the current is limited to [-current_limit, current_limit].
        self._write_word(servo_id, ADDR_GOAL_CURRENT, self.current_limit, "set current")
        self._write_byte(servo_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, "enable torque")

    def _write_byte(self, servo_id, addr, val, label):
        result, err = self.packetHandler.write1ByteTxRx(self.portHandler, servo_id, addr, val)
        if result != COMM_SUCCESS:
            logger.warning(f"{label} servo={servo_id}: {self.packetHandler.getTxRxResult(result)}")
        elif err != 0:
            logger.warning(f"{label} error servo={servo_id}: {self.packetHandler.getRxPacketError(err)}")

    def _write_word(self, servo_id, addr, val, label):
        result, err = self.packetHandler.write2ByteTxRx(self.portHandler, servo_id, addr, val)
        if result != COMM_SUCCESS:
            logger.warning(f"{label} servo={servo_id}: {self.packetHandler.getTxRxResult(result)}")
        elif err != 0:
            logger.warning(f"{label} error servo={servo_id}: {self.packetHandler.getRxPacketError(err)}")

    def shutdown(self):
        for servo_id in self.servo_ids:
            self._write_byte(servo_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "disable torque")
        self.portHandler.closePort()

    # positions -> (left_right_servo, up_down_servo, button_servo)
    def update_positions(self, positions, force=False):
        for i, pos in enumerate(positions):
            if force or self.prev_positions[i] != pos:
                result, err = self.packetHandler.write4ByteTxRx(
                    self.portHandler, self.servo_ids[i], ADDR_GOAL_POSITION, pos
                )
                if result != COMM_SUCCESS:
                    logger.warning(
                        f"set position servo={self.servo_ids[i]}: {self.packetHandler.getTxRxResult(result)}"
                    )
                elif err != 0:
                    logger.warning(
                        f"error setting position servo={self.servo_ids[i]}: {self.packetHandler.getRxPacketError(err)}"
                    )

        self.prev_positions = positions

    def apply_action(self, action, state):
        positions = self.position_from_action_mapping[Action(action)]
        self.update_positions(positions, True)
