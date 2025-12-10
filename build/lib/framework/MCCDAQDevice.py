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

from enum import Enum, IntEnum

from framework.Actions import Action
from framework.ControlDevice import ControlDevice
from framework.Logger import logger


class Signal(Enum):
    LOW = 0
    HIGH = 1


# Layout for the MCC USB-1024LS
class MCCDAQ1024LSLayout:
    """
    The MCC DAQ has 4 ports:
    - PORTA  - A0-A7 mapped to pins 21-28
    - PORTB  - B0-B7 mapped to pins 32-39
    - PORTCL - C0-C3 mapped to pins 1-4
    - PORTCH - C4-C7 mapped to pins 5-8
    """

    # uldaq/ul_enums.py
    class ULDAQPortEnum(IntEnum):
        FIRSTPORTA = 10
        FIRSTPORTB = 11
        FIRSTPORTCL = 12
        FIRSTPORTCH = 13

    # from: https://github.com/mccdaq/uldaq/blob/1d8404159c0fb6d2665461b80acca5bbef5c610a/src/hid/dio/DioUsbDio24.cpp#L177
    class HIDPortEnum(IntEnum):
        FIRSTPORTA = 1
        FIRSTPORTB = 4
        FIRSTPORTCL = 8
        FIRSTPORTCH = 2

    # https://github.com/mccdaq/uldaq/blob/1d8404159c0fb6d2665461b80acca5bbef5c610a/src/hid/dio/DioUsbDio24.h
    class ReportCommand(IntEnum):
        DIN = 0x00  # Read all pins on a port
        DOUT = 0x01  # Write to all pins on a port
        BITIN = 0x02  # Read a single pin
        BITOUT = 0x03  # Write a single pin
        DCONFIG = 0x0D  # Configure direction of a port

    def __init__(self, use_hid: bool):
        self.use_hid = use_hid
        self.PortEnum = self.HIDPortEnum if use_hid else self.ULDAQPortEnum

        self.port_ranges = {
            self.PortEnum.FIRSTPORTA: (21, 29),
            self.PortEnum.FIRSTPORTB: (32, 40),
            self.PortEnum.FIRSTPORTCL: (1, 5),
            self.PortEnum.FIRSTPORTCH: (5, 9),
        }

    def get_port_for_pin(self, pin):
        for port, (start, end) in self.port_ranges.items():
            if start <= pin <= end:
                return port
        return None

    # Convert the pin to the corresponding bit value for the port.
    def get_bit_for_pin(self, port, pin):
        # Pins are 1-based
        start_range, _ = self.port_ranges[port]

        # Find the port the pin belongs to, and offset the port range
        # such that it is in the range [1,number_of_bits].
        # CL = (1,4)
        # CH = (1,4)
        # A  = (1,8)
        # B  = (1,8)
        # And, then offset by 1 to convert to 0-based range for bit.
        bit = pin - (start_range - 1) - 1
        return bit


def get_pins_from_action_mapping(action_to_pin_map) -> dict[Action, tuple[int, ...]]:
    PIN_UP = action_to_pin_map[Action.UP]
    PIN_DOWN = action_to_pin_map[Action.DOWN]
    PIN_RIGHT = action_to_pin_map[Action.RIGHT]
    PIN_LEFT = action_to_pin_map[Action.LEFT]
    PIN_FIRE = action_to_pin_map[Action.FIRE]

    mapping = {
        Action.NOOP: (-1,),
        Action.UP: (PIN_UP,),
        Action.FIRE: (PIN_FIRE,),
        Action.DOWN: (PIN_DOWN,),
        Action.LEFT: (PIN_LEFT,),
        Action.RIGHT: (PIN_RIGHT,),
        Action.UPFIRE: (PIN_UP, PIN_FIRE),
        Action.DOWNFIRE: (PIN_DOWN, PIN_FIRE),
        Action.LEFTFIRE: (PIN_LEFT, PIN_FIRE),
        Action.RIGHTFIRE: (PIN_RIGHT, PIN_FIRE),
        Action.UPLEFT: (PIN_UP, PIN_LEFT),
        Action.UPRIGHT: (PIN_UP, PIN_RIGHT),
        Action.DOWNLEFT: (PIN_DOWN, PIN_LEFT),
        Action.DOWNRIGHT: (PIN_DOWN, PIN_RIGHT),
        Action.UPLEFTFIRE: (PIN_UP, PIN_LEFT, PIN_FIRE),
        Action.UPRIGHTFIRE: (PIN_UP, PIN_RIGHT, PIN_FIRE),
        Action.DOWNLEFTFIRE: (PIN_DOWN, PIN_LEFT, PIN_FIRE),
        Action.DOWNRIGHTFIRE: (PIN_DOWN, PIN_RIGHT, PIN_FIRE),
    }
    return mapping


# NOTE: Implementation is for MCCDAQ USB-1024LS I/O device board
# From https://forums.atariage.com/topic/266868-joystick-pinout-question/#comment-3788375:
# 'SIGNAL_LOW' will trigger the action on Atari, and 'SIGNAL_HIGH' is off.
class DAQDevice(ControlDevice):
    def __init__(
        self,
        model_name: str,
        vendor_id: str,
        product_id: str,
        pin_to_action_str_map,
        active_low=True,
        use_hid_backend=False,
    ):
        super().__init__()
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.use_hid = use_hid_backend
        self.signal_active = Signal.LOW.value if active_low else Signal.HIGH.value
        self.signal_inactive = Signal.HIGH.value if active_low else Signal.LOW.value

        self.layout = MCCDAQ1024LSLayout(self.use_hid)
        self.action_to_pin_map = {}
        for pin, action_str in pin_to_action_str_map.items():
            if Action.has_key(action_str):
                self.action_to_pin_map[Action[action_str]] = int(pin)
            else:
                logger.warning(f"'{action_str}' is not a valid action.")
        self.action_pins = tuple(int(pin) for pin in pin_to_action_str_map.keys())
        self.pins_from_action_map = get_pins_from_action_mapping(self.action_to_pin_map)

        ports = [self.layout.get_port_for_pin(p) for p in self.action_pins]
        assert all(p == ports[0] for p in ports), "Only one port supported at a time"

        self.port = ports[0]
        self.port_state = 0x00

        if self.use_hid:
            from framework.HIDDevice import HIDDevice

            # from: https://github.com/mccdaq/uldaq/blob/1d8404159c0fb6d2665461b80acca5bbef5c610a/src/uldaq.h#L945
            class DigitalDirection(IntEnum):
                INPUT = 1
                OUTPUT = 2

            self._ReportCommand = MCCDAQ1024LSLayout.ReportCommand

            # When using split-port C, we must write all 8 bits at once, but we want to update CL and CH
            # individually. Cache the last known values and combine before writing to device.
            self.port_cl_val = 0
            self.port_ch_val = 0
            self.backend = HIDDevice(vendor_id, product_id)

            # configure the port
            report = bytearray([self._ReportCommand.DCONFIG, self.port, DigitalDirection.OUTPUT])
            self.backend.write_sync(report)
        else:
            import uldaq

            assert self.layout.PortEnum.FIRSTPORTA == uldaq.DigitalPortType.FIRSTPORTA
            assert self.layout.PortEnum.FIRSTPORTB == uldaq.DigitalPortType.FIRSTPORTB
            assert self.layout.PortEnum.FIRSTPORTCL == uldaq.DigitalPortType.FIRSTPORTCL
            assert self.layout.PortEnum.FIRSTPORTCH == uldaq.DigitalPortType.FIRSTPORTCH

            devices = uldaq.get_daq_device_inventory(uldaq.InterfaceType.USB)
            if not devices:
                raise RuntimeError("No DAQ device found.")

            self.device = uldaq.DaqDevice(devices[0])  # TODO: match model_name properly
            assert self.device is not None, "Failed to get first DAQ device"
            self.device.connect()

            self.dio_device = self.device.get_dio_device()
            assert self.dio_device is not None, "Failed to get DIO device"

            self.dio_device.d_config_port(self.port, uldaq.DigitalDirection.OUTPUT)

        self.default_port_state = self._build_default_state()

        # Initialize action pin values to off.
        #  From https://forums.atariage.com/topic/266868-joystick-pinout-question/#comment-3788375:
        # 'SIGNAL_LOW' will trigger the action on Atari, and 'SIGNAL_HIGH' is off.
        self._set_pins(self.port, self.action_pins, self.signal_inactive)

    def shutdown(self):
        # Set the action pins to off
        #  From https://forums.atariage.com/topic/266868-joystick-pinout-question/#comment-3788375:
        # 'SIGNAL_LOW' will trigger the action on Atari, and 'SIGNAL_HIGH' is off.
        self._set_pins(self.port, self.action_pins, self.signal_inactive)

        if self.use_hid:
            if self.backend is not None:
                self.backend.shutdown()
        else:
            if self.device is not None:
                self.device.disconnect()

    def _build_default_state(self):
        bits = self._get_bits_for_pins(self.port, self.action_pins)
        state = 0
        for bit in bits:
            state |= 1 << bit
        return state

    def _send_signal(self, port, state):
        if self.use_hid:
            signal_val = state & 0xFF
            # when using the split port-C, only update the specified nibble
            if port == self.layout.PortEnum.FIRSTPORTCL:
                self.port_cl_val = signal_val & 0x0F
                signal_val = signal_val | (self.port_ch_val << 4)
            elif port == self.layout.PortEnum.FIRSTPORTCH:
                self.port_ch_val = signal_val & 0x0F
                signal_val = (signal_val << 4) | self.port_cl_val

            signal_data = bytearray([self._ReportCommand.DOUT, port, signal_val])
            self.backend.write_sync(signal_data)
        else:
            self.dio_device.d_out(port, state)

    def _set_pins(self, port, pins, value, force=True):
        bits = self._get_bits_for_pins(port, pins)
        updated_state = self.default_port_state

        # construct the data mask for the new set of pins
        # d_in should only be used for debug; ports are configured for OUTPUT,
        # so reads are slow, up to ~15ms.
        # data_mask = self.dio_device.d_in(ports[0])

        for bit in bits:
            if value == self.signal_active:
                updated_state &= ~(1 << bit)
            else:
                updated_state |= 1 << bit

        pins_low = self.port_state & ~updated_state
        pins_high = ~self.port_state & updated_state
        port_state = self.port_state & ~pins_low | pins_high

        if force or self.port_state != port_state:
            self.port_state = port_state
            self._send_signal(port, self.port_state)

        # logger.debug(f"port_state = {bin(self.port_state):08}")

    def _get_bits_for_pins(self, port_enum, pins):
        return [self.layout.get_bit_for_pin(port_enum, p) for p in pins]

    def apply_action(self, action, state):
        pins = self.pins_from_action_map.get(Action(action), (-1,))
        # handle NOOP
        if pins == (-1,):
            pins = self.action_pins
            state = 0

        # To simulate press, pull it low; to simulate release pull it high.
        signal = self.signal_active if state else self.signal_inactive
        self._set_pins(self.port, pins, signal)

    def get_pins(self) -> list[int]:
        return list(self.action_pins)
