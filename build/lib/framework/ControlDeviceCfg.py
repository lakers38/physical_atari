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

from framework.Logger import logger


def create_control_device_from_cfg(**kwargs):
    model_name = kwargs.pop("model_name", None)
    if model_name == "MCC USB-1024LS":
        import framework.MCCDAQDevice as DAQDevice

        logger.info(f"Initializing {model_name}")
        return DAQDevice.DAQDevice(model_name, **kwargs)
    elif model_name == "QinHeng Electronics USB Single Serial":
        logger.info(f"Initializing {model_name} in position mode")
        import framework.RoboTroller as RoboTroller

        return RoboTroller.RoboTroller(model_name, **kwargs)
    else:
        raise ValueError(f"Joystick model not supported: {model_name}")
