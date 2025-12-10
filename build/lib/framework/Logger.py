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

import logging
import os
import re
from enum import Enum
from threading import local

# thread-local storage for frame count
_frame_count_storage = local()


def get_frame_count():
    # -1 as a sentinel value for unset frame_count
    return getattr(_frame_count_storage, 'frame_count', -1)


def set_frame_count(frame_count):
    _frame_count_storage.frame_count = frame_count


class AnsiColor(str, Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        'DEBUG': AnsiColor.CYAN,
        'INFO': AnsiColor.GREEN,
        'WARNING': AnsiColor.YELLOW,
        'ERROR': AnsiColor.RED,
        'CRITICAL': AnsiColor.RED,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelname, AnsiColor.RESET)
        record.colored_levelname = f"{color.value}{record.levelname}{AnsiColor.RESET.value}"
        if not hasattr(record, 'frame_count'):
            record.frame_count = get_frame_count()
        return super().format(record)


class NoColorFormatter(logging.Formatter):
    ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

    def format(self, record):
        if not hasattr(record, 'frame_count'):
            record.frame_count = get_frame_count()
        original = super().format(record)
        return self.ANSI_ESCAPE.sub('', original)


class FrameCountAdapter(logging.LoggerAdapter):
    def __init__(self, logger):
        super().__init__(logger, {})

    def process(self, msg, kwargs):
        frame_count = get_frame_count()
        kwargs.setdefault('extra', {})['frame_count'] = frame_count
        return msg, kwargs


def create_logger():
    logger = logging.getLogger("frame_logger")
    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    formatter = ColorFormatter('[%(asctime)s]: frame:%(frame_count)s: %(colored_levelname)s %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)
    return FrameCountAdapter(logger)


# global logger
logger = create_logger()


# dynamically add the file handler when we know the experiment directory
# alternatively, we can create the file handler at init and move the log
# file to the experiment dir on exit.
def add_file_handler_to_logger(log_file_path):
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        file_handler = logging.FileHandler(log_file_path)
        formatter = NoColorFormatter(
            '[%(asctime)s]: %(process)d:%(thread)d: frame:%(frame_count)s: %(levelname)s %(message)s'
        )
        file_handler.setFormatter(formatter)
        logging.getLogger("frame_logger").addHandler(file_handler)
