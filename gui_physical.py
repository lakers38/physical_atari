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

# real-time interactive gui
import copy
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import zlib
from abc import ABC, abstractmethod
from enum import Enum
from multiprocessing import Lock, Value, shared_memory

import cv2
import dearpygui.dearpygui as dpg
import msgpack
import numpy as np
import psutil
import Xlib.display

from framework import ScoreDetectorConfig
from framework.Actions import Action
from framework.CameraUtils import get_controls, get_index_from_model_name
from framework.Logger import logger, set_frame_count


class SharedFrameData:
    def __init__(self, obs_dim=(160, 210, 3), data_size=1024, lock=None):
        self.target_cam_dim = (640, 480, 3)
        self.shm_frame = shared_memory.SharedMemory(
            create=True,
            size=(
                self.target_cam_dim[0] * self.target_cam_dim[1] * self.target_cam_dim[2] * np.dtype(np.uint8).itemsize
            ),
        )
        self.frame = np.ndarray(
            (self.target_cam_dim[1], self.target_cam_dim[0], self.target_cam_dim[2]),
            dtype=np.uint8,
            buffer=self.shm_frame.buf,
        )

        self.shm_obs = shared_memory.SharedMemory(
            create=True, size=(obs_dim[0] * obs_dim[1] * obs_dim[2] * np.dtype(np.uint8).itemsize)
        )
        self.obs = np.ndarray((obs_dim[1], obs_dim[0], obs_dim[2]), dtype=np.uint8, buffer=self.shm_obs.buf)

        self.shm_data = shared_memory.SharedMemory(create=True, size=data_size)
        self.data_buffer = np.ndarray(data_size, dtype=np.uint8, buffer=self.shm_data.buf)

        self.lock = lock or Lock()
        self.data_available = Value('i', 0)  # len of payload when > 0

    def shutdown(self):
        self.shm_data.unlink()
        self.shm_frame.unlink()
        self.shm_obs.unlink()
        self.data_available.value = 0

    def close(self):
        self.shm_data.close()
        self.shm_frame.close()
        self.shm_obs.close()

    def flatten_data(self, data_dict):
        # for serialization, tuples and numpy arrays need to be converted to lists
        def flatten(value):
            if isinstance(value, np.ndarray):
                return [flatten(v) if isinstance(v, np.ndarray) else v for v in value.tolist()]
            elif isinstance(value, tuple):
                return [flatten(v) for v in value]
            elif isinstance(value, dict):
                return {flatten_key(k): flatten(v) for k, v in value.items()}
            else:
                return value

        def flatten_key(key):
            # validate key is in an expected format
            try:
                hash(key)
                return key
            except TypeError:
                logger.warning(f"invalid key: {key}")
                return str(key)

        data_flat = {}
        for key, value in data_dict.items():
            data_flat[flatten_key(key)] = flatten(value)
        return data_flat

    def write_to_shmem(self, cam_frame, obs_frame, data_dict):
        if cam_frame is not None:
            # TODO: if camera frame is yuyv, handle conversion
            if cam_frame.shape != self.frame.shape:
                cam_frame = cv2.resize(
                    cam_frame, (self.frame.shape[1], self.frame.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            np.copyto(self.frame, cam_frame)

        if obs_frame is not None:
            if obs_frame.shape != self.obs.shape:
                obs_frame = cv2.resize(
                    obs_frame, (self.obs.shape[1], self.obs.shape[0]), interpolation=cv2.INTER_LINEAR
                )
            np.copyto(self.obs, obs_frame)

        if data_dict is not None:
            data_flat = self.flatten_data(data_dict)
            try:
                data_bytes = msgpack.packb(data_flat)
            except Exception as e:
                logger.warning(f"write_to_shmem: Failed to pack data: {e}")
                return

            crc = zlib.crc32(data_bytes)
            data_len = len(data_bytes)
            total_len = data_len + 4  # 4 bytes for crc

            if total_len > len(self.data_buffer):
                logger.warning(
                    f"write_to_shmem: Buffer overflow. Required: {total_len}, available: {len(self.data_buffer)}"
                )
                return

            with self.lock:
                # write data and checksum
                self.data_buffer[:data_len] = np.frombuffer(data_bytes, dtype=np.uint8)
                np.frombuffer(self.data_buffer, dtype=np.uint32, count=1, offset=data_len)[0] = crc
                self.data_available.value = data_len  # NOTE: only data_len, crc excluded

    def read_from_shmem(self, max_retries=2, retry_delay=0.001):
        for attempt in range(max_retries + 1):
            with self.lock:
                data_len = self.data_available.value

                if data_len == 0 or data_len + 4 > len(self.data_buffer):
                    return None, None, None

                try:
                    data_bytes = self.data_buffer[:data_len].copy()
                    stored_crc = np.frombuffer(self.data_buffer, dtype=np.uint32, count=1, offset=data_len)[0]
                except Exception as e:
                    logger.warning(f"read_from_shmem: Exception accessing buffer: {e}")
                    return None, None, None

            # check crc
            actual_crc = zlib.crc32(data_bytes)
            if stored_crc == actual_crc:
                break
            else:
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    logger.warning(
                        f"read_from_shmem: CRC mismatch [stored={stored_crc}, computed={actual_crc}] after retries, skipping frame."
                    )
                    return None, None, None

        try:
            data = msgpack.unpackb(data_bytes, strict_map_key=False)
        except msgpack.exceptions.ExtraData as e:
            logger.warning(f"read_from_shmem: Failed to unpack data: {e}")
            # logger.debug(data_bytes)
            return None, None, None
        except Exception as e:
            logger.warning(f"read_from_shmem: Failed to unpack data: {e}")
            # logger.debug(data_bytes)
            return None, None, None

        with self.lock:
            self.data_available.value = 0

        return self.frame.copy(), self.obs.copy(), data


class BoundingBox:
    def __init__(self, x, y, w, h, num_digits, dims=(0, 0)):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.num_digits = num_digits

        # assumes atari dims as reference
        self.reference_width = 160
        self.reference_height = 210

        self.current_width = self.reference_width
        self.current_height = self.reference_height

        if dims[0] != 0 and dims[1] != 0:
            self.adjust_bounds_for_dims(dims[0], dims[1])

    def __str__(self):
        return f"bb=({self.x}, {self.y}, {self.x + self.w}, {self.y + self.h})"

    # NOTE: scales bounding box from top-left; may not preserve center alignment
    def adjust_bounds_for_dims(self, width, height):
        if width == self.current_width and height == self.current_height:
            return

        scale_x = width / self.current_width
        scale_y = height / self.current_height

        self.x = int(self.x * scale_x)
        self.y = int(self.y * scale_y)
        self.w = int(self.w * scale_x)
        self.h = int(self.h * scale_y)

        self.current_width = width
        self.current_height = height


def get_screen_dimensions():
    # get the useable screen area (screen size minus dock and top bar)
    try:
        output = subprocess.check_output(['xprop', '-root', '_NET_WORKAREA'], encoding='utf-8')
        match = re.search(r'_NET_WORKAREA\(CARDINAL\) = (\d+), (\d+), (\d+), (\d+)', output)
        if match:
            _, _, width, height = map(int, match.groups())
            return width, height
    except Exception as e:
        print(f"Failed to get work area from xprop: {e}")

    display = Xlib.display.Display()
    screen = display.screen()
    return (screen.width_in_pixels, screen.height_in_pixels)


# Provides similar data as the training program for use in validation and editing
class GuiRenderContent:
    def __init__(
        self,
        shared_data,
        config_changed_queue,
        game_config,
        camera_config,
        detection_config,
        device_config,
        score_detector_type,
        obs_dims=(160, 210, 3),
    ):
        from framework.ScoreDetector import ScoreDetector
        from framework.ScreenDetector import ScreenDetector
        from framework.ScreenDetectorFixed import ScreenDetectorFixed

        self.ActionClass = Action

        self.shared_data = shared_data
        self.config_change_queue = config_changed_queue
        self.obs_dims = obs_dims

        with open(camera_config) as cf:
            camera_data = cf.read()

        camera_data = json.loads(camera_data)
        self.camera_name = camera_data["model_name"]
        self.camera_config_data = camera_data["camera_config"]

        self.detection_config = detection_config
        with open(self.detection_config) as df:
            detection_data = df.read()

        detection_data = json.loads(detection_data)
        self.detection_method_name = detection_data["name"]
        if self.detection_method_name == "fixed":
            self.ScreenDetectorClass = ScreenDetectorFixed
        else:
            self.ScreenDetectorClass = ScreenDetector

        self.score_detector_type = score_detector_type
        score_model_type = ScoreDetectorConfig.get_model_type(self.score_detector_type)
        if score_model_type == "network":
            self.ScoreDetectorClass = ScoreDetector
        else:
            raise ValueError(f"Invalid Score detection type {self.score_detector_type}.")

        self.game_config = game_config
        with open(self.game_config) as gf:
            game_config_data = gf.read()
        game_config_data = json.loads(game_config_data)["game_config"]

        self.game_name = game_config_data["name"]
        self.total_lives = game_config_data["lives"]

        with open(device_config) as df:
            device_data = df.read()

        self.device_data = json.loads(device_data)

    def initialize(self):
        from framework.CameraDevice_v4l2 import CameraDevice_v4l2 as CameraDevice
        from framework.ControlDeviceCfg import create_control_device_from_cfg
        from framework.Keyboard import Keyboard

        logger.debug("GuiRenderContent: initialize")

        self.camera = None

        # verify the camera has read access
        retries = 2
        for retry in range(retries):
            self.camera = CameraDevice(self.camera_name, **self.camera_config_data)
            assert self.camera is not None
            if self.camera.validate():
                break

            logger.warning(f"Failed to validate camera: {retry}/{retries} retries...")
            self.camera.shutdown()
            self.camera = None

        if self.camera is None:
            raise ValueError("Failed to read frames from camera.")
        else:
            logger.debug("Camera validated.")

        device = create_control_device_from_cfg(**self.device_data)
        self.keyboard = Keyboard(device)
        self.keyboard.set_input_focus(False)

        # NOTE: set by the update thread; accessed by main thread; we may need to guard with mutex
        self.current_screen_rect = None

        self.running = False
        self.thread = None

        self.start()

    def is_running(self):
        return self.running

    def shutdown(self):
        self.stop()

        if self.camera is not None:
            self.camera.shutdown()
            self.camera = None

        if self.keyboard is not None:
            self.keyboard.shutdown()
            self.keyboard = None

        self.shared_data = None

        logger.debug("GuiRenderContent: shutdown")

    def start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._process_frames, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return

        self.running = False
        self.thread.join()
        self.thread = None

    def has_valid_screen_rect(self):
        return self.current_screen_rect is not None

    def set_input_focus(self, focus: bool):
        if self.keyboard is not None:
            self.keyboard.set_input_focus(focus)

    def _rectify_frame(self, frame, screen_rect, target_width, target_height):
        target_rect = np.float32([(0, 0), (target_width, 0), (target_width, target_height), (0, target_height)])
        transform = cv2.getPerspectiveTransform(screen_rect, target_rect)
        rect_frame = cv2.warpPerspective(frame, transform, (target_width, target_height), flags=cv2.INTER_LINEAR)
        if len(rect_frame.shape) == 2:
            rect_frame = np.expand_dims(rect_frame, axis=-1)
        return rect_frame

    def _process_frames(self):
        target_fps = 60
        target_frame_time = 1.0 / target_fps
        last_frame_time = time.time()
        fps = 0.0

        self.screen_detector = None
        self.score_detector = None

        # Some games do not display the total lives. When detecting number of lives, we need to increment by the
        # number of lives not shown. Of the supported games, these games do not display the number of total lives:
        # ['ms_pacman', 'centipede', "up_n_down", 'qbert', 'krull']
        self.lives_increment = 0

        curr_action = self.ActionClass.NOOP

        while self.running:
            # check for configuration changes and reload the necessary
            # components.
            config_change_type = ConfigureType.CONFIGURE_TYPE_NONE
            if not self.config_change_queue.empty():
                config_change_type = self.config_change_queue.get_nowait()

            if self.screen_detector is None or config_change_type == ConfigureType.CONFIGURE_TYPE_SCREEN_DETECTION:
                logger.debug("Reloading Screen Detection")
                if self.screen_detector is not None:
                    self.screen_detector.shutdown()

                with open(self.detection_config) as df:
                    detection_data = df.read()

                detection_data = json.loads(detection_data)
                method_name = detection_data["name"]
                if method_name == "fixed":
                    self.screen_detector = self.ScreenDetectorClass(
                        self.detection_method_name, **detection_data["detection_config"]
                    )
                else:
                    self.screen_detector = self.ScreenDetectorClass(
                        self.detection_method_name, detection_data["corners"], **detection_data["detection_config"]
                    )

                assert self.screen_detector is not None

            if self.score_detector is None or config_change_type == ConfigureType.CONFIGURE_TYPE_GAME_SCORE_BOXES:
                logger.debug("Reloading Score Detector")

                # load custom user offsets
                score_offsets = {"offset_x": 0, "offset_y": 0}
                lives_offsets = {"offset_x": 0, "offset_y": 0}
                user_offset_file = "configs/.user_offsets.json"
                if os.path.exists(user_offset_file):
                    with open(user_offset_file) as f:
                        offsets = json.load(f)
                        if self.game_name in offsets:
                            game_offsets = offsets[self.game_name]
                            if "score_crop_offset" in game_offsets:
                                score_offsets = game_offsets["score_crop_offset"]
                            if "lives_crop_offset" in game_offsets:
                                lives_offsets = game_offsets["lives_crop_offset"]

                with open(self.game_config) as gf:
                    game_config_data = gf.read()

                score_model_type = ScoreDetectorConfig.get_model_type(self.score_detector_type)
                if score_model_type == "network":
                    score_config_data = json.loads(game_config_data)["score_config"]
                    self.score_detector = self.ScoreDetectorClass(
                        self.game_name,
                        self.score_detector_type,
                        self.total_lives,
                        **score_config_data,
                        score_offsets=score_offsets,
                        lives_offsets=lives_offsets,
                        device='cuda:0',
                    )
                else:
                    raise ValueError(f"Unsupported score detector type={score_model_type}")

                assert self.score_detector is not None

                if self.score_detector.lives_crop_info:
                    num_displayed_lives = self.score_detector.lives_crop_info.num_digits
                    self.lives_increment = max(0, self.total_lives - num_displayed_lives)

            _, action = self.keyboard.update()
            if action is not self.ActionClass.NOOP:
                curr_action = action

            frame_data = self.camera.get_frame()
            frame = frame_data["frame"]
            frame_g = self.camera.convert_to_grayscale(frame)
            frame_num = frame_data["frame_number"]

            try:
                self.current_screen_rect, last_detected_tags = self.screen_detector.get_screen_rect_info(frame_g)
                detected_tag_ids = sorted(last_detected_tags.keys()) if last_detected_tags else []
                if detected_tag_ids:
                    logger.info(f"AprilTags detected: {detected_tag_ids} ({'VALID' if len(detected_tag_ids) == 4 else f'need {4-len(detected_tag_ids)} more'})")
            except Exception as e:
                logger.error(f"Screen detection error: {e}", exc_info=True)
                self.current_screen_rect = None
                last_detected_tags = {}

            frame = self.camera.convert_to_rgb(frame)

            target_width = self.obs_dims[0] * 2
            target_height = self.obs_dims[1] * 2
            frame_rect = (
                None
                if self.current_screen_rect is None
                else self._rectify_frame(frame, self.current_screen_rect, target_width, target_height)
            )

            score = 0
            lives = self.total_lives

            if frame_rect is not None:
                (score, lives) = self.score_detector.get_score_and_lives(frame_rect)
                if lives is not None:
                    lives += self.lives_increment

            if frame_rect is not None and (
                frame_rect.shape[0] != self.obs_dims[1] or frame_rect.shape[1] != self.obs_dims[0]
            ):
                frame_rect = cv2.resize(
                    frame_rect, (self.obs_dims[0], self.obs_dims[1]), interpolation=cv2.INTER_LINEAR
                )

            frame_data = {
                "frame": frame_num,
                "tags": last_detected_tags,
                "lives": lives if lives is not None else self.total_lives,
                "total_lives": self.total_lives,
                "score": score,
                "action": curr_action.name,
                "fps": fps,
            }

            self.shared_data.write_to_shmem(frame, frame_rect, frame_data)

            curr_time = time.time()
            target_time = last_frame_time + target_frame_time
            delta_time = curr_time - last_frame_time
            sleep_time = target_time - curr_time
            if sleep_time < 0.0:
                logger.warning(f"frame took too long: dt={delta_time*1000.0:.2f}ms > target ft={target_frame_time*1000.0:.2f}ms")
                sleep_time = 0.0

            time.sleep(sleep_time)

            last_frame_time = time.time()

        if self.screen_detector is not None:
            self.screen_detector.shutdown()
            self.screen_detector = None

        if self.score_detector is not None:
            self.score_detector = None


def run_cmd(cmd):
    try:
        output = subprocess.check_output(cmd, encoding='utf-8', stderr=subprocess.DEVNULL)
        return 0, output
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output
    except Exception as e:
        return -1, str(e)


def _severity_rank(level):
    order = {"OK": 0, "INFO": 1, "WARNING": 2, "CRITICAL": 3}
    return order.get(level, 0)


def _safe_float(val, default=0.0):
    try:
        val = val.strip().strip('[]')
        return float(val)
    except Exception:
        return default


def _is_flag_active(line):
    line = line.lower()
    return 'active' in line and 'not active' not in line


severity_map = {
    'debug': logger.debug,
    'info': logger.info,
    'warning': logger.warning,
    'error': logger.error,
    'critical': logger.critical,
}


def log_message(severity, message):
    log_func = severity_map.get(severity.lower(), logger.info)
    log_func(message)


class HealthMonitorThread:
    def __init__(self, gpu_id, report_interval_in_sec=15.0):
        self.gpu_id = gpu_id

        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.shutdown_cond = threading.Condition()

        self.report_interval_in_sec = report_interval_in_sec
        self.last_gpu_stats = None
        self.last_sys_stats = None
        self._last_sw_power_cap = None

        self.cpu_history_len = 3
        self.cpu_freq_history = []
        self.cpu_util_history = []

        self.start()

    def shutdown(self):
        self.stop()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_stats, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        with self.shutdown_cond:
            self.shutdown_cond.notify()
        self.thread.join()

    def get_gpu_stats(self):
        with self.lock:
            return self.last_gpu_stats

    def get_sys_stats(self):
        with self.lock:
            return self.last_sys_stats

    def _check_battery_status(self):
        try:
            with open("/sys/class/power_supply/BAT0/status") as f:
                status = f.read().strip()
            with open("/sys/class/power_supply/BAT0/capacity") as f:
                percent = f.read().strip()
            return f"{status}, {percent}%"
        except Exception as e:
            return f"Unavailable ({e})"

    def _check_usb_errors(self):
        try:
            output = subprocess.check_output(
                ['journalctl', '--since', '1 minute ago', '-k'], encoding='utf-8', stderr=subprocess.DEVNULL
            )
            error_keywords = ['usb', 'reset', 'error', 'timeout', 'device descriptor', 'failed']
            for line in output.lower().splitlines():
                if any(keyword in line for keyword in error_keywords):
                    return True
        except Exception as e:
            logger.debug(f"_check_usb_errors: exception {e}")
        return False

    def _get_power_status(self):
        power_info = {}
        try:
            output = subprocess.check_output(['tlp-stat', '-s'], encoding='utf-8', stderr=subprocess.DEVNULL)
            for line in output.splitlines():
                line_lower = line.lower()
                if 'power source' in line_lower and '=' in line:
                    parts = line.split("=", 1)
                    if len(parts) > 1:
                        power_info['power_source'] = parts[1].strip()
                elif 'tlp power save' in line_lower and '=' in line:
                    parts = line.split("=", 1)
                    if len(parts) > 1:
                        power_info['tlp_power_save'] = parts[1].strip().lower() == 'enabled'
        except FileNotFoundError:
            logger.debug("tlp-stat not found; skipping power status check.")
            power_info['power_source'] = 'unknown'
            power_info['tlp_power_save'] = None
        except Exception as e:
            logger.debug(f"_get_power_status: exception {e}")
            power_info['power_source'] = 'unknown'
            power_info['tlp_power_save'] = None

        return power_info

    def _parse_gpu_performance_reasons(self):
        flags = {
            'gpu_idle': False,
            'gpu_sw_power_cap': False,
            'gpu_hw_slowdown': False,
            'gpu_hw_thermal_slowdown': False,
            'gpu_hw_power_brake': False,
            'gpu_sw_thermal_slowdown': False,
            'gpu_sync_boost': False,
            'gpu_display_clock_setting': False,
            'gpu_app_clocks_limited': False,
        }
        try:
            output = subprocess.check_output(['nvidia-smi', '-q', '-d', 'PERFORMANCE'], encoding='utf-8')
            for line in output.splitlines():
                line = line.strip().lower()
                if "idle" in line:
                    flags['gpu_idle'] = _is_flag_active(line)
                elif "applications clocks setting" in line:
                    flags['gpu_app_clocks_limited'] = _is_flag_active(line)
                elif "sw power cap" in line:
                    flags['gpu_sw_power_cap'] = _is_flag_active(line)
                elif "hw slowdown" in line and "thermal" not in line and "power" not in line:
                    flags['gpu_hw_slowdown'] = _is_flag_active(line)
                elif "hw thermal slowdown" in line:
                    flags['gpu_hw_thermal_slowdown'] = _is_flag_active(line)
                elif "hw power brake slowdown" in line:
                    flags['gpu_hw_power_brake'] = _is_flag_active(line)
                elif "sw thermal slowdown" in line:
                    flags['gpu_sw_thermal_slowdown'] = _is_flag_active(line)
                elif "sync boost" in line:
                    flags['gpu_sync_boost'] = _is_flag_active(line)
                elif "display clock setting" in line:
                    flags['gpu_display_clock_setting'] = _is_flag_active(line)
        except Exception as e:
            logger.debug(f"_parse_gpu_performance_reasons exception: {e}")
        return flags

    def _get_gpu_stats(self, gpu_id=0):
        info = {}
        try:
            cmd_list = [
                'nvidia-smi',
                '--query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,pstate,clocks.gr,clocks.mem,power.draw,power.limit',
                '--format=csv,noheader,nounits',
            ]
            status, output = run_cmd(cmd_list)
            if status == 0:
                val_lines = output.strip().splitlines()
                if len(val_lines) <= gpu_id:
                    logger.warning("Failed to get gpu stats: insufficient output.")
                    return info

                vals = val_lines[gpu_id].replace(" ", "").split(',')
                (
                    temp,
                    util,
                    util_mem,
                    total_mem,
                    free_mem,
                    used_mem,
                    pstate,
                    clock_gr,
                    clock_mem,
                    power_draw,
                    power_limit,
                ) = vals

                info.update(
                    {
                        'gpu_temp': _safe_float(temp),
                        'gpu_util': _safe_float(util),
                        'gpu_util_mem': _safe_float(util_mem),
                        'gpu_total_mem': _safe_float(total_mem),
                        'gpu_free_mem': _safe_float(free_mem),
                        'gpu_used_mem': _safe_float(used_mem),
                        'gpu_pstate': pstate,
                        'gpu_clock_graphics_mhz': int(clock_gr) if clock_gr.isdigit() else 0,
                        'gpu_clock_memory_mhz': int(clock_mem) if clock_mem.isdigit() else 0,
                        'gpu_power_draw_watts': _safe_float(power_draw),
                        'gpu_power_limit_watts': _safe_float(power_limit),
                    }
                )
            else:
                logger.warning(f"get_gpu_stats: failed with status={status} error={output}")

            info.update(self._parse_gpu_performance_reasons())
        except Exception as e:
            logger.warning(f"_get_gpu_stats exception: {e}")
        return info

    def _check_cpu_freq_throttling(self, threshold_pct=80):
        throttled = False
        details = {}
        try:
            freqs = psutil.cpu_freq(percpu=True)
            utils = psutil.cpu_percent(percpu=True)
            freq_pcts = []
            active_cores = 0

            for i, (freq, util) in enumerate(zip(freqs, utils)):
                current = freq.current or 0
                max_freq = freq.max or 1  # avoid div by zero
                pct = (current / max_freq) * 100 if max_freq > 0 else 0
                details[f'cpu{i}_freq_pct'] = pct
                details[f'cpu{i}_freq_mhz'] = current
                if util > 10:  # consider active if util > 10%
                    freq_pcts.append(pct)
                    active_cores += 1

            avg_freq_pct = sum(freq_pcts) / active_cores if active_cores else 0
            avg_util = sum(utils) / len(utils) if utils else 0

            details['avg_cpu_freq_pct'] = avg_freq_pct
            details['avg_cpu_util_pct'] = avg_util
            self.cpu_freq_history.append(avg_freq_pct)
            self.cpu_util_history.append(avg_util)
            if len(self.cpu_freq_history) > self.cpu_history_len:
                self.cpu_freq_history.pop(0)
                self.cpu_util_history.pop(0)

            if len(self.cpu_freq_history) >= 2:
                busy = all(util > 30 for util in self.cpu_util_history[-2:])
                low_freq = all(freq < threshold_pct for freq in self.cpu_freq_history[-2:])
                if busy and low_freq:
                    throttled = True

            details['threshold_pct'] = threshold_pct
        except Exception as e:
            details['error'] = str(e)
        return throttled, details

    def _get_system_stats(self):
        info = {}
        try:
            vm = psutil.virtual_memory()
            info['ram'] = vm.percent
            info['total_mem'] = vm.total / (1024 * 1024)
            info['cpu_util'] = psutil.cpu_percent()
            info['disk'] = psutil.disk_usage('/').percent

            temps = psutil.sensors_temperatures()
            coretemps = temps.get('coretemp')
            if coretemps:
                info['cpu_temp'] = temps['coretemp'][0].current

            info.update(self._get_power_status())
            info['usb_errors_detected'] = self._check_usb_errors()

            throttled, freq_details = self._check_cpu_freq_throttling()
            info['cpu_freq_throttled'] = throttled
            info.update(freq_details)
        except Exception as e:
            logger.debug(f"get_system_stats: exception {e}")
        return info

    def _process_stats(self):
        while self.running:
            gpu_info = self._get_gpu_stats(self.gpu_id)
            sys_info = self._get_system_stats()

            with self.lock:
                self.last_gpu_stats = gpu_info
                self.last_sys_stats = sys_info

            with self.shutdown_cond:
                self.shutdown_cond.wait(self.report_interval_in_sec)

    def _check_power_cap_transition(self, gpu_info, sys_info):
        sw_cap = gpu_info.get('gpu_sw_power_cap', False)
        if self._last_sw_power_cap is None:
            self._last_sw_power_cap = sw_cap
            return

        gpu_util = gpu_info.get('gpu_util', 0)
        power_draw = gpu_info.get('gpu_power_draw_watts', 'N/A')
        pstate = gpu_info.get('gpu_pstate', 'N/A')
        clocks = gpu_info.get('gpu_clock_graphics_mhz')
        power_source = sys_info.get('power_source', 'Unknown')
        # battery_status = self._check_battery_status()
        if sw_cap != self._last_sw_power_cap:
            status = "ON" if sw_cap else "OFF"
            logger.debug(
                f"THROTTLE CHANGE={status} | gpu_util={gpu_util}% | power_draw={power_draw}W | p-State={pstate} | clocks={clocks}MHz power_source={power_source}"
            )

        self._last_sw_power_cap = sw_cap

    def _format_cpu_freq_stats(self):
        try:
            freqs = psutil.cpu_freq(percpu=True)
            utils = psutil.cpu_percent(percpu=True)
            cur_freqs = [f.current for f, u in zip(freqs, utils) if u > 10]
            min_freqs = [f.min for f in freqs if f.min]
            max_freqs = [f.max for f in freqs if f.max]

            if not cur_freqs:
                return "CPU Freq: No active cores"

            avg_cur = sum(cur_freqs) / len(cur_freqs)
            min_freq = min(min_freqs)
            max_freq = max(max_freqs)
            avg_util = sum(utils) / len(utils)

            return (
                f"CPU Freq (MHz): cur={avg_cur:.0f} min={min_freq:.0f} max={max_freq:.0f} | "
                f"CPU Util: {avg_util:.0f}%"
            )
        except Exception as e:
            return f"CPU Freq: Error ({e})"

    def get_health_status(self):
        with self.lock:
            gpu_stats = self.last_gpu_stats or {}
            sys_stats = self.last_sys_stats or {}

        # check for changes to power throttling
        self._check_power_cap_transition(gpu_stats, sys_stats)

        messages = []
        severity = "OK"

        if not gpu_stats or not sys_stats:
            return ("No Stats Available", "INFO")

        if sys_stats.get('usb_errors_detected'):
            messages.append("USB Errors")
            severity = "CRITICAL"

        if sys_stats.get('power_source') == 'Battery':
            messages.append("Power Saving Mode")
            severity = max(severity, "WARNING", key=_severity_rank)

        if sys_stats.get('tlp_power_save'):
            messages.append("TLP Enabled")
            severity = max(severity, "INFO", key=_severity_rank)

        throttle_reasons = []
        if gpu_stats.get('gpu_hw_thermal_slowdown'):
            throttle_reasons.append("thermal")
            severity = max(severity, "CRITICAL", key=_severity_rank)
        if gpu_stats.get('gpu_hw_power_brake'):
            throttle_reasons.append("power brake")
            severity = max(severity, "CRITICAL", key=_severity_rank)
        if gpu_stats.get('gpu_hw_slowdown'):
            throttle_reasons.append("hardware")
            severity = max(severity, "CRITICAL", key=_severity_rank)
        if gpu_stats.get('gpu_sw_power_cap'):
            throttle_reasons.append("SW power cap")
            severity = max(severity, "WARNING", key=_severity_rank)
        if gpu_stats.get('gpu_sw_thermal_slowdown'):
            throttle_reasons.append("SW thermal")
            severity = max(severity, "WARNING", key=_severity_rank)
        if gpu_stats.get('gpu_display_clock_setting'):
            throttle_reasons.append("display")
            severity = max(severity, "INFO", key=_severity_rank)

        if throttle_reasons:
            # print(gpu)
            messages.append(f"GPU Throttled ({', '.join(throttle_reasons)})")

        gpu_util = gpu_stats.get('gpu_util', 0)
        pstate = gpu_stats.get('gpu_pstate', 'P8')
        if gpu_util > 50 and pstate not in ('P0', 'P2'):
            messages.append(f"Performance Degraded (P-state {pstate})")
            severity = max(severity, "WARNING", key=_severity_rank)

        if sys_stats.get('cpu_freq_throttled'):
            messages.append("CPU Frequency Throttled")
            severity = max(severity, "WARNING", key=_severity_rank)

        if not messages:
            messages.append("OK")

        # print general stats along with status
        power_draw = gpu_stats.get('gpu_power_draw_watts', 'N/A')
        clocks = gpu_stats.get('gpu_clock_graphics_mhz')
        # power_source = sys.get('power_source', 'Unknown')

        cpu_freq_summary = self._format_cpu_freq_stats()

        stats_line = (
            f"GPU: util={gpu_util:.0f}% pstate={pstate} clock={clocks}MHz power={power_draw}W | " f"{cpu_freq_summary}"
        )
        messages.append(stats_line)

        return (", ".join(messages), severity)


class ConfigureType(Enum):
    CONFIGURE_TYPE_NONE = 0
    CONFIGURE_TYPE_CAMERA_CONTROLS = 1
    CONFIGURE_TYPE_SCREEN_DETECTION = 2
    CONFIGURE_TYPE_GAME_SCORE_BOXES = 3


class ConfigureState(Enum):
    CONFIGURE_STARTED = "configure_started"
    VALIDATION_FAILED = "validation_failed"
    TRAIN_STARTED = "train_started"
    TRAIN_EXITED = "train_exited"


class ValidationFailureReason(Enum):
    VALIDATION_FAILURE_NONE = "none"
    VALIDATION_FAILURE_SCREEN_RECT = "invalid_screen_rect"


# ---------------------------------------
# Editors
# ---------------------------------------


class EditorBase(ABC):
    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def destroy(self):
        pass

    @abstractmethod
    def enable(self, enabled: bool):
        pass

    @abstractmethod
    def draw(self):
        pass

    @abstractmethod
    def key_press_handler(self, sender, app_data):
        pass

    @abstractmethod
    def mouse_click_handler(self, sender, app_data):
        pass

    @abstractmethod
    def mouse_release_handler(self, sender, app_data):
        pass

    @abstractmethod
    def mouse_drag_handler(self, sender, app_data):
        pass

    @abstractmethod
    def focus_changed_handler(self, sender, app_data):
        pass


class CameraControlEditor(EditorBase):
    def __init__(self, config_change_queue, device_idx, focus_window, draw_list):
        self.config_change_queue = config_change_queue
        self.device_idx = device_idx
        self.focus_window = focus_window
        self.control_tags = []
        self.enabled = True

    def create(self, camera_config):
        self.camera_config = camera_config

        ctrls = get_controls(self.device_idx)

        # Header
        dpg.add_text("Camera Controls (from v4l2-ctl)", color=(255, 255, 100))
        dpg.add_separator()

        for name, ctrl in ctrls.items():
            ctrl_type = ctrl.get('type', 'unknown')
            has_range = ('min' in ctrl and ctrl['min'] is not None) and ('max' in ctrl and ctrl['max'] is not None)

            with dpg.group(horizontal=False):
                if ctrl_type == 'bool' or (not has_range and ctrl_type != 'menu'):
                    # Boolean control
                    tag_name = f"{ctrl['name']}_checkbox"
                    current_value = bool(ctrl.get('value', ctrl.get('default', 0)))
                    label = ctrl['name']
                    if 'desc' in ctrl:
                        label += f" ({ctrl['desc']})"

                    dpg.add_checkbox(
                        label=label,
                        default_value=current_value,
                        tag=tag_name,
                        callback=lambda sender, app_data, user_data: self._control_callback(
                            sender, app_data, user_data
                        ),
                        user_data={**ctrl},
                    )
                    dpg.bind_item_theme(tag_name, "checkbox_theme")
                    self.control_tags.append(tag_name)

                elif has_range:
                    # Numeric control with range
                    tag_name = f"{ctrl['name']}_slider"
                    current_value = ctrl.get('value', ctrl.get('default', ctrl['min']))
                    label = ctrl['name']

                    # Value display text
                    value_text_tag = f"{ctrl['name']}_value_text"
                    dpg.add_text(
                        f"{label}: {current_value} (min={ctrl['min']}, max={ctrl['max']})",
                        tag=value_text_tag,
                        color=(200, 200, 200)
                    )

                    dpg.add_slider_int(
                        label="",
                        min_value=ctrl['min'],
                        max_value=ctrl['max'],
                        default_value=current_value,
                        width=-1,
                        tag=tag_name,
                        callback=lambda sender, app_data, user_data: self._control_callback(
                            sender, app_data, user_data
                        ),
                        user_data={**ctrl, 'value_text_tag': value_text_tag},
                    )
                    dpg.bind_item_theme(tag_name, "slider_theme")
                    self.control_tags.append(tag_name)
                    self.control_tags.append(value_text_tag)

                else:
                    # Menu or unknown type
                    dpg.add_text(f"{ctrl['name']}: {ctrl.get('value', 'N/A')} [{ctrl_type}]", color=(150, 150, 150))

        dpg.add_spacer()
        dpg.add_button(
            label="Refresh",
            tag="cam_refresh_button",
            callback=lambda sender, app_data: self._control_refresh_callback(sender, app_data),
        )
        dpg.add_button(
            label="Save",
            tag="cam_save_button",
            callback=lambda sender, app_data: self._save_to_config_callback(sender, app_data),
        )

        dpg.bind_item_theme("cam_refresh_button", "button_theme")
        dpg.bind_item_theme("cam_save_button", "button_theme")

        self.control_tags.append("cam_refresh_button")
        self.control_tags.append("cam_save_button")

    def destroy(self):
        pass

    def enable(self, enabled: bool):
        self.enabled = enabled
        for tag in self.control_tags:
            # Skip text labels (they don't support enabled parameter)
            if tag.endswith('_value_text'):
                continue
            try:
                dpg.configure_item(tag, enabled=enabled)
            except (SystemError, KeyError):
                # Some items don't support enabled parameter, skip them
                pass

    def draw(self):
        # no custom draw required
        pass

    def key_press_handler(self, sender, app_data):
        return False

    def mouse_click_handler(self, sender, app_data):
        return False

    def mouse_release_handler(self, sender, app_data):
        return False

    def mouse_drag_handler(self, sender, app_data):
        return False

    def focus_changed_handler(self, sender, app_data):
        return False

    def _control_callback(self, sender, app_data, user_data):
        ctrl_name = user_data['name']
        ctrl_value = app_data

        if isinstance(ctrl_value, bool):
            ctrl_value = 1 if ctrl_value else 0

        # Update value display text if it exists
        if 'value_text_tag' in user_data:
            value_text_tag = user_data['value_text_tag']
            label = user_data['name']
            ctrl_min = user_data.get('min', 0)
            ctrl_max = user_data.get('max', 100)
            dpg.set_value(value_text_tag, f"{label}: {ctrl_value} (min={ctrl_min}, max={ctrl_max})")

        # Apply the control value
        subprocess.run(['v4l2-ctl', '-d', str(self.device_idx), '--set-ctrl', f'{ctrl_name}={ctrl_value}'])

    # Refresh values from v4l2 in case of external changes
    def _control_refresh_callback(self, sender, app_data):
        ctrls = get_controls(self.device_idx)
        # logger.debug(ctrls)
        for name, ctrl in ctrls.items():
            has_range = ('min' in ctrl and ctrl['min'] is not None) and ('max' in ctrl and ctrl['max'] is not None)

            if has_range:
                tag_name = f"{ctrl['name']}_slider"
                value_text_tag = f"{ctrl['name']}_value_text"

                # Update slider value
                ctrl_value = ctrl['value']
                try:
                    dpg.set_value(tag_name, ctrl_value)
                except:
                    pass

                # Update text label
                try:
                    label = ctrl['name']
                    dpg.set_value(value_text_tag, f"{label}: {ctrl_value} (min={ctrl['min']}, max={ctrl['max']})")
                except:
                    pass
            else:
                tag_name = f"{ctrl['name']}_checkbox"
                ctrl_value = ctrl['value']

                # Check the type of the current value to see if we need to convert to bool
                try:
                    if isinstance(dpg.get_value(tag_name), bool):
                        ctrl_value = True if ctrl_value else False
                    dpg.set_value(tag_name, ctrl_value)
                except:
                    pass

    def _save_to_config_callback(self, sender, app_data):
        ctrls = get_controls(self.device_idx)

        with open(self.camera_config) as cf:
            camera_data = json.load(cf)
        logger.debug(camera_data)

        assert "camera_config" in camera_data
        assert "controls" in camera_data["camera_config"]

        for name, ctrl in ctrls.items():
            camera_data["camera_config"]["controls"][name] = ctrl["value"]

        logger.info(f"Saving camera controls to {self.camera_config}.")
        with open(self.camera_config, 'w') as cf:
            json.dump(camera_data, cf, indent=4)

        self.config_change_queue.put(ConfigureType.CONFIGURE_TYPE_CAMERA_CONTROLS)


class ScreenDetectionEditor(EditorBase):
    def __init__(self, config_change_queue, focus_window, draw_list):
        self.config_change_queue = config_change_queue
        self.focus_window = focus_window
        self.input_focus = False
        self.draw_list = draw_list
        self.control_tags = []
        self.enabled = True

    def create(self, detection_config, game_name, camera_dims, display_dims):
        self.detection_config = detection_config
        self.camera_dims = camera_dims
        self.display_dims = display_dims
        self.scale = [self.display_dims[0] / self.camera_dims[0], self.display_dims[1] / self.camera_dims[1]]

        with open(self.detection_config) as df:
            detection_data = df.read()

        detection_data = json.loads(detection_data)
        self.method_name = detection_data["name"]
        detection_config_data = detection_data["detection_config"]
        self.detection_config_data = copy.deepcopy(detection_config_data)

        self.silhoutte_img_fp32 = None
        self.screen_rect_fixed = None
        self.curr_tags = None

        if self.method_name == "fixed":
            img_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "assets", "images", "screen", f"sil-{game_name}.png"
            )
            if os.path.exists(img_path):
                self.silhoutte_img_fp32 = (
                    cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                )

            # allow for different naming of the screen rect. ideally, standardize this.
            screen_rect_key = list(self.detection_config_data.keys())
            assert len(screen_rect_key) == 1
            screen_rect_key = screen_rect_key[0]
            # TODO: fixed points need to include a reference dimension; verify points are not degenerate
            self.screen_rect_fixed = np.array(self.detection_config_data[screen_rect_key], dtype=np.float32)
            self.screen_rect_ref = np.array([(0, 0), (160, 0), (160, 210), (0, 210)], dtype=np.float32)
            self.warp = cv2.getPerspectiveTransform(self.screen_rect_ref, self.screen_rect_fixed)

            self.selected_sr_point = None
            self.last_drag_delta = (0, 0)
            self.sr_pt_requires_update = True

            self.sr_pt_radius = 6
            self.tex_origin = (0, 0)
            self.tex_size = (self.display_dims[0], self.display_dims[1])

            self.snap_to_grid = False
            self.grid_px = 8
        else:
            self.tags_require_update = True

        if self.method_name == "dt_apriltags":
            for key, value in self.detection_config_data.items():
                # currently no support for nested dicts; if we need them, make a recursive option
                assert not isinstance(value, dict)
                # if isinstance(value, dict):
                #    with dpg.group(label=key):
                #        create_controls(value, parent)
                if isinstance(value, str):
                    dpg.add_text(f"{key}: {value}")
                elif isinstance(value, (int, float)):
                    min_val = 0
                    max_val = value * 4 if isinstance(value, (int, float)) else 100
                    if isinstance(value, float):
                        dpg.add_slider_float(
                            label=key,
                            tag=f"slider_{key}",
                            # step=0.1,
                            default_value=value,
                            min_value=min_val,
                            max_value=max_val,
                            callback=lambda sender, app_data, user_data: self._control_callback(
                                sender, app_data, user_data
                            ),
                            user_data=key,
                        )
                        dpg.bind_item_theme(f"slider_{key}", "slider_theme")
                    else:
                        dpg.add_slider_int(
                            label=key,
                            tag=f"slider_{key}",
                            default_value=value,
                            # step=1,
                            min_value=min_val,
                            max_value=max_val,
                            callback=lambda sender, app_data, user_data: self._control_callback(
                                sender, app_data, user_data
                            ),
                            user_data=key,
                        )
                        dpg.bind_item_theme(f"slider_{key}", "slider_theme")
                    self.control_tags.append(f"slider_{key}")
            dpg.add_button(
                label="Save",
                tag="sr_config_save_button",
                callback=lambda sender, app_data: self._save_to_config_callback(sender, app_data),
            )
            dpg.bind_item_theme("sr_config_save_button", "button_theme")
            self.control_tags.append("sr_config_save_button")
        elif self.method_name == 'fixed':
            dpg.add_text("Click the Camera Frame Window")
            dpg.add_text("<Shift+TAB> to select a screen point and <W,A,S,D> to position.")
            dpg.add_text("Click 'Save' when finished.")
            dpg.add_button(
                label="Save",
                tag="sr_pts_save_button",
                callback=lambda sender, app_data: self._save_to_config_callback(sender, app_data),
            )
            dpg.bind_item_theme("sr_pts_save_button", "button_theme")
            self.control_tags.append("sr_pts_save_button")
        else:
            dpg.add_text("Detection method not recognized.")

    def destroy(self):
        pass

    def enable(self, enabled: bool):
        self.enabled = enabled
        for tag in self.control_tags:
            dpg.configure_item(tag, enabled=enabled)

    def _gui_to_camera(self, points, scale_x, scale_y, off_x=0, off_y=0):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            return (points - np.array([off_x, off_y])) / np.array([scale_x, scale_y])
        elif points.ndim == 2:
            return (points - np.array([off_x, off_y])) / np.array([scale_x, scale_y])
        else:
            raise ValueError(f"_gui_to_camera: invalid ndim={points.ndim}.")

    def _camera_to_gui(self, points, scale_x, scale_y, off_x=0, off_y=0):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            return points * np.array([scale_x, scale_y]) + np.array([off_x, off_y])
        elif points.ndim == 2:
            return points * np.array([scale_x, scale_y]) + np.array([off_x, off_y])
        else:
            raise ValueError(f"_camera_to_gui: invalid ndim={points.ndim}.")

    def compute_overlay(self, src_rgb_f32: np.ndarray, alpha: float = 0.5) -> tuple[np.ndarray, bool]:
        if self.method_name != "fixed":
            return src_rgb_f32, False

        # True when the camera window view is selected
        if not self.input_focus:
            return src_rgb_f32, False

        if self.silhoutte_img_fp32 is None or src_rgb_f32 is None:
            return src_rgb_f32, False

        H, W = src_rgb_f32.shape[:2]

        # recompute the perspective warp transform when the screen rect points have changed this frame
        # this assumes the source and reference points are in camera dimensions
        if self.sr_pt_requires_update:
            self.warp = cv2.getPerspectiveTransform(
                np.asarray(self.screen_rect_ref), np.asarray(self.screen_rect_fixed)
            )

        warped = cv2.warpPerspective(
            self.silhoutte_img_fp32,
            self.warp,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        out = cv2.addWeighted(warped, alpha, src_rgb_f32, 1.0, 0.0)

        return out, True

    def draw(self):
        if self.screen_rect_fixed is not None:
            if not self.sr_pt_requires_update:
                return

            selected_color = (255, 255, 0)
            color = (0, 255, 0)
            pcorners = self._camera_to_gui(self.screen_rect_fixed, self.scale[0], self.scale[1])
            for i, point in enumerate(pcorners):
                dpg.delete_item(f"sr_{i}_pt")
                dpg.draw_circle(
                    center=point,
                    radius=self.sr_pt_radius,
                    fill=(
                        selected_color if self.selected_sr_point is not None and self.selected_sr_point == i else color
                    ),
                    color=(255, 255, 255),
                    parent=self.draw_list,
                    tag=f"sr_{i}_pt",
                )

                x_off = -30 if i == 0 or i == 3 else 10
                y_off = -30 if i < 2 else 10
                dpg.delete_item(f"sr_{i}_id")
                dpg.draw_text(
                    (point[0] + x_off, point[1] + y_off),
                    str(i),
                    color=(
                        selected_color if self.selected_sr_point is not None and self.selected_sr_point == i else color
                    ),
                    size=20,
                    parent=self.draw_list,
                    tag=f"sr_{i}_id",
                )

            self.sr_pt_requires_update = False
        elif self.tags_require_update:
            # delete any previous tags as they can change dynamically throughout the course
            # of training.
            for i in range(4):  # tags
                for t in range(4):  # corners
                    corner_tag = f"tag_{i}_line_{t}"
                    if dpg.does_item_exist(corner_tag):
                        dpg.delete_item(corner_tag)

                id_tag = f"tag_{i}_id"
                if dpg.does_item_exist(id_tag):
                    dpg.delete_item(id_tag)

            if self.curr_tags is not None:
                for i, (tag_id, corners) in enumerate(self.curr_tags.items()):
                    # corners are in camera space, scale to display resolution
                    corners = self._camera_to_gui(corners, self.scale[0], self.scale[1])
                    center_x = 0
                    center_y = 0
                    num_corners = len(corners)
                    for idx in range(num_corners):
                        dpg.draw_line(
                            corners[idx - 1],
                            corners[idx],
                            color=(0, 255, 0),
                            thickness=2,
                            parent=self.draw_list,
                            tag=f"tag_{i}_line_{idx}",
                        )
                        center_x += corners[idx][0]
                        center_y += corners[idx][1]
                    center_x /= num_corners
                    center_y /= num_corners
                    dpg.draw_text(
                        (center_x, center_y),
                        str(tag_id),
                        color=(0, 255, 0),
                        size=25,
                        parent=self.draw_list,
                        tag=f"tag_{i}_id",
                    )

            self.tags_require_update = False

    def _snap_to_tex_grid(self, pos):
        x, y = pos

        local_x = x - self.tex_origin[0]
        local_y = y - self.tex_origin[1]

        # snap
        snap_local_x = round(local_x / self.grid_px) * self.grid_px
        snap_local_y = round(local_y / self.grid_px) * self.grid_px

        return (self.tex_origin[0] + snap_local_x, self.tex_origin[1] + snap_local_y)

    def _clamp_to_tex(self, pos):
        pt_size = self.sr_pt_radius * 2
        min_x = self.tex_origin[0] + pt_size
        min_y = self.tex_origin[1] + pt_size
        max_x = self.tex_origin[0] + (self.tex_size[0] - 1) - pt_size
        max_y = self.tex_origin[1] + (self.tex_size[1] - 1) - pt_size

        x = min(max(pos[0], min_x), max_x)
        y = min(max(pos[1], min_y), max_y)

        return x, y

    def key_press_handler(self, sender, app_data):
        if not self.input_focus:
            return False

        if self.screen_rect_fixed is None or len(self.screen_rect_fixed) != 4:
            return False

        mod_shift = dpg.is_key_down(dpg.mvKey_ModShift)
        shift_tab = mod_shift and (app_data == dpg.mvKey_Tab)

        if shift_tab:
            if self.selected_sr_point is None:
                self.selected_sr_point = 0
            else:
                self.selected_sr_point = (self.selected_sr_point + 1) % 4  # clockwise
            self.sr_pt_requires_update = True
            return True

        if self.selected_sr_point is None:
            return False

        dx, dy = 0, 0
        if app_data == dpg.mvKey_W:
            dy = -1
        elif app_data == dpg.mvKey_S:
            dy = 1
        elif app_data == dpg.mvKey_A:
            dx = -1
        elif app_data == dpg.mvKey_D:
            dx = 1
        else:
            return False

        old_x, old_y = self.screen_rect_fixed[self.selected_sr_point]
        new_x, new_y = old_x + dx, old_y + dy

        if self.snap_to_grid and not dpg.is_key_down(dpg.mvKey_ModShift):
            new_x, new_y = self._snap_to_tex_grid((new_x, new_y))

        new_x, new_y = self._clamp_to_tex((new_x, new_y))
        self.screen_rect_fixed[self.selected_sr_point] = (new_x, new_y)

        self.sr_pt_requires_update = True
        return True

    def mouse_click_handler(self, sender, app_data):
        if self.input_focus:
            if self.screen_rect_fixed is None:
                return

            def is_point_inside(pt, target_pt, tolerance=10):
                dx = pt[0] - target_pt[0]
                dy = pt[1] - target_pt[1]
                return (dx * dx + dy * dy) < (tolerance * tolerance)

            mouse_pos = dpg.get_mouse_pos()

            for i, pt in enumerate(self.screen_rect_fixed):
                if is_point_inside(mouse_pos, pt, 15):
                    self.selected_sr_point = i
                    break

            self.sr_pt_requires_update = True
            return True
        return False

    def mouse_release_handler(self, sender, app_data):
        if self.input_focus:
            if self.screen_rect_fixed is None:
                return

            self.selected_sr_point = None
            self.last_drag_delta = (0, 0)
            self.sr_pt_requires_update = True
            return True
        return False

    def mouse_drag_handler(self, sender, app_data):
        if self.input_focus:
            logger.debug(f"mouse pos={dpg.get_mouse_pos()} delta={dpg.get_mouse_drag_delta()}")
            logger.debug(f"mouse dragged: {app_data}")

            if self.screen_rect_fixed is None:
                return

            if self.selected_sr_point is None:
                return

            curr_delta = dpg.get_mouse_drag_delta()
            dx = curr_delta[0] - self.last_drag_delta[0]
            dy = curr_delta[1] - self.last_drag_delta[1]
            if dx == 0 and dy == 0:
                return False

            old_x, old_y = self.screen_rect_fixed[self.selected_sr_point]
            new_x = old_x + dx
            new_y = old_y + dy

            if self.snap_to_grid and not dpg.is_key_down(dpg.mvKey_ModShift):
                new_x, new_y = self._snap_to_tex_grid((new_x, new_y))

            new_x, new_y = self._clamp_to_tex((new_x, new_y))
            # logger.debug(f"pt:{self.selected_sr_point}:  {new_x},{new_y}")

            self.screen_rect_fixed[self.selected_sr_point] = (new_x, new_y)
            self.last_drag_delta = curr_delta
            self.sr_pt_requires_update = True

            return True
        return False

    def focus_changed_handler(self, sender, app_data):
        self.input_focus = self.focus_window == app_data

    def set_tags(self, tags):
        if tags is None:
            return

        # start = time.time()
        # only update tags if a new set of positions (within a tolerance) have been provided
        if self.curr_tags is None or self.curr_tags.keys() != tags.keys():
            self.tags_require_update = True
            self.curr_tags = tags
        else:
            for key in self.curr_tags:
                try:
                    curr_tags = np.array(self.curr_tags[key], dtype=float)
                    new_tags = np.array(tags[key], dtype=float)

                    # check for malformed tags
                    if curr_tags.ndim != 2 or curr_tags.shape[1] != 2:
                        logger.warning(
                            f"set_tags: Skipping key={key} due to invalid curr_tags shape: {curr_tags.shape}"
                        )
                        continue

                    if new_tags.ndim != 2 or new_tags.shape[1] != 2:
                        logger.warning(f"set_tags: Skipping key={key} due to invalid new_tags shape: {new_tags.shape}")
                        continue
                except Exception as e:
                    logger.warning(f"set_tags: Skipping key={key} due to error: {e}")
                    continue

                diff = np.abs(curr_tags - new_tags)
                if np.any(diff > 0.2):
                    self.tags_require_update = True
                    self.curr_tags = tags
                    break
        # logger.debug(f'tags: {(time.time()-start) * 1000.0:.2f}')

    def _control_callback(self, sender, app_data, user_data):
        ctrl_name = user_data
        ctrl_value = app_data

        if isinstance(ctrl_value, bool):
            ctrl_value = 1 if ctrl_value else 0

        assert ctrl_name in self.detection_config_data
        self.detection_config_data[ctrl_name] = ctrl_value

    def _save_to_config_callback(self, sender, app_data):
        with open(self.detection_config) as df:
            detection_data = json.load(df)

        method_name = detection_data["name"]
        if method_name == "fixed":
            assert "detection_config" in detection_data
            # allow for different naming of the screen rect. ideally, standardize this.
            screen_rect_key = list(detection_data["detection_config"].keys())
            assert len(screen_rect_key) == 1
            screen_rect_key = screen_rect_key[0]

            screen_rect_list = self.screen_rect_fixed.tolist()

            detection_data["detection_config"][screen_rect_key] = screen_rect_list
            logger.info(f"Saving screen rect to {self.detection_config}.")
        elif method_name == "dt_apriltags":
            assert "detection_config" in detection_data
            detection_data["detection_config"] = self.detection_config_data
            logger.info(f"Saving screen detection config to {self.detection_config}")

        with open(self.detection_config, 'w') as df:
            json.dump(detection_data, df, indent=4)

        self.config_change_queue.put(ConfigureType.CONFIGURE_TYPE_SCREEN_DETECTION)


class ScoreDetectionEditor(EditorBase):
    def __init__(self, config_change_queue, focus_window, draw_list):
        self.config_change_queue = config_change_queue
        self.focus_window = focus_window
        self.draw_list = draw_list
        self.control_tags = []
        self.input_focus = False
        self.enabled = True

    def create(self, game_name, score_detector_type, rect_dims, rect_scale, max_rect_scale):
        self.game_name = game_name
        self.rect_dims = rect_dims
        self.rect_scale = rect_scale
        # the rect texture cannot be resized dynamically, so it's created at max and we track the current scale
        self.max_rect_scale = max_rect_scale

        self.selected_bb = 0
        self.bb_requires_update = True

        self.score_detector_type = score_detector_type
        score_model_type = ScoreDetectorConfig.get_model_type(self.score_detector_type)
        if score_model_type == "network":
            self.score_config_file = f"configs/games/{game_name}.json"
            with open(self.score_config_file) as gf:
                game_data = gf.read()
            score_config = json.loads(game_data)["score_config"]
        else:
            raise ValueError(f"Unsupported score detector type={score_detector_type}")

        self.score_crop_info = BoundingBox(**score_config["score_crop_info"], dims=(rect_dims[0], rect_dims[1]))
        if score_config["lives_crop_info"]:
            self.lives_crop_info = BoundingBox(**score_config["lives_crop_info"], dims=(rect_dims[0], rect_dims[1]))
        else:
            self.lives_crop_info = None

        self.score_offsets = {"offset_x": 0, "offset_y": 0}
        self.lives_offsets = {"offset_x": 0, "offset_y": 0}
        self.user_offset_file = "configs/.user_offsets.json"
        if os.path.exists(self.user_offset_file):
            with open(self.user_offset_file) as f:
                offsets = json.load(f)
                if self.game_name in offsets:
                    game_offsets = offsets[self.game_name]
                    if "score_crop_offset" in game_offsets:
                        self.score_offsets = game_offsets["score_crop_offset"]
                    if "lives_crop_offset" in game_offsets:
                        self.lives_offsets = game_offsets["lives_crop_offset"]

        self.default_score_offsets = copy.deepcopy(self.score_offsets)
        self.default_lives_offsets = copy.deepcopy(self.lives_offsets)

        dpg.add_text("Click the Configuration Window.")
        dpg.add_text(
            "Adjust the bounding boxes such that the score and/or lives\n"
            "region is tightly bounded horizontally with as little of the\n"
            "game content overlapping the box.\n\n"
            "<Shift+TAB> to select the box and <W,A,S,D> to position."
        )
        dpg.add_spacer()
        dpg.add_text("Click 'Save' or 'Reset' when finished.")
        dpg.add_spacer()
        dpg.add_button(
            label="Reset",
            tag="bb_reset_button",
            callback=lambda sender, app_data: self._reset_bb_callback(sender, app_data),
        )
        dpg.add_button(
            label="Save",
            tag="bb_save_button",
            callback=lambda sender, app_data: self._save_bb_callback(sender, app_data),
        )
        dpg.bind_item_theme("bb_reset_button", "button_theme")
        dpg.bind_item_theme("bb_save_button", "button_theme")

        self.control_tags.append("bb_reset_button")
        self.control_tags.append("bb_save_button")

    def destroy(self):
        pass

    def enable(self, enabled: bool):
        self.enabled = enabled
        for tag in self.control_tags:
            dpg.configure_item(tag, enabled=enabled)

    def draw(self):
        if not self.bb_requires_update:
            return

        selected_color = (255, 255, 0)
        color = (0, 255, 0)

        # place the image and bounding boxes in the center of the canvas
        start_x = (self.rect_dims[0] * (self.max_rect_scale - self.rect_scale)) // 2
        start_y = (self.rect_dims[1] * (self.max_rect_scale - self.rect_scale)) // 2

        # REVIEW: Once created, ideally we wouldn't need to re-create the drawlist; instead, update the position/scale
        # using .set_item_pos and .set_item_size. However, those functions do not appear to be available
        # for the drawing primitives. For now, recreate the drawlist when the rects need to be updated.
        if self.score_crop_info is not None:
            dpg.delete_item("score_bb")
            off_x = self.score_offsets["offset_x"]
            off_y = self.score_offsets["offset_y"]
            score_x = self.score_crop_info.x + off_x
            score_y = self.score_crop_info.y + off_y
            dpg.draw_rectangle(
                (start_x + (score_x * self.rect_scale), start_y + (score_y * self.rect_scale)),
                (
                    start_x + ((score_x + self.score_crop_info.w) * self.rect_scale),
                    start_y + ((score_y + self.score_crop_info.h) * self.rect_scale),
                ),
                color=(selected_color if self.selected_bb == 0 else color),
                thickness=2,
                tag="score_bb",
                parent=self.draw_list,
            )

        if self.lives_crop_info is not None:
            dpg.delete_item("lives_bb")
            off_x = self.lives_offsets["offset_x"]
            off_y = self.lives_offsets["offset_y"]
            lives_x = self.lives_crop_info.x + off_x
            lives_y = self.lives_crop_info.y + off_y
            dpg.draw_rectangle(
                (start_x + (lives_x * self.rect_scale), start_y + (lives_y * self.rect_scale)),
                (
                    start_x + ((lives_x + self.lives_crop_info.w) * self.rect_scale),
                    start_y + ((lives_y + self.lives_crop_info.h) * self.rect_scale),
                ),
                color=(selected_color if self.selected_bb == 1 else color),
                thickness=2,
                tag="lives_bb",
                parent=self.draw_list,
            )

        self.bb_requires_update = False

    def key_press_handler(self, sender, app_data):
        if self.input_focus:
            mod_shift = dpg.is_key_down(dpg.mvKey_ModShift)
            shift_tab = mod_shift and (app_data == dpg.mvKey_Tab)
            if (
                app_data == dpg.mvKey_W
                or app_data == dpg.mvKey_A
                or app_data == dpg.mvKey_S
                or app_data == dpg.mvKey_D
                or shift_tab
            ):
                if mod_shift and (app_data == dpg.mvKey_Tab):
                    # specifying lives is optional
                    if self.lives_crop_info is not None:
                        self.selected_bb ^= 1
                    # logger.debug(f"bounding box selected={mgr.selected_bb}")
                self._update_rects(app_data)
                return True
        return False

    def mouse_click_handler(self, sender, app_data):
        return False

    def mouse_release_handler(self, sender, app_data):
        return False

    def mouse_drag_handler(self, sender, app_data):
        return False

    def focus_changed_handler(self, sender, app_data):
        self.input_focus = self.focus_window == app_data

    def set_scale(self, scale):
        if scale != self.rect_scale:
            self.rect_scale = scale
            self.bb_requires_update = True

    def _update_rects(self, key):
        step_size = 1

        offsets = self.score_offsets if self.selected_bb == 0 else self.lives_offsets
        if key == dpg.mvKey_W:
            offsets["offset_y"] = offsets["offset_y"] - step_size
        elif key == dpg.mvKey_S:
            offsets["offset_y"] = offsets["offset_y"] + step_size
        elif key == dpg.mvKey_A:
            offsets["offset_x"] = offsets["offset_x"] - step_size
        elif key == dpg.mvKey_D:
            offsets["offset_x"] = offsets["offset_x"] + step_size

        logger.debug(f"offsets={offsets}")
        if self.selected_bb == 0:
            self.score_offsets = offsets
        else:
            self.lives_offsets = offsets

        self.bb_requires_update = True

    def _reset_bb_callback(self, sender, app_data):
        if self.selected_bb == 0:
            self.score_offsets = copy.deepcopy(self.default_score_offsets)
        else:
            self.lives_offsets = copy.deepcopy(self.default_lives_offsets)
        self.bb_requires_update = True

    def _save_bb_callback(self, sender, app_data):
        if os.path.exists(self.user_offset_file):
            with open(self.user_offset_file) as f:
                offsets = json.load(f)
        else:
            offsets = {}

        # if the game does not exist yet in the file, default
        if self.game_name not in offsets:
            offsets[self.game_name] = {}

        # update to user defined offsets
        offsets[self.game_name]["score_crop_offset"] = self.score_offsets
        offsets[self.game_name]["lives_crop_offset"] = self.lives_offsets

        self.default_score_offsets = copy.deepcopy(self.score_offsets)
        self.default_lives_offsets = copy.deepcopy(self.lives_offsets)

        logger.info(f"Saving offsets  to {self.user_offset_file}.")
        with open(self.user_offset_file, 'w') as gf:
            json.dump(offsets, gf, indent=4)

        self.config_change_queue.put(ConfigureType.CONFIGURE_TYPE_GAME_SCORE_BOXES)


class EpisodeGraphManager:
    def __init__(self):
        self.episode_scores = []
        self.episode_ends = []
        self.ave_scores = []
        self.reward_over_time = []

    def add_episode(self, score, end_frame):
        self.episode_scores.append(score)
        self.episode_ends.append(end_frame)

        num_episodes = len(self.episode_ends)

        dpg.set_value("rect_frame_episodes_text", f"Episodes: {num_episodes}")
        dpg.set_value("score_series", [self.episode_ends, self.episode_scores])
        self._autoscale_axis("episode_x_axis", "episode_y_axis", self.episode_scores)

        # moving reward rate
        if num_episodes >= 2:
            k = min(50, num_episodes)
            delta = self.episode_ends[-1] - self.episode_ends[-k]
            if delta > 0:
                recent = sum(self.episode_scores[-k:])
                reward_rate = recent / delta
                self.reward_over_time.append((self.episode_ends[-1], reward_rate))
                x_vals, y_vals = zip(*self.reward_over_time)
                dpg.set_value("ave_reward_series", [x_vals, y_vals])
                self._autoscale_axis("ave_reward_x_axis", "ave_reward_y_axis", y_vals)

    def add_avg_score(self, avg_score):
        self.ave_scores.append(avg_score)
        dpg.set_value("ave_score_series", [self.episode_ends, self.ave_scores])

    def reset(self):
        self.__init__()

        dpg.set_value("ave_reward_series", [list([0.0]), list([0.0])])
        dpg.fit_axis_data('ave_reward_x_axis')
        dpg.fit_axis_data('ave_reward_y_axis')

        dpg.set_value("score_series", [list([0.0]), list([0.0])])
        dpg.fit_axis_data('episode_x_axis')
        dpg.fit_axis_data('episode_y_axis')

        dpg.set_value("ave_score_series", [list([0.0]), list([0.0])])

    def _autoscale_axis(self, x_axis_tag, y_axis_tag, y_vals):
        dpg.fit_axis_data(x_axis_tag)
        if False:
            dpg.fit_axis_data(y_axis_tag)
        else:
            y_arr = np.array(y_vals)
            if len(y_arr) >= 2:
                mean = np.mean(y_arr)
                std_dev = np.std(y_arr)
                upper = mean + 2 * std_dev
            else:
                upper = y_arr[-1]
            dpg.set_axis_limits(y_axis_tag, 0.0, upper)


class FrameGraphManager:
    def __init__(self):
        self.interframe_periods = [0.0] * 300
        self.rewards = [0.0] * 300
        self.terminations = [0.0] * 300
        self.chosen_actions = [0] * 1000

    def update_interframe(self, frame, period):
        n = len(self.interframe_periods)
        k = frame % n
        self.interframe_periods[k] = period
        values = self.interframe_periods[k + 1 :] + self.interframe_periods[: k + 1]
        x_vals = list(range(frame - n, frame))
        dpg.set_value("interframe_period_series", [x_vals, values])
        dpg.fit_axis_data("interframe_x_axis")
        dpg.fit_axis_data("interframe_y_axis")

    def update_reward_termination(self, frame, reward, termination):
        n = len(self.rewards)
        k = frame % n
        self.rewards[k] = reward
        self.terminations[k] = termination
        r = self.rewards[k + 1 :] + self.rewards[: k + 1]
        t = self.terminations[k + 1 :] + self.terminations[: k + 1]
        x_vals = list(range(frame - n, frame))
        dpg.set_value("reward_series", [x_vals, r])
        dpg.set_value("termination_series", [x_vals, t])
        dpg.fit_axis_data("reward_x_axis")
        dpg.fit_axis_data("reward_y_axis")

    def update_action(self, frame, action_value):
        n = len(self.chosen_actions)
        k = frame % n
        self.chosen_actions[k] = action_value
        values = self.chosen_actions[k + 1 :] + self.chosen_actions[: k + 1]
        x_vals = list(range(frame - n, frame))
        dpg.set_value("actions_series", [x_vals, values])
        dpg.fit_axis_data("actions_x_axis")
        dpg.fit_axis_data("actions_y_axis")

    def reset(self):
        self.__init__()

        dpg.set_value("interframe_period_series", [list([0.0]), list([0.0])])
        dpg.fit_axis_data('interframe_x_axis')
        dpg.fit_axis_data('interframe_y_axis')

        dpg.set_value("reward_series", [list([0.0]), list([0.0])])
        dpg.set_value("termination_series", [list([0.0]), list([0.0])])
        dpg.fit_axis_data('reward_x_axis')
        dpg.fit_axis_data('reward_y_axis')

        dpg.set_value("actions_series", [list([0.0]), list([0.0])])
        dpg.fit_axis_data('actions_x_axis')
        dpg.fit_axis_data('actions_y_axis')


class StatGraphManager:
    def __init__(self, max_points):
        self.max_points = max_points
        self.stats_ts = 0
        self.metrics = {}

    def register_metric(self, name, dpg_tag, x_axis_tag, y_axis_tag):
        self.metrics[name] = {"values": [], "dpg_tag": dpg_tag, "x_axis_tag": x_axis_tag, "y_axis_tag": y_axis_tag}

    def update_metric(self, name, new_value):
        if name not in self.metrics:
            logger.warning(f"update_metric: {name} not in metrics")
            return
        metric = self.metrics[name]
        if len(metric["values"]) > self.max_points:
            metric["values"].pop(0)
        metric["values"].append(new_value)

    def draw_visible(self):
        x_min, x_max = max(0, self.stats_ts - self.max_points), self.stats_ts
        x_range = list(range(x_min, x_max))

        if dpg.is_item_visible("sys_stats"):
            for name, metric in self.metrics.items():
                if len(x_range) == 0 or len(metric["values"]) == 0:
                    continue
                dpg.set_value(metric["dpg_tag"], [x_range, metric["values"]])
                dpg.fit_axis_data(metric["x_axis_tag"])
                dpg.fit_axis_data(metric["y_axis_tag"])

        self.stats_ts += 1


class PhysicalGui:
    def __init__(
        self,
        game_config,
        device_config,
        camera_config,
        detection_config,
        score_detector_type,
        obs_dims,
        episode_queue,
        shared_data,
        configure_event=None,
        exit_event=None,
    ):
        from framework.Keyboard import Keys

        self.exit_event = exit_event
        self.configure_event = configure_event
        self.configure_mode = configure_event is not None
        self.configure_state = ConfigureState.CONFIGURE_STARTED
        self.validation_reason = ValidationFailureReason.VALIDATION_FAILURE_NONE

        self.episode_queue = episode_queue
        self.shared_data = shared_data

        self.obs_dims = obs_dims

        # track the available editors
        self.editors = []
        # alert on configuration changed
        self.config_change_queue = queue.Queue()

        self.show_checkpoint_menu = False

        # create a setup config for general run details
        self.setup_config = '.setup.cfg.json'
        setup_data = None
        if not os.path.exists(self.setup_config):
            with open(self.setup_config, "w") as file:
                json.dump({}, file)
        else:
            with open(self.setup_config) as file:
                setup_data = json.load(file)

        self.game_config = game_config
        with open(self.game_config) as gf:
            game_data = gf.read()

        game_data = json.loads(game_data)["game_config"]
        self.game_name = game_data["name"]

        self.device_config = device_config
        self.screen_detection_config = detection_config
        self.score_detector_type = score_detector_type

        self.camera_config = camera_config
        with open(self.camera_config) as cf:
            camera_data = cf.read()
        camera_data = json.loads(camera_data)
        camera_name = camera_data["model_name"]
        camera_config_data = camera_data["camera_config"]

        self.camera_device_idx = get_index_from_model_name(camera_name)
        self.camera_dims = (int(camera_config_data["width"]), int(camera_config_data["height"]))

        # specify a reference dimension for display and resize when appropriate
        self.display_cam_dims = (640, 480)

        self.min_rect_frame_scale = 1
        self.max_rect_frame_scale = self.display_cam_dims[0] // self.obs_dims[0]

        # REVIEW: variables set via callbacks; add mutex to guard access
        self.focused_window = None
        self.rect_frame_scale = (
            4
            if self.obs_dims[0] <= (self.display_cam_dims[0] // 4)
            or self.obs_dims[1] <= (self.display_cam_dims[1] // 4)
            else 1
        )
        self.prev_rect_frame_scale = self.rect_frame_scale
        self.num_game_runs = 1 if setup_data is None or 'num_runs' not in setup_data else setup_data['num_runs']
        self.save_checkpoint = (
            False if setup_data is None or 'save_model' not in setup_data else setup_data['save_model']
        )
        self.load_checkpoint = (
            None if setup_data is None or 'load_model' not in setup_data else setup_data['load_model']
        )

        self.keyboard_controls = ", ".join(f"{'space' if key.value == ' ' else key.value}:{key.name}" for key in Keys)

        self.current_game_run = 0

        # system stats graphing
        self.gpu_id = 0
        self.max_system_stats_points = 150
        self.next_system_stats_time_in_sec = 0.0
        self.system_stats_interval_in_sec = 60.0
        self.health_monitor_thread = HealthMonitorThread(self.gpu_id, self.system_stats_interval_in_sec)

        if self.configure_mode:
            # Render default content until game configuration is complete
            self.content_thread = GuiRenderContent(
                self.shared_data,
                self.config_change_queue,
                self.game_config,
                self.camera_config,
                self.screen_detection_config,
                self.device_config,
                self.score_detector_type,
                obs_dims=self.obs_dims,
            )
            self.content_thread.initialize()
        else:
            self.content_thread = None

        self.create_window()
        self.update_control_button_state()

    def create_window(self):
        # REVIEW: The ubuntu WM appears to trigger a maximize behavior for heights
        # that are close to fullscreen; so although a height less than full screen
        # is requested, it snaps to fullscreen.
        TERMINAL_LINES = 384

        screen_width, screen_height = get_screen_dimensions()
        logger.debug(f"screen_dimensions={screen_width}x{screen_height}")
        window_width = screen_width
        window_height = screen_height - TERMINAL_LINES
        logger.debug(f"gui_dims={window_width}x{window_height}")

        dpg.create_context()
        dpg.setup_dearpygui()

        logger.debug(f"Creating viewport: {window_width}x{window_height}")
        dpg.create_viewport(
            title=f"Physical Atari: {self.game_name}",
            width=window_width,
            height=window_height,
            resizable=True,
            decorated=True,
        )

        # Adjust the font size.
        # NOTE: dearpygui built-in font is a fixed 13px bitmap and can't be scaled without making it blurry;
        # in order to get a larger font we have to specify a system or custom font with size.
        default_font_size = 13
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font_size = 20
        # dpg get_text_size is calculated based on the default font size, which
        # requires a manual scale when using a larger, custom font
        self.font_scale = font_size / default_font_size
        if os.path.exists(font_path):
            logger.debug(f"Adjusting font: {font_path}")
            with dpg.font_registry():
                font = dpg.add_font(font_path, font_size)
            dpg.bind_font(font)

        with dpg.theme(tag="button_theme"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (55, 55, 55, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Border, (128, 128, 128, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (100, 100, 100, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (150, 150, 150, 255), category=dpg.mvThemeCat_Core)

            with dpg.theme_component(dpg.mvButton, enabled_state=True):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))

            with dpg.theme_component(dpg.mvButton, enabled_state=False):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (70, 70, 70, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (160, 160, 160, 255))
                dpg.add_theme_style(dpg.mvStyleVar_Alpha, 0.6)

        with dpg.theme(tag="slider_theme"):
            # use default theme state for 'enabled_state'=True

            for comp in (dpg.mvSliderInt, dpg.mvSliderFloat):
                with dpg.theme_component(comp, enabled_state=False):
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (55, 55, 55, 255))
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (55, 55, 55, 255))
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (55, 55, 55, 255))
                    dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (90, 90, 90, 255))
                    dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (90, 90, 90, 255))
                    dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (160, 160, 160, 255))
                    dpg.add_theme_style(dpg.mvStyleVar_Alpha, 0.6)

        with dpg.theme(tag="checkbox_theme"):
            # use default theme state for 'enabled_state'=True

            with dpg.theme_component(dpg.mvCheckbox, enabled_state=False):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (55, 55, 55, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (55, 55, 55, 255))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (120, 120, 120, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (160, 160, 160, 255))
                dpg.add_theme_style(dpg.mvStyleVar_Alpha, 0.6)

        with dpg.theme(tag="combo_theme"):
            # use default theme state for 'enabled_state'=True

            with dpg.theme_component(dpg.mvCombo, enabled_state=False):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (55, 55, 55, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (55, 55, 55, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (55, 55, 55, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (70, 70, 70, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (70, 70, 70, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (70, 70, 70, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (160, 160, 160, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (160, 160, 160, 255))
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (35, 35, 35, 255))
                dpg.add_theme_style(dpg.mvStyleVar_Alpha, 0.6)

        with dpg.theme(tag="quit_theme"):
            with dpg.theme_component(dpg.mvButton, enabled_state=True):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (110, 60, 170, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (140, 90, 200, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (90, 40, 140, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (230, 210, 255, 255))

            with dpg.theme_component(dpg.mvButton, enabled_state=False):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 40, 90, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (160, 160, 160, 255))
                dpg.add_theme_style(dpg.mvStyleVar_Alpha, 0.6)

        with dpg.theme(tag="valid_text_theme"):  # as valid_text_theme:
            with dpg.theme_component(dpg.mvText):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 255, 0))

        with dpg.theme(tag="invalid_text_theme"):  # as invalid_text_theme:
            with dpg.theme_component(dpg.mvText):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 0, 0))

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.display_cam_dims[0],
                self.display_cam_dims[1],
                default_value=np.zeros((self.display_cam_dims[1], self.display_cam_dims[0], 3), dtype=np.float32),
                format=dpg.mvFormat_Float_rgb,
                tag="camera_texture",
            )
            # dpg textures cannot be resized after the fact, in order to scale the rect_frame the texture has to be created
            # at the max size and inset depending on current scale.
            dpg.add_raw_texture(
                self.obs_dims[0] * self.max_rect_frame_scale,
                self.obs_dims[1] * self.max_rect_frame_scale,
                default_value=np.zeros(
                    (
                        self.obs_dims[1] * self.max_rect_frame_scale,
                        self.obs_dims[0] * self.max_rect_frame_scale,
                        self.obs_dims[2],
                    ),
                    dtype=np.float32,
                ),
                format=dpg.mvFormat_Float_rgb,
                tag="rect_texture",
            )

        with dpg.handler_registry():
            dpg.add_key_press_handler(callback=lambda sender, app_data: self.key_press_handler(sender, app_data))
            dpg.add_mouse_drag_handler(callback=lambda sender, app_data: self.mouse_drag_handler(sender, app_data))
            dpg.add_mouse_click_handler(callback=lambda sender, app_data: self.mouse_click_handler(sender, app_data))
            dpg.add_mouse_release_handler(
                callback=lambda sender, app_data: self.mouse_release_handler(sender, app_data)
            )

        with dpg.item_handler_registry() as focus_handlers:
            dpg.add_item_focus_handler(callback=lambda sender, app_data: self.on_focus_handler(sender, app_data))

        # file dialog for choosing checkpoint to load
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=lambda sender, app_data: self.set_load_checkpoint(sender, app_data["file_path_name"]),
            tag="checkpoint_file_dialog",
            width=500,
            height=400,
        ):
            dpg.add_file_extension(".model", color=(0, 200, 200, 255))
            dpg.add_file_extension(".pt", color=(0, 255, 0, 255))
            dpg.add_file_extension(".ckpt", color=(0, 200, 255, 255))

        # create a dummy parent container and set as primary to establish a layout base; without it,
        # on some configurations, the window may be constructed as borderless without decorations, preventing resize.
        with dpg.window(tag="invisible_root", show=False, width=window_width, height=window_height):
            pass

        """
        TODO: Review: Does dearpygui have an advanced layout manager?
        We may want to define some of this in configs so it is more clear.

        -------------------------------------
        |                  |        |       |
        |------------------|        |       |
        |        |         |----------------|
        |        |         |                |
        |        |         |----------------|
        |        |         |                |
        |        |         |                |
        -------------------------------------
        """

        # ------------------------------------
        # Create the Left Pane Layout
        # ------------------------------------

        left_pane_width = self.display_cam_dims[0] * 2
        left_pane_height = window_height
        left_pane_x_offset = 0
        left_pane_y_offset = 0

        with dpg.window(
            label="Camera Frame",
            pos=(0, left_pane_y_offset),
            width=self.display_cam_dims[0],
            height=self.display_cam_dims[1],
            no_close=True,
            no_resize=True,
            autosize=False,
            no_scrollbar=True,
        ) as camera_window:
            # In order to render an image with drawing primitives on top, a drawlist must be created.
            # Otherwise, the image and drawing primitives will be grouped.
            # dpg.add_image("camera_texture")
            dpg.add_drawlist(
                width=self.display_cam_dims[0],
                height=self.display_cam_dims[1],
                tag="camera_drawlist",
                parent=camera_window,
            )
            dpg.draw_image(
                "camera_texture", (0, 0), (self.display_cam_dims[0], self.display_cam_dims[1]), parent="camera_drawlist"
            )
            # screen_detection drawlist will be added independently based on detection method
            dpg.bind_item_handler_registry(camera_window, focus_handlers)
            # if a tag name is added to the window definition, that will override what the focus handler returns and will need to be accounted for.
            self.camera_window = camera_window

        left_pane_x_offset += left_pane_width // 2

        with dpg.window(
            label="Game Frame",
            pos=(left_pane_x_offset, left_pane_y_offset),
            width=left_pane_width // 2,
            height=left_pane_height - left_pane_y_offset,
            no_close=True,
            no_resize=True,
            autosize=False,
            no_scrollbar=True,
        ) as rect_window:
            # In order to render an image with drawing primitives on top, a drawlist must be created.
            # Otherwise, the image and drawing primitives will be grouped.
            # dpg.add_image("rect_texture", parent=rect_window)
            dpg.add_drawlist(
                width=self.obs_dims[0] * self.max_rect_frame_scale,
                height=self.obs_dims[1] * self.max_rect_frame_scale,
                tag="rect_drawlist",
                parent=rect_window,
            )
            dpg.draw_image(
                "rect_texture",
                (0, 0),
                (self.obs_dims[0] * self.max_rect_frame_scale, self.obs_dims[1] * self.max_rect_frame_scale),
                parent="rect_drawlist",
            )
            dpg.add_text(
                "Mode: Train",
                pos=((self.obs_dims[0] * self.max_rect_frame_scale) - 155, 45),
                color=(255, 255, 255),
                tag="rect_frame_mode_text",
            )
            dpg.add_text(
                "FPS: 0.00",
                pos=((self.obs_dims[0] * self.max_rect_frame_scale) - 100, 70),
                color=(255, 255, 255),
                tag="rect_frame_fps_text",
            )
            dpg.add_spacer()
            dpg.add_text(f"Game: {self.game_name}", color=(245, 0, 245), tag="rect_frame_game_text")
            dpg.add_text("Frame: 0", color=(255, 255, 0), tag="rect_frame_frame_text")
            dpg.add_text("Score: 0", color=(0, 255, 0), tag="rect_frame_score_text")
            dpg.add_text("Lives: 0 / 0", color=(0, 255, 0), tag="rect_frame_lives_text")
            dpg.add_text("Episodes: 0", color=(0, 255, 255), tag="rect_frame_episodes_text")
            dpg.add_text("Action: NOOP", color=(255, 128, 0), tag="rect_frame_action_text")
            with dpg.group(horizontal=False, tag="rect_control_banner"):
                dpg.add_text("Configure Mode:", color=(255, 100, 200), tag="rect_mode_label")
                dpg.add_text(f"{self.keyboard_controls}", wrap=480, color=(200, 200, 200), tag="rect_mode_detail")

            dpg.add_spacer()
            # dpg.add_spacer()
            dpg.add_slider_int(
                label="Frame Scale",
                tag="frame_scale_slider",
                default_value=self.rect_frame_scale,
                min_value=self.min_rect_frame_scale,
                max_value=self.max_rect_frame_scale,
                callback=lambda sender, app_data: self.scale_rect_frame_callback(sender, app_data),
                user_data={'class': self},
            )
            dpg.bind_item_theme("frame_scale_slider", "slider_theme")

            dpg.bind_item_handler_registry(rect_window, focus_handlers)
            # if a tag name is added to the window definition, that will override what the focus handler returns and will need to be accounted for.
            self.rect_window = rect_window

        with dpg.window(
            label="Configuration",
            pos=(0, left_pane_y_offset + self.display_cam_dims[1]),
            width=self.display_cam_dims[0],
            height=left_pane_height - self.display_cam_dims[1] - left_pane_y_offset,
            no_close=True,
            no_resize=True,
            autosize=False,
        ) as config_window:
            with dpg.collapsing_header(label="Camera", tag="camera_header", default_open=False):
                with dpg.child_window(label="Camera Controls", tag="camera_controls", border=False):
                    self.camera_editor = CameraControlEditor(
                        self.config_change_queue, self.camera_device_idx, config_window, None
                    )
                    self.camera_editor.create(self.camera_config)
                    self.camera_editor.draw()
                    self.editors.append(self.camera_editor)

            with dpg.collapsing_header(label="Screen Detection", tag="screen_detection_header", default_open=False):
                with dpg.child_window(label="Screen Detection Controls", tag="detection_controls", border=False):
                    # takes input focus when the camera window is focused
                    self.screen_detection_editor = ScreenDetectionEditor(
                        self.config_change_queue, camera_window, "camera_drawlist"
                    )
                    self.screen_detection_editor.create(
                        self.screen_detection_config, self.game_name, self.camera_dims, self.display_cam_dims
                    )
                    self.screen_detection_editor.draw()
                    self.editors.append(self.screen_detection_editor)

            with dpg.collapsing_header(label="Score Detection", tag="score_detection_header", default_open=False):
                with dpg.child_window(label="Score Detection Controls", tag="score_detection_controls", border=False):
                    self.score_detection_editor = ScoreDetectionEditor(
                        self.config_change_queue, config_window, "rect_drawlist"
                    )
                    self.score_detection_editor.create(
                        self.game_name,
                        self.score_detector_type,
                        self.obs_dims,
                        self.rect_frame_scale,
                        self.max_rect_frame_scale,
                    )
                    self.score_detection_editor.draw()
                    self.editors.append(self.score_detection_editor)

            with dpg.collapsing_header(label="Controller", tag="controller_header", default_open=False):
                with dpg.child_window(label="Controller Testing", tag="controller_testing", border=False):
                    # This header is currently informative only. If specific controls
                    # or other functionality is needed, create a new editor class and setup here.
                    dpg.add_text("Click the Game Frame Window.")
                    dpg.add_text("Manually drive the controller via keyboard commands:\n\n" f"{self.keyboard_controls}")

            if self.configure_mode:
                with dpg.group(horizontal=False):
                    dpg.add_text("Configure Mode.", tag="configure_state_text")
                    dpg.bind_item_theme("configure_state_text", "valid_text_theme")

                    # select number of runs
                    dpg.add_spacer()
                    dpg.add_spacer()
                    dpg.add_combo(
                        [str(i) for i in range(1, 11)],
                        label="Select Number of Runs",
                        tag="run_selection",
                        default_value=str(self.num_game_runs),
                        callback=lambda sender, app_data: self.set_num_game_runs(sender, app_data),
                    )
                    dpg.bind_item_theme("run_selection", "combo_theme")

                    if self.show_checkpoint_menu:
                        # save checkpoint
                        dpg.add_spacer()
                        dpg.add_spacer()
                        dpg.add_checkbox(
                            label="Save Model",
                            tag="save_checkpoint_checkbox",
                            default_value=self.save_checkpoint,
                            callback=lambda sender, app_data: self.set_save_checkpoint(sender, app_data),
                        )
                        dpg.bind_item_theme("save_checkpoint_checkbox", "checkbox_theme")
                        with dpg.group(horizontal=True):
                            # load checkpoint
                            dpg.add_button(
                                label="Load Model" if self.load_checkpoint is None else "Clear Model",
                                tag="checkpoint_toggle_button",
                                callback=lambda sender, app_data: self.toggle_checkpoint_selection(sender, app_data),
                            )
                            dpg.add_text(
                                "None" if self.load_checkpoint is None else self.load_checkpoint,
                                tag="selected_checkpoint_path",
                            )
                            dpg.bind_item_theme("checkpoint_toggle_button", "button_theme")

                    # Start Train
                    dpg.add_spacer()
                    dpg.add_spacer()
                    dpg.add_button(
                        label="Start Training",
                        width=180,
                        height=40,
                        callback=self.start_game,
                        enabled=True,
                        tag="start_button",
                    )
                    dpg.bind_item_theme("start_button", "button_theme")

                    # Quit the program entirely
                    with dpg.group(horizontal=False):
                        dpg.add_spacer(height=325)
                        dpg.add_button(
                            label="Quit",
                            callback=self.quit,
                            width=-1,  # stretch across pane
                            height=40,
                            enabled=True,
                            tag="quit_button",
                        )
                        dpg.bind_item_theme("quit_button", "quit_theme")

            dpg.bind_item_handler_registry(config_window, focus_handlers)
            # if a tag name is added to the window definition, that will override what the focus handler returns and will need to be accounted for.
            self.config_window = config_window

        # ------------------------------------
        # Create the Right Pane Layout
        # ------------------------------------

        # REVIEW: inset, as the window extends beyond the screen dimensions
        right_pane_width = window_width - left_pane_width
        right_pane_height = window_height
        right_pane_x_offset = left_pane_width
        right_pane_y_offset = 0

        num_graph_rows = 4
        graph_height = (right_pane_height - 16) // num_graph_rows

        with dpg.window(
            label="Frame and Episode Stats",
            tag="frame_stats",
            pos=(right_pane_x_offset, right_pane_y_offset),
            width=right_pane_width,
            height=right_pane_height,
            no_close=True,
            no_resize=True,
            autosize=False,
        ):
            with dpg.group(horizontal=True):
                with dpg.plot(label="Interframe Period", height=graph_height, width=right_pane_width // 2 - 16):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Frames", tag='interframe_x_axis')
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Time (s)", tag='interframe_y_axis')
                    dpg.add_scatter_series(
                        x=list([0.0]), y=list([0.0]), label="Period", parent=y_axis, tag='interframe_period_series'
                    )

                with dpg.plot(label="Rewards and Terminations", height=graph_height, width=right_pane_width // 2 - 16):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Frames", tag='reward_x_axis')
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Reward", tag='reward_y_axis')
                    dpg.add_scatter_series(
                        x=list([0.0]), y=list([0.0]), label="Reward", parent=y_axis, tag='reward_series'
                    )
                    dpg.add_scatter_series(
                        x=list([0.0]), y=list([0.0]), label="Termination", parent=y_axis, tag='termination_series'
                    )
            right_pane_y_offset += graph_height * 1

            with dpg.group(horizontal=False):
                with dpg.plot(label="Chosen Actions", height=graph_height, width=right_pane_width - 20):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Frames", tag='actions_x_axis')
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Chosen Actions", tag='actions_y_axis')
                    dpg.add_scatter_series(
                        list([0.0]), list([0.0]), label="Actions", parent=y_axis, tag='actions_series'
                    )

                with dpg.plot(label="Reward over Time", height=graph_height, width=right_pane_width - 20):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Frames", tag='ave_reward_x_axis')
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Reward Rate", tag='ave_reward_y_axis')
                    dpg.add_line_series(
                        x=list([0.0]), y=list([0.0]), label="Reward Rate", parent=y_axis, tag='ave_reward_series'
                    )

                with dpg.plot(label="Episode Scores and Ends", height=graph_height, width=right_pane_width - 20):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Episode End", tag='episode_x_axis')
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Episode Score", tag='episode_y_axis')
                    dpg.add_scatter_series(list([0.0]), list([0.0]), label="Scores", parent=y_axis, tag='score_series')
                    dpg.add_line_series(
                        list([0.0]), list([0.0]), label="Avg Score", parent=y_axis, tag='ave_score_series'
                    )

            right_pane_y_offset += graph_height * 3

        # System stats window, hidden by default
        with dpg.window(label="System Stats", tag="sys_stats", width=480, height=window_height, show=False):
            with dpg.group(label=f"GPU {self.gpu_id} Stats", horizontal=False):
                with dpg.plot(label="GPU Temperature (C)", height=250, width=480):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="gpu_temp_x_axis")
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Temperature (°C)", tag="gpu_temp_y_axis")
                    dpg.add_line_series(
                        x=list([0.0]), y=list([0.0]), label="Temperature", parent=y_axis, tag='gpu_temp_graph'
                    )

                with dpg.plot(label="GPU Utilization", height=250, width=480):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="gpu_util_x_axis")
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Utilization (%)", tag="gpu_util_y_axis")
                    dpg.add_line_series(
                        x=list([0.0]), y=list([0.0]), label="Utilization", parent=y_axis, tag='gpu_util_graph'
                    )

                with dpg.plot(label="GPU Memory Utilization", height=250, width=480):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="gpu_mem_x_axis")
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Memory Usage (%)", tag="gpu_mem_y_axis")
                    dpg.add_line_series(
                        x=list([0.0]),
                        y=list([0.0]),
                        label="Memory Utilization",
                        parent=y_axis,
                        tag='gpu_mem_util_graph',
                    )

            with dpg.group(label="System Stats", horizontal=False):
                with dpg.plot(label="System Memory Usage", height=250, width=480):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="sys_mem_x_axis")
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Memory Usage (MB)", tag="sys_mem_y_axis")
                    dpg.add_line_series(
                        x=list([0.0]), y=list([0.0]), label="Memory Usage", parent=y_axis, tag='sys_mem_graph'
                    )

                with dpg.plot(label="System CPU Utilization", height=250, width=480):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="sys_cpu_util_x_axis")
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Utilization (%)", tag="sys_cpu_util_y_axis")
                    dpg.add_line_series(
                        x=list([0.0]), y=list([0.0]), label="Utilization", parent=y_axis, tag='cpu_util_graph'
                    )

        with dpg.window(label="Help Menu", pos=(0, 0), width=400, height=400, show=False, tag="help_menu"):
            # dpg.add_text("Help Menu", color=(255, 255, 0, 255))
            dpg.add_text("Key Bindings:", color=(255, 255, 0, 255))
            dpg.add_spacer()
            dpg.add_spacer()
            dpg.add_text("Shift+D: Show System Debug Menus")
            dpg.add_text("F: Show the System Font Menu")
            dpg.add_text("H: Toggle help menu visibility")
            dpg.add_text("P: Toggle system stats visibility")
            dpg.add_spacer()
            dpg.add_spacer()

        dpg.show_viewport()
        dpg.set_primary_window("invisible_root", True)

        # vp_width = dpg.get_viewport_client_width()
        # vp_height = dpg.get_viewport_client_height()
        # logger.debug(f"viewport_dimensions={vp_width}x{vp_height}")

    def shutdown(self):
        logger.debug("PhysicalGui: shutdown")

        if self.content_thread is not None:
            self.content_thread.shutdown()

        if self.health_monitor_thread is not None:
            self.health_monitor_thread.shutdown()

        if self.shared_data is not None:
            try:
                self.shared_data.close()
            except Exception as e:
                logger.warning(f"Shared data close failed: {e}")

        dpg.destroy_context()

        if self.configure_event is not None:
            # release the game if the configure event has not been set before shutting down the gui.
            if not self.configure_event.is_set():
                self.configure_event.set()

    def update_control_button_state(self):
        if not self.configure_mode:
            return

        logger.debug(f"update_control_button_state: state={self.configure_state.value}")

        # Enable/Disable buttons based on the current state.
        dpg.configure_item("start_button", enabled=(self.configure_state != ConfigureState.TRAIN_STARTED))
        dpg.configure_item("run_selection", enabled=(self.configure_state != ConfigureState.TRAIN_STARTED))

        if self.show_checkpoint_menu:
            dpg.configure_item(
                "checkpoint_toggle_button", enabled=(self.configure_state != ConfigureState.TRAIN_STARTED)
            )
            dpg.configure_item(
                "save_checkpoint_checkbox", enabled=(self.configure_state != ConfigureState.TRAIN_STARTED)
            )

        for editor in self.editors:
            editor.enable(self.configure_state != ConfigureState.TRAIN_STARTED)

        control_text = "<None>"
        control_text_theme = "valid_text_theme"
        if self.configure_state == ConfigureState.CONFIGURE_STARTED:
            control_text = "Configure or Start Train."
        elif self.configure_state == ConfigureState.VALIDATION_FAILED:
            failure_reason = self.validation_reason.value
            control_text = f"Validation Failed with reason: {failure_reason}.\n Fix the issue and try again."
            control_text_theme = "invalid_text_theme"
        elif self.configure_state == ConfigureState.TRAIN_STARTED:
            control_text = "Configuration Validated. Training In Progress..."

        dpg.set_value("configure_state_text", control_text)
        dpg.bind_item_theme("configure_state_text", control_text_theme)

        mode_text = 'Configure'
        if self.configure_state == ConfigureState.TRAIN_STARTED:
            mode_text = f'Train: Run:{self.current_game_run + 1}/{self.num_game_runs}'
        dpg.set_value("rect_frame_mode_text", f"Mode: {mode_text}")

        mode_text_size = dpg.get_text_size(mode_text)
        if mode_text_size is not None:
            mode_text_pos = [(self.obs_dims[0] * self.max_rect_frame_scale) - (mode_text_size[0] * self.font_scale), 45]
            dpg.set_item_pos("rect_frame_mode_text", pos=mode_text_pos)

        if self.configure_state == ConfigureState.TRAIN_STARTED:
            dpg.set_value("rect_mode_label", "Train Mode:")
            dpg.configure_item("rect_mode_label", color=(255, 170, 0))
            dpg.set_value("rect_mode_detail", "Keyboard control disabled.")
            dpg.configure_item("rect_mode_detail", color=(180, 180, 180))
        else:
            dpg.set_value("rect_mode_label", "Configure Mode:")
            dpg.configure_item("rect_mode_label", color=(255, 100, 200))
            dpg.set_value(
                "rect_mode_detail", f"Use the following keys to test the Atari controller: {self.keyboard_controls}"
            )
            dpg.configure_item("rect_mode_detail", color=(200, 200, 200))

    def _draw_apriltag_boxes(self, cam_frame_fp32, tags):
        """Draw bounding boxes and labels around detected AprilTags based on apriltags.json config"""

        # AprilTags config, equivalent to apriltags.json
        # (id : semantic role)
        apriltag_roles = {
            0: "TOP_LEFT",
            1: "TOP_RIGHT",
            2: "BOTTOM_RIGHT",
            3: "BOTTOM_LEFT"
        }

        # Colors for each role
        role_colors = {
            "TOP_LEFT": (1.0, 0.0, 0.0),       # Red
            "TOP_RIGHT": (0.0, 1.0, 0.0),      # Green
            "BOTTOM_RIGHT": (0.0, 0.0, 1.0),   # Blue
            "BOTTOM_LEFT": (1.0, 1.0, 0.0),    # Yellow
        }

        if cam_frame_fp32 is None or not tags:
            return cam_frame_fp32

        frame = cam_frame_fp32.copy()
        cam_to_display_scale_x = self.display_cam_dims[0] / self.camera_dims[0]
        cam_to_display_scale_y = self.display_cam_dims[1] / self.camera_dims[1]

        for tag_id, corners in tags.items():
            role = apriltag_roles.get(tag_id)
            if role is None:
                continue
            color = role_colors.get(role, (1.0, 1.0, 1.0))

            # Scale corners to display dimensions
            corners_scaled = np.array(corners, dtype=np.float32)
            corners_scaled[:, 0] *= cam_to_display_scale_x
            corners_scaled[:, 1] *= cam_to_display_scale_y
            corners_scaled = corners_scaled.astype(np.int32)

            # Draw polygon around tag
            for i in range(4):
                pt1 = tuple(corners_scaled[i])
                pt2 = tuple(corners_scaled[(i + 1) % 4])
                self._draw_line_on_fp32(frame, pt1, pt2, color, thickness=2)

            # Draw tag label at first corner (role and ID)
            text_pos = tuple(corners_scaled[0])
            # Example: "ID:2 (BOTTOM_RIGHT)"
            self._draw_text_on_fp32(frame, f"ID:{tag_id} ({role})", text_pos, color)

        return frame

    def _draw_line_on_fp32(self, img, pt1, pt2, color, thickness=2):
        """Draw a line on a float32 RGB image (0-1 range)"""
        h, w = img.shape[:2]
        # Simple line drawing using NumPy (for thick lines, we'll just draw circles at points)
        x0, y0 = pt1
        x1, y1 = pt2

        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            # Draw a small circle for thickness
            for dy_off in range(-thickness//2, thickness//2 + 1):
                for dx_off in range(-thickness//2, thickness//2 + 1):
                    px, py = x0 + dx_off, y0 + dy_off
                    if 0 <= px < w and 0 <= py < h:
                        img[py, px] = color

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def _draw_text_on_fp32(self, img, text, pos, color):
        """Draw simple text on a float32 RGB image"""
        # For simplicity, just draw a filled rectangle as background and skip actual text rendering
        # DearPyGUI doesn't easily support text rendering on textures, so we'll just mark the position
        x, y = pos
        h, w = img.shape[:2]
        # Draw a small filled circle to mark the tag position
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                if dx*dx + dy*dy <= 64:  # Circle of radius 8
                    px, py = x + dx, y + dy
                    if 0 <= px < w and 0 <= py < h:
                        img[py, px] = color

    def scale_rect_frame(self, new_scale):
        self.rect_frame_scale = new_scale
        if self.score_detection_editor is not None:
            self.score_detection_editor.set_scale(new_scale)

    def run(self):
        self.frame_graphs = FrameGraphManager()
        self.episode_graphs = EpisodeGraphManager()
        self.reset_per_run_graphs = False

        # system stats (display data for max_system_stats_points)
        self.stats_graph_mgr = StatGraphManager(max_points=self.max_system_stats_points)

        # register metrics
        self.stats_graph_mgr.register_metric("gpu_temp", "gpu_temp_graph", "gpu_temp_x_axis", "gpu_temp_y_axis")
        self.stats_graph_mgr.register_metric("gpu_util", "gpu_util_graph", "gpu_util_x_axis", "gpu_util_y_axis")
        self.stats_graph_mgr.register_metric("gpu_util_mem", "gpu_mem_util_graph", "gpu_mem_x_axis", "gpu_mem_y_axis")
        self.stats_graph_mgr.register_metric("cpu_util", "cpu_util_graph", "sys_cpu_util_x_axis", "sys_cpu_util_y_axis")
        self.stats_graph_mgr.register_metric("total_mem", "sys_mem_graph", "sys_mem_x_axis", "sys_mem_y_axis")

        # there is not a way to resize a dearpygui texture during runtime without deleting from the registry and recreating (review: when deleting,
        # and recreating, using the same name failed). Instead, create a canvas of the max scale and inset the updated frame data.
        rect_frame_canvas = np.zeros(
            (
                self.obs_dims[1] * self.max_rect_frame_scale,
                self.obs_dims[0] * self.max_rect_frame_scale,
                self.obs_dims[2],
            ),
            dtype=np.float32,
        )

        cam_frame_fp32 = np.zeros((self.display_cam_dims[1], self.display_cam_dims[0], 3), dtype=np.float32)
        cam_frame_needs_update = False

        is_running = True

        target_fps = 30
        target_frame_time = 1.0 / target_fps
        last_frame_time = time.time()

        while dpg.is_dearpygui_running() and is_running:
            curr_time = time.time()

            # start = time.time()
            while not self.episode_queue.empty():
                try:
                    data = self.episode_queue.get_nowait()
                    if data is not None:
                        # logger.debug(data)
                        for key, val in data.items():
                            if key == "episode":
                                score, end = val
                                self.episode_graphs.add_episode(score, end)

                            elif key == "episode_avg":
                                self.episode_graphs.add_avg_score(data["episode_avg"])

                except queue.Empty:
                    break
            # logger.debug(f'ep queue: {(time.time()-start) * 1000.0:.2f}')

            # start = time.time()
            cam_frame, obs_frame, data_dict = self.shared_data.read_from_shmem()
            # logger.debug(f'read from shmem ttl {(time.time()-start) * 1000.0:.2f}')
            if cam_frame is not None:
                # start = time.time()
                # logger.debug(val.shape)
                if self.display_cam_dims[0] != cam_frame.shape[1] or self.display_cam_dims[1] != cam_frame.shape[0]:
                    cam_frame = cv2.resize(cam_frame, self.display_cam_dims, interpolation=cv2.INTER_LINEAR)
                cam_frame_fp32 = np.clip(cam_frame.astype(np.float32) * (1.0 / 255.0), 0.0, 1.0)
                cam_frame_needs_update = True
                # logger.debug(f'cam: {(time.time()-start) * 1000.0:.2f}')

            if obs_frame is not None:
                # h, w, c
                # start = time.time()
                assert obs_frame.shape[0] == self.obs_dims[1]
                assert obs_frame.shape[1] == self.obs_dims[0]
                assert obs_frame.shape[2] == self.obs_dims[2]
                val = obs_frame
                if self.rect_frame_scale != 1:
                    val = cv2.resize(
                        val, None, fx=self.rect_frame_scale, fy=self.rect_frame_scale, interpolation=cv2.INTER_LINEAR
                    )
                val_norm = np.clip(val.astype(np.float32) / 255.0, 0.0, 1.0)

                if self.prev_rect_frame_scale != self.rect_frame_scale:
                    rect_frame_canvas = rect_frame_canvas * 0
                    self.prev_rect_frame_scale = self.rect_frame_scale
                scaled_image = rect_frame_canvas
                # center the image within the canvas
                start_x = (rect_frame_canvas.shape[1] - val.shape[1]) // 2
                start_y = (rect_frame_canvas.shape[0] - val.shape[0]) // 2
                scaled_image[start_y : start_y + val.shape[0], start_x : start_x + val.shape[1], :] = val_norm
                dpg.set_value("rect_texture", scaled_image)
                # logger.debug(f"frame_rect={(time.time()-start)*1000:.2f}")

            if data_dict is not None:
                # start = time.time()
                if "frame" in data_dict:
                    frame_num = data_dict["frame"]
                    # when running out of process from the harness, keep the log frame_count
                    # in sync
                    set_frame_count(frame_num)
                    dpg.set_value('rect_frame_frame_text', f"Frame: {frame_num}")

                if "lives" in data_dict:
                    dpg.set_value(
                        "rect_frame_lives_text", f"Lives: {data_dict['lives']} / {data_dict.get('total_lives', '--')}"
                    )

                if "score" in data_dict:
                    dpg.set_value("rect_frame_score_text", f"Score: {data_dict['score']}")

                if "action" in data_dict:
                    chosen_action = data_dict["action"]
                    dpg.set_value("rect_frame_action_text", f"Action: {chosen_action}")
                    chosen_action_value = Action[chosen_action].value
                    self.frame_graphs.update_action(data_dict["frame"], chosen_action_value)
                if "fps" in data_dict:
                    dpg.set_value("rect_frame_fps_text", f"FPS: {data_dict['fps']:.2f}")

                if "tags" in data_dict:
                    if self.screen_detection_editor is not None:
                        self.screen_detection_editor.set_tags(data_dict["tags"])

                if "interframe_period" in data_dict:
                    self.frame_graphs.update_interframe(data_dict["frame"], data_dict["interframe_period"])

                if "reward_termination" in data_dict:
                    r, t = data_dict["reward_termination"]
                    self.frame_graphs.update_reward_termination(data_dict["frame"], r, t)

                if "run_complete" in data_dict:
                    logger.debug(f"Run {data_dict['run_complete']} Complete")
                    self.current_game_run += 1
                    self.reset_per_run_graphs = True
                    self.update_control_button_state()

                if "shutdown" in data_dict:
                    logger.debug("Shutdown detected.")
                    is_running = False
                    break

                # logger.debug(f'data dict : {(time.time()-start) * 1000.0:.2f}')

            if self.reset_per_run_graphs:
                self.frame_graphs.reset()
                self.episode_graphs.reset()
                self.reset_per_run_graphs = False

            cam_frame_dirty = False

            # Draw AprilTag bounding boxes if tags are detected
            if data_dict is not None and "tags" in data_dict and data_dict["tags"]:
                tags = data_dict["tags"]
                # Draw boxes around detected AprilTags
                cam_frame_fp32 = self._draw_apriltag_boxes(cam_frame_fp32, tags)
                cam_frame_dirty = True

            if self.screen_detection_editor is not None:
                cam_frame_fp32, cam_frame_dirty_editor = self.screen_detection_editor.compute_overlay(
                    cam_frame_fp32, alpha=0.4
                )
                cam_frame_dirty = cam_frame_dirty or cam_frame_dirty_editor

            if cam_frame_needs_update or cam_frame_dirty:
                dpg.set_value("camera_texture", np.ascontiguousarray(cam_frame_fp32))
                cam_frame_needs_update = False
                cam_frame_dirty = False

            for editor in self.editors:
                editor.draw()

            # update system stats and graphs (if visible)
            if curr_time >= self.next_system_stats_time_in_sec:
                # start = time.time()
                status, severity = self.health_monitor_thread.get_health_status()
                log_message(severity, f"Health: {status}")

                gpu_info = self.health_monitor_thread.get_gpu_stats() or {}
                sys_info = self.health_monitor_thread.get_sys_stats() or {}

                for key in ["gpu_temp", "gpu_util", "gpu_util_mem"]:
                    if key in gpu_info:
                        self.stats_graph_mgr.update_metric(key, float(gpu_info[key]))

                for key in ["cpu_util", "total_mem"]:
                    if key in sys_info:
                        self.stats_graph_mgr.update_metric(key, int(sys_info[key]))

                self.stats_graph_mgr.draw_visible()
                self.next_system_stats_time_in_sec = curr_time + self.system_stats_interval_in_sec
                # logger.debug(f"frame stats={(time.time()-start)*1000:2f}")

            # start = time.time()
            dpg.render_dearpygui_frame()
            # logger.debug(f"render={(time.time()-start)*1000:2f}")

            curr_time = time.time()
            target_time = last_frame_time + target_frame_time
            # delta_time = curr_time - last_frame_time
            sleep_time = target_time - curr_time
            if sleep_time < 0.0:
                # logger.warning(f"render frame took too long: dt={delta_time*1000.0:.2f}ms > target ft={target_frame_time*1000.0:.2f}ms")
                sleep_time = 0.0

            time.sleep(sleep_time)

            last_frame_time = time.time()

    # -----------------------
    # callbacks
    # -----------------------

    def write_setup_config(self):
        with open(self.setup_config) as cf:
            config_data = json.load(cf)

        config_data['num_runs'] = self.num_game_runs
        config_data['save_model'] = self.save_checkpoint
        if self.load_checkpoint is not None:
            config_data['load_model'] = self.load_checkpoint
        else:
            config_data.pop('load_model', None)

        with open(self.setup_config, "w") as file:
            json.dump(config_data, file)

    def start_game(self):
        logger.info("Configuration Complete. Starting Game.")

        # write the setup config before starting the game
        self.write_setup_config()

        assert self.content_thread is not None

        # Perform validation checks before starting training.
        valid_sr = self.content_thread.has_valid_screen_rect()
        if not valid_sr:
            self.validation_reason = ValidationFailureReason.VALIDATION_FAILURE_SCREEN_RECT
            self.configure_state = ConfigureState.VALIDATION_FAILED
            self.update_control_button_state()
        else:
            # stop the gui renderer, so the camera device is released
            self.content_thread.shutdown()

            assert self.configure_event
            self.configure_event.set()
            self.configure_state = ConfigureState.TRAIN_STARTED
            self.update_control_button_state()

    def quit(self):
        logger.info("Signal Exit.")
        if self.content_thread is not None:
            self.content_thread.shutdown()
            self.content_thread = None

        if self.configure_event and not self.configure_event.is_set():
            self.configure_event.set()

        if self.exit_event:
            self.exit_event.set()

    def set_num_game_runs(self, sender, app_data):
        self.num_game_runs = app_data

    def set_save_checkpoint(self, sender, app_data):
        self.save_checkpoint = app_data

    def set_load_checkpoint(self, sender, app_data):
        if app_data:
            # save as relative path
            self.load_checkpoint = os.path.relpath(app_data, start=os.getcwd())
            dpg.set_value("selected_checkpoint_path", f"{self.load_checkpoint}")
            dpg.configure_item("checkpoint_toggle_button", label="Clear Model")

    def toggle_checkpoint_selection(self, sender, app_data):
        if self.load_checkpoint is not None:
            # clear selection
            self.load_checkpoint = None
            dpg.set_value("selected_checkpoint_path", "None")
            dpg.configure_item("checkpoint_toggle_button", label="Select Model")
        else:
            # open file dialog
            dpg.show_item("checkpoint_file_dialog")

    def scale_rect_frame_callback(self, sender, app_data):
        self.scale_rect_frame(dpg.get_value(sender))

    def toggle_help(self, sender, app_data):
        if dpg.is_item_visible("help_menu"):
            dpg.hide_item("help_menu")
        else:
            dpg.show_item("help_menu")

    def toggle_system_stats(self, sender, app_data):
        if dpg.is_item_visible("sys_stats"):
            dpg.hide_item("sys_stats")
        else:
            dpg.show_item("sys_stats")

    def toggle_camera_controls(self, sender, app_data):
        if dpg.is_item_visible("camera_controls"):
            dpg.hide_item("camera_controls")
        else:
            dpg.show_item("camera_controls")

    def show_font_control(self, sender, app_data):
        dpg.show_font_manager()

    def show_debug_control(self, sender, app_data):
        dpg.show_debug()
        dpg.show_item_registry()

    # this will be called every frame that a window is focused
    def on_focus_handler(self, sender, app_data):
        # logger.debug(f"{dpg.get_frame_count()}: on_focus_handler: focused={app_data}")
        if app_data != self.focused_window:
            for editor in self.editors:
                editor.focus_changed_handler(sender, app_data)

            if app_data == self.rect_window:
                if self.content_thread is not None:
                    self.content_thread.set_input_focus(True)
            elif self.focused_window == self.rect_window:
                if app_data != self.rect_window:
                    if self.content_thread is not None:
                        self.content_thread.set_input_focus(False)
            self.focused_window = app_data

    def mouse_drag_handler(self, sender, app_data):
        for editor in self.editors:
            editor.mouse_drag_handler(sender, app_data)

        return False

    def mouse_click_handler(self, sender, app_data):
        for editor in self.editors:
            editor.mouse_click_handler(sender, app_data)

        return False

    def mouse_release_handler(self, sender, app_data):
        for editor in self.editors:
            editor.mouse_release_handler(sender, app_data)

    def key_press_handler(self, sender, app_data):
        if app_data == dpg.mvKey_Escape:
            logger.debug("ESCAPE")
            dpg.stop_dearpygui()
            return False

        for editor in self.editors:
            editor.key_press_handler(sender, app_data)

        # check for key combos
        mod_shift = dpg.is_key_down(dpg.mvKey_ModShift)

        if app_data == dpg.mvKey_H:
            self.toggle_help(sender, app_data)
            return True
        elif app_data == dpg.mvKey_P:
            self.toggle_system_stats(sender, app_data)
            return True
        elif mod_shift and (app_data == dpg.mvKey_D):
            self.show_debug_control(sender, app_data)
            return True
        elif app_data == dpg.mvKey_F:
            self.show_font_control(sender, app_data)
            return True

        return False


def create_gui_process(
    game_config,
    joystick_config,
    camera_config,
    detection_config,
    score_detector_type,
    obs_dims,
    episode_queue,
    shared_data,
    configure_event=None,
    exit_event=None,
):
    logger.info(f"Physical GUI running at pid={os.getpid()}")
    physical_gui = None

    try:
        physical_gui = PhysicalGui(
            game_config,
            joystick_config,
            camera_config,
            detection_config,
            score_detector_type,
            obs_dims,
            episode_queue,
            shared_data,
            configure_event=configure_event,
            exit_event=exit_event,
        )
        # blocks until termination event received.
        physical_gui.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in GUI process.")
    except Exception as e:
        logger.error(f"Unexpected error in GUI process: {e}", exc_info=True)
    finally:
        logger.info("Physical gui closing...")
        if physical_gui is not None:
            physical_gui.shutdown()
        logger.info("Physical gui shutdown complete")
    sys.exit(0)
