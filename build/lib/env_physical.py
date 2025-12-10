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
import os
import time
from enum import Enum

import cv2
import numpy as np
import torch

from env_base import BaseEnv
from framework import ScoreDetectorConfig
from framework.Actions import Action
from framework.CameraDevice_v4l2 import CameraDevice_v4l2 as CameraDevice
from framework.ControlDeviceCfg import create_control_device_from_cfg
from framework.Joystick import Joystick
from framework.Logger import logger, set_frame_count

supported_games = ["atlantis", "battle_zone", "centipede", "defender", "krull", "ms_pacman", "qbert", "up_n_down"]

# The current logic to determine game_over (technically start of new game)
# is based on game-specific criteria, ie the score has reset to 0 and lives
# are reset to full. That determination requires the game to have been reset,
# which in the set of games currently supported, requires the 'FIRE' button
# to have been pressed at the end of the episode. For games with full_action_space
# in exploration mode, the agent sending a FIRE action within a short period of
# time is likely, but not guaranteed. For games using minimal action sets which
# don't include 'FIRE' this will not be the case. Inject logic to issue a 'FIRE'
# command at a specific frequency once game-specific criteria has been met.

# REVIEW: The current implementation acts outside of the agent, and issues a FIRE command
# based on a reset hint (the episode is likely completing soon or already completed).
# Should the agent 'learn' the reset action as part of curiosity-driven exploration?
# When the hint returns a reset is likely required, should a small negative reward be returned
# until the reset action has been taken? Or the hint flag is added to the agents state?
# Another possibility, hierarchical RL (options/sub-policies) to help learn the reset in
# a more structured way without affecting performance during an episode.

# For some games detecting lives is not possible, or too difficult (ie, atlantis where we'd
# need a recognizer for the different cities (6)). Support multiple hint types to accommodate.


class GameResetHintType(Enum):
    GAME_RESET_HINT_NONE = 0
    GAME_RESET_HINT_SCORE_ONLY = 1
    GAME_RESET_HINT_SCORE_LIVES = 2


class PhysicalEnv(BaseEnv):
    def __init__(
        self,
        game_config,
        camera_config,
        joystick_config,
        detection_config,
        score_detector_type,
        obs_dims=(160, 210, 3),
        reduce_action_set=0,
        device='cpu',
        data_dir=None,
    ):
        # NOTE: Only one module should set the global frame_count.
        # For now, this is handled by the physical env as it owns the camera object,
        # but ultimately the harness might be a better location as it triggers env update.
        set_frame_count(-1)

        self.obs_dims = obs_dims
        self.device = device
        self.data_dir = data_dir

        self.score_detector_type = score_detector_type
        self.joystick = None
        self.camera = None

        # Initialize game-specific components
        with open(game_config) as gf:
            game_data = gf.read()

        game_config = json.loads(game_data)["game_config"]

        self.env_name = game_config["name"]
        assert self.env_name in supported_games, f"Unsupported game {self.env_name}"
        self.total_lives = game_config["lives"]

        legal_action_set = [act for act in Action]

        minimal_action_set = []
        minimal_actions = game_config['minimal_actions']
        if minimal_actions is not None:
            for action in minimal_actions:
                if not Action.has_key(action):
                    logger.warning(f"minimal_action {action} is not a valid action. Skipping.")
                    continue
                minimal_action_set.append(Action[action])
        elif reduce_action_set != 0:
            logger.warning("PhysicalEnv: minimal action space requested but no minimal actions were provided.")
            reduce_action_set = 0

        self.action_set = legal_action_set if (reduce_action_set == 0) else minimal_action_set
        if reduce_action_set == 2 and (self.env_name == 'ms_pacman' or self.env_name == 'qbert'):
            self.action_set = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

        # logger.debug(f"action_set={self.action_set}")

        self.has_fire_in_actionset = Action.FIRE in self.action_set

        # load custom user offsets
        score_offsets = {"offset_x": 0, "offset_y": 0}
        lives_offsets = {"offset_x": 0, "offset_y": 0}
        user_offset_file = "configs/.user_offsets.json"
        if os.path.exists(user_offset_file):
            logger.debug(f"PhysicalEnv: {user_offset_file=}")
            with open(user_offset_file) as f:
                offsets = json.load(f)
                if self.env_name in offsets:
                    game_offsets = offsets[self.env_name]
                    if "score_crop_offset" in game_offsets:
                        score_offsets = game_offsets["score_crop_offset"]
                    if "lives_crop_offset" in game_offsets:
                        lives_offsets = game_offsets["lives_crop_offset"]

        score_model_type = ScoreDetectorConfig.get_model_type(self.score_detector_type)
        if score_model_type == "network":
            from framework.ScoreDetector import ScoreDetector

            score_config = json.loads(game_data)["score_config"]
            self.score_detector = ScoreDetector(
                self.env_name,
                self.score_detector_type,
                self.total_lives,
                **score_config,
                score_offsets=score_offsets,
                lives_offsets=lives_offsets,
                device=self.device,
                data_dir=self.data_dir,
            )
        else:
            raise ValueError(f"Invalid Score detection type {self.score_detector_type}.")

        # Initialize system components
        with open(camera_config) as cf:
            camera_data = cf.read()

        camera_data = json.loads(camera_data)
        camera_name = camera_data["model_name"]
        camera_config = camera_data["camera_config"]
        self.camera = None

        # verify the camera has read access
        retries = 2
        for retry in range(retries):
            self.camera = CameraDevice(camera_name, **camera_config)
            assert self.camera is not None
            if self.camera.validate():
                break

            logger.warning(f"Failed to validate camera: {retry + 1}/{retries} retries...")
            self.camera.shutdown()
            self.camera = None

        if self.camera is None:
            raise ValueError("Failed to read frames from camera.")
        else:
            logger.debug("Camera validated.")

        with open(joystick_config) as jf:
            joystick_data = jf.read()

        joystick_data = json.loads(joystick_data)
        joystick_device = create_control_device_from_cfg(**joystick_data)
        self.joystick = Joystick(joystick_device)

        with open(detection_config) as df:
            detection_data = df.read()
        detection_data = json.loads(detection_data)
        method_name = detection_data["name"]
        detection_config = detection_data["detection_config"]
        if method_name == "fixed":
            from framework.ScreenDetectorFixed import ScreenDetectorFixed

            self.screen_detector = ScreenDetectorFixed(method_name, **detection_config)
        else:
            from framework.ScreenDetector import ScreenDetector

            self.screen_detector = ScreenDetector(method_name, detection_data["corners"], **detection_config)

        # Verify the screen can be detected before proceeding.
        logger.debug("Testing for valid screen rect")
        screen_rect = None
        camera_frame = None
        for _ in range(1000):
            frame_data = self.camera.get_frame()
            camera_frame = frame_data["frame"]
            assert camera_frame is not None

            frame_g = self.camera.convert_to_grayscale(camera_frame)
            screen_rect, _ = self.screen_detector.get_screen_rect_info(frame_g)
            if screen_rect is not None:
                break

        if screen_rect is None:
            raise ValueError("Could not obtain valid screen rect.")

        self.lives_ = self.total_lives
        self.game_over_ = False
        self.frames_since_game_over = 0
        # at game reset, some games do not reset score and lives on the same frame, triggering two
        # different reset detection events.
        self.use_frames_since_game_over_quirk = (
            self.env_name == 'krull'
            or self.env_name == 'centipede'
            or self.env_name == 'defender'
            or self.env_name == 'battle_zone'
            or self.env_name == 'qbert'
        )
        self.observation_cam_frame_num = -1
        self.observation_cam = None
        self.observation_rect = None
        self.past_observation_cam = []

        self.rect_target_width = self.obs_dims[0] * 2
        self.rect_target_height = self.obs_dims[1] * 2
        logger.info(f"Target dimensions for rectification={self.rect_target_width}x{self.rect_target_height}")

        # score detection quirks
        self.prev_lives = self.lives_
        self.prev_score = 0
        self.last_score_change_frame = -1

        # Some games do not display the total lives, when detecting number of lives, we need to increment by the
        # number of lives not shown. Of the supported games, these games do not display the number of total lives:
        # ['ms_pacman', 'centipede', "up_n_down", 'qbert', 'krull']
        self.lives_increment = 0
        if self.score_detector.lives_crop_info:
            num_displayed_lives = self.score_detector.lives_crop_info.num_digits
            self.lives_increment = max(0, self.total_lives - num_displayed_lives)

        self.game_reset_hint_type = GameResetHintType.GAME_RESET_HINT_NONE
        if self.score_detector.supports_lives():
            self.game_reset_hint_type = GameResetHintType.GAME_RESET_HINT_SCORE_LIVES
            self.reset_hint_frames_no_change = 300 if self.env_name != "centipede" else 100
        else:
            self.game_reset_hint_type = GameResetHintType.GAME_RESET_HINT_SCORE_ONLY
            self.reset_hint_frames_no_change = 1200

        self.reset()

        # hit FIRE to start the game in case the unit is in demo mode.
        if not self.has_fire_in_actionset:
            self.joystick.apply_action(Action.FIRE)

    def close(self):
        if self.screen_detector is not None:
            self.screen_detector.shutdown()

        if self.joystick is not None:
            self.joystick.shutdown()

        if self.camera is not None:
            self.camera.shutdown()

    def get_name(self):
        return self.env_name

    def get_action_set(self):
        return self.action_set

    def reset(self):
        self.lives_ = self.total_lives
        self.game_over_ = False

        self.prev_lives = self.lives_
        self.prev_score = 0
        self.last_score_change_frame = -1

    # Returns True if the episode is likely completing soon or already completed.
    def _check_reset_hint(self, frame_num):
        if self.use_frames_since_game_over_quirk and (self.frames_since_game_over < 4):
            return True
        elif self.game_reset_hint_type == GameResetHintType.GAME_RESET_HINT_SCORE_LIVES:
            if self.lives_ <= 1 and (
                self.last_score_change_frame != -1
                and (frame_num - self.last_score_change_frame) > self.reset_hint_frames_no_change
            ):
                return True
        elif self.game_reset_hint_type == GameResetHintType.GAME_RESET_HINT_SCORE_ONLY:
            if (
                self.last_score_change_frame != -1
                and (frame_num - self.last_score_change_frame) > self.reset_hint_frames_no_change
            ):
                return True

        return False

    # screen_rect is oriented CW starting at top-left: TL,TR,BR,BL
    def _rectify_frame(self, frame, screen_rect, target_width, target_height) -> np.ndarray:
        # assert screen_rect.shape == (4, 2), f"Invalid screen_rect shape: {screen_rect.shape}"
        target_rect = np.float32([(0, 0), (target_width, 0), (target_width, target_height), (0, target_height)])
        transform = cv2.getPerspectiveTransform(screen_rect, target_rect)
        rect_frame = cv2.warpPerspective(frame, transform, (target_width, target_height), flags=cv2.INTER_LINEAR)
        if len(rect_frame.shape) == 2:
            rect_frame = np.expand_dims(rect_frame, axis=-1)
        return rect_frame

    def _update_score(self, observation) -> float:
        score, self.lives_ = self.score_detector.get_score_and_lives(observation)
        if self.lives_ is None:
            self.lives_ = self.total_lives
        else:
            self.lives_ += self.lives_increment

        if self.use_frames_since_game_over_quirk and (self.frames_since_game_over < 20):
            pass
        elif self.game_reset_hint_type == GameResetHintType.GAME_RESET_HINT_SCORE_LIVES:
            if (self.prev_score != 0 and score == 0) or (self.lives_ >= self.total_lives and self.prev_lives <= 1):
                logger.info(
                    f"GAME OVER: prev_score: {self.prev_score} curr_score:{score} prev_lives:{self.prev_lives} curr_lives:{self.lives_}"
                )
                self.game_over_ = True
        elif self.game_reset_hint_type == GameResetHintType.GAME_RESET_HINT_SCORE_ONLY:
            if self.prev_score != 0 and score == 0:
                logger.info(f"GAME OVER: prev_score: {self.prev_score} curr_score:{score}")
                self.game_over_ = True

        # handle invalid score (if score detector misses a score consistency check)
        # revisit if we support games with negative rewards
        reward = 0 if self.game_over_ else max(score - self.prev_score, 0)
        # logger.debug(f"act: score={score} reward={reward} game_over={self.game_over_}")

        self.prev_score = score
        self.prev_lives = self.lives_

        return reward

    def act(self, action_):
        # get the raw observation
        # start = time.time()
        torch.cuda.nvtx.range_push("camera_frame_read")
        frame_data = self.camera.get_frame()
        torch.cuda.nvtx.range_pop()

        frame_number = frame_data["frame_number"]
        frame = frame_data["frame"]
        assert frame is not None
        # logger.debug(f"camera: {(time.time()-start)*1000.0}")

        # update the global logger with the current camera frame number.
        set_frame_count(frame_number)

        action = action_

        # check for reset hint, and if active, append FIRE to the
        # action.
        if self._check_reset_hint(frame_number):
            # logger.debug(f"Reset Hint action active.")
            if 'FIRE' not in action.name:
                fire_action_name = f"{action.name}FIRE" if action != Action.NOOP else "FIRE"
                if fire_action_name in Action.__members__:
                    action = Action[fire_action_name]
                else:
                    logger.debug(f"No FIRE variant available for action: {action.name}")

        # begin_time = time.time()
        # start = time.time()
        torch.cuda.nvtx.range_push("apply_action")
        self.joystick.apply_action(action)
        torch.cuda.nvtx.range_pop()
        # logger.debug(f"apply_action: {(time.time()-start)*1000.0}")

        torch.cuda.nvtx.range_push("screen_detect_threaded")
        # provide the latest camera frame for next frame processing, and
        # get the last update from screen detection processing; assumes raw camera data codec is YUYV
        # NOTE: last_detected_tags may be invalid or incomplete (num_tags != 4) and should only
        # be used for debug purposes.
        frame_g = self.camera.convert_to_grayscale(frame)
        screen_rect, last_detected_tags = self.screen_detector.get_screen_rect_info(frame_g)
        torch.cuda.nvtx.range_pop()
        assert screen_rect is not None

        # start = time.time()
        if self.score_detector.tmp_score_crop_img is not None:
            self.past_observation_cam.append(self.score_detector.tmp_score_crop_img)  # for debugging scores
            self.past_observation_cam = self.past_observation_cam[-10:]
        self.observation_cam_frame_num = frame_number

        torch.cuda.nvtx.range_push("color_convert")
        # color convert if color_mode specified as 'rgb'
        if self.obs_dims[2] == 3:
            self.observation_cam = self.camera.convert_to_rgb(frame)
        else:
            self.observation_cam = np.expand_dims(frame_g, axis=-1)
        torch.cuda.nvtx.range_pop()

        # rectify the observation
        torch.cuda.nvtx.range_push("rectify")
        self.observation_rect = self._rectify_frame(
            self.observation_cam, screen_rect, self.rect_target_width, self.rect_target_height
        )
        torch.cuda.nvtx.range_pop()
        # logger.debug(f"rectify + color convert: {(time.time()-start)*1000.0}")

        # determine reward, lives, game_over using the camera frame resolution
        # start = time.time()
        torch.cuda.nvtx.range_push("score")
        reward = self._update_score(self.observation_rect)
        if self.game_over_ and self.frames_since_game_over > 20:
            self.frames_since_game_over = 0
        else:
            self.frames_since_game_over += 1
        torch.cuda.nvtx.range_pop()
        # logger.debug(f"update score: {(time.time()-start)*1000.0}")
        if reward != 0:
            self.last_score_change_frame = frame_number

        # output to expected obs_size
        # start = time.time()
        if self.observation_rect.shape[0] != self.obs_dims[1] or self.observation_rect.shape[1] != self.obs_dims[0]:
            torch.cuda.nvtx.range_push("resize_obs")
            self.observation_rect = cv2.resize(
                self.observation_rect, (self.obs_dims[0], self.obs_dims[1]), interpolation=cv2.INTER_LINEAR
            )
            torch.cuda.nvtx.range_pop()
        # logger.debug(f"resize: {(time.time()-start)*1000.0}")

        info = {
            "score": self.prev_score,
            "tags": last_detected_tags,
            "total_lives": self.total_lives,
        }

        # logger.debug(f"total: {(time.time()-begin_time)*1000.0}")
        return reward, info

    def game_over(self) -> bool:
        return self.game_over_

    def lives(self) -> int:
        return self.lives_

    def get_observation(self) -> np.ndarray:
        return self.observation_rect

    def get_camera_frame(self) -> tuple[int, np.ndarray]:
        return self.observation_cam_frame_num, self.observation_cam
