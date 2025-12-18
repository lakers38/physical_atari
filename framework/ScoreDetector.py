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

import contextlib
import os
import time

import numpy as np
import torch
from typing import Optional

from framework.Logger import logger


class CropInfo:
    def __init__(self, x, y, w, h, num_digits):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.num_digits = num_digits
        # REVIEW: move to config, or assume atari dims?
        self.reference_width = 160
        self.reference_height = 210

    def __str__(self):
        return f"crop=(x={self.x}, y={self.y}, w={self.w}, h={self.h})"


class ScoreDetector:
    def __init__(
        self,
        env_name,
        model_type,
        total_lives,
        checkpoint,
        score_crop_info,
        lives_crop_info=None,
        valid_jumps=[],
        score_offsets={},
        lives_offsets={},
        device="cpu",
        data_dir=None,
    ):

        assert os.path.exists(checkpoint)

        logger.info(f"ScoreDetector: loading score model: {checkpoint} to device={device}")

        self.data_dir = os.getcwd() if data_dir is None else data_dir
        self.env_name = env_name
        self.device = device

        self.score_crop_info = CropInfo(**score_crop_info)
        if score_offsets:
            self.score_crop_info.x += score_offsets["offset_x"]
            self.score_crop_info.y += score_offsets["offset_y"]

        # lives is optional
        if lives_crop_info:
            self.lives_crop_info = CropInfo(**lives_crop_info)
            if lives_offsets:
                self.lives_crop_info.x += lives_offsets["offset_x"]
                self.lives_crop_info.y += lives_offsets["offset_y"]
        else:
            self.lives_crop_info = None

        self.score_validator = None

        self.input_memory_format = torch.contiguous_format
        self.use_mixed_precision = False

        self.model_type = model_type

        load_start_time = time.time()
        if self.model_type == "crnn_ctc":
            from framework.models.score_detector.crnn_ctc import load_model

            self.input_memory_format = torch.channels_last
            self.use_mixed_precision = False if self.device == 'cpu' else True
            self.model = load_model(
                checkpoint,
                self.env_name,
                device=self.device,
                mixed_precision=self.use_mixed_precision,
                memory_format=self.input_memory_format,
            )

            from framework.ScoreValidator import ScoreValidator

            self.validator = ScoreValidator(
                env_name,
                valid_jumps,
                displayed_lives=self.lives_crop_info.num_digits if self.lives_crop_info is not None else total_lives,
                entropy_threshold=self.model.entropy_threshold,
                entropy_ceiling=self.model.entropy_ceiling,
            )
        else:
            raise ValueError(f"Invalid score model type={self.model_type}")

        print(f"score model load: {(time.time() - load_start_time) * 1000.0:.2f}ms")
        # print(self.model)

        # track changes to the regions to avoid unnecessary invocations of the score model
        self.last_score_region = None
        self.last_lives_region = None
        self.region_changed_threshold = 0.8

        self.tmp_score_crop_img = None

        self.ave_time_model = 0
        self.ave_time_total = 0
        self.frames = 0

    def supports_lives(self):
        return self.lives_crop_info is not None

    def get_score_crop_info(self, frame_width, frame_height):
        return self.__adjust_crop(frame_width, frame_height, self.score_crop_info)

    def get_lives_crop_info(self, frame_width, frame_height):
        if self.lives_crop_info:
            return self.__adjust_crop(frame_width, frame_height, self.lives_crop_info)
        else:
            return None

    # np.ndarray: h,w,c
    def get_score_and_lives(self, frame) -> tuple[int, Optional[int]]:
        # total_start = time.time()
        has_lives = self.lives_crop_info is not None
        frame_w, frame_h = frame.shape[1], frame.shape[0]

        score_crop = self.__crop_region(frame, self.get_score_crop_info(frame_w, frame_h), channels_first=False)
        lives_crop = (
            self.__crop_region(frame, self.get_lives_crop_info(frame_w, frame_h), channels_first=False)
            if has_lives
            else None
        )

        self.tmp_score_crop_img = score_crop

        score_crop = self.model.preprocess(score_crop)
        if lives_crop is not None:
            lives_crop = self.model.preprocess(lives_crop, is_lives=True)

        combined_crop = (
            torch.stack([score_crop, lives_crop], dim=0) if lives_crop is not None else torch.stack([score_crop], dim=0)
        )
        combined_crop = combined_crop.contiguous(memory_format=self.input_memory_format)

        # model_start = time.time()
        with torch.no_grad():
            autocast_ctx = (
                torch.amp.autocast(device_type=self.device, dtype=torch.float16)
                if self.use_mixed_precision
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                decoded, confidences = self.model.predict(combined_crop)

        # self.ave_time_model += (time.time() - model_start)

        decoded = decoded.cpu().numpy()
        confidences = confidences.cpu().numpy()

        score, score_confidences, lives, lives_confidences = self.model.convert(
            decoded[0], confidences[0], decoded[1] if has_lives else None, confidences[1] if has_lives else None
        )

        if self.validator is not None:
            valid_score, valid_lives = self.validator.validate(
                score, score_confidences, pred_lives=lives, lives_confidences=lives_confidences
            )
        else:
            valid_score = score
            valid_lives = lives

        # logger.debug(f"{score_preds} -> {valid_score}, {lives_preds} -> {valid_lives}")

        # self.ave_time_total += (time.time()-total_start)
        # self.frames += 1
        # if (self.frames % 10000) == 0:
        #    print(f"model ave: {(self.ave_time_model/self.frames)*1000.0:.2f}/ {(self.ave_time_total/self.frames)*1000.0:.2f}ms")
        #    self.ave_time = 0
        #    self.ave_time_total = 0
        #    self.frames = 0

        return valid_score, valid_lives if has_lives else None

    def __adjust_crop(self, width, height, crop):
        if width == crop.reference_width and height == crop.reference_height:
            return crop

        scale_x = width / crop.reference_width
        scale_y = height / crop.reference_height

        x = int(crop.x * scale_x)
        y = int(crop.y * scale_y)
        x_off = int(crop.w * scale_x)
        y_off = int(crop.h * scale_y)

        return CropInfo(x, y, x_off, y_off, crop.num_digits)

    def __crop_region(self, x, crop_info, channels_first=False):
        if channels_first:
            cropped_image = x[:, crop_info.y : crop_info.y + crop_info.h, crop_info.x : crop_info.x + crop_info.w]
        else:
            cropped_image = x[crop_info.y : crop_info.y + crop_info.h, crop_info.x : crop_info.x + crop_info.w, :]
        return cropped_image

    def _crops_changed(self, score_crop, lives_crop):
        # check for changes to the score and lives regions, if no changes (within a threshold) are detected, return
        # previous values
        score_changed = True
        if self.last_score_region is not None:
            error = np.mean((score_crop - self.last_score_region) ** 2)
            # print(f"score_error={error}")
            score_changed = error >= self.region_changed_threshold

        lives_changed = lives_crop is not None
        if not score_changed and self.last_lives_region is not None and lives_crop is not None:
            error = np.mean((lives_crop - self.last_lives_region) ** 2)
            # print(f"lives_error={error}")
            lives_changed = error >= self.region_changed_threshold

        self.last_score_region = score_crop
        self.last_lives_region = lives_crop

        return score_changed or lives_changed
