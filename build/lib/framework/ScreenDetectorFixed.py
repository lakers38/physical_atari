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

import cv2
import numpy as np

from framework.Logger import logger


class ScreenDetectorFixed:
    def __init__(self, method_name, screen_rect=None, mspacman_rect=None):
        assert method_name == "fixed"
        self.last_detected_tags = {}
        # A mspacman_rect replaces any screen_rect
        if mspacman_rect:  # mspacman_rect in camera pixels, clockwise from upperleft
            # The ALE coordinates of the visible box in mspacman
            mx, my = 160, 171
            mspacman_rect_ale = np.array([(0, 0), (mx, 0), (mx, my), (0, my)], dtype=np.float32)
            mspacman_rect = np.array(mspacman_rect, dtype=np.float32)
            transform = cv2.getPerspectiveTransform(mspacman_rect_ale, mspacman_rect)

            sx, sy = 160, 210
            source_rect = np.array([(0, 0, 1), (sx, 0, 1), (sx, sy, 1), (0, sy, 1)], dtype=np.float32)
            screen_rect = []
            for point in source_rect:
                q = np.matmul(transform, point)
                screen_rect.append((q[0] / q[2], q[1] / q[2]))

            self.screen_rect = np.array(screen_rect, dtype=np.float32)

        elif screen_rect is not None:
            self.screen_rect = np.array(screen_rect, dtype=np.float32)
        else:
            raise ValueError("ScreenDetectorFixed requires either mspacman_rect or screen_rect")

    def shutdown(self):
        pass

    def get_screen_rect_info(self, _frame_g):
        return self.screen_rect, self.last_detected_tags
