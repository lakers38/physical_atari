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

import dt_apriltags
import numpy as np

from framework.Logger import logger

"""
Tag<id>s are expected to be oriented in a CW fashion starting with tag<0> at the top-left corner
of the monitor:

0 -----> 1
|        |
|        |
3<------ 2

The orientation of the tag matters as specific corners are used to determine the screen rect.
Tags should be applied in their default orientation.

dt_apriltags returns tag corner winding is CCW with 0 at the bottom-left.
3<------- 2
|         |
|         |
0 ------->1
"""


class TagID(Enum):
    TAG_ID_TOP_LEFT = 0
    TAG_ID_TOP_RIGHT = 1
    TAG_ID_BOTTOM_RIGHT = 2
    TAG_ID_BOTTOM_LEFT = 3


class ScreenDetector:
    def __init__(
        self, method_name, corners, family, quad_decimate, quad_sigma, refine_edges, decode_sharpening, threaded=True
    ):
        assert method_name == "dt_apriltags"
        self.detector = dt_apriltags.Detector(
            families=family,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
        )

        # Mapping of which corner of the tag should be used
        # to define the screen_rect.
        self.tag_id_corner_idx = {TagID[key].value: idx for key, idx in corners.items()}
        self.threaded = threaded

        self.screen_rect = None
        self.last_detected_tags = {}

        if self.threaded:
            self.running = True
            self.frame_queue = queue.Queue(maxsize=4)
            self.shutdown_cond = threading.Condition()
            self.lock = threading.Lock()
            self.detection_thread = threading.Thread(target=self._process_frames, daemon=True)
            self.detection_thread.start()

    def shutdown(self):
        if self.threaded:
            self.running = False
            with self.shutdown_cond:
                self.shutdown_cond.notify()
            self.detection_thread.join()
        self.detector = None

    # Expects grayscale np.ndarray
    def get_screen_rect_info(self, frame_g):
        if self.threaded:
            # add a frame to the queue for processing and return the previous valid
            # screen info
            try:
                self.frame_queue.put_nowait(frame_g)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame_g)
                except (queue.Full, queue.Empty):
                    pass  # still full or already drained
                except Exception:
                    # logger.warning(f"ScreenDetector: Unexpected queue error: {e}")
                    pass

            with self.lock:
                return self.screen_rect, self.last_detected_tags
        else:
            screen_rect, tag_data = self._detect_screen(frame_g)
            self.screen_rect = screen_rect
            self.last_detected_tags = tag_data
            return screen_rect, tag_data

    def _detect_screen(self, frame_g):
        if frame_g is None:
            logger.warning("ScreenDetector::_detect_screen: invalid frame")
            return None, None

        tags = self._detect_tags(frame_g)
        tag_data = {tag.tag_id: tag.corners for tag in tags}

        if len(tags) == 4:
            sr_pt_dict = {
                tag.tag_id: (
                    tag.corners[self.tag_id_corner_idx[tag.tag_id]][0],
                    tag.corners[self.tag_id_corner_idx[tag.tag_id]][1],
                )
                for tag in tags
            }
            screen_rect = np.float32([sr_pt_dict[i] for i in range(4)])
        else:
            screen_rect = None
            # logger.debug(f"ScreenDetector:_detect_screen: Only detected {len(tags)} tags. Verify camera position and lighting")

        return screen_rect, tag_data

    def _detect_tags(self, frame_g):
        tags = self.detector.detect(frame_g)

        # verify the correct tags have been identified
        # in low-lighting conditions, tags can be mis-identified.
        i = 0
        while i < len(tags):
            if tags[i].tag_id not in self.tag_id_corner_idx:
                logger.warning(f"ScreenDetector: invalid tag_id found = {tags[i].tag_id}")
                tags.pop(i)
            else:
                i += 1

        return tags

    def _process_frames(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                screen_rect, tag_data = self._detect_screen(frame)

                with self.lock:
                    # only update screen_rect with valid data
                    if screen_rect is not None:
                        self.screen_rect = screen_rect

                    # always update tag info as it can be used to determine why screen detection failed.
                    self.last_detected_tags = tag_data

            with self.shutdown_cond:
                self.shutdown_cond.wait(1)
