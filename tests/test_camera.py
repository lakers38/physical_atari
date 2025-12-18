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

import cv2

from framework.CameraDevice_v4l2 import CameraDevice_v4l2 as CameraDevice


def main(args):
    with open(args.camera_config) as cf:
        camera_data = cf.read()

    camera_data = json.loads(camera_data)
    camera_name = camera_data["model_name"]
    camera_config = camera_data["camera_config"]
    camera = CameraDevice(camera_name, **camera_config)

    try:
        print("Starting performance test...")
        start_time = time.time()
        for _ in range(600):
            _ = camera.get_frame()
        print(f"Total_time to read 600 frames={(time.time() - start_time)}s")

        target_fps = camera.get_fps()
        frames = 0
        total_time = 0.0
        while True:
            start_time = time.time()
            _frame_data = camera.get_frame()
            total_time += time.time() - start_time
            frames += 1

            if frames == target_fps:
                print(f"Camera FPS={(target_fps / total_time):.2f}")
                frames = 0
                total_time = 0.0

            """
            start_time = time.time()
            frame = camera.convert_to_rgb(_frame_data["frame"])
            #print(f"convert: {(time.time()-start_time)*1000.0:.2f}")
            cv2.imshow("Camera", frame)
            """
            keycode = cv2.pollKey()
            if keycode == 27:  # Escape Key
                break

    except KeyboardInterrupt:
        pass

    finally:
        camera.shutdown()


def get_argument_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="camera_test.py arguments")
    parser.add_argument('--camera_config', type=str, default="configs/cameras/camera_kiyo_pro.json")
    return parser


if __name__ == '__main__':
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    main(args)
