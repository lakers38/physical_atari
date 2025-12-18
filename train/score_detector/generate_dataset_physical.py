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
import random
import re
import shutil
import subprocess
import time

import cv2
import numpy as np
import pygame
from ale_py import Action, ALEInterface, LoggerMode, roms
from .ale_ram_injection import GAME_RAM_CONFIG, decode_lives, decode_score_bcd, write_lives, write_score
from PIL import Image

ASPECT_RATIO = 4 / 3
ATARI_WIDTH = 160

# TIA: for NTSC expected 262 scanlines per frame / 60hz
# and for PAL expected 312 scanlines per frame / 50Hz

# ALE does not use the full number of physical scanlines
# instead it only processes the 'visible' area that's
# typically drawn by games in an effort to normalize
# the observations across games (discarding any blanking
# regions or offscreen artifacts)
ATARI_HEIGHT_NTSC = 210
ATARI_HEIGHT_PAL = 250

ATARI_HEIGHT = ATARI_HEIGHT_NTSC


def get_connected_displays():
    result = subprocess.run(["xrandr", "--query"], stdout=subprocess.PIPE, text=True)
    output = result.stdout

    regex = re.compile(r"(?P<name>\S+)\s+connected(?:\s+primary)?\s+(?P<res>\d+x\d+)\+(?P<x>\d+)\+(?P<y>\d+)")
    displays = []
    for match in regex.finditer(output):
        width, height = map(int, match.group("res").split("x"))
        displays.append(
            {
                "name": match.group("name"),
                "x": int(match.group("x")),
                "y": int(match.group("y")),
                "width": width,
                "height": height,
            }
        )
    return displays


def get_atari_region2(monitor_width, monitor_height):
    content_h = int(monitor_width * 3 / 4)
    if content_h <= monitor_height:
        x0 = 0
        y0 = (monitor_height - content_h) // 2
        return (x0, y0, monitor_width, content_h)
    else:
        content_w = int(monitor_height * 4 / 3)
        x0 = (monitor_width - content_w) // 2
        y0 = 0
        return (x0, y0, content_w, monitor_height)


def get_atari_region(monitor_width, monitor_height):
    aspect_ratio = 4 / 3
    content_w = int(monitor_height * aspect_ratio)
    if content_w <= monitor_width:
        x0 = (monitor_width - content_w) // 2
        y0 = 0
        return (x0, y0, content_w, monitor_height)
    else:
        content_h = int(monitor_width / aspect_ratio)
        x0 = 0
        y0 = (monitor_height - content_h) // 2
        return (x0, y0, monitor_width, content_h)


def rectify_frame(frame, screen_rect, target_width, target_height):
    target_rect = np.float32([(0, 0), (target_width, 0), (target_width, target_height), (0, target_height)])
    transform = cv2.getPerspectiveTransform(screen_rect, target_rect)
    warped = cv2.warpPerspective(frame, transform, (target_width, target_height), flags=cv2.INTER_LINEAR)
    return np.expand_dims(warped, axis=-1) if len(warped.shape) == 2 else warped


def init_camera(camera_config):
    from framework.CameraDevice_v4l2 import CameraDevice_v4l2 as CameraDevice

    cam = None
    for attempt in range(2):
        cam = CameraDevice(camera_config["model_name"], **camera_config["camera_config"])
        if cam.validate():
            return cam
        print(f"Camera validation failed ({attempt + 1}/2)")
        cam.shutdown()
    raise RuntimeError("Camera could not be validated.")


def init_screen_detector(config):
    if config["name"] == "fixed":
        from framework.ScreenDetectorFixed import ScreenDetectorFixed

        return ScreenDetectorFixed(config["name"], **config["detection_config"])
    else:
        from framework.ScreenDetector import ScreenDetector

        return ScreenDetector(config["name"], config["corners"], **config["detection_config"])


def compute_render_rect(display_w, display_h):
    (
        x,
        y,
        render_w,
        render_h,
    ) = get_atari_region(display_w, display_h)
    """
    if display_w / display_h > ASPECT_RATIO:
        render_h = display_h
        render_w = int(render_h * ASPECT_RATIO)
    else:
        render_w = display_w
        render_h = int(render_w / ASPECT_RATIO)

    return pygame.Rect(
        (display_w - render_w) // 2,
        (display_h - render_h) // 2,
        render_w,
        render_h
    )
    """
    print(f"atari={x} {y} {render_w} {render_h}")
    print(f"rect={(display_w - render_w) // 2} {(display_h - render_h) // 2} {render_w} {render_h}")
    return pygame.Rect(x, y, render_w, render_h)


def setup_display(display_info):
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{display_info['x']},{display_info['y']}"
    os.environ["SDL_VIDEO_CENTERED"] = "0"
    return pygame.display.set_mode((display_info["width"], display_info["height"]), pygame.NOFRAME)


def render_ale_frame(ale, window, render_rect, hold_ms=1000):
    screen_h, screen_w = ale.getScreenDims()
    frame = np.empty((screen_h, screen_w, 3), dtype=np.uint8)
    ale.getScreenRGB(frame)

    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    scaled = pygame.transform.smoothscale(surf, (render_rect.width, render_rect.height))
    window.fill((0, 0, 0))
    window.blit(scaled, render_rect.topleft)
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in [pygame.K_ESCAPE, pygame.K_q]):
            pygame.quit()
            exit()

    pygame.time.wait(hold_ms)


def detect_screen(camera, detector, debug=False):
    for _ in range(1000):
        data = camera.get_frame()
        frame = data["frame"]
        gray = camera.convert_to_grayscale(frame)

        rect, tags = detector.get_screen_rect_info(gray)
        if rect is not None:
            return rect

        if debug:
            rgb = camera.convert_to_rgb(frame)
            for tag_id, corners in tags.items():
                idx = detector.tag_id_corner_idx[tag_id]
                cv2.circle(rgb, tuple(map(int, corners[idx])), 5, (0, 255, 0), -1)
            cv2.imshow("camera", rgb)
            cv2.waitKey(1)
    raise ValueError("Could not obtain valid screen rect.")


def generate_data_physical(args, game, data_dir, debug=False):
    framework_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    with open(os.path.join(framework_path, args.camera_config)) as f:
        camera_config = json.load(f)

    with open(os.path.join(framework_path, args.detection_config)) as f:
        detection_config = json.load(f)

    camera = init_camera(camera_config)
    screen_detector = init_screen_detector(detection_config)
    screen_rect = detect_screen(camera, screen_detector, debug=True)

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    ale = ALEInterface()
    ale.setLoggerMode(LoggerMode.Error)
    ale.loadROM(roms.get_rom_path(game))

    config = GAME_RAM_CONFIG[game]
    score_range = range(0, config["max_score"] + 1, config["score_step"][0])
    lives_range = range(1, config["total_lives"] + 1)

    pygame.init()
    displays = get_connected_displays()
    if args.target_display >= len(displays):
        raise ValueError(f"Invalid display index {args.target_display}")
    display = displays[args.target_display]

    window = setup_display(display)
    pygame.display.set_caption("ALE")
    render_rect = compute_render_rect(display["width"], display["height"])

    print(f"Rendering to display {display['name']} at {display['x']},{display['y']}")
    print(f"Render rect: {render_rect.size}")

    # Generate score images
    dummy_lives = 1 if game == 'defender' or game == 'battle_zone' else 0
    for i, score in enumerate(score_range):
        if debug:
            print(f"--- {game} | Score: {score} | Lives: dummy (0) ---")

        write_score(config, score, ale)
        write_lives(config, dummy_lives, ale)

        if game == 'qbert':
            ale.reset_game()
            for _ in range(5):
                ale.act(Action.FIRE)
            for _ in range(30):
                ale.act(Action.NOOP)
            for j in range(1000):
                write_score(config, score, ale)
                ale.act(ale.getLegalActionSet()[random.randint(0, 3)])
                # TODO: get this from the game config instead of hardcoding
                score_region = ale.getScreenRGB()[5 : 10 + 5, 26 : 56 + 26]
                score_intensity = score_region.mean()
                if score_intensity > 10:
                    if debug:
                        print(f"[INFO] score frame found at step {i} | score_intensity={score_intensity:.1f}")
                    break
        else:
            for _ in range(2):
                ale.act(Action.NOOP)

        if debug:
            ram = ale.getRAM()
            decoded_score = decode_score_bcd(ram, config["score_addr"], config)
            print(f"score={score} decoded={decoded_score}")

        render_ale_frame(ale, window, render_rect, hold_ms=1000)

        # flush stale frames
        for _ in range(5):
            camera.get_frame()
            time.sleep(0.001)

        frame = camera.get_frame()["frame"]
        frame_rgb = camera.convert_to_rgb(frame)

        rectified = rectify_frame(frame_rgb, screen_rect, ATARI_WIDTH, ATARI_HEIGHT)

        score_str = str(score).zfill(config["score_digits"])
        Image.fromarray(rectified).save(os.path.join(data_dir, f"img_score_{score_str}.png"))

        if i % 100 == 0:
            print(f"[INFO] {i + 1} / {len(score_range)} score images generated")

    ale.reset_game()
    for _ in range(5):
        ale.act(Action.NOOP)

    dummy_score = 0

    # Generate lives images
    for i, lives in enumerate(lives_range):
        if debug:
            print(f"--- {game} | Score: dummy (0) | Lives: {lives} ---")

        write_score(config, dummy_score, ale)
        write_lives(config, lives, ale)

        if game == 'qbert':
            ale.reset_game()
            for _ in range(5):
                ale.act(Action.FIRE)
            for _ in range(30):
                ale.act(Action.NOOP)
            for j in range(1000):
                write_lives(config, lives, ale)
                ale.act(ale.getLegalActionSet()[random.randint(0, 3)])
                # TODO: get this from the game config instead of hardcoding
                lives_region = ale.getScreenRGB()[14 : 16 + 14, 33 : 40 + 33]
                lives_intensity = lives_region.mean()
                if lives_intensity > 10:
                    break
        else:
            for _ in range(2):
                ale.act(Action.NOOP)

        if debug:
            ram = ale.getRAM()
            decoded_lives = decode_lives(ram, config['lives_addr'], config)
            print(f"lives={lives} decoded={decoded_lives}")

        render_ale_frame(ale, window, render_rect, hold_ms=1000)

        # flush stale frames
        for _ in range(5):
            camera.get_frame()
            time.sleep(0.001)

        frame = camera.get_frame()["frame"]
        frame_rgb = camera.convert_to_rgb(frame)

        rectified = rectify_frame(frame_rgb, screen_rect, ATARI_WIDTH, ATARI_HEIGHT)

        img = Image.fromarray(rectified)
        img.save(os.path.join(data_dir, f"img_lives_{lives}.png"))

        if i % 2 == 0 or i == len(lives_range) - 1:
            print(f"[INFO] Generated {i + 1} / {len(lives_range)} lives images")

    screen_detector.shutdown()
    camera.shutdown()


# python3 generate_dataset.py ms_pacman --output_dir 'frames/ms_pacman'
# primary display should be set to built-in, multiple display mode: join
def get_argument_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="generate_data.py arguments")
    parser.add_argument('game', type=str, default=None)
    parser.add_argument('--detection_config', type=str, default="configs/screen_detection/april_tags.json")
    parser.add_argument('--camera_config', type=str, default="configs/cameras/camera_kiyo_pro.json")
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.getcwd(), 'results'))
    parser.add_argument('--target_display', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    return parser


if __name__ == '__main__':
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    try:
        generate_data_physical(args, args.game, args.output_dir, debug=args.debug)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")

    exit(0)
