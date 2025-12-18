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

import os
import random
import shutil

from ale_py import Action, ALEInterface, LoggerMode, roms
from .ale_ram_injection import GAME_RAM_CONFIG, decode_lives, decode_score_bcd, write_lives, write_score
from PIL import Image

"""
Generates all valid combinations of score for the game and outputs as {output_dir}/{game}/'img_score_{score:6d}.png'
Generates all valid combinations of lives for the game and outputs as {output_dir}/{game}/'img_lives_{lives}.png'
"""


def generate_data(game, data_dir, debug=False):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    print(f"Generating data for {game} at {data_dir}")

    config = GAME_RAM_CONFIG[game]

    rom_path = roms.get_rom_path(game)
    ale = ALEInterface()
    ale.setLoggerMode(LoggerMode.Error)
    ale.loadROM(rom_path)

    score_range = range(0, config["max_score"] + 1, config["score_step"][0])
    lives_range = range(1, config["total_lives"] + 1)

    # Generate all valid score combos
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

            for i in range(1000):
                write_score(config, score, ale)
                ale.act(ale.getLegalActionSet()[random.randint(0, 3)])
                obs = ale.getScreenRGB()

                # TODO: get this from the game config instead of hardcoding
                score_region = obs[5 : 10 + 5, 26 : 56 + 26]
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

        score_str = str(score).zfill(config["score_digits"])
        img = Image.fromarray(ale.getScreenRGB())
        img.save(os.path.join(data_dir, f"img_score_{score_str}.png"))

        if i % 100 == 0 or i == len(score_range) - 1:
            print(f"[INFO] Generated {i + 1} / {len(score_range)} score images")

    ale.reset_game()
    for _ in range(5):
        ale.act(Action.NOOP)

    dummy_score = 0

    # generate all valid lives combos
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

            for i in range(1000):
                write_lives(config, lives, ale)
                ale.act(ale.getLegalActionSet()[random.randint(0, 3)])
                obs = ale.getScreenRGB()

                # TODO: get this from the game config instead of hardcoding
                lives_region = obs[14 : 16 + 14, 33 : 40 + 33]
                lives_intensity = lives_region.mean()
                if lives_intensity > 10:
                    if debug:
                        print(f"[INFO] lives frame found at step {i} | lives_intensity={lives_intensity:.1f}")
                    break
        else:
            for _ in range(2):
                ale.act(Action.NOOP)

        if debug:
            ram = ale.getRAM()
            decoded_lives = decode_lives(ram, config['lives_addr'], config)
            print(f"lives={lives} decoded={decoded_lives}")

        img = Image.fromarray(ale.getScreenRGB())
        img.save(os.path.join(data_dir, f"img_lives_{lives}.png"))

        if i % 2 == 0 or i == len(lives_range) - 1:
            print(f"[INFO] Generated {i + 1} / {len(lives_range)} lives images")

    print(f"Finished generating data: num_scores={len(score_range)} num_lives={len(lives_range)}.")


# python3 generate_dataset.py ms_pacman --output_dir 'frames/ms_pacman'
def get_argument_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="generate_data.py arguments")
    parser.add_argument('game', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.getcwd(), 'results'))
    parser.add_argument('--debug', action='store_true')
    return parser


if __name__ == '__main__':
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    try:
        generate_data(args.game, args.output_dir, debug=args.debug)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")

    exit(0)
