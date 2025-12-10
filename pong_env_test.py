#!/usr/bin/env python3
"""
Quick sanity check for ALE Pong controls.

Usage:
    python pong_env_test.py --video pong_env_test.mp4

What it does:
    - Resets ALE/Pong-v5
    - Fires once to start play
    - Alternates UP/DOWN in blocks for a configurable number of steps
    - Prints RAM bytes for both paddles before/after moves
      (right/controlled paddle = RAM[49], left/CPU paddle = RAM[50])
    - Optionally records the frames to an MP4 (via imageio-ffmpeg) with a red
      highlight stripe on the controlled (right) paddle side
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import gymnasium as gym
except ImportError as e:
    print("gymnasium is required. Activate the physical_atari env.", file=sys.stderr)
    raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", type=str, default=None,
                        help="If set, save an MP4 of the short move test.")
    parser.add_argument("--total_steps", type=int, default=600,
                        help="Number of post-serve steps to alternate UP/DOWN.")
    parser.add_argument("--block_len", type=int, default=30,
                        help="Steps per UP/DOWN block before switching direction.")
    args = parser.parse_args()

    writer = None
    if args.video:
        try:
            import imageio  # type: ignore

            writer = imageio.get_writer(
                args.video,
                fps=60,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
        except Exception as e:  # pragma: no cover - optional dependency
            print(f"Could not init video writer ({e}); continuing without video.")
            writer = None

    def append_frame(frame, ram_snapshot):
        if writer is None:
            return
        # Mark controlled side with a red stripe
        frame = frame.copy()
        frame[:, -4:, :] = [255, 0, 0]
        # Draw a small blue bar at the controlled paddle's RAM y position
        if ram_snapshot is not None and len(ram_snapshot) > 49:
            h = frame.shape[0]
            y_ram = int(ram_snapshot[49])
            y_px = int(y_ram / 255.0 * (h - 1))
            y0 = max(0, y_px - 3)
            y1 = min(h, y_px + 4)
            frame[y0:y1, -10:-4, :] = [0, 0, 255]
        writer.append_data(frame)

    env = gym.make(
        "ALE/Pong-v5",
        obs_type="rgb",
        full_action_space=True,
        render_mode="rgb_array" if writer else None,
    )
    frame, _ = env.reset(seed=args.seed)
    ram = env.unwrapped.ale.getRAM()
    append_frame(frame, ram)

    print(f"start p_right={ram[49]} p_left={ram[50]}")

    # Fire once to serve the ball
    frame, _, _, _, _ = env.step(1)  # FIRE
    ram = env.unwrapped.ale.getRAM()
    append_frame(frame, ram)

    # Alternate UP/DOWN blocks to clearly show motion
    for t in range(args.total_steps):
        block = (t // args.block_len) % 2  # 0 = UP block, 1 = DOWN block
        action = 2 if block == 0 else 5
        frame, _, _, _, _ = env.step(action)
        ram = env.unwrapped.ale.getRAM()
        append_frame(frame, ram)

    ram = env.unwrapped.ale.getRAM()
    print(f"after {args.total_steps} steps p_right={ram[49]} p_left={ram[50]}")

    env.close()
    if writer:
        writer.close()
        print(f"Saved video to {os.path.abspath(args.video)}")


if __name__ == "__main__":
    main()
