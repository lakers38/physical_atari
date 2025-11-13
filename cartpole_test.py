#!/usr/bin/env python3
"""
Lightweight CartPole sanity test for the MEME agent.

This spins up a vectorized CartPole-v1 environment, wraps the 4-D observations
into 84×84 RGB frames so MEME's vision stack can consume them, and runs the
VectorMEMEAgent for a configurable number of frames. Use this to verify that
the full agent (policy/value heads, replay, training loop) learns a simple
classic-control task outside the bespoke probe environments.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple, Optional

import gymnasium as gym
import numpy as np
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
from vector_agents import VectorAgent
from agent_meme import VectorMEMEAgent


FRAME_SHAPE = (84, 84, 3)
VALUE_CLIP = np.array([2.4, 3.0, 0.3, 3.5], dtype=np.float32)  # sensible CartPole limits


def vector_to_frame(obs: np.ndarray) -> np.ndarray:
    """Encode the 4-D CartPole state as four vertical bars in an RGB image."""
    obs = np.asarray(obs, dtype=np.float32)
    scaled = np.clip(obs / VALUE_CLIP, -1.0, 1.0)
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    bar_width = FRAME_SHAPE[1] // 4
    for i, value in enumerate(scaled):
        start = i * bar_width
        end = FRAME_SHAPE[1] if i == 3 else (i + 1) * bar_width
        intensity = int(np.interp(value, [-1.0, 1.0], [0, 255]))
        frame[:, start:end] = intensity
    return frame


class CartPoleRGBWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=FRAME_SHAPE, dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return vector_to_frame(observation)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


def make_env(seed: int, rank: int, render: bool):
    def thunk():
        render_mode = "human" if render and rank == 0 else None
        env = gym.make("CartPole-v1", render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = CartPoleRGBWrapper(env)
        env.reset(seed=seed + rank)
        return env

    return thunk


def run_cartpole(
    *,
    frames: int,
    num_envs: int,
    seed: int,
    results_dir: str,
    agent_kwargs: dict,
    render: bool = False,
    use_tqdm: bool = True,
    wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None,
    video_every: int = 0,
) -> Tuple[float, float]:
    os.makedirs(results_dir, exist_ok=True)
    env_fns = [make_env(seed, i, render) for i in range(num_envs)]
    venv = gym.vector.SyncVectorEnv(env_fns)

    action_space = venv.single_action_space
    agent: VectorAgent = VectorMEMEAgent(
        num_envs=num_envs,
        seed=seed,
        num_actions=action_space.n,
        results_dir=results_dir,
        total_frames=frames,
        **agent_kwargs,
    )
    agent.reset(num_envs)

    observations, _ = venv.reset(seed=seed)
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    completed = []
    global_step = 0
    log_interval = max(frames // 10, num_envs)
    next_log = log_interval
    progress = tqdm(total=frames, desc="CartPole", unit="frame", disable=not use_tqdm)
    video_frames: list[np.ndarray] = []
    next_video_episode = 1 if video_every > 0 else None

    while global_step < frames:
        actions = agent.act(observations)
        next_obs, rewards, terminations, truncations, infos = venv.step(actions)
        final_info = infos.get("final_info", [])
        agent.observe(next_obs, rewards, terminations, truncations, final_info)
        agent.train_step()

        episode_rewards += rewards
        done = np.logical_or(terminations, truncations)
        for idx, flag in enumerate(done):
            if flag:
                completed.append(float(episode_rewards[idx]))
                episode_rewards[idx] = 0.0
                if wandb_run is not None and video_every > 0 and idx == 0 and next_video_episode is not None:
                    if len(video_frames) >= 1 and len(completed) == next_video_episode:
                        video = np.stack(video_frames, axis=0)
                        wandb_run.log(
                            {f"video/episode_{len(completed)}": wandb.Video(video, fps=30, format="mp4")},
                            step=global_step,
                        )
                        video_frames = []
                        next_video_episode += video_every
                if idx == 0:
                    video_frames = []
        if video_every > 0:
            video_frames.append(next_obs[0])
        observations = next_obs
        global_step += num_envs
        progress.update(num_envs)
        if global_step >= next_log:
            mean = float(np.mean(completed)) if completed else 0.0
            recent = float(np.mean(completed[-10:])) if len(completed) >= 10 else mean
            best = float(np.max(completed)) if completed else 0.0
            print(
                f"[cartpole_test] step={global_step}/{frames} episodes={len(completed)} "
                f"mean_return={mean:.2f} recent_return={recent:.2f} best={best:.2f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "global_step": global_step,
                        "episodes": len(completed),
                        "mean_return": mean,
                        "recent_return": recent,
                        "best_return": best,
                    },
                    step=global_step,
                )
            next_log += log_interval

    venv.close()
    progress.close()
    mean_reward = float(np.mean(completed)) if completed else 0.0
    recent_reward = float(np.mean(completed[-10:])) if completed else 0.0
    return mean_reward, recent_reward


def parse_agent_kwargs(raw_args):
    parsed = {}
    for entry in raw_args:
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed[key] = eval(value, {"__builtins__": {}})
        except Exception:
            parsed[key] = value
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MEME agent on CartPole-v1.")
    parser.add_argument("--frames", type=int, default=50_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default=os.path.join(os.getcwd(), "results", "cartpole_test"))
    parser.add_argument("--agent-arg", action="append", default=[], help="Override MEME arguments key=value")
    parser.add_argument("--render", action="store_true", help="Open a window for env 0 (num_envs should be 1).")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bar output.")
    parser.add_argument("--wandb", action="store_true", help="Log metrics/videos to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="meme-cartpole")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-video-every", type=int, default=0, help="Record env-0 video every N episodes (requires wandb).")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    agent_kwargs = parse_agent_kwargs(args.agent_arg)
    agent_kwargs.setdefault("use_intrinsic_rewards", False)
    agent_kwargs.setdefault("beta_low", 0.0)
    agent_kwargs.setdefault("beta_high", 0.0)
    agent_kwargs.setdefault("seq_len", 1)
    agent_kwargs.setdefault("burn_in", 0)
    agent_kwargs.setdefault("train_interval", 1)
    agent_kwargs.setdefault("batch_size", 1)
    agent_kwargs.setdefault("train_micro_batch", 1)

    if args.render and args.num_envs != 1:
        print("[cartpole_test] render requested but num_envs != 1; only env 0 will display.")

    wandb_run = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            raise RuntimeError("wandb requested but not installed. `pip install wandb`.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "frames": args.frames,
                "num_envs": args.num_envs,
                "seed": args.seed,
                "agent_kwargs": agent_kwargs,
            },
        )

    mean_reward, recent_reward = run_cartpole(
        frames=args.frames,
        num_envs=args.num_envs,
        seed=args.seed,
        results_dir=args.results_dir,
        agent_kwargs=agent_kwargs,
        render=args.render,
        use_tqdm=not args.no_tqdm,
        wandb_run=wandb_run,
        video_every=args.wandb_video_every if wandb_run is not None else 0,
    )
    print(f"[cartpole_test] finished: mean_return={mean_reward:.2f} recent_return={recent_reward:.2f}")
    if wandb_run is not None:
        wandb_run.log({"final/mean_return": mean_reward, "final/recent_return": recent_reward})
        wandb_run.finish()


if __name__ == "__main__":
    main()
