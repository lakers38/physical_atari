#!/usr/bin/env python3
"""
Vectorized training script for ActorCriticSwiftTD on Atari (simulation / optional latency).

Built from scratch following the structure of algorithms/ppo/train_agent_ppo_sim.py,
but using the SwiftTD actor-critic implementation with custom vector handling.

Key differences from the previous version:
- Uses Gymnasium vectorized envs (SubprocVecEnv) with the standard ALE wrappers
- Handles per-environment SwiftTD critics to keep eligibility traces independent
- Normalizes observations to [0,1] and guards against non-finite values
- Minimal logging/checkpointing for robustness; video recording disabled by default
"""

import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Tuple, Optional

import ale_py
import gymnasium as gym
import numpy as np
import torch
from coolname import generate_slug
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, NoopResetEnv
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor
from scipy.ndimage import zoom

# Ensure local imports work when running as a script
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "latency_wrap"))
try:
    from wrapper_v0_2 import LatencyModel, BatchedLatencyModel
    LATENCY_AVAILABLE = True
except ImportError:
    LatencyModel = None
    BatchedLatencyModel = None
    LATENCY_AVAILABLE = False

# Local imports
from actor_critic import PolicyHead, CNNFeatureExtractor  # type: ignore
import swifttd  # type: ignore

# Register ALE environments once
gym.register_envs(ale_py)


# ---------------------------- Environment helpers ---------------------------- #
class ActionSetWrapper(gym.Wrapper):
    """
    Restrict the action space similar to PPO script.
    Modes:
        0: full 18 actions
        1: minimal ALE action set
        2: restricted 4-dir (UP, DOWN, LEFT, RIGHT)
    """

    def __init__(self, env, reduce_action_set=1, game_name=""):
        super().__init__(env)
        self.reduce_action_set = reduce_action_set
        self.game_name = game_name.lower()

        if reduce_action_set == 0:
            self.action_mapping = None
        elif reduce_action_set == 2:
            # ALE indices: UP=2, DOWN=5, LEFT=4, RIGHT=3
            self.action_mapping = [2, 5, 4, 3]
        else:
            self.action_mapping = None

        if self.action_mapping is not None:
            self.action_space = spaces.Discrete(len(self.action_mapping))

    def step(self, action):
        if self.action_mapping is not None:
            action = self.action_mapping[int(action)]
        return self.env.step(action)


class PreprocessWrapper(gym.Wrapper):
    """Grayscale + resize to square, channel-last."""

    def __init__(self, env, frame_size: int):
        super().__init__(env)
        self.frame_size = frame_size
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(frame_size, frame_size, 1),
            dtype=np.uint8,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
        zoom_factors = (self.frame_size / gray.shape[0], self.frame_size / gray.shape[1])
        resized = zoom(gray, zoom_factors, order=1)
        return resized.astype(np.uint8)[..., None]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._preprocess(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._preprocess(obs), reward, terminated, truncated, info


class StackFrames(gym.Wrapper):
    """Simple frame stack along the channel-last dimension."""

    def __init__(self, env, num_stack: int):
        super().__init__(env)
        self.num_stack = num_stack
        h, w, c = env.observation_space.shape
        self.frames: deque = deque(maxlen=num_stack)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(h, w, c * num_stack),
            dtype=np.uint8,
        )

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info


class LatencyWrapper(gym.Wrapper):
    """Single env latency simulation (non-batched)."""

    def __init__(self, env, latency_model_dir):
        super().__init__(env)
        if not LATENCY_AVAILABLE:
            raise ImportError("LatencyModel not available")
        self.latency_model = LatencyModel(directory_with_weights=latency_model_dir)
        self._reset_latency_state()

    def _reset_latency_state(self):
        self.latency_model.action_queue = []
        for _ in range(30):
            self.latency_model.action_queue.append(
                self.latency_model._LatencyModel__one_hot_encode(0, 0, 36)
            )
        self.latency_model.last_action = 0

    def reset(self, **kwargs):
        self._reset_latency_state()
        return self.env.reset(**kwargs)

    def step(self, action):
        ale_action = ale_py.Action(int(action))
        delayed = self.latency_model.act(ale_action)
        return self.env.step(int(delayed))


class VecLatencyWrapper:
    """
    Batched latency wrapper using BatchedLatencyModel.
    Works on a VecEnv and keeps reduced action mapping compatible.
    """

    def __init__(self, venv, latency_model_dir, action_mapping=None, allowed_actions=None):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.latency_model = BatchedLatencyModel(latency_model_dir, self.num_envs) if LATENCY_AVAILABLE else None
        self.action_mapping = action_mapping
        self.allowed_actions = allowed_actions
        self._reverse_map = {full: idx for idx, full in enumerate(action_mapping)} if action_mapping is not None else None
        if action_mapping is not None:
            self.action_space = spaces.Discrete(len(action_mapping))
        else:
            self.action_space = venv.action_space

    def reset(self):
        if self.latency_model is not None:
            self.latency_model.reset_all()
        return self.venv.reset()

    def step(self, actions):
        if self.latency_model is None:
            return self.venv.step(actions)

        # Map reduced→full if needed, then apply batched latency
        mapped = np.array([self.action_mapping[a] for a in actions]) if self.action_mapping is not None else np.array(actions)
        delayed_full = self.latency_model.act_batch(mapped, allowed_actions=self.allowed_actions)

        if self._reverse_map is not None:
            reduced = np.array([self._reverse_map[int(a)] for a in delayed_full], dtype=np.int64)
            return self.venv.step(reduced)

        return self.venv.step(delayed_full)

    def __getattr__(self, name):
        return getattr(self.venv, name)

    # Compatibility with VecEnvWrapper expectations (used by VecVideoRecorder)
    def get_attr(self, attr_name, indices=None):
        if hasattr(self.venv, "get_attr"):
            return self.venv.get_attr(attr_name, indices)
        return getattr(self.venv, attr_name)

    def set_attr(self, attr_name, value, indices=None):
        if hasattr(self.venv, "set_attr"):
            return self.venv.set_attr(attr_name, value, indices)
        setattr(self.venv, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if hasattr(self.venv, "env_method"):
            return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)
        return getattr(self.venv, method_name)(*method_args, **method_kwargs)


def make_single_env(env_id, seed, full_action_space, reduce_action_set, simulate_latency, latency_model_dir,
                   frame_size, frame_stack):
    def thunk():
        env = gym.make(
            env_id,
            obs_type="rgb",
            render_mode="rgb_array",
            full_action_space=full_action_space,
        )
        env.reset(seed=seed)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        # Apply action reduction only when not simulating latency (latency wrapper expects full actions)
        if not simulate_latency:
            env = ActionSetWrapper(env, reduce_action_set=reduce_action_set, game_name=env_id)
        # Grayscale + resize to desired input
        env = PreprocessWrapper(env, frame_size=frame_size)
        env = StackFrames(env, num_stack=frame_stack)
        return env

    return thunk


def create_vector_env(
    env_id: str,
    n_envs: int,
    seed: int,
    reduce_action_set: int,
    simulate_latency: bool,
    latency_model_dir: str,
    frame_stack: int,
    screen_size: int,
):
    # Latency model expects full 18 actions; use full when latency is on
    use_full_action_space = simulate_latency or reduce_action_set == 0
    thunks = [
        make_single_env(
            env_id,
            seed + i,
            full_action_space=use_full_action_space,
            reduce_action_set=reduce_action_set,
            simulate_latency=simulate_latency,
            latency_model_dir=latency_model_dir,
            frame_size=screen_size,
            frame_stack=frame_stack,
        )
        for i in range(n_envs)
    ]

    venv = SyncVectorEnv(thunks)

    # Determine allowed actions for latency masking
    base_env = venv.envs[0].unwrapped
    allowed_actions = None
    action_mapping = None
    if reduce_action_set == 2:
        action_mapping = [2, 5, 4, 3]  # UP, DOWN, LEFT, RIGHT in full space
        allowed_actions = action_mapping
    elif reduce_action_set == 1:
        minimal = getattr(base_env, "_action_set", None)
        if minimal is not None:
            allowed_actions = list(minimal)
            action_mapping = allowed_actions

    if simulate_latency:
        venv = VecLatencyWrapper(
            venv,
            latency_model_dir,
            action_mapping=action_mapping,
            allowed_actions=allowed_actions,
        )
    return venv, action_mapping




# ------------------------------ Training loop ------------------------------- #
def train(args):
    import types

    def _attach_vec_helpers(env):
        if not hasattr(env, "get_attr"):
            def get_attr(self, attr_name, indices=None):
                val = getattr(self, attr_name, None)
                if indices is None or isinstance(indices, (list, tuple, np.ndarray)):
                    return [val] * getattr(self, "num_envs", 1)
                return val
            env.get_attr = types.MethodType(get_attr, env)
        if not hasattr(env, "set_attr"):
            def set_attr(self, attr_name, value, indices=None):
                setattr(self, attr_name, value)
            env.set_attr = types.MethodType(set_attr, env)
        if not hasattr(env, "env_method"):
            def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
                return [getattr(self, method_name)(*method_args, **method_kwargs)]
            env.env_method = types.MethodType(env_method, env)
        return env
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}-swifttd-{args.mode}-{generate_slug(2)}"
    env_dir_name = args.game.replace("/", "_")
    exp_dir = os.path.join(args.output_dir, args.mode, env_dir_name, run_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs", "tensorboard"), exist_ok=True)

    tb_writer = SummaryWriter(log_dir=os.path.join(exp_dir, "logs", "tensorboard"))

    env, action_mapping = create_vector_env(
        env_id=args.game,
        n_envs=args.num_envs,
        seed=args.seed,
        reduce_action_set=args.reduce_action_set,
        simulate_latency=args.mode == "sim_lat",
        latency_model_dir=args.latency_model_dir,
        frame_stack=args.n_stack,
        screen_size=args.frame_size,
    )  # CRITICAL_LINE builds vectorized ALE env + preprocessing
    env = _attach_vec_helpers(env)

    if args.record_videos:
        video_folder = os.path.join(exp_dir, "videos")
        os.makedirs(video_folder, exist_ok=True)
        env = VecVideoRecorder(
            env,
            video_folder=video_folder,
            record_video_trigger=lambda step: (step // args.num_envs) % args.video_freq == 0,
            video_length=args.video_length,
        )

    base_action_space = getattr(env, "single_action_space", env.action_space)
    num_actions = base_action_space.n
    print(f"Action space: {num_actions} | Num envs: {args.num_envs}")

    agent = VectorSwiftTDAgent(
        num_envs=args.num_envs,
        num_actions=num_actions,
        feature_dim=args.num_features,
        actor_hidden_dim=args.actor_hidden_dim,
        n_stack=args.n_stack,
        input_size=args.frame_size,
        device=args.device,
        lambda_=args.lambda_,
        initial_alpha=args.swifttd_alpha,
        gamma=args.gamma,
        eps=args.swifttd_eps,
        max_step_size=args.swifttd_max_step_size,
        step_size_decay=args.swifttd_decay,
        meta_step_size=args.swifttd_meta_step_size,
        eta_min=args.swifttd_eta_min,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        fail_on_nonfinite=args.fail_on_nonfinite,
    )  # CRITICAL_LINE instantiate agent (shared CNN + per-env SwiftTD)

    obs = env.reset()[0]
    _, feats_np = agent.start_episodes(obs)  # CRITICAL_LINE bootstrap V(s) for all envs

    ep_rewards = np.zeros(args.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int32)
    ep_returns = deque(maxlen=100)

    global_step = 0
    start_time = time.time()

    while global_step < args.total_timesteps:
        actions, log_probs, entropy, feats_np = agent.select_actions(obs)  # CRITICAL_LINE π(a|s) across envs
        next_obs, rewards, terminated, truncated, infos = env.step(actions)  # CRITICAL_LINE env transition s→s'
        rewards = np.clip(rewards, -1.0, 1.0)  # CRITICAL_LINE reward shaping for stable TD
        dones = np.logical_or(terminated, truncated)

        metrics = agent.update(obs, actions, rewards, next_obs, dones, log_probs, entropy, feats_np, step_idx=global_step)  # CRITICAL_LINE policy + SwiftTD update

        ep_rewards += rewards
        ep_lengths += 1

        # Handle env resets
        if dones.any():
            for i, done in enumerate(dones):
                if done:
                    ep_returns.append(ep_rewards[i])
                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0
            agent.reset_done(dones, next_obs)

        obs = next_obs
        global_step += args.num_envs

        # Logging
        if global_step % args.log_interval == 0:
            fps = global_step / (time.time() - start_time)
            mean_return = np.mean(ep_returns) if ep_returns else 0.0
            print(
                f"Step {global_step:,} | fps {fps:.1f} | return (100ep mean) {mean_return:.2f} | "
                f"adv {metrics['advantage_mean']:.9f}"
            )
            tb_writer.add_scalar("train/fps", fps, global_step)
            tb_writer.add_scalar("train/actor_loss", metrics["actor_loss"], global_step)
            tb_writer.add_scalar("train/advantage_mean", metrics["advantage_mean"], global_step)
            tb_writer.add_scalar("train/advantage_std", metrics["advantage_std"], global_step)
            if ep_returns:
                tb_writer.add_scalar("eval/mean_return", mean_return, global_step)

        if global_step % args.checkpoint_freq == 0:
            ckpt_path = os.path.join(exp_dir, "models", f"checkpoint_{global_step}.pth")
            agent.save(ckpt_path)
            print(f"✓ Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(exp_dir, "models", "final_model.pth")
    agent.save(final_path)
    print(f"✓ Training complete. Final model saved to {final_path}")
    env.close()
    tb_writer.close()


# ----------------------------------- CLI ------------------------------------ #
def parse_args():
    parser = argparse.ArgumentParser(description="Vectorized ActorCriticSwiftTD training (sim/latency)")
    parser.add_argument("--game", type=str, default="ALE/MsPacman-v5")
    parser.add_argument("--mode", type=str, choices=["sim", "sim_lat"], default="sim")
    parser.add_argument("--latency-model-dir", type=str, default="./latency_wrap")
    parser.add_argument("--reduce-action-set", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-features", type=int, default=512)
    parser.add_argument("--actor-hidden-dim", type=int, default=256)
    parser.add_argument("--n-stack", type=int, default=4)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    parser.add_argument("--swifttd-alpha", type=float, default=1e-4)
    parser.add_argument("--swifttd-eps", type=float, default=1e-5)
    parser.add_argument("--swifttd-max-step-size", type=float, default=0.01)
    parser.add_argument("--swifttd-decay", type=float, default=0.9)
    parser.add_argument("--swifttd-meta-step-size", type=float, default=1e-4)
    parser.add_argument("--swifttd-eta-min", type=float, default=1e-6)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument("--output-dir", type=str, default="./outputs/swifttd_vec")
    parser.add_argument("--allow-nonfinite", action="store_true",
                        help="Do not crash on non-finite values; reset critic instead (default: crash)")
    parser.add_argument("--record-videos", action="store_true", help="Enable periodic vector video recording")
    parser.add_argument("--video-freq", type=int, default=50_000, help="Record a video every N steps")
    parser.add_argument("--video-length", type=int, default=500, help="Length of recorded videos (frames)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "sim_lat" and not LATENCY_AVAILABLE:
        raise SystemExit("LatencyModel not available. Install latency_wrap or use --mode sim.")
    # Fail fast on NaNs by default; can relax with --allow-nonfinite
    args.fail_on_nonfinite = not args.allow_nonfinite
    train(args)


if __name__ == "__main__":
    main()
