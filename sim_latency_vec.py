from __future__ import annotations

import argparse
import importlib
import os
import time
from typing import Any, Dict, Iterable, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import AtariPreprocessing
from imageio import v2 as imageio
from tqdm import tqdm

from latency_wrap.wrapper_v0_2 import LatencyModel
from vector_agents import VectorAgent
from gymnasium import spaces

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None


def _noop_vector() -> list[float]:
    """Return the 36-length NOOP/NOOP one-hot vector LatencyModel expects."""
    vec = [0.0] * 36
    vec[0] = 1.0
    vec[18] = 1.0
    return vec


class LatencyWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        latency_model_dir: str,
        action_mapping: np.ndarray | None = None,
        simulate_latency: bool = True,
        frame_skip: int = 1,
    ):
        super().__init__(env)
        self.simulate_latency = simulate_latency
        self.latency_model = LatencyModel(directory_with_weights=latency_model_dir) if self.simulate_latency else None
        self.frame_skip = max(1, int(frame_skip))
        if action_mapping is not None:
            self._action_map = np.asarray(action_mapping, dtype=np.int64)
        else:
            action_set = getattr(self.env.unwrapped, "_action_set", None)
            if action_set is None:
                raise RuntimeError("Underlying ALE env does not expose _action_set")
            self._action_map = np.asarray(action_set, dtype=np.int64)
        self._reverse_map = {ale_action: idx for idx, ale_action in enumerate(self._action_map)}
        self.action_space = spaces.Discrete(len(self._action_map))
        self.reset_latency_state()

    def reset_latency_state(self) -> None:
        if self.latency_model is None:
            return
        self.latency_model.action_queue = []
        for _ in range(30):
            self.latency_model.action_queue.append(_noop_vector())
        self.latency_model.last_action = 0

    def reset(self, **kwargs):
        while True:
            self.reset_latency_state()
            obs, info = self.env.reset(**kwargs)
            # Randomize starting state so vector env episodes decorrelate during sim runs.
            noop_count = np.random.randint(0, 30)
            terminated = False
            truncated = False
            for _ in range(noop_count):
                obs, _, terminated, truncated, info = self.env.step(0)
                if terminated or truncated:
                    break
            if terminated or truncated:
                continue
            return obs, info

    def step(self, action):
        env_action_idx = int(action)
        ale_action = int(self._action_map[env_action_idx])
        if self.simulate_latency and self.latency_model is not None:
            delayed_ale = int(self.latency_model.act(ale_action))
            env_index = self._reverse_map.get(delayed_ale, env_action_idx)
        else:
            env_index = env_action_idx
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(env_index)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class VectorLatencyWrapper:
    """
    Vectorized latency + frame-skip wrapper applied outside the worker processes.
    Latency is batched across envs to avoid per-env Python overhead.
    """

    def __init__(
        self,
        venv: AsyncVectorEnv,
        latency_weights: str,
        action_mapping: np.ndarray | None,
        frame_skip: int,
        disable_latency: bool,
    ):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.disable_latency = disable_latency
        self.frame_skip = max(1, int(frame_skip))

        # Action mappings
        base_action_set = np.asarray(getattr(venv.env_fns[0](), "unwrapped")._action_set, dtype=np.int64)  # type: ignore
        if action_mapping is not None:
            self._action_map = np.asarray(action_mapping, dtype=np.int64)
        else:
            self._action_map = base_action_set
        self._reverse_map = {ale_action: idx for idx, ale_action in enumerate(self._action_map)}
        self.action_space = spaces.Discrete(len(self._action_map))
        self.single_action_space = self.action_space

        # Latency state
        self.latency_enabled = not disable_latency
        if self.latency_enabled:
            model = LatencyModel(directory_with_weights=latency_weights)
            # copy weights to arrays for batched use
            self.fc_weight = model.fc_weight
            self.fc_bias = model.fc_bias
            self.pred_weight = model.pred_weight
            self.pred_bias = model.pred_bias
            self.action_queue = np.zeros((self.num_envs, 30, 36), dtype=np.float32)
            noop_vec = np.zeros((36,), dtype=np.float32)
            noop_vec[0] = 1.0
            noop_vec[18] = 1.0
            self.action_queue[:] = noop_vec
            self.last_action = np.zeros((self.num_envs,), dtype=np.int64)

    def reset(self):
        return self.venv.reset()

    def step(self, actions: np.ndarray):
        # Map agent actions to ALE actions
        actions = np.asarray(actions, dtype=np.int64)
        ale_actions = self._action_map[actions]

        if self.latency_enabled:
            # Shift queues and insert new one-hots
            self.action_queue[:, :-1, :] = self.action_queue[:, 1:, :]
            hot = np.zeros((self.num_envs, 36), dtype=np.float32)
            hot[np.arange(self.num_envs), ale_actions] = 1.0
            hot[np.arange(self.num_envs), 18 + self.last_action] = 1.0
            self.action_queue[:, -1, :] = hot

            rep = self.action_queue.reshape(self.num_envs, -1)
            logits = rep @ self.fc_weight.T + self.fc_bias
            logits = np.maximum(logits, 0.0)
            logits = logits @ self.pred_weight.T + self.pred_bias
            delayed_ale = np.argmax(logits, axis=1).astype(np.int64)
            self.last_action = delayed_ale
            delayed_actions = np.array([self._reverse_map.get(int(a), int(actions[i])) for i, a in enumerate(delayed_ale)], dtype=np.int64)
        else:
            delayed_actions = np.array([self._reverse_map.get(int(a), int(act)) for a, act in zip(ale_actions, actions)], dtype=np.int64)

        total_reward = np.zeros((self.num_envs,), dtype=np.float32)
        terminated = np.zeros((self.num_envs,), dtype=bool)
        truncated = np.zeros((self.num_envs,), dtype=bool)
        infos = None
        obs = None
        for _ in range(self.frame_skip):
            obs, r, term, trunc, infos = self.venv.step(delayed_actions)
            total_reward += r
            terminated |= term
            truncated |= trunc
            if terminated.any() or truncated.any():
                break
        return obs, total_reward, terminated, truncated, infos


def parse_agent_spec(agent_spec: str) -> Tuple[str, str]:
    if ":" not in agent_spec:
        raise ValueError("Agent spec must be 'module:ClassName'")
    module_name, class_name = agent_spec.split(":", 1)
    return module_name, class_name


def load_agent(agent_spec: str) -> type[VectorAgent]:
    module_name, class_name = parse_agent_spec(agent_spec)
    module = importlib.import_module(module_name)
    try:
        agent_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"{class_name} not found in {module_name}") from exc
    return agent_cls


def parse_agent_kwargs(raw_args: Iterable[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for entry in raw_args:
        if "=" not in entry:
            raise ValueError(f"Invalid agent_arg '{entry}' (expected key = value).")

        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed[key] = eval(value, {"__builtins__": {}})
        except Exception:
            parsed[key] = value
    return parsed


def build_env(args) -> AsyncVectorEnv:
    def make_env(rank: int):
        def thunk():
            env = gym.make(f"ALE/{args.rom}-v5", obs_type='rgb', full_action_space=True)
            env.reset(seed=args.seed + rank)
            # Move grayscale/resize into the env to avoid per-step cv2
            env = AtariPreprocessing(
                env,
                grayscale_obs=True,
                scale_obs=False,
                frame_skip=1,  # we handle frame_skip outside
                screen_size=args.env_screen_size,
            )
            action_map = np.asarray([2, 5, 4, 3], dtype=np.int64) if args.reduce_action_set else None
            env = LatencyWrapper(
                env,
                latency_model_dir=args.latency_weights,
                action_mapping=action_map,
                simulate_latency=not args.disable_latency,
                frame_skip=args.frame_skip,
            )
            return env

        return thunk

    envs = [make_env(i) for i in range(args.num_envs)]
    # Async vectorization to parallelize env stepping; latency handled inside workers
    return AsyncVectorEnv(envs, shared_memory=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Latency-aware vector Atari trainer")
    parser.add_argument("--rom", type=str, default="MsPacman")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_frames", type=int, default=1_000_000)
    parser.add_argument("--agent", type=str, required=True, help="module:Class implementing VectorAgent")
    parser.add_argument("--agent_arg", action="append", default=[], help="Extra agent kwards key=value")
    parser.add_argument("--latency_weights", type=str, default="latency_wrap")
    parser.add_argument("--results_dir", type=str, default=os.path.join(os.getcwd(), "results", "sim_latency_vec"))
    parser.add_argument("--checkpoint_interval", type=int, default=250_000)
    parser.add_argument("--checkpoint_name", type=str, default="vector_agent.pt")
    parser.add_argument("--log_interval", type=int, default=10_000)
    parser.add_argument("--reduce_action_set", action="store_true", help="Restrict to 4 directional actions (UDLR).")
    parser.add_argument("--disable_latency", action="store_true", help="Bypass latency model to benchmark raw env/GPU FPS.")
    parser.add_argument("--frame_skip", type=int, default=1, help="Repeat each selected action this many frames.")
    parser.add_argument("--env_screen_size", type=int, default=128, help="Grayscale resize target for AtariPreprocessing.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="sim-latency-vec")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--record_video", action="store_true", help="Capture env-0 gameplay videos")
    parser.add_argument("--video_dir", type=str, default=os.path.join(os.getcwd(), "videos", "sim_latency_vec"))
    parser.add_argument("--video_every", type=int, default=10, help="Record every Nth env-0 episode")
    parser.add_argument("--video_fps", type=int, default=60)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    venv = build_env(args)
    agent_cls = load_agent(args.agent)
    agent_kwargs = parse_agent_kwargs(args.agent_arg)
    action_space = getattr(venv, "single_action_space", getattr(venv, "action_space", None))
    agent: VectorAgent = agent_cls(
        num_envs=args.num_envs,
        seed=args.seed,
        num_actions=action_space.n,
        results_dir=args.results_dir,
        total_frames=args.total_frames,
        **agent_kwargs,
    )

    agent.reset(args.num_envs)

    run_cfg = vars(args).copy()
    run_cfg["agent_kwargs"] = agent_kwargs
    run_cfg.pop("agent_arg", None)

    run = None
    if args.wandb and wandb is not None:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=run_cfg,
            reinit=True,
        )

    observations, _ = venv.reset()
    episode_rewards = np.zeros(args.num_envs, dtype=np.float32)
    episode_counts = np.zeros(args.num_envs, dtype=np.int64)
    reward_ema = None  # track smoothed episodic reward for logging
    global_step = 0
    start_time = time.time()

    if args.record_video:
        os.makedirs(args.video_dir, exist_ok=True)
        video_frames: list[np.ndarray] = []
        next_record_episode = 1
    else:
        video_frames = []
        next_record_episode = None

    with tqdm(total=args.total_frames, desc="Training", unit="frame", dynamic_ncols=True) as progress:
        while global_step < args.total_frames:
            if args.record_video:
                video_frames.append(observations[0].copy())
            actions = agent.act(observations)
            next_obs, rewards, terminations, truncations, infos = venv.step(actions)

            agent.observe(next_obs, rewards, terminations, truncations, infos.get("final_info", []))
            agent.train_step()

            episode_rewards += rewards
            observations = next_obs
            global_step += args.num_envs
            progress.update(args.num_envs)

            if args.record_video:
                video_frames.append(next_obs[0].copy())

            for idx, final_info in enumerate(infos.get("final_info", [])):
                if final_info is None:
                    continue
                episode_counts[idx] += 1
                score = episode_rewards[idx]
                if reward_ema is None:
                    reward_ema = float(score)
                else:
                    reward_ema = 0.95 * reward_ema + 0.05 * float(score)
                elapsed = time.time() - start_time
                sps = int(global_step / elapsed) if elapsed > 0 else 0
                progress.set_postfix(sps=sps)
                print(f"[env {idx}] episode {episode_counts[idx]} score {score:.1f} step {global_step} SPS {sps}")
                if run is not None:
                    metrics = {"episode_reward": score, "global_step": global_step, "sps": sps}
                    if reward_ema is not None:
                        metrics["episode_reward_ema"] = reward_ema
                    run.log(metrics, step=global_step)
                if args.record_video and idx == 0:
                    if episode_counts[idx] == next_record_episode:
                        video_path = os.path.join(
                            args.video_dir,
                            f"{args.rom}_episode_{episode_counts[idx]:04d}.mp4",
                        )
                        imageio.mimsave(video_path, video_frames, fps=args.video_fps)
                        if run is not None and wandb is not None:
                            run.log({"episode_video": wandb.Video(video_path, fps=args.video_fps)}, step=global_step)
                        next_record_episode += args.video_every
                    video_frames = []
                episode_rewards[idx] = 0.0

            if args.log_interval and global_step % args.log_interval < args.num_envs:
                elapsed = time.time() - start_time
                sps = int(global_step / elapsed) if elapsed > 0 else 0
                progress.set_postfix(sps=sps)
                if run is not None:
                    metrics = {"global_step": global_step, "sps": sps}
                    core = getattr(agent, "core", None)
                    if core is not None:
                        metrics["train/loss"] = getattr(core, "last_loss", 0.0)
                        if getattr(core, "loss_ema", None) is not None:
                            metrics["train/loss_ema"] = core.loss_ema
                        metrics["train/avg_q"] = getattr(core, "last_avg_q", 0.0)
                        metrics["train/max_q"] = getattr(core, "last_max_q", 0.0)
                        metrics["train/epsilon"] = getattr(core, "epsilon", 0.0)
                        metrics["train/td_error"] = getattr(core, "last_td_error", 0.0)
                        metrics["train/grad_norm"] = getattr(core, "last_grad_norm", 0.0)
                        metrics["train/beta"] = getattr(core.replay, "beta", 0.0)
                        if getattr(core.replay, "max_priority", None) is not None:
                            metrics["replay/max_priority"] = core.replay.max_priority
                        metrics["replay/fraction_filled"] = core.replay.size / float(core.replay.capacity)
                        if hasattr(core, "optimizer"):
                            metrics["train/lr"] = core.optimizer.param_groups[0]["lr"]
                        # Average sigma for noisy layers helps track exploration decay.
                        sigmas = []
                        for module in core.network.modules():
                            if hasattr(module, "weight_sigma"):
                                sigmas.append(module.weight_sigma.detach().mean().item())
                        if sigmas:
                            metrics["train/noisy_sigma"] = float(np.mean(sigmas))
                    run.log(metrics, step=global_step)

            if args.checkpoint_interval and global_step % args.checkpoint_interval < args.num_envs:
                ckpt_path = os.path.join(args.results_dir, f"checkpoint_{global_step}.pt")
                agent.save_model(ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

    final_path = os.path.join(args.results_dir, args.checkpoint_name)
    agent.save_model(final_path)
    if run is not None:
        run.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
