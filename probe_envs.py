#!/usr/bin/env python3
"""
Diagnostic probe environments + runner for the MEME agent.

Each environment isolates a single failure mode by keeping everything else
trivial. The environments are tiny (1–2 steps), deterministic when possible,
and emit 84×84 RGB frames so they can pass through the MEME preprocessing
without further changes.

To run the whole probe suite against the MEME agent:

    python probe_envs.py --agent agent_meme:VectorMEMEAgent

Use --env to target a single probe and --frames to shorten/extend training.
"""

from __future__ import annotations

import argparse
import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Type

import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.utils import seeding
    from gymnasium.vector import SyncVectorEnv
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "gymnasium is required for probe_envs. Install via `pip install gymnasium`."
    ) from exc

from agent_utils import preprocess_batch
from vector_agents import VectorAgent


FRAME_SHAPE = (84, 84, 3)


def frame_from_level(level: float) -> np.ndarray:
    """Encode a scalar in [-1, 1] as a solid RGB frame."""
    level = float(np.clip(level, -1.0, 1.0))
    value = int(np.interp(level, [-1.0, 1.0], [0, 255]))
    frame = np.full(FRAME_SHAPE, value, dtype=np.uint8)
    return frame


class BaseProbeEnv(gym.Env):
    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(self, num_actions: int):
        super().__init__()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=0, high=255, shape=FRAME_SHAPE, dtype=np.uint8)
        self._frame_cache: Dict[float, np.ndarray] = {}
        self.np_random, _ = seeding.np_random(None)
        self._terminated = False

    # Gymnasium seeding contract.
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        self._terminated = False
        return self._obs_zero(), {}

    def _obs_zero(self) -> np.ndarray:
        return self._cached_frame(0.0)

    def _cached_frame(self, value: float) -> np.ndarray:
        value = float(np.clip(value, -1.0, 1.0))
        if value not in self._frame_cache:
            self._frame_cache[value] = frame_from_level(value)
        return self._frame_cache[value]

    def _assert_active(self):
        if self._terminated:
            raise RuntimeError("step() called after terminal transition. Call reset().")


class ConstantRewardProbeEnv(BaseProbeEnv):
    """One action, single state, +1 reward. Pure value-network sanity check."""

    name = "value_constant"

    def __init__(self):
        super().__init__(num_actions=1)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        info["state"] = "constant"
        return obs, info

    def step(self, action: int):
        self._assert_active()
        self._terminated = True
        info = {"state": "constant", "expected_value": 1.0}
        return self._obs_zero(), 1.0, True, False, info


class ObservationProbeEnv(BaseProbeEnv):
    """
    One action, random +/- observation with matching reward.
    Identifies failures in propagating observation-dependent values.
    """

    name = "value_observation"

    def __init__(self):
        super().__init__(num_actions=1)
        self._current_signal = 0.0

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._current_signal = float(self.np_random.choice([-1.0, 1.0]))
        obs = self._cached_frame(self._current_signal)
        info.update({"state_value": self._current_signal})
        return obs, info

    def step(self, action: int):
        self._assert_active()
        self._terminated = True
        reward = float(self._current_signal)
        info = {"state_value": self._current_signal, "expected_value": reward}
        return self._cached_frame(self._current_signal), reward, True, False, info


class DiscountProbeEnv(BaseProbeEnv):
    """
    Two-step episode: obs=0 -> reward 0, obs=1 -> reward +1 terminal.
    Validates reward discount/return accumulation.
    """

    name = "discount_check"

    def __init__(self):
        super().__init__(num_actions=1)
        self._phase = 0

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._phase = 0
        return obs, {"phase": 0}

    def step(self, action: int):
        self._assert_active()
        if self._phase == 0:
            self._phase = 1
            obs = self._cached_frame(1.0)
            info = {"phase": 1}
            return obs, 0.0, False, False, info
        self._terminated = True
        info = {"phase": 2, "expected_value_first_obs": 1.0}
        return self._cached_frame(1.0), 1.0, True, False, info


class ActionPreferenceProbeEnv(BaseProbeEnv):
    """
    Two actions, single observation. Action 1 yields +1, action 0 yields -1.
    Ensures policy learning / advantage estimation are wired correctly.
    """

    name = "policy_action_only"

    def __init__(self):
        super().__init__(num_actions=2)

    def step(self, action: int):
        self._assert_active()
        self._terminated = True
        reward = 1.0 if int(action) == 1 else -1.0
        info = {"preferred_action": 1, "reward_sign": reward}
        return self._obs_zero(), reward, True, False, info


class ObservationActionProbeEnv(BaseProbeEnv):
    """
    Two actions, observation indicates which action is correct.
    Tests policy/value interaction + batching (stale obs should fail here).
    """

    name = "policy_observation_action"

    def __init__(self):
        super().__init__(num_actions=2)
        self._current_signal = 1.0

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._current_signal = float(self.np_random.choice([-1.0, 1.0]))
        obs = self._cached_frame(self._current_signal)
        info.update({"state_value": self._current_signal})
        return obs, info

    def step(self, action: int):
        self._assert_active()
        self._terminated = True
        preferred = 1 if self._current_signal > 0 else 0
        reward = 1.0 if int(action) == preferred else -1.0
        info = {"preferred_action": preferred, "state_value": self._current_signal}
        return self._cached_frame(self._current_signal), reward, True, False, info


PROBE_ENVS: Dict[str, Type[BaseProbeEnv]] = {
    cls.name: cls
    for cls in (
        ConstantRewardProbeEnv,
        ObservationProbeEnv,
        DiscountProbeEnv,
        ActionPreferenceProbeEnv,
        ObservationActionProbeEnv,
    )
}


@dataclass
class ProbeSpec:
    name: str
    description: str
    target_recent: float | None
    tolerance: float
    eval_states: Dict[str, float] | None


PROBE_SPECS: Dict[str, ProbeSpec] = {
    "value_constant": ProbeSpec(
        name="value_constant",
        description="Value head should predict +1 for the only state.",
        target_recent=1.0,
        tolerance=0.02,
        eval_states={"constant": 0.0},
    ),
    "value_observation": ProbeSpec(
        name="value_observation",
        description="Value should track observation sign despite 1 action.",
        target_recent=None,
        tolerance=0.0,
        eval_states={"+1": 1.0, "-1": -1.0},
    ),
    "discount_check": ProbeSpec(
        name="discount_check",
        description="Return for the first frame should equal the delayed +1.",
        target_recent=1.0,
        tolerance=0.02,
        eval_states={"phase0": 0.0, "phase1": 1.0},
    ),
    "policy_action_only": ProbeSpec(
        name="policy_action_only",
        description="Policy must learn to pick the +1 action.",
        target_recent=1.0,
        tolerance=0.05,
        eval_states={"base": 0.0},
    ),
    "policy_observation_action": ProbeSpec(
        name="policy_observation_action",
        description="Correct action depends on observation. Value should be +1.",
        target_recent=1.0,
        tolerance=0.05,
        eval_states={"+1": 1.0, "-1": -1.0},
    ),
}


def parse_agent_spec(agent_spec: str) -> Tuple[str, str]:
    module_name, class_name = agent_spec.split(":", 1)
    return module_name, class_name


def load_agent_class(agent_spec: str) -> Type[VectorAgent]:
    module_name, class_name = parse_agent_spec(agent_spec)
    module = importlib.import_module(module_name)
    agent_cls = getattr(module, class_name)
    return agent_cls


def parse_agent_kwargs(args: Sequence[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for raw in args:
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed[key] = eval(value, {"__builtins__": {}})
        except Exception:
            parsed[key] = value
    return parsed


def make_vector_env(env_cls: Type[BaseProbeEnv], num_envs: int, seed: int) -> SyncVectorEnv:
    def make_thunk(rank: int):
        def thunk():
            env = env_cls()
            env.reset(seed=seed + rank)
            return env

        return thunk

    envs = [make_thunk(i) for i in range(num_envs)]
    return SyncVectorEnv(envs)


@dataclass
class ProbeResult:
    name: str
    episodes: int
    mean_reward: float
    recent_reward: float | None
    passed: bool | None
    greedy_actions: Dict[str, int]
    value_estimates: Dict[str, float]


def _stack_from_rgb(core, rgb: np.ndarray) -> torch.Tensor:
    obs = preprocess_batch(rgb[None, ...], core.obs_height, core.obs_width)
    stacked = np.repeat(obs[:, None, :, :], core.stack_size, axis=1)
    tensor = torch.from_numpy(stacked).to(core.device, dtype=torch.float32) / 255.0
    tensor = tensor.unsqueeze(1)  # time dimension
    return tensor


def inspect_values(agent: VectorAgent, states: Dict[str, float]) -> Tuple[Dict[str, int], Dict[str, float]]:
    greedy: Dict[str, int] = {}
    values: Dict[str, float] = {}
    core = getattr(agent, "core", None)
    if core is None or not states:
        return greedy, values

    lstm = core.network.lstm
    hidden_size = lstm.hidden_size
    num_layers = lstm.num_layers

    for tag, level in states.items():
        rgb = frame_from_level(level)
        obs_tensor = _stack_from_rgb(core, rgb)
        hidden = (
            torch.zeros(num_layers, obs_tensor.size(0), hidden_size, device=core.device),
            torch.zeros(num_layers, obs_tensor.size(0), hidden_size, device=core.device),
        )
        with torch.no_grad():
            core.ema_network.eval()
            ext, intr, logits, _ = core.ema_network(obs_tensor, hidden)
        ext = ext.squeeze(1)[0]  # [num_policies, num_actions]
        q_mean = ext.mean(dim=0)
        values[tag] = float(q_mean.max().item())
        greedy[tag] = int(torch.argmax(q_mean).item())
    return greedy, values


def run_probe(
    spec: ProbeSpec,
    env_cls: Type[BaseProbeEnv],
    agent_cls: Type[VectorAgent],
    *,
    frames: int,
    num_envs: int,
    seed: int,
    results_dir: str,
    agent_kwargs: Dict[str, Any],
) -> ProbeResult:
    os.makedirs(results_dir, exist_ok=True)
    venv = make_vector_env(env_cls, num_envs, seed)
    action_space = getattr(venv.unwrapped, "single_action_space", venv.single_action_space)

    base_kwargs = dict(
        num_envs=num_envs,
        seed=seed,
        num_actions=action_space.n,
        results_dir=results_dir,
        total_frames=frames,
        **agent_kwargs,
    )
    probe_overrides = dict(
        seq_len=1,
        burn_in=0,
        train_interval=1,
        batch_size=1,
        train_micro_batch=1,
        buffer_capacity=max(128, num_envs * 8),
        beta_low=0.0,
        beta_high=0.0,
        use_intrinsic_rewards=False,
        epsilon_start=0.0,
        epsilon_end=0.0,
        num_policies=1,
    )
    for key, value in probe_overrides.items():
        base_kwargs.setdefault(key, value)

    agent: VectorAgent = agent_cls(**base_kwargs)
    agent.reset(num_envs)

    observations, _ = venv.reset(seed=seed)
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    completed: List[float] = []
    global_step = 0

    while global_step < frames:
        actions = agent.act(observations)
        next_obs, rewards, terminations, truncations, infos = venv.step(actions)
        final_info = infos.get("final_info", [])
        agent.observe(next_obs, rewards, terminations, truncations, final_info)
        agent.train_step()

        episode_rewards += rewards
        done_mask = np.logical_or(terminations, truncations)
        for idx, done in enumerate(done_mask):
            if done:
                completed.append(float(episode_rewards[idx]))
                episode_rewards[idx] = 0.0
        observations = next_obs
        global_step += num_envs

    venv.close()
    recent = np.mean(completed[-min(len(completed), 20):]) if completed else None
    mean_reward = float(np.mean(completed)) if completed else 0.0

    passed = None
    if spec.target_recent is not None and recent is not None:
        passed = recent >= spec.target_recent - spec.tolerance

    greedy, values = inspect_values(agent, spec.eval_states or {})
    return ProbeResult(
        name=spec.name,
        episodes=len(completed),
        mean_reward=mean_reward,
        recent_reward=None if recent is None else float(recent),
        passed=passed,
        greedy_actions=greedy,
        value_estimates=values,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MEME agent probe environments.")
    parser.add_argument("--agent", type=str, default="agent_meme:VectorMEMEAgent")
    parser.add_argument("--agent-arg", action="append", default=[], help="Extra agent kwargs key=value")
    parser.add_argument("--env", action="append", default=[], choices=sorted(PROBE_ENVS.keys()))
    parser.add_argument("--frames", type=int, default=10_000, help="Frames per probe")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(os.getcwd(), "results", "probe_envs"),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    agent_cls = load_agent_class(args.agent)
    agent_kwargs = parse_agent_kwargs(args.agent_arg)

    targets = args.env or list(PROBE_ENVS.keys())
    for env_name in targets:
        spec = PROBE_SPECS[env_name]
        env_cls = PROBE_ENVS[env_name]
        print(f"\n[probe] {env_name}: {spec.description}")
        result = run_probe(
            spec,
            env_cls,
            agent_cls,
            frames=args.frames,
            num_envs=args.num_envs,
            seed=args.seed,
            results_dir=os.path.join(args.results_dir, env_name),
            agent_kwargs=agent_kwargs,
        )
        recent = "n/a" if result.recent_reward is None else f"{result.recent_reward:.3f}"
        status = "?" if result.passed is None else ("PASS" if result.passed else "FAIL")
        print(
            f"episodes={result.episodes:4d} "
            f"mean_return={result.mean_reward:.3f} "
            f"recent_return={recent} "
            f"status={status}"
        )
        if result.value_estimates:
            print("  value estimates:")
            for tag, value in result.value_estimates.items():
                greedy = result.greedy_actions.get(tag, -1)
                print(f"    {tag:>6s}: value={value:.3f} greedy_action={greedy}")


if __name__ == "__main__":
    main()
