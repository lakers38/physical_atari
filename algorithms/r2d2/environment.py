from collections import deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np

from . import config
from utils.latency_wrap.wrapper_v0_2 import LatencyModel
from framework.Logger import logger

gym.register_envs(ale_py)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            done = terminated or truncated
            if done:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


class LatencyWrapper(gym.Wrapper):
    def __init__(self, env, latency_model_dir="./utils/latency_wrap"):
        super().__init__(env)
        self.latency_model = LatencyModel(directory_with_weights=latency_model_dir)
        logger.info("r2d2: LatencyWrapper initialized with weights from %s", latency_model_dir)

    def step(self, action):
        ale_action = ale_py.Action(int(action))
        delayed_action = self.latency_model.act(ale_action)
        return self.env.step(int(delayed_action))

    def reset(self, **kwargs):
        self.latency_model.action_queue = []
        for _ in range(30):
            self.latency_model.action_queue.append(self.latency_model._LatencyModel__one_hot_encode(0, 0, 36))
        self.latency_model.last_action = 0
        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width),
            dtype=np.uint8,
        )

    def observation(self, obs):
        obs = cv2.resize(obs, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(n_frames, *shape),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        return np.stack(self.frames, axis=0)


def create_env(
    env_name=config.game_name,
    noop_start=True,
    render_mode=None,
    simulate_latency=False,
    latency_model_dir="./utils/latency_wrap",
):
    env = gym.make(
        env_name,
        obs_type='grayscale',
        frameskip=4,
        repeat_action_probability=0,
        full_action_space=True,
        render_mode=render_mode,
    )

    if simulate_latency:
        env = LatencyWrapper(env, latency_model_dir=latency_model_dir)
        logger.info("r2d2: create_env latency simulation enabled for %s", env_name)

    env = WarpFrame(env)
    env = FrameStack(env, n_frames=4)

    if noop_start:
        env = NoopResetEnv(env)

    return env
