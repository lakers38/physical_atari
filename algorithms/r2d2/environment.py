'''Gymnasium environment wrapper for R2D2.
Updated for Gymnasium 1.1.1 API (5-value step, 2-value reset)'''
import gymnasium as gym
import ale_py
import numpy as np
import cv2
from r2d2 import config

# Register ALE environments
gym.register_envs(ale_py)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """start the game with no-op actions to provide random starting positions
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        obs, info = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            done = terminated or truncated
            if done:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)



class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        """
        super().__init__(env)
        self._width = width
        self._height = height

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(1, self._height, self._width),
            dtype=np.uint8,
        )

    def observation(self, obs):

        obs = cv2.resize(
            obs, (self._width, self._height), interpolation=cv2.INTER_AREA
        )

        obs = np.expand_dims(obs, 0)

        return obs


def create_env(env_name=config.game_name, noop_start=True, render_mode=None):
    env = gym.make(
        env_name,
        obs_type='grayscale',
        frameskip=4,
        repeat_action_probability=0,
        full_action_space=True,
        render_mode=render_mode
    )

    env = WarpFrame(env)
    if noop_start:
        env = NoopResetEnv(env)

    return env
