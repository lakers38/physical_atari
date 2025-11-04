'''Gymnasium environment wrapper for R2D2.
Updated for Gymnasium 1.1.1 API (5-value step, 2-value reset)'''
import os
import sys
import gymnasium as gym
import ale_py
import numpy as np
import cv2
from collections import deque
import config

# Import latency model for hardware latency simulation
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'latency_wrap'))
try:
    from wrapper_v0_2 import LatencyModel
    LATENCY_AVAILABLE = True
except ImportError:
    LATENCY_AVAILABLE = False
    LatencyModel = None

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


class LatencyWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that applies the LatencyModel to simulate hardware latency.

    The LatencyModel maintains a history of the past 30 actions and uses a neural
    network to predict which action should actually be executed, simulating the
    real-world delay between action selection and execution.
    """

    def __init__(self, env, latency_model_dir="./latency_wrap"):
        """
        Args:
            env: The Gymnasium environment to wrap
            latency_model_dir: Directory containing the LatencyModel weights
        """
        super().__init__(env)
        if not LATENCY_AVAILABLE or LatencyModel is None:
            raise ImportError(
                "LatencyModel not available. Please ensure latency_wrap/wrapper_v0_2.py exists."
            )
        self.latency_model = LatencyModel(directory_with_weights=latency_model_dir)
        print(f"[LatencyWrapper] Initialized with weights from {latency_model_dir}")

    def step(self, action):
        """
        Step function with latency simulation.

        Args:
            action: The action selected by the agent

        Returns:
            obs, reward, terminated, truncated, info
        """
        # Convert action to ALE action format and apply latency model
        ale_action = ale_py.Action(int(action))
        delayed_action = self.latency_model.act(ale_action)

        # Execute the delayed action in the environment
        return self.env.step(int(delayed_action))

    def reset(self, **kwargs):
        """Reset the environment and latency model state"""
        # Reset the latency model's action queue to NOOPs
        self.latency_model.action_queue = []
        for _ in range(30):
            self.latency_model.action_queue.append(
                self.latency_model._LatencyModel__one_hot_encode(0, 0, 36)
            )
        self.latency_model.last_action = 0

        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        Outputs (84, 84) - single frame without channel dimension.
        """
        super().__init__(env)
        self._width = width
        self._height = height

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width),  # (84, 84)
            dtype=np.uint8,
        )

    def observation(self, obs):
        # Resize to 84x84
        obs = cv2.resize(
            obs, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        # Return without adding channel dimension - FrameStack will handle stacking
        return obs


class FrameStack(gym.Wrapper):
    """
    Stack the last n_frames observations.

    This provides temporal information by stacking recent frames.
    Output shape: (n_frames, height, width) e.g., (4, 84, 84)
    """
    def __init__(self, env, n_frames=4):
        """
        Args:
            env: Environment to wrap
            n_frames: Number of frames to stack (default: 4)
        """
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

        # Update observation space to reflect stacked frames
        shape = env.observation_space.shape  # Should be (84, 84)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(n_frames, *shape),  # (4, 84, 84)
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        """Reset environment and initialize frame stack"""
        obs, info = self.env.reset(**kwargs)
        # Fill the frame stack with the initial observation
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        """Step environment and update frame stack"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        """Stack frames along first axis"""
        return np.stack(self.frames, axis=0)


def create_env(env_name=config.game_name, noop_start=True, render_mode=None,
               simulate_latency=False, latency_model_dir="./latency_wrap"):
    """
    Create Atari environment with preprocessing and optional latency simulation.

    Args:
        env_name: Atari environment name (e.g., "ALE/MsPacman-v5")
        noop_start: If True, apply NoopResetEnv wrapper
        render_mode: Render mode for the environment (None or "rgb_array")
        simulate_latency: If True, apply LatencyWrapper to simulate hardware delays
        latency_model_dir: Directory containing LatencyModel weights

    Returns:
        Wrapped Gymnasium environment
    """
    env = gym.make(
        env_name,
        obs_type='grayscale',
        frameskip=4,
        repeat_action_probability=0,
        full_action_space=True,
        render_mode=render_mode
    )

    # Apply latency wrapper BEFORE other wrappers if requested
    # This ensures latency affects the raw actions
    if simulate_latency:
        if not LATENCY_AVAILABLE:
            raise ImportError(
                "Latency simulation requested but LatencyModel not available. "
                "Please ensure latency_wrap/wrapper_v0_2.py exists."
            )
        env = LatencyWrapper(env, latency_model_dir=latency_model_dir)
        print(f"[create_env] Latency simulation enabled for {env_name}")

    # Apply WarpFrame to resize to 84x84
    env = WarpFrame(env)

    # Apply FrameStack to stack last 4 frames (provides temporal info)
    env = FrameStack(env, n_frames=4)

    # Apply NoopReset after frame stacking
    if noop_start:
        env = NoopResetEnv(env)

    return env
