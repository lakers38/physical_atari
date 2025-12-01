"""
Environment Setup for SwiftTD

Creates Atari environments with frame stacking and optional latency simulation.
Reuses stable-baselines3's environment utilities for consistency with PPO/R2D2.
"""

import os
import sys
import numpy as np
import gymnasium as gym
import ale_py

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, VecVideoRecorder

# Import the latency model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'latency_wrap'))
from wrapper_v0_2 import LatencyModel

# Register ALE environments
gym.register_envs(ale_py)


class VecLatencyWrapper:
    """
    Vectorized environment wrapper for latency simulation.
    Works with VecEnv from stable-baselines3.

    This wrapper applies the LatencyModel to simulate hardware latency,
    mimicking the delay between action selection and execution on real hardware.
    """

    def __init__(self, venv, latency_model_dir):
        """
        Args:
            venv: VecEnv to wrap
            latency_model_dir: Directory containing LatencyModel weights
        """
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space

        # Create separate latency model for each parallel environment
        self.latency_models = [
            LatencyModel(directory_with_weights=latency_model_dir)
            for _ in range(self.num_envs)
        ]
        print(f"[VecLatencyWrapper] Initialized {self.num_envs} latency models")

    def step_async(self, actions):
        """Apply latency model to actions before sending to environments"""
        delayed_actions = []
        for i, action in enumerate(actions):
            ale_action = ale_py.Action(int(action))
            delayed_action = self.latency_models[i].act(ale_action)
            delayed_actions.append(int(delayed_action))

        self.venv.step_async(np.array(delayed_actions))

    def step_wait(self):
        """Wait for step to complete"""
        return self.venv.step_wait()

    def step(self, actions):
        """Synchronous step"""
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        """Reset all environments and latency models"""
        for model in self.latency_models:
            model.action_queue = []
            for _ in range(30):
                model.action_queue.append(model._LatencyModel__one_hot_encode(0, 0, 36))
            model.last_action = 0

        return self.venv.reset()

    def __getattr__(self, name):
        """Forward attribute access to wrapped venv"""
        return getattr(self.venv, name)


def create_swifttd_env(
    env_name,
    n_stack=4,
    seed=0,
    simulate_latency=False,
    latency_model_dir=None,
    monitor_path=None,
    video_path=None,
    record_video=False,
    video_freq=50000,
    video_length=500
):
    """
    Create Atari environment for SwiftTD training.

    This function creates a single Atari environment (n_envs=1) for online learning,
    with frame stacking and optional latency simulation.

    Args:
        env_name: Name of the Atari environment (e.g., "ALE/MsPacman-v5")
        n_stack: Number of frames to stack (default: 4)
        seed: Random seed
        simulate_latency: If True, apply LatencyModel wrapper
        latency_model_dir: Directory containing LatencyModel weights
        monitor_path: Path for monitoring logs
        video_path: Path for video recordings
        record_video: If True, record videos during training
        video_freq: Record video every N steps
        video_length: Number of frames per video

    Returns:
        Vectorized environment with frame stacking and optional latency simulation
        Observation shape: (1, n_stack, 84, 84) - VecEnv adds batch dimension
    """
    # Create base Atari environment with standard preprocessing
    # IMPORTANT: Use full_action_space=True because LatencyModel expects 18 actions
    # n_envs=1 for online learning (no parallel environments)
    env = make_atari_env(
        env_name,
        n_envs=1,
        seed=seed,
        env_kwargs={'full_action_space': True}
    )

    print(f"[SwiftTD Env] Created {env_name}")
    print(f"  - n_envs: 1 (online learning)")
    print(f"  - full_action_space: True (18 actions)")

    # Apply latency wrapper BEFORE frame stacking
    # This ensures the latency affects the raw actions
    if simulate_latency:
        if latency_model_dir is None:
            latency_model_dir = "./latency_wrap"
        print(f"  - Latency simulation: ENABLED")
        print(f"  - Latency model dir: {latency_model_dir}")
        env = VecLatencyWrapper(env, latency_model_dir=latency_model_dir)
    else:
        print(f"  - Latency simulation: DISABLED")

    # Apply frame stacking (standard n_stack=4 for Atari)
    print(f"  - Frame stacking: {n_stack} frames")
    env = VecFrameStack(env, n_stack=n_stack)

    # Add monitoring
    if monitor_path:
        os.makedirs(monitor_path, exist_ok=True)
        print(f"  - Monitoring: {monitor_path}")
        env = VecMonitor(
            env,
            filename=os.path.join(monitor_path, f"{env_name.replace('/', '_')}_monitor.csv")
        )

    # Add video recording
    if record_video and video_path:
        print(f"  - Video recording: Every {video_freq} steps to {video_path}")
        env = VecVideoRecorder(
            env,
            video_path,
            record_video_trigger=lambda x: x % video_freq == 0,
            video_length=video_length,
        )

    print(f"  - Observation space: {env.observation_space.shape}")
    print(f"  - Action space: {env.action_space.n} actions")

    return env


def create_eval_env(
    env_name,
    n_stack=4,
    seed=100,
    simulate_latency=False,
    latency_model_dir=None
):
    """
    Create evaluation environment (no monitoring, deterministic).

    Args:
        env_name: Name of the Atari environment
        n_stack: Number of frames to stack
        seed: Random seed (different from training env)
        simulate_latency: If True, apply LatencyModel wrapper
        latency_model_dir: Directory containing LatencyModel weights

    Returns:
        Evaluation environment
    """
    return create_swifttd_env(
        env_name=env_name,
        n_stack=n_stack,
        seed=seed,
        simulate_latency=simulate_latency,
        latency_model_dir=latency_model_dir,
        monitor_path=None
    )

