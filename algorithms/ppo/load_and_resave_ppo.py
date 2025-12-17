#!/usr/bin/env python3
"""
Load PPO model from unzipped .pth files and re-save with numpy 1.x compatibility
Uses the already-extracted policy.pth, policy.optimizer.pth files
"""

import os

import ale_py
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Register ALE environments
gym.register_envs(ale_py)


def create_matching_env(env_name="ALE/MsPacman-v5"):
    """
    Create environment with EXACT same settings as training
    (from train_agent_ppo_sim.py)
    """
    env = make_atari_env(
        env_name,
        n_envs=1,
        seed=0,
        env_kwargs={'full_action_space': True},  # 18 actions
        wrapper_kwargs={"screen_size": 128},  # 128x128
    )
    env = VecFrameStack(env, n_stack=16)  # 16 frames stacked (matching your training)

    print("Environment created:")
    print(f"  Observation shape: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n} actions")

    return env


def main():
    print("=" * 70)
    print("Loading PPO model from unzipped .pth files")
    print("=" * 70)

    # Check that files exist
    required_files = ["policy.pth", "policy.optimizer.pth"]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"\n✗ Missing files: {missing}")
        print("  Make sure policy.pth and policy.optimizer.pth are in current directory")
        return

    print("\n✓ Found required .pth files")

    # Create matching environment
    print("\nCreating environment (matching training config)...")
    env = create_matching_env("ALE/MsPacman-v5")

    # Create a new PPO model with the correct architecture
    print("\nCreating new PPO model...")
    model = PPO(
        "CnnPolicy",
        env,
        device="cpu",  # Use CPU for loading to avoid CUDA issues
        verbose=1,
    )

    # Load the saved weights
    print("\nLoading policy weights from policy.pth...")
    try:
        policy_state = torch.load("policy.pth", map_location="cpu")
        model.policy.load_state_dict(policy_state)
        print("  ✓ Policy loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading policy: {e}")
        return

    print("\nLoading optimizer state from policy.optimizer.pth...")
    try:
        optimizer_state = torch.load("policy.optimizer.pth", map_location="cpu")
        model.policy.optimizer.load_state_dict(optimizer_state)
        print("  ✓ Optimizer loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading optimizer: {e}")
        return

    # Save with numpy 1.x compatibility
    print("\nSaving model as ppo_10m_numpy1.zip...")
    model.save("ppo_10m_numpy1")
    print("  ✓ Model saved!")

    # Verify the saved file exists
    if os.path.exists("ppo_10m_numpy1.zip"):
        size_mb = os.path.getsize("ppo_10m_numpy1.zip") / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")

    print("\n" + "=" * 70)
    print("SUCCESS! Model re-saved with numpy 1.x compatibility")
    print("=" * 70)
    print("\nUse in harness_physical.py with:")
    print("  --load_model=ppo_10m_numpy1")
    print()


if __name__ == "__main__":
    main()
