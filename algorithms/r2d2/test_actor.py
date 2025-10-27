#!/usr/bin/env python3
"""
Comprehensive test script for R2D2 Actor components
Tests each component in isolation before full integration
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from r2d2.environment import create_env
from r2d2.model import Network, AgentState
from r2d2.actor import LocalBuffer, Actor
from r2d2 import config

def test_environment():
    """Test 1: Environment creation and basic operations"""
    print("=" * 60)
    print("TEST 1: Environment Creation")
    print("=" * 60)

    env = create_env(env_name='ALE/MsPacman-v5', noop_start=True)
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Action space size: {env.action_space.n}")

    obs, info = env.reset()
    print(f"✓ Reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Expected: (1, 84, 84)")
    assert obs.shape == (1, 84, 84), f"Wrong obs shape: {obs.shape}"

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Step successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Reward: {reward}")

    env.close()
    print()
    return env.action_space.n


def test_model(action_dim):
    """Test 2: Model forward pass"""
    print("=" * 60)
    print("TEST 2: Model Forward Pass")
    print("=" * 60)

    model = Network(action_dim)
    model.eval()
    print(f"✓ Model created with action_dim={action_dim}")

    # Create dummy observation
    obs = np.random.randint(0, 255, (1, 84, 84), dtype=np.uint8)

    # Create AgentState
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
    state = AgentState(obs_tensor, action_dim)
    print(f"✓ AgentState created")
    print(f"  obs shape: {state.obs.shape}")
    print(f"  last_action shape: {state.last_action.shape}")
    print(f"  last_reward shape: {state.last_reward.shape}")

    # Forward pass
    with torch.no_grad():
        q_value, hidden = model(state)

    print(f"✓ Forward pass successful")
    print(f"  q_value shape: {q_value.shape}")
    print(f"  Expected: ({action_dim},)")
    assert q_value.shape == (action_dim,), f"Wrong q_value shape: {q_value.shape}"

    print(f"  hidden[0] shape: {hidden[0].shape}")
    print(f"  hidden[1] shape: {hidden[1].shape}")
    print(f"  Expected: (1, 1, 512)")

    # Test action selection
    action = q_value.argmax().item()
    print(f"✓ Action selection successful: action={action}")
    assert 0 <= action < action_dim, f"Invalid action: {action}"

    # Test hidden state conversion for buffer
    hidden_np = torch.cat(hidden).squeeze(1).numpy()
    print(f"✓ Hidden state conversion successful")
    print(f"  hidden_np shape: {hidden_np.shape}")
    print(f"  Expected: (2, 512)")
    assert hidden_np.shape == (2, 512), f"Wrong hidden_np shape: {hidden_np.shape}"

    print()
    return q_value, hidden_np


def test_local_buffer(action_dim):
    """Test 3: LocalBuffer operations"""
    print("=" * 60)
    print("TEST 3: LocalBuffer Operations")
    print("=" * 60)

    buffer = LocalBuffer(action_dim)
    print(f"✓ LocalBuffer created")
    print(f"  block_length: {buffer.block_length}")
    print(f"  learning_steps: {buffer.learning_steps}")

    # Initialize buffer
    init_obs = np.random.randint(0, 255, (1, 84, 84), dtype=np.uint8)
    buffer.reset(init_obs)
    print(f"✓ Buffer reset")
    print(f"  obs_buffer length: {len(buffer.obs_buffer)}")
    print(f"  hidden_buffer length: {len(buffer.hidden_buffer)}")
    print(f"  hidden_buffer[0] shape: {buffer.hidden_buffer[0].shape}")

    # Add some transitions
    num_steps = min(buffer.block_length, 100)
    print(f"\n  Adding {num_steps} transitions...")

    for i in range(num_steps):
        action = np.random.randint(0, action_dim)
        reward = np.random.randn()
        next_obs = np.random.randint(0, 255, (1, 84, 84), dtype=np.uint8)
        q_value = np.random.randn(action_dim).astype(np.float32)
        hidden = np.random.randn(2, 512).astype(np.float32)

        buffer.add(action, reward, next_obs, q_value, hidden)

        if (i + 1) % 20 == 0:
            print(f"    {i+1} transitions added, buffer size: {buffer.size}")

    print(f"✓ Transitions added successfully")
    print(f"  Final buffer size: {buffer.size}")
    print(f"  obs_buffer length: {len(buffer.obs_buffer)}")
    print(f"  qval_buffer length: {len(buffer.qval_buffer)}")

    # Test finish (with bootstrap)
    print(f"\n  Testing buffer.finish() with bootstrap Q-values...")
    last_qval = np.random.randn(action_dim).astype(np.float32)

    try:
        block, priorities, episode_reward = buffer.finish(last_qval)
        print(f"✓ Buffer finish successful (with bootstrap)")
        print(f"  block.obs shape: {block.obs.shape}")
        print(f"  block.actions shape: {block.action.shape}")
        print(f"  block.n_step_reward shape: {block.n_step_reward.shape}")
        print(f"  block.hiddens shape: {block.hidden.shape}")
        print(f"  priorities shape: {priorities.shape}")
        print(f"  episode_reward: {episode_reward}")
    except Exception as e:
        print(f"✗ Buffer finish FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Reset and test finish without bootstrap (episode done)
    print(f"\n  Testing buffer.finish() without bootstrap (episode done)...")
    buffer.reset(init_obs)

    for i in range(50):
        action = np.random.randint(0, action_dim)
        reward = np.random.randn()
        next_obs = np.random.randint(0, 255, (1, 84, 84), dtype=np.uint8)
        q_value = np.random.randn(action_dim).astype(np.float32)
        hidden = np.random.randn(2, 512).astype(np.float32)
        buffer.add(action, reward, next_obs, q_value, hidden)

    try:
        block, priorities, episode_reward = buffer.finish(None)
        print(f"✓ Buffer finish successful (episode done)")
        print(f"  episode_reward: {episode_reward}")
    except Exception as e:
        print(f"✗ Buffer finish FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

    print()
    return buffer


def test_full_episode(action_dim):
    """Test 4: Full episode simulation"""
    print("=" * 60)
    print("TEST 4: Full Episode Simulation")
    print("=" * 60)

    env = create_env(env_name='ALE/MsPacman-v5', noop_start=True)
    model = Network(action_dim)
    model.eval()
    buffer = LocalBuffer(action_dim)

    print(f"✓ Environment, model, and buffer created")

    # Reset
    obs, info = env.reset()
    buffer.reset(obs)

    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
    agent_state = AgentState(obs_tensor, action_dim)

    print(f"✓ Episode reset")

    # Run episode for a few steps
    max_steps = 50
    print(f"\n  Running {max_steps} steps...")

    for step in range(max_steps):
        # Get Q-values
        with torch.no_grad():
            q_value, hidden = model(agent_state)

        # Select action
        action = q_value.argmax().item()

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Convert hidden for buffer
        hidden_np = torch.cat(hidden).squeeze(1).numpy()

        # Add to buffer
        buffer.add(action, reward, next_obs, q_value.numpy(), hidden_np)

        # Update state
        agent_state.update(next_obs, action, reward, hidden)

        if (step + 1) % 10 == 0:
            print(f"    Step {step+1}: reward={reward:.2f}, done={done}, buffer_size={buffer.size}")

        if done:
            print(f"  Episode ended at step {step+1}")
            break

    print(f"✓ Episode simulation successful")

    # Test buffer finish
    print(f"\n  Testing buffer finish...")
    try:
        if done:
            block, priorities, episode_reward = buffer.finish(None)
        else:
            with torch.no_grad():
                q_value, hidden = model(agent_state)
            block, priorities, episode_reward = buffer.finish(q_value.numpy())

        print(f"✓ Buffer finish successful")
        print(f"  Block created with {block.num_sequences} sequences")
    except Exception as e:
        print(f"✗ Buffer finish FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    env.close()
    print()


def main():
    print("\n" + "=" * 60)
    print("R2D2 ACTOR COMPONENT TESTS")
    print("=" * 60 + "\n")

    try:
        # Test 1: Environment
        action_dim = test_environment()

        # Test 2: Model
        test_model(action_dim)

        # Test 3: LocalBuffer
        test_local_buffer(action_dim)

        # Test 4: Full episode
        test_full_episode(action_dim)

        print("=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        print("\nIf all tests passed, the Actor should work correctly!")
        print("If any tests failed, review the error messages above.\n")

    except Exception as e:
        print(f"\n✗ FATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
