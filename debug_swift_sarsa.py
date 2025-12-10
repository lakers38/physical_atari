#!/usr/bin/env python3
"""
Debug script to diagnose Swift-Sarsa issues on Pong.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/praneet/code/physical_atari')

# Import ale_py first to register environments
import ale_py
import gymnasium as gym

from agent_swift_sarsa import AtariFeatureExtractor, SwiftSarsaCore
from swiftsarsa import SwiftSarsaBinaryFeatures


def test_feature_extraction():
    """Test that feature extraction produces sensible output."""
    print("\n=== Testing Feature Extraction ===")
    
    # Create env
    env = gym.make("ALE/Pong-v5", obs_type='rgb', full_action_space=True)
    obs, _ = env.reset()
    
    print(f"Raw observation shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min()}, {obs.max()}]")
    
    # Create extractor with SwiftTD settings
    extractor = AtariFeatureExtractor(
        height=105,
        width=80,
        num_bins=8,
        num_actions=4,  # reduced action set
        use_grayscale=False,
        use_frame_diff=False,
        use_cumulant=True,
    )
    
    print(f"\nExtractor config:")
    print(f"  Total features: {extractor.total_features}")
    print(f"  Pixel features: {extractor.pixel_features}")
    print(f"  Expected active: {extractor.get_active_feature_count()}")
    
    # Extract features
    features = extractor.extract(obs)
    feature_indices = [idx for idx, _ in features]
    print(f"\nExtracted features:")
    print(f"  Count: {len(feature_indices)}")
    print(f"  Min index: {min(feature_indices)}")
    print(f"  Max index: {max(feature_indices)}")
    print(f"  Unique: {len(set(feature_indices))}")
    
    # Check distribution of features
    features_arr = np.array(feature_indices)
    print(f"\nFeature distribution:")
    print(f"  Pixel features (< {extractor.pixel_features}): {np.sum(features_arr < extractor.pixel_features)}")
    print(f"  Action features: {np.sum((features_arr >= extractor.pixel_features) & (features_arr < extractor.pixel_features + 4))}")
    print(f"  Cumulant features: {np.sum(features_arr >= extractor.pixel_features + 4)}")
    
    env.close()
    return True


def test_q_value_differentiation():
    """Test that Q-values differentiate between actions."""
    print("\n=== Testing Q-Value Differentiation ===")
    
    # Create env
    env = gym.make("ALE/Pong-v5", obs_type='rgb', full_action_space=True)
    obs, _ = env.reset()
    
    # Create Swift-Sarsa learner
    num_actions = 4
    
    # Create extractor
    extractor = AtariFeatureExtractor(
        height=105,
        width=80,
        num_bins=8,
        num_actions=num_actions,
        use_grayscale=False,
        use_frame_diff=False,
        use_cumulant=True,
    )
    
    num_features = extractor.total_features
    
    learner = SwiftSarsaBinaryFeatures(
        num_features,
        num_actions,
        0.9,       # lambda
        1e-5,      # alpha
        1e-2,      # theta
        1.0,       # eta
        0.999,     # decay
        1e-6,      # epsilon
        3.06e-7,   # eta_min
    )
    
    features = extractor.extract(obs)
    
    # Get initial Q-values
    q_values = learner.get_action_values(features)
    print(f"Initial Q-values: {q_values}")
    print(f"  All zeros? {all(q == 0 for q in q_values)}")
    
    # Run some random steps and learn
    print("\nRunning 100 random steps with learning...")
    action_counts = [0, 0, 0, 0]
    
    for step in range(100):
        action = np.random.randint(4)
        action_counts[action] += 1
        
        # Map to ALE action
        ale_action = [2, 5, 4, 3][action]  # reduced action set
        next_obs, reward, done, truncated, _ = env.step(ale_action)
        
        # Learn
        gamma = 0.0 if done else 0.99
        learner.learn(features, float(reward), gamma, action)
        
        # Update features
        extractor.set_prev_action(action)
        if reward != 0:
            extractor.set_cumulant(reward)
        features = extractor.extract(next_obs)
        
        if done:
            learner.reset_episode()
            extractor.reset()
            obs, _ = env.reset()
            features = extractor.extract(obs)
    
    # Get Q-values after learning
    q_values_after = learner.get_action_values(features)
    print(f"\nQ-values after 100 steps: {q_values_after}")
    print(f"Action counts: {action_counts}")
    print(f"Q-value range: {max(q_values_after) - min(q_values_after):.6f}")
    
    env.close()
    return True


def test_action_selection():
    """Test action selection with exploration."""
    print("\n=== Testing Action Selection ===")
    
    # Simulate action selection
    np.random.seed(42)
    
    # Test with different Q-value scenarios
    scenarios = [
        ("All zeros", [0.0, 0.0, 0.0, 0.0]),
        ("One slightly higher", [0.0, 0.01, 0.0, 0.0]),
        ("One much higher", [0.0, 1.0, 0.0, 0.0]),
        ("Negative with one higher", [-1.0, -0.5, -1.0, -1.0]),
    ]
    
    temperature = 0.1
    epsilon = 0.1
    
    for name, q_values in scenarios:
        print(f"\nScenario: {name}")
        print(f"  Q-values: {q_values}")
        
        # Count actions over many samples
        action_counts = [0, 0, 0, 0]
        for _ in range(1000):
            # Epsilon-greedy
            if np.random.random() < epsilon:
                action = np.random.randint(4)
            else:
                # Softmax
                q_arr = np.array(q_values)
                q_shifted = q_arr - np.max(q_arr)
                q_scaled = np.clip(q_shifted / max(temperature, 1e-8), -50, 50)
                exp_q = np.exp(q_scaled)
                probs = exp_q / (np.sum(exp_q) + 1e-10)
                action = np.random.choice(4, p=probs)
            action_counts[action] += 1
        
        print(f"  Action distribution (1000 samples): {action_counts}")
        print(f"  Softmax probs: ", end="")
        q_arr = np.array(q_values)
        q_shifted = q_arr - np.max(q_arr)
        q_scaled = np.clip(q_shifted / max(temperature, 1e-8), -50, 50)
        exp_q = np.exp(q_scaled)
        probs = exp_q / (np.sum(exp_q) + 1e-10)
        print(f"{probs}")


def test_full_episode():
    """Run a full episode and see what happens."""
    print("\n=== Running Full Episode ===")
    
    env = gym.make("ALE/Pong-v5", obs_type='rgb', full_action_space=True)
    obs, _ = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    
    core = SwiftSarsaCore(
        num_envs=1,
        num_actions=4,
        seed=42,
        alpha_init=1e-5,
        theta=1e-2,
        eta=1.0,
        temperature=0.5,  # Higher temperature
        exploration_epsilon=0.3,  # More exploration
    )
    
    print(f"Core config:")
    print(f"  Feature extractor total: {core.extractors[0].total_features}")
    print(f"  Feature extractor pixels: {core.extractors[0].pixel_features}")
    
    core.reset(obs[None, ...])
    
    total_reward = 0
    action_counts = [0, 0, 0, 0]
    step_count = 0
    rewards_received = []
    
    print("\nRunning episode with verbose action logging...")
    
    while True:
        actions = core.act(obs[None, ...])
        action = int(actions[0])
        action_counts[action] += 1
        
        # Map to ALE action
        ale_action = [2, 5, 4, 3][action]
        next_obs, reward, done, truncated, _ = env.step(ale_action)
        total_reward += reward
        
        if reward != 0:
            rewards_received.append((step_count, reward))
        
        # Log first 20 actions AND after rewards
        if step_count < 20 or (reward != 0 and len(rewards_received) <= 5):
            # Get Q-values BEFORE observe
            q_before = core.agents[0].get_action_values(core.last_features[0])
            
            core.observe(
                next_obs[None, ...],
                np.array([reward]),
                np.array([done]),
                np.array([truncated]),
            )
            
            # Get Q-values AFTER observe (need to extract features first)
            features_after = core.extractors[0].extract(next_obs)
            q_after = core.agents[0].get_action_values(features_after)
            
            print(f"  Step {step_count}: action={action}, reward={reward:.0f}")
            print(f"    Q before: {[f'{q:.6f}' for q in q_before]}")
            print(f"    Q after:  {[f'{q:.6f}' for q in q_after]}")
            print(f"    TD error: {core.last_td_error:.6f}")
        else:
            core.observe(
                next_obs[None, ...],
                np.array([reward]),
                np.array([done]),
                np.array([truncated]),
            )
        
        obs = next_obs
        step_count += 1
        
        if done or truncated:
            break
    
    print(f"\nEpisode summary:")
    print(f"  Total reward: {total_reward}")
    print(f"  Steps: {step_count}")
    print(f"  Action counts: {action_counts}")
    print(f"  Rewards received: {rewards_received[:10]}...")
    
    # Final Q-values
    features_final = core.extractors[0].extract(obs)
    q_final = core.agents[0].get_action_values(features_final)
    print(f"  Final Q-values: {q_final}")
    
    env.close()


if __name__ == "__main__":
    test_feature_extraction()
    test_q_value_differentiation()
    test_action_selection()
    test_full_episode()
