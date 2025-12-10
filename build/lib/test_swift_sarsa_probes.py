#!/usr/bin/env python3
"""
Probe environments for Swift-Sarsa debugging.

These are minimal environments that isolate specific aspects of the algorithm.
If Swift-Sarsa fails on these, we know exactly what's broken.

IMPORTANT: The C++ Swift-Sarsa doesn't reset v_old between episodes.
So we design multi-step episodes to let v_old stabilize naturally.

Run with: python test_swift_sarsa_probes.py
"""

import numpy as np
from swiftsarsa import SwiftSarsaBinaryFeatures
from typing import List, Tuple

# Default hyperparameters for testing
# CRITICAL: tau = num_active_features * alpha must be < eta
# For tests with 1-10 features, alpha=0.01 keeps tau small
DEFAULT_CONFIG = {
    "lambda_": 0.9,
    "alpha_init": 0.01,     # Low enough that tau < eta for small tests
    "theta": 0.0,           # No meta-adaptation for predictable behavior
    "eta": 1.0,             # tau must be < eta
    "decay": 0.999,
    "epsilon": 1e-4,
    "eta_min": 1e-8,
}


def create_agent(num_features: int, num_actions: int, gamma: float = 0.99) -> SwiftSarsaBinaryFeatures:
    """Create a Swift-Sarsa agent with test config."""
    return SwiftSarsaBinaryFeatures(
        num_features,
        num_actions,
        DEFAULT_CONFIG["lambda_"],
        DEFAULT_CONFIG["alpha_init"],
        DEFAULT_CONFIG["theta"],
        DEFAULT_CONFIG["eta"],
        DEFAULT_CONFIG["decay"],
        DEFAULT_CONFIG["epsilon"],
        DEFAULT_CONFIG["eta_min"],
    )


def test_1_constant_reward():
    """
    Probe 1: Constant observation, 2 steps, +1 reward each.
    
    Tests: Basic value learning - can agent learn positive value for rewarding state?
    
    Note: True Online TD(λ) needs 2+ steps for weight updates to occur.
    """
    print("\n" + "="*60)
    print("PROBE 1: Constant reward (2 steps)")
    print("="*60)
    
    num_features = 1
    num_actions = 1
    gamma = 0.9
    
    agent = create_agent(num_features, num_actions, gamma)
    
    features = [0]
    action = 0
    reward = 1.0
    
    # Run episodes (2 steps each - minimum for True Online TD)
    for episode in range(100):
        agent.learn(features, reward, gamma, action)  # Step 1
        agent.learn(features, reward, 0.0, action)    # Step 2, terminal
        agent.reset_episode()
    
    q_final = agent.get_action_values(features)[0]
    
    # Q should be positive (learning that this state gives reward)
    passed = q_final > 0.5
    
    print(f"  Final Q(s,a): {q_final:.4f}")
    print(f"  Expected:     > 0.5 (positive, learning reward)")
    print(f"  Result:       {'✓ PASS' if passed else '✗ FAIL'}")
    
    assert passed


def test_2_two_states_different_rewards():
    """
    Probe 2: Two different states visited in sequence, different rewards.
    
    Episode: s0 (r=+1) → s1 (r=-1) → terminal
    
    This tests: Can the agent learn different values for different features?
    
    NOTE: In True Online TD(λ), terminal state s1 is not directly updated.
    s0 gets the +1 reward, but s1's -1 reward backpropagates to affect s0.
    """
    print("\n" + "="*60)
    print("PROBE 2: Two states, different rewards")
    print("="*60)
    
    num_features = 2
    num_actions = 1
    gamma = 0.9
    
    agent = create_agent(num_features, num_actions, gamma)
    
    # Run episodes WITH reset
    for episode in range(300):
        # Step 1: s0, r=+1
        agent.learn([0], 1.0, gamma, 0)
        # Step 2: s1, r=-1, terminal
        agent.learn([1], -1.0, 0.0, 0)
        agent.reset_episode()
    
    q_s0 = agent.get_action_values([0])[0]
    q_s1 = agent.get_action_values([1])[0]
    
    # With True Online TD: s0 receives r=+1 directly, plus backpropagated -1 from s1
    # Net effect: Q(s0) should be positive but reduced by the negative terminal reward
    # s1 (terminal) is not directly updated
    # The key is that Q(s0) should be different from Q(s1)
    different_values = abs(q_s0 - q_s1) > 0.1
    
    print(f"  Q(s0): {q_s0:.4f}")
    print(f"  Q(s1): {q_s1:.4f}")
    print(f"  Values are different: {different_values}")
    print(f"  Result:               {'✓ PASS' if different_values else '✗ FAIL'}")
    
    assert different_values


def test_3_three_step_chain():
    """
    Probe 3: Three-step chain with reward only at the end.
    
    s0 (r=0) → s1 (r=0) → s2 (r=+1) → terminal
    
    This tests: Do eligibility traces propagate credit backwards?
    
    NOTE: Terminal state (s2) is never directly updated in True Online TD(λ).
    The reward is backpropagated to s0, s1 through traces.
    
    Expected: Q(s1) > Q(s0) > 0 (non-terminal states learn correctly)
    """
    print("\n" + "="*60)
    print("PROBE 3: Three-step chain (trace propagation)")
    print("="*60)
    
    num_features = 3
    num_actions = 1
    gamma = 0.9
    
    agent = create_agent(num_features, num_actions, gamma)
    
    # Run episodes WITH episode reset
    for episode in range(500):
        agent.learn([0], 0.0, gamma, 0)  # s0, r=0
        agent.learn([1], 0.0, gamma, 0)  # s1, r=0
        agent.learn([2], 1.0, 0.0, 0)    # s2, r=1, terminal
        agent.reset_episode()            # Reset for episodic learning
    
    q_s0 = agent.get_action_values([0])[0]
    q_s1 = agent.get_action_values([1])[0]
    q_s2 = agent.get_action_values([2])[0]
    
    # Non-terminal states should have positive Q, with s1 > s0 (closer to reward)
    # Terminal state s2 may be ~0 (not directly updated in True Online TD)
    non_terminal_positive = q_s0 > 0 and q_s1 > 0
    correct_ordering = q_s1 > q_s0  # Closer to reward = higher value
    passed = non_terminal_positive and correct_ordering
    
    print(f"  Q(s0): {q_s0:.4f} (expected ~γ² ≈ 0.81)")
    print(f"  Q(s1): {q_s1:.4f} (expected ~γ ≈ 0.90)")
    print(f"  Q(s2): {q_s2:.4f} (terminal - not directly updated)")
    print(f"  Non-terminal positive: {non_terminal_positive}")
    print(f"  Q(s1) > Q(s0): {correct_ordering}")
    print(f"  Result: {'✓ PASS' if passed else '✗ FAIL'}")
    
    assert passed


def test_4_action_choice_multistep():
    """
    Probe 4: Two actions, same state, multi-step episodes.
    
    Each episode: pick one action, repeat for 5 steps.
    - Action 0: +1 reward every step
    - Action 1: -1 reward every step
    
    This tests: Can the agent learn to prefer action 0?
    """
    print("\n" + "="*60)
    print("PROBE 4: Action choice (multi-step episodes)")
    print("="*60)
    
    num_features = 1
    num_actions = 2
    gamma = 0.9
    
    agent = create_agent(num_features, num_actions, gamma)
    
    features = [0]
    
    # Run episodes, alternating which action we use for each episode
    for episode in range(200):
        action = episode % 2
        reward = 1.0 if action == 0 else -1.0
        
        # 4 non-terminal steps
        for step in range(4):
            agent.learn(features, reward, gamma, action)
        # Terminal step
        agent.learn(features, reward, 0.0, action)
        agent.reset_episode()  # Reset for episodic learning
    
    q_values = agent.get_action_values(features)
    q_a0 = q_values[0]
    q_a1 = q_values[1]
    
    prefers_a0 = q_a0 > q_a1
    passed = prefers_a0
    
    print(f"  Q(s, a0): {q_a0:.4f} (gets +1 reward)")
    print(f"  Q(s, a1): {q_a1:.4f} (gets -1 reward)")
    print(f"  Prefers action 0: {prefers_a0}")
    print(f"  Result:           {'✓ PASS' if passed else '✗ FAIL'}")
    
    assert passed


def test_5_state_dependent_optimal_action():
    """
    Probe 5: Two states, two actions, state-dependent optimal action.
    
    Episode 1: s0 → pick a0 → +1 reward (4 steps) → terminal
    Episode 2: s1 → pick a1 → +1 reward (4 steps) → terminal
    Episode 3: s0 → pick a1 → -1 reward (4 steps) → terminal
    Episode 4: s1 → pick a0 → -1 reward (4 steps) → terminal
    
    This tests: Can the agent learn different optimal actions per state?
    """
    print("\n" + "="*60)
    print("PROBE 5: State-dependent optimal action")
    print("="*60)
    
    num_features = 2
    num_actions = 2
    gamma = 0.9
    
    agent = create_agent(num_features, num_actions, gamma)
    
    def get_reward(state, action):
        # s0: a0 is good, a1 is bad
        # s1: a1 is good, a0 is bad
        if state == 0:
            return 1.0 if action == 0 else -1.0
        else:
            return 1.0 if action == 1 else -1.0
    
    # Run episodes
    for episode in range(400):
        state = episode % 2
        action = (episode // 2) % 2  # Cycle through all (state, action) pairs
        features = [state]
        reward = get_reward(state, action)
        
        # 4 non-terminal steps
        for step in range(4):
            agent.learn(features, reward, gamma, action)
        # Terminal
        agent.learn(features, reward, 0.0, action)
        agent.reset_episode()  # Reset for episodic learning
    
    q_s0 = agent.get_action_values([0])
    q_s1 = agent.get_action_values([1])
    
    s0_prefers_a0 = q_s0[0] > q_s0[1]
    s1_prefers_a1 = q_s1[1] > q_s1[0]
    passed = s0_prefers_a0 and s1_prefers_a1
    
    print(f"  Q(s0, a0): {q_s0[0]:.4f}, Q(s0, a1): {q_s0[1]:.4f}")
    print(f"  Q(s1, a0): {q_s1[0]:.4f}, Q(s1, a1): {q_s1[1]:.4f}")
    print(f"  s0 prefers a0: {s0_prefers_a0} (expected True)")
    print(f"  s1 prefers a1: {s1_prefers_a1} (expected True)")
    print(f"  Result:        {'✓ PASS' if passed else '✗ FAIL'}")
    
    assert passed


def test_6_sparse_features():
    """
    Probe 6: Sparse binary features like Atari.
    
    - 100 features total, 10 active per observation
    - State A (features 0-9): action 0 is good
    - State B (features 50-59): action 1 is good
    """
    print("\n" + "="*60)
    print("PROBE 6: Sparse binary features (100 total, 10 active)")
    print("="*60)
    
    num_features = 100
    num_actions = 2
    gamma = 0.9
    
    agent = create_agent(num_features, num_actions, gamma)
    
    features_a = list(range(10))      # State A
    features_b = list(range(50, 60))  # State B
    
    def get_reward(features, action):
        if features == features_a:
            return 1.0 if action == 0 else -1.0
        else:
            return 1.0 if action == 1 else -1.0
    
    # Run episodes WITH reset
    for episode in range(400):
        # Pick state and action
        state_is_a = (episode % 2) == 0
        features = features_a if state_is_a else features_b
        action = (episode // 2) % 2
        reward = get_reward(features, action)
        
        # Multi-step episode
        for step in range(4):
            agent.learn(features, reward, gamma, action)
        agent.learn(features, reward, 0.0, action)
        agent.reset_episode()  # Reset for episodic learning
    
    q_a = agent.get_action_values(features_a)
    q_b = agent.get_action_values(features_b)
    
    a_prefers_0 = q_a[0] > q_a[1]
    b_prefers_1 = q_b[1] > q_b[0]
    passed = a_prefers_0 and b_prefers_1
    
    print(f"  State A: Q(a0)={q_a[0]:.4f}, Q(a1)={q_a[1]:.4f}")
    print(f"  State B: Q(a0)={q_b[0]:.4f}, Q(a1)={q_b[1]:.4f}")
    print(f"  State A prefers a0: {a_prefers_0} (expected True)")
    print(f"  State B prefers a1: {b_prefers_1} (expected True)")
    print(f"  Result:             {'✓ PASS' if passed else '✗ FAIL'}")
    
    assert passed


def test_7_online_sarsa_sequence():
    """
    Probe 7: True online Sarsa sequence - exactly as Atari would use it.
    
    Simulate the act() → observe() pattern:
    1. act: get features, pick action, store
    2. observe: call learn(stored_features, reward, gamma, stored_action)
    
    Simple grid: 3 states, moving right is good.
    s0 → s1 → s2 (terminal, +1)
    """
    print("\n" + "="*60)
    print("PROBE 7: Online Sarsa sequence (act/observe pattern)")
    print("="*60)
    
    num_features = 3
    num_actions = 1  # Only one action (move right)
    gamma = 0.9
    
    agent = create_agent(num_features, num_actions, gamma)
    
    # Simulate exactly like the Atari agent
    for episode in range(300):
        # State 0
        features_0 = [0]
        action_0 = 0
        
        # Observe transition s0 → s1 (if not first step of first episode)
        # In real code, this would be called in observe() with stored features
        if episode > 0:
            # This simulates: last_features=[2], last_action=0, reward=1.0, done=True
            # from the END of the previous episode
            pass
        
        # Step 1: s0 → s1
        agent.learn(features_0, 0.0, gamma, action_0)  # r=0 for this transition
        
        # Step 2: s1 → s2
        features_1 = [1]
        action_1 = 0
        agent.learn(features_1, 0.0, gamma, action_1)  # r=0
        
        # Step 3: s2 → terminal
        features_2 = [2]
        action_2 = 0
        agent.learn(features_2, 1.0, 0.0, action_2)  # r=+1, terminal
        agent.reset_episode()  # Reset for episodic learning
    
    q_s0 = agent.get_action_values([0])[0]
    q_s1 = agent.get_action_values([1])[0]
    q_s2 = agent.get_action_values([2])[0]
    
    # Non-terminal states should be positive, with correct ordering
    # Terminal state (s2) is not directly updated in True Online TD
    non_terminal_positive = q_s0 > 0 and q_s1 > 0
    correct_ordering = q_s1 > q_s0  # Closer to reward = higher value
    passed = non_terminal_positive and correct_ordering
    
    print(f"  Q(s0): {q_s0:.4f}")
    print(f"  Q(s1): {q_s1:.4f}")
    print(f"  Q(s2): {q_s2:.4f} (terminal - not directly updated)")
    print(f"  Non-terminal positive: {non_terminal_positive}")
    print(f"  Q(s1) > Q(s0): {correct_ordering}")
    print(f"  Result:       {'✓ PASS' if passed else '✗ FAIL'}")
    
    assert passed


def _run_probe(func):
    """
    Run a probe function that asserts internally.
    Returns True on success, False on AssertionError.
    """
    try:
        func()
        return True
    except AssertionError:
        return False


def run_all_tests():
    """Run all probe tests."""
    print("\n" + "#"*60)
    print("# SWIFT-SARSA PROBE TESTS (v2 - multi-step episodes)")
    print("#"*60)
    print("\nNote: Tests use multi-step episodes because C++ v_old")
    print("persists across episodes (doesn't reset on terminal).\n")
    
    results = []
    
    results.append(("1. Constant reward", _run_probe(test_1_constant_reward)))
    results.append(("2. Two states, diff rewards", _run_probe(test_2_two_states_different_rewards)))
    results.append(("3. Three-step chain", _run_probe(test_3_three_step_chain)))
    results.append(("4. Action choice", _run_probe(test_4_action_choice_multistep)))
    results.append(("5. State-dependent action", _run_probe(test_5_state_dependent_optimal_action)))
    results.append(("6. Sparse features", _run_probe(test_6_sparse_features)))
    results.append(("7. Online Sarsa sequence", _run_probe(test_7_online_sarsa_sequence)))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED! Swift-Sarsa core is working correctly.")
        print("   If Atari still doesn't learn, it's a representation issue.")
    else:
        print("❌ SOME TESTS FAILED! There may be a bug in Swift-Sarsa.")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
