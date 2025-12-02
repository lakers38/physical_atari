"""
Smoke tests for SwiftTDAgent

Tests basic learning capabilities on simple bandit-style problems.
"""

import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from agent_actor_critic import SwiftTDAgent

# Import pytest only if available (for running via pytest)
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create dummy pytest decorator for standalone execution
    class DummyMark:
        @staticmethod
        def parametrize(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

    class DummyPytest:
        mark = DummyMark()

    pytest = DummyPytest()


def _make_agent(overrides=None):
    """Create a SwiftTDAgent with default params for testing."""
    params = dict(
        num_actions=4,
        feature_dim=512,
        actor_hidden_dim=256,
        n_stack=1,  # Single frame for testing
        input_size=128,
        device='cpu',
        # SwiftTD hyperparameters (aggressive for fast learning)
        lambda_=0.0,  # No eligibility traces for bandit
        gamma=0.0,  # Bandit-style (no discounting)
        initial_alpha=1e-1,  # High learning rate for fast convergence
        learning_rate=1e-2,  # High LR for actor
        entropy_coef=0.1,  # Higher entropy for exploration in tests
    )

    if overrides:
        params.update(overrides)

    agent = SwiftTDAgent(**params)
    return agent


# Cache a single observation for bandit-style tests (same state every time)
_CACHED_OBS = None

def _make_obs():
    """Create a dummy observation (128x128x1 single grayscale frame)."""
    global _CACHED_OBS
    if _CACHED_OBS is None:
        # Create a fixed non-zero observation for bandit tests
        # (all-zero observations lead to all-zero features, breaking SwiftTD learning)
        # Use a simple pattern so features are non-zero but consistent
        _CACHED_OBS = np.ones((128, 128, 1), dtype=np.uint8) * 128
    return _CACHED_OBS.copy()


@pytest.mark.parametrize("reward_prob,threshold,steps", [(1.0, 0.5, 256)])
def test_bandit_prefers_rewarded_action(reward_prob, threshold, steps):
    """
    Test that the agent learns to prefer the rewarded action in a bandit setting.

    Reward action 0 with high probability, expect policy to converge to action 0.
    """
    np.random.seed(0)
    torch.manual_seed(0)

    agent = _make_agent()
    action_counts = np.zeros(4, dtype=int)
    target_action = 0

    obs = _make_obs()

    # Initialize episode
    agent.start_episodes(obs[np.newaxis])

    for t in range(steps):
        # Select action for current step
        actions, log_probs, entropy, feats = agent.select_actions(obs[np.newaxis])
        action = actions[0]
        action_counts[action] += 1

        # Compute reward based on current action
        reward = 0.0
        if action == target_action and np.random.rand() < reward_prob:
            reward = 1.0

        # Next observation (same in bandit setting)
        next_obs = _make_obs()
        done = False

        # Update agent
        agent.update(
            obs[np.newaxis], actions, [reward],
            next_obs[np.newaxis], [done],
            log_probs, entropy, feats
        )

        obs = next_obs

        if t % 64 == 0:
            print(f"Step {t}: action={action}, reward={reward:.1f}, counts={action_counts}")

    # Check final policy
    obs_test = _make_obs()
    features_torch, features_np = agent._extract_features(obs_test[np.newaxis])
    logits = agent.actor(features_torch)
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

    print(f"\nFinal action counts: {action_counts}")
    print(f"Final policy probs: {probs}")

    assert probs[target_action] > threshold, \
        f"Agent should prefer action {target_action}, but probs={probs}"


@pytest.mark.parametrize("life_loss_every", [8])
def test_bandit_with_life_loss_terminals(life_loss_every):
    """
    Test that learning survives frequent terminals (life loss) and still prefers rewarded action.
    """
    reward_prob, threshold, steps = 1.0, 0.5, 256
    np.random.seed(1)
    torch.manual_seed(1)

    agent = _make_agent()
    action_counts = np.zeros(4, dtype=int)
    target_action = 0

    obs = _make_obs()
    agent.start_episodes(obs[np.newaxis])

    for t in range(steps):
        # Select action
        actions, log_probs, entropy, feats = agent.select_actions(obs[np.newaxis])
        action = actions[0]
        action_counts[action] += 1

        # Compute reward
        reward = 0.0
        if action == target_action and np.random.rand() < reward_prob:
            reward = 1.0

        # Terminal every N steps
        done = (t + 1) % life_loss_every == 0

        # Next observation
        next_obs = _make_obs()

        # Update agent
        agent.update(
            obs[np.newaxis], actions, [reward],
            next_obs[np.newaxis], [done],
            log_probs, entropy, feats
        )

        # Reset on terminal
        if done:
            agent.reset_done(done, next_obs[np.newaxis])

        obs = next_obs

        if t % 32 == 0:
            print(f"Step {t}: action={action}, reward={reward:.1f}, "
                  f"done={done}, counts={action_counts}")

    # Check final policy
    obs_test = _make_obs()
    features_torch, features_np = agent._extract_features(obs_test[np.newaxis])
    logits = agent.actor(features_torch)
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

    print(f"\nFinal action counts: {action_counts}")
    print(f"Final policy probs: {probs}")

    assert probs[target_action] > threshold, \
        f"Agent should prefer action {target_action} despite terminals, but probs={probs}"


def test_value_function_learns_returns():
    """
    Test that the value function (critic) learns to predict returns accurately.

    In a deterministic bandit where action 0 gives reward +1 and others give 0,
    the advantages should converge toward zero as V(s) learns the expected return.
    """
    np.random.seed(2)
    torch.manual_seed(2)

    agent = _make_agent()
    target_action = 0
    steps = 256

    obs = _make_obs()
    agent.start_episodes(obs[np.newaxis])

    advantages = []
    recent_advantages = []

    for t in range(steps):
        # Select action
        actions, log_probs, entropy, feats = agent.select_actions(obs[np.newaxis])
        action = actions[0]

        # Deterministic reward
        reward = 1.0 if action == target_action else 0.0

        # Next observation
        next_obs = _make_obs()
        done = False

        # Update agent and track advantage
        metrics = agent.update(
            obs[np.newaxis], actions, [reward],
            next_obs[np.newaxis], [done],
            log_probs, entropy, feats
        )

        advantages.append(metrics["advantage"])
        recent_advantages.append(metrics["advantage"])
        if len(recent_advantages) > 50:
            recent_advantages.pop(0)

        obs = next_obs

        if t % 64 == 0:
            recent_mean = np.mean(recent_advantages)
            recent_std = np.std(recent_advantages)
            print(f"Step {t}: action={action}, reward={reward:.1f}, "
                  f"advantage={metrics['advantage']:.3f}, "
                  f"recent_adv_mean={recent_mean:.3f}±{recent_std:.3f}")

    # Check that advantages are trending toward zero (value function learning)
    early_advantages = advantages[:50]
    late_advantages = advantages[-50:]

    early_mean_abs = np.mean(np.abs(early_advantages))
    late_mean_abs = np.mean(np.abs(late_advantages))

    print(f"\nEarly advantage magnitude: {early_mean_abs:.3f}")
    print(f"Late advantage magnitude: {late_mean_abs:.3f}")
    print(f"Improvement: {early_mean_abs - late_mean_abs:.3f}")

    # Value function should improve (advantages should decrease in magnitude)
    assert late_mean_abs < early_mean_abs, \
        f"Value function should learn: late advantages ({late_mean_abs:.3f}) should be smaller than early ({early_mean_abs:.3f})"


if __name__ == "__main__":
    # Run tests manually
    print("=" * 60)
    print("Running Smoke Tests for SwiftTDAgent")
    print("=" * 60)

    print("\n[Test 1/3] Bandit prefers rewarded action...")
    test_bandit_prefers_rewarded_action(1.0, 0.5, 256)
    print("✓ PASSED")

    print("\n[Test 2/3] Bandit with life loss terminals...")
    test_bandit_with_life_loss_terminals(8)
    print("✓ PASSED")

    print("\n[Test 3/3] Value function learns returns...")
    test_value_function_learns_returns()
    print("✓ PASSED")

    print("\n" + "=" * 60)
    print("All smoke tests PASSED! ✓")
    print("=" * 60)
