"""Smoke tests for SAC."""

import numpy as np
import torch

from algorithms.sac.sac import SACAgent
from framework.Logger import logger


def _make_agent(overrides=None):
    params = dict(
        num_actions=4,
        feature_dim=64,
        actor_hidden_dim=64,
        value_hidden_dim=64,
        n_stack=4,
        input_size=84,
        device='cpu',
        gamma=0.0,
        learning_rate=3e-3,
        entropy_coef=0.0,
        auto_entropy_tuning=False,
    )

    if overrides:
        params.update(overrides)

    agent = SACAgent(**params)
    return agent


_CACHED_OBS = None


def _make_obs():
    global _CACHED_OBS
    if _CACHED_OBS is None:
        _CACHED_OBS = np.ones((84, 84, 4), dtype=np.uint8) * 128
    return _CACHED_OBS.copy()


def test_bandit_prefers_rewarded_action():
    reward_prob, threshold, steps = 1.0, 0.5, 256
    np.random.seed(0)
    torch.manual_seed(0)

    agent = _make_agent()
    action_counts = np.zeros(4, dtype=int)
    target_action = 0

    obs = _make_obs()

    agent.start_episodes(obs[np.newaxis])

    for t in range(steps):
        actions, log_probs, entropy, _ = agent.select_actions(obs[np.newaxis])
        action = actions[0]
        action_counts[action] += 1

        reward = 0.0
        if action == target_action and np.random.rand() < reward_prob:
            reward = 1.0

        next_obs = _make_obs()
        done = False

        agent.update(
            obs[np.newaxis],
            actions,
            [reward],
            next_obs[np.newaxis],
            [done],
        )

        obs = next_obs

        if t % 32 == 0:
            logger.info(
                f"Step {t}: action={action}, reward={reward:.1f}, counts={action_counts}, entropy={entropy.item():.3f}, log_probs={log_probs.item():.3f}"
            )

    obs_test = _make_obs()
    features_torch, _features_np = agent._extract_features(obs_test[np.newaxis])
    logits = agent.actor(features_torch)
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

    logger.info(f"\nFinal action counts: {action_counts}")
    logger.info(f"Final policy probs: {probs}")

    assert probs[target_action] > threshold, f"Agent should prefer action {target_action}, but probs={probs}"


def test_bandit_with_life_loss_terminals():
    life_loss_every = 8
    reward_prob, threshold, steps = 1.0, 0.5, 256
    np.random.seed(1)
    torch.manual_seed(1)

    agent = _make_agent()
    action_counts = np.zeros(4, dtype=int)
    target_action = 0

    obs = _make_obs()
    agent.start_episodes(obs[np.newaxis])

    for t in range(steps):
        actions, _log_probs, _entropy, _ = agent.select_actions(obs[np.newaxis])
        action = actions[0]
        action_counts[action] += 1

        reward = 0.0
        if action == target_action and np.random.rand() < reward_prob:
            reward = 1.0

        done = (t + 1) % life_loss_every == 0

        next_obs = _make_obs()

        agent.update(
            obs[np.newaxis],
            actions,
            [reward],
            next_obs[np.newaxis],
            [done],
        )

        if done:
            agent.reset_done(done, next_obs[np.newaxis])

        obs = next_obs

        if t % 32 == 0:
            logger.info(f"Step {t}: action={action}, reward={reward:.1f}, done={done}, counts={action_counts}")

    obs_test = _make_obs()
    features_torch, _features_np = agent._extract_features(obs_test[np.newaxis])
    logits = agent.actor(features_torch)
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

    logger.info(f"\nFinal action counts: {action_counts}")
    logger.info(f"Final policy probs: {probs}")

    assert (
        probs[target_action] > threshold
    ), f"Agent should prefer action {target_action} despite terminals, but probs={probs}"


def test_value_function_learns_returns():
    np.random.seed(2)
    torch.manual_seed(2)

    agent = _make_agent()
    target_action = 0
    steps = 256

    obs = _make_obs()
    agent.start_episodes(obs[np.newaxis])

    q_values = []

    for t in range(steps):
        actions, _log_probs, _entropy, _ = agent.select_actions(obs[np.newaxis])
        action = actions[0]

        reward = 1.0 if action == target_action else 0.0

        with torch.no_grad():
            feats_t, _ = agent._extract_features(obs[np.newaxis])
            q = agent.q1(feats_t).detach().cpu().numpy()[0]
            q_values.append(q)

        next_obs = _make_obs()
        done = False

        metrics = agent.update(
            obs[np.newaxis],
            actions,
            [reward],
            next_obs[np.newaxis],
            [done],
        )

        obs = next_obs

        if t % 64 == 0:
            q = q_values[-1]
            logger.info(
                f"Step {t}: action={action}, reward={reward:.1f}, "
                f"q_target={metrics['value_target']:.3f}, q_taken={metrics['value_pred']:.3f}, "
                f"q={q}"
            )

    early_q = np.mean([q[target_action] for q in q_values[:50]])
    late_q = np.mean([q[target_action] for q in q_values[-50:]])

    logger.info(f"\nEarly Q(target): {early_q:.3f}")
    logger.info(f"Late Q(target): {late_q:.3f}")

    assert (
        late_q > early_q
    ), f"Q(target) should increase: early={early_q:.3f}, late={late_q:.3f}"


def test_lifetime_return_error_decreases():
    np.random.seed(3)
    torch.manual_seed(3)

    agent = _make_agent(overrides=dict(gamma=0.0, entropy_coef=0.0, auto_entropy_tuning=False))
    obs = _make_obs()
    agent.start_episodes(obs[np.newaxis])

    true_return = 1.0
    steps = 256
    errors = []

    for t in range(steps):
        actions, _log_probs, _entropy, _feats = agent.select_actions(obs[np.newaxis])

        metrics = agent.update(
            obs[np.newaxis],
            actions,
            [true_return],
            obs[np.newaxis],
            [False],
        )

        errors.append((true_return - metrics["value_pred"]) ** 2)

        if t % 64 == 0:
            logger.info(
                f"Step {t}: V_pred={metrics['value_pred']:.3f}, "
                f"squared_error={(true_return - metrics['value_pred']) ** 2:.4f}"
            )

    early_error = np.mean(errors[:50])
    late_error = np.mean(errors[-50:])

    logger.info(f"\nLifetime error decrease: early={early_error:.4f}, late={late_error:.4f}")
    assert (
        late_error < early_error
    ), f"Lifetime return error should decrease: early={early_error:.4f}, late={late_error:.4f}"


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Running Smoke Tests for Soft Actor-Critic Agent")
    logger.info("=" * 60)

    logger.info("\n[Test 1/3] Bandit prefers rewarded action...")
    test_bandit_prefers_rewarded_action()
    logger.info("✓ PASSED")

    logger.info("\n[Test 2/3] Bandit with life loss terminals...")
    test_bandit_with_life_loss_terminals()
    logger.info("✓ PASSED")

    logger.info("\n[Test 3/3] Value function learns returns...")
    test_value_function_learns_returns()
    logger.info("✓ PASSED")

    logger.info("\n[Test 4/4] Lifetime return error decreases...")
    test_lifetime_return_error_decreases()
    logger.info("✓ PASSED")

    logger.info("\n" + "=" * 60)
    logger.info("All smoke tests PASSED! ✓")
    logger.info("=" * 60)
