"""Smoke tests for `algorithms/ppo/agent_ppo.py`."""
import time

import numpy as np
import pytest
import torch

from framework.Logger import logger

from .agent_ppo import Agent


def _make_agent(overrides=None):
    """Create a PPO agent with small buffers for quick smoke testing."""
    params = dict(
        data_dir="/tmp",
        n_steps=16,
        n_stack=1,
        batch_size=2,
        n_epochs=4,
        frame_skip=1,
        num_actions=4,
        seed=0,
        learning_rate=1e-2,
        gamma=0.0,
        total_frames=1000,
        ent_coef=0.0,
        use_wandb=False,
    )

    if overrides:
        params.update(overrides)
    agent = Agent(**params)
    return agent


@pytest.mark.parametrize("reward_prob,threshold,steps", [(1.0, 0.5, 128)])
def test_bandit_prefers_rewarded_action(reward_prob, threshold, steps):
    """With no terminals, the policy should prefer the rewarded action."""
    agent = _make_agent(overrides=None)
    action_counts = np.zeros(4, dtype=int)
    target_action = 0
    prev_action = None

    for t in range(steps):
        reward = 0
        if prev_action is not None and prev_action == target_action and np.random.rand() < reward_prob:
            reward = 1

        obs = np.zeros((210, 160, 3), dtype=np.uint8)
        action = agent.frame(obs, reward, end_of_episode=0)
        action_counts[action] += 1
        prev_action = action
        logger.info(
            f"o: {t}\ta: {action}\tr: {reward}\tbuffer_size: {agent.rollout_buffer.size()}\tfslt:{agent.frames_since_train}"
        )

        time.sleep(0.01)
        if agent.training_thread is not None and not agent.training_thread.is_alive():
            agent.training_thread.join()
            agent.training_thread = None

    if agent.training_thread is not None:
        agent.training_thread.join(timeout=10)

    stacked_frames = np.zeros((1, 128, 128))
    obs_tensor = torch.as_tensor(stacked_frames).unsqueeze(0).to(agent.actor_model.device)
    probs = agent.actor_model.policy.get_distribution(obs_tensor).distribution.probs.detach().cpu().numpy()
    assert probs[0][0] > 0.5, f"Distribution should skew towards action 0, but distribution was: {probs[0]}"


@pytest.mark.parametrize("life_loss_every", [3])
def test_bandit_with_life_loss_terminals(life_loss_every):
    """Ensure learning survives frequent life-loss terminals and still prefers the rewarded action."""
    reward_prob, threshold, steps = 1.0, 0.5, 256
    np.random.seed(1)
    agent = _make_agent()
    action_counts = np.zeros(4, dtype=int)
    prev_action = None

    for t in range(steps):
        reward = 0
        if prev_action is not None and prev_action == 0 and np.random.rand() < reward_prob:
            reward = 1

        end_of_episode = 1 if (t + 1) % life_loss_every == 0 else 0

        obs = np.zeros((210, 160, 3), dtype=np.uint8)
        action = agent.frame(obs, reward, end_of_episode=end_of_episode)
        action_counts[action] += 1
        prev_action = action
        time.sleep(0.01)
        logger.info(f"o: {t}\ta: {action}\tr: {reward}\tbuffer_size: {agent.rollout_buffer.size()}\tend: {end_of_episode}")

        if agent.training_thread is not None and not agent.training_thread.is_alive():
            agent.training_thread.join()
            agent.training_thread = None

    if agent.training_thread is not None:
        agent.training_thread.join(timeout=10)

    stacked_frames = np.zeros((1, 128, 128))
    obs_tensor = torch.as_tensor(stacked_frames).unsqueeze(0).to(agent.actor_model.device)
    probs = agent.actor_model.policy.get_distribution(obs_tensor).distribution.probs.detach().cpu().numpy()
    assert probs[0][0] > threshold, f"Life-loss bandit failed: counts {action_counts}, probs {probs[0]}"


def test_bandit_reward_flip_adapts_policy():
    """Reward action 0 first, then flip to action 1 and expect the policy to follow."""
    reward_prob = 1.0
    flip_at = 96
    steps = 256
    threshold_pre = 0.4
    threshold_post = 0.4

    np.random.seed(2)
    agent = _make_agent()
    action_counts = np.zeros(4, dtype=int)
    prev_action = None

    pre_flip_probs = None

    for t in range(steps):
        current_target = 0 if t < flip_at else 1
        reward = 0
        if prev_action is not None and prev_action == current_target and np.random.rand() < reward_prob:
            reward = 1

        obs = np.zeros((210, 160, 3), dtype=np.uint8)
        action = agent.frame(obs, reward, end_of_episode=0)
        action_counts[action] += 1
        prev_action = action
        time.sleep(0.01)
        logger.info(f"o: {t}\ta: {action}\tr: {reward}\ttarget: {current_target}\tbuffer_size: {agent.rollout_buffer.size()}")

        if agent.training_thread is not None and not agent.training_thread.is_alive():
            agent.training_thread.join()
            agent.training_thread = None

        if t == flip_at:
            stacked_frames = np.zeros((1, 128, 128))
            obs_tensor = torch.as_tensor(stacked_frames).unsqueeze(0).to(agent.actor_model.device)
            pre_flip_probs = (
                agent.actor_model.policy.get_distribution(obs_tensor).distribution.probs[0].detach().cpu().numpy()
            )
            logger.info(f"Post-flip snapshot probs: {pre_flip_probs}")

    if agent.training_thread is not None:
        agent.training_thread.join(timeout=10)

    stacked_frames = np.zeros((1, 128, 128))
    obs_tensor = torch.as_tensor(stacked_frames).unsqueeze(0).to(agent.actor_model.device)
    probs = agent.actor_model.policy.get_distribution(obs_tensor).distribution.probs[0].detach().cpu().numpy()

    assert pre_flip_probs[0] > threshold_pre, f"Pre-flip preference missing: probs {pre_flip_probs}"
    assert probs[1] > threshold_post, f"Post-flip preference missing: probs {probs}"
