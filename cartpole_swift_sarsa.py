import argparse
import random
from typing import List

import gymnasium as gym
import numpy as np

from swiftsarsa import SwiftSarsaBinaryFeatures


def make_feature_bins(num_bins: int) -> List[np.ndarray]:
    # Hand-tuned clip ranges based on CartPole obs bounds.
    # cart position, cart velocity, pole angle, pole velocity at tip
    ranges = [(-4.8, 4.8), (-5.0, 5.0), (-0.418, 0.418), (-5.0, 5.0)]
    bins = []
    for low, high in ranges:
        bins.append(np.linspace(low, high, num_bins - 1, dtype=np.float32))
    return bins


def featurize(obs: np.ndarray, bins: List[np.ndarray]) -> List[int]:
    idxs: List[int] = []
    offset = 0
    for i, b in enumerate(bins):
        v = float(np.clip(obs[i], b[0], b[-1]))
        bin_idx = int(np.digitize(v, b, right=False))
        idxs.append(offset + bin_idx)
        offset += len(b) + 1  # number of bins for this dimension
    return idxs


def select_action(q_values: np.ndarray, eps: float, temp: float, use_softmax: bool) -> int:
    if np.random.random() < eps:
        return int(np.random.randint(len(q_values)))
    if use_softmax:
        shifted = q_values - np.max(q_values)
        probs = np.exp(np.clip(shifted / temp, -50, 50))
        probs = probs / (probs.sum() + 1e-12)
        if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
            return int(np.random.randint(len(q_values)))
        return int(np.random.choice(len(q_values), p=probs))
    return int(np.argmax(q_values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=2e-3)
    parser.add_argument("--meta_step", type=float, default=1e-4,
                        help="Meta step size (theta) for Swift-Sarsa; matches paper runner defaults.")
    parser.add_argument("--eta", type=float, default=1.0,
                        help="Step-size budget (eta) for Swift-Sarsa.")
    parser.add_argument("--swift_decay", type=float, default=0.999,
                        help="Per-feature step-size decay factor.")
    parser.add_argument("--swift_trace_epsilon", type=float, default=1e-6,
                        help="Trace culling threshold epsilon; traces dropped when z <= last_alpha * epsilon.")
    parser.add_argument("--swift_eta_min", type=float, default=3.06e-7,
                        help="Minimum per-feature step size (eta_min).")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Starting epsilon for exploration.")
    parser.add_argument("--epsilon_final", type=float, default=None, help="Final epsilon; if unset, no decay.")
    parser.add_argument("--epsilon_decay_episodes", type=int, default=0,
                        help="Episodes over which to linearly decay epsilon (0 = no decay).")
    parser.add_argument("--temperature", type=float, default=0.999)
    parser.add_argument("--pure_epsilon", action="store_true", help="Disable softmax; use epsilon-greedy only.")
    parser.add_argument("--softmax_only", action="store_true",
                        help="Ignore epsilon and use softmax (temperature) exploration only.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (gamma).")
    parser.add_argument("--discount", type=float, default=None,
                        help="Optional alias for gamma; if set, overrides --gamma.")
    parser.add_argument("--num_bins", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=0, help="If >0, run greedy eval episodes after training.")
    parser.add_argument("--eval_temperature", type=float, default=0.1, help="Softmax temperature during eval.")
    parser.add_argument("--eval_softmax", action="store_true", help="Use softmax (temp) during eval instead of greedy argmax.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)

    eps_start = args.epsilon
    eps_final = args.epsilon if args.epsilon_final is None else args.epsilon_final
    decay_episodes = max(args.epsilon_decay_episodes, 0)
    current_eps = eps_start

    bins = make_feature_bins(args.num_bins)
    total_features = sum(len(b) + 1 for b in bins)
    num_actions = env.action_space.n

    learner = SwiftSarsaBinaryFeatures(
        total_features,
        num_actions,
        0.9,                                  # lambda
        max(args.alpha, 1e-8),
        max(args.meta_step, 0.0),
        args.eta,
        args.swift_decay,
        args.swift_trace_epsilon,
        args.swift_eta_min,
    )

    returns = []
    for ep in range(args.episodes):
        learner.reset_episode()
        obs, _ = env.reset()
        features = featurize(obs, bins)
        q_values = np.array(learner.get_action_values(features))
        effective_eps = 0.0 if args.softmax_only else current_eps
        action = select_action(q_values, effective_eps, args.temperature, use_softmax=not args.pure_epsilon or args.softmax_only)


        learner.learn(features, 0.0, 0.0, action)
        
        ep_ret = 0.0
        done = False
        step = 0
        while not done and step < args.max_steps:
            next_obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += float(r)
            done = bool(terminated or truncated)
            reward = float(r)
            effective_gamma = args.discount if args.discount is not None else args.gamma
            g = 0.0 if done else effective_gamma

            

            features_next = featurize(next_obs, bins)
            q_next = np.array(learner.get_action_values(features_next))
            action_next = select_action(q_next, effective_eps, args.temperature, use_softmax=not args.pure_epsilon or args.softmax_only)

            if args.alpha > 0:
                learner.learn(features_next, reward, g, action)

            features = features_next
            action = action_next
            step += 1

        returns.append(ep_ret)
        # Linear epsilon decay
        if decay_episodes > 0:
            frac = min(ep + 1, decay_episodes) / decay_episodes
            current_eps = eps_start + (eps_final - eps_start) * frac
        else:
            current_eps = eps_start
        if args.log_every and (ep + 1) % args.log_every == 0:
            recent = returns[-args.log_every:]
            print(
                f"ep {ep + 1}  return {ep_ret:.1f}  "
                f"mean_last_{args.log_every}={np.mean(recent):.2f}  "
                f"eps={effective_eps:.3f}"
            )

    print(f"Finished {args.episodes} episodes. Mean return={np.mean(returns):.2f}, best={np.max(returns):.2f}")

    # Greedy/softmax eval with frozen weights
    if args.eval_episodes > 0:
        eval_returns = []
        for ep in range(args.eval_episodes):
            obs, _ = env.reset()
            features = featurize(obs, bins)
            q_values = np.array(learner.get_action_values(features))
            # Eval: no exploration epsilon; softmax optional for tie-breaking
            eval_eps = 0.0
            action = select_action(q_values, eval_eps, args.eval_temperature,
                                   use_softmax=args.eval_softmax or args.softmax_only)

            ep_ret = 0.0
            done = False
            step = 0
            while not done and step < args.max_steps:
                next_obs, r, terminated, truncated, _ = env.step(action)
                ep_ret += float(r)
                done = bool(terminated or truncated)

                features_next = featurize(next_obs, bins)
                q_next = np.array(learner.get_action_values(features_next))
                action = select_action(q_next, eval_eps, args.eval_temperature,
                                       use_softmax=args.eval_softmax or args.softmax_only)

                features = features_next
                step += 1

            eval_returns.append(ep_ret)

        print(
            f"Eval ({args.eval_episodes} eps, eps=0, "
            f"{'softmax' if args.eval_softmax else 'greedy'}, temp={args.eval_temperature}): "
            f"mean={np.mean(eval_returns):.2f}, best={np.max(eval_returns):.2f}"
        )

    env.close()


if __name__ == "__main__":
    main()
