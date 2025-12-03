# Soft Actor-Critic Agents for Atari

Lightweight SAC-style agents (single-env and vectorized) built with a shared CNN encoder plus two-layer MLP policy/value heads. Training scripts support optional latency simulation and reduced action sets.

## Contents
- `agent_actor_critic.py`: Single-env agent.
- `agent_actor_critic_vec.py`: Vectorized agent.
- `train_sac_sim.py`: Training script for a single environment (with optional latency model).
- `train_sac_sim_vec.py`: Training script for vectorized environments.
- `smoke_tests_actor_critic.py`: Simple bandit-style sanity checks.

## Quickstart
```bash
# Single-env training (example)
~/miniconda3/envs/robotroller/bin/python algorithms/sac/train_sac_sim.py --env ALE/MsPacman-v5 --timesteps 100000

# Vectorized training (example with 4 envs)
~/miniconda3/envs/robotroller/bin/python algorithms/sac/train_sac_sim_vec.py --env ALE/MsPacman-v5 --timesteps 400000 --num-envs 4
```

## Notes
- Latency simulation is available via `--mode sim_lat` and respects reduced action sets (`--reduce-action-set`); do not disable reduced action sets when enabling latency.
- Models checkpoint CNN, policy head, and value head together for straightforward resume.
