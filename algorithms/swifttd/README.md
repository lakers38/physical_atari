# SwiftTD Actor-Critic for Atari

Implementation of an **Actor-Critic agent with SwiftTD critic** for Atari reinforcement learning.

## Overview

This implementation combines:
- **Actor-Critic architecture** (A2C) for policy learning
- **SwiftTD** adaptive step-size TD(λ) for the critic
- **Nature DQN CNN** for feature extraction (128x128 input)
- **Online learning** with immediate updates (no replay buffer)

## Architecture

```
Input: (128, 128, 4) stacked grayscale frames
   ↓
[CNN Feature Extractor]  (Nature DQN architecture)
   ↓ (512 features)
   ├─→ [Actor Head: 512 → 256 → actions]  ← Policy gradient
   └─→ [SwiftTD Critic: 512 → V(s)]       ← Adaptive TD(λ)
```

**Key Features:**
- **Shared CNN backbone**: Trained with TD backprop
- **Actor (policy)**: Stochastic policy with entropy regularization
- **Critic (value)**: SwiftTD with per-feature adaptive step sizes
- **128x128 input**: Matches PPO setup (vs standard 84x84)

## Quick Start

### 1. Install Dependencies

```bash
# SwiftTD library (included in public/)
cd public/swifttd
pip install -e .
cd ../..

# Other dependencies
pip install torch gymnasium ale-py scipy coolname
pip install wandb  # Optional, for logging
```

### 2. Run Smoke Tests

```bash
# Verify implementation works
~/miniconda3/envs/robotroller/bin/python smoke_tests_actor_critic.py
```

Expected output:
```
[Test 1/7] Bandit prefers rewarded action... ✓ PASSED
[Test 2/7] Bandit with life loss terminals... ✓ PASSED
[Test 3/7] Bandit reward flip adapts policy... ⊘ SKIPPED
[Test 4/7] Value learning... ✓ PASSED
[Test 5/7] Entropy decreases with learning... ✓ PASSED
[Test 6/7] Save and load... ✓ PASSED
```

### 3. Train on Atari

```bash
# Quick test (Pong, 500k steps, ~30 min)
~/miniconda3/envs/robotroller/bin/python train_actor_critic_sim.py \
    --game ALE/Pong-v5 \
    --total-timesteps 500000

# Full training (MsPacman, 10M steps, ~24 hours)
~/miniconda3/envs/robotroller/bin/python train_actor_critic_sim.py \
    --game ALE/MsPacman-v5 \
    --total-timesteps 10000000 \
    --use-wandb
```

See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for detailed parameter explanations and tuning tips.

## Using with sim_latency_vec.py

The `SwiftTDVectorAgent` wrapper allows you to use this implementation with the unified `sim_latency_vec.py` training harness:

### Quick Start

```bash
# Using the helper script
cd algorithms/swifttd
./run_swifttd_latency.sh

# Or manually specify all parameters
cd ../..
~/miniconda3/envs/robotroller/bin/python sim_latency_vec.py \
    --rom MsPacman \
    --num_envs 4 \
    --total_frames 1000000 \
    --agent algorithms.swifttd.swifttd_vector_agent:SwiftTDVectorAgent \
    --agent_arg "feature_dim=512" \
    --agent_arg "learning_rate=1e-4" \
    --agent_arg "device=cuda" \
    --record_video \
    --video_every 10
```

### Custom Hyperparameters

Pass SwiftTD hyperparameters via `--agent_arg`:

```bash
~/miniconda3/envs/robotroller/bin/python sim_latency_vec.py \
    --rom Pong \
    --num_envs 8 \
    --agent algorithms.swifttd.swifttd_vector_agent:SwiftTDVectorAgent \
    --agent_arg "feature_dim=256" \
    --agent_arg "actor_hidden_dim=128" \
    --agent_arg "learning_rate=3e-4" \
    --agent_arg "lambda_=0.9" \
    --agent_arg "initial_alpha=1e-3" \
    --agent_arg "gamma=0.99" \
    --agent_arg "entropy_coef=0.01"
```

### Test the Wrapper

```bash
~/miniconda3/envs/robotroller/bin/python test_swifttd_wrapper.py
```

## Files

| File | Description |
|------|-------------|
| `actor_critic.py` | ActorCriticSwiftTD agent (single-env) |
| `swifttd_vector_agent.py` | VectorAgent wrapper for sim_latency_vec.py |
| `model.py` | CNN feature extractor (Nature DQN architecture) |
| `smoke_tests_actor_critic.py` | Unit tests for agent functionality |
| `test_swifttd_wrapper.py` | Tests for VectorAgent wrapper |
| `train_actor_critic_sim.py` | Training script with VectorSwiftTDAgent |
| `run_swifttd_latency.sh` | Helper script for sim_latency_vec.py |
| `TRAINING_GUIDE.md` | Comprehensive training and tuning guide |
| `public/swifttd/` | SwiftTD C++ library (from paper authors) |

## Algorithm Details

### Actor-Critic with SwiftTD

**Actor Update** (Policy Gradient):
```python
# Compute advantage
advantage = reward + γ * V(s') - V(s)

# Policy gradient loss
actor_loss = -log π(a|s) * advantage - β * H(π)
            └─────┬──────┘             └──┬──┘
           Policy gradient      Entropy bonus
```

**Critic Update** (SwiftTD):
```python
# SwiftTD manages adaptive step sizes α_i per feature
δ = reward + γ * V(s') - V(s)  # TD error
e = γλ * e + ∇V(s)             # Eligibility traces

# Per-feature step size adaptation
α_i = α_i * exp(θ * δ * e_i)   # IDBD meta-learning
α_i = clip(α_i * decay, η_min, η_max)

# Weight update
w_i += α_i * δ * e_i
```

**Key difference from standard A2C**: The critic uses SwiftTD's adaptive per-feature step sizes instead of a fixed learning rate.

### SwiftTD Hyperparameters

From the [SwiftTD paper](https://khurramjaved.com/swifttd.pdf):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `initial_alpha` | α₀ | 1e-3 | Initial step size |
| `max_step_size` | η | 0.1 | Maximum step size (prevents instability) |
| `step_size_decay` | - | 0.9995 | Gradual annealing |
| `meta_step_size` | θ | 1e-4 | Meta-learning rate for step-size adaptation |
| `lambda_` | λ | 0.95 | Eligibility trace decay |
| `gamma` | γ | 0.99 | Discount factor |

## Performance Expectations

### Training Progress (MsPacman, typical run):

| Timesteps | Episode Reward | Notes |
|-----------|---------------|-------|
| 0-100k | 200-500 | Random exploration |
| 100k-1M | 500-1000 | Learning basic patterns |
| 1M-5M | 1000-2000 | Improving strategy |
| 5M-10M | 2000-3000+ | Refinement |

**Note**: Performance varies significantly by game and hyperparameters. See TRAINING_GUIDE.md for tuning tips.

## Comparison with Other Methods

| Method | Update Rule | Sample Efficiency | Stability | Complexity |
|--------|-------------|-------------------|-----------|------------|
| **ActorCriticSwiftTD** | Online A2C + adaptive TD | Low (online) | Medium | High (SwiftTD params) |
| **PPO** | Batched policy gradient | High (mini-batches) | High (clipping) | Medium |
| **DQN** | Q-learning + replay | High (replay) | Medium (target net) | Medium |
| **A2C** | Online actor-critic | Low (online) | Medium | Low |

**When to use ActorCriticSwiftTD:**
- ✅ Researching adaptive step-size methods
- ✅ Testing SwiftTD's effectiveness vs fixed learning rates
- ✅ Low-memory constraints (no replay buffer)
- ✅ Want simple online learning

**When to use PPO instead:**
- ✅ Need stable, production-ready training
- ✅ Have tuned hyperparameters from literature
- ✅ Want maximum sample efficiency

## Debugging & Troubleshooting

### Common Issues

**1. Values exploding (>1000)**
```bash
# Reduce max step size
--swifttd-max-step-size 0.01
```

**2. Not learning / random performance**
```bash
# Increase learning rates and exploration
--learning-rate 3e-4 --entropy-coef 0.05
```

**3. Training crashes with NaN**
```bash
# Reduce all learning rates
--learning-rate 1e-5 --swifttd-alpha 1e-4 --swifttd-max-step-size 0.01
```

**4. "Features are all zeros" error**
- Check observations are non-zero (not using `np.zeros` for input)
- Verify CNN is properly initialized and processing frames

See **Troubleshooting** section in TRAINING_GUIDE.md for more details.

## Citation

If you use this implementation, please cite the SwiftTD paper:

```bibtex
@article{javed2024swifttd,
  title={SwiftTD: A Fast and Robust Algorithm for Temporal Difference Learning},
  author={Javed, Khurram and White, Martha},
  journal={Reinforcement Learning Journal},
  year={2024}
}
```

## References

- **SwiftTD Paper**: https://khurramjaved.com/swifttd.pdf
- **Interactive Demo**: https://khurramjaved.com/swifttd.html
- **A2C Paper**: Mnih et al. "Asynchronous Methods for Deep RL" (2016)
- **Nature DQN**: Mnih et al. "Human-level control through deep RL" (2015)

## License

SwiftTD library (`public/swifttd/`) is provided by the paper authors. Check their repository for license details.
