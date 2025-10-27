# SwiftTD Q-Learning for Atari

This directory contains a SwiftTD-based Q-learning implementation for Atari games, designed to be consistent with the PPO and R2D2 training setups.

## Overview

**SwiftTD** is an algorithm for temporal difference learning with adaptive step sizes. This implementation uses 18 separate SwiftTD learners (one per action) for Q-learning on Atari games.

### Architecture

```
Environment (frame-stacked) → CNN Feature Extractor → 18 SwiftTD Q-Learners → ε-greedy Action Selection
                                 (512-dim features)      Q(s,a₀)...Q(s,a₁₇)
```

**Key Components:**
- **Frame Stacking**: 4 consecutive frames (4, 84, 84) for temporal information
- **CNN**: Nature DQN architecture adapted for 4-channel input → 512 features
- **18 Q-Learners**: Separate SwiftTD learners for each action
- **Online Learning**: Updates after each step (no replay buffer)
- **Latency Simulation**: Optional LatencyModel wrapper for sim_lat mode

## Files

- `config.py` - Hyperparameters and training settings
- `model.py` - CNN feature extractor (Nature DQN architecture)
- `agent.py` - SwiftTDAgent class with 18 Q-learners
- `environment.py` - Environment creation with frame stacking and latency
- `train_agent_swifttd_sim.py` - Main training script
- `__init__.py` - Package initialization

## Training

### Basic Usage

```bash
# Train without latency (baseline)
python algorithms/swifttd/train_agent_swifttd_sim.py \
  --env ALE/MsPacman-v5 \
  --timesteps 1000000 \
  --mode sim \
  --device cuda

# Train with latency simulation
python algorithms/swifttd/train_agent_swifttd_sim.py \
  --env ALE/MsPacman-v5 \
  --timesteps 1000000 \
  --mode sim_lat \
  --latency-model-dir ./latency_wrap \
  --device cuda
```

### Command-Line Arguments

- `--env`: Atari environment name (default: ALE/MsPacman-v5)
- `--timesteps`: Total training timesteps (default: 1M)
- `--mode`: Training mode - `sim` (no latency) or `sim_lat` (with LatencyModel)
- `--latency-model-dir`: Directory containing LatencyModel weights
- `--output-dir`: Base directory for outputs (default: outputs/swifttd/)
- `--device`: Device for training - `cuda`, `cpu`, or `mps`
- `--seed`: Random seed
- `--wandb`: Enable Weights & Biases logging
- `--wandb-project`: WandB project name
- `--wandb-entity`: WandB entity/team name

### Output Structure

```
outputs/swifttd/
  └── {mode}/
      └── {env_name}/
          └── {run_name}/
              ├── config.txt
              ├── logs/
              │   └── monitor/
              ├── checkpoints/
              └── final_model.pth
```

## SwiftTD Hyperparameters

Key hyperparameters in `config.py`:

```python
# SwiftTD learner parameters
lambda_ = 0.95           # Eligibility trace decay
initial_alpha = 1e-3     # Initial learning rate
gamma = 0.99             # Discount factor
max_step_size = 0.1      # Maximum step size
step_size_decay = 0.9995 # Decay rate per step
meta_step_size = 1e-4    # Meta learning rate

# Exploration
epsilon_start = 1.0      # Initial exploration
epsilon_end = 0.01       # Final exploration
epsilon_decay_steps = 250_000  # Decay period

# CNN training
cnn_learning_rate = 1e-4 # CNN optimizer learning rate
num_features = 512       # Feature dimension
```

## How It Works

### Q-Learning with Multiple Learners

1. **18 Separate Learners**: One SwiftTD learner per action learns Q(s, aᵢ)
2. **Action Selection**:
   - Extract features φ(s) from CNN
   - Query all 18 learners to get Q-values
   - Select action: ε-greedy (random with prob ε, else argmax Q)
3. **Update**:
   - Only update the learner for the action taken
   - SwiftTD handles TD(λ) updates with eligibility traces
   - Adaptive step sizes for stable learning

### Feature Extraction

- **Input**: 4 stacked grayscale frames (4, 84, 84)
- **CNN**: 3 conv layers + FC layer (Nature DQN)
- **Output**: 512-dimensional feature vector
- **Shared**: Same CNN used for all 18 Q-learners

### Training Loop

```python
for step in range(total_timesteps):
    # Update epsilon (linear decay)
    epsilon = epsilon_schedule(step)

    # Select action using ε-greedy
    action = agent.select_action(obs, epsilon)

    # Take step
    next_obs, reward, done, info = env.step(action)

    # Update Q-learner for this action
    agent.update(obs, action, reward, next_obs, done)

    # Evaluation, checkpointing, logging
```

## Latency Simulation

When `--mode sim_lat` is used:
- LatencyModel wrapper simulates hardware delay
- Applied BEFORE frame stacking
- Models delay between action selection and execution
- Helps train robust policies for physical deployment

## Differences from PPO/R2D2

| Feature | PPO | R2D2 | SwiftTD |
|---------|-----|------|---------|
| Algorithm | Policy gradient | Off-policy Q-learning | On-policy Q-learning |
| Replay | None | Large buffer | None (online) |
| Architecture | Actor-Critic | LSTM + Dueling DQN | 18 linear learners |
| Parallelism | 4 envs | 8 actors | 1 env |
| Step size | Fixed | Fixed | Adaptive |
| Eligibility traces | No | No | Yes (λ=0.95) |

## Example Training Session

```bash
# Quick test (500K steps)
python algorithms/swifttd/train_agent_swifttd_sim.py \
  --env ALE/Pong-v5 \
  --timesteps 500000 \
  --mode sim \
  --device cuda \
  --wandb

# Full training with latency (1M steps)
python algorithms/swifttd/train_agent_swifttd_sim.py \
  --env ALE/MsPacman-v5 \
  --timesteps 1000000 \
  --mode sim_lat \
  --device cuda \
  --wandb \
  --wandb-project physical-atari
```

## Monitoring

- **Console**: Episode rewards, lengths, FPS
- **WandB**: Real-time metrics, evaluation results
- **Monitor logs**: CSV files in `logs/monitor/`
- **Checkpoints**: Saved every 100K steps

## Future Improvements

1. **CNN Training**: Currently CNN is initialized but not updated. Could implement:
   - Store raw observations for backprop through CNN
   - Update CNN using TD errors from Q-learners

2. **Target Network**: Add target network for more stable Q-learning

3. **Sparse Features**: Use `SwiftTDBinaryFeatures` with feature hashing for efficiency

4. **Parallel Actors**: Like R2D2 but with shared SwiftTD learners

## References

- SwiftTD paper: Javed, Sharifnassab, and Sutton (2024)
- SwiftTD repo: https://github.com/khurramjaved96/swifttd
- Nature DQN: Mnih et al. (2015)
