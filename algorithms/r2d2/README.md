# R2D2 Implementation for Physical Atari

This directory contains an R2D2-inspired algorithm implementation for training agents on Atari games.

## Overview

This implementation is inspired by **R2D2** (Recurrent Experience Replay in Distributed Reinforcement Learning) and uses:
- **LSTM networks** to handle temporal dependencies
- **Experience replay** for off-policy learning
- **Target network** for stable Q-learning
- **Epsilon-greedy exploration** with decay
- **Single-process training** for simplicity and cross-platform compatibility

**Note**: This is a simplified, single-process version. The original R2D2 paper uses distributed training with multiple actors, which can be implemented later if needed (see Option 3 in design docs).

## File Structure

```
r2d2/
├── __init__.py           # Package initialization
├── config.py             # Default hyperparameters (used for reference)
├── model.py              # Network architecture (CNN + LSTM + Dueling DQN)
├── priority_tree.py      # Sum tree (for future prioritized replay)
├── replay_buffer.py      # Replay buffer (distributed version - not used in single-process)
├── actor.py              # Actor class (distributed version - not used in single-process)
├── learner.py            # Learner class (distributed version - not used in single-process)
└── README.md             # This file
```

**Note**: `train_agent_r2d2_sim.py` contains a simplified single-process implementation with inline replay buffer and training logic.

## Usage

### Training in Simulation

Train R2D2 in Gymnasium with or without latency simulation:

```bash
# Pure simulation (no latency)
python train_agent_r2d2_sim.py \
  --env ALE/Breakout-v5 \
  --mode sim \
  --timesteps 1000000 \
  --device cpu

# Simulation with hardware latency
python train_agent_r2d2_sim.py \
  --env ALE/Breakout-v5 \
  --mode sim_lat \
  --timesteps 1000000 \
  --device cuda \
  --latency-model-dir ./latency_wrap
```

### Training on Physical Hardware

Transfer a trained model to the physical Robotroller:

```bash
python harness_physical.py \
  --agent_type=agent_r2d2 \
  --load_model=outputs_sim/sim_lat/ALE_Breakout-v5/run_xyz/models/final_model.pth \
  --total_frames=500000 \
  --game_config=configs/games/breakout.json
```

### Key Hyperparameters

**Training:**
- `--learning-rate`: Learning rate (default: 1e-4)
- `--gamma`: Discount factor (default: 0.997)
- `--batch-size`: Batch size for training (default: 32)
- `--buffer-capacity`: Replay buffer size (default: 100k)
- `--learning-starts`: Start training after N steps (default: 10k)
- `--train-freq`: Train every N steps (default: 4)
- `--target-update-freq`: Update target network (default: 2500)

**Exploration:**
- `--epsilon-start`: Starting epsilon (default: 0.4)
- `--epsilon-end`: Final epsilon (default: 0.01)
- `--epsilon-decay-steps`: Epsilon decay duration (default: 250k)

**Physical Hardware (via harness_physical.py):**
- `--r2d2_epsilon`: Fixed epsilon (default: 0.01)
- `--r2d2_hidden_dim`: LSTM hidden dimension (default: 512)
- `--r2d2_frame_skip`: Act every N frames (default: 4)

## Algorithm Details

### Network Architecture

1. **CNN Feature Extractor** (Nature DQN):
   - Conv(1→32, 8×8, stride=4) → ReLU
   - Conv(32→64, 4×4, stride=2) → ReLU
   - Conv(64→64, 3×3, stride=1) → ReLU
   - Flatten → Linear(3136→512) → ReLU

2. **LSTM**: Takes concatenated [features, last_action_one_hot, last_reward] as input

3. **Dueling DQN Head**: Separate advantage and value streams

### Training Process

1. **Experience Collection**: Agent interacts with environment using epsilon-greedy policy
2. **Replay Buffer**: Stores transitions (obs, action, reward, next_obs, done) with LSTM hidden states
3. **Training Loop**:
   - Sample random batch from replay buffer
   - Compute Q-values using online network
   - Compute target Q-values using target network
   - Update online network with gradient descent
   - Periodically update target network (every 2500 steps)
   - Gradient clipping (norm=40)

### Simplified Design

This single-process implementation:
- Uses simple replay buffer (`collections.deque`)
- No multiprocessing/threading (no pickling issues)
- Works on all platforms (macOS, Linux, Windows)
- LSTM maintains hidden state across episode but resets on done
- Target network for stable Q-learning

## Comparison with PPO

| Feature | R2D2 | PPO |
|---------|------|-----|
| Algorithm Type | Off-policy Q-learning | On-policy policy gradient |
| Memory | Recurrent (LSTM) | Feed-forward |
| Experience Replay | Yes (prioritized) | No |
| Parallelization | Distributed actors + centralized learner | Vectorized environments |
| Sample Efficiency | Higher (reuses experience) | Lower (on-policy) |
| Training Stability | Value rescaling + target network | Clipped surrogate objective |

## References

- [Recurrent Experience Replay in Distributed Reinforcement Learning (Kapturowski et al., 2019)](https://openreview.net/forum?id=r1lyTjAqYX)
- [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236) - DQN
- [Dueling Network Architectures for Deep Reinforcement Learning (Wang et al., 2016)](https://arxiv.org/abs/1511.06581)
