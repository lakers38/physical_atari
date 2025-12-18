# DQN (Deep Q-Network)

## Algorithm

**Type:** Off-policy Q-learning

## Training in Simulation

```bash
python utils/sim_latency_vec.py \
  --agent agent_dqn \
  --env ALE/MsPacman-v5 \
  --total_timesteps 1000000 \
  --device cuda
```

## Physical Hardware

**Note:** DQN was not extensively trained on the physical harness due to low sample efficiency.

For evaluation only:
```bash
python harness_physical.py \
  --agent_type=agent_dqn \
  --load_file=outputs/dqn/models/dqn_final.pth \
  --total_frames=10000
```

## Architecture

CNN (3 conv layers) → FC(512) → FC(num_actions) outputting Q-values
