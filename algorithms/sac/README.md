# SAC (Soft Actor-Critic)

## Algorithm

**Type:** Off-policy actor-critic with maximum entropy

**Key Features:**
- Automatic entropy temperature tuning
- Off-policy learning with replay buffer for sample efficiency
- Twin Q-networks (clipped double Q-learning) reduce overestimation

## Training in Simulation

```bash
python algorithms/sac/train_sim.py \
  --env ALE/MsPacman-v5 \
  --total_timesteps 1000000 \
  --device cuda
```

With latency simulation:
```bash
python algorithms/sac/train_sim.py \
  --env ALE/MsPacman-v5 \
  --mode sim_lat \
  --latency-model-dir ./utils/latency_wrap
```

## Physical Hardware

```bash
python harness_physical.py \
  --agent_type=agent_sac \
  --load_file=outputs/sac/models/sac_final.pth \
  --game_config=configs/games/ms_pacman.json \
  --total_frames=500000
```

## Architecture

CNN feature extractor → Twin Q-networks + Policy network
