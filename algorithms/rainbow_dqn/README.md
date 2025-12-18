# Rainbow DQN (Combined DQN Improvements)

## Algorithm

**Type:** Off-policy Q-learning with multiple enhancements

**Key Features:**
- Double Q-learning
- Prioritized experience replay
- Dueling network architecture

**Why Rainbow:** Combines six orthogonal DQN improvements into a single agent.

## Training in Simulation

```bash
python algorithms/rainbow_dqn/train_sim.py \
  --env ALE/MsPacman-v5 \
  --total_timesteps 10000000 \
  --device cuda
```

With latency simulation:
```bash
python algorithms/rainbow_dqn/train_sim.py \
  --env ALE/MsPacman-v5 \
  --mode sim_lat \
  --latency-model-dir ./utils/latency_wrap
```

## Physical Hardware

```bash
python harness_physical.py \
  --agent_type=agent_rainbow \
  --load_file=outputs/rainbow/models/rainbow_final.pth \
  --game_config=configs/games/ms_pacman.json \
  --total_frames=500000
```

## Architecture

CNN → Dueling head → Distributional output
