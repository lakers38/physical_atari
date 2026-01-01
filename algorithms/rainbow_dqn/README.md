# Rainbow DQN (Combined DQN Improvements)

## Algorithm

**Type:** Off-policy Q-learning with multiple enhancements

**Key Features:**
- Double Q-learning
- Prioritized experience replay
- Dueling network architecture
- Distributional Learning
- Noisy Nets
- Mult-step Returns

**Why Rainbow:** Combines six orthogonal DQN improvements into a single agent.

## Training in Simulation

```bash
python algorithms/utils/sim_latency_vec.py \
  --agent agent_rainbow \
  --env ALE/MsPacman-v5 \
  --total_timesteps 10000000 \
  --device cuda
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
