# R2D2 (Recurrent Experience Replay in Distributed Reinforcement Learning)

## Algorithm

**Type:** Off-policy Q-learning with recurrent networks

**Key Features:**
- Distributed training with multiple actors and centralized learner
- LSTM network to handle partial observability and temporal dependencies

**Why R2D2:** Combines the sample efficiency of DQN with recurrent networks to handle Atari's frame-stacking problem more elegantly. The distributed architecture allows for diverse exploration strategies across actors.

## Training in Simulation

```bash
python algorithms/r2d2/train_sim.py \
  --env ALE/MsPacman-v5 \
  --timesteps 1000000 \
  --num-actors 4 \
  --device cuda
```

With latency simulation:
```bash
python algorithms/r2d2/train_sim.py \
  --env ALE/MsPacman-v5 \
  --mode sim_lat \
  --latency-model-dir ./utils/latency_wrap
```

## Physical Hardware

**Note:** R2D2 was not trained on the physical harness because it's primary advantage is distributed training across environments with different exploration parameters. Our physical setup only had one robot, which makes R2D2 less optimal than RainbowDQN. 

For evaluation only (using pre-trained model):
```bash
python harness_physical.py \
  --agent_type=agent_r2d2 \
  --load_file=outputs/r2d2/models/final_model.pth \
  --total_frames=10000
```

## Architecture

CNN (Nature DQN) → LSTM → Dueling DQN Head (Value + Advantage streams)
