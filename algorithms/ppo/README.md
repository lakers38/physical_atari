# PPO (Proximal Policy Optimization)

## Algorithm

**Type:** On-policy policy gradient

**Key Features:**
- Clipped surrogate objective prevents destructively large policy updates

**Why PPO:** Reliable and stable policy gradient method.

## Training in Simulation

```bash
python algorithms/ppo/train_sim.py \
  --env ALE/MsPacman-v5 \
  --total_timesteps 10000000 \
  --device cuda
```

With latency simulation:
```bash
python algorithms/ppo/train_sim.py \
  --env ALE/MsPacman-v5 \
  --mode sim_lat \
  --latency-model-dir ./utils/latency_wrap
```

## Physical Hardware

```bash
python harness_physical.py \
  --agent_type=agent_ppo \
  --load_file=outputs/ppo/models/ppo_final.pth \
  --game_config=configs/games/ms_pacman.json \
  --total_frames=500000
```

## Architecture

CNN (Nature DQN style) → FC(512) → Policy head (actor) + Value head (critic)
