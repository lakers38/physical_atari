# Delay Target (Latency-Aware Q-Learning)

## Algorithm

**Type:** Off-policy Q-learning adapted for hardware latency

## Training in Simulation

Requires latency simulation:
```bash
python algorithms/delay_target/sim_latency.py \
  --agent agent_delay_target \
  --env ALE/MsPacman-v5 \
  --mode sim_lat \
  --latency-model-dir ./utils/latency_wrap \
  --total_timesteps 2000000
```

## Physical Hardware

This algorithm is specifically designed for physical hardware:
```bash
python harness_physical.py \
  --agent_type=agent_delay_target \
  --total_frames=500000
```

## Architecture

CNN → FC(512) → Q-values, with delayed target network accounting for action queue state
