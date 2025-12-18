# Delay Target (Latency-Aware Q-Learning)

## Algorithm

**Type:** Off-policy Q-learning adapted for hardware latency

**Key Features:**
- Explicitly models action-observation delay in Q-learning targets
- Maintains action queue matching physical hardware buffer
- Adjusted n-step returns accounting for delayed feedback
- Can use any base Q-learning algorithm (DQN, Rainbow, etc.)

**Why Delay Target:** Purpose-built for the physical Atari setup. Directly addresses the fundamental challenge of hardware latency (actions take ~250ms to execute) by incorporating delay into the Bellman equation.

## Training in Simulation

Requires latency simulation:
```bash
python algorithms/delay_target/train_sim.py \
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
  --load_file=outputs/delay_target/models/dt_final.pth \
  --game_config=configs/games/ms_pacman.json \
  --camera_config=configs/cameras/camera_elgato.json \
  --joystick_config=configs/controllers/robotroller.json \
  --detection_config=configs/screen_detection/april_tags.json \
  --total_frames=500000
```

## Architecture

CNN → FC(512) → Q-values, with delayed target network accounting for action queue state
