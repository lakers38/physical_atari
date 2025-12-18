# Swift SARSA (Fast On-Policy TD Learning)

## Algorithm

**Type:** On-policy temporal difference learning (SARSA variant)

**Key Features:**
- Linear function approximation with tile coding
- On-policy learning (learns from actions actually taken)
- Uses PPO-trained CNN as fixed feature extractor

**Why Swift SARSA:** Linear learner on top of fixed featurization backbone (CNN). The linear learner enables on the fly updates and learning with tuning based on bootstrapped TD values.

## Training in Simulation

Train PPO backbone first:
```bash
python algorithms/ppo/train_sim.py --env ALE/MsPacman-v5 --total_timesteps 5000000
```

Then train Swift SARSA:
```bash
python algorithms/swift_sarsa/train_sim.py \
  --env ALE/MsPacman-v5 \
  --ppo_model=outputs/ppo/models/ppo_final.pth \
  --total_timesteps 1000000
```

## Physical Hardware

**Note:** Swift SARSA was not extensively trained on the physical harness because the PPO feature extractor did not perform well in simulation.

For evaluation:
```bash
python harness_physical.py \
  --agent_type=agent_ss \
  --load_file=outputs/swift_sarsa/models/ss_final.npz \
  --total_frames=10000
```

## Architecture

Pretrained PPO CNN (frozen) → Feature vector → C++ linear function learner
