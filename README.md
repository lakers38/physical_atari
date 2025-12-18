# Robo Atari

Robo Atari is a fork of [physical_atari](https://github.com/Keen-Technologies/physical_atari/). 

Most of our proposed modifications are in `algorithms/`. We have made some small modifications to the physical harness code to ensure compatibility with our agents.

```
algorithms/                          # RL agents + sim training scripts
├── dqn/                             # DQN agent (sim only)
├── rainbow_dqn/                     # Rainbow DQN agent (physical + sim)
├── r2d2/                            # R2D2 agent (sim only)
├── ppo/                             # PPO agent + (physical + sim)
├── sac/                             # SAC agent (physical + sim)
├── delay_target/                    # Delay-target agent (physical + sim)
└── swift_sarsa/                     # Swift SARSA agent (sim only)

utils/                               # Shared utilities (e.g., latency model wrapper)
```

Entry points are typically either:
- `python -m algorithms.<algo>.train_sim ...` for simulation training (where available)
- `harness_physical.py --agent_type=...` for physical runs (loads `algorithms/<algo>/agent_*.py`)


Below is the original, unmodified repository README

# Physical Atari

**Physical Atari** is a platform for evaluating reinforcement learning (RL) algorithms.

Many RL algorithms are evaluated primarily with simulations, driven by the ease of running experiments that can be replicated by other researchers. Although this is a successful research approach, it is widely recognized in science and engineering that a simulation can only capture part of the complexity of the real system.  With so much RL research using simulators, there is the danger that improvements observed in a simulator does not translate to their real-world counterparts.  This is especially true for RL research developed with the Atari Learning Environment (ALE), as the corresponding physical systems were not easily accessible.  Some of algorithms developed with the ALE have been deployed to other real-world settings, but many RL algorithms have only been tested in the ALE.

The **Physical Atari** platform provides a software and hardware interface between reinforcement learning agents and a modern version of a physical Atari.  This interface enables the evaluation of RL algorithms developed for the ALE with a real-world instantiation.  The physical platform exposes several timing concerns that are not present in the ALE.  The physical Atari system operates in real-time and is not turn based (it does not wait for an agent action).  Physical systems have non-negligible latency and physical systems have unmodelled dynamics (sensor and actuation noise).  Unlike traditional environments that use pixel-perfect emulators (e.g., ALE), this setup integrates a physical **Atari 2600+** console, a camera-based observation pipeline, and a real-time control system using physical actuators.


This platform provides three contributions for the RL research community.
- A physical platform for evaluating RL algorithms that have been primarily developed with the ALE
- An implementation of a game-independent RL algorithm on this platform that learns under real-time constraints to reliably surpass standard benchmark performance within five hours (1 million frames) on multiple Atari games.
- The platform provides insight into the limitations of our simulators.  Discrepancies in performance between the simulator and reality suggests changes to our simulated environments and the metrics for evaluating RL  algorithms.

---

## Overview

The system consists of three main components:

- **Environment**: A modern [Atari 2600+](https://www.amazon.com/Atari-2600/dp/B0CG7LMFKY) console, outputting real 4:3 video over HDMI.  The console is pin-compatible with original Atari game cartridges and joysticks.
- **Agent**: The learning algorithms and supporting control logic run on a gaming laptop or workstation.
- **Interface**:
  - **Observation**: Video is captured by a USB camera at 60 frames per second
  - **Action**: Agent selected actions are sent to the console by one of the following.
    - A **mechanical actuator** (RoboTroller) that physically moves the CX40+ joystick
    - A **digital I/O module** that bypasses the joystick and sends signals directly to the controller port via the DB9 cable

This setup enables the study of RL algorithms in the physical world, in the presence of many real-world concerns (domain shifts, latency, and noise).

---

## System Setup

A complete hardware/software setup guide is available here:
[**System Setup**](docs/setup.md)

---

## Components

| Component           | Description |
|---------------------|-------------|
| **Console**         | [Atari 2600+](https://www.amazon.com/Atari-2600/dp/B0CG7LMFKY) with CX40+ joystick |
| **Monitor**         | Any gaming monitor with native 16:9 resolution and 60Hz refresh rate |
| **Camera**          | [Razer Kiyo Pro (1080p)](https://www.amazon.com/dp/B08T1MWX6J) — supports 60FPS uncompressed |
| **Control (Option 1)** | Mechanical joystick control via servo-based actuator [RoboTroller](https://robotroller.keenagi.com) |
| **Control (Option 2)** | Digital I/O module — e.g., [MCC USB-1024LS](https://microdaq.com/usb-1024ls-24-bit-digital-input-output-i-o-module.php) |

> See [setup.md](docs/setup.md) for placement, lighting, USB bandwidth, tag positioning, and system setup.

---

## Design of the RL software interface

In the textbook picture, an RL agent interacts with its environment by exchanging signals for reward, observation, and action.  In the episodic domains there is also a signal for the end of episode. In most Atari/Gym interfaces this is extended to support signals for end-of-life, sequence truncation, and supporting a minimal action set in a game.  These additional signals are useful for accelerating early performance in Atari games, and ease of experimentation is a significant factor for the physical Atari platform. We have chosen to expose the additional signals to our learning agents.

We want the RL algorithms to have an interface that supports real-time interaction with the real world.  We have changed the agent/environment calling conventions a common choice (where the agent directly calls the environment) to an interface where the experiment infrastructure sends the signals to the agent and to the environment.

The primary agent/environment interface operates at 60fps.  Observations are received from the video camera (with some internal buffers) and sent to the agent.  Actions selected by the agent are converted into commands for the robotroller to move the joystick (which also has latencies).  More effort is required to extract signals for rewards, lives, and the end of episode from the observed video, and these are described below.

## Games

We restricted our attention to console games that only require a **fire button press** to start gameplay.  Many Atari games require toggling the physical **reset switch** on the Atari console to restart the game, and so were not used.


The following games are known to work with this setup and cover a range of visual styles and control demands:

_Recommended_

- **Ms. Pac-Man**
- **Centipede**
- **Up 'n Down**
- **Krull**

_Less Tested_
- **Q*Bert**
- **Battle Zone**
- **Atlantis**
- **Defender**

## Detecting Score, Lives, and the End of Game


Detection of the game score, lives, and the end of game requires custom logic for the Physical Atari.  In the ALE, bits from the internal game state are used to compute these signals in a custom manner for each game.  For the physical Atari, these signals must be computed from the video screen.  Multiple steps are required for extracting these signals, and it is the most brittle part of the physical atari platform.

For the first step, camera images are rectified by identifying the corners of the Atari game screen in the camera image and applying a standard linear transformation.  We have tried multiple approaches here. One reliable approach is to manually identify the four screen corners, as these do not vary by Atari game.  This approach breaks down if the system is subject to jostling or vibration.  Two other approaches we have examined are the use of April tags, and whole screen recognition.

For the second step we manually identify boxes around the score and lives for each game.

For the third step, the score is read from the video.  The digits used in each atari game differs substantially, and they are also distinct from digits in standard online datasets such as MNIST.  We again tried multiple approaches.  The most reliable was to collect a dataset for each game of images and numerical scores, and then train a supervised learning classifier for each.  For training the classifier, the captured images are augmented with several transformations, to account for small variations in calibration, lighting, and geometry.  A similar process is used to detect the number of lives in the game.

Some additional custom logic is used on top of the neural net classifiers to extract reliable signals.  Score differences are validated for consistency with the game scores observed in the simulator, so scores can't change by an unrealizable amount between frames.  Additional logic is present to recover from transient errors, and to detect a plausible end of the game.  When the game is presumed to be over, a FIRE action is sent to restart, and the end of game is sent to the learning agent.

// A per-game CRNN model is used to extract the score directly from screen pixels. These models are trained on ALE-rendered frames, using known score regions for supervision.

// For some games, a more targeted model may be used instead.

### Additional Considerations for ROM and Hardware Variability

Several additional ROM and hardware factors can impact signal extraction accuracy. Different ROM revisions may introduce subtle or significant variations, including difficulty adjustments, bug fixes, or changes in visual indicators such as the displayed number of lives. Region-specific differences between NTSC and PAL cartridges could also potentially result in rendering variations, timing discrepancies, or different memory layouts, although the extent to which these factors necessitate separate processing pipelines remains to be fully tested.

Additionally, prototype versions of games may differ notably from retail releases, leading to mismatches if the ALE emulator is based on a retail ROM and the physical cartridge represents an earlier or alternative version. Furthermore, variations in bankswitching techniques, used by certain cartridges to extend memory addressing, could also explain inconsistencies between the simulated ALE environment and physical hardware, particularly if RAM addresses differ for critical indicators like lives or score. These factors collectively underscore the importance of careful validation and customization when transitioning from simulation to the physical Atari platform.

---

## Research Challenges

This project focuses on bridging the **reality gap** in empirical reinforcement learning research.

**Differences between the emulator and reality include:**
- No turn-taking in a real-time environment
- Visual statistics (emulator vs real camera)
- Latency in video capture and actuator response
- Imperfect lighting, reflections, and image noise
- Score detection errors under variable lighting and resolution

Both trained policies and reinforcement learning algorithms can degrade significantly when exposed to these real-world conditions, even if they perform well in simulation.

---

## Launching

To run the physical setup, build the docker environment with:

```
./docker_build.sh
```

Run the docker environment with:

```
./docker_run.sh
```

Launch the physical harness:

```
python physical_harness.py
```

An example run for the Ms Pacman game, for a custom configuration

```
python3 harness_physical.py \
 --detection_config=configs/screen_detection/fixed.json \
 --game_config=configs/games/ms_pacman.json \
 --agent_type=agent_delay_target \
 --reduce_action_set=2 --gpu=0 \
 --joystick_config=configs/controllers/robotroller.json \
 --total_frames=1_000_000
```


---

## System Performance and Profiling

Running the physical setup reliably requires that the system meets strict performance requirements — especially with regard to CPU and GPU power settings, thermal limits, and scheduling behavior.

Modern systems often default to power-saving configurations that can cause unexpected latency, frame delays, or jitter. These issues are especially problematic in real-time or hardware-in-the-loop setups.

- See [Performance Setup](./docs/setup.md#system-performance-validation) for validating system configuration and fixing system-level performance issues.
- See [Profiling Guide](./docs/profiling.md) for details on collecting and analyzing performance data using NVIDIA Nsight Systems and NVTX annotations.

---

## License

This project is licensed under the Apache 2.0 License.

Unless otherwise noted, this license applies to all source code and pre-trained model files included in the repository.
