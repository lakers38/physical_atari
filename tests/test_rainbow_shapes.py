import os
import sys

import numpy as np
import torch

# Add project root to path for direct test runs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_rainbow import RainbowCore, RainbowNetwork


def test_rainbow_network_handles_128x128_stack16():
    net = RainbowNetwork(in_channels=16, num_actions=6, num_atoms=51, obs_height=128, obs_width=128)
    dummy = torch.zeros((2, 16, 128, 128), dtype=torch.float32)
    support = torch.linspace(-10, 10, 51)

    probs, log_probs = net(dummy)
    assert probs.shape == (2, 6, 51)
    assert log_probs.shape == (2, 6, 51)

    q_values = net.q_values(dummy, support)
    assert q_values.shape == (2, 6)


def test_rainbow_core_stacks_and_buffers_shape():
    core = RainbowCore(
        num_envs=2,
        seed=0,
        num_actions=6,
        total_frames=1_000,
        stack_size=16,
        obs_height=128,
        obs_width=128,
        buffer_size=32,  # keep memory footprint small for the test
        gpu=-1,
    )

    # raw observations from ALE are 210x160x3; they get resized internally
    obs = np.zeros((2, 210, 160, 3), dtype=np.uint8)
    core.reset(obs)
    actions = core.act(obs)

    assert actions.shape == (2,)
    assert core.state_stacks.shape == (2, 16, 128, 128)
    assert core.replay.states.shape[1:] == (16, 128, 128)
