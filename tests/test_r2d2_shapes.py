
import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.r2d2.model import Network, AgentState
from algorithms.r2d2 import config
from agent_r2d2 import Agent

def test_model_shapes():
    print("Testing Network shapes...")
    # Config should now be (16, 128, 128)
    print(f"Config obs_shape: {config.obs_shape}")
    assert config.obs_shape == (16, 128, 128), f"Expected (16, 128, 128), got {config.obs_shape}"

    action_dim = 6
    model = Network(action_dim, obs_shape=config.obs_shape)
    
    # Create dummy input: (batch=1, seq=1, channels=16, H=128, W=128)
    # AgentState expects (batch, channels, H, W) for single step
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    obs = torch.zeros((1, 16, 128, 128)).to(device)
    state = AgentState(obs, action_dim)
    
    # Forward pass
    q_values, hidden = model(state)
    
    print(f"Q-values shape: {q_values.shape}")
    assert q_values.shape == (action_dim,), f"Expected ({action_dim},), got {q_values.shape}"
    print("Network forward pass successful.")

def test_agent_wrapper():
    print("\nTesting Agent wrapper...")
    # Create agent
    agent = Agent(data_dir="/tmp", seed=0, num_actions=6, total_frames=1000)
    
    # Verify agent config
    print(f"Agent resize_to_128: {getattr(agent, 'resize_to_128', False)}")
    print(f"Agent n_frames: {getattr(agent, 'n_frames', 0)}")
    
    # Simulate a frame from harness (160, 210, 3)
    frame_rgb = np.zeros((210, 160, 3), dtype=np.uint8)
    
    # Pass 17 frames to ensure stack fills and rotates
    for i in range(17):
        action = agent.frame(frame_rgb, reward=0, end_of_episode=False)
        
    print(f"Agent frame processing successful. Last action: {action}")
    
    # Check internal stack
    print(f"Internal stack size: {len(agent.frames)}")
    assert len(agent.frames) == 16, f"Expected stack size 16, got {len(agent.frames)}"
    
    # Check stacked observation shape in agent_state
    if agent.agent_state:
        print(f"AgentState obs shape: {agent.agent_state.obs.shape}")
        # Expect (1, 16, 128, 128)
        assert agent.agent_state.obs.shape == (1, 16, 128, 128), f"Expected (1, 16, 128, 128), got {agent.agent_state.obs.shape}"

if __name__ == "__main__":
    try:
        test_model_shapes()
        test_agent_wrapper()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
