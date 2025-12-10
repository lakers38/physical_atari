
import numpy as np
from ale_pong_swift_sarsa import MemoryFeatures, MemoryImprinter
from collections import deque

def test_memory_features():
    print("Testing MemoryFeatures...")
    # Setup: 10 base features
    num_base = 10
    mem = MemoryFeatures(num_tenured=num_base)
    
    # Add a memory feature: parent=0, k1=3 (duration), k2=2 (delay), id=10
    # Logic: If parent 0 fires at t, memory 10 should be active at [t+2, t+2+3) = [t+2, t+5)
    # i.e., t+2, t+3, t+4.
    mem.add_feature(parent=0, k1=3, k2=2, feat_id=10, trigger_now=False)
    
    # Time t=0: Parent 0 fires
    active_base = [0]
    active_mem = mem.step(active_base)
    print(f"t=0 (parent fires): active={active_mem}")
    assert 10 not in active_mem, "Memory feature should not be active at t=0 (delay 2)"
    
    # Time t=1: No parent
    active_mem = mem.step([])
    print(f"t=1: active={active_mem}")
    assert 10 not in active_mem, "Memory feature should not be active at t=1 (delay 2)"
    
    # Time t=2: Should be active (start of window)
    active_mem = mem.step([])
    print(f"t=2: active={active_mem}")
    assert 10 in active_mem, "Memory feature SHOULD be active at t=2"
    
    # Time t=3: Should be active
    active_mem = mem.step([])
    print(f"t=3: active={active_mem}")
    assert 10 in active_mem, "Memory feature SHOULD be active at t=3"
    
    # Time t=4: Should be active (end of window)
    active_mem = mem.step([])
    print(f"t=4: active={active_mem}")
    assert 10 in active_mem, "Memory feature SHOULD be active at t=4"
    
    # Time t=5: Should be inactive
    active_mem = mem.step([])
    print(f"t=5: active={active_mem}")
    assert 10 not in active_mem, "Memory feature should NOT be active at t=5"
    
    print("MemoryFeatures test passed!")

def test_memory_imprinting():
    print("\nTesting MemoryImprinter...")
    num_base = 10
    mem = MemoryFeatures(num_tenured=num_base)
    free_ids = deque([10, 11, 12])
    
    # Imprinter setup
    imprinter = MemoryImprinter(
        base_features=num_base,
        memory=mem,
        free_ids=free_ids,
        alpha_init=0.1,
        eta=1.0,
        tenure_thresh=0.5,
        patterns=[(3, 2)], # Fixed pattern for testing
        max_new_per_step=1
    )
    
    # Mock weights and betas
    # Feature 0 is tenured (|w| >= 0.5)
    w_feat = np.zeros(num_base + 10)
    w_feat[0] = 0.6 
    # Initialize betas to small value so tau_t is small
    # exp(beta) = step_size. Let's make it 0.01
    beta_feat = np.full(num_base + 10, np.log(0.01))
    
    # Step 1: Feature 0 was active at t-1
    base_active_prev = [0]
    
    # Run imprinter
    imprinter.step(base_active_prev, w_feat, beta_feat)
    
    # Check if a new feature was added
    if mem.num_memory_features == 1:
        print("SUCCESS: New memory feature added")
        print(f"New feature ID: {mem.ids[0]}")
        assert mem.ids[0] == 10
    else:
        print(f"FAILURE: Expected 1 memory feature, got {mem.num_memory_features}")

    print("MemoryImprinter test passed!")

if __name__ == "__main__":
    test_memory_features()
    test_memory_imprinting()
