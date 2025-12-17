#!/usr/bin/env python3
"""
Test ReplayBuffer components in isolation
"""

import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from r2d2 import config
from r2d2.replay_buffer import Block, ReplayBuffer


def create_dummy_block(block_length=120, learning_steps=80, action_dim=18):
    """Create a dummy block for testing"""
    num_sequences = np.ceil(block_length / learning_steps).astype(int)

    # Create dummy data
    obs = np.random.randint(0, 255, (block_length + 1, 1, 84, 84), dtype=np.uint8)
    last_action = np.random.randint(0, 2, (block_length + 1, action_dim), dtype=bool)
    last_reward = np.random.randn(block_length + 1).astype(np.float32)
    actions = np.random.randint(0, action_dim, block_length, dtype=np.uint8)
    n_step_reward = np.random.randn(block_length).astype(np.float32)
    gamma = np.random.uniform(0.9, 0.999, block_length).astype(np.float32)
    hiddens = np.random.randn(num_sequences, 2, 512).astype(np.float32)

    burn_in_steps = np.array([min(i * learning_steps, 40) for i in range(num_sequences)], dtype=np.uint8)
    learning_steps_arr = np.array(
        [min(learning_steps, block_length - i * learning_steps) for i in range(num_sequences)], dtype=np.uint8
    )
    forward_steps = np.array(
        [min(5, block_length + 1 - np.sum(learning_steps_arr[: i + 1])) for i in range(num_sequences)], dtype=np.uint8
    )

    block = Block(
        obs=obs,
        last_action=last_action,
        last_reward=last_reward,
        action=actions,
        n_step_reward=n_step_reward,
        gamma=gamma,
        hidden=hiddens,
        num_sequences=num_sequences,
        burn_in_steps=burn_in_steps,
        learning_steps=learning_steps_arr,
        forward_steps=forward_steps,
    )

    priorities = np.random.uniform(0.1, 1.0, np.ceil(120 / 80).astype(int)).astype(np.float32)
    episode_reward = None

    return block, priorities, episode_reward


def test_priority_tree():
    """Test 1: PriorityTree in isolation"""
    print("=" * 60)
    print("TEST 1: PriorityTree")
    print("=" * 60)

    from r2d2.priority_tree import PriorityTree

    capacity = 1000
    tree = PriorityTree(capacity, alpha=0.6, beta=0.4)
    print(f"✓ PriorityTree created with capacity={capacity}")
    print(f"  tree.size: {tree.size}")
    print(f"  tree.capacity: {tree.capacity}")

    # Add some priorities
    idxes = np.array([0, 1, 2, 3, 4])
    priorities = np.array([0.000005, 1000.0, 0.000000003, 0.8, 0.6])

    tree.update(idxes, priorities)
    print("✓ Updated 5 priorities")
    print(f"  tree.size: {tree.size}")

    # Sample
    try:
        sampled_idxes, is_weights = tree.sample(batch_size=20)
        print("✓ Sampled batch_size=5")
        print(f"  sampled_idxes: {sampled_idxes}")
        print(f"  is_weights shape: {is_weights.shape}")
    except Exception as e:
        print(f"✗ Sampling failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False

    print()
    return True


def test_replay_buffer_basic():
    """Test 2: ReplayBuffer creation and basic operations"""
    print("=" * 60)
    print("TEST 2: ReplayBuffer Basic Operations")
    print("=" * 60)

    # Create queues
    sample_queue_list = [mp.Queue() for _ in range(2)]
    batch_queue = mp.Queue(8)
    priority_queue = mp.Queue(8)

    buffer = ReplayBuffer(
        sample_queue_list=sample_queue_list,
        batch_queue=batch_queue,
        priority_queue=priority_queue,
        buffer_capacity=10000,
        batch_size=32,
        stats_queue=[],
    )

    print("✓ ReplayBuffer created")
    print(f"  buffer_capacity: {buffer.buffer_capacity}")
    print(f"  num_blocks: {buffer.num_blocks}")
    print(f"  batch_size: {buffer.batch_size}")
    print(f"  size: {buffer.size}")

    # Add blocks manually
    print("\n  Adding blocks...")
    num_blocks_to_add = 5

    for i in range(num_blocks_to_add):
        block, priorities, episode_reward = create_dummy_block()
        buffer.add(block, priorities, episode_reward)
        print(f"    Block {i + 1} added, buffer.size={buffer.size}")

    print(f"✓ Added {num_blocks_to_add} blocks")
    print(f"  Final buffer.size: {buffer.size}")

    # Try to sample
    print("\n  Testing sample_batch()...")
    try:
        data = buffer.sample_batch()
        print("✓ Sampled batch successfully")
        print(f"  Batch data elements: {len(data)}")
        if len(data) > 0:
            print(f"  obs shape: {data[0].shape}")
            print(f"  action shape: {data[5].shape}")
    except Exception as e:
        print(f"✗ Sampling failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False

    print()
    return True


def test_replay_buffer_with_queues():
    """Test 3: ReplayBuffer with queue-based communication"""
    print("=" * 60)
    print("TEST 3: ReplayBuffer with Queue Communication")
    print("=" * 60)

    # Create queues
    num_actors = 2
    sample_queue_list = [mp.Queue() for _ in range(num_actors)]
    batch_queue = mp.Queue(8)
    priority_queue = mp.Queue(8)

    buffer = ReplayBuffer(
        sample_queue_list=sample_queue_list,
        batch_queue=batch_queue,
        priority_queue=priority_queue,
        buffer_capacity=10000,
        batch_size=32,
        stats_queue=[],
    )

    print(f"✓ ReplayBuffer created with {num_actors} actor queues")

    # Simulate actors adding data
    print("\n  Simulating actors adding blocks via queues...")
    for i, sample_queue in enumerate(sample_queue_list):
        for j in range(3):
            block, priorities, episode_reward = create_dummy_block()
            sample_queue.put([block, priorities, episode_reward])
            print(f"    Actor {i} added block {j + 1} to queue")

    # Process the queues (simulate add_data thread)
    print("\n  Processing queues (simulating add_data thread)...")
    blocks_added = 0
    for sample_queue in sample_queue_list:
        while not sample_queue.empty():
            data = sample_queue.get_nowait()
            buffer.add(*data)
            blocks_added += 1

    print(f"✓ Processed {blocks_added} blocks from queues")
    print(f"  buffer.size: {buffer.size}")

    # Test sampling
    print("\n  Testing batch sampling...")
    try:
        data = buffer.sample_batch()
        print("✓ Batch sampled successfully")
    except Exception as e:
        print(f"✗ Sampling failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test priority updates
    print("\n  Testing priority updates via queue...")
    idxes = np.array([0, 1, 2, 3, 4])
    td_errors = np.random.uniform(0.1, 1.0, 5).astype(np.float32)
    old_ptr = buffer.block_ptr
    loss = 0.5

    priority_queue.put((idxes, td_errors, old_ptr, loss))

    # Process priority update
    if not priority_queue.empty():
        data = priority_queue.get_nowait()
        buffer.update_priorities(*data)
        print("✓ Priority update processed")
        print(f"  training_steps: {buffer.training_steps}")

    print()
    return True


def test_edge_cases():
    """Test 4: Edge cases"""
    print("=" * 60)
    print("TEST 4: Edge Cases")
    print("=" * 60)

    sample_queue_list = [mp.Queue() for _ in range(1)]
    batch_queue = mp.Queue(8)
    priority_queue = mp.Queue(8)

    buffer = ReplayBuffer(
        sample_queue_list=sample_queue_list,
        batch_queue=batch_queue,
        priority_queue=priority_queue,
        buffer_capacity=10000,
        batch_size=32,
        stats_queue=[],
    )

    print("✓ Buffer created (empty)")

    # Test 1: Sample when empty
    print("\n  Test: Sampling when buffer is empty...")
    try:
        # This should fail gracefully or be prevented
        if buffer.size < config.learning_starts:
            print(f"  ✓ Buffer correctly reports size ({buffer.size}) < learning_starts ({config.learning_starts})")
        else:
            data = buffer.sample_batch()
            print("  ✗ Sampled from empty buffer (this might be wrong)")
    except Exception as e:
        print(f"  Expected error: {type(e).__name__}: {str(e)[:60]}...")

    # Test 2: Add minimum data and sample
    print("\n  Test: Add minimum data then sample...")

    # Add enough blocks to exceed learning_starts
    blocks_needed = int(np.ceil(config.learning_starts / 80)) + 5
    print(f"  Adding {blocks_needed} blocks...")

    for i in range(blocks_needed):
        block, priorities, episode_reward = create_dummy_block()
        buffer.add(block, priorities, episode_reward)

    print(f"  Buffer size: {buffer.size}")

    try:
        data = buffer.sample_batch()
        print("  ✓ Successfully sampled after adding sufficient data")
    except Exception as e:
        print(f"  ✗ Failed to sample: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    # Test 3: Circular buffer wrap-around
    print("\n  Test: Circular buffer wrap-around...")
    initial_ptr = buffer.block_ptr

    # Add many blocks to force wrap-around
    for i in range(buffer.num_blocks + 5):
        block, priorities, episode_reward = create_dummy_block()
        buffer.add(block, priorities, episode_reward)

    print(f"  Initial block_ptr: {initial_ptr}")
    print(f"  Final block_ptr: {buffer.block_ptr}")
    print("  ✓ Buffer pointer wrapped around" if buffer.block_ptr < initial_ptr else "  Buffer hasn't wrapped yet")

    print()
    return True


def test_priority_tree_sample_robustness():
    """Test 5: PriorityTree sample() robustness with edge cases"""
    print("=" * 60)
    print("TEST 5: PriorityTree sample() Edge Cases")
    print("=" * 60)

    from r2d2.priority_tree import PriorityTree

    capacity = 1000
    tree = PriorityTree(capacity, alpha=0.6, beta=0.4)

    # Add some priorities
    idxes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    priorities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tree.update(idxes, priorities)

    print("✓ Tree initialized with 10 priorities")

    # Test various batch sizes that might trigger the bug
    test_batch_sizes = [1, 2, 3, 7, 8, 16, 32, 64, 63, 65, 100]

    all_passed = True
    for batch_size in test_batch_sizes:
        try:
            sampled_idxes, is_weights = tree.sample(batch_size)

            # Verify shapes
            if len(sampled_idxes) != batch_size:
                print(f"  ✗ FAIL: batch_size={batch_size}, got {len(sampled_idxes)} idxes (expected {batch_size})")
                all_passed = False
            elif len(is_weights) != batch_size:
                print(f"  ✗ FAIL: batch_size={batch_size}, got {len(is_weights)} weights (expected {batch_size})")
                all_passed = False
            else:
                print(f"  ✓ batch_size={batch_size} OK")

        except ValueError as e:
            print(f"  ✗ FAIL: batch_size={batch_size} raised ValueError: {e}")
            all_passed = False
        except Exception as e:
            print(f"  ✗ FAIL: batch_size={batch_size} raised {type(e).__name__}: {e}")
            all_passed = False

    # Test with different priority distributions that might cause floating point issues
    print("\n  Testing with extreme priority values...")

    extreme_test_cases = [
        ("very small", np.array([1e-10, 1e-9, 1e-8, 1e-7, 1e-6])),
        ("very large", np.array([1e6, 1e7, 1e8, 1e9, 1e10])),
        ("mixed scale", np.array([1e-10, 1.0, 1e10, 1e-5, 1e5])),
        ("all same", np.ones(10)),
        ("alternating", np.array([0.1, 1000.0] * 5)),
    ]

    for test_name, priorities in extreme_test_cases:
        tree2 = PriorityTree(capacity, alpha=0.6, beta=0.4)
        idxes = np.arange(len(priorities))
        tree2.update(idxes, priorities)

        try:
            sampled_idxes, is_weights = tree2.sample(batch_size=8)
            if len(sampled_idxes) != 8 or len(is_weights) != 8:
                print(f"  ✗ FAIL: {test_name} priorities - wrong shapes")
                all_passed = False
            else:
                print(f"  ✓ {test_name} priorities OK")
        except Exception as e:
            print(f"  ✗ FAIL: {test_name} priorities raised {type(e).__name__}: {e}")
            all_passed = False

    print()
    return all_passed


def test_replay_buffer_sample_robustness():
    """Test 6: ReplayBuffer sample_batch() with resampling logic"""
    print("=" * 60)
    print("TEST 6: ReplayBuffer sample_batch() Robustness")
    print("=" * 60)

    sample_queue_list = [mp.Queue() for _ in range(1)]
    batch_queue = mp.Queue(8)
    priority_queue = mp.Queue(8)
    stats_queue = mp.Queue(8)

    buffer = ReplayBuffer(
        sample_queue_list=sample_queue_list,
        batch_queue=batch_queue,
        priority_queue=priority_queue,
        stats_queue=stats_queue,
        buffer_capacity=10000,
        batch_size=8,
    )

    # Add blocks with varying num_sequences
    print("  Adding blocks with different num_sequences...")

    # Add some full blocks
    for i in range(10):
        block, priorities, episode_reward = create_dummy_block(block_length=120, learning_steps=80)
        buffer.add(block, priorities, episode_reward)

    # Add some partial blocks (shorter sequences)
    for i in range(5):
        block, priorities, episode_reward = create_dummy_block(block_length=60, learning_steps=80)
        buffer.add(block, priorities, episode_reward)

    print("✓ Added 15 blocks (10 full, 5 partial)")
    print(f"  buffer.size: {buffer.size}")

    # Test multiple samples to catch edge cases
    print(f"\n  Testing 100 samples with batch_size={buffer.batch_size}...")

    all_passed = True
    for i in range(100):
        try:
            data = buffer.sample_batch()

            # Verify batch has correct size
            batch_obs = data[0]
            if batch_obs.shape[0] != buffer.batch_size:
                print(f"  ✗ FAIL: Sample {i} has wrong batch size: {batch_obs.shape[0]} != {buffer.batch_size}")
                all_passed = False
                break

        except Exception as e:
            print(f"  ✗ FAIL: Sample {i} raised {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False
            break

    if all_passed:
        print("  ✓ All 100 samples completed successfully")

    # Test with different batch sizes
    print("\n  Testing various batch sizes...")

    for batch_size in [1, 2, 7, 8, 16, 32]:
        buffer.batch_size = batch_size
        try:
            data = buffer.sample_batch()
            batch_obs = data[0]
            if batch_obs.shape[0] != batch_size:
                print(f"  ✗ FAIL: batch_size={batch_size} returned {batch_obs.shape[0]} samples")
                all_passed = False
            else:
                print(f"  ✓ batch_size={batch_size} OK")
        except Exception as e:
            print(f"  ✗ FAIL: batch_size={batch_size} raised {type(e).__name__}: {e}")
            all_passed = False

    print()
    return all_passed


def test_buffer_capacity():
    """Test 7: Buffer capacity is respected"""
    print("=" * 60)
    print("TEST 7: Buffer Capacity Limit")
    print("=" * 60)

    buffer_capacity = 10000
    sample_queue_list = [mp.Queue() for _ in range(1)]
    batch_queue = mp.Queue(8)
    priority_queue = mp.Queue(8)

    buffer = ReplayBuffer(
        sample_queue_list=sample_queue_list,
        batch_queue=batch_queue,
        priority_queue=priority_queue,
        buffer_capacity=buffer_capacity,
        batch_size=32,
        stats_queue=[],
    )

    print(f"✓ Buffer created with capacity={buffer_capacity}")
    print(f"  num_blocks: {buffer.num_blocks}")
    print(f"  block_len: {buffer.block_len}")
    print(f"  seq_per_block: {buffer.seq_per_block}")
    print(f"  num_sequences: {buffer.num_sequences}")

    # Calculate actual capacity
    actual_capacity = buffer.num_blocks * buffer.block_len
    print(f"  Actual capacity (num_blocks * block_len): {actual_capacity}")

    if actual_capacity != buffer_capacity:
        print(f"  ⚠ WARNING: Actual capacity {actual_capacity} != requested capacity {buffer_capacity}")
        print(f"  Lost capacity: {buffer_capacity - actual_capacity} steps")

    # Test: Fill buffer beyond capacity
    print("\n  Test: Adding blocks to exceed capacity...")
    blocks_to_add = buffer.num_blocks + 20  # Add more than capacity

    for i in range(blocks_to_add):
        block, priorities, episode_reward = create_dummy_block()
        buffer.add(block, priorities, episode_reward)

        if (i + 1) % 20 == 0:
            print(f"    Added {i + 1} blocks, buffer.size={buffer.size}, block_ptr={buffer.block_ptr}")

    print(f"\n✓ Added {blocks_to_add} blocks (more than num_blocks={buffer.num_blocks})")
    print(f"  Final buffer.size: {buffer.size}")
    print(f"  Final block_ptr: {buffer.block_ptr}")

    # Check if size exceeds capacity
    if buffer.size > buffer_capacity:
        print(f"  ✗ FAIL: buffer.size ({buffer.size}) > buffer_capacity ({buffer_capacity})")
        print(f"  Buffer exceeded capacity by {buffer.size - buffer_capacity} steps!")
        return False
    elif buffer.size > actual_capacity:
        print(f"  ✗ FAIL: buffer.size ({buffer.size}) > actual_capacity ({actual_capacity})")
        print(f"  Buffer exceeded actual capacity by {buffer.size - actual_capacity} steps!")
        return False
    else:
        print(f"  ✓ Buffer size ({buffer.size}) <= capacity ({buffer_capacity})")

    # Test circular wrap
    print("\n  Test: Verify circular buffer wrapped correctly...")
    expected_ptr = blocks_to_add % buffer.num_blocks
    if buffer.block_ptr == expected_ptr:
        print(f"  ✓ block_ptr wrapped correctly: {buffer.block_ptr} == {expected_ptr}")
    else:
        print(f"  ✗ block_ptr incorrect: {buffer.block_ptr} != {expected_ptr}")
        return False

    print()
    return True


def main():
    print("\n" + "=" * 60)
    print("REPLAY BUFFER COMPONENT TESTS")
    print("=" * 60 + "\n")

    all_passed = True

    # Test 1: Priority Tree
    if not test_priority_tree():
        all_passed = False

    # Test 2: Basic operations
    if not test_replay_buffer_basic():
        all_passed = False

    # Test 3: Queue communication
    if not test_replay_buffer_with_queues():
        all_passed = False

    # Test 4: Edge cases
    if not test_edge_cases():
        all_passed = False

    # Test 5: Priority Tree sample() robustness
    if not test_priority_tree_sample_robustness():
        all_passed = False

    # Test 6: ReplayBuffer sample_batch() robustness
    if not test_replay_buffer_sample_robustness():
        all_passed = False

    # Test 7: Buffer capacity
    if not test_buffer_capacity():
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
