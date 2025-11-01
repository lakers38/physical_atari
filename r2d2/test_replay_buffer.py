#!/usr/bin/env python3
"""Test cases for ReplayBuffer index arithmetic and circular buffer logic"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
from unittest.mock import Mock
from r2d2.replay_buffer import ReplayBuffer, Block
from r2d2 import config


class TestReplayBufferIndexArithmetic:
    """Test suite for index arithmetic in ReplayBuffer"""

    @pytest.fixture
    def mock_queues(self):
        """Create mock queues for ReplayBuffer initialization"""
        sample_queue_list = [Mock()]
        batch_queue = Mock()
        priority_queue = Mock()
        return sample_queue_list, batch_queue, priority_queue

    @pytest.fixture
    def buffer(self, mock_queues):
        """Create a ReplayBuffer with known configuration"""
        # Use small, predictable sizes for testing
        sample_queue_list, batch_queue, priority_queue = mock_queues

        # Override config values for testing
        buffer_capacity = 80  # Small capacity for easy testing
        sequence_len = 20
        batch_size = 4

        # block_length from config (assuming 40 for this test)
        original_block_length = config.block_length
        config.block_length = 40

        buf = ReplayBuffer(
            sample_queue_list,
            batch_queue,
            priority_queue,
            buffer_capacity=buffer_capacity,
            sequence_len=sequence_len,
            alpha=0.6,
            beta=0.4,
            batch_size=batch_size
        )

        yield buf

        # Restore original config
        config.block_length = original_block_length

    def test_buffer_dimensions(self, buffer):
        """Verify buffer calculates dimensions correctly"""
        # With capacity=80, sequence_len=20, block_len=40:
        # num_sequences = 80 / 20 = 4
        # num_blocks = 80 / 40 = 2
        # seq_per_block = 40 / 20 = 2

        assert buffer.buffer_capacity == 80
        assert buffer.sequence_len == 20
        assert buffer.num_sequences == 4
        assert buffer.block_len == 40
        assert buffer.num_blocks == 2
        assert buffer.seq_per_block == 2

    def test_add_block_index_calculation(self, buffer):
        """Verify add() calculates correct sequence indices for priority tree"""
        # Create a dummy block
        block = self._create_dummy_block(buffer, num_sequences=2)
        priorities = np.array([1.0, 2.0])

        # Mock the priority_tree.update to capture what indices are passed
        original_update = buffer.priority_tree.update
        captured_idxes = None

        def capture_update(idxes, prios):
            nonlocal captured_idxes
            captured_idxes = idxes.copy()
            original_update(idxes, prios)

        buffer.priority_tree.update = capture_update

        # Add block at position 0
        buffer.add(block, priorities, episode_reward=None)

        # block_ptr was 0, so idxes should be [0*2, 1*2) = [0, 1]
        assert np.array_equal(captured_idxes, np.array([0, 1]))

        # Add another block at position 1
        buffer.add(block, priorities, episode_reward=None)

        # block_ptr was 1, so idxes should be [1*2, 2*2) = [2, 3]
        assert np.array_equal(captured_idxes, np.array([2, 3]))

    def test_sample_block_sequence_conversion(self, buffer):
        """Verify sampling correctly converts sequence indices to block/sequence pairs"""
        # Add two blocks to the buffer
        block1 = self._create_dummy_block(buffer, num_sequences=2)
        block2 = self._create_dummy_block(buffer, num_sequences=2)
        priorities = np.array([1.0, 1.0])

        buffer.add(block1, priorities, episode_reward=None)
        buffer.add(block2, priorities, episode_reward=None)
        buffer.priority_tree.size = 4  # Mark as having 4 sequences

        # Mock sample to return specific indices
        buffer.priority_tree.sample = Mock(return_value=(
            np.array([0, 1, 2, 3]),  # All 4 sequence indices
            np.array([1.0, 1.0, 1.0, 1.0])  # IS weights
        ))

        # Sample a batch
        data = buffer.sample_batch()

        # Verify we sampled from priority tree
        buffer.priority_tree.sample.assert_called_once_with(buffer.batch_size)

        # The sample should complete without errors
        # (verifies block_idxes and sequence_idxes calculations worked)
        assert data is not None

    def test_sequence_index_to_block_conversion_logic(self, buffer):
        """Directly test the index conversion logic used in sample_batch"""
        # seq_per_block = 2
        # Sequence index 0 -> block 0, sequence 0
        # Sequence index 1 -> block 0, sequence 1
        # Sequence index 2 -> block 1, sequence 0
        # Sequence index 3 -> block 1, sequence 1

        idxes = np.array([0, 1, 2, 3])
        block_idxes = idxes // buffer.seq_per_block
        sequence_idxes = idxes % buffer.seq_per_block

        assert np.array_equal(block_idxes, np.array([0, 0, 1, 1]))
        assert np.array_equal(sequence_idxes, np.array([0, 1, 0, 1]))

    def _create_dummy_block(self, buffer, num_sequences=2):
        """Helper to create a dummy block with correct shapes"""
        # Total timesteps in block = block_len
        total_steps = buffer.block_len

        # For simplicity, split evenly across sequences
        learning_steps_per_seq = buffer.sequence_len

        return Block(
            obs=np.zeros((total_steps, 84, 84, 4), dtype=np.uint8),
            last_action=np.zeros((total_steps, 1), dtype=np.int64),
            last_reward=np.zeros((total_steps, 1), dtype=np.float32),
            action=np.zeros(total_steps, dtype=np.int64),
            n_step_reward=np.zeros(total_steps, dtype=np.float32),
            gamma=np.ones(total_steps, dtype=np.float32) * 0.99,
            hidden=np.zeros((num_sequences, 2, 512), dtype=np.float32),
            num_sequences=num_sequences,
            burn_in_steps=np.array([5] * num_sequences, dtype=np.int32),
            learning_steps=np.array([learning_steps_per_seq] * num_sequences, dtype=np.int32),
            forward_steps=np.array([0] * num_sequences, dtype=np.int32),
        )


class TestCircularBufferBehavior:
    """Test suite for circular buffer wraparound logic"""

    @pytest.fixture
    def mock_queues(self):
        """Create mock queues for ReplayBuffer initialization"""
        sample_queue_list = [Mock()]
        batch_queue = Mock()
        priority_queue = Mock()
        return sample_queue_list, batch_queue, priority_queue

    @pytest.fixture
    def small_buffer(self, mock_queues):
        """Create a small ReplayBuffer for wraparound testing"""
        sample_queue_list, batch_queue, priority_queue = mock_queues

        # Very small buffer: 2 blocks, 2 sequences per block
        original_block_length = config.block_length
        config.block_length = 20

        buf = ReplayBuffer(
            sample_queue_list,
            batch_queue,
            priority_queue,
            buffer_capacity=40,
            sequence_len=10,
            alpha=0.6,
            beta=0.4,
            batch_size=2
        )

        yield buf

        config.block_length = original_block_length

    def test_buffer_wraparound(self, small_buffer):
        """Verify buffer pointer wraps around correctly"""
        buffer = small_buffer

        # num_blocks should be 2
        assert buffer.num_blocks == 2
        assert buffer.block_ptr == 0

        # Add first block
        block = self._create_simple_block(buffer)
        buffer.add(block, np.array([1.0, 1.0]), episode_reward=None)
        assert buffer.block_ptr == 1

        # Add second block
        buffer.add(block, np.array([2.0, 2.0]), episode_reward=None)
        assert buffer.block_ptr == 0  # Should wrap to 0

        # Add third block (overwrites first)
        buffer.add(block, np.array([3.0, 3.0]), episode_reward=None)
        assert buffer.block_ptr == 1

    def test_size_calculation_on_overwrite(self, small_buffer):
        """Verify size is updated correctly when overwriting blocks"""
        buffer = small_buffer

        # Create blocks with known learning_steps
        block1 = self._create_simple_block(buffer, learning_steps=[10, 10])
        block2 = self._create_simple_block(buffer, learning_steps=[10, 10])
        block3 = self._create_simple_block(buffer, learning_steps=[8, 8])

        # Add first block: size = 0 + 20 = 20
        buffer.add(block1, np.array([1.0, 1.0]), episode_reward=None)
        assert buffer.size == 20

        # Add second block: size = 20 + 20 = 40
        buffer.add(block2, np.array([2.0, 2.0]), episode_reward=None)
        assert buffer.size == 40

        # Add third block (overwrites first): size = 40 - 20 + 16 = 36
        buffer.add(block3, np.array([3.0, 3.0]), episode_reward=None)
        assert buffer.size == 36

        # Add fourth block (overwrites second): size = 36 - 20 + 16 = 32
        buffer.add(block3, np.array([4.0, 4.0]), episode_reward=None)
        assert buffer.size == 32

    def test_overwrite_replaces_data(self, small_buffer):
        """Verify that wraparound actually replaces old blocks"""
        buffer = small_buffer

        # Add two blocks
        block1 = self._create_simple_block(buffer)
        block2 = self._create_simple_block(buffer)

        buffer.add(block1, np.array([1.0, 1.0]), episode_reward=None)
        buffer.add(block2, np.array([2.0, 2.0]), episode_reward=None)

        # Store reference to block at position 0
        old_block_0 = buffer.buffer[0]
        assert old_block_0 is block1

        # Add third block (should overwrite position 0)
        block3 = self._create_simple_block(buffer)
        buffer.add(block3, np.array([3.0, 3.0]), episode_reward=None)

        # Verify position 0 now contains block3
        assert buffer.buffer[0] is block3
        assert buffer.buffer[0] is not old_block_0

    def test_episode_reward_tracking(self, small_buffer):
        """Verify episode rewards are accumulated correctly"""
        buffer = small_buffer

        block = self._create_simple_block(buffer)

        assert buffer.num_episodes == 0
        assert buffer.episode_reward == 0

        # Add block with no episode end
        buffer.add(block, np.array([1.0, 1.0]), episode_reward=None)
        assert buffer.num_episodes == 0
        assert buffer.episode_reward == 0

        # Add block with episode reward
        buffer.add(block, np.array([1.0, 1.0]), episode_reward=100.5)
        assert buffer.num_episodes == 1
        assert buffer.episode_reward == 100.5

        # Add another episode
        buffer.add(block, np.array([1.0, 1.0]), episode_reward=200.3)
        assert buffer.num_episodes == 2
        assert buffer.episode_reward == 300.8

    def test_buffer_starts_empty(self, small_buffer):
        """Verify buffer initializes in empty state"""
        buffer = small_buffer

        assert buffer.size == 0
        assert buffer.block_ptr == 0
        assert buffer.env_steps == 0
        assert all(block is None for block in buffer.buffer)

    def _create_simple_block(self, buffer, learning_steps=None):
        """Helper to create a simple block for testing"""
        if learning_steps is None:
            learning_steps = [10, 10]  # Default: 2 sequences of 10 steps each

        learning_steps = np.array(learning_steps, dtype=np.int32)
        num_sequences = len(learning_steps)
        total_steps = np.sum(learning_steps)

        return Block(
            obs=np.zeros((total_steps + 10, 84, 84, 4), dtype=np.uint8),  # +10 for burn-in/forward
            last_action=np.zeros((total_steps + 10, 1), dtype=np.int64),
            last_reward=np.zeros((total_steps + 10, 1), dtype=np.float32),
            action=np.zeros(total_steps, dtype=np.int64),
            n_step_reward=np.zeros(total_steps, dtype=np.float32),
            gamma=np.ones(total_steps, dtype=np.float32) * 0.99,
            hidden=np.zeros((num_sequences, 2, 512), dtype=np.float32),
            num_sequences=num_sequences,
            burn_in_steps=np.array([5] * num_sequences, dtype=np.int32),
            learning_steps=learning_steps,
            forward_steps=np.array([0] * num_sequences, dtype=np.int32),
        )


if __name__ == "__main__":
    print("Running ReplayBuffer test suite...")
    pytest.main([__file__, "-v"])
