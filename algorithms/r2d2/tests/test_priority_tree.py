#!/usr/bin/env python3
"""Test cases for PriorityTree (sum tree for prioritized experience replay)"""

import numpy as np
import pytest

from algorithms.r2d2.priority_tree import PriorityTree


class TestPriorityTree:
    """Test suite for PriorityTree data structure"""

    def test_initialization(self):
        """Verify tree initializes with correct structure and zeros"""
        tree = PriorityTree(capacity=8, alpha=0.6, beta=0.4)

        # Tree should have 2*capacity - 1 nodes
        assert len(tree.tree) == 15  # 2*8 - 1
        # All nodes should start at zero
        assert np.all(tree.tree == 0)
        # Root priority should be zero initially
        assert tree.total_priority() == 0

    def test_single_update_propagation(self):
        """Verify priority updates propagate correctly up the tree"""
        tree = PriorityTree(capacity=4, alpha=1.0, beta=1.0)

        # Update index 0 with priority 10.0
        tree.update(np.array([0]), np.array([10.0]))

        # Check leaf node (index 3 in tree for data index 0)
        # Tree structure: [root, left, right, leaf0, leaf1, leaf2, leaf3]
        #                 [  0,    1,     2,     3,     4,     5,     6]
        # For capacity=4: leaves start at index 3 (capacity-1)
        leaf_idx = 0 + tree.capacity - 1  # = 3
        assert tree.tree[leaf_idx] == 10.0

        # Check parent nodes propagated correctly
        # Root should equal sum of all children = 10.0
        assert tree.tree[0] == 10.0

    def test_multiple_updates(self):
        """Verify multiple priorities are stored and summed correctly"""
        tree = PriorityTree(capacity=4, alpha=1.0, beta=1.0)

        # Update indices 0, 1, 2 with priorities 10, 20, 30
        tree.update(np.array([0, 1, 2]), np.array([10.0, 20.0, 30.0]))

        # Total priority should be sum of all
        assert tree.total_priority() == 60.0

        # Individual leaf nodes should have correct values
        assert tree.tree[0 + tree.capacity - 1] == 10.0  # index 0
        assert tree.tree[1 + tree.capacity - 1] == 20.0  # index 1
        assert tree.tree[2 + tree.capacity - 1] == 30.0  # index 2

    def test_update_existing_priority(self):
        """Verify updating an existing priority correctly adjusts the tree"""
        tree = PriorityTree(capacity=4, alpha=1.0, beta=1.0)

        # Initial update
        tree.update(np.array([0, 1]), np.array([10.0, 20.0]))
        assert tree.total_priority() == 30.0

        # Update index 0 from 10 to 50
        tree.update(np.array([0]), np.array([50.0]))

        # Total should now be 50 + 20 = 70
        assert tree.total_priority() == 70.0
        assert tree.tree[0 + tree.capacity - 1] == 50.0

    def test_alpha_exponent(self):
        """Verify alpha exponent is applied to priorities"""
        tree = PriorityTree(capacity=4, alpha=2.0, beta=1.0)

        # Update with priority 3.0, should be stored as 3.0^2.0 = 9.0
        tree.update(np.array([0]), np.array([3.0]))

        # Check that alpha was applied: |priority|^alpha
        assert tree.tree[0 + tree.capacity - 1] == 9.0
        assert tree.total_priority() == 9.0

    def test_sampling_distribution(self):
        """Verify sampling is proportional to priorities"""
        tree = PriorityTree(capacity=100, alpha=1.0, beta=1.0)

        # Set up: index 0 has priority 90, indices 1-99 have priority 1 each
        # Total = 90 + 99 = 189
        # Index 0 should be sampled ~47.6% of the time
        priorities = np.ones(100)
        priorities[0] = 90.0
        tree.update(np.arange(100), priorities)
        tree.size = 100  # Mark tree as full

        # Sample many times and check distribution
        num_samples = 10000
        samples, _ = tree.sample(num_samples)

        # Count how many times index 0 was sampled
        count_idx_0 = np.sum(samples == 0)
        expected_ratio = 90.0 / 189.0  # ~0.476
        actual_ratio = count_idx_0 / num_samples

        # Should be close to expected (within 5%)
        assert abs(actual_ratio - expected_ratio) < 0.05

    def test_zero_priority_handling(self):
        """Verify tree handles zero priorities without division by zero"""
        tree = PriorityTree(capacity=4, alpha=1.0, beta=1.0)

        # Update with mix of zero and non-zero priorities
        tree.update(np.array([0, 1, 2]), np.array([0.0, 10.0, 20.0]))

        # Total should be 30
        assert tree.total_priority() == 30.0

        # Sampling should not crash (even though index 0 has zero priority)
        tree.size = 3
        idxes, weights = tree.sample(10)

        # Index 0 should rarely be sampled (nearly zero priority with epsilon)
        # Due to epsilon, it has tiny probability, but should be heavily undersampled
        count_idx_0 = np.sum(idxes == 0)
        assert count_idx_0 <= 2  # Should be very rare (at most 2 out of 10)

        # IS weights should be valid (no inf or nan)
        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0)

    def test_negative_priority_handling(self):
        """Verify tree takes absolute value of priorities"""
        tree = PriorityTree(capacity=4, alpha=1.0, beta=1.0)

        # Update with negative priority
        tree.update(np.array([0]), np.array([-10.0]))

        # Should be stored as abs(-10.0)^alpha = 10.0
        assert tree.tree[0 + tree.capacity - 1] == 10.0

    def test_capacity_limits(self):
        """Verify behavior at capacity boundaries"""
        tree = PriorityTree(capacity=4, alpha=1.0, beta=1.0)

        # Update all 4 slots
        tree.update(np.arange(4), np.array([10.0, 20.0, 30.0, 40.0]))
        tree.size = 4

        # Now overwrite index 0 (like circular buffer)
        tree.update(np.array([0]), np.array([100.0]))

        # Total should be 100 + 20 + 30 + 40 = 190
        assert tree.total_priority() == 190.0


def test_display_functions():
    """Demonstrate and test display functions"""
    tree = PriorityTree(capacity=8, alpha=1.0, beta=1.0)

    # Add some priorities
    tree.update(np.array([0, 1, 2, 3, 4]), np.array([10.0, 20.0, 30.0, 40.0, 5.0]))
    tree.size = 5

    print("\n" + "=" * 80)
    print("DISPLAY FUNCTION DEMO")
    print("=" * 80)

    # Show tree structure
    tree.display(precision=2)

    # Show compact view
    tree.display_compact(precision=2)

    # This test just ensures display functions don't crash
    assert True


if __name__ == "__main__":
    # Run tests with pytest
    print("Running PriorityTree test suite...")
    print("\nTo see display demo, run: pytest test_priority_tree.py::test_display_functions -v -s")
    pytest.main([__file__, "-v"])
