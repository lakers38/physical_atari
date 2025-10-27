
"""Priority tree (sum tree) for prioritized experience replay"""

import numpy as np


class PriorityTree:
    """Sum tree for efficient prioritized sampling"""

    def __init__(self, capacity, alpha, beta):
        """
        Args:
            capacity: Maximum number of elements
            alpha: Priority exponent (how much prioritization to use)
            beta: Importance sampling exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0
        self.size = 0

    def update(self, idxes, priorities):
        """
        Update priorities for given indices

        Args:
            idxes: Array of data indices to update
            priorities: Array of new priorities
        """
        priorities = np.abs(priorities) ** self.alpha

        for idx, priority in zip(idxes, priorities):
            tree_idx = idx + self.capacity - 1
            delta = priority - self.tree[tree_idx]
            self.tree[tree_idx] = priority

            # Propagate change up the tree
            while tree_idx != 0:
                tree_idx = (tree_idx - 1) // 2
                self.tree[tree_idx] += delta

        # Update size to track maximum valid index
        if len(idxes) > 0:
            max_idx = np.max(idxes) + 1
            self.size = max(self.size, max_idx)

    def sample(self, batch_size):
        """
        Sample batch_size indices based on priorities

        Args:
            batch_size: Number of samples to draw

        Returns:
            idxes: Array of sampled indices
            is_weights: Importance sampling weights
        """
        # Small epsilon to prevent division by zero
        eps = 1e-8

        # Stratified sampling: divide priority range into batch_size segments
        p_sum = self.tree[0]
        segment_size = p_sum / batch_size

        # Vectorized: sample uniformly from each segment
        # Use explicit array creation to avoid floating point precision issues with np.arange
        prefixsums = np.array([i * segment_size for i in range(batch_size)], dtype=np.float64)
        prefixsums += np.random.uniform(0, segment_size, batch_size)

        # Vectorized tree traversal
        import math
        num_layers = int(math.log2(self.capacity)) + 1
        idxes = np.zeros(batch_size, dtype=np.int64)

        for _ in range(num_layers - 1):
            # Left child indices
            left_children = idxes * 2 + 1
            # Compare with left child values
            nodes = self.tree[left_children]
            # Go left if prefixsum < left_value, else go right
            idxes = np.where(prefixsums < nodes, left_children, left_children + 1)
            # Subtract left child value when going right
            prefixsums = np.where(idxes % 2 == 0, prefixsums - self.tree[idxes - 1], prefixsums)

        # Get priorities and convert tree indices to data indices
        priorities = self.tree[idxes]
        idxes = idxes - (self.capacity - 1)

        # Calculate importance sampling weights
        min_priority = np.min(self.tree[self.capacity-1:self.capacity-1+self.size])

        # Simplified IS weight formula: (p_i / min_p)^(-beta)
        # Add epsilon to prevent division by zero
        is_weights = np.power((priorities + eps) / (min_priority + eps), -self.beta)

        return idxes, is_weights

    def _retrieve(self, idx, value):
        """
        Traverse tree to find leaf index corresponding to cumulative value

        Args:
            idx: Current tree index
            value: Target cumulative value

        Returns:
            Leaf index in tree
        """
        left = 2 * idx + 1
        right = left + 1

        # If leaf node, return
        if left >= len(self.tree):
            return idx

        # Traverse left or right based on cumulative sum
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def total_priority(self):
        """Get total priority (root of tree)"""
        return self.tree[0]

    def display(self, max_depth=None, precision=2):
        """
        Display the tree structure in a human-readable format

        Args:
            max_depth: Maximum depth to display (None for all levels)
            precision: Number of decimal places for values

        Example output for capacity=4:
                     60.00           <- Level 0 (root)
                    /     \
                30.00     30.00      <- Level 1
                /  \      /  \
            10.00 20.00 30.00 0.00   <- Level 2 (leaves, data indices 0-3)
        """
        import math

        # Calculate tree depth
        depth = int(math.log2(self.capacity)) + 1
        if max_depth is not None:
            depth = min(depth, max_depth)

        print(f"\nPriorityTree Display (capacity={self.capacity}, size={self.size})")
        print(f"Total Priority: {self.total_priority():.{precision}f}")
        print(f"Alpha: {self.alpha}, Beta: {self.beta}")
        print("=" * 80)

        # Display level by level
        for level in range(depth):
            # Calculate nodes at this level
            level_start = 2**level - 1
            level_end = min(2**(level+1) - 1, len(self.tree))
            num_nodes = level_end - level_start

            # Calculate spacing
            max_width = 80
            node_width = max_width // (2**level)

            # Build level string
            level_str = ""
            for i in range(level_start, level_end):
                value = self.tree[i]

                # Add data index annotation for leaf nodes
                if i >= self.capacity - 1:
                    data_idx = i - (self.capacity - 1)
                    node_str = f"[{data_idx}]{value:.{precision}f}"
                else:
                    node_str = f"{value:.{precision}f}"

                # Center the node string
                padding = (node_width - len(node_str)) // 2
                level_str += " " * padding + node_str + " " * (node_width - padding - len(node_str))

            print(f"Level {level}: {level_str}")

        print("=" * 80)
        print(f"Leaf nodes are marked with [data_index] prefix\n")

    def display_compact(self, precision=2):
        """
        Display tree in compact list format showing internal nodes and leaf nodes separately

        Args:
            precision: Number of decimal places for values
        """
        print(f"\nPriorityTree Compact View (capacity={self.capacity}, size={self.size})")
        print(f"Total Priority: {self.total_priority():.{precision}f}")
        print("=" * 80)

        # Internal nodes (non-leaves)
        print("Internal Nodes (cumulative sums):")
        internal_nodes = []
        for i in range(self.capacity - 1):
            internal_nodes.append(f"[{i}]:{self.tree[i]:.{precision}f}")
        print("  " + ", ".join(internal_nodes))

        # Leaf nodes (actual data priorities)
        print("\nLeaf Nodes (data priorities):")
        leaf_nodes = []
        for i in range(self.capacity):
            tree_idx = i + self.capacity - 1
            value = self.tree[tree_idx]
            if value > 0 or i < self.size:  # Only show non-zero or within size
                leaf_nodes.append(f"data[{i}]:{value:.{precision}f}")
        print("  " + ", ".join(leaf_nodes if leaf_nodes else ["(empty)"]))

        print("=" * 80 + "\n")
