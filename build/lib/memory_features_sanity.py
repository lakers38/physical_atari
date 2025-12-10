#!/usr/bin/env python3
"""Tiny sanity check for MemoryFeatures against Fig. 9.2."""

from typing import Dict, List

from ale_pong_swift_sarsa import MemoryFeatures


def run_demo():
    """
    Fire a single parent feature at t=0 and t=8 and print which memory
    features activate over time. Uses the Fig. 9.2 timing pairs.
    """
    timing_pairs = [
        (2, 2),  # φ[m]
        (3, 1),  # φ[n]
        (1, 3),  # φ[o]
    ]

    memory = MemoryFeatures(
        num_tenured=5,
        parent_indices=[0],  # only parent feature 0
        timing_pairs=timing_pairs,
    )

    # Map memory feature ids to human-readable labels
    labels = [f"φ[{name}]" for name in ("m", "n", "o")]
    id_to_label: Dict[int, str] = {
        mem_id: labels[i] for i, mem_id in enumerate(memory._ids)  # type: ignore[attr-defined]
    }

    # Parent fires at t=0 and t=8 (matching the textbook example)
    parent_fires: Dict[int, List[int]] = {0: [0], 8: [0]}

    print("t  parent  active_memory_features")
    for t in range(13):  # t=0..12
        active_tenured = parent_fires.get(t, [])
        active = memory.step(active_tenured)
        active_memory = [
            id_to_label[i] for i in active if i in id_to_label
        ]
        print(f"{t:2d}    {int(bool(active_tenured))}      {active_memory}")


if __name__ == "__main__":
    run_demo()
