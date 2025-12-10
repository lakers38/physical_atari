#!/usr/bin/env python3
"""
Unit test harness for MemoryFeatures (Figure 9.2 style).

Tests memory feature timing with various (k1, k2) combinations.
Run: python test_imprinting.py
"""

from ale_pong_swift_sarsa import MemoryFeatures


def test_single_memory_feature():
    """
    Test a single memory feature with k1=3 (duration), k2=2 (delay).
    
    Expected behavior when parent (feature 0) fires at t=0:
    - t=0: triggered (phase=0), NOT active (phase < k2)
    - t=1: phase=1, NOT active
    - t=2: phase=2, ACTIVE (k2 <= phase < k2+k1)
    - t=3: phase=3, ACTIVE
    - t=4: phase=4, ACTIVE
    - t=5: phase=5 >= k2+k1=5, goes idle, NOT active
    """
    print("=" * 60)
    print("TEST 1: Single Memory Feature (k1=3, k2=2)")
    print("=" * 60)
    print("Parent 0 fires at t=0, expect:")
    print("  - Delay (k2=2): NOT active at t=0,1")
    print("  - Duration (k1=3): ACTIVE at t=2,3,4")
    print("  - After: idle at t=5+")
    print()
    
    mem = MemoryFeatures(
        num_tenured=5,
        parent_indices=[0],         # Only create memory for parent 0
        timing_pairs=[(3, 2)],      # (k1=3, k2=2)
    )
    
    print(f"Created {mem.num_memory_features} memory feature(s)")
    print()
    
    print(f"{'t':>3} | {'Parent 0':>10} | {'Active Features':>30}")
    print("-" * 50)
    
    expected_active = [False, False, True, True, True, False, False, False, False, False]
    all_pass = True
    
    for t in range(10):
        # Parent 0 fires only at t=0
        if t == 0:
            active_tenured = [0, 1, 2]
        else:
            active_tenured = [1, 2]
        
        active = mem.step(active_tenured)
        # Memory feature has id = num_tenured = 5
        mem_active = 5 in active
        parent_on = 0 in active_tenured
        
        check = "✓" if mem_active == expected_active[t] else "✗"
        if mem_active != expected_active[t]:
            all_pass = False
        
        print(f"{t:>3} | {'ON' if parent_on else 'off':>10} | {str(active):>30} {check}")
    
    print()
    print("PASS" if all_pass else "FAIL")
    print()
    return all_pass


def test_memory_retrigger():
    """
    Test that memory feature can be re-triggered after going idle.
    Parent fires at t=0 and t=8.
    """
    print("=" * 60)
    print("TEST 2: Memory Feature Re-trigger")
    print("=" * 60)
    print("Parent 0 fires at t=0 and t=8")
    print("Memory: k1=2, k2=1")
    print("Expect active at t=1,2 and t=9,10")
    print()
    
    mem = MemoryFeatures(
        num_tenured=5,
        parent_indices=[0],
        timing_pairs=[(2, 1)],  # (k1=2, k2=1)
    )
    mem_id = 5  # first memory feature
    
    print(f"{'t':>3} | {'Parent 0':>10} | {'Mem Active?':>12}")
    print("-" * 40)
    
    # Expected: active at t=1,2 (from trigger at t=0) and t=9,10 (from trigger at t=8)
    expected = [False, True, True, False, False, False, False, False, False, True, True, False]
    all_pass = True
    
    for t in range(12):
        parent_on = (t == 0 or t == 8)
        active_tenured = [0] if parent_on else []
        
        active = mem.step(active_tenured)
        mem_active = mem_id in active
        
        check = "✓" if mem_active == expected[t] else "✗"
        if mem_active != expected[t]:
            all_pass = False
        
        print(f"{t:>3} | {'ON' if parent_on else 'off':>10} | {'YES' if mem_active else 'no':>12} {check}")
    
    print()
    print("PASS" if all_pass else "FAIL")
    print()
    return all_pass


def test_figure_9_2():
    """
    Reproduce Figure 9.2 from the dissertation.
    
    Three memory features connected to the same parent (feature 0):
    - φ[m]: k1=2, k2=2 → active at t+2, t+3
    - φ[n]: k1=3, k2=1 → active at t+1, t+2, t+3
    - φ[o]: k1=1, k2=3 → active at t+3
    
    Parent fires at t=0 and t=8.
    """
    print("=" * 60)
    print("TEST 3: Figure 9.2 - Three Memory Features, Same Parent")
    print("=" * 60)
    print("Parent 0 fires at t=0 and t=8")
    print("φ[m]: k1=2, k2=2 → active at t+2, t+3")
    print("φ[n]: k1=3, k2=1 → active at t+1, t+2, t+3")  
    print("φ[o]: k1=1, k2=3 → active at t+3")
    print()
    
    # Each parent gets ALL timing pairs, so we use a single parent
    # with three timing pairs to get the Figure 9.2 setup
    mem = MemoryFeatures(
        num_tenured=5,
        parent_indices=[0],  # Single parent
        timing_pairs=[
            (2, 2),  # φ[m]: k1=2, k2=2
            (3, 1),  # φ[n]: k1=3, k2=1
            (1, 3),  # φ[o]: k1=1, k2=3
        ],
    )
    
    # IDs: 5, 6, 7 for the three features
    m_id, n_id, o_id = 5, 6, 7
    
    print(f"{'t':>3} | {'Parent':>7} | {'φ[m]':>6} | {'φ[n]':>6} | {'φ[o]':>6}")
    print("-" * 45)
    
    # Expected from Figure 9.2 (parent fires at t=0 and t=8):
    # t:   0  1  2  3  4  5  6  7  8  9 10 11 12
    # m:   -  -  ✓  ✓  -  -  -  -  -  -  ✓  ✓  -
    # n:   -  ✓  ✓  ✓  -  -  -  -  -  ✓  ✓  ✓  -
    # o:   -  -  -  ✓  -  -  -  -  -  -  -  ✓  -
    expected_m = [False, False, True, True, False, False, False, False, False, False, True, True, False]
    expected_n = [False, True, True, True, False, False, False, False, False, True, True, True, False]
    expected_o = [False, False, False, True, False, False, False, False, False, False, False, True, False]
    
    all_pass = True
    
    for t in range(13):
        parent_on = (t == 0 or t == 8)
        active_tenured = [0] if parent_on else []
        
        active = mem.step(active_tenured)
        m_on = m_id in active
        n_on = n_id in active
        o_on = o_id in active
        
        check_m = "✓" if m_on == expected_m[t] else "✗"
        check_n = "✓" if n_on == expected_n[t] else "✗"
        check_o = "✓" if o_on == expected_o[t] else "✗"
        
        if m_on != expected_m[t] or n_on != expected_n[t] or o_on != expected_o[t]:
            all_pass = False
        
        print(f"{t:>3} | {'ON' if parent_on else 'off':>7} | {'YES' if m_on else 'no':>4}{check_m} | {'YES' if n_on else 'no':>4}{check_n} | {'YES' if o_on else 'no':>4}{check_o}")
    
    print()
    print("PASS" if all_pass else "FAIL")
    print()
    return all_pass


def test_multiple_parents():
    """
    Test memory features for multiple different parents.
    """
    print("=" * 60)
    print("TEST 4: Multiple Parents")
    print("=" * 60)
    print("Parent 0 fires at t=0, Parent 1 fires at t=2")
    print("Both have memory with k1=2, k2=1")
    print()
    
    mem = MemoryFeatures(
        num_tenured=5,
        parent_indices=[0, 1],
        timing_pairs=[(2, 1)],  # (k1=2, k2=1)
    )
    
    # IDs: 5 for parent 0, 6 for parent 1
    mem0_id, mem1_id = 5, 6
    
    print(f"{'t':>3} | {'P0':>4} | {'P1':>4} | {'Mem0':>6} | {'Mem1':>6}")
    print("-" * 40)
    
    # Expected:
    # Parent 0 fires at t=0 -> mem0 active at t=1,2
    # Parent 1 fires at t=2 -> mem1 active at t=3,4
    expected_m0 = [False, True, True, False, False, False]
    expected_m1 = [False, False, False, True, True, False]
    
    all_pass = True
    
    for t in range(6):
        active_tenured = []
        if t == 0:
            active_tenured.append(0)
        if t == 2:
            active_tenured.append(1)
        
        active = mem.step(active_tenured)
        m0_on = mem0_id in active
        m1_on = mem1_id in active
        
        check_0 = "✓" if m0_on == expected_m0[t] else "✗"
        check_1 = "✓" if m1_on == expected_m1[t] else "✗"
        
        if m0_on != expected_m0[t] or m1_on != expected_m1[t]:
            all_pass = False
        
        p0 = "ON" if 0 in active_tenured else "off"
        p1 = "ON" if 1 in active_tenured else "off"
        print(f"{t:>3} | {p0:>4} | {p1:>4} | {'YES' if m0_on else 'no':>4}{check_0} | {'YES' if m1_on else 'no':>4}{check_1}")
    
    print()
    print("PASS" if all_pass else "FAIL")
    print()
    return all_pass


def test_reset():
    """
    Test that reset() properly clears all memory phases.
    """
    print("=" * 60)
    print("TEST 5: Reset")
    print("=" * 60)
    print("Trigger memory, then reset, expect it to be idle")
    print()
    
    mem = MemoryFeatures(
        num_tenured=5,
        parent_indices=[0],
        timing_pairs=[(2, 1)],  # (k1=2, k2=1)
    )
    mem_id = 5
    
    # Trigger memory
    active = mem.step([0])
    active = mem.step([])  # t=1, should be active
    
    print(f"After trigger at t=0, step to t=1: {mem_id} in active = {mem_id in active}")
    
    # Reset
    mem.reset()
    print("Called reset()")
    
    # Step again - should NOT be active (was reset)
    active = mem.step([])
    result = mem_id not in active
    print(f"After reset, step: {mem_id} in active = {mem_id in active}")
    
    print()
    print("PASS" if result else "FAIL")
    print()
    return result


if __name__ == "__main__":
    results = []
    results.append(test_single_memory_feature())
    results.append(test_memory_retrigger())
    results.append(test_figure_9_2())
    results.append(test_multiple_parents())
    results.append(test_reset())
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
