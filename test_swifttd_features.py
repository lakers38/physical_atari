#!/usr/bin/env python3
"""
Probe tests for SwiftTD-style feature extraction and Swift-Sarsa learning.

These tests verify:
1. Feature extractor produces correct shapes and counts
2. Binning follows SwiftTD paper (p // 32)
3. Cumulant encoding works correctly (3 binary features: -1/0/+1)
4. Swift-Sarsa learns on simple probe environments
"""

import numpy as np
import sys

# Add path for imports
sys.path.insert(0, '/home/praneet/code/physical_atari')

from agent_swift_sarsa import AtariFeatureExtractor, SwiftSarsaCore
from swiftsarsa import SwiftSarsaBinaryFeatures


def test_feature_extractor_shape():
    """Test that feature extractor produces correct dimensions."""
    print("\n=== Test 1: Feature Extractor Shape ===")
    
    extractor = AtariFeatureExtractor(
        height=105,
        width=80,
        num_bins=8,
        num_actions=18,
        use_grayscale=False,
        use_frame_diff=False,
        use_cumulant=True,
        K_actions=0,
        K_rewards=0,
        use_reward_gap=False,
    )
    
    # Check total features: 105*80*3*8 + 18 + 3 = 201,621
    expected_total = 105 * 80 * 3 * 8 + 18 + 3
    assert extractor.total_features == expected_total, \
        f"Expected {expected_total} features, got {extractor.total_features}"
    print(f"✓ Total features: {extractor.total_features} (expected {expected_total})")
    
    # Check pixel features: 105*80*3*8 = 201,600
    expected_pixels = 105 * 80 * 3 * 8
    assert extractor.pixel_features == expected_pixels, \
        f"Expected {expected_pixels} pixel features, got {extractor.pixel_features}"
    print(f"✓ Pixel features: {extractor.pixel_features} (expected {expected_pixels})")
    
    # Create a random frame and extract features
    frame = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
    features = extractor.extract(frame)
    feature_indices = [idx for idx, _ in features]
    
    # Should have 105*80*3 + 1 (prev action) + 1 (zero cumulant) active features
    expected_active = 105 * 80 * 3 + 1 + 1  # pixels + action + zero-cumulant bit
    assert len(feature_indices) == expected_active, \
        f"Expected {expected_active} active features, got {len(feature_indices)}"
    print(f"✓ Active features (with zero cumulant): {len(feature_indices)} (expected {expected_active})")
    
    # All indices should be in valid range
    assert all(0 <= f < extractor.total_features for f in feature_indices), \
        "Some feature indices out of range!"
    print(f"✓ All feature indices in valid range [0, {extractor.total_features})")
    
    print("PASSED\n")
    return True


def test_binning_logic():
    """Test that binning follows SwiftTD paper: bin = p // 32."""
    print("\n=== Test 2: Binning Logic (p // 32) ===")
    
    extractor = AtariFeatureExtractor(
        height=1,  # Tiny for testing
        width=1,
        num_bins=8,
        num_actions=4,
        use_grayscale=True,  # Single channel for simplicity
        use_frame_diff=False,
        use_cumulant=False,
        K_actions=0,
        K_rewards=0,
        use_reward_gap=False,
    )
    
    # Test each bin boundary
    test_cases = [
        (0, 0),    # 0 // 32 = 0
        (31, 0),   # 31 // 32 = 0
        (32, 1),   # 32 // 32 = 1
        (63, 1),   # 63 // 32 = 1
        (64, 2),   # 64 // 32 = 2
        (127, 3),  # 127 // 32 = 3
        (128, 4),  # 128 // 32 = 4
        (191, 5),  # 191 // 32 = 5
        (192, 6),  # 192 // 32 = 6
        (223, 6),  # 223 // 32 = 6
        (224, 7),  # 224 // 32 = 7
        (255, 7),  # 255 // 32 = 7
    ]
    
    for pixel_value, expected_bin in test_cases:
        # Create 1x1 grayscale frame
        frame = np.array([[[pixel_value]]], dtype=np.uint8)
        # Resize will keep it as-is for 1x1
        frame_resized = np.full((1, 1, 1), pixel_value, dtype=np.uint8)
        
        extractor.reset()
        features = extractor.extract(frame_resized)
        
        # First feature should be the bin index (base_index + bin)
        # For 1x1x1 image, base_index = 0, so feature = bin
        pixel_feature = features[0][0]
        
        assert pixel_feature == expected_bin, \
            f"Pixel value {pixel_value}: expected bin {expected_bin}, got {pixel_feature}"
        print(f"✓ Pixel {pixel_value} → bin {pixel_feature} (expected {expected_bin})")
    
    print("PASSED\n")
    return True


def test_cumulant_encoding():
    """Test that cumulant is encoded as 2 binary features."""
    print("\n=== Test 3: Cumulant Encoding (2 features) ===")
    
    extractor = AtariFeatureExtractor(
        height=1,
        width=1,
        num_bins=8,
        num_actions=4,
        use_grayscale=True,
        use_frame_diff=False,
        use_cumulant=True,
        K_actions=0,
        K_rewards=0,
        use_reward_gap=False,
    )
    
    frame = np.array([[[128]]], dtype=np.uint8)
    
    # Total features: 1*1*1*8 + 4 + 3 = 15
    # Pixel features: 8, Action features: 4, Cumulant features: 3
    # Cumulant base index: 8 + 4 = 12
    cumulant_base = extractor.pixel_features + extractor.num_actions
    assert cumulant_base == 12, f"Expected cumulant base 12, got {cumulant_base}"
    
    # Test positive reward
    extractor.reset()
    extractor.set_cumulant(1.0)  # Positive reward
    features = extractor.extract(frame)
    feature_indices = {idx for idx, _ in features}
    assert (cumulant_base + 2) in feature_indices, "Positive cumulant feature should be active"
    assert cumulant_base not in feature_indices, "Negative cumulant feature should NOT be active"
    assert (cumulant_base + 1) not in feature_indices, "Zero cumulant feature should NOT be active"
    print(f"✓ Positive reward: feature {cumulant_base + 2} active")
    
    # Test negative reward
    extractor.reset()
    extractor.set_cumulant(-1.0)  # Negative reward
    features = extractor.extract(frame)
    feature_indices = {idx for idx, _ in features}
    assert cumulant_base in feature_indices, "Negative cumulant feature should be active"
    assert (cumulant_base + 1) not in feature_indices, "Zero cumulant feature should NOT be active"
    assert (cumulant_base + 2) not in feature_indices, "Positive cumulant feature should NOT be active"
    print(f"✓ Negative reward: feature {cumulant_base} active")
    
    # Test zero reward
    extractor.reset()
    extractor.set_cumulant(0.0)  # Zero reward
    features = extractor.extract(frame)
    feature_indices = {idx for idx, _ in features}
    assert (cumulant_base + 1) in feature_indices, "Zero cumulant feature should be active"
    assert cumulant_base not in feature_indices, "Negative cumulant feature should NOT be active"
    assert (cumulant_base + 2) not in feature_indices, "Positive cumulant feature should NOT be active"
    print(f"✓ Zero reward: only zero cumulant feature active")
    
    print("PASSED\n")
    return True


def test_swift_sarsa_learns_simple():
    """Test that Swift-Sarsa learns on a simple continuing task."""
    print("\n=== Test 4: Swift-Sarsa Learning (Continuing Task) ===")
    
    # Continuing task where we consistently take the same action:
    # - Run episodes where we ONLY take action 0 and get +1 reward
    # - Run episodes where we ONLY take action 1 and get -1 reward
    # We expect Q(s, a0) > Q(s, a1) after learning
    
    num_features = 10
    num_actions = 2
    
    learner = SwiftSarsaBinaryFeatures(
        num_features,
        num_actions,
        0.9,      # lambda
        0.1,      # alpha (high for fast learning)
        0.0,      # theta (no meta-learning)
        2.0,      # eta (need higher for 3 features * 0.1 = 0.3 tau)
        0.999,    # decay
        1e-6,     # epsilon
        1e-10,    # eta_min
    )
    
    # State features (sparse binary)
    state_features = [0, 1, 2]  # 3 active features
    
    # Run episodes with ONLY action 0 (good action, +1 reward)
    for ep in range(50):
        learner.reset_episode()
        for step in range(10):
            learner.learn(state_features, 1.0, 0.99, 0)
    
    # Run episodes with ONLY action 1 (bad action, -1 reward)
    for ep in range(50):
        learner.reset_episode()
        for step in range(10):
            learner.learn(state_features, -1.0, 0.99, 1)
    
    # Check Q-values
    q_values = learner.get_action_values(state_features)
    print(f"Q-values after training: Q(s,a0)={q_values[0]:.4f}, Q(s,a1)={q_values[1]:.4f}")
    
    # Q(s, a0) should be positive (always got +1)
    # Q(s, a1) should be negative (always got -1)
    assert q_values[0] > 0, f"Expected Q(s,a0) > 0, got {q_values[0]}"
    assert q_values[1] < 0, f"Expected Q(s,a1) < 0, got {q_values[1]}"
    assert q_values[0] > q_values[1], \
        f"Expected Q(s,a0) > Q(s,a1), got {q_values[0]} vs {q_values[1]}"
    print(f"✓ Q(s, good_action) = {q_values[0]:.4f} > 0")
    print(f"✓ Q(s, bad_action) = {q_values[1]:.4f} < 0")
    
    print("PASSED\n")
    return True


def test_swift_sarsa_core_integration():
    """Test SwiftSarsaCore with feature extractor integration."""
    print("\n=== Test 5: SwiftSarsaCore Integration ===")
    
    core = SwiftSarsaCore(
        num_envs=1,
        num_actions=4,
        seed=42,
        screen_height=105,
        screen_width=80,
        num_bins=8,
        use_grayscale=False,
        use_frame_diff=False,
        use_cumulant=True,
        K_actions=0,
        K_rewards=0,
        use_reward_gap=False,
        alpha_init=4e-6,
        theta=0.0,
        eta=1.0,
    )
    
    # Check feature count
    expected_features = 105 * 80 * 3 * 8 + 4 + 3  # pixels + actions + cumulant
    actual_features = core.extractors[0].total_features
    assert actual_features == expected_features, \
        f"Expected {expected_features} features, got {actual_features}"
    print(f"✓ Total features: {actual_features}")
    
    # Test act/observe cycle
    frame = np.random.randint(0, 256, (1, 210, 160, 3), dtype=np.uint8)
    
    core.reset(frame)
    actions = core.act(frame)
    
    assert actions.shape == (1,), f"Expected shape (1,), got {actions.shape}"
    assert 0 <= actions[0] < 4, f"Action out of range: {actions[0]}"
    print(f"✓ Action selection works: action={actions[0]}")
    
    # Observe transition
    next_frame = np.random.randint(0, 256, (1, 210, 160, 3), dtype=np.uint8)
    core.observe(next_frame, np.array([1.0]), np.array([False]), np.array([False]))
    print(f"✓ Observe transition works")
    
    # Check TD error is computed
    print(f"✓ TD error: {core.last_td_error:.6f}")
    
    print("PASSED\n")
    return True


def test_active_feature_count():
    """Verify active feature count matches expectation."""
    print("\n=== Test 6: Active Feature Count ===")
    
    extractor = AtariFeatureExtractor(
        height=105,
        width=80,
        num_bins=8,
        num_actions=18,
        use_grayscale=False,
        use_frame_diff=False,
        use_cumulant=True,
        K_actions=0,
        K_rewards=0,
        use_reward_gap=False,
    )
    
    # Random frame
    frame = np.random.randint(0, 256, (105, 80, 3), dtype=np.uint8)
    
    # No cumulant set
    extractor.reset()
    features_no_cumulant = extractor.extract(frame)
    expected_no_cumulant = 105 * 80 * 3 + 1 + 1  # pixels + action + zero-cumulant bit
    assert len(features_no_cumulant) == expected_no_cumulant, \
        f"Expected {expected_no_cumulant}, got {len(features_no_cumulant)}"
    print(f"✓ Without cumulant: {len(features_no_cumulant)} active (expected {expected_no_cumulant})")
    
    # With positive cumulant
    extractor.reset()
    extractor.set_cumulant(1.0)
    features_pos = extractor.extract(frame)
    expected_pos = 105 * 80 * 3 + 1 + 1  # pixels + action + cumulant
    assert len(features_pos) == expected_pos, \
        f"Expected {expected_pos}, got {len(features_pos)}"
    print(f"✓ With positive cumulant: {len(features_pos)} active (expected {expected_pos})")
    
    # With negative cumulant
    extractor.reset()
    extractor.set_cumulant(-1.0)
    features_neg = extractor.extract(frame)
    expected_neg = 105 * 80 * 3 + 1 + 1  # pixels + action + cumulant
    assert len(features_neg) == expected_neg, \
        f"Expected {expected_neg}, got {len(features_neg)}"
    print(f"✓ With negative cumulant: {len(features_neg)} active (expected {expected_neg})")
    
    print("PASSED\n")
    return True


def test_unique_features():
    """Verify all extracted features are unique (no duplicates)."""
    print("\n=== Test 7: Feature Uniqueness ===")
    
    extractor = AtariFeatureExtractor(
        height=105,
        width=80,
        num_bins=8,
        num_actions=18,
        use_grayscale=False,
        use_frame_diff=False,
        use_cumulant=True,
        K_actions=0,
        K_rewards=0,
        use_reward_gap=False,
    )
    
    # Random frame
    frame = np.random.randint(0, 256, (105, 80, 3), dtype=np.uint8)
    
    extractor.set_cumulant(1.0)
    features = extractor.extract(frame)
    
    feature_indices = [idx for idx, _ in features]
    unique_features = set(feature_indices)
    assert len(feature_indices) == len(unique_features), \
        f"Duplicate features found! {len(feature_indices)} total, {len(unique_features)} unique"
    print(f"✓ All {len(feature_indices)} features are unique")
    
    print("PASSED\n")
    return True


def run_all_tests():
    """Run all probe tests."""
    print("=" * 60)
    print("SwiftTD Feature Extraction & Swift-Sarsa Probe Tests")
    print("=" * 60)
    
    tests = [
        test_feature_extractor_shape,
        test_binning_logic,
        test_cumulant_encoding,
        test_swift_sarsa_learns_simple,
        test_swift_sarsa_core_integration,
        test_active_feature_count,
        test_unique_features,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
