
import numpy as np
from agent_swift_sarsa import AtariFeatureExtractor

def test_preprocessing():
    # Setup matching the user's description and ale_pong_swift_sarsa.py defaults
    # User says: 210x160x3 -> 105x80x3 -> 105x80x8 (per channel) -> 201,600
    # Plus 18 actions
    # Plus 1 cumulant
    # Total 201,619
    
    # In ale_pong_swift_sarsa.py:
    # K_actions=0, K_rewards=0, use_reward_gap=False
    # num_actions=18 (if not reduced)
    
    extractor = AtariFeatureExtractor(
        height=105,
        width=80,
        num_bins=8,
        num_actions=18,
        use_grayscale=False,
        use_frame_diff=False,
        K_actions=0,
        K_rewards=0,
        use_reward_gap=False
    )
    
    print(f"Total features configured: {extractor.total_features}")
    
    # Create dummy frame: 210x160x3
    frame = np.zeros((210, 160, 3), dtype=np.uint8)
    
    # Extract
    feature_pairs = extractor.extract(frame)
    
    print(f"Extracted feature pairs count: {len(feature_pairs)}")
    indices = [idx for idx, val in feature_pairs]
    print(f"Max index: {max(indices) if indices else 'None'}")
    
    # Check count
    # We expect sparse indices. The number of active features is what 'indices' contains.
    # The user is talking about the DIMENSION of the feature vector.
    # "flatten it to get a vector with 201,600 binary valued components"
    # "append the previous one-hot coded action ... and the cumulant ... to get the final feature vector with 201,619 components"
    
    # So extractor.total_features should be 201,619.
    
    if extractor.total_features == 201619:
        print("SUCCESS: Total features match 201,619")
    else:
        print(f"FAILURE: Total features {extractor.total_features} != 201,619")
        print(f"Difference: {201619 - extractor.total_features}")

    # Check active features
    # 105*80*3 (pixels) + 1 (action) + 1 (cumulant) = 25200 + 1 + 1 = 25202
    expected_active = 105 * 80 * 3 + 1 + 1
    if len(feature_pairs) == expected_active:
        print(f"SUCCESS: Active features match {expected_active}")
    else:
        print(f"FAILURE: Active features {len(feature_pairs)} != {expected_active}")
        
    # Check cumulant value
    # Default is 0.0
    cumulant_pair = feature_pairs[-1]
    if cumulant_pair[0] == extractor.cumulant_base and cumulant_pair[1] == 0.0:
        print("SUCCESS: Default cumulant is 0.0")
    else:
        print(f"FAILURE: Default cumulant {cumulant_pair} != (base, 0.0)")
        
    # Test negative cumulant
    extractor.update_reward(-1.0)
    feature_pairs_neg = extractor.extract(frame)
    cumulant_pair_neg = feature_pairs_neg[-1]
    if cumulant_pair_neg[0] == extractor.cumulant_base and cumulant_pair_neg[1] == -1.0:
        print("SUCCESS: Negative cumulant is -1.0")
    else:
        print(f"FAILURE: Negative cumulant {cumulant_pair_neg} != (base, -1.0)")

if __name__ == "__main__":
    test_preprocessing()
