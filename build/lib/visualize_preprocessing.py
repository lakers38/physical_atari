
import numpy as np
import gymnasium as gym
import cv2
from agent_swift_sarsa import AtariFeatureExtractor
from shimmy.registration import register_gymnasium_envs
import matplotlib.pyplot as plt

def visualize():
    # Register envs
    try:
        register_gymnasium_envs()
    except Exception:
        pass

    # Create env
    env = gym.make("ALE/Pong-v5", obs_type="rgb", full_action_space=True)
    frame, _ = env.reset(seed=42)
    
    # Run a few steps to get a non-empty screen
    for _ in range(50):
        frame, _, _, _, _ = env.step(0) # NOOP

    # Setup extractor
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
    
    # Extract features
    feature_pairs = extractor.extract(frame)
    indices = [idx for idx, val in feature_pairs if idx < extractor.pixel_features]
    
    print(f"Extracted {len(indices)} pixel features")
    
    # Reconstruct image
    # Shape: 105x80x3
    reconstructed = np.zeros((105, 80, 3), dtype=np.uint8)
    
    num_bins = extractor.num_bins
    num_channels = extractor.num_channels
    width = extractor.width
    
    for idx in indices:
        bin_val = idx % num_bins
        pixel_idx = idx // num_bins
        c = pixel_idx % num_channels
        pixel_idx //= num_channels
        x = pixel_idx % width
        y = pixel_idx // width
        
        # Approx pixel value: bin center
        # bin = p // 32. Range [0, 7].
        # 0 -> 0-31 (center 16)
        # 7 -> 224-255 (center 240)
        val = bin_val * 32 + 16
        reconstructed[y, x, c] = val
        
    # Save original (resized) and reconstructed
    original_resized = cv2.resize(frame, (80, 105), interpolation=cv2.INTER_AREA)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_resized)
    axes[0].set_title("Original (Resized 105x80)")
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed)
    axes[1].set_title("Reconstructed from Features")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("preprocessing_visualization.png")
    print("Saved preprocessing_visualization.png")

if __name__ == "__main__":
    visualize()
