"""
CNN Feature Extractor for SwiftTD

Nature DQN architecture adapted for frame-stacked input (4 channels).
Extracts 512-dimensional features from (4, 84, 84) observations.
"""

import torch
import torch.nn as nn
import numpy as np


class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor based on Nature DQN architecture.

    Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
    Output: (batch, 512) - feature vector
    """

    def __init__(self, n_stack=4, feature_dim=512):
        """
        Args:
            n_stack: Number of stacked frames (default: 4)
            feature_dim: Output feature dimension (default: 512)
        """
        super().__init__()
        self.n_stack = n_stack
        self.feature_dim = feature_dim

        # Nature DQN convolutional layers
        # Input: (n_stack, 84, 84)
        self.conv = nn.Sequential(
            nn.Conv2d(n_stack, 32, kernel_size=8, stride=4),  # → (32, 20, 20)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),       # → (64, 9, 9)
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),       # → (64, 7, 7)
            nn.ReLU(True),
            nn.Flatten()                                       # → 3136
        )

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, feature_dim),  # 3136 → 512
            nn.ReLU(True)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, n_stack, 84, 84)

        Returns:
            features: Tensor of shape (batch, feature_dim)
        """
        # Normalize pixel values to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = self.conv(x)
        features = self.fc(x)
        return features

    def extract_features_numpy(self, obs):
        """
        Extract features from numpy observation.

        Args:
            obs: Numpy array of shape (n_stack, 84, 84) or (batch, n_stack, 84, 84)

        Returns:
            features: Numpy array of shape (feature_dim,) or (batch, feature_dim)
        """
        # Convert to tensor
        if obs.ndim == 3:
            obs = obs[np.newaxis, ...]  # Add batch dimension
            squeeze = True
        else:
            squeeze = False

        obs_tensor = torch.from_numpy(obs).float()

        # Move to same device as model
        device = next(self.parameters()).device
        obs_tensor = obs_tensor.to(device)

        # Extract features
        with torch.no_grad():
            features = self.forward(obs_tensor)

        # Convert back to numpy
        features_np = features.cpu().numpy()

        if squeeze:
            features_np = features_np[0]  # Remove batch dimension

        return features_np


def test_model():
    """Test the CNN feature extractor."""
    print("Testing CNNFeatureExtractor...")

    model = CNNFeatureExtractor(n_stack=4, feature_dim=512)

    # Test with random input
    batch_size = 2
    x = torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8)

    print(f"Input shape: {x.shape}")

    # Forward pass
    features = model(x)
    print(f"Output shape: {features.shape}")
    print(f"Output range: [{features.min():.3f}, {features.max():.3f}]")

    # Test numpy interface
    x_numpy = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
    features_numpy = model.extract_features_numpy(x_numpy)
    print(f"Numpy output shape: {features_numpy.shape}")

    print("✓ Model test passed!")


if __name__ == "__main__":
    test_model()
