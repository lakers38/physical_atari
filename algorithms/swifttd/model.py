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

    Input: (batch, n_stack, H, W) - stacked grayscale frames
    Output: (batch, feature_dim) - feature vector

    Supports 84x84 and 128x128 input sizes.
    """

    def __init__(self, n_stack=4, feature_dim=512, input_size=84):
        """
        Args:
            n_stack: Number of stacked frames (default: 4)
            feature_dim: Output feature dimension (default: 512)
            input_size: Input image size (84 or 128, default: 84)
        """
        super().__init__()
        self.n_stack = n_stack
        self.feature_dim = feature_dim
        self.input_size = input_size

        # Nature DQN convolutional layers
        # Same architecture works for both 84x84 and 128x128
        self.conv = nn.Sequential(
            nn.Conv2d(n_stack, 32, kernel_size=8, stride=4),  # 84→20, 128→31
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),       # 20→9, 31→14
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),       # 9→7, 14→12
            nn.ReLU(True),
            nn.Flatten()
        )

        # Calculate conv output size
        conv_out_size = self._get_conv_output_size(input_size)

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, feature_dim),
            nn.ReLU(True)
        )

        # Initialize weights
        self._initialize_weights()

    def _get_conv_output_size(self, input_size):
        """Calculate the output size of conv layers."""
        # Conv1: kernel=8, stride=4, padding=0
        size = (input_size - 8) // 4 + 1
        # Conv2: kernel=4, stride=2, padding=0
        size = (size - 4) // 2 + 1
        # Conv3: kernel=3, stride=1, padding=0
        size = (size - 3) // 1 + 1
        # Output: 64 channels × size × size
        return 64 * size * size

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
            x: Tensor of shape (batch, n_stack, H, W) where H=W=input_size

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
