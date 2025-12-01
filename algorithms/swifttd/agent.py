"""
SwiftTD Q-Learning Agent

Implements Q-learning with 18 separate SwiftTD learners (one per action).
Uses ε-greedy exploration and CNN feature extraction.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import swifttd

# Handle both direct execution and module import
try:
    from .model import CNNFeatureExtractor
    from . import config as default_config
except ImportError:
    from model import CNNFeatureExtractor
    import config as default_config


class SwiftTDAgent:
    """
    SwiftTD Q-learning agent for Atari.

    Uses 18 separate SwiftTD learners (one per action) for Q-learning.
    Features are extracted using a CNN, then fed to each learner.
    """

    def __init__(
        self,
        num_actions=18,
        num_features=512,
        n_stack=4,
        device='cuda',
        # SwiftTD hyperparameters
        lambda_=0.95,
        initial_alpha=1e-3,
        gamma=0.99,
        eps=1e-5,
        max_step_size=0.1,
        step_size_decay=0.9995,
        meta_step_size=1e-4,
        eta_min=1e-10,
        # CNN training
        cnn_learning_rate=1e-4,
        cnn_update_frequency=4,
        gradient_clip=10.0,
        # Exploration
        epsilon=1.0
    ):
        """
        Args:
            num_actions: Number of actions (default: 18 for Atari full action space)
            num_features: CNN feature dimension (default: 512)
            n_stack: Number of stacked frames (default: 4)
            device: Device for CNN ('cuda' or 'cpu')
            lambda_: Eligibility trace decay
            initial_alpha: Initial SwiftTD learning rate
            gamma: Discount factor
            eps: Numerical stability constant
            max_step_size: Maximum step size for SwiftTD
            step_size_decay: Step size decay rate
            meta_step_size: Meta learning rate
            eta_min: Minimum step size
            cnn_learning_rate: Learning rate for CNN
            cnn_update_frequency: Update CNN every N steps
            gradient_clip: Gradient clipping value
            epsilon: Exploration rate
        """
        self.num_actions = num_actions
        self.num_features = num_features
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.epsilon = epsilon
        self.cnn_update_frequency = cnn_update_frequency
        self.gradient_clip = gradient_clip

        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(n_stack=n_stack, feature_dim=num_features)
        self.cnn.to(self.device)
        self.cnn_optimizer = torch.optim.Adam(self.cnn.parameters(), lr=cnn_learning_rate)

        # Create 18 SwiftTD learners (one per action)
        # SwiftTD API: SwiftTDNonSparse(num_of_features, lambda, alpha, gamma, epsilon, eta, decay, meta_step_size, eta_min)
        # Note: Must use positional args because 'lambda' is a Python keyword
        self.q_learners = []
        for i in range(num_actions):
            learner = swifttd.SwiftTDNonSparse(
                num_features,      # num_of_features
                lambda_,           # lambda
                initial_alpha,     # alpha
                gamma,             # gamma
                eps,               # epsilon
                max_step_size,     # eta
                step_size_decay,   # decay
                meta_step_size,    # meta_step_size
                eta_min            # eta_min
            )
            self.q_learners.append(learner)

        # Training state
        self.step_count = 0
        self.last_features = None
        self.last_action = None
        self.accumulated_td_errors = []

        print(f"✓ SwiftTDAgent initialized:")
        print(f"  - {num_actions} SwiftTD Q-learners")
        print(f"  - {num_features} features from CNN")
        print(f"  - Device: {self.device}")
        print(f"  - Epsilon: {epsilon:.3f}")

    def extract_features(self, obs):
        """
        Extract features from observation using CNN.

        Args:
            obs: Observation array of shape (1, 84, 84, 4) from VecFrameStack
             or (84, 84, 4) or (4, 84, 84)

        Returns:
            features: Numpy array of shape (num_features,)
        """
        # Handle VecEnv output (batch dimension)
        if obs.ndim == 4:
            obs = obs[0]  # Take first environment → (84, 84, 4)

        # VecFrameStack outputs (H, W, C), but PyTorch expects (C, H, W)
        if obs.shape[-1] == self.cnn.n_stack:
            # Shape is (84, 84, 4) - transpose to (4, 84, 84)
            obs = np.transpose(obs, (2, 0, 1))

        return self.cnn.extract_features_numpy(obs)

    def get_q_values(self, obs):
        """
        Get Q-values for all actions.

        Args:
            obs: Observation array

        Returns:
            q_values: Array of Q-values for each action (num_actions,)
        """
        features = self.extract_features(obs)

        # Query each learner (pass reward=0 to get prediction without update)
        q_values = []
        for learner in self.q_learners:
            q = learner.step(features.tolist(), 0.0)
            q_values.append(q)

        return np.array(q_values)

    def select_action(self, obs, epsilon=None):
        """
        Select action using ε-greedy policy.

        Args:
            obs: Observation array
            epsilon: Exploration rate (uses self.epsilon if None)

        Returns:
            action: Selected action (int)
        """
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.random() < epsilon:
            # Random action
            return np.random.randint(0, self.num_actions)
        else:
            # Greedy action
            q_values = self.get_q_values(obs)
            return int(np.argmax(q_values))

    def update(self, obs, action, reward, next_obs, done):
        """
        Update the agent after taking a step.

        Args:
            obs: Previous observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode is done
        """
        # Extract features
        features = self.extract_features(obs)
        next_features = self.extract_features(next_obs)

        # Compute TD target
        if done:
            td_target = reward
        else:
            # Get max Q-value for next state
            next_q_values = []
            for learner in self.q_learners:
                q = learner.step(next_features.tolist(), 0.0)
                next_q_values.append(q)
            max_next_q = max(next_q_values)
            td_target = reward + default_config.gamma * max_next_q

        # Get current Q-value
        current_q = self.q_learners[action].step(features.tolist(), 0.0)

        # Compute TD error
        td_error = td_target - current_q

        # Update the Q-learner for the taken action with the actual reward
        # SwiftTD will handle the TD update internally
        self.q_learners[action].step(features.tolist(), reward)

        # Accumulate TD error for CNN update
        self.accumulated_td_errors.append((features, td_error, action))

        self.step_count += 1

        # Update CNN periodically using accumulated TD errors
        if self.step_count % self.cnn_update_frequency == 0 and len(self.accumulated_td_errors) > 0:
            self._update_cnn()
            self.accumulated_td_errors = []

    def _update_cnn(self):
        """Update CNN using accumulated TD errors."""
        if len(self.accumulated_td_errors) == 0:
            return

        # Prepare batch of experiences
        features_list = []
        td_errors_list = []

        for features, td_error, action in self.accumulated_td_errors:
            features_list.append(features)
            td_errors_list.append(td_error)

        # Convert to tensors
        # Note: We need to reconstruct observations to backprop through CNN
        # For now, we'll use a simplified approach with feature-level gradients
        # This is a limitation - ideally we'd store raw observations

        # Skip CNN update for now - would need to store raw observations
        # TODO: Implement proper CNN training with stored observations
        pass

    def set_epsilon(self, epsilon):
        """Set exploration rate."""
        self.epsilon = epsilon

    def save(self, save_path):
        """
        Save agent checkpoint.

        Args:
            save_path: Path to save checkpoint (without extension)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            'cnn_state_dict': self.cnn.state_dict(),
            'cnn_optimizer_state_dict': self.cnn_optimizer.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            # Note: SwiftTD learners don't have native save/load
            # Would need to implement custom serialization
        }

        torch.save(checkpoint, save_path + '.pth')
        print(f"✓ Checkpoint saved to {save_path}.pth")

    def load(self, load_path):
        """
        Load agent checkpoint.

        Args:
            load_path: Path to checkpoint (without extension)
        """
        checkpoint = torch.load(load_path + '.pth', map_location=self.device)

        self.cnn.load_state_dict(checkpoint['cnn_state_dict'])
        self.cnn_optimizer.load_state_dict(checkpoint['cnn_optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.epsilon = checkpoint['epsilon']

        print(f"✓ Checkpoint loaded from {load_path}.pth")
        print(f"  - step_count: {self.step_count}")
        print(f"  - epsilon: {self.epsilon:.3f}")


def test_agent():
    """Test SwiftTDAgent."""
    print("Testing SwiftTDAgent...")

    agent = SwiftTDAgent(
        num_actions=18,
        num_features=512,
        device='cpu',
        epsilon=0.1
    )

    # Test with random observation
    obs = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)

    # Test action selection
    action = agent.select_action(obs)
    print(f"Selected action: {action}")

    # Test Q-values
    q_values = agent.get_q_values(obs)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values}")

    # Test update
    next_obs = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
    agent.update(obs, action, 1.0, next_obs, False)

    print("✓ Agent test passed!")


if __name__ == "__main__":
    test_agent()
