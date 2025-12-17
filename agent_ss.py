# agent_ss.py
#
# Swift SARSA agent for physical Atari using PPO backbone for feature extraction
# This agent uses a pretrained PPO CNN as a feature extractor and Swift SARSA for control

import os
from collections import deque
from typing import Optional

import cv2
import numpy as np

# Import Swift SARSA C++ bindings
import swift_sarsa
import torch
import torch.nn as nn

from framework.Logger import logger


class PPOFeatureExtractor:
    """
    Extracts features from a pretrained PPO model's CNN encoder.
    Based on PPOBackbone from train_sim.py
    """

    def __init__(self, device: str, ppo_model_path: Optional[str] = None, frame_size: int = 84):
        self.device = torch.device(device)
        self.frame_size = frame_size
        self.in_channels = 4  # Default to 4 stacked frames

        # Nature CNN architecture (standard SB3 CnnPolicy)
        # Expects 4 stacked grayscale frames (4 channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Load weights from PPO checkpoint if provided
        if ppo_model_path is not None:
            self._load_ppo_weights(ppo_model_path)

        self.encoder.eval()
        self.encoder.to(self.device)

        # Compute feature dimension dynamically with dummy forward pass
        # (only if weights weren't loaded, which would have already computed it)
        if not hasattr(self, 'feature_dim'):
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.in_channels, frame_size, frame_size, device=self.device)
                dummy_output = self.encoder(dummy_input)
                flattened = dummy_output.view(dummy_output.size(0), -1)
                self.feature_dim = flattened.size(1)
                logger.info(f"agent_ss: Computed feature_dim={self.feature_dim} for frame_size={frame_size}")

        logger.info(f"agent_ss: PPO feature extractor initialized (feature_dim={self.feature_dim})")

    def _load_ppo_weights(self, weights_path: str):
        """Load CNN weights from SB3 PPO checkpoint."""
        try:
            from stable_baselines3 import PPO

            logger.info(f"agent_ss: Loading PPO model from {weights_path}")
            ppo_model = PPO.load(weights_path, device=self.device)

            # Extract feature extractor CNN weights (it's a Sequential module)
            feature_extractor = ppo_model.policy.features_extractor.cnn

            # Detect input channels from loaded model
            # Access first conv layer directly from the Sequential
            first_conv = None
            for module in feature_extractor.children():
                if isinstance(module, nn.Conv2d):
                    first_conv = module
                    break

            if first_conv is not None:
                detected_channels = first_conv.weight.shape[1]
                logger.info(f"agent_ss: Detected {detected_channels} input channels from checkpoint")

                # Recreate encoder with correct channels
                self.encoder = nn.Sequential(
                    nn.Conv2d(detected_channels, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU(),
                )
                self.in_channels = detected_channels

            # Copy weights
            self.encoder.load_state_dict(feature_extractor.state_dict())

            # Move encoder to device and set to eval mode before dummy forward pass
            self.encoder.to(self.device)
            self.encoder.eval()

            # Compute actual feature dimension with dummy forward pass
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.in_channels, self.frame_size, self.frame_size, device=self.device)
                dummy_output = self.encoder(dummy_input)
                # Flatten to get feature dimension
                flattened = dummy_output.view(dummy_output.size(0), -1)
                self.feature_dim = flattened.size(1)
                logger.info(f"agent_ss: Computed feature_dim={self.feature_dim} from dummy forward pass")

            logger.info("agent_ss: Successfully loaded CNN weights from PPO checkpoint")

        except Exception as e:
            logger.error(f"agent_ss: Failed to load PPO weights: {e}")
            logger.warning("agent_ss: Using randomly initialized features")

    def extract_features(self, stacked_frames: np.ndarray) -> np.ndarray:
        """
        Extract features from stacked frames.

        Args:
            stacked_frames: (4, 84, 84) stacked grayscale frames

        Returns:
            feature vector: (3136,) flattened features
        """
        with torch.no_grad():
            # Normalize to [0, 1]
            x = torch.from_numpy(stacked_frames).float() / 255.0
            x = x.unsqueeze(0).to(self.device)  # Add batch dimension: (1, 4, 84, 84)

            features = self.encoder(x)  # (1, 64, 7, 7)
            features = features.view(features.size(0), -1)  # (1, 3136)

            # Normalize features (same as train_sim.py PPOBackbone)
            features = features / 15.0

            return features.squeeze(0).cpu().numpy().astype(np.float32)


def dense_to_sparse(feature_vec: np.ndarray) -> list[tuple[int, float]]:
    """Convert dense feature vector to sparse format for Swift SARSA."""
    flat = feature_vec.flatten()
    return [(int(i), float(v)) for i, v in enumerate(flat)]


class Agent:
    """Swift SARSA agent for physical Atari using PPO backbone"""

    def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
        # Configuration
        self.num_actions = num_actions
        self.total_frames = total_frames
        self.seed = seed
        self.data_dir = data_dir
        self.gpu = kwargs.get('gpu', 0)

        # Swift SARSA hyperparameters (can be overridden via kwargs)
        self.lambda_ = kwargs.get('lambda_', 0.95)
        self.alpha = kwargs.get('alpha', 1e-7)
        self.meta_step_size = kwargs.get('meta_step_size', 1e-3)
        self.eta = kwargs.get('eta', 1.0)
        self.decay = kwargs.get('decay', 0.999)
        self.epsilon = kwargs.get('epsilon', 0.10)
        self.eta_min = kwargs.get('eta_min', 1e-8)

        # Exploration parameters
        self.exploration = kwargs.get('exploration', 'softmax')  # 'softmax' or 'epsilon_greedy'
        self.eps_greedy_start = kwargs.get('eps_greedy_start', 1.0)
        self.eps_greedy_end = kwargs.get('eps_greedy_end', 0.05)
        self.eps_greedy_end_timestamp = kwargs.get('eps_greedy_end_timestamp', 100_000)
        self.softmax_temp = kwargs.get('softmax_temp', 0.1)

        # Model paths
        self.ppo_weights_path = kwargs.get('ppo_weights_path', None)
        self.sarsa_weights_path = kwargs.get('sarsa_weights_path', None)
        self.load_file = kwargs.get('load_file', None)

        # Frame processing
        self.frame_skip = kwargs.get('frame_skip', 4)
        self.resize_to_84 = True
        self.gamma = kwargs.get('gamma', 0.99)

        logger.info("agent_ss: Initializing Swift SARSA agent")
        logger.info(f"agent_ss: lambda={self.lambda_}, alpha={self.alpha}, meta_step_size={self.meta_step_size}")
        logger.info(f"agent_ss: exploration={self.exploration}, softmax_temp={self.softmax_temp}")

        # Device
        device = f"cuda:{self.gpu}" if torch.cuda.is_available() and self.gpu >= 0 else "cpu"
        logger.info(f"agent_ss: Using device = {device}")

        # Initialize PPO feature extractor
        frame_size = 84 if self.resize_to_84 else 210
        self.feature_extractor = PPOFeatureExtractor(device, self.ppo_weights_path, frame_size=frame_size)
        self.feature_dim = self.feature_extractor.feature_dim

        # Initialize Swift SARSA algorithm
        self.algo = swift_sarsa.SwiftSarsa(
            self.feature_dim,
            num_actions,
            self.lambda_,
            self.alpha,
            self.meta_step_size,
            self.eta,
            self.decay,
            self.epsilon,
            self.eta_min,
        )

        # Load Swift SARSA weights if provided
        if self.sarsa_weights_path and os.path.exists(self.sarsa_weights_path):
            self._load_sarsa_weights(self.sarsa_weights_path)
        elif self.load_file and os.path.exists(self.load_file):
            self._load_sarsa_weights(self.load_file)

        # Frame buffering (stack 4 frames)
        self.frame_buffer = deque(maxlen=4)
        self.step_count = 0
        self.last_action = 0

        # Training tracking
        self.train_losses = []  # Required by harness_physical.py
        self.global_step = 0

        # Episode tracking for SARSA learning
        self.last_feature = None
        self.last_reward = 0.0

        logger.info(f"agent_ss: Initialized successfully (feature_dim={self.feature_dim}, num_actions={num_actions})")

    def _load_sarsa_weights(self, path: str):
        """Load Swift SARSA weights from npz file."""
        try:
            if not path.endswith('.npz'):
                path = path + '.npz'

            data = np.load(path)
            weights = data['weights']
            beta = data['beta']
            last_alpha = data['last_alpha']

            # Verify dimensions match
            expected_size = self.feature_dim * self.num_actions
            if len(weights) != expected_size:
                logger.error(f"agent_ss: Weight size mismatch. Expected {expected_size}, got {len(weights)}")
                logger.error(f"agent_ss: Feature dim={self.feature_dim}, num_actions={self.num_actions}")
                return

            # Set weights in the algorithm
            self.algo.set_weights(weights.tolist(), beta.tolist(), last_alpha.tolist())

            logger.info(f"agent_ss: Successfully loaded SARSA weights from {path}")
            logger.info(f"agent_ss: Weights shape: {weights.shape}, beta shape: {beta.shape}, last_alpha shape: {last_alpha.shape}")

        except Exception as e:
            logger.error(f"agent_ss: Failed to load SARSA weights from {path}: {e}")

    def preprocess_frame(self, observation_rgb8):
        """Preprocess single frame: resize and convert to grayscale"""
        # observation_rgb8 is (160, 210, 3) from physical env
        # Note: OpenCV uses (H, W, C) format

        # Convert to grayscale
        frame = cv2.cvtColor(observation_rgb8, cv2.COLOR_RGB2GRAY)  # (210, 160)

        if self.resize_to_84:
            # Resize to 84x84 (standard Atari preprocessing)
            frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

        return frame

    def _compute_epsilon(self) -> float:
        """Linearly interpolate epsilon from start to end over the schedule."""
        if self.global_step >= self.eps_greedy_end_timestamp:
            return self.eps_greedy_end
        progress = self.global_step / max(self.eps_greedy_end_timestamp, 1)
        return self.eps_greedy_start + progress * (self.eps_greedy_end - self.eps_greedy_start)

    def select_action(self, feature_vec: np.ndarray) -> tuple[int, list[float], float]:
        """
        Select action using Swift SARSA policy.

        Returns:
            action: selected action index
            values: Q-values for all actions
            entropy: policy entropy
        """
        features = dense_to_sparse(feature_vec)
        values = self.algo.get_action_values(features)
        entropy = 0.0

        if self.exploration == "softmax":
            logits = torch.tensor(values, dtype=torch.float32)
            probs = torch.softmax(logits / max(self.softmax_temp, 1e-6), dim=0)
            action = int(torch.multinomial(probs, 1).item())
            entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
        else:
            # Epsilon-greedy
            eps_greedy = self._compute_epsilon()
            if np.random.random() < eps_greedy:
                action = np.random.randint(self.num_actions)
            else:
                action = int(np.argmax(values))
            # Calculate entropy
            p_rand = eps_greedy / self.num_actions
            p_greedy = 1.0 - eps_greedy + p_rand
            entropy = float(-(p_greedy * np.log(max(p_greedy, 1e-12)) +
                            (self.num_actions - 1) * p_rand * np.log(max(p_rand, 1e-12))))

        return action, values, entropy

    def frame(self, observation_rgb8, reward, end_of_episode):
        """
        Called every frame by harness.

        Args:
            observation_rgb8: RGB observation (160, 210, 3) uint8
            reward: Scalar reward from environment
            end_of_episode: Boolean indicating episode termination

        Returns:
            action_index: Integer action index to execute
        """
        self.step_count += 1
        self.global_step += 1

        # Preprocess and buffer frame
        processed_frame = self.preprocess_frame(observation_rgb8)
        self.frame_buffer.append(processed_frame)

        # Only act every frame_skip frames
        if self.step_count % self.frame_skip != 0:
            return self.last_action

        # Wait until we have 4 frames stacked
        if len(self.frame_buffer) < 4:
            return self.last_action

        # Stack 4 grayscale frames: [(H, W), ...] -> (4, H, W)
        stacked_frames = np.stack(list(self.frame_buffer), axis=0)  # (4, 84, 84)

        # Extract features using PPO backbone
        feature = self.feature_extractor.extract_features(stacked_frames)

        # SARSA learning step (if we have a previous state)
        if self.last_feature is not None:
            gamma = 0.0 if end_of_episode else self.gamma
            features_sparse = dense_to_sparse(self.last_feature)

            # Learn from previous transition
            td_error = self.algo.learn(features_sparse, float(self.last_reward), float(gamma), int(self.last_action))

            # Track TD error as "loss" for harness compatibility
            self.train_losses.append(abs(td_error))

        # Reset episode state if needed
        if end_of_episode:
            self.last_feature = None
            self.last_reward = 0.0
            # Clear frame buffer on episode end
            self.frame_buffer.clear()
            return 0  # Return no-op action on episode end

        # Select action for current state
        action, values, entropy = self.select_action(feature)

        # Log Q-values and entropy periodically for debugging
        if self.global_step % 1000 == 0:
            logger.debug(f"agent_ss: step={self.global_step}, action={action}, "
                        f"Q_max={max(values):.3f}, Q_min={min(values):.3f}, entropy={entropy:.3f}")

        # Store for next SARSA update
        self.last_feature = feature
        self.last_action = action
        self.last_reward = reward

        return action

    def save_model(self, filename):
        """Save Swift SARSA weights to disk"""
        try:
            # Get learnable parameters from the algorithm
            weights = self.algo.get_weights()
            beta = self.algo.get_beta()
            last_alpha = self.algo.get_last_alpha()

            # Save as numpy arrays
            save_dict = {
                'weights': np.array(weights, dtype=np.float32),
                'beta': np.array(beta, dtype=np.float32),
                'last_alpha': np.array(last_alpha, dtype=np.float32),
                'num_actions': self.num_actions,
                'feature_dim': self.feature_dim,
                'global_step': self.global_step,
            }

            # Add .npz extension if not present
            if not filename.endswith('.npz'):
                filename = filename + '.npz'

            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savez_compressed(filename, **save_dict)
            logger.info(f"agent_ss: Model saved to {filename}")
        except Exception as e:
            logger.error(f"agent_ss: Error saving model: {e}")
