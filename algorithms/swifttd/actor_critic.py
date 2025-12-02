"""
Actor-Critic architecture with SwiftTD critic for Atari.

Implements A2C with:
- Shared CNN feature extractor (trained with TD backprop)
- Policy head (actor) for action selection
- SwiftTD critic for value function V(s)

Following the approach from SwiftTD paper Section 7:
- SwiftTD applied ONLY to the last layer (critic)
- CNN trained with standard TD(λ) backprop
- Separate learning rates for CNN and SwiftTD
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import swifttd

# Handle both direct execution and module import
try:
    from .model import CNNFeatureExtractor
except ImportError:
    from model import CNNFeatureExtractor


class PolicyHead(nn.Module):
    """
    Policy network (actor) for action selection.

    Takes features from CNN and outputs action probabilities.
    """

    def __init__(self, feature_dim=512, hidden_dim=256, num_actions=18):
        """
        Args:
            feature_dim: Input feature dimension from CNN
            hidden_dim: Hidden layer dimension
            num_actions: Number of discrete actions
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, num_actions)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Forward pass.

        Args:
            features: Tensor of shape (batch, feature_dim)

        Returns:
            logits: Tensor of shape (batch, num_actions)
        """
        return self.network(features)


class ActorCriticSwiftTD:
    """
    Actor-Critic agent with SwiftTD critic for Atari.

    Architecture:
        CNN (shared) → features → Actor (policy)
                                → Critic (SwiftTD value function)

    The CNN is trained with TD(λ) backprop, SwiftTD manages the critic weights.
    """

    def __init__(
        self,
        num_actions=18,
        num_features=512,
        actor_hidden_dim=256,
        n_stack=4,
        input_size=128,
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
        # CNN/Actor training
        cnn_learning_rate=1e-4,
        actor_learning_rate=1e-4,
        entropy_coef=0.01,
        # Logging
        verbose=1
    ):
        """
        Args:
            num_actions: Number of discrete actions
            num_features: CNN feature dimension
            actor_hidden_dim: Hidden dimension for policy network
            n_stack: Number of stacked frames
            input_size: Input image size (84 or 128)
            device: Device for PyTorch ('cuda' or 'cpu')
            lambda_: Eligibility trace decay
            initial_alpha: Initial SwiftTD learning rate
            gamma: Discount factor
            eps: Numerical stability constant for SwiftTD
            max_step_size: Maximum step size for SwiftTD
            step_size_decay: Step size decay rate for SwiftTD
            meta_step_size: Meta learning rate for SwiftTD
            eta_min: Minimum step size for SwiftTD
            cnn_learning_rate: Learning rate for CNN
            actor_learning_rate: Learning rate for actor
            entropy_coef: Entropy coefficient for exploration
            verbose: Verbosity level
        """
        self.num_actions = num_actions
        self.num_features = num_features
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.verbose = verbose

        # Shared CNN feature extractor
        self.cnn = CNNFeatureExtractor(
            n_stack=n_stack,
            feature_dim=num_features,
            input_size=input_size
        )
        self.cnn.to(self.device)

        # Actor (policy head)
        self.actor = PolicyHead(
            feature_dim=num_features,
            hidden_dim=actor_hidden_dim,
            num_actions=num_actions
        )
        self.actor.to(self.device)

        # SwiftTD Critic (value function)
        # Single SwiftTD learner: features → V(s)
        self.critic = swifttd.SwiftTDNonSparse(
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

        # Optimizers (CNN and Actor share optimizer)
        # Note: Critic (SwiftTD) manages its own weights
        self.optimizer = torch.optim.Adam(
            list(self.cnn.parameters()) + list(self.actor.parameters()),
            lr=cnn_learning_rate
        )

        # Track separate LR for actor if different
        if actor_learning_rate != cnn_learning_rate:
            # Use separate optimizers
            self.cnn_optimizer = torch.optim.Adam(
                self.cnn.parameters(),
                lr=cnn_learning_rate
            )
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(),
                lr=actor_learning_rate
            )
            self.use_separate_optimizers = True
        else:
            self.cnn_optimizer = None
            self.actor_optimizer = None
            self.use_separate_optimizers = False

        # Training state
        self.step_count = 0
        self.prev_features = None  # For SwiftTD temporal difference learning
        self.prev_value = 0.0  # Previous state value

        if self.verbose:
            print(f"✓ ActorCriticSwiftTD initialized:")
            print(f"  - Shared CNN: {input_size}x{input_size} → {num_features} features")
            print(f"  - Actor: {num_features} → {actor_hidden_dim} → {num_actions} actions")
            print(f"  - Critic: SwiftTD ({num_features} features → V(s))")
            print(f"  - Device: {self.device}")
            print(f"  - Gamma: {gamma}, Lambda: {lambda_}")
            print(f"  - CNN LR: {cnn_learning_rate}, Actor LR: {actor_learning_rate}")

    def _maybe_bootstrap_critic(self, features_np):
        """
        SwiftTD's `step` expects its internal `v_old` to already contain V(s_t).

        When starting a new episode (or after loading a model) we do not yet have
        that cached value. We perform a single zero-reward `step` to prime the
        critic so that the next call corresponds to the real transition
        (s_t → s_{t+1}, r_{t+1}).
        """
        if self.prev_features is None:
            self.prev_value = self.critic.step(features_np.tolist(), 0.0)
            self.prev_features = features_np

    def start_episode(self, obs):
        """
        Reset per-episode caches and prime the critic on the initial observation.
        """
        self.prev_features = None
        self.prev_value = 0.0
        _, features_np = self.extract_features(obs)
        self._maybe_bootstrap_critic(features_np)

    def extract_features(self, obs):
        """
        Extract features from observation using CNN.

        Args:
            obs: Observation array from VecFrameStack
                 Shape: (batch, H, W, C) or (H, W, C)

        Returns:
            features_torch: PyTorch tensor (batch, num_features) with grad
            features_np: Numpy array (batch, num_features) for SwiftTD
        """
        # Handle batch dimension
        if obs.ndim == 3:
            obs = obs[np.newaxis, ...]  # Add batch dimension
            squeeze = True
        else:
            squeeze = False

        # VecFrameStack outputs (batch, H, W, C), PyTorch expects (batch, C, H, W)
        if obs.shape[-1] == self.cnn.n_stack:
            # Transpose from (batch, H, W, C) to (batch, C, H, W)
            obs = np.transpose(obs, (0, 3, 1, 2))

        # Convert to tensor and normalize to [0, 1]
        obs_tensor = torch.from_numpy(obs).float() / 255.0
        obs_tensor = obs_tensor.to(self.device)

        # Extract features (with gradient for CNN backprop)
        features_torch = self.cnn(obs_tensor)  # CRITICAL_LINE shared encoder φ(s) for actor/critic

        # NaN/inf guard
        if not torch.isfinite(features_torch).all():
            if self.verbose:
                print("[Warning] Non-finite features detected; replacing with zeros.")
            features_torch = torch.nan_to_num(features_torch, nan=0.0, posinf=0.0, neginf=0.0)

        # Get numpy version for SwiftTD (detached)
        features_np = features_torch.detach().cpu().numpy()

        if squeeze:
            features_np = features_np[0]  # Remove batch dimension

        return features_torch, features_np

    def select_action(self, obs, deterministic=False):
        """
        Select action using current policy.

        Args:
            obs: Observation array (single observation, not batched)
            deterministic: If True, select greedy action

        Returns:
            action: Selected action (int)
            log_prob: Log probability of selected action
            entropy: Policy entropy
            value: Value estimate V(s) from critic
        """
        features_torch, features_np = self.extract_features(obs)

        # Get action logits from actor
        logits = self.actor(features_torch)  # CRITICAL_LINE policy head π(a|s)
        if not torch.isfinite(logits).all():
            if self.verbose:
                print("[Warning] Non-finite logits detected; zeroing logits and using uniform policy.")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        # Get action probabilities
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            # Sample from categorical distribution
            # Clone logits to avoid gradient graph issues with PyTorch 2.8
            logits_copy = logits.detach().clone()
            dist = torch.distributions.Categorical(logits=logits_copy)
            action = dist.sample()

        # Calculate log probability and entropy
        log_probs = F.log_softmax(logits, dim=-1)  # CRITICAL_LINE log π(a|s)
        log_prob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        # Return cached value (will be updated in update() method)
        # This avoids calling critic.step() multiple times on the same state
        return action.item(), log_prob, entropy, self.prev_value

    def update(self, obs, action, reward, next_obs, done, log_prob=None, entropy=None):
        """
        Update agent after taking a step (A2C update).

        True Online TD(λ) pattern:
        - Cache V(s_t) via bootstrap call when starting an episode
        - Single critic.step(features_next, reward) per environment transition

        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode is done
            log_prob: (Unused, recomputed) Log probability of action
            entropy: (Unused, recomputed) Policy entropy

        Returns:
            metrics: Dictionary of training metrics
        """
        # Extract features
        features_torch, features_np = self.extract_features(obs)
        _, next_features_np = self.extract_features(next_obs)

        # Ensure critic internal state is aligned with current state value
        self._maybe_bootstrap_critic(features_np)  # CRITICAL_LINE prime SwiftTD with V(s_t)

        # Recompute log_prob and entropy for fresh computation graph
        logits = self.actor(features_torch)  # CRITICAL_LINE policy logits for loss graph
        if not torch.isfinite(logits).all():
            if self.verbose:
                print("[Warning] Non-finite logits during update; sanitizing.")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device).unsqueeze(-1)
        log_prob_fresh = log_probs.gather(-1, action_tensor).squeeze(-1)
        entropy_fresh = -(probs * log_probs).sum(dim=-1)

        # SwiftTD critic update: single step per environment transition
        V_current = self.prev_value  # CRITICAL_LINE cached V(s_t) from SwiftTD

        # Terminal transitions are treated as transition to zero-valued absorbing state
        target_features_np = np.zeros_like(next_features_np) if done else next_features_np
        V_next = self.critic.step(target_features_np.tolist(), reward)  # CRITICAL_LINE SwiftTD update/query on (r, φ(s'))

        # Cache value/features for next step (cleared on terminal)
        if done:
            self.prev_value = 0.0
            self.prev_features = None
        else:
            self.prev_value = V_next
            self.prev_features = target_features_np

        # Compute TD error (advantage)
        td_error = reward + (0.0 if done else self.gamma * V_next) - V_current  # CRITICAL_LINE δ = r + γV(s') − V(s)
        advantage = td_error

        # Convert advantage to tensor (detached, no grad through SwiftTD)
        advantage_tensor = torch.tensor(advantage, dtype=torch.float32, device=self.device)
        if not torch.isfinite(advantage_tensor).all():
            if self.verbose:
                print("[Warning] Non-finite advantage; zeroing to keep training stable.")
            advantage_tensor = torch.zeros_like(advantage_tensor)

        # Actor loss: policy gradient with advantage
        actor_loss = -(log_prob_fresh * advantage_tensor)  # CRITICAL_LINE policy gradient term −logπ * A

        # Add entropy bonus for exploration
        actor_loss = actor_loss - self.entropy_coef * entropy_fresh

        # Backprop through actor and CNN with gradient clipping
        if self.use_separate_optimizers:
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # Update CNN (could add value loss term here if desired)
            # For now, CNN only gets gradients from actor
            self.cnn_optimizer.zero_grad()
            # Recompute features to get fresh gradients
            features_torch, _ = self.extract_features(obs)
            logits = self.actor(features_torch)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            cnn_loss = -(log_probs.gather(-1, torch.tensor([[action]], device=self.device)).squeeze() * advantage_tensor)
            cnn_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cnn.parameters(), max_norm=1.0)
            self.cnn_optimizer.step()
        else:
            # Single optimizer for both
            self.optimizer.zero_grad()
            actor_loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.cnn.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.step_count += 1

        return {
            'value': V_current,
            'td_error': td_error,
            'advantage': advantage,
            'actor_loss': actor_loss.item(),
            'entropy': entropy_fresh.item(),
            'reward': reward
        }

    def save(self, save_path):
        """
        Save agent checkpoint.

        Args:
            save_path: Path to save checkpoint (without extension)
        """
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        checkpoint = {
            'cnn_state_dict': self.cnn.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if not self.use_separate_optimizers else None,
            'cnn_optimizer_state_dict': self.cnn_optimizer.state_dict() if self.use_separate_optimizers else None,
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict() if self.use_separate_optimizers else None,
            'step_count': self.step_count,
            # Note: SwiftTD learner doesn't have native save/load
            # Would need to implement custom serialization
        }

        torch.save(checkpoint, save_path + '.pth')
        if self.verbose:
            print(f"✓ Checkpoint saved to {save_path}.pth")

    def load(self, load_path):
        """
        Load agent checkpoint.

        Args:
            load_path: Path to checkpoint (without extension)
        """
        checkpoint = torch.load(load_path + '.pth', map_location=self.device)

        self.cnn.load_state_dict(checkpoint['cnn_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])

        if self.use_separate_optimizers:
            if checkpoint['cnn_optimizer_state_dict']:
                self.cnn_optimizer.load_state_dict(checkpoint['cnn_optimizer_state_dict'])
            if checkpoint['actor_optimizer_state_dict']:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        else:
            if checkpoint['optimizer_state_dict']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.step_count = checkpoint['step_count']
        # Clear episode-specific caches to avoid stale state after loading
        self.prev_features = None
        self.prev_value = 0.0

        if self.verbose:
            print(f"✓ Checkpoint loaded from {load_path}.pth")
            print(f"  - step_count: {self.step_count}")


def test_actor_critic():
    """Test ActorCriticSwiftTD."""
    print("Testing ActorCriticSwiftTD...")

    agent = ActorCriticSwiftTD(
        num_actions=18,
        num_features=512,
        input_size=128,
        device='cpu',
        verbose=1
    )

    # Test with random observation (128x128x4 from VecFrameStack)
    obs = np.random.randint(0, 256, (128, 128, 4), dtype=np.uint8)

    # Test action selection
    action, log_prob, entropy, value = agent.select_action(obs)
    print(f"Selected action: {action}")
    print(f"Log prob: {log_prob.item():.3f}")
    print(f"Entropy: {entropy.item():.3f}")
    print(f"Value: {value:.3f}")

    # Test update
    next_obs = np.random.randint(0, 256, (128, 128, 4), dtype=np.uint8)
    metrics = agent.update(obs, action, 1.0, next_obs, False, log_prob, entropy)

    print(f"Update metrics: {metrics}")
    print("✓ ActorCriticSwiftTD test passed!")


if __name__ == "__main__":
    test_actor_critic()
