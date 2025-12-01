"""
SwiftTD Configuration

Hyperparameters for SwiftTD Q-learning on Atari games.
"""

# Environment settings
game_name = "ALE/MsPacman-v5"
n_stack = 4  # Number of frames to stack
obs_shape = (n_stack, 84, 84)  # Stacked frame shape

# SwiftTD hyperparameters
# These are passed to each of the 18 SwiftTD learners
num_features = 512  # CNN output dimension
lambda_ = 0.95  # Eligibility trace decay (λ)
initial_alpha = 1e-3  # Initial learning rate
gamma = 0.99  # Discount factor
eps = 1e-5  # Small constant for numerical stability
max_step_size = 0.1  # Maximum allowed step size
step_size_decay = 0.9995  # Step size decay rate per step
meta_step_size = 1e-4  # Meta learning rate for step-size adaptation
eta_min = 1e-10  # Minimum value of the step-size parameter

# Action space
num_actions = 18  # Atari full action space

# Training settings
training_steps = 1_000_000  # Total training timesteps
eval_frequency = 50_000  # Evaluate every N steps
save_frequency = 100_000  # Save checkpoint every N steps

# Exploration
epsilon_start = 1.0  # Initial exploration rate
epsilon_end = 0.01  # Final exploration rate
epsilon_decay_steps = 250_000  # Steps to decay from start to end

# CNN training
cnn_learning_rate = 1e-4  # Learning rate for CNN feature extractor
cnn_update_frequency = 4  # Update CNN every N steps
gradient_clip = 10.0  # Gradient clipping value

# Evaluation
eval_episodes = 10  # Number of episodes for evaluation
eval_epsilon = 0.001  # Low epsilon for evaluation (nearly greedy)

# Logging
log_frequency = 1000  # Log training stats every N steps
video_frequency = 50_000  # Record video every N steps
video_length = 500  # Number of frames per video
