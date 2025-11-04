"""Default configuration for R2D2"""

# Observation settings
obs_shape = (4, 84, 84)  # (channels, height, width) - grayscale

# Network architecture
hidden_dim = 512  # LSTM hidden dimension

# Training hyperparameters
lr = 1e-3  # Learning rate
eps_adam = 1.5e-4  # Adam epsilon
grad_norm = 40  # Gradient clipping norm
gamma = 0.997  # Discount factor

# Replay buffer
buffer_capacity = 250_000  # Total capacity in frames
block_length = 120  # Length of each stored block (burn_in + learning)
burn_in_steps = 40  # Steps to warm up LSTM
learning_steps = 80  # Steps used for learning
forward_steps = 5  # N-step return lookahead

# Prioritized replay
prio_exponent = 0.9  # Alpha: priority exponent
importance_sampling_exponent = 0.6  # Beta: IS exponent

# Training
batch_size = 64  # Number of sequences per batch
learning_starts = 50_000  # Start learning after this many frames
training_steps = 1_000_000  # Total training steps
target_net_update_interval = 2500  # Update target network every N steps
save_interval = 10_000  # Save model every N steps

# Distributed training
num_actors = 4  # Number of parallel actors
base_explore_eps = 0.6  # Base epsilon for exploration
alpha = 7  # Epsilon schedule exponent

# Environment
max_episode_steps = 108_000  # 30 minutes at 60 fps

# Logging
log_interval = 10  # Log every N seconds

# Game (can be overridden)
game_name = "ALE/MsPacman-v5"
