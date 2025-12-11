#!/usr/bin/env python3
"""
Hybrid Swift-SARSA Agent (C++ Backend + PyTorch Body)
-----------------------------------------------------
1. Body: Shallow CNN (PyTorch) learns dense features via Backprop.
2. Head: SwiftSarsa (C++) learns values via Meta-Learning & Traces.
3. Bridge: Uses 'Proxy Rewards' to align Delayed Targets with C++ Traces.
"""

import argparse
import numpy as np
import gymnasium as gym
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from collections import deque
from shimmy.registration import register_gymnasium_envs

# Import compiled C++ extension (module name matches pybind definition)
import swift_sarsa as swiftsarsa

# ----------------------------------------------------------------------
# 1. Feature Extraction (CNN)
# ----------------------------------------------------------------------

class DenseAtariFeatureExtractor(nn.Module):
    """
    Dissertation Sec 7.6: Shallow CNN for dense features.
    Input: (24, 105, 80) Binary Stack -> Output: Flattened Vector
    """
    def __init__(self, input_channels=24, height=105, width=80):
        super().__init__()
        self.height = height
        self.width = width
        
        # 24 input channels -> 25 filters, 3x3, stride 2
        self.conv1 = nn.Conv2d(input_channels, 25, kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # Compute output dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, height, width)
            out = self.conv1(dummy)
            self.feat_dim = out.numel()

    def preprocess(self, frame) -> torch.Tensor:
        """Resize -> Bin -> One-Hot -> Stack"""
        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        binned = resized // 32  # Quantize to 8 bins
        
        planes = []
        for c in range(3):
            # Efficient One-Hot: (105, 80) -> (8, 105, 80)
            channel_bins = np.eye(8, dtype=np.float32)[binned[:, :, c]]
            planes.append(channel_bins.transpose(2, 0, 1))
            
        tensor = np.concatenate(planes, axis=0)
        return torch.from_numpy(tensor).float()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        return x

# ----------------------------------------------------------------------
# 2. Online Normalization (Equation 10.10)
# ----------------------------------------------------------------------

class OnlineNormalizer:
    """
    Whitens features (Mean=0, Var=1) online.
    Essential for the stability of scalar step-size adaptation in Swift.
    """
    def __init__(self, num_features, alpha=1e-4, device="cpu"):
        self.mean = torch.zeros(num_features, device=device)
        self.var = torch.ones(num_features, device=device)
        self.alpha = alpha
        self.epsilon = 1e-5
        self.count = 0

    def normalize(self, x):
        # Stop updating stats after 1M steps to stabilize the manifold
        if self.count < 1_000_000: 
            with torch.no_grad():
                batch_mean = x.mean(0)
                batch_var = x.var(0, unbiased=False)
                self.mean = (1 - self.alpha) * self.mean + self.alpha * batch_mean
                self.var = (1 - self.alpha) * self.var + self.alpha * batch_var
                self.count += 1
        
        return (x - self.mean) / (torch.sqrt(self.var) + self.epsilon)

# ----------------------------------------------------------------------
# 3. Hybrid Agent
# ----------------------------------------------------------------------

class HybridAgent(nn.Module):
    def __init__(
        self, 
        num_actions, 
        input_shape=(24, 105, 80),
        alpha_swift=1e-4, 
        alpha_cnn=1e-5, 
        meta_step=1e-3,
        eta=0.03,
        lambda_val=0.9,
        swift_decay=0.999,
        delay_k=10,
        gamma=0.99,
        device="cpu"
    ):
        super().__init__()
        self.num_actions = num_actions
        self.gamma = gamma
        self.delay_k = delay_k
        self.device = device
        
        # --- A. PyTorch Body ---
        self.extractor = DenseAtariFeatureExtractor()
        self.feat_dim = self.extractor.feat_dim
        print(f"CNN Feature Dimension: {self.feat_dim}")
        
        self.normalizer = OnlineNormalizer(self.feat_dim, device=device)
        self.cnn_optim = torch.optim.Adam(self.extractor.parameters(), lr=alpha_cnn)
        self.to(device)

        # --- B. C++ Head ---
        self.learner = swiftsarsa.SwiftSarsa(
            self.feat_dim,      # num_features
            num_actions,        # num_actions
            lambda_val,         # lambda
            alpha_swift,        # alpha
            meta_step,          # meta_step
            eta,                # eta
            swift_decay,        # decay
            1e-6,               # epsilon (trace cull)
            1e-8                # eta_min
        )
        
        # Buffer: Stores (raw_state_tensor_cpu, action, reward)
        self.buffer = deque(maxlen=delay_k + 2)

    def _to_sparse(self, tensor):
        """
        Convert PyTorch Tensor -> C++ Sparse List [(idx, val)].
        Efficiently extracts non-zero elements (ReLU sparsity).
        """
        flat = tensor.detach().flatten().cpu()
        # Find indices of non-zeros
        indices = torch.nonzero(flat).squeeze(1).numpy()
        values = flat[indices].numpy()
        # Zip for Pybind11 vector<pair<int, float>>
        return list(zip(indices.tolist(), values.tolist()))

    def predict(self, state_tensor):
        """Get Q-values for action selection."""
        with torch.no_grad():
            phi = self.extractor(state_tensor)
            phi = self.normalizer.normalize(phi)
            sparse_phi = self._to_sparse(phi)
            q_vals = self.learner.get_action_values(sparse_phi)
        return q_vals, sparse_phi

    def step(self, raw_state, action, reward, next_raw_state, done):
        """
        Hybrid Update Step with Delay Compensation.
        """
        # 1. Preprocess & Buffer
        state_t = self.extractor.preprocess(raw_state).unsqueeze(0).to(self.device)
        self.buffer.append((state_t, action, reward))
        
        # Prime C++ learner on first step (Set v_old = Q(s0))
        if len(self.buffer) == 1:
            with torch.no_grad():
                phi_0 = self.normalizer.normalize(self.extractor(state_t))
                self.learner.learn(self._to_sparse(phi_0), 0.0, 0.0, action)
            return 0.0

        if len(self.buffer) <= self.delay_k + 1:
            return 0.0

        # --- 2. Prepare Delayed Experience ---
        # We update transition: S_{t-k} -> S_{t-k+1}
        # Using Target derived from: S_t (The "Future")
        
        s_old_tensor, a_old, _ = self.buffer[0]
        s_next_old_tensor, a_next_old, _ = self.buffer[1]
        
        # Move delayed states to GPU for processing
        s_old = s_old_tensor.to(self.device)
        s_next_old = s_next_old_tensor.to(self.device)

        # --- 3. Calculate N-Step Target (Latency Compensated) ---
        accum_r = 0.0
        curr_gamma = 1.0
        # Sum rewards from buffer[1] (result of a_old) to current
        for i in range(1, len(self.buffer)):
            accum_r += self.buffer[i][2] * curr_gamma
            curr_gamma *= self.gamma
            
        # Bootstrap from CURRENT real-time state (S_t)
        # This jumps the delay gap
        with torch.no_grad():
            q_curr, _ = self.predict(state_t)
            boot_val = 0.0 if done else q_curr[action] # On-policy: Q(s_t, a_t)
            
        target_val = accum_r + curr_gamma * boot_val

        # --- 4. Update C++ Head (Meta-Learning) ---
        # We feed S_{t-k+1} to C++.
        # C++ naturally calculates: delta = r + gamma * V(S_{t-k+1}) - V(S_{t-k}).
        # We want: delta = Target - V(S_{t-k}).
        # So we set input r = Proxy Reward.
        
        with torch.no_grad():
            phi_next_old = self.normalizer.normalize(self.extractor(s_next_old))
            sparse_next_old = self._to_sparse(phi_next_old)
            
            # Get V(S_{t-k+1}) using current C++ weights
            q_next_old = self.learner.get_action_values(sparse_next_old)
            v_s_next = q_next_old[a_next_old]

        # Proxy Reward Trick
        proxy_reward = target_val - (self.gamma * v_s_next)
        
        # This updates C++ weights/traces for S_{t-k} based on the target
        self.learner.learn(sparse_next_old, proxy_reward, self.gamma, a_next_old)

        # --- 5. Update PyTorch Body (Backprop) ---
        # We minimize MSE between Target and CNN prediction for S_{t-k}
        # We treat C++ weights as fixed constants here.
        
        # A. Re-compute S_{t-k} with gradients
        phi_old_grad = self.normalizer.normalize(self.extractor(s_old))
        
        # B. Get C++ weights for action a_old
        w_flat = torch.tensor(self.learner.get_weights(), device=self.device)
        w_matrix = w_flat.view(self.num_actions, self.feat_dim)
        w_fixed = w_matrix[a_old].detach()
        
        # C. Predict Q = (CNN_out * C++_weights)
        q_pred = torch.dot(phi_old_grad.flatten(), w_fixed)
        
        # D. Backprop
        loss = 0.5 * (target_val - q_pred) ** 2
        
        self.cnn_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.extractor.parameters(), 1.0)
        self.cnn_optim.step()
        
        return loss.item()

    def reset_episode(self):
        self.learner.reset_episode()
        self.buffer.clear()

# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------

def select_action(q_values, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(len(q_values))
    return int(np.argmax(q_values))

def current_epsilon(step, start, end, decay_steps):
    if step >= decay_steps: return end
    return start - (step / decay_steps) * (start - end)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", type=str, default="MsPacman")
    parser.add_argument("--decisions", type=int, default=2_000_000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--delay_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    
    # Neural Hyperparameters (Optimized)
    parser.add_argument("--alpha_swift", type=float, default=1e-4)
    parser.add_argument("--alpha_cnn", type=float, default=1e-5)
    parser.add_argument("--meta_step", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=0.03)
    
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    try: register_gymnasium_envs()
    except: pass
    
    env = gym.make(f"ALE/{args.rom}-v5", obs_type="rgb", full_action_space=False)
    num_actions = env.action_space.n
    print(f"ROM: {args.rom} | Actions: {num_actions} | Delay: {args.delay_k}")
    
    extractor = DenseAtariFeatureExtractor()
    agent = HybridAgent(
        num_actions,
        alpha_swift=args.alpha_swift,
        alpha_cnn=args.alpha_cnn,
        meta_step=args.meta_step,
        eta=args.eta,
        delay_k=args.delay_k,
        device=device
    )
    
    log_file = open(f"log_hybrid_{args.rom}.txt", "w", buffering=1)
    def log(s): print(s); log_file.write(s+"\n")

    step = 0
    ep_idx = 0
    returns = []

    while step < args.decisions:
        raw_state, _ = env.reset(seed=args.seed + ep_idx)
        agent.reset_episode()
        
        done = False
        ep_reward = 0
        
        # Initial Action
        state_t = agent.extractor.preprocess(raw_state).unsqueeze(0).to(device)
        q_vals, _ = agent.predict(state_t)
        eps = current_epsilon(step, 1.0, 0.05, 200000)
        action = select_action(q_vals, eps)
        
        # Prime the buffer/learner sequence
        agent.step(raw_state, action, 0.0, raw_state, False)
        
        while not done:
            next_raw_state, r, term, trunc, _ = env.step(action)
            done = term or trunc
            step += 1
            ep_reward += r
            
            # Select Next Action
            next_state_t = agent.extractor.preprocess(next_raw_state).unsqueeze(0).to(device)
            q_next, _ = agent.predict(next_state_t)
            eps = current_epsilon(step, 1.0, 0.05, 200000)
            next_action = select_action(q_next, eps)
            
            # Hybrid Step
            loss = agent.step(raw_state, action, np.clip(r, -1, 1), next_raw_state, done)
            
            raw_state = next_raw_state
            action = next_action
            
            if step % 5000 == 0:
                log(f"Step {step}: Loss={loss:.6f}")

        ep_idx += 1
        returns.append(ep_reward)
        mean_100 = np.mean(returns[-100:]) if returns else 0.0
        log(f"Ep {ep_idx}: Reward={ep_reward:.1f} Mean100={mean_100:.2f} Steps={step}")
        
    env.close()
    log_file.close()

if __name__ == "__main__":
    main()
