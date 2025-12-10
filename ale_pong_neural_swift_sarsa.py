#!/usr/bin/env python3
"""
Swift-Sarsa trainer with Section 9.2 imprinting on Atari Pong.

Usage:
    python ale_pong_swift_sarsa.py --rom Pong --reduce_action_set
    python ale_pong_swift_sarsa.py --rom Pong --reduce_action_set --use_imprinting
"""

from __future__ import annotations

import argparse
import math
import random
import numpy as np
import gymnasium as gym
from collections import deque
from typing import List, Sequence, Tuple
from shimmy.registration import register_gymnasium_envs

from swiftsarsa import SwiftSarsaBinaryFeatures
from agent_swift_sarsa import AtariFeatureExtractor


# ---------------------------------------------------------------------------
# MemoryFeatures (Figure 9.2 style)
# ---------------------------------------------------------------------------

class MemoryFeatures:
    """
    Memory feature generator as in SwiftTD temporal imprinting (Figure 9.2).

    For each chosen parent feature i and timing pair (k1, k2),
    we create a memory feature m with:
        - duration k1   (number of steps it is active)
        - delay    k2   (steps after trigger before it turns on)

    If the parent fires at time t, then m is active for
        t + k2, ..., t + k2 + k1 - 1
    provided it was idle when the parent fired.
    """

    def __init__(
        self,
        num_tenured: int,
    ):
        """
        Args:
            num_tenured: total number of base/tenured features. They occupy
                         global indices [0, ..., num_tenured - 1].
        """
        self.num_tenured = num_tenured

        # Per-memory-feature parameters
        self._parents: List[int] = []   # parent index (global, < num_tenured)
        self._k1: List[int] = []        # duration
        self._k2: List[int] = []        # delay
        self._phase: List[int] = []     # -1 = idle, >=0 = steps since trigger
        self._ids: List[int] = []       # global feature index of this memory feat
        self._index: dict[int, int] = {}  # feat_id -> array index

    @property
    def num_memory_features(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> List[int]:
        """Return current global IDs of all live memory features."""
        return list(self._ids)

    def add_feature(self, parent: int, k1: int, k2: int, feat_id: int, trigger_now: bool = True) -> None:
        """Dynamically add a memory feature with a given global id."""
        if not (0 <= parent < self.num_tenured):
            raise ValueError(f"Parent index {parent} out of range [0, {self.num_tenured})")
        if k1 <= 0:
            raise ValueError(f"Duration k1 must be positive, got {k1}")
        if k2 < 0:
            raise ValueError(f"Delay k2 must be >= 0, got {k2}")
        if feat_id in self._index:
            raise ValueError(f"feat_id {feat_id} already in use in MemoryFeatures")

        j = len(self._ids)
        self._parents.append(parent)
        self._k1.append(k1)
        self._k2.append(k2)
        self._ids.append(feat_id)
        self._phase.append(0 if trigger_now else -1)
        self._index[feat_id] = j

    def is_parent(self, feat_id: int) -> bool:
        """Check if a feature is currently a parent of any other memory feature."""
        return feat_id in self._parents

    def retire_feature(self, feat_id: int) -> bool:
        """
        Remove a memory feature completely (for reaping / reuse).
        Returns True if it was present, False otherwise.
        """
        if feat_id not in self._index:
            return False

        idx = self._index.pop(feat_id)
        last = len(self._ids) - 1

        if idx != last:
            last_id = self._ids[last]
            self._ids[idx] = last_id
            self._parents[idx] = self._parents[last]
            self._k1[idx] = self._k1[last]
            self._k2[idx] = self._k2[last]
            self._phase[idx] = self._phase[last]
            self._index[last_id] = idx

        self._ids.pop()
        self._parents.pop()
        self._k1.pop()
        self._k2.pop()
        self._phase.pop()

        return True

    def step(self, active_tenured: Sequence[int]) -> List[int]:
        """
        Advance one time step and return all active feature indices.

        Args:
            active_tenured: global indices of tenured features that are 1 at
                            this time step. MUST be a subset of
                            [0, num_tenured).

        Returns:
            Sorted list of active feature indices (tenured + memory).
        """
        tenured_set = set(active_tenured)

        # Start from currently-active base/tenured features
        active = set(tenured_set)

        # Update each memory feature
        for j in range(len(self._ids)):
            parent = self._parents[j]
            k1 = self._k1[j]
            k2 = self._k2[j]
            phase = self._phase[j]

            # 1) Trigger if idle and parent fires at this step
            if phase == -1 and parent in tenured_set:
                phase = 0
                self._phase[j] = 0

            # 2) If counting since last trigger, maybe active
            if phase >= 0:
                # Active in window [k2, k2 + k1)
                if k2 <= phase < k2 + k1:
                    active.add(self._ids[j])

                # Advance internal clock
                phase += 1
                self._phase[j] = phase

                # 3) After delay+duration, go back to idle
                if phase >= k2 + k1:
                    self._phase[j] = -1

        return sorted(active)

    def reset(self) -> None:
        """Reset all memory feature phases (for episode boundaries)."""
        for j in range(len(self._phase)):
            self._phase[j] = -1


class MemoryImprinter:
    """Online controller that instantiates new memory features based on tenured parents."""

    def __init__(
        self,
        base_features: int,
        memory: MemoryFeatures,
        free_ids: deque,
        alpha_init: float,
        eta: float,
        tenure_thresh: float,
        patterns: List[Tuple[int, int]],
        max_new_per_step: int,
    ):
        self.base_features = base_features
        self.memory = memory
        self.free_ids = free_ids
        self.alpha_init = alpha_init
        self.eta = eta
        self.tenure_thresh = tenure_thresh
        self.patterns = patterns
        self.max_new_per_step = max_new_per_step

    def _is_tenured(self, weight: float) -> bool:
        return abs(weight) >= self.tenure_thresh

    def step(
        self,
        base_active_prev: Sequence[int],
        w_feat: Sequence[float],
        beta_feat: Sequence[float],
    ) -> None:
        """Run one imprinting decision (after learning on time t)."""
        active_prev = [i for i in base_active_prev if i < self.base_features]

        if not active_prev or not self.patterns:
            return

        # tau_t budget over ϕ_{t-1}
        tau_t = 0.0
        for i in active_prev:
            tau_t += math.exp(beta_feat[i])

        # Tenured parents that were active at t-1
        parents = [i for i in active_prev if self._is_tenured(w_feat[i])]
        if not parents:
            return

        # Create new memory features while budget and capacity permit
        created = 0
        while (
            created < self.max_new_per_step
            and self.free_ids
            and tau_t + self.alpha_init <= self.eta
        ):
            parent = random.choice(parents)
            k1, k2 = random.choice(self.patterns)
            feat_id = self.free_ids.popleft()
            self.memory.add_feature(parent, k1, k2, feat_id, trigger_now=True)

            # Only counts toward tau_t immediately if the new feature is active now (k2 == 0).
            if k2 == 0:
                tau_t += self.alpha_init

            created += 1


class MemoryReaper:
    """
    Periodically removes memory features that stay idle (low |w|) and unused for long.
    Freed IDs are returned to free_ids for reuse.
    """

    def __init__(
        self,
        base_features: int,
        memory: MemoryFeatures,
        free_ids: deque,
        idle_thresh: float,
        patience: int,
        check_every: int = 500,
    ):
        self.base_features = base_features
        self.memory = memory
        self.free_ids = free_ids
        self.idle_thresh = idle_thresh
        self.patience = patience
        self.check_every = check_every
        self.step_count = 0
        self.inactive_steps: dict[int, int] = {}  # feat_id -> steps since last activation

    def step(
        self,
        active_features_t: Sequence[int],
        w_feat: Sequence[float],
    ) -> None:
        """
        Update inactivity counters and occasionally reap dead-ish memory features.

        active_features_t: full active feature indices at time t (base + memory)
        w_feat: max-abs weights per feature across actions (len = total_features)
        """
        self.step_count += 1

        active_mem = {f for f in active_features_t if f >= self.base_features}

        # Reset inactivity for active memory features
        for fid in active_mem:
            self.inactive_steps[fid] = 0

        # Increment inactivity for all current memory features
        for fid in self.memory.ids:
            if fid not in active_mem:
                self.inactive_steps[fid] = self.inactive_steps.get(fid, 0) + 1

        if self.step_count % self.check_every != 0:
            return

        # Identify removal candidates
        to_remove: List[int] = []
        for fid in self.memory.ids:
            if self.inactive_steps.get(fid, 0) >= self.patience:
                if abs(w_feat[fid]) < self.idle_thresh:
                    # CRITICAL: Do not remove if it is a parent of another feature (ghost feature)
                    if not self.memory.is_parent(fid):
                        to_remove.append(fid)

        for fid in to_remove:
            if self.memory.retire_feature(fid):
                self.free_ids.append(fid)
                self.inactive_steps.pop(fid, None)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def parse_int_list(arg: str, default: List[int]) -> List[int]:
    """Parse comma-separated ints with a fallback default."""
    if not arg:
        return default
    try:
        vals = [int(x.strip()) for x in arg.split(",") if x.strip() != ""]
        return vals if vals else default
    except Exception:
        return default


def select_action(q_values: np.ndarray, eps: float, temp: float, use_softmax: bool = True) -> int:
    """Epsilon-greedy with optional softmax exploration."""
    # Guard against NaNs/Infs from upstream; fall back to zeros.
    if not np.all(np.isfinite(q_values)):
        q_values = np.nan_to_num(q_values, nan=0.0, posinf=0.0, neginf=0.0)

    if np.random.random() < eps:
        return np.random.randint(len(q_values))
    if use_softmax:
        q_shifted = q_values - np.max(q_values)
        exp_q = np.exp(np.clip(q_shifted / temp, -50, 50))
        probs = exp_q / (exp_q.sum() + 1e-10)
        # If numerical issues still yield invalid probabilities, choose uniformly.
        if not np.all(np.isfinite(probs)) or np.any(probs < 0) or probs.sum() <= 0:
            return int(np.random.randint(len(q_values)))
        return int(np.random.choice(len(q_values), p=probs))
    return int(np.argmax(q_values))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decisions", type=int, default=500_000)
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--reduce_action_set", action="store_true")
    # Memory features (Figure 9.2 style)
    parser.add_argument("--use_memory", action="store_true",
                        help="Enable memory features (Figure 9.2)")
    parser.add_argument("--ema_decay", type=float, default=0.95,
                        help="EMA decay for return tracking")
    parser.add_argument("--plot_path", type=str, default=None,
                        help="If set, save a PNG plot of return and EMA to this path")
    parser.add_argument("--alpha", type=float, default=1e-7,
                        help="Initial alpha (meta step uses theta). Set <=0 to disable learning.")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Epsilon for exploration")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Softmax temperature")
    parser.add_argument("--pure_epsilon", action="store_true",
                        help="Disable softmax; use pure epsilon-greedy")
    parser.add_argument("--debug_features_every", type=int, default=0,
                        help="If >0, print base feature stats every N global steps")
    parser.add_argument("--max_memory_features", type=int, default=5000,
                        help="Max number of memory features to allocate capacity for")
    parser.add_argument("--max_new_memory_per_step", type=int, default=5,
                        help="Upper bound on new memory features created each step")
    parser.add_argument("--tenure_threshold", type=float, default=0.05,
                        help="|w| threshold to consider a feature tenured")
    parser.add_argument("--memory_durations", type=str, default="1,4,16",
                        help="Comma-separated durations (k1) to sample for memory features")
    parser.add_argument("--memory_delays", type=str, default="0,4,16,32,64,128,256,512",
                        help="Comma-separated delays (k2) to sample for memory features")
    parser.add_argument("--idle_threshold", type=float, default=0.005,
                        help="|w| threshold to consider a memory feature idle (eligible for removal)")
    parser.add_argument("--reaper_patience", type=int, default=5000,
                        help="Steps of inactivity before a low-weight memory feature can be removed")
    parser.add_argument("--reaper_every", type=int, default=500,
                        help="Check for memory feature removal every N decisions")
    args = parser.parse_args()

    # Environment
    try:
        register_gymnasium_envs()
    except Exception:
        pass

    env = gym.make(f"ALE/{args.rom}-v5", obs_type="rgb", full_action_space=True)
    frame, _ = env.reset(seed=args.seed)

    # Action mapping for Pong
    if args.reduce_action_set and args.rom.lower() == "pong":
        # Minimal, sensible Pong set: NOOP, UP, DOWN
        action_map = np.array([0, 2, 3], dtype=np.int64)
        num_actions = len(action_map)
    else:
        action_map = None
        num_actions = env.action_space.n

    try:
        meanings = env.unwrapped.get_action_meanings()
        print(f"Action meanings: {meanings}")
    except Exception:
        pass

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Hyperparams
    gamma = 0.99
    temperature = args.temperature
    exploration_eps = args.epsilon
    reward_clip = 1.0
    learning_enabled = args.alpha > 0
    alpha = max(args.alpha, 1e-12)  # avoid log(0) NaNs in backend
    prev_feature_set = None  # for debug change tracking

    # Neural Feature Extractor (CNN + SwiftTD)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class NeuralFeatureExtractor:
        """
        Extracts features using a fixed random CNN as described:
        Input: 105x80x24 (RGB binned)
        Conv: 25 filters, 3x3x24, stride 2, ReLU
        Output: 52x40x25 -> Flatten -> 52000 features
        """
        def __init__(self, device="cpu"):
            self.device = device
            self.num_bins = 8
            
            # Fixed random CNN
            # Input channels = 24 (current) + 24 (diff) = 48
            self.conv = nn.Conv2d(in_channels=48, out_channels=25, kernel_size=3, stride=2, padding=0)
            
            # Initialize weights U(-1, 1) as per description
            nn.init.uniform_(self.conv.weight, -1.0, 1.0)
            nn.init.uniform_(self.conv.bias, -1.0, 1.0)
            
            self.conv.to(device)
            self.conv.eval() # Fixed features, no training
            
            self.total_features = 52 * 39 * 25
            self.cumulant_base = self.total_features 
            self.total_features += 3 # Add 3 for cumulant (-1, 0, 1)
            
            self.reward_hist = deque([0]*10, maxlen=10) # Dummy for compatibility
            self.current_cumulant = 0.0
            
            self.prev_tensor = None

        def update_reward(self, reward: float):
            self.reward_hist.append(reward)
            # Simple cumulant: sign of reward
            if reward > 0:
                self.current_cumulant = 1.0
            elif reward < 0:
                self.current_cumulant = -1.0
            else:
                self.current_cumulant = 0.0

        def extract(self, frame: np.ndarray) -> List[int]:
            # Frame: (210, 160, 3) -> Resize to (105, 80)
            # We need to do the binning manually to create the 24-channel input
            
            # 1. Resize
            import cv2
            resized = cv2.resize(frame, (80, 105), interpolation=cv2.INTER_AREA) # (105, 80, 3)
            
            # 2. Binning (8 bins per channel)
            tensor = torch.zeros((1, 24, 105, 80), dtype=torch.float32, device=self.device)
            
            for c in range(3): # R, G, B
                # Quantize to 0..7
                bins = (resized[:, :, c] // 32).astype(np.int64) # (105, 80)
                bins = np.clip(bins, 0, 7)
                
                offset = c * 8
                t_bins = torch.from_numpy(bins).to(self.device)
                one_hot = F.one_hot(t_bins, num_classes=8).permute(2, 0, 1).float() # (8, 105, 80)
                tensor[0, offset:offset+8, :, :] = one_hot

            # 3. Frame Difference
            if self.prev_tensor is None:
                diff = torch.zeros_like(tensor)
            else:
                diff = tensor - self.prev_tensor
            
            self.prev_tensor = tensor.clone()
            
            # Concatenate: (1, 48, 105, 80)
            input_tensor = torch.cat([tensor, diff], dim=1)

            # 4. Conv
            with torch.no_grad():
                out = self.conv(input_tensor) # (1, 25, 52, 39)
                out = F.relu(out)
                
            # 5. Flatten and extract non-zeros
            flat = out.view(-1) # (N_features,)
            
            # Find non-zero indices
            indices = torch.nonzero(flat).squeeze(1)
            
            indices_np = indices.cpu().numpy()
            
            # Convert to list of ints
            features = indices_np.tolist()
                
            # Add cumulant (3-bit encoding)
            if self.current_cumulant == -1.0:
                features.append(self.cumulant_base)
            elif self.current_cumulant == 0.0:
                features.append(self.cumulant_base + 1)
            elif self.current_cumulant == 1.0:
                features.append(self.cumulant_base + 2)
            
            return features

        def reset(self):
            """Reset state for new episode (clears reward history)."""
            self.reward_hist.clear()
            self.current_cumulant = 0.0

        def set_prev_action(self, action: int):
            """Dummy method for compatibility."""
            pass

    extractor = NeuralFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Neural Extractor: {extractor.total_features} features")
    base_features = extractor.total_features
    active_per_frame = getattr(extractor, "_active_per_frame", None)
    if active_per_frame is not None:
        tau = alpha * active_per_frame
        print(
            f"Feature stats: total={base_features}, active_per_frame≈{active_per_frame}, "
            f"alpha={alpha}, alpha*active≈{tau:.3e}"
        )

    # Memory timing patterns (discrete grid)
    durations = parse_int_list(args.memory_durations, [1, 4, 16])
    delays = parse_int_list(args.memory_delays, [0, 4, 16, 32, 64, 128, 256, 512])
    patterns = [(k1, k2) for k1 in durations for k2 in delays]

    # Memory features (Figure 9.2 style) – allocate capacity, fill online
    if args.use_memory:
        max_mem = args.max_memory_features
        total_features = base_features + max_mem
        free_mem_ids = deque(range(base_features, base_features + max_mem))
        memory = MemoryFeatures(num_tenured=base_features)
        print(f"Memory capacity: {max_mem} (patterns={len(patterns)})")
    else:
        memory = None
        free_mem_ids = deque()
        total_features = base_features

    # Swift-Sarsa learner
    learner = SwiftSarsaBinaryFeatures(
        total_features,
        num_actions,
        0.9,      # lambda
        alpha,    # alpha
        1e-3,     # theta
        1.0,      # eta
        0.999,    # decay
        1e-6,     # epsilon
        3.06e-7,  # eta_min
    )

    if args.use_memory:
        imprinter = MemoryImprinter(
            base_features=base_features,
            memory=memory,
            free_ids=free_mem_ids,
            alpha_init=learner.get_alpha_init(),
            eta=learner.get_eta(),
            tenure_thresh=args.tenure_threshold,
            patterns=patterns,
            max_new_per_step=args.max_new_memory_per_step,
        )
        reaper = MemoryReaper(
            base_features=base_features,
            memory=memory,
            free_ids=free_mem_ids,
            idle_thresh=args.idle_threshold,
            patience=args.reaper_patience,
            check_every=args.reaper_every,
        )
    else:
        imprinter = None
        reaper = None

    print(f"Base features: {base_features}, Total: {total_features}")

    # -------------------------------------------------------------------------
    # Initialize for first state s_0
    # -------------------------------------------------------------------------
    extractor.reset()
    if memory is not None:
        memory.reset()

    base_indices = extractor.extract(frame)
    
    if memory:
        memory_indices = memory.step(base_indices)
        features = memory_indices # memory.step returns sorted(base + memory)
    else:
        features = base_indices
        
    base_indices_prev = base_indices # Store indices for next step's imprinting logic

    q_values = np.array(learner.get_action_values(features))
    action = select_action(q_values, exploration_eps, temperature, use_softmax=not args.pure_epsilon)

    # Store prev state/action for learn() call
    prev_features = features
    prev_action = action

    extractor.set_prev_action(action)

    # Stats
    episode_reward = 0.0
    reward_ema = None
    episode_idx = 0
    global_step = 0
    returns = []
    ema_series = []
    td_ema = None
    last_avg_q = 0.0
    last_max_q = 0.0
    action_counts = np.zeros(num_actions, dtype=np.int64)
    td_ema = None
    last_avg_q = 0.0
    last_max_q = 0.0

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    while global_step < args.decisions:
        # Map to ALE action and step environment
        ale_action = int(action_map[prev_action]) if action_map is not None else prev_action

        total_r = 0.0
        done = trunc = False
        for _ in range(args.frame_skip):
            next_frame, r, terminated, truncated, _ = env.step(ale_action)
            total_r += float(r)
            frame = next_frame
            done = bool(terminated)
            trunc = bool(truncated)
            if done or trunc:
                break

        reward = np.clip(total_r, -reward_clip, reward_clip)
        g = 0.0 if (done or trunc) else gamma

        # Learn on (prev_features, prev_action) with reward from this transition
        # Update reward in extractor for NEXT step's extraction
        extractor.update_reward(reward)
        
        q_curr = np.array(learner.get_action_values(prev_features))
        if learning_enabled:
            learner.learn(prev_features, reward, g, prev_action)

        # Online imprinting (uses base features from time t)
        if imprinter is not None:
            w_feat = learner.get_feature_weights_max_over_actions()
            beta_feat = learner.get_feature_betas_max_over_actions()
            imprinter.step(base_indices_prev, w_feat, beta_feat)

        # Extract features for next state
        base_indices_next = extractor.extract(frame)
        
        if memory:
            features_next = memory.step(base_indices_next)
        else:
            features_next = base_indices_next

        # Online removal / recycling for memory features
        if reaper is not None:
            reaper.step(features_next, w_feat)

        if args.debug_features_every and (global_step % args.debug_features_every == 0):
            cur_set = set(base_indices_next)
            change_frac = None
            if prev_feature_set is not None and len(cur_set) > 0:
                inter = len(cur_set & prev_feature_set)
                change_frac = 1.0 - inter / len(cur_set)
            prev_feature_set = cur_set

            sample = list(cur_set)
            random.shuffle(sample)
            sample = sample[:5]

            print(
                f"debug step {global_step}: "
                f"active_base={len(cur_set)} "
                f"change_frac={change_frac} "
                f"sample={sample}"
            )

        # Get Q-values and select next action
        q_values_next = np.array(learner.get_action_values(features_next))
        last_avg_q = float(q_values_next.mean())
        last_max_q = float(q_values_next.max())
        delta_hat = reward + g * last_max_q - float(q_curr.max())
        if td_ema is None:
            td_ema = delta_hat
        else:
            td_ema = args.ema_decay * td_ema + (1 - args.ema_decay) * delta_hat
        action_next = select_action(q_values_next, exploration_eps, temperature, use_softmax=not args.pure_epsilon)
        action_counts[action_next] += 1

        # Update extractor state
        extractor.set_prev_action(action_next)

        # Shift for next iteration
        base_indices_prev = base_indices_next
        prev_features = features_next
        prev_action = action_next

        # Bookkeeping
        episode_reward += total_r
        global_step += 1
        
        if global_step % 1000 == 0:
            print(
                f"step {global_step}  "
                f"ep_rew {episode_reward:.1f}  "
                f"avg_q {last_avg_q:.3f}  "
                f"max_q {last_max_q:.3f}  "
                f"mem_feats={memory.num_memory_features if memory else 0}"
            )

        # Episode end
        if done or trunc:
            if reward_ema is None:
                reward_ema = episode_reward
            else:
                reward_ema = args.ema_decay * reward_ema + (1 - args.ema_decay) * episode_reward

            returns.append(episode_reward)
            ema_series.append(reward_ema)

            mem_str = f"  mem_feats={memory.num_memory_features}" if memory else ""
            td_str = f"{td_ema:.3f}" if td_ema is not None else "nan"
            action_str = f"actions {action_counts.tolist()}"
            learn_str = "frozen" if not learning_enabled else "learn"
            print(
                f"ep {episode_idx}  "
                f"reward {episode_reward:.1f}  "
                f"ema {reward_ema:.2f}  "
                f"step {global_step}  "
                f"avg_q {last_avg_q:.3f}  "
                f"max_q {last_max_q:.3f}  "
                f"td_ema {td_str}  "
                f"{action_str}  "
                f"{learn_str}"
                f"{mem_str}"
            )

            episode_idx += 1
            episode_reward = 0.0

            # Reset for new episode
            frame, _ = env.reset()
            extractor.reset()
            if memory is not None:
                memory.reset()
            learner.reset_episode()

            # Initialize for new s_0
            base_indices = extractor.extract(frame)
            
            if memory:
                features = memory.step(base_indices)
            else:
                features = base_indices
                base_indices_prev = base_indices
            q_values = np.array(learner.get_action_values(features))
            action = select_action(q_values, exploration_eps, temperature, use_softmax=not args.pure_epsilon)

    if args.plot_path:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(returns, label="Return", alpha=0.3)
        plt.plot(ema_series, label="EMA", color="red")
        plt.legend()
        plt.title(f"Pong Swift-Sarsa (Binary) - {args.rom}")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.savefig(args.plot_path)
        print(f"Saved plot to {args.plot_path}")

if __name__ == "__main__":
    main()
