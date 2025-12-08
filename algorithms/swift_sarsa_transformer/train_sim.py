#!/usr/bin/env python3
"""
Train a Swift-Sarsa controller on Ms. Pacman in simulation using a transformer backbone
to produce features. The code mirrors the SAC training scaffolding (argparse, wandb,
frame skip/stack, preprocessing, recording) but keeps a linear control layer on top
of a swappable transformer encoder (default: RF-DETR Nano/Small/Medium/etc).
"""

import argparse
import json
import os
import pickle
import random
import sys
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from scipy.ndimage import zoom

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import torchvision.models as tv_models
import wandb
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, NoopResetEnv

import ale_py  # noqa: F401 - registers ALE envs
import swift_sarsa

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "latency_wrap"))

from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge  # type: ignore
from rfdetr.util.misc import nested_tensor_from_tensor_list  # type: ignore
from datetime import datetime


# -----------------------------
# Environment wrappers
# -----------------------------
class PreprocessWrapper(gym.Wrapper):
    """Resize frames and optionally convert to grayscale; keeps channel-last layout."""

    def __init__(self, env, frame_size: int, grayscale: bool = False):
        super().__init__(env)
        self.frame_size = frame_size
        self.grayscale = grayscale
        channels = 1 if grayscale else env.observation_space.shape[-1]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(frame_size, frame_size, channels),
            dtype=np.uint8,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self.grayscale:
            frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])[..., None]
        zoom_factors = (
            self.frame_size / frame.shape[0],
            self.frame_size / frame.shape[1],
            1,
        )
        resized = zoom(frame, zoom_factors, order=1)
        return resized.astype(np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._preprocess(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._preprocess(obs), reward, terminated, truncated, info


class ActionSetWrapper(gym.Wrapper):
    """
    Restrict the action space similar to agent_delay_target logic.
    - reduce_action_set=0: full action set
    - reduce_action_set=1: ALE minimal action set
    - reduce_action_set=2: 4 directional actions for Ms. Pacman / Q*bert
    """

    def __init__(self, env, reduce_action_set=1, game_name=""):
        super().__init__(env)
        self.reduce_action_set = reduce_action_set
        self.game_name = game_name.lower()

        if reduce_action_set == 0:
            self.action_mapping = None
        elif reduce_action_set == 2:
            # ALE action indices: UP=2, DOWN=5, LEFT=4, RIGHT=3
            self.action_mapping = [2,5,4,3] # Switch back to [2, 5, 4, 3] for full action space
            print(f"[ActionSetWrapper] Restricting {game_name} to 4 directional actions only")
        else:
            self.action_mapping = None

        if self.action_mapping is not None:
            self.action_space = spaces.Discrete(len(self.action_mapping))
            print(f"[ActionSetWrapper] Action space reduced to {len(self.action_mapping)} actions: {self.action_mapping}")

    def step(self, action):
        if self.action_mapping is not None:
            action = self.action_mapping[action]
        return self.env.step(action)

# -----------------------------
# Backbone interface
# -----------------------------
class FeatureBackbone:
    """Minimal interface for swapping different feature encoders."""
    feature_dim: int
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RFDetrBackbone(FeatureBackbone):
    """
    RF-DETR backbone that returns a pooled feature vector from the last backbone level.
    Designed for fast inference (no grads) and easy swapping by model name.
    """

    def __init__(self, model_name: str, device: str, frame_size: int):
        model_map = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "medium": RFDETRMedium,
            "large": RFDETRLarge,
        }
        if model_name not in model_map:
            raise ValueError(f"Unknown RF-DETR model '{model_name}'. Options: {list(model_map.keys())}")
        model_cls = model_map[model_name]
        # Instantiate wrapper with explicit resolution tied to frame_size
        self.wrapper = model_cls(device=device, resolution=frame_size)
        self.model = self.wrapper.model.model  # underlying torch module (LW-DETR)
        self.model.eval()
        self.device = torch.device(device)
        self.means = self.wrapper.means
        self.stds = self.wrapper.stds
        self.resolution = frame_size
        self.feature_dim = self.model.transformer.d_model

    def _prep_obs(self, obs: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(obs).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # CHW
        assert tensor.shape == (3, self.resolution, self.resolution), f"Expected (3, {self.resolution}, {self.resolution}), got {tensor.shape}"
        tensor = TVF.normalize(tensor, self.means, self.stds)
        return tensor

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = self._prep_obs(obs).to(self.device)
            nested = nested_tensor_from_tensor_list([tensor])
            assert nested.tensors.shape == (1,3,self.resolution,self.resolution), f"Expected (1, 3, {self.resolution}, {self.resolution}), got {nested.tensors.shape}"
            feats_out = self.model.backbone(nested)
            feat_list = feats_out[0] if isinstance(feats_out, tuple) else feats_out
            last = feat_list[-1]
            assert hasattr(last, "tensors"), f"Expected last to have tensors, got {type(last)}"
            feat = last.tensors
            pooled = feat.mean(dim=[2, 3])
            pooled = pooled.squeeze(0)
            pooled = pooled / (pooled.norm(dim=0, keepdim=True) + 1e-8)
            return pooled.cpu().numpy().astype(np.float32)


class ResNetBackbone(FeatureBackbone):
    """Pretrained ResNet that outputs the pooled penultimate feature vector."""

    def __init__(self, model_name: str, device: str, frame_size: int):
        if model_name != "resnet_18":
            raise ValueError("Only resnet_18 supported for now")
        # Load pretrained weights
        self.model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove classifier head; use avgpool + flatten
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.means = tv_models.ResNet18_Weights.IMAGENET1K_V1.transforms().mean
        self.stds = tv_models.ResNet18_Weights.IMAGENET1K_V1.transforms().std
        self.resolution = frame_size
        self.feature_dim = 512  # resnet18 penultimate dim

    def _prep_obs(self, obs: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(obs).float() / 255.0
        if tensor.shape[-1] == 1:
            tensor = tensor.repeat(1, 1, 3)
        elif tensor.shape[-1] > 3:
            raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[-1]}")
        tensor = tensor.permute(2, 0, 1)  # CHW
        tensor = TVF.resize(tensor, (self.resolution, self.resolution), antialias=True)
        tensor = TVF.normalize(tensor, self.means, self.stds)
        return tensor

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = self._prep_obs(obs).unsqueeze(0).to(self.device)  # B=1
            feats = self.model(x)
            # normalize features to zero mean and unit variance
            # feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
            feats = feats / 40.0
            return feats.squeeze(0).cpu().numpy().astype(np.float32)


class CartPoleIdentityBackbone(FeatureBackbone):
    """
    Optionally discretizes and binarizes CartPole observations based on standard bounds/bins.
    If discretization is False, simply returns the (flattened) feature vector as is.
    Cart Position:      min=-4.8, max=4.8    (20 bins)
    Cart Velocity:      unbounded            (20 bins, clip at -3/3)
    Pole Angle:         min=-0.418, max=0.418 (20 bins)
    Pole Angular Vel:   unbounded            (20 bins, clip at -5/5)
    Outputs a 1-hot concatenation of all 4 variable buckets: 80-dim binary.
    """
    CART_POSITION_BINS = 8
    CART_VELOCITY_BINS = 8
    POLE_ANGLE_BINS = 8
    POLE_ANGVEL_BINS = 8

    CART_POSITION_MIN = -4.8
    CART_POSITION_MAX = 4.8

    CART_VELOCITY_MIN = -5.0
    CART_VELOCITY_MAX = 5.0

    POLE_ANGLE_MIN = -0.418
    POLE_ANGLE_MAX = 0.418

    POLE_ANGVEL_MIN = -5.0
    POLE_ANGVEL_MAX = 5.0

    def __init__(self, obs_space, discretize: bool = True):
        assert isinstance(obs_space, spaces.Box)
        assert len(obs_space.shape) == 1  # e.g. CartPole: (4,)
        self.discretize = discretize
        self.feature_dim = (
            self.CART_POSITION_BINS +
            self.CART_VELOCITY_BINS +
            self.POLE_ANGLE_BINS +
            self.POLE_ANGVEL_BINS
        )

    def _discretize(self, val, vmin, vmax, nbins):
        # Clip to range, then bin
        clipped = np.clip(val, vmin, vmax)
        bins = np.linspace(vmin, vmax, nbins + 1)
        idx = np.digitize([clipped], bins) - 1  # digitize returns 1-based
        return int(np.clip(idx[0], 0, nbins - 1))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if not self.discretize:
            # Just return the flattened continuous input observations as the feature vector
            return obs.flatten().astype(np.float32)
        # Discretize each variable
        pos = self._discretize(
            obs[0], self.CART_POSITION_MIN, self.CART_POSITION_MAX, self.CART_POSITION_BINS
        )
        vel = self._discretize(
            obs[1], self.CART_VELOCITY_MIN, self.CART_VELOCITY_MAX, self.CART_VELOCITY_BINS
        )
        ang = self._discretize(
            obs[2], self.POLE_ANGLE_MIN, self.POLE_ANGLE_MAX, self.POLE_ANGLE_BINS
        )
        angvel = self._discretize(
            obs[3], self.POLE_ANGVEL_MIN, self.POLE_ANGVEL_MAX, self.POLE_ANGVEL_BINS
        )
        # 1-hot encode each variable, then concatenate
        feature = np.zeros(self.feature_dim, dtype=np.float32)
        feature[pos] = 1.0
        feature[self.CART_POSITION_BINS + vel] = 1.0
        feature[self.CART_POSITION_BINS + self.CART_VELOCITY_BINS + ang] = 1.0
        feature[self.CART_POSITION_BINS + self.CART_VELOCITY_BINS + self.POLE_ANGLE_BINS + angvel] = 1.0
        return feature


def dense_to_sparse(feature_vec: np.ndarray) -> List[Tuple[int, float]]:
    flat = feature_vec.flatten()
    return [(int(i), float(v)) for i, v in enumerate(flat)]

def _stat_dict(values: List[float], prefix: str) -> Dict[str, float]:
    if not values:
        return {
            f"sarsa/{prefix}_mean": 0.0,
            f"sarsa/{prefix}_min": 0.0,
            f"sarsa/{prefix}_max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        f"sarsa/{prefix}_mean": float(arr.mean()),
        f"sarsa/{prefix}_min": float(arr.min()),
        f"sarsa/{prefix}_max": float(arr.max()),
    }

def collect_swiftsarsa_stats(agent) -> Dict[str, float]:
    """Collect mean/min/max for key SwiftSarsa internal buffers."""
    stats: Dict[str, float] = {}
    try:
        stats.update(_stat_dict(agent.algo.get_weights(), "w"))
        stats.update(_stat_dict(agent.algo.get_beta(), "beta"))
        stats.update(_stat_dict(agent.algo.get_last_alpha(), "last_alpha"))
    except Exception as e:
        print(f"[stats] failed to collect SwiftSarsa internals: {e}")
    return stats


@dataclass
class SwiftSarsaConfig:
    lambda_: float
    alpha: float
    meta_step_size: float
    eta: float
    decay: float
    epsilon: float
    eta_min: float
    exploration: str
    eps_greedy_start: float
    eps_greedy_end: float
    eps_greedy_end_timestamp: int
    softmax_temp: float


class SwiftSarsaAgent:
    def __init__(self, num_actions: int, feature_dim: int, cfg: SwiftSarsaConfig):
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.cfg = cfg
        self.algo = swift_sarsa.SwiftSarsa(
            feature_dim,
            num_actions,
            cfg.lambda_,
            cfg.alpha,
            cfg.meta_step_size,
            cfg.eta,
            cfg.decay,
            cfg.epsilon,
            cfg.eta_min,
        )

    def _compute_epsilon(self, global_step: int) -> float:
        """Linearly interpolate epsilon from start to end over the schedule."""
        if global_step >= self.cfg.eps_greedy_end_timestamp:
            return self.cfg.eps_greedy_end
        progress = global_step / max(self.cfg.eps_greedy_end_timestamp, 1)
        return self.cfg.eps_greedy_start + progress * (self.cfg.eps_greedy_end - self.cfg.eps_greedy_start)

    def select_action(self, feature_vec: np.ndarray, global_step: int = 0) -> Tuple[int, List[float], float, float]:
        features = dense_to_sparse(feature_vec)
        values = self.algo.get_action_values(features)
        entropy = 0.0
        if self.cfg.exploration == "softmax":
            logits = torch.tensor(values, dtype=torch.float32)
            probs = torch.softmax(logits / max(self.cfg.softmax_temp, 1e-6), dim=0)
            action = int(torch.multinomial(probs, 1).item())
            entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
            eps_greedy = 0.0  # not used for softmax
        else:
            eps_greedy = self._compute_epsilon(global_step)
            if random.random() < eps_greedy:
                action = random.randrange(self.num_actions)
            else:
                action = int(np.argmax(values))
            # epsilon-greedy distribution entropy
            p_rand = eps_greedy / self.num_actions
            p_greedy = 1.0 - eps_greedy + p_rand
            entropy = float(-(p_greedy * math.log(max(p_greedy, 1e-12)) +
                              (self.num_actions - 1) * p_rand * math.log(max(p_rand, 1e-12))))
        return action, values, entropy, eps_greedy

    def learn(self, feature_vec: np.ndarray, reward: float, gamma: float, action: int) -> float:
        features = dense_to_sparse(feature_vec)
        return self.algo.learn(features, float(reward), float(gamma), int(action))


# -----------------------------
# Training helpers
# -----------------------------
def create_cartpole_env(seed: int, video_path: str):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordEpisodeStatistics(env)
    env = RecordVideo(env, video_folder=video_path, episode_trigger=lambda ep: ep % 100 == 0, name_prefix="training", video_length=500)
    env.reset(seed=seed)
    return env

def create_single_atari_env(
    env_name: str,
    seed: int,
    reduce_action_set: int,
    frame_skip: int,
    frame_size: int,
    grayscale: bool,
    video_path: Optional[str],
    video_freq: int,
    video_length: int,
):
    """Create Atari env with preprocessing and optional video recording."""
    use_full_action_space = reduce_action_set in (0, 2)
    env = gym.make(
        env_name,
        obs_type="rgb",
        render_mode="rgb_array" if video_path else None,
        full_action_space=use_full_action_space,
    )

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=frame_skip)

    if reduce_action_set == 2:
        env = ActionSetWrapper(env, reduce_action_set, env_name)

    env = PreprocessWrapper(env, frame_size=frame_size, grayscale=grayscale)
    env = RecordEpisodeStatistics(env)

    if video_path:
        print(f"[Video] Recording every {video_freq} episodes to {video_path}")
        env = RecordVideo(
            env,
            video_folder=video_path,
            episode_trigger=lambda ep: ep % video_freq == 0,
            name_prefix="training",
            video_length=video_length,
        )

    env.reset(seed=seed)
    return env


def init_wandb(args, feature_dim: int):
    if args.disable_wandb:
        return None
    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name or f"swift-sarsa-ms-pacman-{int(time.time())}",
        config=vars(args) | {"feature_dim": feature_dim},
    )
    return run


def save_agent_state(agent: SwiftSarsaAgent, path: str, note: str = ""):
    # SwiftSarsa (pybind) is not picklable; skip saving to avoid noisy errors.
    return


def train_loop(env: gym.Env, backbone: FeatureBackbone, agent: SwiftSarsaAgent, args, paths: Dict[str, str]):
    def stack_features(feat_queue: deque) -> np.ndarray:
        return np.concatenate(list(feat_queue), axis=0)

    obs, info = env.reset(seed=args.seed)
    feat_queue: deque = deque(maxlen=args.frame_stack)
    t_inf = time.time()
    first_feat = backbone(obs)
    first_inf_time = time.time() - t_inf
    for _ in range(args.frame_stack):
        feat_queue.append(first_feat)
    stacked = stack_features(feat_queue)

    global_step = 0
    action, values, entropy, current_eps = agent.select_action(stacked, global_step)
    episode_reward = 0.0
    episode_len = 0
    episode_q_vals: List[float] = []
    episode_q_taken: List[float] = []
    episode_feat_norms: List[float] = []
    episode_entropies: List[float] = []
    episode_deltas: List[float] = []
    episode_start_time = time.time()
    best_reward = -float("inf")
    # Timing stats
    timing_window = deque(maxlen=1000)
    sarsa_timing = deque(maxlen=1000)
    inference_timing = deque(maxlen=1000)
    inference_timing.append(first_inf_time)
    start_wall = time.time()

    logger = []
    wandb_run = init_wandb(args, agent.feature_dim)

    while global_step < args.total_frames:
        t0 = time.time()
        next_obs, reward, terminated, truncated, info = env.step(action)
        env_step_time = time.time() - t0
        done = terminated or truncated
        episode_reward += reward
        episode_len += 1
        episode_q_vals.extend(values)
        episode_q_taken.append(float(values[action]))
        episode_feat_norms.append(float(np.linalg.norm(stacked)))
        episode_entropies.append(entropy)

        gamma = args.gamma if not done else 0.0
        t1 = time.time()
        agent.learn(stacked, reward, gamma, action)
        sarsa_timing.append(time.time() - t1)
        try:
            episode_deltas.append(float(agent.algo.get_last_delta()))
        except Exception:
            pass

        if done:
            elapsed = max(time.time() - episode_start_time, 1e-6)
            fps = episode_len / elapsed
            q_arr = np.array(episode_q_vals, dtype=np.float32) if episode_q_vals else np.array([0.0], dtype=np.float32)
            feat_arr = np.array(episode_feat_norms, dtype=np.float32) if episode_feat_norms else np.array([0.0], dtype=np.float32)
            ent_arr = np.array(episode_entropies, dtype=np.float32) if episode_entropies else np.array([0.0], dtype=np.float32)
            delta_arr = np.array(episode_deltas, dtype=np.float32) if episode_deltas else np.array([0.0], dtype=np.float32)
            q_min = float(q_arr.min()) if q_arr.size else 0.0
            feat_min = float(feat_arr.min()) if feat_arr.size else 0.0
            feat_max = float(feat_arr.max()) if feat_arr.size else 0.0
            ent_min = float(ent_arr.min()) if ent_arr.size else 0.0
            ent_max = float(ent_arr.max()) if ent_arr.size else 0.0
            delta_min = float(delta_arr.min()) if delta_arr.size else 0.0
            delta_max = float(delta_arr.max()) if delta_arr.size else 0.0
            q_taken = np.array(episode_q_taken, dtype=np.float32) if episode_q_taken else np.array([0.0], dtype=np.float32)
            q_taken_min = float(q_taken.min()) if q_taken.size else 0.0
            q_taken_max = float(q_taken.max()) if q_taken.size else 0.0
            q_taken_mean = float(q_taken.mean()) if q_taken.size else 0.0
            q_taken_std = float(q_taken.std()) if q_taken.size else 0.0
            avg_env = np.mean(timing_window) if timing_window else 0.0
            avg_sarsa = np.mean(sarsa_timing) if sarsa_timing else 0.0
            avg_infer = np.mean(inference_timing) if inference_timing else 0.0
            if wandb_run:
                wandb_log = {
                    "reward/episode_reward": episode_reward,
                    "reward/episode_len": episode_len,
                    "env/q_max": float(q_arr.max()),
                    "env/q_min": q_min,
                    "env/q_mean": float(q_arr.mean()),
                    "env/q_taken_mean": q_taken_mean,
                    "env/q_taken_min": q_taken_min,
                    "env/q_taken_max": q_taken_max,
                    "env/policy_entropy_mean": float(ent_arr.mean()),
                    "env/policy_entropy_min": ent_min,
                    "env/policy_entropy_max": ent_max,
                    "env/td_error_mean": float(delta_arr.mean()),
                    "env/td_error_min": delta_min,
                    "env/td_error_max": delta_max,
                    "exploration/epsilon_greedy": current_eps,
                }
                wandb_log.update(collect_swiftsarsa_stats(agent))
                wandb_run.log(wandb_log)
            logger.append((global_step, episode_reward, episode_len))
            # Checkpointing disabled (pybind object not picklable)
            episode_reward = 0.0
            episode_len = 0
            episode_q_vals.clear()
            episode_feat_norms.clear()
            episode_entropies.clear()
            episode_deltas.clear()
            episode_start_time = time.time()
            next_obs, info = env.reset()
            feat_queue.clear()
            t_inf = time.time()
            reset_feat = backbone(next_obs)
            inference_timing.append(time.time() - t_inf)
            for _ in range(args.frame_stack):
                feat_queue.append(reset_feat)
            stacked = stack_features(feat_queue)
            action, values, entropy, current_eps = agent.select_action(stacked, global_step)
        else:
            t_inf = time.time()
            new_feat = backbone(next_obs)
            inference_timing.append(time.time() - t_inf)
            feat_queue.append(new_feat)
            stacked = stack_features(feat_queue)
            action, values, entropy, current_eps = agent.select_action(stacked, global_step)

        global_step += 1
        timing_window.append(env_step_time)

        if global_step % 1000 == 0:
            avg_env = np.mean(timing_window) if timing_window else 0.0
            avg_sarsa = np.mean(sarsa_timing) if sarsa_timing else 0.0
            avg_infer = np.mean(inference_timing) if inference_timing else 0.0
            fps_wall = global_step / max(time.time() - start_wall, 1e-6)
            recent_rewards = [entry[1] for entry in logger[-5:]]
            avg_reward_5 = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            print(
                f"[stats] step={global_step} "
                f"avg_env_step_ms={avg_env*1000:.3f} "
                f"avg_reward_5ep={avg_reward_5:.2f} "
                f"avg_len={episode_len:.2f} "
                f"feature_norm={episode_feat_norms[-1] if episode_feat_norms else 0:.4f}"
            )
    if wandb_run:
        wandb_run.finish()
    return logger


def build_backbone(args, sample_obs: np.ndarray) -> FeatureBackbone:
    if args.backbone.startswith("rfdetr_"):
        model_name = args.backbone.split("_", 1)[1]
        return RFDetrBackbone(model_name=model_name, device=args.device, frame_size=args.frame_size)
    if args.backbone.startswith("resnet_"):
        return ResNetBackbone(model_name="resnet_18", device=args.device, frame_size=args.frame_size)
    raise ValueError(f"Unknown backbone option {args.backbone}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Swift-Sarsa on Ms. Pacman with transformer features (simulation).")
    parser.add_argument("--env_name", type=str, default="ALE/MsPacman-v5")
    parser.add_argument("--game_name", type=str, default="ms_pacman")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_frames", type=int, default=200_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--frame_stack", type=int, default=2)
    parser.add_argument("--frame_size", type=int, default=64)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--noop_max", type=int, default=30)
    parser.add_argument("--reduce_action_set", type=int, default=2)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--video_freq", type=int, default=250)
    parser.add_argument("--video_length", type=int, default=500)
    parser.add_argument("--no_videos", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs/swift_sarsa_transformer")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--checkpoint_freq", type=int, default=50000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--wandb_project", type=str, default="swift-sarsa-transformer")
    parser.add_argument("--disable_wandb", action="store_true")

    # Swift-Sarsa hyperparameters
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=1e-7)
    parser.add_argument("--meta_step_size", type=float, default=1e-3)
    parser.add_argument("--decay", type=float, default=0.999)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--eta_min", type=float, default=1e-8)
    parser.add_argument("--exploration", type=str, choices=["epsilon_greedy", "softmax"], default="softmax")
    parser.add_argument("--eps_greedy_start", type=float, default=1.0)
    parser.add_argument("--eps_greedy_end", type=float, default=0.05)
    parser.add_argument("--eps_greedy_end_timestamp", type=int, default=100000)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--softmax_temp", type=float, default=.1)

    # Backbone
    parser.add_argument("--backbone", type=str, default="rfdetr_nano", help="Options: rfdetr_nano/small/medium/large or resnet_18.")

    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_slug = args.run_name or f"{timestamp}-swift-sarsa-sim"
    env_dir = args.env_name.replace("/", "_")
    experiment_dir = os.path.join(args.output_dir, "sim", env_dir, run_slug)

    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "models", "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "models", "best_model"), exist_ok=True)

    config_path = os.path.join(experiment_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(json.dumps(vars(args), indent=2))

    if not args.no_videos and args.video_path is None:
        args.video_path = os.path.join(experiment_dir, "videos")
    if args.env_name == "CartPole-v1":
        env = create_cartpole_env(args.seed, args.video_path)
    else:
        env = create_single_atari_env(
            env_name=args.env_name,
            seed=args.seed,
            reduce_action_set=args.reduce_action_set,
            frame_skip=args.frame_skip,
            frame_size=args.frame_size,
            grayscale=args.grayscale,
            video_path=None if args.no_videos else args.video_path,
            video_freq=args.video_freq,
            video_length=args.video_length,
        )
    sample_obs, _ = env.reset(seed=args.seed)
    if args.env_name == "CartPole-v1":
        backbone = CartPoleIdentityBackbone(env.observation_space)
    else:
        backbone = build_backbone(args, sample_obs)
    agent_cfg = SwiftSarsaConfig(
        lambda_=args.lambda_,
        alpha=args.alpha,
        meta_step_size=args.meta_step_size,
        eta=args.eta,
        decay=args.decay,
        epsilon=args.epsilon,
        eta_min=args.eta_min,
        exploration=args.exploration,
        eps_greedy_start=args.eps_greedy_start,
        eps_greedy_end=args.eps_greedy_end,
        eps_greedy_end_timestamp=args.eps_greedy_end_timestamp,
        softmax_temp=args.softmax_temp,
    )
    num_actions = env.action_space.n
    # assert num_actions == 2
    agent = SwiftSarsaAgent(num_actions=num_actions, feature_dim=backbone.feature_dim * args.frame_stack, cfg=agent_cfg)

    print(f"Starting training: total_frames={args.total_frames}, actions={num_actions}, feature_dim={backbone.feature_dim * args.frame_stack}")
    paths = {
        "experiment_dir": experiment_dir,
        "checkpoints": os.path.join(experiment_dir, "models", "checkpoints"),
        "best_model": os.path.join(experiment_dir, "models", "best_model", "best.pkl"),
        "final_model": os.path.join(experiment_dir, "models", "final_model.pkl"),
    }
    train_loop(env, backbone, agent, args, paths)


if __name__ == "__main__":
    main()
