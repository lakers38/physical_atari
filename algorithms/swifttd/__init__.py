"""
SwiftTD Actor-Critic for Atari

SwiftTD is an algorithm for temporal difference learning with adaptive step sizes.
This implementation provides both single-env and vectorized actor-critic agents.
"""

from .model import CNNFeatureExtractor
from .actor_critic import ActorCriticSwiftTD, PolicyHead

__all__ = ['CNNFeatureExtractor', 'ActorCriticSwiftTD', 'PolicyHead']
