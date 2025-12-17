"""
Soft Actor-Critic style agents for Atari with shared CNN encoders and MLP
policy/value heads. Includes single-env and vectorized variants.
"""

from .sac import SACAgent

__all__ = ["SACAgent"]
