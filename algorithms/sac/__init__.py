"""
Soft Actor-Critic style agents for Atari with shared CNN encoders and MLP
policy/value heads. Includes single-env and vectorized variants.
"""

from .agent_actor_critic import SACAgent
from .agent_actor_critic_vec import SACAgent as SwiftTDAgentVec

__all__ = ["SACAgent", "SwiftTDAgentVec"]
