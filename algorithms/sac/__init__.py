"""
Soft Actor-Critic style agents for Atari with shared CNN encoders and MLP
policy/value heads. Includes single-env and vectorized variants.
"""

from .agent_actor_critic import SwiftTDAgent
from .agent_actor_critic_vec import SwiftTDAgent as SwiftTDAgentVec

__all__ = ["SwiftTDAgent", "SwiftTDAgentVec"]
