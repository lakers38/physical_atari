"""
SwiftTD Q-Learning for Atari

SwiftTD is an algorithm for temporal difference learning with adaptive step sizes.
This implementation uses 18 separate SwiftTD learners (one per action) for Q-learning.
"""

from .agent import SwiftTDAgent
from .model import CNNFeatureExtractor
from . import config

__all__ = ['SwiftTDAgent', 'CNNFeatureExtractor', 'config']
