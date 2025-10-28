from __future__ import annotations
from typing import Protocol, Iterable, Dict
import numpy as np


class VectorAgent(Protocol):
    def reset(self, num_envs: int) -> None: ...

    def act_(self, observations: np.ndarray) -> np.ndarray: ...

    def observe(
        self,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
        infos: Iterable[Dict],
    ) -> None: ...

    def train_step(self) -> None: ...

    def save_model(self, path: str) -> None: ...

    def load_model(self, path: str) -> None: ...
