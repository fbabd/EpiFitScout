"""Superposition dataclass: rigid-body rotation + translation from MASTER alignment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Superposition:
    """Rigid-body transform that aligns a hit fragment onto the query fragment.

    Rotation and translation are stored as they come from MASTER's output (or
    from the Kabsch algorithm): to map hit coords into the query frame, apply:
        aligned = coords @ rotation.T + translation
    """

    rotation: np.ndarray = field(repr=False, compare=False, hash=False)
    translation: np.ndarray = field(repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.rotation.shape != (3, 3):
            raise ValueError(f"rotation must be (3, 3), got {self.rotation.shape}")
        if self.translation.shape != (3,):
            raise ValueError(f"translation must be (3,), got {self.translation.shape}")

    def apply(self, coords: np.ndarray) -> np.ndarray:
        """Apply rotation and translation to coordinate array.

        Args:
            coords: (N, 3) or (N, K, 3) coordinate array.

        Returns:
            Transformed coordinates, same shape as input.
        """
        if coords.ndim == 2:
            # (N, 3) → straightforward
            return coords @ self.rotation.T + self.translation
        elif coords.ndim == 3:
            # (N, K, 3) — apply to last dimension, broadcast over K
            return np.einsum("nkj,ij->nki", coords, self.rotation) + self.translation
        else:
            raise ValueError(f"coords must be 2D or 3D, got {coords.ndim}D")

    def inverse(self) -> "Superposition":
        """Return the inverse transform (maps query frame back to hit frame)."""
        r_inv = self.rotation.T
        t_inv = -(self.translation @ self.rotation)
        return Superposition(rotation=r_inv, translation=t_inv)

    @classmethod
    def identity(cls) -> "Superposition":
        """Return the identity transform (no rotation, no translation)."""
        return cls(rotation=np.eye(3), translation=np.zeros(3))
