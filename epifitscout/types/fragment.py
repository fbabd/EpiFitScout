"""Fragment and ScoredHit dataclasses — canonical data representations for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Fragment:
    """A backbone structural fragment extracted from a protein structure.

    coords shape: (N, 4, 3) — N residues × [N, CA, C, O] × [x, y, z].
    Residue range is 1-indexed, inclusive, matching PDB residue sequence numbers.
    """

    pdb_id: str
    chain: str
    residue_range: tuple[int, int]
    coords: np.ndarray = field(repr=False, compare=False, hash=False)
    sequence: str = ""
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.coords.ndim != 3 or self.coords.shape[1] != 4 or self.coords.shape[2] != 3:
            raise ValueError(
                f"coords must be (N, 4, 3), got {self.coords.shape}"
            )

    def __hash__(self) -> int:
        return hash((self.pdb_id, self.chain, self.residue_range))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fragment):
            return NotImplemented
        return (
            self.pdb_id == other.pdb_id
            and self.chain == other.chain
            and self.residue_range == other.residue_range
        )

    @property
    def length(self) -> int:
        """Number of residues in this fragment."""
        return self.coords.shape[0]

    @property
    def ca_coords(self) -> np.ndarray:
        """CA coordinates only, shape (N, 3)."""
        return self.coords[:, 1, :]

    @property
    def ca_centroid(self) -> np.ndarray:
        """Mean CA position, shape (3,)."""
        return self.ca_coords.mean(axis=0)

    @property
    def cdr_type(self) -> str | None:
        """CDR type if annotated (e.g. 'H3', 'L1'), else None."""
        return self.metadata.get("cdr_type")

    @property
    def canonical_form(self) -> str | None:
        """CDR canonical form if annotated, else None."""
        return self.metadata.get("canonical_form")


@dataclass
class ScoredHit:
    """A fragment hit with all pipeline scores attached."""

    fragment: Fragment
    superposition: Any          # Superposition — forward ref avoids circular import
    rmsd_cdr: float
    complementarity_score: float
    final_score: float

    def __repr__(self) -> str:
        return (
            f"ScoredHit(pdb={self.fragment.pdb_id}:{self.fragment.chain} "
            f"res={self.fragment.residue_range} "
            f"cdr={self.fragment.cdr_type} "
            f"rmsd={self.rmsd_cdr:.3f} "
            f"comp={self.complementarity_score:.3f} "
            f"score={self.final_score:.4f})"
        )
