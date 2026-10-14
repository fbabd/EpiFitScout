"""EpitopeSurface: the antigen residues a CDR site actually faces.

The epitope used to be a fixed sequence window (``centre ± 8 residues``), which
has two problems: it is contiguous, while real epitopes are often patches of
surface drawn from separate parts of the chain, and it is chosen by residue
number rather than by proximity, so most of it can sit 10-19 Å from the loop
with nothing to complement.

This type instead holds whichever antigen residues are *spatially* near the CDR
site, as all-atom point sets, in no particular order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["EpitopeSurface"]


@dataclass(frozen=True)
class EpitopeSurface:
    """Antigen residues near a CDR site, kept as all-atom point sets.

    Attributes:
        chain: Antigen chain identifier.
        residue_numbers: PDB residue number of each selected residue.
        residue_names: Three-letter name of each selected residue.
        atom_coords: ``(n_atoms, 3)`` every atom of every selected residue.
        atom_residue: ``(n_atoms,)`` index into ``residue_numbers`` saying which
            residue each atom belongs to.
        body_coords: ``(n_body_atoms, 3)`` every atom of the whole antigen
            chain, not just the selected residues. Used only to orient outward
            surface normals: the selected patch is a thin shell with no
            interior, so a normal estimated from the patch alone points along
            the surface rather than away from the protein.
    """

    chain: str
    residue_numbers: tuple[int, ...]
    residue_names: tuple[str, ...]
    atom_coords: np.ndarray = field(repr=False)
    atom_residue: np.ndarray = field(repr=False)
    body_coords: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if self.atom_coords.ndim != 2 or self.atom_coords.shape[1] != 3:
            raise ValueError(f"atom_coords must be (A, 3), got {self.atom_coords.shape}")
        if len(self.atom_residue) != len(self.atom_coords):
            raise ValueError("atom_residue must have one entry per atom")
        if len(self.residue_numbers) != len(self.residue_names):
            raise ValueError("residue_numbers and residue_names must agree in length")
        if self.body_coords.ndim != 2 or self.body_coords.shape[1] != 3:
            raise ValueError(f"body_coords must be (A, 3), got {self.body_coords.shape}")

    @property
    def n_residues(self) -> int:
        return len(self.residue_numbers)

    @property
    def n_atoms(self) -> int:
        return len(self.atom_coords)

    def residue_centroids(self) -> np.ndarray:
        """``(n_residues, 3)`` mean atom position of each residue."""
        return np.stack([
            self.atom_coords[self.atom_residue == i].mean(axis=0)
            for i in range(self.n_residues)
        ])

    @classmethod
    def from_fragment(cls, fragment: Any) -> EpitopeSurface:
        """Build a surface from a backbone-only :class:`Fragment`.

        A fallback for callers that still describe the epitope as a fragment.
        The result carries **no side chains**, so contact distances come out
        larger than they should and outward normals are poor: a loop traced as
        a curve has no interior to point away from. Prefer
        :meth:`epifitscout.query.builder.QueryBuilder.get_epitope_surface`,
        which reads every atom from the structure.
        """
        coords = fragment.coords
        lo = getattr(fragment, "residue_range", (1, len(coords)))[0]
        chain = getattr(fragment, "chain", "?") or "?"
        residues = [(lo + i, "UNK", coords[i]) for i in range(len(coords))]
        return cls.from_residues(chain, residues)

    @classmethod
    def from_residues(
        cls,
        chain: str,
        residues: list[tuple[int, str, np.ndarray]],
        body: np.ndarray | None = None,
    ) -> EpitopeSurface:
        """Build from ``(residue_number, residue_name, coords)`` triples.

        Args:
            chain: Antigen chain identifier.
            residues: The selected epitope residues.
            body: All atoms of the whole antigen chain. Defaults to the selected
                atoms, which makes normals unreliable; pass the full chain.
        """
        if not residues:
            raise ValueError("cannot build an EpitopeSurface from no residues")
        numbers = tuple(int(n) for n, _, _ in residues)
        names = tuple(str(nm) for _, nm, _ in residues)
        coords = np.concatenate([c for _, _, c in residues], axis=0)
        index = np.concatenate([
            np.full(len(c), i, dtype=np.int32) for i, (_, _, c) in enumerate(residues)
        ])
        return cls(
            chain=chain, residue_numbers=numbers, residue_names=names,
            atom_coords=coords.astype(np.float64), atom_residue=index,
            body_coords=(coords if body is None else body).astype(np.float64),
        )
