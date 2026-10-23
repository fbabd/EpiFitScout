"""GraftCandidate: one retrieved loop, carried through the stages that judge it.

A candidate accumulates evidence in a fixed order, and each stage is allowed to
see only what the earlier ones produced:

1. **retrieval** — the loop, its donor stems, and the transform that lays those
   stems onto the template's. Set at construction.
2. **feasibility** — can this loop be bonded into the framework at all
   (:mod:`epifitscout.graft.feasibility`). Candidates that fail are dropped, not
   scored.
3. **interface** — what the packed, all-atom graft does against the epitope
   (:mod:`epifitscout.build.interface`). Measured only for the models that
   are reported, and never used to choose them.
4. **ranking** — sorted on that fit, clash-free models first.

Nothing here refers to the native loop. That is deliberate: the native loop is
the thing a benchmark asks the method to predict, so a candidate that knew about
it could not be ranked honestly. It appears only in evaluation, downstream of
every decision this object records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from epifitscout.types.fragment import Fragment
from epifitscout.types.superposition import Superposition

__all__ = ["GraftCandidate"]


@dataclass
class GraftCandidate:
    """A database loop proposed as a graft, with its placement and scores.

    Attributes:
        loop: The candidate loop, in its **donor's** crystal frame. These are
            the residues that fell between the matched stems.
        n_stem: Donor residues matching the template's N-side stem.
        c_stem: Donor residues matching the template's C-side stem.
        superposition: Transform from the donor frame into the template frame,
            obtained by least-squares fit of the donor stems onto the
            template's. Apply it to place any donor coordinate at the CDR site.
        master_rmsd: The RMSD MASTER reported for the stem match, in Å.
        stem_rmsd_ca: CA RMSD of the stems after the fit. Reproduces
            ``master_rmsd`` and so doubles as a check that the residues reloaded
            from the database are the ones MASTER aligned.
        stem_rmsd_bb: Full-backbone (N, CA, C, O) RMSD of the stems after the
            fit — the residual of the transform actually used.
        n_collide: Set by the selection stage; loop backbone atoms inside the
            antigen. Ranks above :attr:`shape`, because a loop in the wrong
            place is not a candidate however well its profile matches.
        shape: Set by the selection stage; a
            :class:`~epifitscout.scoring.shape.ShapeScore` — how well the placed
            loop's backbone complements the epitope. This is the only thing the
            ranking sorts on, and it is computed before any packing.
        feasibility: Set by stage 2; a
            :class:`~epifitscout.graft.feasibility.GraftFeasibility`.
        interface: Set after packing; an
            :class:`~epifitscout.build.interface.InterfaceScore`. A
            measurement of the chosen model, reported but never ranked on.
        model_path: Where the packed all-atom model was written, if it was kept.
        metadata: Free-form annotation.
    """

    loop: Fragment
    n_stem: Fragment
    c_stem: Fragment
    superposition: Superposition
    master_rmsd: float
    stem_rmsd_ca: float
    stem_rmsd_bb: float
    feasibility: Any = None
    shape: Any = None
    n_collide: int = 0
    interface: Any = None
    model_path: Any = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #

    @property
    def donor(self) -> str:
        """``PDBID_CHAIN`` of the structure this loop was taken from."""
        return f"{self.loop.pdb_id}_{self.loop.chain}"

    @property
    def source(self) -> str:
        """Donor and the loop's residue range within it."""
        lo, hi = self.loop.residue_range
        return f"{self.donor}:{lo}-{hi}"

    @property
    def length(self) -> int:
        """Residues in the candidate loop."""
        return self.loop.length

    @property
    def sequence(self) -> str:
        """One-letter sequence of the candidate loop."""
        return self.loop.sequence

    # ------------------------------------------------------------------ #
    # Placement
    # ------------------------------------------------------------------ #

    @property
    def placed_loop(self) -> np.ndarray:
        """``(N, 4, 3)`` loop backbone in the template frame, ready to graft."""
        return self.superposition.apply(self.loop.coords)

    @property
    def placed_stems(self) -> np.ndarray:
        """``(S, 4, 3)`` both donor stems in the template frame, N-to-C.

        Not grafted — the template keeps its own framework. Used to judge how
        well the fit landed and to close the junctions.
        """
        return self.superposition.apply(
            np.concatenate([self.n_stem.coords, self.c_stem.coords], axis=0)
        )

    def place(self, coords: np.ndarray) -> np.ndarray:
        """Apply this candidate's transform to arbitrary donor coordinates."""
        return self.superposition.apply(coords)

    # ------------------------------------------------------------------ #
    # Stage bookkeeping
    # ------------------------------------------------------------------ #

    @property
    def is_feasible(self) -> bool:
        """True once stage 2 has run and the graft can be bonded in."""
        return bool(getattr(self.feasibility, "passed", False))

    def __repr__(self) -> str:
        parts = [
            f"GraftCandidate({self.source}",
            f"len={self.length}",
            f"master_rmsd={self.master_rmsd:.3f}",
        ]
        if self.feasibility is not None:
            parts.append("feasible" if self.is_feasible else "infeasible")
        if self.shape is not None:
            parts.append(f"shape={self.shape.score:.3f}")
        if self.n_collide:
            parts.append(f"collide={self.n_collide}")
        return ", ".join(parts) + ")"
