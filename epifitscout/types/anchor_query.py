"""AnchorQuery: the two framework stems flanking a CDR site, and the gap between them.

Why the query is not the loop
-----------------------------
EpiFitScout v1 searched with the CDR loop itself. That has two consequences, and
they are the reason this type exists.

*It is the answer.* Every hit is then within the RMSD cutoff of the native loop
by construction, so CA RMSD, DockQ, Fnat and iRMSD measure the query rather than
the prediction. Nothing downstream can repair that.

*It is unnecessary.* The loop's placement in a graft is not free. The retrieved
loop is attached to the template's framework at both ends, and the framework is
already in the same frame as the antigen, so the rigid-body transform that puts
the loop at the CDR site is fully determined by the stems. Querying the stems
asks the database the question the graft actually poses — *which real loops can
bridge this gap in this geometry* — and asks it without ever looking at the
answer.

It also lifts the length ceiling. A single-segment query requires the whole loop
to superpose within the cutoff, and the number of real loops meeting that
condition collapses as the loop lengthens (v1 returned nothing for 10 of 101
complexes, all CDR-H3 ≥ 15). A stem query requires only the stems to superpose,
so retrieval depth is set by framework geometry, which does not vary with loop
length.

What MASTER does with it
------------------------
MASTER accepts a query of disjoint segments and ``--gapLen`` constrains the
number of residues that map between adjacent segments in a match. Segments are
read from the query PDB in N-to-C order, and a discontinuity in residue
numbering is enough to separate them — :meth:`AnchorQuery.to_pdb_string` writes
both stems in one chain with the loop's numbering left out.

A match is therefore a database chain carrying two stems whose geometry
reproduces this one, separated by exactly ``gap_length`` residues. Those gap
residues are the candidate loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from epifitscout.types.fragment import Fragment

__all__ = ["AnchorQuery"]

#: Residue-number offset placed between the two stems when the query is written
#: as PDB. Any gap separates the segments; the real loop length is passed to
#: MASTER through ``--gapLen`` and need not be encoded in the numbering.
_WRITE_GAP = 50


@dataclass(frozen=True)
class AnchorQuery:
    """A CDR site described by its flanking framework stems.

    Attributes:
        pdb_id: Template identifier, for labelling only.
        chain: Antibody chain the CDR sits on.
        n_stem: Framework residues immediately before the loop, N-to-C.
        c_stem: Framework residues immediately after the loop, N-to-C.
        gap_length: Residues between the stems, i.e. the loop length to retrieve.
        loop_range: ``(first, last)`` residue numbers of the loop in the
            template, the span a graft replaces.
        metadata: Free-form annotation.
    """

    pdb_id: str
    chain: str
    n_stem: Fragment
    c_stem: Fragment
    gap_length: int
    loop_range: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.gap_length < 1:
            raise ValueError(f"gap_length must be positive, got {self.gap_length}")
        if self.n_stem.length < 3 or self.c_stem.length < 3:
            raise ValueError(
                "each stem needs at least 3 residues: the graft transform is a "
                "least-squares fit to the stem backbone, and fewer than 3 "
                "residues per stem leaves the rotation poorly determined "
                f"(got {self.n_stem.length} and {self.c_stem.length})"
            )

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #

    @property
    def n_segments(self) -> int:
        return 2

    @property
    def stem_coords(self) -> np.ndarray:
        """``(n_stem + c_stem, 4, 3)`` backbone atoms of both stems, N-to-C."""
        return np.concatenate([self.n_stem.coords, self.c_stem.coords], axis=0)

    @property
    def stem_ca(self) -> np.ndarray:
        """``(n_stem + c_stem, 3)`` CA atoms of both stems, N-to-C.

        This is the atom set MASTER reports its RMSD over, so it is also what
        reproduces that number as a consistency check.
        """
        return self.stem_coords[:, 1, :]

    @property
    def takeoff(self) -> np.ndarray:
        """``(3,)`` C atom of the last N-stem residue: where the loop starts."""
        return self.n_stem.coords[-1, 2, :]

    @property
    def landing(self) -> np.ndarray:
        """``(3,)`` N atom of the first C-stem residue: where the loop ends."""
        return self.c_stem.coords[0, 0, :]

    @property
    def span(self) -> float:
        """Distance in Å the loop must bridge, take-off C to landing N.

        The quantity that decides whether a loop of this length can close at
        all: an extended chain covers roughly 3.3 Å per residue, so a span much
        beyond ``3.3 * gap_length`` is unbridgeable and one far below it needs
        the loop to double back.
        """
        return float(np.linalg.norm(self.landing - self.takeoff))

    @property
    def span_per_residue(self) -> float:
        """:attr:`span` divided by :attr:`gap_length`, in Å."""
        return self.span / self.gap_length

    def describe(self) -> str:
        """One-line summary."""
        return (
            f"{self.pdb_id}:{self.chain} loop {self.loop_range[0]}-"
            f"{self.loop_range[1]} ({self.gap_length} res), stems "
            f"{self.n_stem.residue_range[0]}-{self.n_stem.residue_range[1]} and "
            f"{self.c_stem.residue_range[0]}-{self.c_stem.residue_range[1]}, "
            f"span {self.span:.1f} Å ({self.span_per_residue:.2f} Å/res)"
        )

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_pdb_string(self) -> str:
        """Both stems as one PDB chain with a numbering gap between them.

        MASTER reads disjoint segments in the order they appear in the file and
        ``--gapLen`` fixes that order N-to-C, so the N-stem is written first.
        Verified against the two-chain alternative: both give byte-identical
        match lists.
        """
        from epifitscout.utils.io import fragment_to_pdb_string

        chain = self.chain or "X"
        first = fragment_to_pdb_string(
            self.n_stem.coords, sequence=self.n_stem.sequence,
            chain_id=chain, start_resnum=1,
        )
        second = fragment_to_pdb_string(
            self.c_stem.coords, sequence=self.c_stem.sequence,
            chain_id=chain, start_resnum=self.n_stem.length + _WRITE_GAP,
        )
        body = [ln for ln in first.splitlines() if ln.startswith("ATOM")]
        body += [ln for ln in second.splitlines() if ln.startswith("ATOM")]
        body.append("END")
        return "\n".join(body)

    def gap_len_argument(self, tolerance: int = 0) -> str:
        """The ``--gapLen`` value for this query.

        Args:
            tolerance: Residues of slack either side of :attr:`gap_length`.
                Zero — the default — retrieves only loops of exactly the length
                the graft needs, which is what keeps the replaced span and the
                retrieved loop one-to-one. Non-zero returns loops that would
                lengthen or shorten the CDR, which is a different experiment and
                needs the template span adjusted to match.
        """
        if tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")
        if tolerance == 0:
            return str(self.gap_length)
        lo = max(1, self.gap_length - tolerance)
        return f"{lo}-{self.gap_length + tolerance}"
