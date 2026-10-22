"""Does the placed loop pass through the antigen?

The shape score answers how well a loop's backbone profile matches the
epitope's, and that question has no steric content at all: the correlation it
maximises is *best* when the loop is pressed into the surface, because that is
when the two profiles agree most closely. Ranked on shape alone, the loops that
score highest are reliably the ones furthest inside the antigen -- measured over
17 targets, 47 of 85 returned models carried more than twenty clashes.

So shape needs a counterweight, and this is it: a count of backbone atoms that
have no business being where they are. It gates nothing on its own. The pipeline
sorts clean candidates above colliding ones and lets shape decide within each
group, which orders without ever emptying a pool -- requiring a clean candidate
outright was tried in v2 and cost 13 of 101 targets their only answer.

Backbone only
-------------
A retrieved fragment carries backbone coordinates; its side chains are built
later by the packer. So this sees roughly the 39% of real clashes that involve
the loop's own backbone, and is blind to the side-chain clashes that make up the
rest. It is a check that the loop is in a plausible *place*, not that the model
is clash-free.
"""

from __future__ import annotations

import numpy as np

__all__ = ["COLLISION", "count_collisions"]

#: Backbone-to-antigen distance below which the two interpenetrate, Å.
#: Calibrated on 86 native antibody–antigen complexes: the closest any native
#: CDR backbone comes to its antigen is 2.12 Å, so nothing real is rejected
#: here. 2.5 Å was tried and is wrong -- it rejects 12% of those natives.
COLLISION = 2.0


def count_collisions(
    placed_loop: np.ndarray,
    antigen_coords: np.ndarray,
    cutoff: float = COLLISION,
) -> int:
    """Loop-backbone atoms closer to the antigen than any real interface gets.

    Args:
        placed_loop: ``(N, 4, 3)`` or ``(N, 3)`` loop backbone, already in the
            template frame.
        antigen_coords: ``(A, 3)`` antigen heavy atoms.
        cutoff: Interpenetration distance, Å.

    Returns:
        How many loop backbone atoms lie within ``cutoff`` of any antigen atom.
        Atoms, not pairs: one atom buried against six others is one problem,
        and counting pairs would let a single deep atom outweigh a broadly
        misplaced loop.
    """
    if not len(placed_loop) or not len(antigen_coords):
        return 0

    from scipy.spatial import cKDTree

    coords = placed_loop.reshape(-1, 3)
    distances, _ = cKDTree(antigen_coords).query(coords, k=1)
    return int(np.count_nonzero(distances < cutoff))
