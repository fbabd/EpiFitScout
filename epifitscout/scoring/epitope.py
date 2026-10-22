"""The epitope: which antigen residues a retrieved loop is scored against.

The caller may name the epitope outright. When they do not, it is taken as the
antigen residues within :data:`EPITOPE_CUTOFF` of the template's own CDR, which
is what the tool's original specification asked for -- the user marks the CDR
and the epitope, and the epitope is found by proximity when unmarked.

A note on what this does and does not use
-----------------------------------------
Finding the epitope by proximity reads the template's native CDR coordinates.
That is legitimate for the task: the input is a *bound* complex and the question
is which loops fit the site this antibody already binds, so the site is given.
It is **not** legitimate to let those coordinates reach the search -- the query
is still built from the framework stems alone, and nothing here is passed to
MASTER. Keep the distinction when measuring conformational recovery: the
epitope tells a candidate where to point, so a benchmark that calls the
retrieved conformation a blind prediction is overstating it.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["EPITOPE_CUTOFF", "MIN_EPITOPE_RESIDUES", "epitope_ca"]

#: Antigen residues with any heavy atom this close to the CDR are the epitope, Å.
#: The tight cutoff: only residues actually in contact, rather than the whole
#: facing patch.
EPITOPE_CUTOFF = 4.5

#: The shape scorer needs a profile, and a profile needs a curve. Below four
#: CA atoms the torsion term is undefined and the depth term is two points and
#: a slope, so a smaller epitope is reported as unusable rather than scored.
MIN_EPITOPE_RESIDUES = 4


def epitope_ca(
    loop_coords: np.ndarray,
    antigen_coords: np.ndarray,
    antigen_residue_index: np.ndarray,
    antigen_is_ca: np.ndarray,
    cutoff: float = EPITOPE_CUTOFF,
) -> np.ndarray:
    """CA coordinates of the antigen residues nearest the loop.

    Args:
        loop_coords: ``(L, 3)`` heavy-atom coordinates of the CDR.
        antigen_coords: ``(A, 3)`` antigen heavy-atom coordinates.
        antigen_residue_index: ``(A,)`` integer residue id per antigen atom.
        antigen_is_ca: ``(A,)`` bool mask marking the CA atom of each residue.
        cutoff: Contact distance, Å.

    Returns:
        ``(M, 3)`` CA coordinates, ordered by residue index so the profile runs
        along the antigen chain rather than in discovery order. Empty when the
        antigen or the loop has no atoms.
    """
    if not len(loop_coords) or not len(antigen_coords):
        return np.empty((0, 3))

    from scipy.spatial import cKDTree

    tree = cKDTree(loop_coords)
    near = tree.query_ball_point(antigen_coords, cutoff)
    contact_residues = {
        int(antigen_residue_index[i]) for i, hits in enumerate(near) if hits
    }
    if not contact_residues:
        return np.empty((0, 3))

    ca_idx = np.flatnonzero(antigen_is_ca)
    keep = [i for i in ca_idx if int(antigen_residue_index[i]) in contact_residues]
    if not keep:
        return np.empty((0, 3))

    keep.sort(key=lambda i: int(antigen_residue_index[i]))
    return np.asarray(antigen_coords[np.asarray(keep)])
