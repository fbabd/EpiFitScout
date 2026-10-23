"""Scoring that decides — stages 5 to 7, on backbone coordinates only.

:mod:`~epifitscout.scoring.epitope`
    Which antigen residues a candidate is judged against.
:mod:`~epifitscout.scoring.shape`
    Depth-profile and torsion complementarity to that epitope. This ranks.
:mod:`~epifitscout.scoring.collision`
    Loop backbone inside the antigen. Shape has no steric content and is
    maximised by pressing into the surface, so it needs this counterweight.

Everything here runs before anything is built, which is what makes the
all-atom measurements in :mod:`epifitscout.build` independent evidence rather
than a restatement of the objective.
"""

from epifitscout.scoring.collision import COLLISION, count_collisions
from epifitscout.scoring.epitope import EPITOPE_CUTOFF, epitope_ca
from epifitscout.scoring.shape import ShapeComplementarityScorer, ShapeScore

__all__ = [
    "COLLISION",
    "EPITOPE_CUTOFF",
    "ShapeComplementarityScorer",
    "ShapeScore",
    "count_collisions",
    "epitope_ca",
]
