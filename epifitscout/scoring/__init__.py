"""Scoring: the selector, and the measurements made on what it selects.

:mod:`~epifitscout.scoring.shape`
    How much of a placed loop sits cleanly against the epitope, counted from
    backbone distances. **This is what ranks candidates**, and it is the whole
    of the selection.
:mod:`~epifitscout.scoring.interface`
    Buried area, contacts, clashes, hydrogen bonds, salt bridges and trapped
    void, measured all-atom on the models that are reported. Output and
    evidence — none of it ranks.
:mod:`~epifitscout.scoring.sasa`
    Shrake–Rupley surface area and the interface void, from first principles.
"""

from __future__ import annotations

from epifitscout.scoring.epitope import epitope_ca
from epifitscout.scoring.interface import InterfaceScore, InterfaceScorer
from epifitscout.scoring.shape import ShapeComplementarityScorer, ShapeScore

__all__ = [
    "InterfaceScore", "InterfaceScorer",
    "ShapeComplementarityScorer", "ShapeScore", "epitope_ca",
]
