"""Grafting: placing a retrieved loop into the template, and judging the join.

Three steps, in the order the pipeline runs them:

:mod:`~epifitscout.graft.feasibility`
    Can the loop be bonded in at all? Junction geometry on backbone
    coordinates, cheap enough for every retrieved candidate, and a hard gate —
    an unclosable graft is never scored.
:mod:`~epifitscout.build.assemble`
    Write the model and rebuild its side chains. Produces the all-atom
    structure the interface measurement needs.
:mod:`~epifitscout.build.packer`
    Driving an external packer (FASPR), and checking what it returns.
"""

from epifitscout.build.assemble import AssembledModel, TemplateSite, write_graft
from epifitscout.graft.feasibility import GraftFeasibility, assess, assess_all

__all__ = [
    "AssembledModel",
    "GraftFeasibility",
    "TemplateSite",
    "assess",
    "assess_all",
    "write_graft",
]
