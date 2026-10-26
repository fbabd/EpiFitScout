"""Retrieve, score and rank real CDR loops for one site — stages 1 to 8.

EpiFitScout proper. Ranks on backbone shape complementarity to the epitope,
with candidates whose backbone sits inside the antigen sorted below the rest.
Nothing is grafted, packed or measured, so this needs no packer and writes no
models.

``epifitscout graft`` runs these same stages and then builds models from the
result. The order it reports is identical: building never reorders anything.
"""

from __future__ import annotations

from epifitscout.tools.graft_cdr import main as _graft_main


def main(argv: list[str] | None = None) -> int:
    """Run stages 1-8 and print the ranking."""
    return _graft_main(argv, build=False)
