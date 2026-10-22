#!/usr/bin/env python3
"""Worked v2 example: graft new CDR-H3 loops into the bundled 1a2y complex.

Run it from the repository root::

    python example/example_v2_graft.py

1a2y is HyHEL-10 Fv bound to hen egg lysozyme. Chain B is the heavy chain, A is
the antigen, and CDR-H3 is IMGT 105-117 — thirteen residue *numbers* covering
ten real residues, because 110-112 are absent. That alone is worth seeing: a
span's length is not ``last - first + 1`` in antibody numbering, and the query
counts real residues.

The example walks the stages one at a time rather than calling
``epifitscout.graft()``, because the point is to show what each one decides.

**Leakage is excluded twice over, and the example shows why once is not enough.**
The chain database is built from SAbDab and 1A2Y is in it, so without exclusion
the search retrieves this very loop at RMSD ~0 and reports a perfect result that
means nothing. Excluding the *entry* is not enough either: HyHEL-10 is deposited
many times, and with only 1A2Y refused the top five hits are all its exact
CDR-H3 from 1A7O, 1A7P, 1VFA, 1A7R and 1KIR. The run below is done both ways so
the difference is visible.
"""

from __future__ import annotations

import logging
from pathlib import Path

from epifitscout.config.schema import MASTERConfig, PipelineConfig
from epifitscout.graft import feasibility as feas
from epifitscout.graft.assemble import TemplateSite
from epifitscout.pipeline.graft_pipeline import GraftPipeline
from epifitscout.query.builder import QueryBuilder
from epifitscout.scoring.contact_fit import score_fit

STRUCTURE = Path(__file__).resolve().parent / "1a2y.pdb"
AB_CHAIN, ANTIGEN_CHAIN = "B", "A"
CDR_H3 = (105, 117)
POOL, TOP = 50, 5


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    work = Path("example/test_run/v2")
    work.mkdir(parents=True, exist_ok=True)

    # --- stage 0: what are we asking for? -------------------------------- #
    builder = QueryBuilder(STRUCTURE)
    query = builder.get_anchor_query(AB_CHAIN, *CDR_H3)
    print(f"\nquery:  {query.describe()}")
    print(f"        native loop {query.metadata['native_loop_sequence']} "
          f"({query.gap_length} residues) — read only to count them, "
          f"never searched with")
    print(f"        stems {query.n_stem.sequence} ... {query.c_stem.sequence}, "
          f"gapLen {query.gap_len_argument()}")

    site = TemplateSite.load(
        STRUCTURE, AB_CHAIN, query.loop_range, (ANTIGEN_CHAIN,)
    )
    print(f"        site is bridgeable: {feas.query_is_bridgeable(query)} "
          f"({query.span_per_residue:.2f} Å per residue, a chain can reach 3.3)")

    pipeline = GraftPipeline(PipelineConfig(
        master=MASTERConfig(max_hits=200), pool_size=POOL, top_k=TOP,
        max_workers=4, log_level="WARNING",
    ))

    # --- stages 1-2: retrieve and place ---------------------------------- #
    native = query.metadata["native_loop_sequence"]
    leaky = pipeline.search(query, work / "leaky", exclude_donors={"1A2Y"})
    same = sum(1 for c in leaky if c.sequence.upper() == native.upper())
    print(f"\nstage 1-2  with only the 1A2Y entry excluded: {len(leaky)} loops, "
          f"{same} of them\n           the native sequence itself, from other "
          f"entries of the same antibody")

    candidates = pipeline.search(
        query, work, exclude_donors={"1A2Y"}, exclude_sequences={native},
    )
    print(f"           excluding that sequence too: {len(candidates)} loops "
          f"bridge this gap")
    if not candidates:
        print("           nothing retrieved — is the database installed?")
        return 1
    best_fit = min(candidates, key=lambda c: c.stem_rmsd_bb)
    print(f"           tightest stem fit {best_fit.stem_rmsd_bb:.2f} Å "
          f"({best_fit.source})")

    # --- stage 3: can they be bonded in? -------------------------------- #
    feasible = feas.assess_all(candidates, query, site.framework_coords)
    print(f"\nstage 3    {len(feasible)} of {len(candidates)} can be grafted")
    rejected = [c for c in candidates if not c.is_feasible]
    if rejected:
        example = rejected[0]
        print(f"           e.g. {example.source} rejected: "
              f"{'; '.join(example.feasibility.reasons)}")
    if feasible:
        print(f"           e.g. {feasible[0].source}: "
              f"{feasible[0].feasibility.describe()}")

    # --- stage 4: the selection ----------------------------------------- #
    for candidate in feasible:
        candidate.fit = score_fit(candidate.placed_loop, site.antigen_coords)
    fitting = sorted(
        (c for c in feasible if c.fit.fits), key=lambda c: -c.fit.score
    )
    print(f"\nstage 4    {len(fitting)} of {len(feasible)} sit against the "
          f"epitope without colliding")
    if fitting:
        print(f"           best: {fitting[0].source} — {fitting[0].fit.describe()}")

    # --- stages 5-6: rank, then measure what came out -------------------- #
    scored = pipeline.measure(fitting[:TOP], query, site, work, keep_models=True)
    ranked = sorted(
        scored,
        key=lambda c: (bool(c.interface.is_clean), c.fit.score), reverse=True,
    )
    print("\nstage 5-6  ranked on contact fit, then measured all-atom\n")
    print(f"{'donor':<16} {'sequence':<14} {'fit':>5}  {'BSA':>6} {'cl':>3} "
          f"{'hb':>3} {'sb':>3} {'gap':>5}")
    for candidate in ranked:
        score = candidate.interface
        print(
            f"{candidate.donor:<16} {candidate.sequence:<14} "
            f"{candidate.fit.score:5.2f}  {score.loop_buried_area:6.0f} "
            f"{score.clashes:3d} {score.hbonds:3d} {score.salt_bridges:3d} "
            f"{score.gap_index:5.2f}"
        )

    print(
        "\n`fit` is what chose these. Everything to its right was measured "
        "afterwards and\nnot used to choose, which is what lets it be read as "
        "evidence. BSA and clashes\nare printed together on purpose: burial "
        "rewards interpenetration, so an area\nwith no steric term beside it "
        "ranks a loop driven into the antigen first."
    )
    if ranked and ranked[0].model_path:
        print(f"\nmodels in {work / 'models'}; best: {ranked[0].model_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
