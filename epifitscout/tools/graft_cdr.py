#!/usr/bin/env python3
"""Retrieve and rank real CDR loops for one site, and build models from them.

Give it a bound antibody–antigen complex, say which chain carries the CDR and
which residues to replace, and it writes ranked all-atom models::

    epifitscout graft --structure 7abc.pdb --cdr H 95 102 --antigen A \\
        --out-dir models/7abc

What it does, in one paragraph: the two framework stems either side of the CDR
become a two-segment MASTER query with ``--gapLen`` fixed to the loop's length,
which returns every real loop in the database that can bridge that gap in that
geometry. The CDR's own coordinates are never part of the query, so the loops
that come back are a prediction rather than a restatement of the one already
there. Each is placed by fitting its donor's stems onto the template's — the
only placement a graft allows — gated on whether its two new peptide bonds are
real bonds, and then **chosen by how well its backbone complements the
epitope**, with candidates sitting inside the antigen sorted below the rest.
That ranking is settled before anything is built; only the models that come
out are packed, relaxed and measured, and none of those measurements change
the order. ``epifitscout search`` stops at the ranking. See
:mod:`epifitscout.pipeline.graft_pipeline`.

On ``--cdr``: state the span explicitly when comparing against another method.
IMGT and Chothia CDR-H3 differ by exactly two residues, and left alone that has
the two methods designing different windows while every metric is reported as
though they were the same.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from epifitscout.config.schema import MASTERConfig, PipelineConfig
from epifitscout.pipeline.graft_pipeline import GraftPipeline

logger = logging.getLogger(__name__)


def _parser(build: bool = True) -> argparse.ArgumentParser:
    """Arguments for ``graft``, or for ``search`` when ``build`` is False.

    The two commands share every search and ranking option, because they run
    the same stages 1-8 and must return the same order. Only the options that
    exist to control building a model are hidden from ``search``, since it
    builds none.
    """
    from epifitscout.tools import search_cdr

    p = argparse.ArgumentParser(
        prog="epifitscout graft" if build else "epifitscout search",
        description=__doc__ if build else search_cdr.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--structure", type=Path, required=True,
                   help="bound antibody-antigen complex (PDB or mmCIF)")
    p.add_argument("--cdr", nargs=3, metavar=("CHAIN", "FIRST", "LAST"),
                   required=True, help="chain and inclusive residue range to replace")
    p.add_argument("--antigen", nargs="+", metavar="CHAIN", required=True,
                   help="antigen chain(s)")
    p.add_argument("--out-dir", type=Path, required=build,
                   default=None,
                   help="where models and MASTER files are written"
                        if build else
                        "keep the MASTER files here (default: a temporary "
                        "directory, discarded)")
    p.add_argument("--top", type=int, default=10,
                   help="candidates to report (default: 10)")
    p.add_argument("--pool", type=int, default=50,
                   help="feasible candidates to pack and measure. This is the "
                        "search budget: hold it equal across methods being "
                        "compared (default: 50)")
    p.add_argument("--max-hits", type=int, default=500,
                   help="matches MASTER returns before the gate (default: 500)")
    p.add_argument("--rmsd-cut", type=float, default=2.0,
                   help="stem RMSD cutoff in A (default: 2.0)")
    p.add_argument("--stem", type=int, default=4,
                   help="framework residues per stem in the query (default: 4)")
    if build:
        p.add_argument("--packer", default="FASPR",
                       help="side-chain packer, or 'none' for backbone-only "
                            "models, which makes the interface numbers "
                            "diagnostic rather than results (default: FASPR)")
    p.add_argument("--exclude", nargs="*", default=[], metavar="PDBID",
                   help="donors the search must not return. For a benchmark "
                        "this is the leakage control: the database is built "
                        "from SAbDab, so a SAbDab target can otherwise "
                        "retrieve its own native loop and rank it first")
    p.add_argument("--exclude-native-loop", action="store_true",
                   help="also refuse any retrieved loop sharing an 8-residue "
                        "stretch with the template's own CDR. Excluding the "
                        "target's PDB ID does not cover the same antibody "
                        "deposited under another accession — on 1a2y, five "
                        "other entries return its exact CDR-H3. Any benchmark "
                        "wants this on")
    p.add_argument("--database", type=Path, default=None,
                   help="a pds.list to search instead of the default. Also how "
                        "to exclude many donors at once, by filtering the list")
    p.add_argument("--workers", type=int, default=6,
                   help="parallel MASTER shards and packing threads (default: 6)")
    if build:
        p.add_argument("--no-keep-models", action="store_true",
                       help="delete each model after scoring it")
    p.add_argument("--quiet", action="store_true", help="errors only")
    return p


def main(argv: list[str] | None = None, build: bool = True) -> int:
    args = _parser(build).parse_args(argv)
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    chain, first, last = args.cdr[0], int(args.cdr[1]), int(args.cdr[2])
    packer = getattr(args, "packer", "FASPR")
    keep_models = build and not getattr(args, "no_keep_models", False)
    cfg = PipelineConfig(
        master=MASTERConfig(
            database_path=args.database,
            rmsd_threshold=args.rmsd_cut,
            max_hits=args.max_hits,
        ),
        pool_size=args.pool,
        top_k=args.top,
        stem_length=args.stem,
        max_workers=args.workers,
        log_level="ERROR" if args.quiet else "INFO",
    )

    result = GraftPipeline(cfg).run(
        template=args.structure,
        ab_chain=chain,
        loop_span=(first, last),
        antigen_chains=tuple(args.antigen),
        out_dir=args.out_dir,
        exclude_donors={d.upper() for d in args.exclude},
        exclude_native_loop=args.exclude_native_loop,
        packer_name=packer,
        keep_models=keep_models,
        build=build,
    )

    print(f"\n{result.query.describe()}")
    print("  " + "  ".join(f"{k} {v}" for k, v in result.funnel.items()))
    if result.parse_stats is not None:
        print(f"  retrieval: {result.parse_stats.summary()}")

    if not result.ok:
        print(
            "\nNo candidate came through. The funnel above says where they went:\n"
            "  retrieved 0   nothing in the database bridges this gap in this\n"
            "                geometry — widen --rmsd-cut, or check the span\n"
            "  feasible 0    loops were found but none can be bonded in; the\n"
            "                stems may be poorly modelled\n"
            "  scored 0      models were built but could not be packed or\n"
            "                measured — run with --packer none to see them",
            file=sys.stderr,
        )
        return 1

    if not build:
        print(f"\n{'rank':>4}  {'donor':<16} {'sequence':<22} {'shape':>6}  detail")
        for i, candidate in enumerate(result.top, start=1):
            print(
                f"{i:>4}  {candidate.donor:<16} {candidate.sequence:<22} "
                f"{candidate.shape.score:6.3f}  {candidate.shape.describe()}"
            )
        print(
            f"\nEpitope: {result.epitope_residues} residues. "
            f"{result.clear_of_antigen} of {result.funnel.get('feasible', 0)} "
            "feasible candidates are clear of the antigen.\n"
            "Ranked on backbone shape complementarity alone; nothing was "
            "grafted, packed or measured.\nUse `epifitscout graft` for "
            "all-atom models, which returns this same order."
        )
        return 0

    print(f"\n{'rank':>4}  {'donor':<16} {'sequence':<22} {'shape':>6}  interface")
    for i, candidate in enumerate(result.top, start=1):
        print(
            f"{i:>4}  {candidate.donor:<16} {candidate.sequence:<22} "
            f"{candidate.shape.score:6.3f}  {candidate.interface.describe()}"
        )
        print(f"        {candidate.shape.describe()}")
        if candidate.model_path is not None:
            print(f"        {candidate.model_path}")

    clean = sum(1 for c in result.top if c.interface.is_clean)
    print(
        f"\n{clean} of {len(result.top)} reported models have no clash with the "
        "antigen.\n`fit` is what the ranking used — the fraction of the loop in "
        "its longest clean\ncontact run. Everything in the interface column was "
        "measured afterwards and\nnot used to choose, so it can be read as "
        "evidence about the ranking."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
