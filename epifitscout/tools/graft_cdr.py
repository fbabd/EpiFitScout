#!/usr/bin/env python3
"""Retrieve, graft and rank real CDR loops for one complex.

The v2 command. Give it a bound antibody–antigen complex, say which chain
carries the CDR and which residues to replace, and it writes ranked all-atom
models::

    epifitscout graft --structure 7abc.pdb --cdr H 95 102 --antigen A \\
        --out-dir models/7abc

What it does, in one paragraph: the two framework stems either side of the CDR
become a two-segment MASTER query with ``--gapLen`` fixed to the loop's length,
which returns every real loop in the database that can bridge that gap in that
geometry. The CDR's own coordinates are never part of the query, so the loops
that come back are a prediction rather than a restatement of the one already
there. Each is placed by fitting its donor's stems onto the template's — the
only placement a graft allows — gated on whether its two new peptide bonds are
real bonds, and then **chosen by how much of it sits cleanly against the
epitope**, counted from backbone distances. Only the models that come out are
rebuilt all-atom and measured. See
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


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="epifitscout graft",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--structure", type=Path, required=True,
                   help="bound antibody-antigen complex (PDB or mmCIF)")
    p.add_argument("--cdr", nargs=3, metavar=("CHAIN", "FIRST", "LAST"),
                   required=True, help="chain and inclusive residue range to replace")
    p.add_argument("--antigen", nargs="+", metavar="CHAIN", required=True,
                   help="antigen chain(s)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="where models and MASTER files are written")
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
    p.add_argument("--packer", default="FASPR",
                   help="side-chain packer, or 'none' for backbone-only models, "
                        "which makes the interface numbers diagnostic rather "
                        "than results (default: FASPR)")
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
    p.add_argument("--no-keep-models", action="store_true",
                   help="delete each model after scoring it")
    p.add_argument("--quiet", action="store_true", help="errors only")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    chain, first, last = args.cdr[0], int(args.cdr[1]), int(args.cdr[2])
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
        packer_name=args.packer,
        keep_models=not args.no_keep_models,
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

    print(f"\n{'rank':>4}  {'donor':<16} {'sequence':<22} {'fit':>5}  interface")
    for i, candidate in enumerate(result.top, start=1):
        print(
            f"{i:>4}  {candidate.donor:<16} {candidate.sequence:<22} "
            f"{candidate.shape.score:5.2f}  {candidate.interface.describe()}"
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
