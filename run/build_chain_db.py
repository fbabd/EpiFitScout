"""Build the EpiFitScout chain database.

Two input modes (mutually exclusive):

  --metadata   Build from SAbDab summary CSV (default)
  --pdb_list   Build from a plain text file of PDB IDs and chains

PDB list format (one entry per line)::

    1abc          # extract all protein chains
    1abc H        # heavy chain only
    1abc H L      # heavy and light chains
    # comment lines and blank lines are ignored

Usage::

    # Full SAbDab database
    uv run python run/build_chain_db.py

    # Dev test: first 50 SAbDab entries
    uv run python run/build_chain_db.py --limit 50

    # Custom PDB list
    uv run python run/build_chain_db.py --pdb_list my_pdbs.txt

    # Custom paths
    uv run python run/build_chain_db.py \\
        --create_pds MASTER/master-v1.6/bin/createPDS \\
        --metadata data/SAbDab/sabdab_metadata.csv \\
        --output_dir data/sabdab_chains.db \\
        --rcsb_cache data/rcsb_cache

    # Dry-run: count entries only, do not build
    uv run python run/build_chain_db.py --dry_run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Resolve project root so the script works regardless of cwd
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

from epifitscout.db.chain_db_builder import ChainDbBuilder
from epifitscout.db.pdb_list_reader import PdbListReader
from epifitscout.db.rcsb_downloader import RcsbDownloader
from epifitscout.db.sabdab_metadata import SAbDabMetadataReader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build EpiFitScout chain database from SAbDab metadata."
    )
    p.add_argument(
        "--create_pds",
        type=Path,
        default=_PROJECT_ROOT / "MASTER/master-v1.6/bin/createPDS",
        help="Path to MASTER createPDS binary.",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=_PROJECT_ROOT / "data/SAbDab/sabdab_metadata.csv",
        help="Path to SAbDab summary CSV (default input mode).",
    )
    p.add_argument(
        "--pdb_list",
        type=Path,
        default=None,
        help=(
            "Path to a plain text PDB list file. "
            "Format: one entry per line — '1abc', '1abc H', or '1abc H L'. "
            "If given, --metadata is ignored."
        ),
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=_PROJECT_ROOT / "data/sabdab_chains.db",
        help="Output database root directory.",
    )
    p.add_argument(
        "--rcsb_cache",
        type=Path,
        default=_PROJECT_ROOT / "data/rcsb_cache",
        help="Directory for cached RCSB PDB downloads.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of SAbDab entries processed (default: all).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Count entries and exit without building.",
    )
    p.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # Validate binary
    # ------------------------------------------------------------------ #
    if not args.create_pds.exists():
        logger.error("createPDS binary not found: %s", args.create_pds)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Load entries (SAbDab CSV or PDB list)
    # ------------------------------------------------------------------ #
    downloader = RcsbDownloader(cache_dir=args.rcsb_cache)
    builder = ChainDbBuilder(
        db_dir=args.output_dir,
        downloader=downloader,
        create_pds_binary=args.create_pds,
    )

    if args.pdb_list is not None:
        if not args.pdb_list.exists():
            logger.error("PDB list file not found: %s", args.pdb_list)
            sys.exit(1)
        reader_list = PdbListReader(args.pdb_list)
        assignments = reader_list.chain_assignments()
        if args.limit is not None:
            logger.info("Applying limit: %d / %d assignments", args.limit, len(assignments))
            assignments = assignments[: args.limit]
        logger.info(
            "Mode: PDB list  |  Assignments: %d  |  Output DB: %s",
            len(assignments), args.output_dir,
        )
        if args.dry_run:
            logger.info("Dry-run mode — exiting without building.")
            return
        pds_list = builder.build_from_list(assignments)

    else:
        if not args.metadata.exists():
            logger.error("SAbDab metadata CSV not found: %s", args.metadata)
            sys.exit(1)
        sabdab_reader = SAbDabMetadataReader(args.metadata)
        entries = sabdab_reader.entries()
        if args.limit is not None:
            logger.info("Applying limit: %d / %d entries", args.limit, len(entries))
            entries = entries[: args.limit]
        total_chains = sum(len(e.chain_assignments()) for e in entries)
        logger.info(
            "Mode: SAbDab CSV  |  Entries: %d  |  Chains: %d  |  Output DB: %s",
            len(entries), total_chains, args.output_dir,
        )
        if args.dry_run:
            logger.info("Dry-run mode — exiting without building.")
            return
        pds_list = builder.build_from_sabdab(entries)

    logger.info("Done. pds.list written to: %s", pds_list)


if __name__ == "__main__":
    main()
