#!/usr/bin/env python3
"""Build the EpiFitScout chain database, or bring an existing one up to date.

Reads a SAbDab summary file, downloads the structures it names from RCSB, cuts
out each heavy and light chain, and converts every chain to the format MASTER
searches. Run it again later with a newer SAbDab file to update: it fetches only
what is new and refreshes only the search targets that are out of date.

    # first time, or an update; the same command either way
    python database_builder/build_database.py

    # what would an update do? (nothing is downloaded or written)
    python database_builder/build_database.py --dry-run

    # after switching to a different MASTER version
    python database_builder/build_database.py --rebuild-pds

    # a quick test database
    python database_builder/build_database.py --limit 50 --output-dir ./test.db

See database_builder/README.md for the full workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from epifitscout import paths  # noqa: E402
from epifitscout.db.chain_db_builder import ChainDbBuilder  # noqa: E402
from epifitscout.db.pdb_list_reader import PdbListReader  # noqa: E402
from epifitscout.db.rcsb_downloader import RcsbDownloader  # noqa: E402
from epifitscout.db.sabdab_metadata import SAbDabMetadataReader  # noqa: E402
from epifitscout.db.verify import verify_database  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_SUMMARY = HERE / "sabdab_summary.tsv"
SABDAB_URL = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--metadata", type=Path, default=DEFAULT_SUMMARY,
        help="SAbDab summary file (tab- or comma-separated). Default: %(default)s",
    )
    p.add_argument(
        "--pdb-list", type=Path, default=None,
        help="Build from a plain list of PDB ids/chains instead of SAbDab. One "
             "entry per line: '1abc', '1abc H' or '1abc H L'.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=paths.DB_DIR,
        help="Database directory. Default: %(default)s",
    )
    p.add_argument(
        "--rcsb-cache", type=Path, default=HERE / "rcsb_cache",
        help="Where downloaded RCSB structures are kept. Safe to delete after a "
             "build. Default: %(default)s",
    )
    p.add_argument(
        "--create-pds", type=Path, default=None,
        help="Path to createPDS. Default: MASTER/bin/createPDS",
    )
    p.add_argument("--limit", type=int, default=None, help="Process only the first N entries.")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be added; download and write nothing.",
    )
    p.add_argument(
        "--rebuild-pds", action="store_true",
        help="Regenerate every PDS target, even those that look up to date. Needed "
             "after changing MASTER versions.",
    )
    p.add_argument(
        "--retry-skipped", action="store_true",
        help="Re-attempt chains previously recorded as unextractable.",
    )
    p.add_argument("--workers", type=int, default=4, help="Parallel createPDS processes.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("build_database")

    try:
        create_pds = paths.find_create_pds(args.create_pds)
    except paths.SetupError as exc:
        log.error("\n%s", exc)
        return 1

    # ---- read the input list ------------------------------------------------
    source: dict[str, str] = {}
    try:
        if args.pdb_list is not None:
            if not args.pdb_list.is_file():
                log.error("PDB list not found: %s", args.pdb_list)
                return 1
            assignments = PdbListReader(args.pdb_list).chain_assignments()
            if args.limit is not None:
                assignments = assignments[: args.limit]
            n_requested = len(assignments)
            source = {"source": "pdb-list", "input": args.pdb_list.name}
            entries = None
        else:
            if not args.metadata.is_file():
                log.error(
                    "SAbDab summary file not found: %s\n"
                    "Download the summary from %s and save it there, or pass "
                    "--metadata <file>.",
                    args.metadata, SABDAB_URL,
                )
                return 1
            entries = SAbDabMetadataReader(args.metadata).entries()
            if args.limit is not None:
                entries = entries[: args.limit]
            n_requested = sum(len(e.chain_assignments()) for e in entries)
            source = {
                "source": "sabdab",
                "input": args.metadata.name,
                "input_sha256": _sha256(args.metadata),
            }
            assignments = None
    except (ValueError, OSError) as exc:
        log.error("%s", exc)
        return 1

    out = args.output_dir
    existing = {p.stem for p in (out / "pdb").glob("*.pdb")} if (out / "pdb").is_dir() else set()

    # ---- dry run: say what an update would do -------------------------------
    if args.dry_run:
        if entries is not None:
            wanted = [
                f"{e.pdb_id}_{cid}_{ctype}"
                for e in entries for cid, ctype in e.chain_assignments()
            ]
            new = [w for w in wanted if w not in existing]
            n_entries = len({e.pdb_id for e in entries})
        else:
            new, n_entries = [], 0
        log.info("Input        : %d chains from %d PDB entries", n_requested, n_entries)
        log.info("In database  : %d chain files in %s", len(existing), out)
        if entries is not None:
            log.info("To add       : %d chains (some may be unextractable and skipped)", len(new))
            if new:
                log.info("  e.g. %s", ", ".join(new[:5]))
        log.info("Dry run — nothing downloaded or written.")
        return 0

    # ---- build / update -----------------------------------------------------
    log.info("Database: %s  (%d chain files present)", out, len(existing))
    builder = ChainDbBuilder(
        db_dir=out,
        downloader=RcsbDownloader(cache_dir=args.rcsb_cache),
        create_pds_binary=create_pds,
        force_pds=args.rebuild_pds,
        retry_skipped=args.retry_skipped,
        workers=args.workers,
    )
    try:
        if entries is not None:
            builder.build_from_sabdab(entries)
        else:
            builder.build_from_list(assignments or [])
    except KeyboardInterrupt:
        log.error("Interrupted. Re-run the same command to resume; finished work is kept.")
        return 130

    builder.write_build_info(**source)

    check = verify_database(out)
    print()
    print(builder.report.summary())
    for warning in check.warnings:
        print(f"warning: {warning}")
    if not check.ok:
        print("The database is NOT usable yet:")
        for problem in check.problems:
            print(f"  - {problem}")
        return 1
    print(f"Database ready: {out}  ({check.n_listed} searchable chains)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
