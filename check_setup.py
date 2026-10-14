#!/usr/bin/env python3
"""Check that EpiFitScout is set up and working.

    python check_setup.py            # are MASTER and the database in place?
    python check_setup.py --search   # also run a real search end to end (~10 s - 3 min)

Exit status is 0 when everything needed to search is present.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from epifitscout import paths
from epifitscout.db.verify import verify_database

OK, BAD = "[ok]  ", "[FAIL]"


def _check_binaries() -> bool:
    good = True
    explained: set[str] = set()
    for label, finder in (("master", paths.find_master_binary),
                          ("createPDS", paths.find_create_pds)):
        try:
            print(f"{OK} {label:<10} {finder()}")
        except paths.SetupError as exc:
            good = False
            message = str(exc)
            # Both binaries usually go missing together and share one fix; say
            # it once rather than printing the same instructions twice.
            body = message.split("\n", 1)[1] if "\n" in message else message
            if body in explained:
                print(f"{BAD} {label:<10} not found (same fix as above)")
                continue
            explained.add(body)
            print(f"{BAD} {label}")
            print("      " + message.replace("\n", "\n      "))
    return good


def _check_packer() -> None:
    """Report the optional side-chain packer. Never fails the check: packing is
    only needed for ``tools/build_complex.py --side-chains packed``."""
    sys.path.insert(0, str(paths.REPO_ROOT / "tools"))
    try:
        from build_complex import check_packer, find_packer  # type: ignore[import-not-found]
    except ImportError:
        return

    exe = find_packer("FASPR")
    if exe is None:
        print("[--]   FASPR      not installed (optional; needed only for "
              "--side-chains packed)")
        print(f"      see {paths.REPO_ROOT / 'FASPR/README.md'}")
        return
    problem = check_packer(exe, "FASPR")
    if problem is None:
        print(f"{OK} FASPR      {exe}")
    else:
        print(f"[--]   FASPR      {problem.splitlines()[0]}")
        print(f"      see {paths.REPO_ROOT / 'FASPR/README.md'}")


def _check_database() -> bool:
    try:
        pds_list = paths.find_pds_list()
    except paths.SetupError as exc:
        print(f"{BAD} database")
        print("      " + str(exc).replace("\n", "\n      "))
        return False

    check = verify_database(pds_list.parent)
    if not check.ok:
        print(f"{BAD} database   {pds_list.parent}")
        for problem in check.problems:
            print(f"      - {problem}")
        return False

    print(f"{OK} database   {pds_list.parent}")
    print(f"      {check.n_listed} searchable chains "
          f"({check.n_pdb} coordinate files, {check.n_pds} search targets)")
    if check.info.get("updated_at"):
        src = check.info.get("input", "unknown source")
        print(f"      last built {check.info['updated_at'][:10]} from {src}")
    for warning in check.warnings:
        print(f"      note: {warning}")
    return True


def _run_search() -> bool:
    import epifitscout

    example = Path(__file__).resolve().parent / "example" / "1a2y.pdb"
    if not example.is_file():
        print(f"{BAD} search      example structure missing: {example}")
        return False
    qb = epifitscout.QueryBuilder(example)
    cdr, epi = qb.get_fragment("B", 99, 110), qb.get_fragment("C", 18, 27)
    started = time.time()
    try:
        hits = epifitscout.search(cdr, epi, max_hits=20, log_level="WARNING")
    except Exception as exc:
        print(f"{BAD} search      {type(exc).__name__}: {exc}")
        return False
    elapsed = time.time() - started
    if not hits:
        print(f"{BAD} search      ran in {elapsed:.0f}s but returned no hits")
        return False
    best = hits[0]
    print(f"{OK} search      {len(hits)} hits in {elapsed:.0f}s; best "
          f"{best.fragment.pdb_id}:{best.fragment.chain} "
          f"RMSD {best.rmsd_cdr:.3f}, score {best.final_score:.3f}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search", action="store_true",
                        help="Also run the bundled 1a2y example search.")
    args = parser.parse_args()

    print(f"EpiFitScout repository: {paths.REPO_ROOT}\n")
    good = _check_binaries()
    good = _check_database() and good
    _check_packer()
    if good and args.search:
        good = _run_search()

    print()
    if good:
        print("Ready." if args.search else
              "Ready. Add --search to run a real example search.")
        return 0
    print("Not ready: fix the items marked FAIL above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
