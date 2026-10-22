#!/usr/bin/env python3
"""Check that EpiFitScout is set up and working.

    python check_setup.py            # are MASTER and the database in place?
    python check_setup.py --search   # also run a real search end to end (~10 s - 3 min)

Exit status is 0 when everything needed to search is present.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
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
    """Report the side-chain packer.

    Does not fail the check, because a search still runs without it — but it
    says plainly that the results will not mean much. Stage 5 measures an
    all-atom interface, and without a packer the models are backbone-only at
    the graft, which reads as a loose interface on every term.
    """
    try:
        from epifitscout.graft.packer import check_packer, find_packer
    except ImportError:
        return

    exe = find_packer("FASPR")
    if exe is None:
        print("[--]   FASPR      not installed — needed for results: without a "
              "packer the")
        print("                  grafts are backbone-only and every interface "
              "term reads loose")
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
    """Run the v2 pipeline end to end on the bundled example.

    The example's own entry is excluded from the search, as a benchmark would
    exclude it: the database is built from SAbDab and 1A2Y is in it, so without
    the exclusion this would retrieve the native loop at RMSD ~0 and report a
    perfect result that means nothing.
    """
    import epifitscout

    example = paths.REPO_ROOT / "example" / "1a2y.pdb"
    if not example.is_file():
        print(f"{BAD} search      example structure missing: {example}")
        return False

    started = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="epifitscout_check_") as tmp:
            result = epifitscout.graft(
                example, "B", (105, 117), ("A",),
                out_dir=Path(tmp), exclude_donors={"1A2Y"},
                pool_size=5, top_k=3, keep_models=False, max_workers=4,
            )
    except Exception as exc:
        print(f"{BAD} search      {type(exc).__name__}: {exc}")
        return False
    elapsed = time.time() - started

    funnel = " ".join(f"{k}={v}" for k, v in result.funnel.items())
    if not result.ok:
        print(f"{BAD} search      ran in {elapsed:.0f}s but returned nothing "
              f"({funnel})")
        return False
    best = result.top[0]
    print(f"{OK} search      {funnel} in {elapsed:.0f}s")
    print(f"      best {best.source} {best.sequence} — {best.interface.describe()}")
    # Only 1A2Y's own entry was excluded, so a different entry holding the same
    # antibody can still be retrieved — and usually is: 1G7M carries this very
    # CDR-H3. Worth seeing here, because it is the leakage mode that PDB-ID
    # exclusion alone misses and that a benchmark has to filter by sequence.
    native = (
        "ARERDYRLDY" if best.sequence.upper() == "ARERDYRLDY" else None
    )
    if native:
        print("      note: that is 1a2y's own CDR-H3, from another entry for the "
              "same antibody.")
        print("      Only the 1A2Y entry was excluded; a benchmark must also "
              "exclude donors")
        print("      that share a stretch of the native loop — see "
              "benchmark/run_v2_benchmark.py.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search", action="store_true",
                        help="Also run the bundled 1a2y example search.")
    args = parser.parse_args(argv)

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
