"""Check that a chain database is complete and internally consistent.

One definition of "this database is usable", shared by the updater, the dataset
packaging script and ``check_setup.py`` so they cannot drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["DbCheck", "verify_database"]

#: A PDB newer than its PDS by more than this is reported as stale.
_STALE_TOLERANCE_S = 2.0


@dataclass
class DbCheck:
    """Result of :func:`verify_database`."""

    ok: bool
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_listed: int = 0
    n_pdb: int = 0
    n_pds: int = 0
    info: dict[str, Any] = field(default_factory=dict)


def verify_database(db_dir: Path | str) -> DbCheck:
    """Verify a database directory.

    A database is *usable* when ``pds.list`` is present and every entry in it
    resolves to a non-empty PDS file with a matching PDB. Staleness (a PDB newer
    than its PDS) and unlisted files are reported as warnings, since they do not
    stop a search but do mean an update is due.

    Args:
        db_dir: The ``sabdab_chains.db`` directory.
    """
    db = Path(db_dir)
    result = DbCheck(ok=False)

    if not db.is_dir():
        result.problems.append(f"directory does not exist: {db}")
        return result

    pdb_dir, pds_dir = db / "pdb", db / "pds"
    for label, directory in (("pdb", pdb_dir), ("pds", pds_dir)):
        if not directory.is_dir():
            result.problems.append(f"missing {label}/ directory")
    pds_list = db / "pds.list"
    if not pds_list.is_file():
        result.problems.append("missing pds.list")
    if result.problems:
        return result

    result.n_pdb = sum(1 for _ in pdb_dir.glob("*.pdb"))
    result.n_pds = sum(1 for _ in pds_dir.glob("*.pds"))

    listed = [ln.strip() for ln in pds_list.read_text().splitlines() if ln.strip()]
    result.n_listed = len(listed)
    if not listed:
        result.problems.append("pds.list is empty")
        return result

    missing: list[str] = []
    empty: list[str] = []
    no_pdb: list[str] = []
    stale = 0
    for rel in listed:
        pds = db / rel
        try:
            size = pds.stat().st_size
        except FileNotFoundError:
            missing.append(rel)
            continue
        if size == 0:
            empty.append(rel)
        pdb = pdb_dir / f"{pds.stem}.pdb"
        try:
            if pdb.stat().st_mtime > pds.stat().st_mtime + _STALE_TOLERANCE_S:
                stale += 1
        except FileNotFoundError:
            no_pdb.append(rel)

    if missing:
        result.problems.append(
            f"{len(missing)} entries in pds.list point at files that do not exist "
            f"(e.g. {missing[0]})"
        )
    if empty:
        result.problems.append(
            f"{len(empty)} PDS files are empty (e.g. {empty[0]}); rebuild them"
        )
    if no_pdb:
        result.problems.append(
            f"{len(no_pdb)} PDS files have no matching PDB, so their coordinates "
            f"cannot be loaded (e.g. {no_pdb[0]})"
        )
    if stale:
        result.warnings.append(
            f"{stale} PDS files are older than their PDB; run the updater with "
            "--rebuild-pds to refresh them"
        )
    unlisted = result.n_pds - (result.n_listed - len(missing))
    if unlisted > 0:
        result.warnings.append(f"{unlisted} PDS files are not listed in pds.list")

    info_path = db / "BUILD_INFO.json"
    if info_path.is_file():
        try:
            result.info = json.loads(info_path.read_text())
        except json.JSONDecodeError:
            result.warnings.append("BUILD_INFO.json is unreadable")

    result.ok = not result.problems
    return result
