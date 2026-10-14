"""ChainDbBuilder: build and incrementally update a MASTER-searchable chain database.

Accepts chain assignments from any source reader (SAbDabMetadataReader or
PdbListReader), downloads PDB files from RCSB on demand, and maintains a
MASTER-compatible database with separate ``pdb/`` and ``pds/`` directories.
Paths in ``pds.list`` and ``db.list`` are stored relative to the database root
so the database can be moved across machines.

Output layout::

    <db_dir>/
    ├── pdb/                ← full-chain PDB files (coordinate source)
    │   ├── 1ABC_H_H.pdb
    │   └── 1ABC_A_L.pdb
    ├── pds/                ← MASTER binary format, one per PDB
    │   ├── 1ABC_H_H.pds
    │   └── 1ABC_A_L.pds
    ├── pds.list            ← relative paths, one per line (pass to MASTER)
    ├── db.list             ← relative paths to PDB files
    ├── metadata.json       ← per-chain annotation
    ├── skipped.json        ← chains that cannot be extracted, so updates skip them
    └── BUILD_INFO.json     ← when and from what the database was last built

Updating
--------
Running a build against an existing database is an *update*, and is safe to
repeat. The rules, each of which guards against a specific failure:

* A chain whose PDB already exists is **not re-downloaded** — an updater that
  never had the RCSB cache must not re-fetch the whole database.
* A PDS is **rebuilt** when it is missing, empty, or older than its PDB. A PDB
  regenerated from a revised RCSB entry therefore refreshes its search target.
* PDS files are written to a temporary name and renamed on success, so a failed
  or interrupted ``createPDS`` can never leave a truncated target that later
  runs would trust.
* The list files are derived from what is actually on disk, so a run over a
  subset of entries never truncates the database.
* A PDS with no PDB is removed.

Usage::

    from epifitscout.db.sabdab_metadata import SAbDabMetadataReader
    from epifitscout.db.rcsb_downloader import RcsbDownloader
    from epifitscout.db.chain_db_builder import ChainDbBuilder

    entries  = SAbDabMetadataReader(Path("sabdab_metadata.csv")).entries()
    builder  = ChainDbBuilder(
        db_dir=Path("data/sabdab_chains.db"),
        downloader=RcsbDownloader(cache_dir=Path("rcsb_cache")),
        create_pds_binary=Path("MASTER/bin/createPDS"),
    )
    pds_list = builder.build_from_sabdab(entries)
    print(builder.report)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from epifitscout.db.rcsb_downloader import RcsbDownloader, StructureFile
from epifitscout.db.sabdab_metadata import SAbDabEntry
from epifitscout.utils.io import extract_backbone_coords_by_range, fragment_to_pdb_string

logger = logging.getLogger(__name__)

#: A PDB is considered newer than its PDS only beyond this many seconds, so
#: filesystems and archive formats that round timestamps never trigger a
#: spurious full rebuild.
_MTIME_TOLERANCE_S = 2.0

_CREATE_PDS_TIMEOUT_S = 120


@dataclass
class BuildReport:
    """What a build or update run actually did."""

    requested: int = 0
    already_present: int = 0
    added: int = 0
    skipped_known: int = 0
    skipped_new: int = 0
    download_failed: int = 0
    pds_built: int = 0
    pds_reused: int = 0
    pds_failed: int = 0
    orphans_removed: int = 0
    in_database: int = 0

    def summary(self) -> str:
        return (
            f"requested {self.requested} chains: "
            f"{self.already_present} already present, {self.added} added, "
            f"{self.skipped_known + self.skipped_new} skipped "
            f"({self.skipped_new} newly), {self.download_failed} download failures. "
            f"PDS: {self.pds_built} built, {self.pds_reused} up to date, "
            f"{self.pds_failed} failed, {self.orphans_removed} orphans removed. "
            f"Database now holds {self.in_database} searchable chains."
        )


class ChainDbBuilder:
    """Builds and updates a MASTER-searchable protein chain database.

    Args:
        db_dir: Root directory for the database.
        downloader: RcsbDownloader for fetching PDB files. May be None when
            only refreshing PDS targets with :meth:`sync_pds`.
        create_pds_binary: Path to MASTER's ``createPDS`` binary.
        force_pds: Rebuild every PDS even if it looks up to date. Use after
            changing MASTER versions.
        retry_skipped: Re-attempt chains recorded in ``skipped.json`` instead of
            skipping them.
        workers: Parallel ``createPDS`` processes.
    """

    def __init__(
        self,
        db_dir: Path,
        downloader: RcsbDownloader | None,
        create_pds_binary: Path,
        *,
        force_pds: bool = False,
        retry_skipped: bool = False,
        workers: int = 4,
    ) -> None:
        self._db_dir = Path(db_dir)
        self._pdb_dir = self._db_dir / "pdb"
        self._pds_dir = self._db_dir / "pds"
        self._db_list = self._db_dir / "db.list"
        self._pds_list = self._db_dir / "pds.list"
        self._meta_path = self._db_dir / "metadata.json"
        self._skipped_path = self._db_dir / "skipped.json"
        self._info_path = self._db_dir / "BUILD_INFO.json"
        self._downloader = downloader
        self._create_pds = Path(create_pds_binary)
        self._force_pds = force_pds
        self._retry_skipped = retry_skipped
        self._workers = max(1, workers)
        self.report = BuildReport()

        if not self._create_pds.exists():
            raise FileNotFoundError(
                f"createPDS binary not found at {self._create_pds}"
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build_from_sabdab(self, entries: list[SAbDabEntry]) -> Path:
        """Build or update the database from SAbDab metadata entries.

        Returns:
            Path to ``pds.list`` (pass to MASTERConfig.database_path).
        """
        assignments: list[tuple[str, str, str]] = []
        for entry in entries:
            for chain_id, chain_type in entry.chain_assignments():
                assignments.append((entry.pdb_id, chain_id, chain_type))
        return self._build(assignments)

    def build_from_list(
        self,
        chain_assignments: list[tuple[str, str | None]],
    ) -> Path:
        """Build or update the database from a generic (pdb_id, chain_id | None) list.

        When chain_id is None, all protein chains in the PDB are extracted.

        Returns:
            Path to ``pds.list``.
        """
        self._require_downloader()
        expanded: list[tuple[str, str, str]] = []
        for pdb_id, chain_id in chain_assignments:
            if chain_id is None:
                try:
                    sf = self._downloader.download(pdb_id)  # type: ignore[union-attr]
                    for cid in _discover_chains(sf.path, sf.fmt):
                        expanded.append((pdb_id, cid, ""))
                except Exception as exc:
                    logger.warning("Skipping %s (download error): %s", pdb_id, exc)
            else:
                expanded.append((pdb_id, chain_id, ""))
        return self._build(expanded)

    def sync_pds(self) -> Path:
        """Bring PDS targets and list files in line with the PDB files on disk.

        Public so a database can be refreshed without any input list — for
        example after swapping in a different MASTER build.

        Returns:
            Path to ``pds.list``.
        """
        self._pdb_dir.mkdir(parents=True, exist_ok=True)
        self._pds_dir.mkdir(parents=True, exist_ok=True)
        self._sync_pds()
        return self._pds_list

    def write_build_info(self, **extra: Any) -> None:
        """Record when and from what the database was last built."""
        info: dict[str, Any] = {}
        if self._info_path.exists():
            try:
                info = json.loads(self._info_path.read_text())
            except json.JSONDecodeError:
                info = {}
        info.update(
            updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_chains=self.report.in_database,
            **extra,
        )
        _atomic_write(self._info_path, json.dumps(info, indent=2) + "\n")

    def load_metadata(self) -> dict[str, dict]:
        """Return chain metadata keyed by relative PDB path string."""
        if not self._meta_path.exists():
            return {}
        with self._meta_path.open() as fh:
            return json.load(fh)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _build(self, assignments: list[tuple[str, str, str]]) -> Path:
        """Core update loop: extract missing chains → refresh PDS → write lists."""
        self._require_downloader()
        self._pdb_dir.mkdir(parents=True, exist_ok=True)
        self._pds_dir.mkdir(parents=True, exist_ok=True)

        self.report = BuildReport(requested=len(assignments))
        metadata = self.load_metadata()
        skipped = self._load_skipped()

        # A PDB shared by several chains is downloaded once per run. None marks
        # a download that already failed, so it is not retried per chain.
        fetched: dict[str, StructureFile | None] = {}
        total = len(assignments)

        for i, (pdb_id, chain_id, chain_type) in enumerate(assignments, 1):
            suffix = f"_{chain_type}" if chain_type else ""
            filename = f"{pdb_id}_{chain_id}{suffix}.pdb"
            out_pdb = self._pdb_dir / filename
            rel = f"pdb/{filename}"

            if out_pdb.is_file() and out_pdb.stat().st_size > 0:
                self.report.already_present += 1
                metadata.setdefault(
                    rel, {"pdb_id": pdb_id, "chain": chain_id, "chain_type": chain_type}
                )
            elif filename in skipped and not self._retry_skipped:
                self.report.skipped_known += 1
            else:
                reason = self._extract_chain(fetched, pdb_id, chain_id, chain_type, out_pdb)
                if reason is None:
                    self.report.added += 1
                    skipped.pop(filename, None)
                    metadata[rel] = {
                        "pdb_id": pdb_id, "chain": chain_id, "chain_type": chain_type,
                    }
                elif reason.startswith("download:"):
                    # Transient; deliberately not remembered, so the next run retries.
                    self.report.download_failed += 1
                else:
                    self.report.skipped_new += 1
                    skipped[filename] = reason

            if i % 500 == 0 or i == total:
                logger.info(
                    "Chains: %d/%d | %d present, %d added, %d skipped",
                    i, total, self.report.already_present, self.report.added,
                    self.report.skipped_known + self.report.skipped_new,
                )

        self._save_skipped(skipped)
        _atomic_write(self._meta_path, json.dumps(metadata, indent=2))
        self._sync_pds()

        logger.info("%s", self.report.summary())
        return self._pds_list

    def _require_downloader(self) -> None:
        if self._downloader is None:
            raise RuntimeError(
                "This ChainDbBuilder was created without a downloader, so it can "
                "only refresh PDS targets (sync_pds)."
            )

    def _extract_chain(
        self,
        fetched: dict[str, StructureFile | None],
        pdb_id: str,
        chain_id: str,
        chain_type: str,
        out_path: Path,
    ) -> str | None:
        """Write one chain's PDB. Returns None on success, else a reason string.

        Reasons beginning ``download:`` are transient and are not remembered.
        """
        if pdb_id not in fetched:
            try:
                fetched[pdb_id] = self._downloader.download(pdb_id)  # type: ignore[union-attr]
            except Exception as exc:
                logger.warning("Download failed for %s: %s", pdb_id, exc)
                fetched[pdb_id] = None
                return f"download: {exc}"
        sf = fetched[pdb_id]
        if sf is None:
            return "download: failed earlier in this run"

        try:
            coords, resnums, sequence = extract_backbone_coords_by_range(
                sf.path, chain_id, res_start=1, res_end=9999, fmt=sf.fmt
            )
        except ValueError as exc:
            logger.debug("Skipping %s chain %s: %s", pdb_id, chain_id, exc)
            return f"extract: {exc}"

        if len(resnums) < 4:
            logger.debug(
                "Skipping %s chain %s: too short (%d residues)",
                pdb_id, chain_id, len(resnums),
            )
            return f"too short ({len(resnums)} residues)"

        try:
            pdb_str = fragment_to_pdb_string(
                coords,
                sequence=sequence,
                chain_id=chain_id,
                start_resnum=resnums[0],
            )
            _atomic_write(out_path, pdb_str)
        except Exception as exc:
            logger.warning("Failed to write PDB for %s chain %s: %s", pdb_id, chain_id, exc)
            return f"write: {exc}"
        return None

    # ---- PDS maintenance ------------------------------------------------ #

    def _pds_is_current(self, pdb: Path) -> bool:
        """True when this PDB's PDS exists, is non-empty, and is not older than it."""
        pds = self._pds_dir / f"{pdb.stem}.pds"
        try:
            pds_stat = pds.stat()
            pdb_stat = pdb.stat()
        except FileNotFoundError:
            return False
        if pds_stat.st_size == 0:
            return False
        return pdb_stat.st_mtime <= pds_stat.st_mtime + _MTIME_TOLERANCE_S

    def _convert_one(self, pdb: Path) -> str | None:
        """Run createPDS for one chain. Returns None on success, else an error."""
        final = self._pds_dir / f"{pdb.stem}.pds"
        tmp = self._pds_dir / f".{pdb.stem}.pds.tmp"
        cmd = [
            str(self._create_pds), "--type", "target",
            "--pdb", str(pdb), "--pds", str(tmp),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=_CREATE_PDS_TIMEOUT_S
            )
        except subprocess.TimeoutExpired:
            tmp.unlink(missing_ok=True)
            return f"createPDS timed out after {_CREATE_PDS_TIMEOUT_S}s"
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            return f"createPDS could not be executed: {exc}"

        if result.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            detail = (result.stderr or result.stdout or "no output").strip()[:200]
            return f"createPDS failed (code {result.returncode}): {detail}"

        os.replace(tmp, final)
        return None

    def _sync_pds(self) -> None:
        """Rebuild stale PDS targets, drop orphans, and rewrite the list files."""
        # Debris from a run that was killed mid-write.
        for leftover in (*self._pds_dir.glob(".*.tmp"), *self._pdb_dir.glob(".*.tmp")):
            leftover.unlink(missing_ok=True)

        pdbs = sorted(self._pdb_dir.glob("*.pdb"))
        pdb_stems = {p.stem for p in pdbs}

        for pds in self._pds_dir.glob("*.pds"):
            if pds.stem not in pdb_stems:
                pds.unlink()
                self.report.orphans_removed += 1

        todo = [p for p in pdbs if self._force_pds or not self._pds_is_current(p)]
        self.report.pds_reused += len(pdbs) - len(todo)

        failed: dict[str, str] = {}
        if todo:
            logger.info(
                "Building PDS targets: %d of %d need (re)building (%d workers)",
                len(todo), len(pdbs), self._workers,
            )
            with ThreadPoolExecutor(max_workers=self._workers) as pool:
                futures = {pool.submit(self._convert_one, p): p for p in todo}
                for n, future in enumerate(as_completed(futures), 1):
                    pdb = futures[future]
                    error = future.result()
                    if error is None:
                        self.report.pds_built += 1
                    else:
                        self.report.pds_failed += 1
                        failed[pdb.name] = error
                        logger.warning("PDS failed for %s: %s", pdb.name, error)
                    if n % 1000 == 0 or n == len(todo):
                        logger.info("PDS: %d/%d", n, len(todo))

        usable = [p.stem for p in pdbs if (self._pds_dir / f"{p.stem}.pds").is_file()
                  and p.name not in failed]
        _atomic_write(self._pds_list, "".join(f"pds/{s}.pds\n" for s in usable))
        _atomic_write(self._db_list, "".join(f"pdb/{s}.pdb\n" for s in usable))
        self.report.in_database = len(usable)

        # Keep metadata aligned with what is searchable: add the unrecorded,
        # drop the entries whose files are gone.
        metadata = self.load_metadata()
        wanted = {f"pdb/{s}.pdb" for s in usable}
        changed = False
        for key in list(metadata):
            if key not in wanted:
                del metadata[key]
                changed = True
        for key in wanted - metadata.keys():
            parts = Path(key).stem.split("_")
            metadata[key] = {
                "pdb_id": parts[0],
                "chain": parts[1] if len(parts) > 1 else "",
                "chain_type": parts[2] if len(parts) > 2 else "",
            }
            changed = True
        if changed or not self._meta_path.exists():
            _atomic_write(self._meta_path, json.dumps(metadata, indent=2))

    # ---- skipped.json --------------------------------------------------- #

    def _load_skipped(self) -> dict[str, str]:
        if not self._skipped_path.exists():
            return {}
        try:
            data = json.loads(self._skipped_path.read_text())
        except json.JSONDecodeError:
            logger.warning("Ignoring unreadable %s", self._skipped_path)
            return {}
        return data if isinstance(data, dict) else {}

    def _save_skipped(self, skipped: dict[str, str]) -> None:
        _atomic_write(self._skipped_path, json.dumps(skipped, indent=2, sort_keys=True))


def _atomic_write(path: Path, text: str) -> None:
    """Write *text* to *path* via a temporary file and a rename."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _discover_chains(struct_path: Path, fmt: str = "pdb") -> list[str]:
    """Return unique chain IDs with ATOM records in a structure file, in order."""
    if fmt == "cif":
        from Bio import PDB as biopdb
        parser = biopdb.MMCIFParser(QUIET=True)
        structure = parser.get_structure("tmp", str(struct_path))
        model = next(iter(structure))
        return list(model.child_dict.keys())

    seen: list[str] = []
    seen_set: set[str] = set()
    with struct_path.open() as fh:
        for line in fh:
            if line.startswith("ATOM") and len(line) >= 22:
                chain_id = line[21]
                if chain_id not in seen_set:
                    seen_set.add(chain_id)
                    seen.append(chain_id)
    return seen
