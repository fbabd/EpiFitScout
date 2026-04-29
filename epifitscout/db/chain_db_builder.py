"""ChainDbBuilder: build a MASTER-searchable database from protein chains.

Accepts chain assignments from any source reader (SAbDabMetadataReader or
PdbListReader), downloads PDB files from RCSB on demand, and produces a
MASTER-compatible database with separate pdb/ and pds/ subdirectories.
Paths in pds.list and db.list are stored relative to the database root so
the database can be moved across machines.

Output layout::

    <db_dir>/
    ├── pdb/                ← full-chain PDB files (coordinate source)
    │   ├── 1ABC_H_H.pdb
    │   └── 1ABC_A_L.pdb
    ├── pds/                ← MASTER binary format
    │   ├── 1ABC_H_H.pds
    │   └── 1ABC_A_L.pds
    ├── pds.list            ← relative paths, one per line (pass to MASTER)
    ├── db.list             ← relative paths to PDB files
    └── metadata.json       ← per-chain annotation

Usage::

    from epifitscout.db.sabdab_metadata import SAbDabMetadataReader
    from epifitscout.db.rcsb_downloader import RcsbDownloader
    from epifitscout.db.chain_db_builder import ChainDbBuilder

    entries  = SAbDabMetadataReader(Path("data/SAbDab/sabdab_metadata.csv")).entries()
    dl       = RcsbDownloader(cache_dir=Path("data/rcsb_cache"))
    builder  = ChainDbBuilder(
        db_dir=Path("data/sabdab_chains.db"),
        downloader=dl,
        create_pds_binary=Path("MASTER/bin/createPDS"),
    )
    pds_list = builder.build_from_sabdab(entries)
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from epifitscout.db.rcsb_downloader import RcsbDownloader, StructureFile
from epifitscout.db.sabdab_metadata import SAbDabEntry
from epifitscout.utils.io import extract_backbone_coords_by_range, fragment_to_pdb_string

logger = logging.getLogger(__name__)


class ChainDbBuilder:
    """Builds a MASTER-searchable protein chain database.

    Args:
        db_dir: Root directory for the database.
        downloader: RcsbDownloader instance for fetching PDB files.
        create_pds_binary: Path to MASTER's ``createPDS`` binary.
    """

    def __init__(
        self,
        db_dir: Path,
        downloader: RcsbDownloader,
        create_pds_binary: Path,
    ) -> None:
        self._db_dir = Path(db_dir)
        self._pdb_dir = self._db_dir / "pdb"
        self._pds_dir = self._db_dir / "pds"
        self._db_list = self._db_dir / "db.list"
        self._pds_list = self._db_dir / "pds.list"
        self._meta_path = self._db_dir / "metadata.json"
        self._downloader = downloader
        self._create_pds = Path(create_pds_binary)

        if not self._create_pds.exists():
            raise FileNotFoundError(
                f"createPDS binary not found at {self._create_pds}"
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build_from_sabdab(self, entries: list[SAbDabEntry]) -> Path:
        """Build database from SAbDab metadata entries.

        Args:
            entries: List of SAbDabEntry from SAbDabMetadataReader.

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
        """Build database from a generic (pdb_id, chain_id | None) list.

        When chain_id is None, all protein chains in the PDB are extracted.

        Args:
            chain_assignments: From PdbListReader.chain_assignments().

        Returns:
            Path to ``pds.list``.
        """
        expanded: list[tuple[str, str, str]] = []
        for pdb_id, chain_id in chain_assignments:
            if chain_id is None:
                try:
                    sf = self._downloader.download(pdb_id)
                    for cid in _discover_chains(sf.path, sf.fmt):
                        expanded.append((pdb_id, cid, ""))
                except Exception as exc:
                    logger.warning("Skipping %s (download error): %s", pdb_id, exc)
            else:
                expanded.append((pdb_id, chain_id, ""))
        return self._build(expanded)

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
        """Core build loop: download → extract → write PDB → convert PDS → lists."""
        self._pdb_dir.mkdir(parents=True, exist_ok=True)
        self._pds_dir.mkdir(parents=True, exist_ok=True)

        pdb_rel_paths: list[str] = []
        pds_rel_paths: list[str] = []
        metadata: dict[str, dict] = {}
        skipped = 0
        total = len(assignments)

        for i, (pdb_id, chain_id, chain_type) in enumerate(assignments, 1):
            try:
                rcsb_sf = self._downloader.download(pdb_id)
            except Exception as exc:
                logger.warning("Download failed for %s: %s", pdb_id, exc)
                skipped += 1
                continue

            out_pdb = self._write_chain_pdb(rcsb_sf, pdb_id, chain_id, chain_type)
            if out_pdb is None:
                skipped += 1
                continue

            out_pds = self._convert_to_pds(out_pdb)
            if out_pds is None:
                skipped += 1
                out_pdb.unlink(missing_ok=True)
                continue

            pdb_rel = str(out_pdb.relative_to(self._db_dir))
            pds_rel = str(out_pds.relative_to(self._db_dir))
            pdb_rel_paths.append(pdb_rel)
            pds_rel_paths.append(pds_rel)
            metadata[pdb_rel] = {
                "pdb_id": pdb_id,
                "chain": chain_id,
                "chain_type": chain_type,
            }

            if i % 500 == 0 or i == total:
                logger.info(
                    "Progress: %d/%d | %d written, %d skipped",
                    i, total, len(pdb_rel_paths), skipped,
                )

        self._db_list.write_text("\n".join(pdb_rel_paths) + "\n")
        self._pds_list.write_text("\n".join(pds_rel_paths) + "\n")
        with self._meta_path.open("w") as fh:
            json.dump(metadata, fh, indent=2)

        logger.info(
            "Chain DB built: %d chains written, %d skipped. pds.list: %s",
            len(pds_rel_paths), skipped, self._pds_list,
        )
        return self._pds_list

    def _write_chain_pdb(
        self,
        rcsb_sf: StructureFile,
        pdb_id: str,
        chain_id: str,
        chain_type: str,
    ) -> Path | None:
        """Extract one chain from an RCSB structure file and write to pdb/ dir."""
        suffix = f"_{chain_type}" if chain_type else ""
        filename = f"{pdb_id}_{chain_id}{suffix}.pdb"
        out_path = self._pdb_dir / filename

        if out_path.exists():
            return out_path

        try:
            coords, resnums, sequence = extract_backbone_coords_by_range(
                rcsb_sf.path, chain_id, res_start=1, res_end=9999, fmt=rcsb_sf.fmt
            )
        except ValueError as exc:
            logger.debug("Skipping %s chain %s: %s", pdb_id, chain_id, exc)
            return None

        if len(resnums) < 4:
            logger.debug(
                "Skipping %s chain %s: too short (%d residues)",
                pdb_id, chain_id, len(resnums),
            )
            return None

        try:
            pdb_str = fragment_to_pdb_string(
                coords,
                sequence=sequence,
                chain_id=chain_id,
                start_resnum=resnums[0],
            )
            out_path.write_text(pdb_str)
        except Exception as exc:
            logger.warning(
                "Failed to write PDB for %s chain %s: %s", pdb_id, chain_id, exc
            )
            return None

        return out_path

    def _convert_to_pds(self, pdb_path: Path) -> Path | None:
        """Convert a PDB file to MASTER PDS format into pds/ dir."""
        pds_path = self._pds_dir / pdb_path.with_suffix(".pds").name
        if pds_path.exists():
            return pds_path

        cmd = [
            str(self._create_pds),
            "--type", "target",
            "--pdb", str(pdb_path),
            "--pds", str(pds_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning(
                "createPDS failed for %s (code %d): %s",
                pdb_path.name, result.returncode, result.stderr[:200],
            )
            return None

        return pds_path


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
