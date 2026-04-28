"""MASTERRunner: subprocess wrapper for the MASTER C++ binary (v1.5/1.6).

MASTER requires PDS-format files, not raw PDB. This module:
  1. Writes the query Fragment as a temporary PDB file.
  2. Converts it to PDS using the accompanying ``createPDS`` binary.
  3. Resolves relative paths in pds.list to absolute (MASTER requires absolute).
  4. Runs MASTER search against the resolved PDS list.

The database pds.list may contain paths relative to its parent directory —
this allows the database to be moved across machines. MASTERRunner creates a
temporary resolved copy before passing ``--targetList`` to MASTER.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from epifitscout.config.schema import MASTERConfig
from epifitscout.types.fragment import Fragment
from epifitscout.utils.io import fragment_to_pdb_string

logger = logging.getLogger(__name__)


def resolve_pds_list(list_path: Path) -> Path:
    """Return a temp file with all paths in list_path resolved to absolute.

    Relative paths are resolved relative to ``list_path.parent``. Absolute
    paths are kept as-is. The caller is responsible for deleting the temp file.

    Args:
        list_path: Path to the pds.list file (may contain relative paths).

    Returns:
        Path to a temporary file with all-absolute paths.
    """
    base = list_path.parent
    resolved_lines: list[str] = []
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        if not p.is_absolute():
            p = (base / p).resolve()
        resolved_lines.append(str(p))

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".list", delete=False, prefix="epifitscout_"
    )
    tmp.write("\n".join(resolved_lines) + "\n")
    tmp.flush()
    return Path(tmp.name)


class MASTERExecutionError(RuntimeError):
    """Raised when MASTER or createPDS exits with a non-zero return code."""


class MASTERRunner:
    """Wraps the MASTER C++ binary via subprocess.

    Usage::

        runner = MASTERRunner(cfg)
        match_path = runner.run(query_fragment, output_dir=Path("/tmp/run1"))
    """

    def __init__(self, cfg: MASTERConfig) -> None:
        self._cfg = cfg
        self._binary = Path(cfg.binary_path)
        self._db = Path(cfg.database_path)
        self._create_pds = self._binary.parent / "createPDS"

    def run(self, query: Fragment, output_dir: Path) -> Path:
        """Run MASTER search for a query fragment.

        Args:
            query: The query fragment.
            output_dir: Directory to write intermediate and output files.

        Returns:
            Path to the MASTER match output file.

        Raises:
            MASTERExecutionError: If MASTER or createPDS exits with non-zero code.
            FileNotFoundError: If binary or database path does not exist.
        """
        self._validate_paths()
        output_dir.mkdir(parents=True, exist_ok=True)

        query_pdb = self._write_query_pdb(query, output_dir)
        query_pds = self._convert_to_pds(query_pdb, output_dir)
        match_file = output_dir / "matches.txt"

        resolved_list = resolve_pds_list(self._db)
        try:
            cmd = self._build_command(query_pds, match_file, resolved_list)
            logger.info("Running MASTER: %s", " ".join(str(c) for c in cmd))

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._cfg.timeout_seconds,
                )
            except subprocess.TimeoutExpired as e:
                raise MASTERExecutionError(
                    f"MASTER timed out after {self._cfg.timeout_seconds}s"
                ) from e

            if result.returncode != 0:
                raise MASTERExecutionError(
                    f"MASTER exited with code {result.returncode}.\nstderr:\n{result.stderr}"
                )

            logger.info("MASTER finished. stdout: %s", result.stdout[:200])
        finally:
            resolved_list.unlink(missing_ok=True)

        return match_file

    def run_with_tempdir(self, query: Fragment) -> Path:
        """Run MASTER using a managed temporary directory."""
        tmp = tempfile.mkdtemp(prefix="epifitscout_master_")
        return self.run(query, output_dir=Path(tmp))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _write_query_pdb(self, query: Fragment, dest_dir: Path) -> Path:
        """Serialise query Fragment to PDB format."""
        query_path = dest_dir / "query.pdb"
        pdb_str = fragment_to_pdb_string(
            query.coords,
            sequence=query.sequence,
            chain_id=query.chain or "X",
            start_resnum=query.residue_range[0],
        )
        query_path.write_text(pdb_str)
        logger.debug("Wrote query PDB: %s (%d residues)", query_path, query.length)
        return query_path

    def _convert_to_pds(self, pdb_path: Path, dest_dir: Path) -> Path:
        """Convert a PDB file to MASTER PDS format using createPDS."""
        pds_path = dest_dir / "query.pds"
        cmd = [
            str(self._create_pds), "--type", "query",
            "--pdb", str(pdb_path), "--pds", str(pds_path),
        ]
        logger.debug("Running createPDS: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise MASTERExecutionError(
                f"createPDS failed (code {result.returncode}).\nstderr:\n{result.stderr}"
            )
        return pds_path

    def _build_command(
        self, query_pds: Path, match_file: Path, resolved_list: Path
    ) -> list[str]:
        """Construct the MASTER command-line argument list."""
        cmd = [
            str(self._binary),
            "--query", str(query_pds),
            "--targetList", str(resolved_list),
            "--rmsdCut", str(self._cfg.rmsd_threshold),
            "--matchOut", str(match_file),
        ]
        if self._cfg.max_hits > 0:
            cmd += ["--topN", str(self._cfg.max_hits)]
        return cmd

    def _validate_paths(self) -> None:
        """Check binary and database exist before launching subprocess."""
        if not self._binary.exists():
            raise FileNotFoundError(f"MASTER binary not found at {self._binary}.")
        if not self._create_pds.exists():
            raise FileNotFoundError(f"createPDS binary not found at {self._create_pds}.")
        if not self._db.exists():
            raise FileNotFoundError(f"MASTER database not found at {self._db}.")
