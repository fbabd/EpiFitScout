"""MASTERRunner: subprocess wrapper for the MASTER C++ binary (v1.5/1.6).

MASTER requires PDS-format files, not raw PDB. This module:
  1. Writes the query Fragment as a temporary PDB file.
  2. Converts it to PDS using the accompanying ``createPDS`` binary.
  3. Resolves relative paths in pds.list to absolute (MASTER requires absolute).
  4. Runs MASTER search against the resolved PDS list.

When ``max_workers > 1``, the database is split into shards and searched in
parallel — one MASTER process per shard — then results are merged and sorted
by RMSD before returning.
"""

from __future__ import annotations

import atexit
import logging
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from epifitscout.config.schema import MASTERConfig
from epifitscout.types.fragment import Fragment
from epifitscout.utils.io import fragment_to_pdb_string

logger = logging.getLogger(__name__)

_RMSD_RE = re.compile(r"^\s*(\S+)\s+")


def _rmtree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def resolve_pds_list(list_path: Path) -> Path:
    """Return a temp file with all paths in list_path resolved to absolute."""
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

    When ``max_workers == 1`` (default), runs a single MASTER process against
    the full database. When ``max_workers > 1``, splits the database into
    shards and searches them in parallel, merging results by RMSD.

    Args:
        cfg: MASTER configuration.
        max_workers: Number of parallel MASTER processes. 1 = sequential.

    Usage::

        runner = MASTERRunner(cfg, max_workers=6)
        match_path = runner.run(query_fragment, output_dir=Path("/tmp/run1"))
    """

    def __init__(self, cfg: MASTERConfig, max_workers: int = 1) -> None:
        self._cfg = cfg
        self._binary = Path(cfg.binary_path)
        self._db = Path(cfg.database_path)
        self._create_pds = self._binary.parent / "createPDS"
        self._max_workers = max_workers

        # Resolve full PDS list once.
        self._resolved_list = resolve_pds_list(self._db)
        atexit.register(self._resolved_list.unlink, missing_ok=True)

        # Pre-build shards if parallel search is requested.
        if max_workers > 1:
            self._shards = self._make_shards(max_workers)
            for shard in self._shards:
                atexit.register(shard.unlink, missing_ok=True)
            logger.info(
                "MASTERRunner: %d shards, max_workers=%d",
                len(self._shards), max_workers,
            )
        else:
            self._shards = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, query: Fragment, output_dir: Path) -> Path:
        """Run MASTER search — single process, full database.

        Args:
            query: The query fragment.
            output_dir: Directory to write intermediate and output files.

        Returns:
            Path to the MASTER match output file.

        Raises:
            MASTERExecutionError: If MASTER or createPDS exits with non-zero code.
        """
        self._validate_paths()
        output_dir.mkdir(parents=True, exist_ok=True)

        query_pdb = self._write_query_pdb(query, output_dir)
        query_pds = self._convert_to_pds(query_pdb, output_dir)
        match_file = output_dir / "matches.txt"

        cmd = self._build_command(query_pds, match_file, self._resolved_list)
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
        return match_file

    def run_sharded(self, query: Fragment, output_dir: Path) -> Path:
        """Run MASTER in parallel across database shards, return merged results.

        Args:
            query: The query fragment.
            output_dir: Directory to write intermediate and output files.

        Returns:
            Path to the merged match file (same format as ``run()``).

        Raises:
            MASTERExecutionError: If any MASTER shard process fails.
        """
        if not self._shards:
            logger.warning("run_sharded() called but no shards — falling back to run()")
            return self.run(query, output_dir)

        self._validate_paths()
        output_dir.mkdir(parents=True, exist_ok=True)

        query_pdb = self._write_query_pdb(query, output_dir)
        query_pds = self._convert_to_pds(query_pdb, output_dir)

        def _run_shard(args: tuple[int, Path]) -> Path:
            i, shard_list = args
            match_file = output_dir / f"matches_{i}.txt"
            cmd = self._build_command(query_pds, match_file, shard_list)
            logger.debug("Shard %d: %s", i, " ".join(str(c) for c in cmd))
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._cfg.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise MASTERExecutionError(
                    f"MASTER shard {i} timed out after {self._cfg.timeout_seconds}s"
                ) from exc
            if result.returncode != 0:
                raise MASTERExecutionError(
                    f"MASTER shard {i} failed (code {result.returncode}).\n"
                    f"stderr:\n{result.stderr}"
                )
            logger.debug("Shard %d done.", i)
            return match_file

        logger.info("Running MASTER across %d shards in parallel...", len(self._shards))
        with ThreadPoolExecutor(max_workers=len(self._shards)) as pool:
            shard_match_files = list(pool.map(_run_shard, enumerate(self._shards)))

        return self._merge_match_files(shard_match_files, output_dir)

    def run_with_tempdir(self, query: Fragment) -> Path:
        """Run MASTER using a managed temporary directory.

        The temp directory is registered for cleanup on process exit via atexit.
        """
        tmp = tempfile.mkdtemp(prefix="epifitscout_master_")
        atexit.register(_rmtree, tmp)
        if self._max_workers > 1:
            return self.run_sharded(query, output_dir=Path(tmp))
        return self.run(query, output_dir=Path(tmp))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _make_shards(self, n: int) -> list[Path]:
        """Split the resolved PDS list into n shard temp files."""
        lines = [
            l for l in self._resolved_list.read_text().splitlines() if l.strip()
        ]
        chunk_size = -(-len(lines) // n)  # ceil division
        chunks = [lines[i: i + chunk_size] for i in range(0, len(lines), chunk_size)]

        shard_paths: list[Path] = []
        for i, chunk in enumerate(chunks):
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".list",
                delete=False,
                prefix=f"epifitscout_shard{i}_",
            )
            tmp.write("\n".join(chunk) + "\n")
            tmp.flush()
            shard_paths.append(Path(tmp.name))
        return shard_paths

    def _merge_match_files(self, paths: list[Path], output_dir: Path) -> Path:
        """Merge shard match files, sort by RMSD, keep global top max_hits."""
        scored_lines: list[tuple[float, str]] = []

        for p in paths:
            if not p.exists():
                logger.warning("Shard match file missing: %s", p)
                continue
            for line in p.read_text().splitlines():
                m = _RMSD_RE.match(line)
                if m is None:
                    continue
                try:
                    rmsd = float(m.group(1))
                except ValueError:
                    continue
                scored_lines.append((rmsd, line))

        scored_lines.sort(key=lambda x: x[0])
        if self._cfg.max_hits > 0:
            scored_lines = scored_lines[: self._cfg.max_hits]

        merged_path = output_dir / "matches.txt"
        merged_path.write_text("\n".join(line for _, line in scored_lines) + "\n")
        logger.info(
            "Merged %d shards → %d hits → %s",
            len(paths), len(scored_lines), merged_path,
        )
        return merged_path

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
        self, query_pds: Path, match_file: Path, target_list: Path
    ) -> list[str]:
        """Construct the MASTER command-line argument list."""
        cmd = [
            str(self._binary),
            "--query", str(query_pds),
            "--targetList", str(target_list),
            "--rmsdCut", str(self._cfg.rmsd_threshold),
            "--matchOut", str(match_file),
        ]
        if self._cfg.max_hits > 0:
            cmd += ["--topN", str(self._cfg.max_hits)]
        if self._cfg.n_threads > 0:
            cmd += ["--nThreads", str(self._cfg.n_threads)]
        return cmd

    def _validate_paths(self) -> None:
        """Check binary and database exist before launching subprocess."""
        if not self._binary.exists():
            raise FileNotFoundError(f"MASTER binary not found at {self._binary}.")
        if not self._create_pds.exists():
            raise FileNotFoundError(f"createPDS binary not found at {self._create_pds}.")
        if not self._db.exists():
            raise FileNotFoundError(f"MASTER database not found at {self._db}.")