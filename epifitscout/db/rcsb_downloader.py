"""RcsbDownloader: fetch PDB or mmCIF files from RCSB and cache locally.

Tries PDB format first; falls back to mmCIF (available for all RCSB entries)
when RCSB returns HTTP 404 for the PDB format.

Usage::

    downloader = RcsbDownloader(cache_dir=Path("data/rcsb_cache"))
    sf = downloader.download("1abc")   # StructureFile(path=..., fmt="pdb"|"cif")
"""

from __future__ import annotations

import logging
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


class RcsbDownloadError(RuntimeError):
    """Raised when a structure file cannot be downloaded from RCSB."""


@dataclass(frozen=True)
class StructureFile:
    """A locally cached structure file and its format."""

    path: Path
    fmt: str  # "pdb" or "cif"


class RcsbDownloader:
    """Downloads and caches PDB/mmCIF files from RCSB.

    Tries PDB format first; on HTTP 404 falls back to mmCIF.

    Args:
        cache_dir: Directory to store downloaded files. Created if absent.

    Usage::

        dl = RcsbDownloader(Path("data/rcsb_cache"))
        sf = dl.download("1abc")   # StructureFile(path=..., fmt="pdb"|"cif")
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, pdb_id: str) -> StructureFile:
        """Return a cached structure file, downloading from RCSB if needed.

        Tries PDB format first; on HTTP 404 falls back to mmCIF.

        Args:
            pdb_id: 4-character PDB ID (case-insensitive).

        Returns:
            StructureFile with path and fmt ("pdb" or "cif").

        Raises:
            RcsbDownloadError: If both formats fail to download.
        """
        pdb_id = pdb_id.lower()
        pdb_dest = self._cache_dir / f"{pdb_id}.pdb"
        cif_dest = self._cache_dir / f"{pdb_id}.cif"

        if pdb_dest.exists():
            logger.debug("Cache hit (pdb): %s", pdb_dest)
            return StructureFile(path=pdb_dest, fmt="pdb")
        if cif_dest.exists():
            logger.debug("Cache hit (cif): %s", cif_dest)
            return StructureFile(path=cif_dest, fmt="cif")

        try:
            self._fetch(pdb_id, _PDB_URL, pdb_dest)
            return StructureFile(path=pdb_dest, fmt="pdb")
        except RcsbDownloadError as exc:
            if "HTTP 404" not in str(exc):
                raise
            logger.info("%s has no PDB format — falling back to mmCIF", pdb_id.upper())

        self._fetch(pdb_id, _CIF_URL, cif_dest)
        return StructureFile(path=cif_dest, fmt="cif")

    def is_cached(self, pdb_id: str) -> bool:
        """Return True if either format is already in the local cache."""
        pid = pdb_id.lower()
        return (
            (self._cache_dir / f"{pid}.pdb").exists()
            or (self._cache_dir / f"{pid}.cif").exists()
        )

    def _fetch(self, pdb_id: str, url_template: str, dest: Path) -> None:
        url = url_template.format(pdb_id=pdb_id)
        logger.info("Downloading %s ...", url)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
        except urllib.error.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            raise RcsbDownloadError(
                f"RCSB returned HTTP {exc.code} for {pdb_id.upper()}: {url}"
            ) from exc
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise RcsbDownloadError(
                f"Failed to download {pdb_id.upper()}: {exc}"
            ) from exc
        logger.debug("Saved %s → %s", pdb_id.upper(), dest)
