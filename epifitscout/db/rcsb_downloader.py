"""RcsbDownloader: fetch PDB files from RCSB and cache locally.

Downloads the biological-assembly PDB file for a given PDB ID from
https://files.rcsb.org/download/{pdb_id}.pdb. Files are cached in a
user-specified directory; re-running with the same ID is a no-op.

Usage::

    downloader = RcsbDownloader(cache_dir=Path("data/rcsb_cache"))
    pdb_path = downloader.download("1abc")
"""

from __future__ import annotations

import logging
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)

_RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


class RcsbDownloadError(RuntimeError):
    """Raised when a PDB file cannot be downloaded from RCSB."""


class RcsbDownloader:
    """Downloads and caches PDB files from RCSB.

    Args:
        cache_dir: Directory to store downloaded PDB files. Created if absent.

    Usage::

        dl = RcsbDownloader(Path("data/rcsb_cache"))
        pdb_path = dl.download("1abc")   # downloads once; subsequent calls return cache
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, pdb_id: str) -> Path:
        """Return path to a cached PDB file, downloading from RCSB if needed.

        Args:
            pdb_id: 4-character PDB ID (case-insensitive).

        Returns:
            Path to the local PDB file.

        Raises:
            RcsbDownloadError: If download fails (HTTP error, timeout, etc.).
        """
        pdb_id = pdb_id.lower()
        dest = self._cache_dir / f"{pdb_id}.pdb"

        if dest.exists():
            logger.debug("Cache hit: %s", dest)
            return dest

        url = _RCSB_URL.format(pdb_id=pdb_id)
        logger.info("Downloading %s from RCSB ...", pdb_id.upper())

        tmp = dest.with_suffix(".tmp")
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
        return dest

    def is_cached(self, pdb_id: str) -> bool:
        """Return True if the PDB file is already in the local cache."""
        return (self._cache_dir / f"{pdb_id.lower()}.pdb").exists()
