"""RcsbDownloader: fetch PDB or mmCIF files from RCSB and cache locally.

Tries PDB format first; falls back to mmCIF (available for all RCSB entries)
when RCSB returns HTTP 404 for the PDB format.

Usage::

    downloader = RcsbDownloader(cache_dir=Path("data/rcsb_cache"))
    sf = downloader.download("1abc")   # StructureFile(path=..., fmt="pdb"|"cif")
"""

from __future__ import annotations

import logging
import os
import shutil
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

_TIMEOUT_S = 60
_ATTEMPTS = 3

#: Where Linux distributions keep their CA bundle. A Python built elsewhere
#: (uv, conda, python.org) often looks only in its own build-time location, so
#: HTTPS fails with CERTIFICATE_VERIFY_FAILED on a machine where ``curl`` works.
_CA_BUNDLES = (
    "/etc/pki/tls/certs/ca-bundle.crt",      # RHEL, CentOS, Rocky, Fedora
    "/etc/ssl/certs/ca-certificates.crt",    # Debian, Ubuntu
    "/etc/ssl/ca-bundle.pem",                # openSUSE
    "/etc/ssl/cert.pem",                     # Alpine, macOS
)

_CERT_HELP = (
    "Python could not find your system's CA certificates. Point SSL_CERT_FILE "
    "at them, e.g. SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt"
)


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context that can verify RCSB on a wide range of systems.

    Order: ``SSL_CERT_FILE`` (the user's explicit choice), the platform's own
    CA bundle, ``certifi`` if installed, then Python's built-in default.
    """
    candidates: list[str] = []
    if env := os.environ.get("SSL_CERT_FILE"):
        candidates.append(env)
    candidates.extend(_CA_BUNDLES)
    for cafile in candidates:
        if os.path.isfile(cafile):
            return ssl.create_default_context(cafile=cafile)
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


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
        context = _ssl_context()
        last: Exception | None = None

        for attempt in range(1, _ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(url, timeout=_TIMEOUT_S, context=context) as resp:
                    with open(tmp, "wb") as out:
                        shutil.copyfileobj(resp, out)
                tmp.rename(dest)
                logger.debug("Saved %s → %s", pdb_id.upper(), dest)
                return
            except urllib.error.HTTPError as exc:
                # 404 and friends are answers, not glitches; retrying will not help.
                tmp.unlink(missing_ok=True)
                raise RcsbDownloadError(
                    f"RCSB returned HTTP {exc.code} for {pdb_id.upper()}: {url}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                tmp.unlink(missing_ok=True)
                reason = getattr(exc, "reason", exc)
                if isinstance(reason, ssl.SSLCertVerificationError):
                    raise RcsbDownloadError(
                        f"Cannot verify RCSB's certificate for {pdb_id.upper()}: "
                        f"{reason}. {_CERT_HELP}"
                    ) from exc
                last = exc
                if attempt < _ATTEMPTS:
                    logger.debug("Attempt %d for %s failed (%s); retrying",
                                 attempt, pdb_id.upper(), exc)
                    time.sleep(2 * attempt)

        raise RcsbDownloadError(
            f"Failed to download {pdb_id.upper()} after {_ATTEMPTS} attempts: {last}"
        ) from last
