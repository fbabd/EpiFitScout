"""PdbListReader: load chain assignments from a plain PDB-ID list file.

Supports a simple text format — one entry per line::

    1abc          # all protein chains extracted
    1abc H        # heavy chain only
    1abc H L      # heavy and light chains

Blank lines and lines starting with ``#`` are ignored.

This is the generic input path for non-SAbDab proteins.  When no chain IDs
are given, all ATOM-record chains present in the downloaded PDB are used.

Usage::

    reader = PdbListReader(Path("my_pdb_list.txt"))
    for pdb_id, chain_id in reader.chain_assignments():
        ...
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PdbListReader:
    """Reads a plain-text PDB ID list and yields (pdb_id, chain_id) pairs.

    When a line specifies chain IDs, only those chains are yielded.
    When no chain IDs are given, ``chain_assignments()`` yields
    ``(pdb_id, None)`` to signal "extract all chains" — the caller
    (ChainDbBuilder) is responsible for discovering chain IDs from the PDB.

    Args:
        list_path: Path to the PDB list file.

    Usage::

        reader = PdbListReader(Path("proteins.txt"))
        assignments = reader.chain_assignments()
        # [(pdb_id, chain_id_or_None), ...]
    """

    def __init__(self, list_path: Path) -> None:
        self._list_path = Path(list_path)
        if not self._list_path.exists():
            raise FileNotFoundError(f"PDB list file not found: {self._list_path}")

    def chain_assignments(self) -> list[tuple[str, str | None]]:
        """Parse the list file and return (pdb_id, chain_id | None) pairs.

        None as chain_id means "extract all chains" from that PDB.
        Multiple chain IDs on one line expand to multiple pairs.
        """
        results: list[tuple[str, str | None]] = []

        with self._list_path.open() as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                pdb_id = parts[0].upper()

                if len(parts) == 1:
                    results.append((pdb_id, None))
                else:
                    for chain_id in parts[1:]:
                        results.append((pdb_id, chain_id))

        logger.info(
            "Loaded %d chain assignments from %s", len(results), self._list_path.name
        )
        return results
