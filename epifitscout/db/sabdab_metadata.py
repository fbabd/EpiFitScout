"""SAbDabMetadataReader: parse a SAbDab summary file into chain assignments.

Reads the summary file SAbDab distributes — which is **tab-separated** despite
its CSV-style name — and also a comma-separated export of the same columns. The
delimiter is detected from the header line, so either works unchanged.

Columns used (extra columns are ignored)::

    pdb, Hchain, Lchain, antigen_chain          required
    H1, H2, H3, L1, L2, L3                      optional CDR sequences

SAbDab writes ``NA`` for an absent chain; that, ``nan``, ``none`` and blank are
all treated as "no chain".

Usage::

    reader = SAbDabMetadataReader(Path("database_builder/sabdab_summary.tsv"))
    for entry in reader.entries():
        print(entry.pdb_id, entry.hchain, entry.lchain)
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SAbDabEntry:
    """One H/L chain pair from the SAbDab metadata CSV.

    Attributes:
        pdb_id: Uppercase 4-character PDB ID.
        hchain: Heavy chain ID (empty string if absent).
        lchain: Light chain ID (empty string if absent).
        antigen_chain: Antigen chain ID(s), space-separated (may be empty).
        cdr_seqs: Dict mapping CDR type (H1-L3) to sequence string.
    """

    pdb_id: str
    hchain: str
    lchain: str
    antigen_chain: str
    cdr_seqs: dict[str, str] = field(default_factory=dict)

    def chain_assignments(self) -> list[tuple[str, str]]:
        """Return list of (chain_id, chain_type) pairs for this entry.

        Only includes chains that are non-empty strings.
        """
        pairs: list[tuple[str, str]] = []
        if self.hchain:
            pairs.append((self.hchain, "H"))
        if self.lchain:
            pairs.append((self.lchain, "L"))
        return pairs


class SAbDabMetadataReader:
    """Reads the SAbDab summary CSV and yields SAbDabEntry objects.

    Deduplicates by (pdb_id, hchain, lchain) — SAbDab often repeats the
    same pair when there are multiple antigen chains.

    Args:
        csv_path: Path to the SAbDab metadata CSV file.

    Usage::

        reader = SAbDabMetadataReader(Path("data/SAbDab/sabdab_metadata.csv"))
        entries = list(reader.entries())
    """

    _CDR_COLS = ("H1", "H2", "H3", "L1", "L2", "L3")

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = Path(csv_path)
        if not self._csv_path.exists():
            raise FileNotFoundError(f"SAbDab metadata CSV not found: {self._csv_path}")

    def entries(self) -> list[SAbDabEntry]:
        """Parse CSV and return deduplicated list of SAbDabEntry objects."""
        seen: set[tuple[str, str, str]] = set()
        results: list[SAbDabEntry] = []

        with self._csv_path.open(newline="") as fh:
            header = fh.readline()
            fh.seek(0)
            delimiter = "\t" if header.count("\t") > header.count(",") else ","
            reader = csv.DictReader(fh, delimiter=delimiter)

            missing = {"pdb", "Hchain", "Lchain"} - set(reader.fieldnames or [])
            if missing:
                found = ", ".join((reader.fieldnames or [])[:8]) or "(no header)"
                raise ValueError(
                    f"{self._csv_path.name} is not a SAbDab summary file: missing "
                    f"column(s) {sorted(missing)}. Columns found: {found}"
                )

            for row in reader:
                pdb_id = row.get("pdb", "").strip().upper()
                hchain = row.get("Hchain", "").strip()
                lchain = row.get("Lchain", "").strip()

                if not pdb_id:
                    continue

                # Normalise "nan" / empty to empty string
                if hchain.lower() in ("nan", "none", "na", ""):
                    hchain = ""
                if lchain.lower() in ("nan", "none", "na", ""):
                    lchain = ""
                if not hchain and not lchain:
                    continue

                key = (pdb_id, hchain, lchain)
                if key in seen:
                    continue
                seen.add(key)

                cdr_seqs = {
                    col: row.get(col, "").strip()
                    for col in self._CDR_COLS
                    if row.get(col, "").strip()
                }

                results.append(
                    SAbDabEntry(
                        pdb_id=pdb_id,
                        hchain=hchain,
                        lchain=lchain,
                        antigen_chain=row.get("antigen_chain", "").strip(),
                        cdr_seqs=cdr_seqs,
                    )
                )

        if not results:
            raise ValueError(
                f"{self._csv_path.name} contains no usable rows (every row lacked "
                "a PDB id or both chains)."
            )
        logger.info(
            "Loaded %d unique H/L pairs from %s", len(results), self._csv_path.name
        )
        return results
