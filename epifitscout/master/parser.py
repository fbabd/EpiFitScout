"""Parse MASTER match output files into (Fragment, Superposition, rmsd) tuples.

MASTER v1.5/1.6 match line format::

    <rmsd>  <path/to/target.pds>  [(seg_start,seg_end)]

PDS files live in the ``pds/`` subdirectory of the database; the corresponding
PDB files (used to reload coordinates) are in the sibling ``pdb/`` subdirectory.
The fragment PDB filename encodes identity as ``{PDBID}_{CHAIN}_{TYPE}.pdb``,
e.g. ``12E8_H_H.pdb`` or ``1ABC_A_.pdb``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from epifitscout.types.fragment import Fragment
from epifitscout.types.superposition import Superposition
from epifitscout.utils.io import extract_backbone_coords_by_range

logger = logging.getLogger(__name__)

# e.g.: "  0.87428 /abs/path/to/pds/4KUZ_H_H.pds [(0,18)]"
_MATCH_RE = re.compile(
    r"^\s*(?P<rmsd>\S+)\s+(?P<path>\S+)\s+\[\((?P<seg_start>\d+),(?P<seg_end>\d+)\)\]"
)


class MASTERParseError(ValueError):
    """Raised when the MASTER output file cannot be parsed."""


def parse_match_file(
    match_path: Path,
    rmsd_threshold: float = 2.0,
    max_hits: int = 500,
) -> list[tuple[Fragment, Superposition, float]]:
    """Parse a MASTER match output file.

    Coordinates are loaded from the PDB file in the ``pdb/`` sibling directory
    of the ``pds/`` directory where the matched PDS file lives.

    Args:
        match_path: Path to the MASTER --matchOut file.
        rmsd_threshold: Discard hits with RMSD above this value.
        max_hits: Maximum number of hits to return.

    Returns:
        List of (Fragment, Superposition, rmsd) tuples sorted by RMSD ascending.
    """
    if not match_path.exists():
        raise FileNotFoundError(f"MASTER match file not found: {match_path}")

    lines = match_path.read_text().splitlines()
    results: list[tuple[Fragment, Superposition, float]] = []

    for line in lines:
        m = _MATCH_RE.match(line)
        if m is None:
            continue

        try:
            rmsd_val = float(m.group("rmsd"))
        except ValueError:
            continue

        if rmsd_val > rmsd_threshold:
            continue

        pds_path = Path(m.group("path"))
        pdb_path = _pdb_path_for_pds(pds_path)

        if pdb_path is None or not pdb_path.exists():
            logger.debug("PDB not found for hit: %s", pds_path)
            continue

        seg_start = int(m.group("seg_start"))
        seg_end = int(m.group("seg_end"))

        try:
            fragment = _load_fragment(pdb_path, seg_start, seg_end)
        except Exception as exc:
            logger.warning("Skipping %s: %s", pdb_path.name, exc)
            continue

        results.append((fragment, Superposition.identity(), rmsd_val))

        if len(results) >= max_hits:
            break

    results.sort(key=lambda x: x[2])
    logger.info(
        "Parsed %d hits (rmsd ≤ %.2f) from %s",
        len(results), rmsd_threshold, match_path.name,
    )
    return results


def _pdb_path_for_pds(pds_path: Path) -> Path | None:
    """Resolve the PDB coordinate file from a PDS path.

    Handles two layouts:
    - New:    ``<db>/pds/NAME.pds``    → ``<db>/pdb/NAME.pdb``
    - Legacy: ``<db>/chains/NAME.pds`` → ``<db>/chains/NAME.pdb``
    """
    pds_dir = pds_path.parent
    if pds_dir.name == "pds":
        pdb_dir = pds_dir.parent / "pdb"
        return pdb_dir / pds_path.with_suffix(".pdb").name
    return pds_path.with_suffix(".pdb")


def _load_fragment(pdb_path: Path, seg_start: int, seg_end: int) -> Fragment:
    """Load a Fragment from a chain PDB file, sliced by MASTER segment indices.

    PDB filenames follow ``{PDBID}_{CHAIN}_{TYPE}.pdb``,
    e.g. ``12E8_H_H.pdb`` or ``1ABC_A_.pdb``.
    """
    stem_parts = pdb_path.stem.split("_")
    pdb_id = stem_parts[0] if len(stem_parts) >= 1 else pdb_path.stem
    chain = stem_parts[1] if len(stem_parts) >= 2 else "A"
    chain_type = stem_parts[2] if len(stem_parts) >= 3 else ""

    coords, resnums, sequence = extract_backbone_coords_by_range(
        pdb_path, chain, res_start=1, res_end=9999
    )

    end_idx = min(seg_end + 1, len(resnums))
    coords = coords[seg_start:end_idx]
    resnums = resnums[seg_start:end_idx]
    sequence = sequence[seg_start:end_idx]

    if len(resnums) == 0:
        raise ValueError(
            f"Empty fragment after slicing [{seg_start}:{end_idx}] in {pdb_path.name}"
        )

    return Fragment(
        pdb_id=pdb_id,
        chain=chain,
        residue_range=(resnums[0], resnums[-1]),
        coords=coords,
        sequence=sequence,
        metadata={
            "chain_type": chain_type,
            "source_pdb_path": str(pdb_path),
            "seg_start": seg_start,
            "seg_end": seg_end,
        },
    )
