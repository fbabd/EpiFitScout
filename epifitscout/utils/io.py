"""I/O utilities: load Fragment from PDB file, save/load fragment coords."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Backbone atom names in order matching Fragment.coords axis 1
BACKBONE_ATOMS = ("N", "CA", "C", "O")


def extract_backbone_coords(
    pdb_path: Path,
    chain_id: str,
    residue_numbers: list[int],
) -> np.ndarray:
    """Extract backbone coords from a PDB file for specified residues.

    Args:
        pdb_path: Path to the PDB file.
        chain_id: Chain identifier (single character).
        residue_numbers: List of residue sequence numbers (1-indexed) to extract.

    Returns:
        (N, 4, 3) array with backbone atoms [N, CA, C, O] for each residue.

    Raises:
        ValueError: If a required atom is missing for any requested residue.
    """
    try:
        from Bio import PDB as biopdb
    except ImportError as e:
        raise ImportError("BioPython is required for PDB parsing.") from e

    parser = biopdb.PDBParser(QUIET=True)
    structure = parser.get_structure("tmp", str(pdb_path))

    # Use first model
    model = next(iter(structure))

    try:
        chain = model[chain_id]
    except KeyError:
        raise ValueError(f"Chain '{chain_id}' not found in {pdb_path}")

    res_set = set(residue_numbers)
    res_map: dict[int, dict[str, np.ndarray]] = {}

    for residue in chain:
        het, resseq, _ = residue.get_id()
        if het.strip():
            continue  # skip HETATM
        if resseq not in res_set:
            continue
        # Resolve backbone atoms — try OXT as fallback for missing O at C-terminus
        atoms: dict[str, np.ndarray] = {}
        for atom_name in BACKBONE_ATOMS:
            if atom_name in residue:
                atoms[atom_name] = residue[atom_name].get_vector().get_array()
            elif atom_name == "O" and "OXT" in residue:
                atoms[atom_name] = residue["OXT"].get_vector().get_array()

        # Skip residue if any backbone atom is still missing
        missing = [a for a in BACKBONE_ATOMS if a not in atoms]
        if missing:
            logger.debug(
                "Skipping residue %d chain %s of %s — missing backbone atom(s): %s",
                resseq, chain_id, pdb_path.name, missing,
            )
            continue

        res_map[resseq] = atoms

    coords_list: list[np.ndarray] = []
    for resnum in residue_numbers:
        if resnum not in res_map:
            logger.debug("Residue %d not found in chain %s of %s — skipping", resnum, chain_id, pdb_path)
            continue
        atoms = res_map[resnum]
        row = np.zeros((4, 3), dtype=np.float32)
        for i, atom_name in enumerate(BACKBONE_ATOMS):
            row[i] = atoms[atom_name]
        coords_list.append(row)

    if not coords_list:
        raise ValueError(f"No complete residues found for chain {chain_id} of {pdb_path}")

    return np.stack(coords_list, axis=0).astype(np.float32)


def _predict_carbonyl_O(
    ca: np.ndarray,
    c: np.ndarray,
    n_next: np.ndarray,
) -> np.ndarray:
    """Predict carbonyl O position from peptide plane geometry.

    The carbonyl O is sp2 hybridised — it lies in the plane of CA(i)-C(i)-N(i+1)
    on the opposite side of C from the bisector of the CA and N_next directions.
    C-O bond length is ~1.23 Å.

    Args:
        ca:     CA coordinates of residue i.
        c:      C  coordinates of residue i.
        n_next: N  coordinates of residue i+1.

    Returns:
        Predicted (3,) O coordinate array.
    """
    ca_dir = ca - c
    n_dir  = n_next - c
    ca_norm = np.linalg.norm(ca_dir)
    n_norm  = np.linalg.norm(n_dir)
    if ca_norm < 1e-6 or n_norm < 1e-6:
        raise ValueError("Degenerate peptide geometry — cannot predict O")
    bisector = ca_dir / ca_norm + n_dir / n_norm
    bisector_norm = np.linalg.norm(bisector)
    if bisector_norm < 1e-6:
        raise ValueError("Degenerate peptide geometry — cannot predict O")
    o_dir = -bisector / bisector_norm   # O is opposite the CA/N bisector
    return (c + o_dir * 1.23).astype(np.float32)


def extract_backbone_coords_by_range(
    pdb_path: Path,
    chain_id: str,
    res_start: int,
    res_end: int,
) -> tuple[np.ndarray, list[int], str]:
    """Extract backbone coords for all residues in [res_start, res_end] (inclusive).

    Handles IMGT insertion codes (e.g. 111A, 112A) by including all residues
    whose integer resseq falls within the range.

    Args:
        pdb_path: Path to PDB file.
        chain_id: Chain identifier.
        res_start: First residue number (inclusive).
        res_end: Last residue number (inclusive).

    Returns:
        Tuple of (coords, residue_numbers, sequence) where:
          - coords: (N, 4, 3) float32 backbone array
          - residue_numbers: list of integer resseq numbers (may repeat for insertion codes)
          - sequence: one-letter amino acid string
    """
    try:
        from Bio import PDB as biopdb
        from Bio.SeqUtils import seq1
    except ImportError as e:
        raise ImportError("BioPython is required for PDB parsing.") from e

    parser = biopdb.PDBParser(QUIET=True)
    structure = parser.get_structure("tmp", str(pdb_path))
    model = next(iter(structure))

    try:
        chain = model[chain_id]
    except KeyError:
        raise ValueError(f"Chain '{chain_id}' not found in {pdb_path}")

    # First pass: collect all residues in range.
    # Each candidate: (resseq, aa, atom_coords_dict, complete_row | None)
    # atom_coords_dict is kept for partial residues so interpolation can use it.
    candidates: list[tuple[int, str, dict[str, np.ndarray], np.ndarray | None]] = []

    for residue in chain:
        het, resseq, _icode = residue.get_id()
        if het.strip():
            continue  # skip HETATM
        if resseq < res_start or resseq > res_end:
            continue

        # Resolve backbone atoms — try OXT as fallback for missing O at C-terminus
        atom_coords: dict[str, np.ndarray] = {}
        for atom_name in BACKBONE_ATOMS:
            if atom_name in residue:
                atom_coords[atom_name] = residue[atom_name].get_vector().get_array()
            elif atom_name == "O" and "OXT" in residue:
                atom_coords[atom_name] = residue["OXT"].get_vector().get_array()

        missing = [a for a in BACKBONE_ATOMS if a not in atom_coords]

        if len(missing) == 4:
            # No ATOM records at all — ghost residue from SEQRES.
            # Not written to PDB so createPDS never sees it; skip silently.
            logger.debug(
                "Skipping ghost residue %d%s chain %s of %s — no backbone atoms present",
                resseq, _icode.strip(), chain_id, pdb_path.name,
            )
            continue
        elif missing:
            # Partially present — store atom_coords for possible interpolation.
            candidates.append((resseq, seq1(residue.resname), atom_coords, None))
        else:
            row = np.zeros((4, 3), dtype=np.float32)
            for j, atom_name in enumerate(BACKBONE_ATOMS):
                row[j] = atom_coords[atom_name]
            candidates.append((resseq, seq1(residue.resname), atom_coords, row))

    if not candidates:
        raise ValueError(
            f"No residues found in chain {chain_id} range {res_start}-{res_end} of {pdb_path}"
        )

    # Second pass: attempt O interpolation for residues missing only O.
    # Carbonyl O lies in the peptide plane (CA-C-N_next), ~1.23 Å from C.
    # Only possible when N of the next residue is available.
    for i, (resseq, aa, atom_coords, row) in enumerate(candidates):
        if row is not None:
            continue
        missing = [a for a in BACKBONE_ATOMS if a not in atom_coords]
        if missing != ["O"]:
            continue  # only handle the single-missing-O case

        # Find N of the next residue in the candidate list
        n_next: np.ndarray | None = None
        for j in range(i + 1, len(candidates)):
            next_coords = candidates[j][2]
            if "N" in next_coords:
                n_next = next_coords["N"]
                break

        if n_next is None:
            continue  # can't predict without N(i+1) — leave as None

        ca = atom_coords["CA"]
        c  = atom_coords["C"]
        o_pred = _predict_carbonyl_O(ca, c, n_next)
        atom_coords["O"] = o_pred

        complete_row = np.zeros((4, 3), dtype=np.float32)
        for j, atom_name in enumerate(BACKBONE_ATOMS):
            complete_row[j] = atom_coords[atom_name]
        candidates[i] = (resseq, aa, atom_coords, complete_row)
        logger.debug(
            "Interpolated carbonyl O for residue %d chain %s of %s",
            resseq, chain_id, pdb_path.name,
        )

    # Third pass: terminal incomplete residues are trimmed; mid-range ones
    # invalidate the whole fragment (backbone gap corrupts structural geometry).
    first_complete = next((i for i, (_, _, _, r) in enumerate(candidates) if r is not None), None)
    last_complete  = next((i for i, (_, _, _, r) in enumerate(reversed(candidates)) if r is not None), None)

    if first_complete is None:
        raise ValueError(
            f"No residues with complete backbone in chain {chain_id} "
            f"range {res_start}-{res_end} of {pdb_path}"
        )

    last_complete_idx = len(candidates) - 1 - last_complete

    for i, (resseq, aa, _, row) in enumerate(candidates):
        if row is None and first_complete < i < last_complete_idx:
            raise ValueError(
                f"Mid-range residue {resseq} in chain {chain_id} of {pdb_path.name} "
                f"has incomplete backbone — skipping entire fragment"
            )

    # Build output from the trimmed (terminal-stripped) slice
    rows: list[np.ndarray] = []
    resnums: list[int] = []
    seq_chars: list[str] = []

    for i, (resseq, aa, _, row) in enumerate(candidates):
        if row is None:
            logger.debug(
                "Trimming incomplete terminal residue %d chain %s of %s",
                resseq, chain_id, pdb_path.name,
            )
            continue
        rows.append(row)
        resnums.append(resseq)
        seq_chars.append(aa)

    if not rows:
        raise ValueError(
            f"No complete residues remain in chain {chain_id} "
            f"range {res_start}-{res_end} of {pdb_path}"
        )

    return np.stack(rows, axis=0), resnums, "".join(seq_chars)


def save_coords(coords: np.ndarray, path: Path) -> None:
    """Save a (N, 4, 3) backbone coordinate array to a .npy file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), coords)


def load_coords(path: Path) -> np.ndarray:
    """Load a (N, 4, 3) backbone coordinate array from a .npy file."""
    coords: np.ndarray = np.load(str(path))
    return coords


def fragment_to_pdb_string(
    coords: np.ndarray,
    sequence: str = "",
    chain_id: str = "A",
    start_resnum: int = 1,
) -> str:
    """Serialise a (N, 4, 3) backbone fragment as a minimal PDB-format string.

    Used by MASTERRunner to write query files.

    Args:
        coords: (N, 4, 3) backbone coordinate array.
        sequence: One-letter amino acid sequence (optional; falls back to ALA).
        chain_id: Chain identifier character.
        start_resnum: Starting residue number.

    Returns:
        PDB-format string with ATOM records.
    """
    from Bio.Data.IUPACData import protein_letters_1to3

    n_res = coords.shape[0]
    lines: list[str] = []
    atom_serial = 1

    for i in range(n_res):
        resnum = start_resnum + i
        one_letter = sequence[i] if i < len(sequence) else "A"
        three_letter = protein_letters_1to3.get(one_letter.upper(), "ALA")

        for j, atom_name in enumerate(BACKBONE_ATOMS):
            x, y, z = coords[i, j]
            # Skip zero-filled placeholder atoms
            if x == 0.0 and y == 0.0 and z == 0.0 and atom_name != "CA":
                continue
            lines.append(
                f"ATOM  {atom_serial:5d}  {atom_name:<3s} {three_letter:>3s} "
                f"{chain_id}{resnum:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>1s}"
            )
            atom_serial += 1

    lines.append("END")
    return "\n".join(lines)
