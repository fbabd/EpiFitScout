"""BioPython-free replacements for the two epifitscout.utils.io functions the
pipeline calls. BioPython is a broken namespace shadow in this environment.

Behaviour matches the originals, including insertion-code handling (residues are
kept in file order and keyed on (resseq, icode), so IMGT 111A/112A survive).
"""
from __future__ import annotations

import numpy as np

BACKBONE_ATOMS = ("N", "CA", "C", "O")

AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
          "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
          "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
          "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O"}
AA1to3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
          "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
          "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
          "Y": "TYR", "V": "VAL"}


def _parse(path, chain_id, lo=None, hi=None):
    """Return (coords (N,4,3) float32, resnums list[int], sequence str)."""
    order, atoms, names = [], {}, {}
    with open(path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resname = line[17:20].strip().upper()
            if line.startswith("HETATM") and resname != "MSE":
                continue
            if line[21] != chain_id:
                continue
            name = line[12:16].strip()
            if name not in BACKBONE_ATOMS:
                continue
            if line[16] not in (" ", "A"):          # altloc: keep blank or A
                continue
            try:
                rn = int(line[22:26])
            except ValueError:
                continue
            if lo is not None and not (lo <= rn <= hi):
                continue
            key = (rn, line[26])                     # (resseq, icode)
            if key not in atoms:
                atoms[key] = {}
                names[key] = resname
                order.append(key)
            atoms[key][name] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))

    keys = [k for k in order if all(a in atoms[k] for a in BACKBONE_ATOMS)]
    if not keys:
        raise ValueError(f"No complete backbone residues for chain {chain_id} in {path}")
    coords = np.array([[atoms[k][a] for a in BACKBONE_ATOMS] for k in keys], dtype=np.float32)
    resnums = [k[0] for k in keys]
    seq = "".join(AA3to1.get(names[k].upper(), "X") for k in keys)
    return coords, resnums, seq


def extract_backbone_coords_by_range(pdb_path, chain_id, res_start, res_end, fmt="pdb"):
    return _parse(str(pdb_path), chain_id, int(res_start), int(res_end))


def fragment_to_pdb_string(coords, sequence="", chain_id="A", start_resnum=1):
    lines, serial = [], 1
    for i in range(coords.shape[0]):
        resnum = start_resnum + i
        one = sequence[i] if i < len(sequence) else "A"
        three = AA1to3.get(one.upper(), "ALA")
        for j, atom_name in enumerate(BACKBONE_ATOMS):
            x, y, z = coords[i, j]
            if x == 0.0 and y == 0.0 and z == 0.0 and atom_name != "CA":
                continue
            lines.append(
                f"ATOM  {serial:5d}  {atom_name:<3s} {three:>3s} "
                f"{chain_id}{resnum:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>1s}")
            serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def install():
    """Patch the pipeline's import sites."""
    import epifitscout.master.parser as parser
    import epifitscout.master.runner as runner
    import epifitscout.utils.io as io
    io.extract_backbone_coords_by_range = extract_backbone_coords_by_range
    io.fragment_to_pdb_string = fragment_to_pdb_string
    runner.fragment_to_pdb_string = fragment_to_pdb_string
    parser.extract_backbone_coords_by_range = extract_backbone_coords_by_range
