"""Driving an external side-chain packer, and checking what it gave back.

Side chains are not optional in v2, they are the stage-2 substrate. A retrieved
loop arrives as backbone only — the chain database stores nothing else — and
complementarity is a side-chain property: how a loop meshes with an epitope is
decided by the atoms this module puts back. v1 ranked on backbone geometry and
measured a quantity with almost no spread; packing first is what makes the
interface measurement mean anything.

So every candidate that passes the feasibility gate is rebuilt all-atom here,
with the donor's sequence and the template's environment, before it is scored.

What is packed, and what is left alone
--------------------------------------
The graft and the residues within a shell of it. The rest of the complex keeps
the template's own experimental rotamers, because rewriting a deposited model to
place one loop trades known coordinates for predicted ones. ``repack_all``
lifts that restriction.

Nothing the packer returns is trusted. FASPR exits 0 and writes an empty file
when its ``-s`` sequence length disagrees with the structure; it silently drops
heteroatoms; and a misaligned sequence comes back as a complex with different
residue identities and no error at all. Each of those is checked for by name in
:func:`_verify_pack` and :func:`run_packer`.

This module was lifted out of ``epifitscout.tools.build_complex`` so the
pipeline can pack without going through the command-line tool; the tool now
imports it.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from epifitscout.paths import REPO_ROOT

logger = logging.getLogger(__name__)

__all__ = [
    "BACKBONE",
    "BREAK_THRESHOLD",
    "DEFAULT_REPACK_SHELL",
    "HBOND_MIN",
    "ONE_TO_THREE",
    "PEPTIDE_BOND",
    "THREE_TO_ONE",
    "VDW_MIN",
    "WATER_NAMES",
    "check_packer",
    "closest_contact",
    "faspr_residues",
    "find_packer",
    "repack_sequence",
    "run_packer",
    "summarise_residues",
]

BACKBONE = ("N", "CA", "C", "O")

#: Ideal peptide bond length; used to judge whether a graft joins the framework.
PEPTIDE_BOND = 1.33
#: Beyond this the backbone is considered broken at the junction.
BREAK_THRESHOLD = 2.0

#: Packers this module knows how to drive, and the argument order they expect.
_PACKERS = {
    "FASPR": ["{exe}", "-i", "{inp}", "-o", "{out}"],
    "Scwrl4": ["{exe}", "-i", "{inp}", "-o", "{out}"],
}

#: FASPR locates its rotamer library as ``dirname(argv[0])/dun2010bbdep.bin``,
#: so the file must sit beside the real executable and the executable must be
#: invoked by a path containing a separator.
FASPR_ROTAMER_LIB = "dun2010bbdep.bin"

#: Default radius around the grafted loop whose side chains are also repacked.
#: Residues outside it keep the template's own rotamers.
DEFAULT_REPACK_SHELL = 5.0

#: Waters whose atoms fall this close to the graft are displaced by it.
WATER_DISPLACEMENT = 2.6

#: Residue names for solvent water. Waters are excluded from every steric and
#: interface measurement: ordered solvent was modelled around the template's own
#: loop, so a water in a graft's way is displaced rather than clashed with, and
#: leaving them in counts solvent as part of the epitope.
WATER_NAMES = frozenset({"HOH", "DOD", "WAT"})

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL", "X": "GLY",
}

#: Shortest heavy-atom separation seen at real interfaces, by whether both atoms
#: are polar. Two polar atoms may hydrogen bond and come genuinely close; once a
#: carbon is involved there is no such interaction and the contact is steric.
#: Being element-aware matters in practice: a 2.4 Å Tyr-OH···Gln-CG contact is a
#: clash, while 2.65 Å Gln-OE1···Gln-NE2 at the same interface is a good
#: hydrogen bond, and one distance cut calls both the same thing.
#:
#: Calibrated on 102 native antibody-antigen interfaces rather than on van der
#: Waals radii, which are far stricter than deposited structures: the median
#: native closest approach is 2.74 Å polar and 3.17 Å with a carbon, but the
#: tails run much shorter. These cuts flag about 2% of natives, so a flag means
#: "a real interface would not do this" instead of "textbook radii overlap".
#: A vdW-derived 3.0 Å carbon cut flags 12.7% of natives and cries wolf.
HBOND_MIN = 2.2
VDW_MIN = 2.8
_POLAR = frozenset("NOS")


# --------------------------------------------------------------------------- #
# Finding a packer
# --------------------------------------------------------------------------- #


def find_packer(name: str) -> Path | None:
    """Locate a packer executable, returning a *resolved* absolute path.

    Searched in order: ``<NAME>_BIN``, the packer's own folder in the
    repository (``FASPR/bin/``, laid out like ``MASTER/bin/``), then
    ``PATH``. The path is resolved through symlinks because FASPR derives its
    rotamer-library location from ``argv[0]``: a symlink in ``~/bin`` would send
    it looking in the wrong directory.
    """
    env = os.environ.get(f"{name.upper()}_BIN")
    if env:
        candidate = Path(env)
        if candidate.is_dir():
            candidate = candidate / name
        if candidate.is_file():
            return candidate.resolve()

    for directory in (REPO_ROOT / name / "bin", REPO_ROOT / name):
        for stem in (name, name.lower(), name.upper()):
            candidate = directory / stem
            if candidate.is_file():
                return candidate.resolve()

    for stem in (name, name.lower(), name.upper()):
        found = shutil.which(stem)
        if found:
            return Path(found).resolve()
    return None


def check_packer(exe: Path, name: str) -> str | None:
    """Return a human-readable problem with this packer install, or None."""
    if name == "FASPR" and not (exe.parent / FASPR_ROTAMER_LIB).is_file():
        return (
            f"{exe} is present but {FASPR_ROTAMER_LIB} is not beside it.\n"
            f"  FASPR hard-codes that filename and looks for it next to the\n"
            f"  executable, so copy it into {exe.parent}/ from the FASPR source tree."
        )
    return None


# --------------------------------------------------------------------------- #
# PDB line helpers
# --------------------------------------------------------------------------- #


def _atom_key(line: str) -> tuple[str, int, str]:
    return line[21], int(line[22:26]), line[26]


def _coord(line: str) -> np.ndarray:
    return np.array([float(line[30 + 8 * i: 38 + 8 * i]) for i in range(3)])


def element_of(line: str) -> str:
    """Element symbol, from columns 77-78 or else the atom name."""
    symbol = line[76:78].strip().upper()
    return symbol if symbol else line[12:16].strip().lstrip("0123456789")[:1].upper()


def summarise_residues(numbers: Sequence[int], limit: int = 6) -> str:
    """Condense residue numbers to ranges, so a detected epitope is reviewable.

    A proximity-detected epitope is not a contiguous span, and printing 20 bare
    numbers hides whether the selection is sensible. "38-46, 92, 135-137" does
    not.
    """
    if not numbers:
        return "none"
    runs: list[list[int]] = [[numbers[0], numbers[0]]]
    for n in sorted(numbers)[1:]:
        if n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    parts = [str(a) if a == b else f"{a}-{b}" for a, b in runs]
    if len(parts) > limit:
        return ", ".join(parts[:limit]) + f", +{len(parts) - limit} more"
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Selective repacking
# --------------------------------------------------------------------------- #


def faspr_residues(lines: list[str]) -> list[tuple[tuple[str, int, str], str]]:
    """The residues FASPR will read, in the order it reads them.

    FASPR takes standard-amino-acid ATOM records in file order as one sequence
    stream, ignoring chain boundaries, HETATM and alternate locations beyond the
    first. ``-s`` is matched against that stream positionally, so this ordering
    has to be exact: verified against FASPR output on an 812-residue complex,
    where every residue name came back unchanged.
    """
    seen: dict[tuple[str, int, str], str] = {}
    for line in lines:
        if not line.startswith("ATOM") or line[16] not in (" ", "A"):
            continue
        name = line[17:20].strip()
        if name not in THREE_TO_ONE:
            continue
        seen.setdefault(_atom_key(line), name)
    return list(seen.items())


def repack_sequence(
    lines: list[str],
    chain: str,
    span: tuple[int, int],
    shell: float,
) -> tuple[str, int]:
    """A FASPR ``-s`` sequence that repacks only the graft and its neighbours.

    FASPR repacks every side chain in the file unless told otherwise, which
    would silently rewrite the antigen and the antibody framework as well. A
    lower-case letter fixes that residue's conformation, so everything beyond
    ``shell`` Å of the grafted loop is lower-cased and left exactly as it was.

    Returns the sequence and how many residues it leaves free to move.
    """
    residues = faspr_residues(lines)
    atoms: dict[tuple[str, int, str], list[np.ndarray]] = {}
    for line in lines:
        if line.startswith("ATOM") and line[16] in (" ", "A"):
            atoms.setdefault(_atom_key(line), []).append(_coord(line))

    lo, hi = span
    loop = {k for k, _ in residues if k[0] == chain and lo <= k[1] <= hi}
    if not loop:
        raise ValueError(f"no grafted residues found in chain {chain} {lo}-{hi}")
    loop_coords = np.array([c for k in loop for c in atoms[k]])

    free = set(loop)
    for key, _ in residues:
        if key in free:
            continue
        near = np.array(atoms[key])
        if float(
            np.linalg.norm(near[:, None, :] - loop_coords[None, :, :], axis=-1).min()
        ) < shell:
            free.add(key)

    sequence = "".join(
        THREE_TO_ONE[name] if key in free else THREE_TO_ONE[name].lower()
        for key, name in residues
    )
    return sequence, len(free)


def _merge_heteroatoms(
    packed: list[str],
    original: list[str],
    graft: np.ndarray | None = None,
) -> tuple[list[str], int, list[str]]:
    """Put back the HETATM records FASPR discards.

    FASPR drops every heteroatom — glycans, ligands, metals, waters. On a
    glycosylated antigen those are part of the interface environment, so they
    are restored verbatim and the atom serials renumbered.

    Restoring them wholesale would be wrong, though. The template's ordered
    waters were modelled around the template's *own* loop; where the graft now
    sits, some of them overlap it, by as little as 0.5 Å in practice. A water is
    displaceable solvent, so one in the graft's way is deleted. Anything else —
    a glycan, a ligand, a metal — is not silently removed: it is kept and
    returned as a warning, because that is a modelling decision to make rather
    than to hide.

    Returns the merged lines, the number of waters dropped, and the non-water
    heteroatoms left overlapping the graft.
    """
    het = [x for x in original if x.startswith("HETATM")]
    if not het:
        return packed, 0, []

    dropped, overlapping = 0, []
    if graft is not None and len(graft):
        keep: list[str] = []
        by_residue: dict[tuple[str, int, str], list[str]] = {}
        for line in het:
            by_residue.setdefault(_atom_key(line), []).append(line)
        for key, lines in by_residue.items():
            closest = min(
                float(np.linalg.norm(graft - _coord(line), axis=1).min())
                for line in lines
            )
            if closest >= WATER_DISPLACEMENT:
                keep.extend(lines)
            elif lines[0][17:20].strip() in WATER_NAMES:
                dropped += 1
            else:
                keep.extend(lines)
                overlapping.append(
                    f"{lines[0][17:20].strip()} {key[0]}{key[1]} at {closest:.2f} A"
                )
        het = keep

    body = [x for x in packed if not x.startswith(("END", "MASTER"))]
    out, serial = [], 1
    for line in body + het:
        if line.startswith(("ATOM", "HETATM")):
            out.append(f"{line[:6]}{serial:>5}{line[11:]}")
            serial += 1
        else:
            out.append(line)
    out.append("END\n")
    return out, dropped, overlapping


def _verify_pack(original: list[str], packed: list[str]) -> str | None:
    """Confirm the packer only moved side chains. Returns a problem, or None."""

    def table(
        lines: list[str],
    ) -> dict[tuple[str, int, str], tuple[str, dict[str, np.ndarray]]]:
        out: dict[tuple[str, int, str], tuple[str, dict[str, np.ndarray]]] = {}
        for line in lines:
            if not line.startswith("ATOM") or line[16] not in (" ", "A"):
                continue
            key = _atom_key(line)
            name, atoms = out.setdefault(key, (line[17:20].strip(), {}))
            atoms[line[12:16].strip()] = _coord(line)
        return out

    before, after = table(original), table(packed)
    missing = set(before) - set(after)
    if missing:
        return f"{len(missing)} residues were dropped, e.g. {sorted(missing)[:3]}"
    mutated = [k for k in before if before[k][0] != after[k][0]]
    if mutated:
        key = mutated[0]
        return (
            f"{len(mutated)} residues changed identity, e.g. {key} "
            f"{before[key][0]} -> {after[key][0]} — the sequence was misaligned"
        )
    shifted = max(
        (float(np.linalg.norm(before[k][1][a] - after[k][1][a]))
         for k in before for a in BACKBONE
         if a in before[k][1] and a in after[k][1]),
        default=0.0,
    )
    if shifted > 0.1:
        return f"the backbone moved by up to {shifted:.2f} A; only side chains may move"
    return None


def run_packer(
    exe: Path,
    name: str,
    pdb: Path,
    *,
    chain: str | None = None,
    span: tuple[int, int] | None = None,
    shell: float = DEFAULT_REPACK_SHELL,
) -> bool:
    """Repack side chains in place. Returns True on success.

    With ``chain`` and ``span`` given, only the grafted loop and the residues
    within ``shell`` Å of it are repacked; everything else is pinned. Pass
    ``shell`` as ``float("inf")`` to repack the whole complex.

    The result is checked rather than trusted. FASPR exits 0 and writes nothing
    when the ``-s`` sequence length disagrees with the structure, and it never
    reports the mismatch, so a silent truncation would otherwise pass as a
    successful pack.
    """
    template = _PACKERS.get(name)
    if template is None:
        return False
    original = pdb.read_text().splitlines(keepends=True)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "packed.pdb"
        cmd = [part.format(exe=exe, inp=pdb, out=out) for part in template]

        if (
            name == "FASPR" and chain is not None and span is not None
            and shell != float("inf")
        ):
            try:
                sequence, n_free = repack_sequence(original, chain, span, shell)
            except (ValueError, KeyError) as exc:
                logger.warning("cannot build a selective-repack sequence: %s", exc)
                return False
            seq_file = Path(tmp) / "sequence.txt"
            seq_file.write_text(sequence + "\n")
            cmd += ["-s", str(seq_file)]
            logger.debug(
                "repacking %d residues (loop + %g A shell), %d fixed",
                n_free, shell, len(sequence) - n_free,
            )

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("packer failed: %s", exc)
            return False
        if result.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            why = detail[-1][:160] if detail else "no output was written"
            logger.warning("packer exited %d: %s", result.returncode, why)
            return False

        packed = out.read_text().splitlines(keepends=True)

    problem = _verify_pack(original, packed)
    if problem is not None:
        logger.warning("packed model rejected: %s", problem)
        return False

    graft = None
    if chain is not None and span is not None:
        lo, hi = span
        graft = np.array([
            _coord(line) for line in packed
            if line.startswith("ATOM") and line[21] == chain
            and lo <= int(line[22:26]) <= hi
        ])
    merged, dropped, overlapping = _merge_heteroatoms(packed, original, graft)
    if dropped:
        logger.debug("dropped %d waters displaced by the graft", dropped)
    if overlapping:
        logger.warning(
            "heteroatoms still overlap the graft: %s", ", ".join(overlapping[:4])
        )
    pdb.write_text("".join(merged))
    return True


def closest_contact(
    pdb: Path, chain: str, span: tuple[int, int], ag_chain: str
) -> tuple[float, str | None]:
    """Closest approach from the grafted loop to the antigen, and any clash.

    Returns the closest heavy-atom distance and a description of the worst
    chemically unacceptable contact, or None if there is none. Kept for the
    command-line tool's per-model report; the pipeline uses the full
    :mod:`epifitscout.build.interface` measurement instead.
    """
    lo, hi = span
    loop: list[tuple[str, np.ndarray, str]] = []
    antigen: list[tuple[str, np.ndarray, str]] = []
    for line in pdb.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")) or element_of(line) == "H":
            continue
        label = (f"{line[17:20].strip()}{line[22:26].strip()}{line[26].strip()} "
                 f"{line[12:16].strip()}")
        entry = (element_of(line), _coord(line), label)
        if line[21] == chain and lo <= int(line[22:26]) <= hi:
            loop.append(entry)
        elif line[21] == ag_chain:
            antigen.append(entry)
    if not loop or not antigen:
        return float("nan"), None

    coords_a = np.array([c for _, c, _ in loop])
    coords_b = np.array([c for _, c, _ in antigen])
    distances = np.linalg.norm(coords_a[:, None, :] - coords_b[None, :, :], axis=-1)

    worst: tuple[float, str] | None = None
    for i, j in zip(*np.where(distances < VDW_MIN), strict=True):
        polar = loop[i][0] in _POLAR and antigen[j][0] in _POLAR
        allowed = HBOND_MIN if polar else VDW_MIN
        d = float(distances[i, j])
        if d < allowed and (worst is None or d - allowed < worst[0]):
            worst = (d - allowed, f"{loop[i][2]} - {antigen[j][2]} at {d:.2f} A "
                                  f"(needs {allowed:.1f} A)")
    return float(distances.min()), (worst[1] if worst else None)
