#!/usr/bin/env python3
"""Assemble grafted antibody-antigen complexes from a search.

``epifitscout.search`` returns ranked *fragments*. This turns the top ones into
complete structures: the template complex with its CDR loop replaced by the
retrieved fragment, antigen included, written as PDB files you can open, model
further, or hand to a packer.

Side chains, in increasing order of rigour:

``--side-chains none`` (default)
    Backbone only at the grafted loop; the rest of the template keeps its own
    atoms. Honest about what was retrieved — the fragment database stores
    backbone only.

``--side-chains native``
    Carry over the fragment's own side chains from its original PDB entry,
    moved by the same rigid transform. Fast and needs no extra software, but
    those rotamers were packed against the fragment's *original* antigen, so in
    the new context they are a starting point, not an answer.

``--side-chains packed --packer FASPR``
    Rebuild side chains from the sequence with an external packer. This is the
    correct treatment and the prerequisite for any real buried-surface or
    Lawrence & Colman Sc calculation. See ``FASPR/README.md`` to install FASPR.

    The graft and the residues within ``--repack-shell`` Å of it are repacked;
    the rest of the complex keeps the template's own rotamers, because rewriting
    the whole experimental model to place one loop is not a trade worth making.
    ``--repack-all`` lifts that restriction.

Every output records how well the loop joins the framework. A peptide bond is
1.33 Å; much beyond 2 Å and the chain is broken, however good the score. It
also records the closest approach to the antigen, with an element-aware clash
check, because the ranking score is computed on backbone alone and cannot see a
side chain driven into the antigen.

Example::

    python tools/build_complex.py \\
        --structure database_builder/rcsb_cache/3gbn.pdb \\
        --cdr H 93 102 --epitope B 38 46 \\
        --top 5 --side-chains native --out-dir ../grafts
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import epifitscout  # noqa: E402
from epifitscout.config.schema import MASTERConfig, PipelineConfig  # noqa: E402
from epifitscout.db.rcsb_downloader import RcsbDownloader  # noqa: E402
from epifitscout.pipeline.search_pipeline import FragmentSearchPipeline  # noqa: E402
from epifitscout.types.fragment import ScoredHit  # noqa: E402
from epifitscout.utils.geometry import kabsch_superpose  # noqa: E402
from epifitscout.utils.io import detect_format, extract_residue_atoms  # noqa: E402

BACKBONE = ("N", "CA", "C", "O")
#: Ideal peptide bond length; used to judge whether the graft joins the framework.
PEPTIDE_BOND = 1.33
#: Beyond this the backbone is considered broken at the junction.
BREAK_THRESHOLD = 2.0

_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL", "X": "GLY",
}

#: Packers this script knows how to drive, and the argument order they expect.
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

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


@dataclass
class GraftReport:
    """How a single graft turned out."""

    rank: int
    source: str
    rmsd: float
    score: float
    n_term_gap: float
    c_term_gap: float
    clashes: int
    matched: int
    side_chains: str
    path: Path
    #: Closest heavy-atom approach from the grafted loop to the antigen in the
    #: finished model, NaN when not measured.
    contact: float = float("nan")
    #: The worst chemically unacceptable contact in the finished model, if any.
    clash_detail: str | None = None

    @property
    def joins_cleanly(self) -> bool:
        return max(self.n_term_gap, self.c_term_gap) < BREAK_THRESHOLD

    @property
    def contact_is_too_close(self) -> bool:
        return self.clash_detail is not None

    def line(self) -> str:
        flag = "" if self.joins_cleanly else "   CHAIN BREAK"
        if self.contact_is_too_close:
            flag += f"   CLASH {self.contact:.2f} A"
        contact = "" if self.contact != self.contact else f"contact {self.contact:4.2f} A  "
        return (
            f"{self.rank:>3}  {self.source:<12} rmsd {self.rmsd:4.2f}  "
            f"score {self.score:.3f}  join {self.n_term_gap:4.2f}/{self.c_term_gap:4.2f} A  "
            f"clashes {self.clashes}  {contact}{self.path.name}{flag}"
        )


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

    # The packer's own folder in the repository, laid out like MASTER/bin/.
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


def _atom_key(line: str) -> tuple[str, int, str]:
    return line[21], int(line[22:26]), line[26]


def _coord(line: str) -> np.ndarray:
    return np.array([float(line[30 + 8 * i: 38 + 8 * i]) for i in range(3)])


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
        if name not in _THREE_TO_ONE:
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
        if float(np.linalg.norm(near[:, None, :] - loop_coords[None, :, :], axis=-1).min()) < shell:
            free.add(key)

    sequence = "".join(
        _THREE_TO_ONE[name] if key in free else _THREE_TO_ONE[name].lower()
        for key, name in residues
    )
    return sequence, len(free)


#: Waters whose atoms fall this close to the graft are displaced by it.
WATER_DISPLACEMENT = 2.6


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
        keep, by_residue = [], {}
        for line in het:
            by_residue.setdefault(_atom_key(line), []).append(line)
        for key, lines in by_residue.items():
            closest = min(
                float(np.linalg.norm(graft - _coord(line), axis=1).min()) for line in lines
            )
            if closest >= WATER_DISPLACEMENT:
                keep.extend(lines)
            elif lines[0][17:20].strip() in ("HOH", "DOD", "WAT"):
                dropped += 1
            else:
                keep.extend(lines)
                overlapping.append(f"{lines[0][17:20].strip()} {key[0]}{key[1]} "
                                   f"at {closest:.2f} A")
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


def native_side_chains(
    hit: ScoredHit, downloader: RcsbDownloader
) -> dict[int, list[tuple[str, np.ndarray]]] | None:
    """Fetch the fragment's own side chains, moved into the grafted position.

    The fragment database stores backbone only, so the side chains come from the
    original RCSB entry. They are placed by the rigid transform that maps the
    fragment's original backbone onto the grafted backbone, which is exact.

    Returns None when the original entry cannot supply them — a missing entry,
    a residue-range mismatch, or incomplete backbone.
    """
    frag = hit.fragment
    try:
        sf = downloader.download(frag.pdb_id.lower())
        residues = extract_residue_atoms(sf.path, frag.chain, fmt=detect_format(sf.path))
    except Exception:
        return None

    lo, hi = frag.residue_range
    window = [r for r in residues if lo <= r[0] <= hi]
    if len(window) != frag.length:
        return None

    from Bio.PDB import MMCIFParser, PDBParser

    parser = (MMCIFParser(QUIET=True) if detect_format(sf.path) == "cif"
              else PDBParser(QUIET=True))
    try:
        chain = parser.get_structure("x", str(sf.path))[0][frag.chain]
    except Exception:
        return None

    original_bb, per_residue = [], []
    for num, _, _ in window:
        try:
            res = chain[(" ", num, " ")]
            original_bb.append([res[a].coord for a in BACKBONE])
        except KeyError:
            return None
        per_residue.append([
            (a.get_name(), np.array(a.coord, dtype=np.float64))
            for a in res if a.get_name() not in BACKBONE and a.element != "H"
        ])

    grafted = hit.superposition.apply(frag.coords)
    _, rot, trans = kabsch_superpose(
        np.array(original_bb, dtype=np.float64).reshape(-1, 3),
        grafted.reshape(-1, 3),
    )
    return {
        i: [(name, coord @ rot.T + trans) for name, coord in atoms]
        for i, atoms in enumerate(per_residue)
    }


def write_complex(
    template: Path,
    ab_chain: str,
    span: tuple[int, int],
    hit: ScoredHit,
    out_path: Path,
    side_chains: dict[int, list[tuple[str, np.ndarray]]] | None,
) -> None:
    """Write the template with its CDR replaced by the grafted fragment."""
    from Bio.PDB import PDBIO, MMCIFParser, PDBParser
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Residue import Residue

    parser = (MMCIFParser(QUIET=True) if detect_format(template) == "cif"
              else PDBParser(QUIET=True))
    structure = parser.get_structure("t", str(template))
    model = next(iter(structure))
    chain = model[ab_chain]

    lo, hi = span
    # Reuse the template's own residue identifiers for the replaced stretch.
    # Antibody CDRs are routinely numbered with insertion codes (100, 100A,
    # 100B...), so a span of 93-102 can hold more than ten residues; renumbering
    # the graft sequentially would collide with residues further along the chain.
    replaced = [r for r in chain if r.id[0] == " " and lo <= r.id[1] <= hi]
    replaced_ids = [r.id for r in replaced]
    for residue in replaced:
        chain.detach_child(residue.id)

    if len(replaced_ids) != hit.fragment.length:
        raise ValueError(
            f"template has {len(replaced_ids)} residues in {ab_chain}/{lo}-{hi} but the "
            f"fragment has {hit.fragment.length}; they must match one to one"
        )

    grafted = hit.superposition.apply(hit.fragment.coords)
    sequence = hit.fragment.sequence or "X" * hit.fragment.length
    serial = 1
    for i in range(hit.fragment.length):
        resname = _ONE_TO_THREE.get(sequence[i].upper(), "GLY")
        residue = Residue(replaced_ids[i], resname, "")
        for j, atom_name in enumerate(BACKBONE):
            residue.add(Atom(atom_name, grafted[i, j, :], 0.0, 1.0, " ",
                             f" {atom_name:<3}"[:4], serial, atom_name[0]))
            serial += 1
        for name, coord in (side_chains or {}).get(i, []):
            element = name[0] if name[0].isalpha() else "C"
            residue.add(Atom(name, coord, 0.0, 1.0, " ", f" {name:<3}"[:4],
                             serial, element))
            serial += 1
        chain.add(residue)

    # keep residues in sequence order so the file reads as a real chain
    chain.child_list.sort(key=lambda r: (r.id[1], r.id[2]))

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path))


def junction_gaps(
    template: Path, ab_chain: str, span: tuple[int, int], hit: ScoredHit
) -> tuple[float, float]:
    """Distance from the grafted termini to the framework stubs they must bond to."""
    qb = epifitscout.QueryBuilder(template)
    lo, hi = span
    flank = qb.get_fragment(ab_chain, lo - 1, hi + 1)
    pre_c = flank.coords[0, 2, :]        # C of the residue before the loop
    post_n = flank.coords[-1, 0, :]      # N of the residue after
    grafted = hit.superposition.apply(hit.fragment.coords)
    return (
        float(np.linalg.norm(grafted[0, 0, :] - pre_c)),
        float(np.linalg.norm(grafted[-1, 2, :] - post_n)),
    )


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

        if name == "FASPR" and chain is not None and span is not None and shell != float("inf"):
            try:
                sequence, n_free = repack_sequence(original, chain, span, shell)
            except (ValueError, KeyError) as exc:
                print(f"    cannot build a selective-repack sequence: {exc}")
                return False
            seq_file = Path(tmp) / "sequence.txt"
            seq_file.write_text(sequence + "\n")
            cmd += ["-s", str(seq_file)]
            print(f"    repacking {n_free} residues (loop + {shell:g} A shell), "
                  f"{len(sequence) - n_free} fixed")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"    packer failed: {exc}")
            return False
        if result.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            why = detail[-1][:160] if detail else "no output was written"
            print(f"    packer exited {result.returncode}: {why}")
            return False

        packed = out.read_text().splitlines(keepends=True)

    problem = _verify_pack(original, packed)
    if problem is not None:
        print(f"    packed model rejected: {problem}")
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
        print(f"    dropped {dropped} waters displaced by the graft")
    if overlapping:
        print(f"    WARNING: heteroatoms still overlap the graft: "
              f"{', '.join(overlapping[:4])}")
    pdb.write_text("".join(merged))
    return True


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


def _element(line: str) -> str:
    """Element symbol, from columns 77-78 or else the atom name."""
    symbol = line[76:78].strip().upper()
    return symbol if symbol else line[12:16].strip().lstrip("0123456789")[:1].upper()


def closest_contact(
    pdb: Path, chain: str, span: tuple[int, int], ag_chain: str
) -> tuple[float, str | None]:
    """Closest approach from the grafted loop to the antigen, and any clash.

    The ranking score is computed on backbone alone, so it cannot see a side
    chain driven into the antigen; measuring it on the finished model is the
    only way to know. Returns the closest heavy-atom distance and a description
    of the worst chemically unacceptable contact, or None if there is none.
    """
    lo, hi = span
    loop: list[tuple[str, np.ndarray, str]] = []
    antigen: list[tuple[str, np.ndarray, str]] = []
    for line in pdb.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")) or _element(line) == "H":
            continue
        label = (f"{line[17:20].strip()}{line[22:26].strip()}{line[26].strip()} "
                 f"{line[12:16].strip()}")
        entry = (_element(line), _coord(line), label)
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


def _verify_pack(original: list[str], packed: list[str]) -> str | None:
    """Confirm the packer only moved side chains. Returns a problem, or None."""
    def table(lines: list[str]) -> dict[tuple[str, int, str], tuple[str, dict[str, np.ndarray]]]:
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
        return (f"{len(mutated)} residues changed identity, e.g. {key} "
                f"{before[key][0]} -> {after[key][0]} — the sequence was misaligned")
    shifted = max(
        (float(np.linalg.norm(before[k][1][a] - after[k][1][a]))
         for k in before for a in BACKBONE
         if a in before[k][1] and a in after[k][1]),
        default=0.0,
    )
    if shifted > 0.1:
        return f"the backbone moved by up to {shifted:.2f} A; only side chains may move"
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", type=Path, required=True,
                   help="Template antibody-antigen complex.")
    p.add_argument("--cdr", nargs=3, metavar=("CHAIN", "FIRST", "LAST"), required=True)
    p.add_argument("--epitope", nargs=3, metavar=("CHAIN", "FIRST", "LAST"), required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--top", type=int, default=10, help="How many hits to build.")
    p.add_argument("--include-self", action="store_true",
                   help="Keep hits from the template's own PDB entry.")
    p.add_argument("--side-chains", choices=("none", "native", "packed"), default="none")
    p.add_argument("--packer", default="FASPR",
                   help="Packer executable for --side-chains packed (FASPR or Scwrl4).")
    p.add_argument("--repack-shell", type=float, default=DEFAULT_REPACK_SHELL,
                   metavar="A",
                   help="Also repack residues within this distance of the grafted "
                        f"loop (default {DEFAULT_REPACK_SHELL:g} A). Everything "
                        "further away keeps the template's own rotamers.")
    p.add_argument("--repack-all", action="store_true",
                   help="Repack every side chain in the complex, antigen and "
                        "framework included. Rarely what you want: it rewrites "
                        "the experimental model you are grafting into.")
    p.add_argument("--max-hits", type=int, default=500)
    p.add_argument("--rmsd-threshold", type=float, default=2.0)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)

    ab_chain, lo, hi = args.cdr[0], int(args.cdr[1]), int(args.cdr[2])
    ag_chain, elo, ehi = args.epitope[0], int(args.epitope[1]), int(args.epitope[2])

    packer_exe = None
    if args.side_chains == "packed":
        packer_exe = find_packer(args.packer)
        if packer_exe is None:
            print(f"ERROR: packer '{args.packer}' was not found in "
                  f"{args.packer.upper()}_BIN, "
                  f"{REPO_ROOT / args.packer / 'bin'}/, or PATH.\n"
                  f"  FASPR:  https://github.com/tommyhuangthu/FASPR\n"
                  f"          git clone, then 'g++ -O3 -o FASPR src/*.cpp', and copy\n"
                  f"          both FASPR and {FASPR_ROTAMER_LIB} into "
                  f"{REPO_ROOT / 'FASPR/bin'}/\n"
                  f"          see {REPO_ROOT / 'FASPR/README.md'}\n"
                  f"  Scwrl4: http://dunbrack.fccc.edu/lab/scwrl (registration)\n"
                  f"Or use --side-chains native, which needs no extra software.",
                  file=sys.stderr)
            return 1
        problem = check_packer(packer_exe, args.packer)
        if problem is not None:
            print(f"ERROR: {problem}", file=sys.stderr)
            return 1
        print(f"packer: {packer_exe}")

    qb = epifitscout.QueryBuilder(args.structure)
    cdr = qb.get_fragment(ab_chain, lo, hi)
    epitope = qb.get_epitope_surface(ag_chain, residues=(elo, ehi))
    print(f"template {args.structure.name}: CDR {ab_chain}/{lo}-{hi} ({cdr.length} residues), "
          f"epitope {ag_chain}/{elo}-{ehi} ({epitope.n_residues} residues), "
          f"reach {qb.epitope_reach(cdr, epitope):.1f} A")

    body = np.concatenate(
        [c for _, _, c in extract_residue_atoms(args.structure, ab_chain)], axis=0
    )
    cfg = PipelineConfig(
        master=MASTERConfig(rmsd_threshold=args.rmsd_threshold,
                            max_hits=args.max_hits, timeout_seconds=900),
        max_workers=args.workers, log_level="WARNING",
    )
    hits = FragmentSearchPipeline(cfg).search(cdr, epitope, antibody_body=body)
    self_id = args.structure.stem.upper()
    if not args.include_self:
        hits = [h for h in hits if h.fragment.pdb_id.upper() != self_id]
    hits = [h for h in hits if h.local_fit is None or h.local_fit.fits][: args.top]
    if not hits:
        print("no hits survived the collision filter")
        return 1
    print(f"building {len(hits)} complexes\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    downloader = RcsbDownloader(cache_dir=REPO_ROOT / "database_builder/rcsb_cache")
    reports: list[GraftReport] = []

    for rank, hit in enumerate(hits, 1):
        tag = f"{rank:02d}_{hit.fragment.pdb_id}_{hit.fragment.chain}"
        out_path = args.out_dir / f"{tag}.pdb"
        sc = None
        mode = "none"
        if args.side_chains in ("native", "packed"):
            sc = native_side_chains(hit, downloader)
            mode = "native" if sc else "none (unavailable)"
        write_complex(args.structure, ab_chain, (lo, hi), hit, out_path, sc)
        if args.side_chains == "packed" and packer_exe is not None:
            packed = run_packer(
                packer_exe, args.packer, out_path, chain=ab_chain, span=(lo, hi),
                shell=float("inf") if args.repack_all else args.repack_shell,
            )
            mode = "packed" if packed else f"{mode} (pack failed)"
        n_gap, c_gap = junction_gaps(args.structure, ab_chain, (lo, hi), hit)
        report = GraftReport(
            rank=rank, source=f"{hit.fragment.pdb_id}:{hit.fragment.chain}",
            rmsd=hit.rmsd_cdr, score=hit.complementarity_score,
            n_term_gap=n_gap, c_term_gap=c_gap,
            clashes=hit.local_fit.n_clash if hit.local_fit else 0,
            matched=hit.local_fit.matched_length if hit.local_fit else 0,
            side_chains=mode, path=out_path,
        )
        if args.side_chains != "none":
            report.contact, report.clash_detail = closest_contact(
                out_path, ab_chain, (lo, hi), ag_chain
            )
        reports.append(report)
        print(report.line())

    broken = [r for r in reports if not r.joins_cleanly]
    unresolved = [r for r in reports if r.side_chains.startswith("none") and
                  args.side_chains != "none"]
    print(f"\n{len(reports)} complexes written to {args.out_dir}")
    print(f"  ideal peptide bond is {PEPTIDE_BOND} A; "
          f"{len(reports) - len(broken)}/{len(reports)} join cleanly")
    if broken:
        print(f"  CHAIN BREAK in {len(broken)}: "
              f"{', '.join(r.source for r in broken)} — these need loop closure")
    if unresolved:
        print(f"  side chains unavailable for {len(unresolved)}: "
              f"{', '.join(r.source for r in unresolved)}")
    too_close = [r for r in reports if r.contact_is_too_close]
    if too_close:
        print(f"  CLASH in {len(too_close)}: the ranking score sees backbone only,\n"
              f"        so a side chain can still run into the antigen. These need\n"
              f"        minimisation or a different rotamer:")
        for r in too_close:
            print(f"        {r.source}  {r.clash_detail}")
    if args.side_chains == "native":
        print("  note: native rotamers were packed against the fragment's original\n"
              "        antigen, so treat them as a starting point, not an answer.\n"
              "        Use --side-chains packed for a proper repack.")
    summary = args.out_dir / "grafts.tsv"
    summary.write_text(
        "rank\tsource\trmsd\tscore\tn_gap\tc_gap\tclashes\tmatched\tcontact\t"
        "side_chains\tfile\n"
        + "".join(
            f"{r.rank}\t{r.source}\t{r.rmsd:.3f}\t{r.score:.4f}\t{r.n_term_gap:.2f}\t"
            f"{r.c_term_gap:.2f}\t{r.clashes}\t{r.matched}\t"
            f"{'' if r.contact != r.contact else f'{r.contact:.2f}'}\t"
            f"{r.side_chains}\t{r.path.name}\n"
            for r in reports
        )
    )
    print(f"  summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
