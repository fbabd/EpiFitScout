"""Relieve side-chain clashes by turning chi angles, after the packer has run.

Why this exists
---------------
FASPR packs against the antigen and does it well, but it chooses from discrete
rotamers and its own tolerance is looser than the element-aware criterion the
interface scorer applies. What comes out is a field of near-misses: over 85
grafted models, 61% of clashing atom pairs were side-chain-only, at a median
overlap of 0.43 Å. Nothing is in the wrong *place*; things are a few degrees off
the right angle.

So this turns the angles. For every side chain with a clash against the antigen,
each rotatable bond is scanned and set to the value that relieves the overlap
best. Rotating about a chi bond moves only the atoms distal to it and preserves
every bond length and angle exactly, so the result is still a chemically valid
side chain -- which Cartesian minimisation of the same atoms would not
guarantee.

What it will not do
-------------------
It cannot fix a backbone that is inside the antigen: that was the other 39% of
clashes, and it needs the loop itself to move. It also never moves the antigen,
and it will not invent contacts -- a side chain turned out of a clash is usually
turned out of contact too, so expect clashes to fall and contacts to fall with
them. Whether what remains is a *better* interface is a question for the
scorer, not for this module.
"""

from __future__ import annotations

import logging

import numpy as np

from epifitscout.graft import packer as pk

logger = logging.getLogger(__name__)

__all__ = ["CHI_ATOMS", "relax_side_chains"]

#: Atoms defining each chi bond, and the first atom that moves with it.
#: Rotating about ``(b, c)`` moves every side-chain atom distal to ``c``.
CHI_ATOMS: dict[str, list[tuple[str, str]]] = {
    "ARG": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE")],
    "ASN": [("CA", "CB"), ("CB", "CG")],
    "ASP": [("CA", "CB"), ("CB", "CG")],
    "CYS": [("CA", "CB")],
    "GLN": [("CA", "CB"), ("CB", "CG"), ("CG", "CD")],
    "GLU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD")],
    "HIS": [("CA", "CB"), ("CB", "CG")],
    "ILE": [("CA", "CB"), ("CB", "CG1")],
    "LEU": [("CA", "CB"), ("CB", "CG")],
    "LYS": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE")],
    "MET": [("CA", "CB"), ("CB", "CG"), ("CG", "SD")],
    "PHE": [("CA", "CB"), ("CB", "CG")],
    "SER": [("CA", "CB")],
    "THR": [("CA", "CB")],
    "TRP": [("CA", "CB"), ("CB", "CG")],
    "TYR": [("CA", "CB"), ("CB", "CG")],
    "VAL": [("CA", "CB")],
}

#: Backbone atoms never move here -- only side chains turn.
_BACKBONE = frozenset({"N", "CA", "C", "O", "OXT"})

#: Angles tried for each bond, degrees. A full turn at 10° costs 36 distance
#: evaluations per bond, which is nothing beside the pack that preceded it.
_SCAN = np.arange(0, 360, 10.0)


def _rotate(coords: np.ndarray, origin: np.ndarray, axis: np.ndarray,
            degrees: float) -> np.ndarray:
    """Rotate ``coords`` about the line through ``origin`` along ``axis``."""
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    rel = coords - origin
    return (
        rel * c
        + np.cross(axis, rel) * s
        + axis * np.dot(rel, axis)[:, None] * (1.0 - c)
        + origin
    )


def _penalty(moved: np.ndarray, moved_elements: list[str],
             antigen: np.ndarray, antigen_elements: list[str],
             fixed: np.ndarray) -> float:
    """How badly these atoms overlap the antigen, plus their own molecule.

    Depth of overlap rather than a count, so the scan has a gradient to follow:
    a pair at 2.7 Å and one at 1.2 Å are both "a clash", and only a depth tells
    the scan which way it is improving.
    """
    if not len(moved):
        return 0.0

    total = 0.0
    for target, elements in ((antigen, antigen_elements), (fixed, None)):
        if not len(target):
            continue
        d = np.linalg.norm(moved[:, None, :] - target[None, :, :], axis=-1)
        for i, el_i in enumerate(moved_elements):
            polar_i = el_i in pk._POLAR
            for j in np.flatnonzero(d[i] < pk.VDW_MIN):
                allowed = (
                    pk.HBOND_MIN
                    if polar_i and elements is not None
                    and elements[j] in pk._POLAR
                    else pk.VDW_MIN
                )
                if d[i, j] < allowed:
                    total += float(allowed - d[i, j]) ** 2
    return total


def relax_side_chains(
    lines: list[str],
    ab_chain: str,
    span: tuple[int, int],
    antigen_chains: tuple[str, ...],
    passes: int = 2,
) -> tuple[list[str], int]:
    """Turn clashing side chains of the grafted loop out of the antigen.

    Args:
        lines: PDB lines of the packed model.
        ab_chain: Chain carrying the graft.
        span: ``(first, last)`` residue numbers of the graft.
        antigen_chains: Chains treated as antigen.
        passes: Sweeps over the loop. Residues interact, so one pass can leave
            a clash another created; two is enough in practice and a third
            rarely moves anything.

    Returns:
        ``(lines, n_changed)`` -- the model with side chains turned, and how
        many residues actually moved.
    """
    lo, hi = span
    atoms: list[tuple[int, str, str, str, int, np.ndarray]] = []
    for idx, line in enumerate(lines):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[16] not in (" ", "A"):
            continue
        name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21]
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        atoms.append((idx, name, resname, chain, resnum, pk._coord(line)))

    if not atoms:
        return lines, 0

    antigen_idx = [
        i for i, a in enumerate(atoms)
        if a[3] in antigen_chains
        and a[2] not in pk.WATER_NAMES
        and pk.element_of(lines[a[0]]) != "H"
    ]
    antigen = np.array([atoms[i][5] for i in antigen_idx]) if antigen_idx else np.empty((0, 3))
    antigen_el = [pk.element_of(lines[atoms[i][0]]) for i in antigen_idx]

    if not len(antigen):
        return lines, 0

    coords = {i: a[5].copy() for i, a in enumerate(atoms)}
    changed = 0

    for _ in range(passes):
        moved_this_pass = 0
        for resnum in range(lo, hi + 1):
            members = [
                i for i, a in enumerate(atoms)
                if a[3] == ab_chain and a[4] == resnum
            ]
            if not members:
                continue
            resname = atoms[members[0]][2]
            bonds = CHI_ATOMS.get(resname)
            if not bonds:
                continue

            by_name = {atoms[i][1]: i for i in members}
            side = [
                i for i in members
                if atoms[i][1] not in _BACKBONE
                and pk.element_of(lines[atoms[i][0]]) != "H"
            ]
            if not side:
                continue

            # Everything that is not this residue is an obstacle it must avoid.
            other = np.array([
                coords[i] for i in range(len(atoms))
                if i not in set(members) and i not in set(antigen_idx)
                and pk.element_of(lines[atoms[i][0]]) != "H"
            ])

            for base, tip in bonds:
                if base not in by_name or tip not in by_name:
                    break
                origin = coords[by_name[base]]
                axis = coords[by_name[tip]] - origin
                if np.linalg.norm(axis) < 1e-6:
                    break

                # Atoms distal to the bond: everything in the side chain
                # further along the chain than `tip`, which for these
                # definitions is every side-chain atom bar those already named.
                order = [b[1] for b in bonds[: bonds.index((base, tip)) + 1]]
                distal = [
                    i for i in side
                    if atoms[i][1] not in order and atoms[i][1] != base
                ]
                if not distal:
                    continue

                block = np.array([coords[i] for i in distal])
                elements = [pk.element_of(lines[atoms[i][0]]) for i in distal]
                best_angle, best_cost = 0.0, _penalty(
                    block, elements, antigen, antigen_el, other
                )
                if best_cost == 0.0:
                    continue
                for angle in _SCAN[1:]:
                    trial = _rotate(block, origin, axis, float(angle))
                    cost = _penalty(trial, elements, antigen, antigen_el, other)
                    if cost < best_cost:
                        best_angle, best_cost = float(angle), cost
                if best_angle:
                    turned = _rotate(block, origin, axis, best_angle)
                    for i, c in zip(distal, turned, strict=True):
                        coords[i] = c
                    moved_this_pass += 1
        changed += moved_this_pass
        if not moved_this_pass:
            break

    if not changed:
        return lines, 0

    out = list(lines)
    for i, a in enumerate(atoms):
        c = coords[i]
        if np.allclose(c, a[5]):
            continue
        line = out[a[0]]
        out[a[0]] = f"{line[:30]}{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}{line[54:]}"
    logger.debug("relaxed %d side-chain bonds in %s %d-%d", changed, ab_chain, lo, hi)
    return out, changed
