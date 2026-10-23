"""What a packed graft actually does against the epitope.

This is stage 2 of the method: given the loop placed where the graft dictates
and rebuilt all-atom, measure the interface it makes. Everything is computed on
the model as written to disk, so what is scored is exactly what a user would
open.

None of this ranks
------------------
The ranking is decided by :mod:`epifitscout.scoring.shape`, on backbone
geometry, before any of this runs. These measurements are made on the handful
of models that are actually reported, and they exist to describe and to
evaluate them — which is exactly what makes them usable as evidence about the
ranking rather than restatements of it.

What is measured, and why each term is here
-------------------------------------------
``buried_area``
    Antibody–antigen buried surface, by Shrake–Rupley: the area the two hide
    from solvent on binding. The quantity interface size is normally quoted in.
    Across candidates for one target only the loop differs, so differences in it
    are the loop's doing.
``loop_buried_area``
    The same difference restricted to the loop's own atoms — how much of *this
    loop* the antigen covers.
``contacts``, ``contact_residues``
    Heavy-atom pairs within the contact cutoff, and how many loop residues take
    part. Extent of the interface rather than its quality.
``clashes``
    Heavy-atom pairs closer than a real interface ever puts them, element-aware
    (see :mod:`epifitscout.build.packer`): polar pairs may approach 2.2 Å
    because they can hydrogen bond, anything involving carbon may not come
    inside 2.8 Å. Thresholds calibrated on 102 native interfaces, where they
    flag ~2% — so a clash means "a real interface would not do this".
``hbonds``, ``salt_bridges``
    Polar complementarity: a donor–acceptor pair within 3.5 Å whose angle at the
    acceptor exceeds 90°, and an oppositely charged group pair within 4.0 Å.
    These are the terms that reward a loop presenting the right chemistry to the
    epitope rather than merely enough surface.
``gap_volume``, ``gap_index``
    Empty space trapped between loop and epitope, and that volume per Å² of
    buried area — Laskowski's index, by grid approximation. A second, cruder
    reading of the same thing Sc measures: a retrieved loop was folded against
    a different antigen, so burying area while leaving cavities is its
    characteristic failure, and no area or contact count detects it.

The confound this exists to avoid
---------------------------------
Burial and contact counts reward interpenetration. In the v1 benchmark a random
length-matched control buried more surface (923 Å²) and made more contacts
(11.7) than every real method including the native loop, purely by being driven
into the antigen. Any of these numbers is therefore meaningless without the
steric terms beside it, which is why :class:`InterfaceScore` keeps them together
and reports ``is_clean`` alongside.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from epifitscout.config.schema import InterfaceConfig
from epifitscout.build import packer as pk
from epifitscout.build.sasa import atom_radii, interface_void, shrake_rupley

logger = logging.getLogger(__name__)

__all__ = ["AtomSet", "InterfaceScore", "InterfaceScorer", "read_atoms"]

#: Atoms that can donate a hydrogen bond, by element. Without explicit
#: hydrogens a nitrogen is treated as a donor and an oxygen as both donor and
#: acceptor, which is the usual convention for heavy-atom-only analysis.
_DONORS = frozenset("NO")
_ACCEPTORS = frozenset("ON")

#: How far from the partner an atom must be before its buried area is certainly
#: zero, in Å. Two expanded surfaces can only touch within r_i + r_j + 2*probe,
#: which is at most 1.8 + 1.8 + 2.8 = 6.4 Å; 8 Å leaves margin.
_SASA_REACH = 8.0

#: One parsed atom: coordinates, element, atom name, residue name, and the
#: (chain, number, insertion code) that identifies its residue.
_Record = tuple[np.ndarray, str, str, str, tuple[str, int, str]]

#: Charged groups, by residue and atom name. Histidine is included: at
#: physiological pH a fraction is protonated, and an ND1/NE2 contact with a
#: carboxylate is a real electrostatic interaction.
_POSITIVE = {
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
    ("LYS", "NZ"),
    ("HIS", "ND1"), ("HIS", "NE2"),
}
_NEGATIVE = {
    ("ASP", "OD1"), ("ASP", "OD2"),
    ("GLU", "OE1"), ("GLU", "OE2"),
}


@dataclass(frozen=True)
class AtomSet:
    """Heavy atoms of a selection, with what each term needs to know about them."""

    coords: np.ndarray = field(repr=False)
    elements: tuple[str, ...] = field(repr=False)
    atom_names: tuple[str, ...] = field(repr=False)
    residue_names: tuple[str, ...] = field(repr=False)
    residue_keys: tuple[tuple[str, int, str], ...] = field(repr=False)

    def __len__(self) -> int:
        return len(self.coords)

    @property
    def radii(self) -> np.ndarray:
        return atom_radii(list(self.elements))

    def is_charged(self, sign: int) -> np.ndarray:
        """``(N,)`` mask of atoms in a positively (+1) or negatively (-1) charged group."""
        table = _POSITIVE if sign > 0 else _NEGATIVE
        return np.array(
            [(rn, an) in table for rn, an in zip(self.residue_names, self.atom_names,
                                                 strict=True)],
            dtype=bool,
        )


@dataclass(frozen=True)
class InterfaceScore:
    """Every interface term for one model. Units: Å², Å³, counts.

    Attributes:
        buried_area: Antibody–antigen buried surface area, Å².
        loop_buried_area: The graft's own buried area, Å².
        contacts: Loop–antigen heavy-atom pairs within the contact cutoff.
        contact_residues: Loop residues with at least one such pair.
        clashes: Element-aware unacceptable contacts, loop to antigen.
        hbonds: Loop–antigen hydrogen bonds.
        salt_bridges: Loop–antigen salt bridges.
        gap_volume: Void trapped between loop and antigen, Å³.
        gap_index: ``gap_volume / loop_buried_area``, Å. Lower is tighter.
        min_distance: Closest loop–antigen heavy-atom approach, Å.
    """

    buried_area: float
    loop_buried_area: float
    contacts: int
    contact_residues: int
    clashes: int
    hbonds: int
    salt_bridges: int
    gap_volume: float
    gap_index: float
    min_distance: float

    @property
    def is_clean(self) -> bool:
        """True when nothing clashes. Every other term presupposes this."""
        return self.clashes == 0

    def as_dict(self) -> dict[str, float]:
        """Flat mapping, for tables and for z-scoring."""
        return {
            "buried_area": self.buried_area,
            "loop_buried_area": self.loop_buried_area,
            "contacts": float(self.contacts),
            "contact_residues": float(self.contact_residues),
            "clashes": float(self.clashes),
            "hbonds": float(self.hbonds),
            "salt_bridges": float(self.salt_bridges),
            "gap_volume": self.gap_volume,
            "gap_index": self.gap_index,
            "min_distance": self.min_distance,
        }

    def describe(self) -> str:
        """One-line summary."""
        return (
            f"BSA {self.buried_area:.0f} Å² (loop {self.loop_buried_area:.0f}), "
            f"{self.contacts} contacts over {self.contact_residues} residues, "
            f"{self.hbonds} H-bonds, {self.salt_bridges} salt bridges, "
            f"gap index {self.gap_index:.2f} Å, "
            f"{self.clashes} clashes"
        )


class InterfaceScorer:
    """Measures the interface of a packed graft.

    One instance serves a whole target, though it holds no state: buried area
    is summed over interface atoms only, which is exact and leaves nothing
    worth caching.

    Args:
        cfg: Geometric criteria; defaults to :class:`InterfaceConfig`.
    """

    def __init__(self, cfg: InterfaceConfig | None = None) -> None:
        self._cfg = cfg or InterfaceConfig()

    @property
    def config(self) -> InterfaceConfig:
        return self._cfg

    def prime(self, site: object) -> None:
        """Kept as a no-op so callers need not know this is no longer needed.

        An earlier version pre-computed the antigen's whole unbound surface area
        once per target. Buried area is now summed over interface atoms only
        (see :meth:`score`), which is both exact and far cheaper, so there is
        nothing left to cache.
        """

    def score(
        self,
        model: Path,
        ab_chain: str,
        span: tuple[int, int],
        antigen_chains: tuple[str, ...] | list[str],
    ) -> InterfaceScore:
        """Measure one model.

        Args:
            model: A packed PDB file: the template with the graft in place and
                the antigen present. Scoring a backbone-only model is possible
                but measures a different thing and will read as a loose
                interface throughout.
            ab_chain: Chain carrying the graft.
            span: ``(first, last)`` residue numbers of the graft.
            antigen_chains: Chains treated as antigen.

        Returns:
            An :class:`InterfaceScore`.

        Raises:
            ValueError: If either side of the interface has no atoms.
        """
        cfg = self._cfg
        antibody, antigen, loop_mask = read_atoms(
            model, ab_chain, span, tuple(antigen_chains)
        )
        if not len(antibody) or not len(antigen):
            raise ValueError(
                f"{model.name}: antibody has {len(antibody)} atoms and antigen "
                f"{len(antigen)}; both sides are needed"
            )
        if not loop_mask.any():
            raise ValueError(
                f"{model.name}: no atoms in {ab_chain}/{span[0]}-{span[1]}, so "
                "there is no graft to measure"
            )

        loop = _subset(antibody, loop_mask)

        # ---- buried surface -------------------------------------------- #
        # BSA = SASA(antibody) + SASA(antigen) - SASA(complex), which is the sum
        # over atoms of (area alone - area in the complex). An atom out of the
        # partner's reach has the same area in both states and contributes
        # exactly zero, so only atoms near the interface need measuring. Every
        # atom still occludes; what is skipped is arithmetic that cancels.
        from scipy.spatial import cKDTree

        ab_radii, ag_radii = antibody.radii, antigen.radii
        tree_ag = cKDTree(antigen.coords)
        tree_ab = cKDTree(antibody.coords)
        ab_near = np.flatnonzero(
            tree_ag.query_ball_point(
                antibody.coords, _SASA_REACH, return_length=True
            )
        )
        ag_near = np.flatnonzero(
            tree_ab.query_ball_point(
                antigen.coords, _SASA_REACH, return_length=True
            )
        )

        buried_area = 0.0
        loop_buried = 0.0
        if len(ab_near) and len(ag_near):
            all_coords = np.concatenate([antibody.coords, antigen.coords])
            all_radii = np.concatenate([ab_radii, ag_radii])
            n_ab = len(antibody)
            points, probe = cfg.sasa_points, cfg.probe_radius

            ab_alone = shrake_rupley(
                antibody.coords, ab_radii, probe, points, indices=ab_near
            )
            ag_alone = shrake_rupley(
                antigen.coords, ag_radii, probe, points, indices=ag_near
            )
            ab_bound = shrake_rupley(
                all_coords, all_radii, probe, points, indices=ab_near
            )
            ag_bound = shrake_rupley(
                all_coords, all_radii, probe, points, indices=n_ab + ag_near
            )
            buried_area = float(
                (ab_alone - ab_bound).sum() + (ag_alone - ag_bound).sum()
            )
            loop_buried = float((ab_alone - ab_bound)[loop_mask[ab_near]].sum())

        # ---- contacts, clashes ----------------------------------------- #
        pairs = tree_ag.query_ball_point(loop.coords, cfg.contact_cutoff)
        contacts = sum(len(group) for group in pairs)
        contact_residues = len({
            loop.residue_keys[i] for i, group in enumerate(pairs) if group
        })
        clashes, min_distance = _clash_count(loop, antigen)

        # ---- polar complementarity ------------------------------------- #
        hbonds = _hbond_count(loop, antigen, cfg)
        salt = _salt_bridge_count(loop, antigen, cfg.salt_bridge_distance)

        # ---- packing quality ------------------------------------------- #
        gap_volume = interface_void(
            loop.coords, loop.radii, antigen.coords, ag_radii,
            probe=cfg.probe_radius, spacing=cfg.grid_spacing, margin=cfg.grid_margin,
        )
        gap_index = gap_volume / loop_buried if loop_buried > 1.0 else float("inf")

        return InterfaceScore(
            buried_area=buried_area,
            loop_buried_area=loop_buried,
            contacts=int(contacts),
            contact_residues=int(contact_residues),
            clashes=int(clashes),
            hbonds=int(hbonds),
            salt_bridges=int(salt),
            gap_volume=float(gap_volume),
            gap_index=float(gap_index),
            min_distance=float(min_distance),
        )


# --------------------------------------------------------------------------- #
# Reading a model
# --------------------------------------------------------------------------- #


def read_atoms(
    model: Path,
    ab_chain: str,
    span: tuple[int, int],
    antigen_chains: tuple[str, ...],
) -> tuple[AtomSet, AtomSet, np.ndarray]:
    """Split a model into antibody and antigen heavy atoms.

    Hydrogens are dropped — the packer does not add them and the criteria here
    are all heavy-atom ones — and only the first alternate location of any atom
    is kept. Water is dropped too: ordered solvent would otherwise occlude the
    surfaces whose burial is being measured and be counted as part of the
    epitope. Other heteroatoms on the antigen side are kept, because a glycan
    at the interface is part of what a loop has to fit against.

    Returns:
        ``(antibody, antigen, loop_mask)`` where ``loop_mask`` indexes the
        antibody set and marks the grafted residues.
    """
    lo, hi = span
    ab: list[_Record] = []
    ag: list[_Record] = []
    mask: list[bool] = []

    for line in Path(model).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[16] not in (" ", "A"):
            continue
        element = pk.element_of(line)
        if element == "H" or line[17:20].strip() in pk.WATER_NAMES:
            continue
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        chain = line[21]
        record = (
            pk._coord(line), element, line[12:16].strip(), line[17:20].strip(),
            (chain, resnum, line[26]),
        )
        if chain in antigen_chains:
            ag.append(record)
        elif line.startswith("ATOM"):
            ab.append(record)
            mask.append(chain == ab_chain and lo <= resnum <= hi)

    return (
        _atomset(ab),
        _atomset(ag),
        np.array(mask, dtype=bool) if mask else np.zeros(0, dtype=bool),
    )


def _atomset(records: list[_Record]) -> AtomSet:
    if not records:
        return AtomSet(
            coords=np.zeros((0, 3)), elements=(), atom_names=(),
            residue_names=(), residue_keys=(),
        )
    coords, elements, names, resnames, keys = zip(*records, strict=True)
    return AtomSet(
        coords=np.asarray(coords, dtype=np.float64),
        elements=tuple(elements),
        atom_names=tuple(names),
        residue_names=tuple(resnames),
        residue_keys=tuple(keys),
    )


def _subset(atoms: AtomSet, mask: np.ndarray) -> AtomSet:
    idx = np.flatnonzero(mask)
    return AtomSet(
        coords=atoms.coords[idx],
        elements=tuple(atoms.elements[i] for i in idx),
        atom_names=tuple(atoms.atom_names[i] for i in idx),
        residue_names=tuple(atoms.residue_names[i] for i in idx),
        residue_keys=tuple(atoms.residue_keys[i] for i in idx),
    )


# --------------------------------------------------------------------------- #
# Individual terms
# --------------------------------------------------------------------------- #


def _clash_count(loop: AtomSet, antigen: AtomSet) -> tuple[int, float]:
    """Element-aware clashes between loop and antigen, and closest approach.

    Uses the thresholds calibrated on native interfaces in
    :mod:`epifitscout.build.packer`: a polar–polar pair is allowed to 2.2 Å
    because it may be a hydrogen bond, any pair involving carbon only to 2.8 Å.
    One distance cutoff for both would either call good hydrogen bonds clashes
    or let steric overlaps pass.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(antigen.coords)
    groups = tree.query_ball_point(loop.coords, pk.VDW_MIN)
    clashes = 0
    for i, group in enumerate(groups):
        if not group:
            continue
        polar_i = loop.elements[i] in pk._POLAR
        d = np.linalg.norm(antigen.coords[group] - loop.coords[i], axis=1)
        for j, dist in zip(group, d, strict=True):
            allowed = (
                pk.HBOND_MIN
                if polar_i and antigen.elements[j] in pk._POLAR
                else pk.VDW_MIN
            )
            if dist < allowed:
                clashes += 1
    min_distance = float(tree.query(loop.coords, k=1)[0].min())
    return clashes, min_distance


def _hbond_count(loop: AtomSet, antigen: AtomSet, cfg: InterfaceConfig) -> int:
    """Hydrogen bonds between loop and antigen, by heavy-atom geometry.

    Distance plus an angle at the acceptor. The angle needs the acceptor's
    antecedent — the atom it is covalently bound to — which is found as its
    nearest neighbour inside the same residue rather than from a topology
    table, so it works on any residue the packer may have written. Without the
    angle, every N/O pair at 3.5 Å counts and the term degenerates into a
    second contact count.
    """
    from scipy.spatial import cKDTree

    donors = np.array([e in _DONORS for e in loop.elements], dtype=bool)
    acceptors = np.array([e in _ACCEPTORS for e in antigen.elements], dtype=bool)
    if not donors.any() or not acceptors.any():
        return 0

    loop_polar = _subset(loop, donors)
    ag_polar_idx = np.flatnonzero(acceptors)
    tree = cKDTree(antigen.coords[ag_polar_idx])
    groups = tree.query_ball_point(loop_polar.coords, cfg.hbond_distance)

    antecedents = _antecedents(antigen)
    count = 0
    for i, group in enumerate(groups):
        for g in group:
            j = int(ag_polar_idx[g])
            base = antecedents[j]
            if base < 0:
                count += 1               # no antecedent found; distance alone
                continue
            angle = _angle(
                antigen.coords[base], antigen.coords[j], loop_polar.coords[i]
            )
            if angle >= cfg.hbond_angle:
                count += 1
    return count


def _antecedents(atoms: AtomSet) -> np.ndarray:
    """``(N,)`` index of each atom's covalently bonded neighbour, or -1.

    The nearest atom within 1.8 Å that belongs to the same residue: a covalent
    bond is 1.2–1.6 Å and the next-nearest non-bonded atom in a residue is
    further than that, so proximity identifies the bond without a topology
    table.
    """
    from scipy.spatial import cKDTree

    if not len(atoms):
        return np.zeros(0, dtype=int)
    tree = cKDTree(atoms.coords)
    out = np.full(len(atoms), -1, dtype=int)
    for i, group in enumerate(tree.query_ball_point(atoms.coords, 1.8)):
        best, best_d = -1, 1.8
        for j in group:
            if j == i or atoms.residue_keys[j] != atoms.residue_keys[i]:
                continue
            d = float(np.linalg.norm(atoms.coords[j] - atoms.coords[i]))
            if d < best_d:
                best, best_d = j, d
        out[i] = best
    return out


def _salt_bridge_count(loop: AtomSet, antigen: AtomSet, cutoff: float) -> int:
    """Oppositely charged group pairs within *cutoff*, counted once per pair."""
    from scipy.spatial import cKDTree

    total = 0
    for sign in (1, -1):
        loop_idx = np.flatnonzero(loop.is_charged(sign))
        ag_idx = np.flatnonzero(antigen.is_charged(-sign))
        if not len(loop_idx) or not len(ag_idx):
            continue
        tree = cKDTree(antigen.coords[ag_idx])
        seen: set[tuple[tuple[str, int, str], tuple[str, int, str]]] = set()
        for i, group in enumerate(tree.query_ball_point(loop.coords[loop_idx], cutoff)):
            for g in group:
                seen.add((
                    loop.residue_keys[loop_idx[i]],
                    antigen.residue_keys[ag_idx[g]],
                ))
        total += len(seen)
    return total


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle a-b-c in degrees, with *b* at the vertex."""
    v1, v2 = a - b, c - b
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-8 or n2 < 1e-8:
        return 180.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))
