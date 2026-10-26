"""Can this loop actually be bonded into the framework?

A retrieved loop is placed by a least-squares fit of its donor's stems onto the
template's. The fit is a compromise: it leaves a residual, and that residual
lands on the two atoms that have to form new peptide bonds. So a candidate with
an excellent stem RMSD can still be ungraftable, and a candidate that is
ungraftable must not be scored, ranked or reported — a complementarity score for
a structure whose backbone is broken describes nothing.

This module is that gate. It runs before any packing or interface measurement,
on backbone coordinates only, so it is cheap enough to apply to every retrieved
candidate.

Why the gate is closure geometry and not stem RMSD
--------------------------------------------------
v1 had no gate, and the natural candidate for one — the fit residual — turns out
not to work. Framework stems are conserved, so they superpose well on every
antibody: measured over a full search, stem RMSD bunches hard against whatever
cutoff is set (median 1.7 Å at a 2.0 Å cutoff, only ~2% below 1.0 Å) and
separates almost nothing. It is a filter for "is this an antibody framework",
not a measure of graftability.

Closure geometry does separate, because it is the quantity that actually has to
be right. A peptide bond is 1.33 Å with a CA–C–N angle of 116° and a C–N–CA
angle of 122°, and a loop whose ends point the wrong way fails those regardless
of how well its stems fit. The lever arm makes this worse the longer the loop:
a sub-ångström stem fit can still leave the far junction several ångströms open.

Both junctions are measured on the model as it will be written — no refinement
is attempted. A borderline candidate could often be closed by a cycle of CCD or
a short minimisation; not doing it means this gate under-counts feasible loops
rather than over-counting them, which is the safe direction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from epifitscout.config.schema import FeasibilityConfig
from epifitscout.types.anchor_query import AnchorQuery
from epifitscout.types.candidate import GraftCandidate

logger = logging.getLogger(__name__)

__all__ = [
    "GraftFeasibility",
    "IDEAL_BOND",
    "IDEAL_CA_C_N",
    "IDEAL_C_N_CA",
    "assess",
    "assess_all",
    "query_is_bridgeable",
]

#: Ideal trans-peptide geometry (Engh & Huber 1991).
IDEAL_BOND = 1.329      # C–N, Å
IDEAL_CA_C_N = 116.2    # degrees
IDEAL_C_N_CA = 121.7    # degrees

# Backbone atom positions within a (4, 3) residue block.
_N, _CA, _C, _O = 0, 1, 2, 3


@dataclass(frozen=True)
class GraftFeasibility:
    """Junction geometry of a graft, and whether it is acceptable.

    Attributes:
        n_bond, c_bond: C–N distance at the N-side and C-side junction, in Å
            (ideal 1.33).
        n_angle_c, c_angle_c: CA–C–N angle at each junction, in degrees
            (ideal 116.2).
        n_angle_n, c_angle_n: C–N–CA angle at each junction, in degrees
            (ideal 121.7).
        n_omega, c_omega: CA–C–N–CA dihedral at each junction, in degrees.
            Reported, not gated: trans is ~180° and cis ~0°, and cis junctions
            are rare but real.
        self_clashes: Loop backbone atoms closer than the cutoff to the
            framework they are being inserted into, excluding the bonded
            flanking residues.
        stem_rmsd_bb: The fit residual carried over from retrieval, in Å.
        passed: True when nothing below failed.
        reasons: One short string per failure, empty when it passed.
    """

    n_bond: float
    c_bond: float
    n_angle_c: float
    n_angle_n: float
    c_angle_c: float
    c_angle_n: float
    n_omega: float
    c_omega: float
    self_clashes: int
    stem_rmsd_bb: float
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def worst_bond(self) -> float:
        """Junction bond length furthest from ideal, in Å."""
        return max(abs(self.n_bond - IDEAL_BOND), abs(self.c_bond - IDEAL_BOND))

    @property
    def worst_angle(self) -> float:
        """Junction bond angle furthest from ideal, in degrees."""
        return max(
            abs(self.n_angle_c - IDEAL_CA_C_N), abs(self.c_angle_c - IDEAL_CA_C_N),
            abs(self.n_angle_n - IDEAL_C_N_CA), abs(self.c_angle_n - IDEAL_C_N_CA),
        )

    def describe(self) -> str:
        """One-line summary."""
        verdict = "feasible" if self.passed else "; ".join(self.reasons)
        return (
            f"bonds {self.n_bond:.2f}/{self.c_bond:.2f} Å, "
            f"worst angle {self.worst_angle:.0f}° off, "
            f"self-clashes {self.self_clashes} — {verdict}"
        )


def query_is_bridgeable(
    query: AnchorQuery, cfg: FeasibilityConfig | None = None
) -> bool:
    """Whether a loop of this length could span this site at all.

    A property of the *site*, checked once per target rather than per candidate:
    the fit forces every candidate to span the same distance, so if the site is
    unbridgeable no retrieved loop can help.
    """
    cfg = cfg or FeasibilityConfig()
    return query.span_per_residue <= cfg.max_span_per_residue


def assess(
    candidate: GraftCandidate,
    query: AnchorQuery,
    framework_coords: np.ndarray | None = None,
    cfg: FeasibilityConfig | None = None,
) -> GraftFeasibility:
    """Measure both junctions of a placed candidate and decide on it.

    Args:
        candidate: A placed :class:`~epifitscout.types.candidate.GraftCandidate`.
        query: The query whose stems define the framework being joined to. Its
            last N-stem residue and first C-stem residue are the two residues
            the new bonds are made to.
        framework_coords: ``(A, 3)`` template atoms the loop must not clash
            with. Pass the antibody **without** the replaced span and without
            the two residues flanking it, since those are bonded to the loop.
            None skips the self-clash test.
        cfg: Thresholds; defaults to :class:`FeasibilityConfig`.

    Returns:
        A :class:`GraftFeasibility`.
    """
    cfg = cfg or FeasibilityConfig()
    loop = candidate.placed_loop

    # The framework residues the loop is bonded to, in the template's frame.
    before = query.n_stem.coords[-1]
    after = query.c_stem.coords[0]

    n_bond = _distance(before[_C], loop[0][_N])
    n_angle_c = _angle(before[_CA], before[_C], loop[0][_N])
    n_angle_n = _angle(before[_C], loop[0][_N], loop[0][_CA])
    n_omega = _dihedral(before[_CA], before[_C], loop[0][_N], loop[0][_CA])

    c_bond = _distance(loop[-1][_C], after[_N])
    c_angle_c = _angle(loop[-1][_CA], loop[-1][_C], after[_N])
    c_angle_n = _angle(loop[-1][_C], after[_N], after[_CA])
    c_omega = _dihedral(loop[-1][_CA], loop[-1][_C], after[_N], after[_CA])

    self_clashes = 0
    if framework_coords is not None and len(framework_coords):
        from scipy.spatial import cKDTree

        tree = cKDTree(np.asarray(framework_coords, dtype=np.float64))
        self_clashes = int(
            tree.query_ball_point(
                loop.reshape(-1, 3), cfg.self_clash_cutoff, return_length=True
            ).sum()
        )

    reasons: list[str] = []
    for side, bond in (("N", n_bond), ("C", c_bond)):
        if not cfg.bond_min <= bond <= cfg.bond_max:
            reasons.append(f"{side}-junction bond {bond:.2f} Å")
    for side, (a_c, a_n) in (
        ("N", (n_angle_c, n_angle_n)), ("C", (c_angle_c, c_angle_n))
    ):
        if abs(a_c - IDEAL_CA_C_N) > cfg.angle_tolerance:
            reasons.append(f"{side}-junction CA-C-N {a_c:.0f}°")
        if abs(a_n - IDEAL_C_N_CA) > cfg.angle_tolerance:
            reasons.append(f"{side}-junction C-N-CA {a_n:.0f}°")
    if candidate.stem_rmsd_bb > cfg.max_stem_rmsd:
        reasons.append(f"stem fit {candidate.stem_rmsd_bb:.2f} Å")
    if self_clashes:
        reasons.append(f"{self_clashes} clash(es) with own framework")

    return GraftFeasibility(
        n_bond=n_bond, c_bond=c_bond,
        n_angle_c=n_angle_c, n_angle_n=n_angle_n,
        c_angle_c=c_angle_c, c_angle_n=c_angle_n,
        n_omega=n_omega, c_omega=c_omega,
        self_clashes=self_clashes,
        stem_rmsd_bb=candidate.stem_rmsd_bb,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def assess_all(
    candidates: list[GraftCandidate],
    query: AnchorQuery,
    framework_coords: np.ndarray | None = None,
    cfg: FeasibilityConfig | None = None,
) -> list[GraftCandidate]:
    """Assess every candidate in place and return the feasible ones.

    The verdict is attached to each candidate either way, so a rejected one can
    still be inspected; only the returned list goes on to be scored.
    """
    for candidate in candidates:
        candidate.feasibility = assess(candidate, query, framework_coords, cfg)
    kept = [c for c in candidates if c.is_feasible]
    if candidates:
        tally: dict[str, int] = {}
        for c in candidates:
            for reason in c.feasibility.reasons:
                key = reason.split(" ")[0] + " " + reason.split(" ")[1]
                tally[key] = tally.get(key, 0) + 1
        logger.info(
            "Feasibility: %d of %d graftable%s",
            len(kept), len(candidates),
            "" if len(kept) == len(candidates)
            else "; commonest failure: " + max(tally, key=lambda k: tally[k]),
        )
    return kept


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(b) - np.asarray(a)))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle a-b-c in degrees, with *b* at the vertex."""
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    v2 = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    cos = float(np.clip(float(np.dot(v1, v2)) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _dihedral(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> float:
    """Dihedral a-b-c-d in degrees, in (-180, 180]."""
    p = [np.asarray(x, dtype=np.float64) for x in (a, b, c, d)]
    b0, b1, b2 = p[1] - p[0], p[2] - p[1], p[3] - p[2]
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    m = np.cross(n1, b1 / max(float(np.linalg.norm(b1)), 1e-8))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m, n2))
    return float(np.degrees(np.arctan2(y, x)))
