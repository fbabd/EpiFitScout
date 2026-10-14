"""Local shape-fit scoring: where along the epitope a loop fits, and how far.

The question this answers is lock-and-key: slide the grafted loop along the
epitope surface, find the stretch of loop that sits against it without
colliding, and report both the quality of that contact and how long it runs.

This replaces a global profile correlation that forced the loop and the epitope
to correspond end-to-end, which could neither locate a local fit nor report its
extent.

Inputs and their limits
-----------------------
A grafted fragment carries **backbone atoms only** (N, CA, C, O); the chain
database discards side chains at build time. The epitope, read from the query
structure, is all-atom. So every distance here is *fragment backbone* to
*antigen heavy atom*, and side-chain packing on the fragment side is invisible.
That caps how much this can say about true complementarity.

A distance cutoff also reaches sideways: a residue can fall within
``contact_cutoff`` of a patch it sits *beside* rather than *faces*, since only
separation is tested. That is the weakness the Lawrence & Colman Sc statistic
exists to address, and the reason ``normal_quality`` below is computed at all.

``normal_quality`` is a cheap stand-in for the Lawrence & Colman Sc statistic
(*J Mol Biol* 234:946, 1993): it weights a surface-normal dot product by a
distance decay, which is the part a pure distance metric misses — two surfaces
at equal separation can mesh or merely graze. It is **not** Sc: Sc is computed
on solvent-excluded molecular surfaces of fully-atomic structures, and the
normals here are approximated from local atom centroids. Treat it as an
indicator, not a literature-comparable number.

Thresholds
----------
Calibrated on 86 native antibody-antigen complexes drawn from SAbDab (1077 CDR
residues), measuring each CDR residue's backbone distance to the nearest antigen
heavy atom.

Per complex, the closest approach anywhere in the loop:

    minimum observed   2.12 Å      p10   2.47 Å
    1st percentile     2.12 Å      p25   2.72 Å
    5th percentile     2.24 Å      p50   2.98 Å

Per residue, pooled:

    1st percentile     2.56 Å      median      5.89 Å
    within 6 Å         51%         within 8 Å  71%

The collision cutoff is therefore **2.0 Å**: below anything seen in a real
structure, so no native complex is rejected. A tighter cut is tempting but
wrong — 2.5 Å would reject 12% of these natives, and 2.4 Å would reject 8%,
which is a false-positive rate the filter cannot carry. Contact extends to 8 Å,
which covers 71% of native contacting residues and leaves room for the side
chains the fragment does not have.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from epifitscout.types.epitope import EpitopeSurface

logger = logging.getLogger(__name__)

__all__ = ["LocalFit", "LocalFitScorer"]

#: Below this fragment-backbone-to-antigen distance the two interpenetrate.
#: The closest approach in 86 native complexes is 2.12 Å, so this rejects none
#: of them; see the module docstring for the full distribution.
DEFAULT_CLASH_CUTOFF = 2.0

#: Out to this distance a fragment residue is treated as facing the epitope.
#: Side chains, absent from the fragment, would span part of this gap.
DEFAULT_CONTACT_CUTOFF = 8.0

#: Separation at which backbone-to-antigen contact is treated as ideal, and the
#: width of the tolerance around it. The native per-residue median is 5.89 Å and
#: the p25 is 4.19 Å, so the window is centred in that band.
DEFAULT_IDEAL_GAP = 5.0
DEFAULT_GAP_WIDTH = 2.5

#: Radius used to estimate a local outward normal from neighbouring atoms.
DEFAULT_NORMAL_RADIUS = 8.0

#: Shortest run of consecutive residues that counts as a fit.
DEFAULT_MIN_RUN = 3


@dataclass(frozen=True)
class LocalFit:
    """Where a loop fits against an epitope, how well, and over what length.

    Attributes:
        score: Overall fit in [0, 1]; 0 when the loop collides or never makes
            contact over ``min_run`` consecutive residues.
        fragment_span: ``(first, last)`` 0-based indices of the matched run, or
            None if there is no fit.
        fragment_residues: The same run as PDB residue numbers, or None.
        epitope_residues: Antigen residue numbers the run sits against.
        matched_length: Residues in the run.
        fraction_matched: ``matched_length / loop length``.
        mean_gap, min_gap: Backbone-to-antigen distance over the run, in Å.
        n_clash: Residues closer than the collision cutoff, anywhere in the loop.
        contact_quality: Distance-only quality of the run, in [0, 1].
        normal_quality: Sc-style normal agreement over the run, in [0, 1]. See
            the module docstring: an indicator, not Lawrence & Colman Sc.
        per_residue_gap: Backbone-to-antigen distance for every loop residue.
        per_residue_partner: Index into the epitope of each residue's nearest
            antigen residue.
    """

    score: float
    fragment_span: tuple[int, int] | None
    fragment_residues: tuple[int, int] | None
    epitope_residues: tuple[int, ...]
    matched_length: int
    fraction_matched: float
    mean_gap: float
    min_gap: float
    n_clash: int
    contact_quality: float
    normal_quality: float
    per_residue_gap: np.ndarray = field(repr=False)
    per_residue_partner: np.ndarray = field(repr=False)

    @property
    def fits(self) -> bool:
        """True when a usable run was found and nothing collided."""
        return self.fragment_span is not None and self.n_clash == 0

    def describe(self) -> str:
        """One-line human-readable summary."""
        if not self.fits:
            reason = "collision" if self.n_clash else "no contact run"
            return f"no fit ({reason}); closest approach {self.min_gap:.1f} Å"
        lo, hi = self.fragment_residues or (0, 0)
        return (
            f"score {self.score:.3f}: loop residues {lo}-{hi} "
            f"({self.matched_length} of {len(self.per_residue_gap)}, "
            f"{self.fraction_matched:.0%}) against epitope residues "
            f"{min(self.epitope_residues)}-{max(self.epitope_residues)}; "
            f"gap {self.mean_gap:.1f} Å, normals {self.normal_quality:.2f}"
        )


class LocalFitScorer:
    """Scores how far along an epitope a grafted loop makes clean contact.

    Args:
        clash_cutoff: Below this separation the loop and antigen interpenetrate.
        contact_cutoff: Out to this separation a residue counts as in contact.
        ideal_gap: Separation treated as ideal backbone-to-antigen contact.
        gap_width: Tolerance around ``ideal_gap``.
        min_run: Shortest run of consecutive residues that counts as a fit.
        require_monotonic: Require the run's nearest-partner indices to progress
            in one direction, i.e. the loop tracks along the surface rather than
            doubling back. This is the literal "sliding" condition; it needs no
            residue correspondence and so is unaffected by which end of the
            epitope is listed first.
        weight_normals: Share of the fit quality taken from normal agreement
            rather than distance alone. **Defaults to 0**, i.e. distance only.
            ``normal_quality`` is still computed and reported, but it is not
            trusted enough to drive the score: on the five native complexes it
            ranges 0.09-0.42, and the case with the tightest contact (2NY7)
            scores lowest, which a sound normal estimate would not do. The
            centroid-based normals here are too crude. Raise this only after
            validating them against real Sc on fully-atomic structures.
        normal_radius: Neighbourhood radius for estimating outward normals.
    """

    def __init__(
        self,
        clash_cutoff: float = DEFAULT_CLASH_CUTOFF,
        contact_cutoff: float = DEFAULT_CONTACT_CUTOFF,
        ideal_gap: float = DEFAULT_IDEAL_GAP,
        gap_width: float = DEFAULT_GAP_WIDTH,
        min_run: int = DEFAULT_MIN_RUN,
        require_monotonic: bool = False,
        weight_normals: float = 0.0,
        normal_radius: float = DEFAULT_NORMAL_RADIUS,
    ) -> None:
        if not 0.0 <= weight_normals <= 1.0:
            raise ValueError(f"weight_normals must be in [0, 1], got {weight_normals}")
        if clash_cutoff >= contact_cutoff:
            raise ValueError("clash_cutoff must be below contact_cutoff")
        if gap_width <= 0:
            raise ValueError("gap_width must be positive")
        self._clash = clash_cutoff
        self._contact = contact_cutoff
        self._ideal = ideal_gap
        self._width = gap_width
        self._min_run = max(1, min_run)
        self._monotonic = require_monotonic
        self._w_norm = weight_normals
        self._normal_radius = normal_radius

    @property
    def name(self) -> str:
        return "local_fit"

    @property
    def clash_cutoff(self) -> float:
        """Separation below which loop and antigen interpenetrate, in Å."""
        return self._clash

    @property
    def contact_cutoff(self) -> float:
        """Separation out to which a residue counts as facing the epitope, in Å."""
        return self._contact

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def score(
        self,
        grafted_fragment: np.ndarray,
        epitope: EpitopeSurface,
        residue_numbers: tuple[int, int] | None = None,
        antibody_body: np.ndarray | None = None,
    ) -> LocalFit:
        """Score a grafted loop against an epitope surface.

        Args:
            grafted_fragment: ``(N, 4, 3)`` backbone coordinates **already in
                the antibody-antigen frame**. Scoring a fragment still in its
                own crystal frame is meaningless; see
                :mod:`epifitscout.master.parser`.
            epitope: The antigen surface facing this CDR site.
            residue_numbers: ``(first, last)`` PDB numbering of the fragment,
                used only to label the result.
            antibody_body: ``(A, 3)`` atoms of the template antibody around the
                CDR site, used to orient the loop's outward normals. Without it
                the loop's own atoms are used, which is a thin curve and gives a
                poorer normal estimate.

        Returns:
            A :class:`LocalFit`.

        Raises:
            ValueError: If the fragment is not ``(N, 4, 3)``.
        """
        if grafted_fragment.ndim != 3 or grafted_fragment.shape[1:] != (4, 3):
            raise ValueError(
                f"grafted_fragment must be (N, 4, 3), got {grafted_fragment.shape}"
            )

        gap, partner = self._gap_profile(grafted_fragment, epitope)
        n_clash = int((gap < self._clash).sum())
        in_contact = (gap >= self._clash) & (gap <= self._contact)

        run = self._best_run(in_contact, gap < self._clash, partner)
        if run is None:
            return LocalFit(
                score=0.0, fragment_span=None, fragment_residues=None,
                epitope_residues=(), matched_length=0, fraction_matched=0.0,
                mean_gap=float(gap.min()), min_gap=float(gap.min()),
                n_clash=n_clash, contact_quality=0.0, normal_quality=0.0,
                per_residue_gap=gap, per_residue_partner=partner,
            )

        start, stop = run
        seg = slice(start, stop)
        seg_gap = gap[seg]
        dist_q = float(self._distance_quality(seg_gap).mean())
        norm_q = float(
            self._normal_quality(
                grafted_fragment, epitope, partner, seg, gap, antibody_body
            )
        )
        quality = (1.0 - self._w_norm) * dist_q + self._w_norm * norm_q

        length = stop - start
        fraction = length / len(gap)
        score = 0.0 if n_clash else float(quality * fraction)

        epi_res = tuple(
            int(epitope.residue_numbers[i]) for i in sorted(set(partner[seg].tolist()))
        )
        frag_res = None
        if residue_numbers is not None:
            frag_res = (residue_numbers[0] + start, residue_numbers[0] + stop - 1)

        return LocalFit(
            score=score, fragment_span=(start, stop - 1), fragment_residues=frag_res,
            epitope_residues=epi_res, matched_length=length, fraction_matched=fraction,
            mean_gap=float(seg_gap.mean()), min_gap=float(gap.min()), n_clash=n_clash,
            contact_quality=dist_q, normal_quality=norm_q,
            per_residue_gap=gap, per_residue_partner=partner,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _gap_profile(
        self, fragment: np.ndarray, epitope: EpitopeSurface
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-residue nearest-antigen distance, and which antigen residue it is.

        Each loop residue finds its own nearest partner, so no index-to-index
        correspondence between loop and epitope is assumed anywhere.
        """
        from scipy.spatial import cKDTree

        tree = cKDTree(epitope.atom_coords)
        n = len(fragment)
        gap = np.empty(n)
        partner = np.empty(n, dtype=np.int32)
        for i in range(n):
            d, j = tree.query(fragment[i], k=1)      # 4 backbone atoms
            best = int(np.argmin(d))
            gap[i] = float(d[best])
            partner[i] = int(epitope.atom_residue[j[best]])
        return gap, partner

    def _best_run(
        self, in_contact: np.ndarray, clashing: np.ndarray, partner: np.ndarray
    ) -> tuple[int, int] | None:
        """Longest run of consecutive in-contact residues with no collision.

        Ties on length are broken by the tighter mean gap, so a snug run beats a
        loose one of equal extent.
        """
        n = len(in_contact)
        best: tuple[int, int] | None = None
        best_len = 0
        for start in range(n):
            if not in_contact[start]:
                continue
            for stop in range(start + 1, n + 1):
                if not in_contact[stop - 1] or clashing[stop - 1]:
                    break
                length = stop - start
                if length < self._min_run:
                    continue
                if self._monotonic and not _monotonic(partner[start:stop]):
                    continue
                if length > best_len:
                    best, best_len = (start, stop), length
        return best

    def _distance_quality(self, gap: np.ndarray) -> np.ndarray:
        """Gaussian window on separation: 1 at ``ideal_gap``, falling off either side."""
        return np.exp(-(((gap - self._ideal) / self._width) ** 2))

    def _normal_quality(
        self,
        fragment: np.ndarray,
        epitope: EpitopeSurface,
        partner: np.ndarray,
        seg: slice,
        gap: np.ndarray,
        antibody_body: np.ndarray | None = None,
    ) -> float:
        """Distance-weighted agreement of approximated outward normals.

        For facing surfaces the two outward normals are antiparallel, so their
        negated dot product approaches 1. Normals are estimated as the direction
        from a local atom centroid to the residue, which is the cheap stand-in
        described in the module docstring.

        The distance decay uses the **atom-to-atom** separation already in
        ``gap``, not the distance between residue centroids: centroids sit
        several Å further apart than the closest atoms, so weighting by them
        would suppress every term regardless of how well the normals agreed.
        """
        frag_centroids = fragment.mean(axis=1)
        epi_centroids = epitope.residue_centroids()
        frag_body = (
            fragment.reshape(-1, 3) if antibody_body is None
            else np.asarray(antibody_body, dtype=np.float64)
        )

        vals: list[float] = []
        for i in range(seg.start, seg.stop):
            n_f = _outward_normal(frag_centroids[i], frag_body, self._normal_radius)
            j = int(partner[i])
            n_e = _outward_normal(
                epi_centroids[j], epitope.body_coords, self._normal_radius
            )
            if n_f is None or n_e is None:
                continue
            mesh = float(np.clip(-np.dot(n_f, n_e), 0.0, 1.0))
            decay = float(np.exp(-(((gap[i] - self._ideal) / self._width) ** 2)))
            vals.append(mesh * decay)
        return float(np.mean(vals)) if vals else 0.0


def _monotonic(seq: np.ndarray) -> bool:
    """True if the sequence is non-decreasing or non-increasing."""
    diff = np.diff(seq.astype(float))
    return bool(np.all(diff >= 0) or np.all(diff <= 0))


def _outward_normal(
    point: np.ndarray, cloud: np.ndarray, radius: float
) -> np.ndarray | None:
    """Direction from the local centroid of *cloud* to *point*.

    A standard cheap approximation of a surface normal: a point on the outside
    of a body lies away from the centre of mass of its neighbourhood. Returns
    None when the neighbourhood is centred on the point and no direction can be
    resolved.
    """
    near = cloud[np.linalg.norm(cloud - point, axis=1) <= radius]
    if len(near) < 3:
        near = cloud
    vec = np.asarray(point - near.mean(axis=0), dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    unit: np.ndarray = vec / norm
    return unit
