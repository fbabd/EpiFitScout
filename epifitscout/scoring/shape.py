"""Shape complementarity scorer using interface depth profiles and backbone torsion.

Theory
------
Two components are combined:

1. Depth profile complementarity (weight 0.7)
   The CDR hit is already in the query frame (MASTER superposition), so the
   CDR and epitope share a common coordinate system.  Define the interface
   normal as the unit vector from the CDR centroid to the epitope centroid,
   with the backbone-direction component removed (via SVD on combined CAs) to
   prevent lateral loop extent from contaminating the depth signal.
   Project each CA onto this corrected normal to get a 1-D "depth profile".
   Complementary surfaces have *anti-correlated* depth profiles: where the CDR
   bulges toward the epitope (high projection), the epitope is recessed (low
   projection).  Linear interpolation to a common length handles length
   differences while preserving positional correspondence (N-term to N-term,
   C-term to C-term).  NOTE: DTW is intentionally NOT used here — DTW
   minimises paired distance, which maps CDR peaks onto epitope troughs (same
   value) rather than same position, destroying the anti-correlation signal.

2. Backbone torsion similarity (weight 0.3)
   Torsion τ from the Frenet-Serret CA curve is frame-independent and captures
   the 3D twist / handedness of the loop.  Complementary loops that maintain
   contact along their length tend to share a similar 3D twist.  Linear
   interpolation is used (same rationale as above).

DEPTH SIGN — corrected from v1
------------------------------
v1 scored ``depth_score = -pearson(...)``, on the reasoning that where the CDR
bulges toward the epitope the epitope is recessed, so the profiles should be
anti-correlated. **That reasoning does not survive the geometry.** Both profiles
are projected onto the *same* normal, the one running CDR → epitope. A bulge on
the loop and the recess it nests into both lie *further along that axis*, so two
complementary surfaces are correlated. Anti-correlation would be right only if
the epitope's depth were measured along the opposite direction, which it is not.

Measured on three synthetic pairs whose answer is not in doubt, correlation
tracks fit quality monotonically while its negation inverts the ranking:

    loop nests perfectly   pearson = +1.00   gap std 0.00 Å
    flat lid over a bowl   pearson =  0.00   gap std 0.70 Å
    opposite curvature     pearson = -1.00   gap std 1.39 Å

So the sign is positive here. This matches v2's ``contact_fit`` note that the
v1 profile correlation "had no discriminating power, and its sign was
inverted" -- it was ranking interpenetrating loops above complementary ones,
which is a sufficient explanation for v1's term appearing to do nothing.
:func:`sanity_check_sign` is the regression guard; ``tests/test_shape.py``
fails if the sign flips back.

Final score
-----------
    depth_score =  pearson(depth_cdr_aligned, depth_epi_aligned)   # co   = good
    tau_score   =  pearson(tau_cdr_aligned,   tau_epi_aligned)      # co   = good
    raw         = 0.7 * depth_score + 0.3 * tau_score              ∈ [-1, 1]
    score       = clip((raw + 1) / 2, 0, 1)                        ∈ [0,  1]

Provenance and a warning
------------------------
This is the v1 scorer, restored deliberately. Two prior findings bear on it and
neither is a reason to skip it, but both are reasons to measure it here rather
than assume it works:

* v1's own head-to-head ranked one set of MASTER hits with and without this
  term and found no difference in sequence recovery (AAR best-of-10 0.319 with
  against 0.330 without, sign test p = 0.58). That test judged the term on a
  quantity it never claimed to improve. The same note records what it *did* do:
  hits that were clashing or had no sustained contact scored 0 and were buried.
* v2's ``contact_fit`` module states this profile correlation had "no
  discriminating power, and its sign was inverted". The sign is checkable, and
  :func:`sanity_check_sign` does exactly that, so a silent inversion cannot
  survive here unnoticed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ShapeComplementarityScorer",
    "ShapeScore",
    "sanity_check_sign",
]


# ---------------------------------------------------------------------------
# Frenet-Serret torsion
# ---------------------------------------------------------------------------

def _smooth_ca(ca: np.ndarray) -> np.ndarray:
    """3-point moving average to stabilise torsion; endpoints unchanged."""
    if len(ca) < 3:
        return ca.copy()
    out = ca.copy()
    out[1:-1] = (ca[:-2] + ca[1:-1] + ca[2:]) / 3.0
    return out


def _torsion_profile(ca: np.ndarray) -> np.ndarray:
    """Compute discrete backbone torsion τ along a CA trace.

    Args:
        ca: (N, 3) CA coordinates (smoothed recommended).

    Returns:
        (N-3,) torsion values (signed, radians-equivalent).

    Raises:
        ValueError: if fewer than 4 residues.
    """
    if len(ca) < 4:
        raise ValueError(f"Need ≥ 4 CA atoms for torsion, got {len(ca)}")

    diff = ca[1:] - ca[:-1]                              # (N-1, 3)
    seg_len = np.linalg.norm(diff, axis=1, keepdims=True)
    seg_len = np.where(seg_len < 1e-8, 1e-8, seg_len)
    T = diff / seg_len                                   # (N-1, 3) tangents

    cross = np.cross(T[:-1], T[1:])                     # (N-2, 3) binormal direction
    cross_norm = np.linalg.norm(cross, axis=1, keepdims=True)
    cross_norm = np.where(cross_norm < 1e-8, 1e-8, cross_norm)
    B = cross / cross_norm                               # (N-2, 3) binormals

    dT = T[1:] - T[:-1]                                 # (N-2, 3)
    dT_norm = np.linalg.norm(dT, axis=1, keepdims=True)
    dT_norm = np.where(dT_norm < 1e-8, 1e-8, dT_norm)
    N_vec = dT / dT_norm                                 # (N-2, 3) principal normals

    dB = B[1:] - B[:-1]                                 # (N-3, 3)
    avg_seg = (seg_len[:-1] + seg_len[1:]) / 2.0        # (N-2, 1)
    avg_seg2 = (avg_seg[:-1] + avg_seg[1:]) / 2.0       # (N-3, 1)

    tau = -np.einsum("ij,ij->i", dB, N_vec[:-1]) / avg_seg2[:, 0]  # (N-3,)

    # Zero out near-zero torsion (planar loops — noise would give misleading r)
    if np.max(np.abs(tau)) < 1e-4:
        return np.zeros_like(tau)
    return tau


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def _interp_align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample two 1-D signals to a common length via linear interpolation.

    Preserves positional correspondence: 0% through a aligns with 0% through b.
    This is correct for both anti-correlated (depth) and co-correlated (torsion)
    profiles.

    Unlike DTW, this does NOT minimise pairwise distance — DTW would map CDR
    peaks to epitope troughs (same value) rather than same relative position,
    destroying the anti-correlation signal needed for depth scoring.
    """
    n_out = max(len(a), len(b))
    xa = np.linspace(0.0, 1.0, len(a))
    xb = np.linspace(0.0, 1.0, len(b))
    xc = np.linspace(0.0, 1.0, n_out)
    return np.interp(xc, xa, a), np.interp(xc, xb, b)


# ---------------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------------

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; returns 0.0 if either signal is constant."""
    if len(x) < 2:
        return 0.0
    sx, sy = x.std(), y.std()
    if sx < 1e-8 or sy < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Interface normal (backbone-corrected)
# ---------------------------------------------------------------------------

def _interface_normal(cdr_ca: np.ndarray, epi_ca: np.ndarray) -> np.ndarray:
    """Interface normal from CDR to epitope, with the CDR backbone direction
    projected out.

    The raw centroid-to-centroid vector picks up lateral (along-backbone)
    components when the CDR loop extends partially along the approach axis.
    SVD on the CDR CAs alone identifies the principal axis of the CDR loop
    (the direction it extends along); that component is removed from n_hat so
    that depth profiles measure only the true cross-loop variation.

    Key: only the CDR is used for the SVD, NOT the combined CDR+EPI.  Using
    the combined data would cause the interface direction itself (which
    dominates variance in well-separated pairs) to be mistakenly identified
    as the backbone direction and removed.

    Falls back to the raw centroid-to-centroid direction if the correction
    collapses the vector (CDR runs end-on toward epitope).
    """
    interface_vec = epi_ca.mean(0) - cdr_ca.mean(0)
    dist = float(np.linalg.norm(interface_vec))
    if dist < 1e-6:
        return np.zeros(3)

    n_hat = interface_vec / dist

    # SVD on CDR only: PC1 = principal axis of the CDR loop
    cdr_c = cdr_ca - cdr_ca.mean(0)
    try:
        _, _, Vt = np.linalg.svd(cdr_c, full_matrices=False)
        backbone_dir = Vt[0]
        n_corrected = n_hat - np.dot(n_hat, backbone_dir) * backbone_dir
        n_corrected_norm = float(np.linalg.norm(n_corrected))
        if n_corrected_norm > 1e-6:
            return n_corrected / n_corrected_norm
    except np.linalg.LinAlgError:
        pass

    return n_hat  # fallback: CDR runs end-on, no correction needed


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

class ShapeScore:
    """The shape score and the two components it is made of.

    Keeping the parts is not decoration: the depth term and the torsion term
    disagree often enough that a combined number alone cannot be debugged, and
    the whole reason this scorer is back is that its previous evaluation could
    not separate "the term did nothing" from "the term was measured wrong".
    """

    __slots__ = ("score", "depth", "tau", "n_epitope")

    def __init__(self, score: float, depth: float, tau: float, n_epitope: int) -> None:
        self.score = score
        self.depth = depth
        self.tau = tau
        self.n_epitope = n_epitope

    def describe(self) -> str:
        return (
            f"shape {self.score:.3f} (depth {self.depth:+.2f}, "
            f"torsion {self.tau:+.2f}, epitope {self.n_epitope} res)"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ShapeScore({self.describe()})"


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class ShapeComplementarityScorer:
    """3D loop shape complementarity via depth profiles and backbone torsion.

    Component 1 — Depth profile (weight_depth, default 0.7):
        Project CDR and epitope CAs onto the backbone-corrected CDR→epitope
        interface normal.  Anti-correlated depth profiles (convex CDR into
        concave epitope) score high.

    Component 2 — Torsion similarity (weight_tau, default 0.3):
        Frame-independent Frenet-Serret torsion τ of the CA trace captures 3D
        twist.  Co-correlated τ profiles score high.

    Both components return values in [-1, 1]; combined raw score is mapped to
    [0, 1] via (raw + 1) / 2.

    Args:
        weight_depth: Weight for depth anti-correlation component. Default 0.7.
        weight_tau:   Weight for backbone torsion co-correlation. Default 0.3.
    """

    def __init__(
        self,
        weight_depth: float = 0.7,
        weight_tau: float = 0.3,
    ) -> None:
        self._w_depth = weight_depth
        self._w_tau = weight_tau

    @property
    def name(self) -> str:
        return "shape"

    def score(
        self,
        hit_cdr_aligned: np.ndarray,
        query_epitope: np.ndarray,
    ) -> ShapeScore:
        """Compute shape complementarity.

        Args:
            hit_cdr_aligned: (N, 4, 3) or (N, 3) CDR hit backbone coords,
                             already superposed onto the query frame.
            query_epitope:   (M, 4, 3) or (M, 3) query epitope CA coords.

        Returns:
            A :class:`ShapeScore`; ``score`` is in [0, 1], higher is better.
        """
        cdr_ca = self._get_ca(hit_cdr_aligned)
        epi_ca = self._get_ca(query_epitope)

        if len(cdr_ca) < 4 or len(epi_ca) < 4:
            return ShapeScore(0.0, 0.0, 0.0, len(epi_ca))

        n_hat = _interface_normal(cdr_ca, epi_ca)
        if np.linalg.norm(n_hat) < 1e-6:
            return ShapeScore(0.0, 0.0, 0.0, len(epi_ca))

        depth_cdr = cdr_ca @ n_hat
        depth_epi = epi_ca @ n_hat
        depth_cdr = depth_cdr - depth_cdr.mean()
        depth_epi = depth_epi - depth_epi.mean()

        dc_a, de_a = _interp_align(depth_cdr, depth_epi)
        # Correlation, not anti-correlation. See DEPTH SIGN in the module
        # docstring: both profiles are projected onto the *same* normal, so a
        # loop bulge and the recess it nests into both lie further along it.
        depth_score = _pearson(dc_a, de_a)

        tau_score = 0.0
        try:
            tau_cdr = _torsion_profile(_smooth_ca(cdr_ca))
            tau_epi = _torsion_profile(_smooth_ca(epi_ca))
            if len(tau_cdr) >= 2 and len(tau_epi) >= 2:
                tc_a, te_a = _interp_align(tau_cdr, tau_epi)
                tau_score = _pearson(tc_a, te_a)
        except ValueError:
            pass

        raw = self._w_depth * depth_score + self._w_tau * tau_score
        return ShapeScore(
            float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0)),
            float(depth_score), float(tau_score), len(epi_ca),
        )

    @staticmethod
    def _get_ca(coords: np.ndarray) -> np.ndarray:
        """Extract (N, 3) CA coords from (N, 4, 3) or (N, 3)."""
        if coords.ndim == 3 and coords.shape[1] == 4:
            return coords[:, 1, :]
        return coords


# ---------------------------------------------------------------------------
# Sign check
# ---------------------------------------------------------------------------

def sanity_check_sign() -> tuple[float, float]:
    """Score a genuinely complementary pair against an interpenetrating one.

    v2's ``contact_fit`` records that this scorer's sign was inverted. That is a
    testable claim, so it is tested rather than argued about: a convex loop laid
    against a matching concave surface must score above the same loop pushed
    through it. Returns ``(complementary, clashing)``; the first must exceed the
    second or the sign is wrong.
    """
    t = np.linspace(-1.0, 1.0, 12)

    # A concave epitope surface: a trough opening toward +z.
    epitope = np.stack([t * 6.0, np.zeros_like(t), t ** 2 * 2.0], axis=1)

    # A convex loop that nests into it, held 4 Å above the trough floor.
    loop_fit = np.stack([t * 6.0, np.zeros_like(t), t ** 2 * 2.0 + 4.0], axis=1)

    # The same loop with its curvature flipped: it bulges where the surface
    # bulges, so the two interpenetrate at the ends.
    loop_bad = np.stack([t * 6.0, np.zeros_like(t), -t ** 2 * 2.0 + 4.0], axis=1)

    scorer = ShapeComplementarityScorer()
    return (
        scorer.score(loop_fit, epitope).score,
        scorer.score(loop_bad, epitope).score,
    )
