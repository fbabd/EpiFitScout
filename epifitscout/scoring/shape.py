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

Final score
-----------
    depth_score = -pearson(depth_cdr_aligned, depth_epi_aligned)   # anti = good
    tau_score   =  pearson(tau_cdr_aligned,   tau_epi_aligned)      # co   = good
    raw         = 0.7 * depth_score + 0.3 * tau_score              ∈ [-1, 1]
    score       = clip((raw + 1) / 2, 0, 1)                        ∈ [0,  1]
"""

from __future__ import annotations

import numpy as np


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


def _dtw_align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DTW-align two 1-D signals; return paired values after warping.

    Kept for external use / visualisation.  The main scorer uses _interp_align
    instead — DTW is incorrect for anti-correlated depth profiles because it
    minimises paired distance rather than preserving positional correspondence.
    """
    from scipy.spatial.distance import cdist
    m, n = len(a), len(b)
    cost = cdist(a[:, None], b[:, None], metric="sqeuclidean")

    acc = np.full((m, n), np.inf)
    acc[0, 0] = cost[0, 0]
    for i in range(1, m):
        acc[i, 0] = acc[i - 1, 0] + cost[i, 0]
    for j in range(1, n):
        acc[0, j] = acc[0, j - 1] + cost[0, j]
    for i in range(1, m):
        for j in range(1, n):
            acc[i, j] = cost[i, j] + min(
                acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]
            )

    i, j = m - 1, n - 1
    pa: list[float] = []
    pb: list[float] = []
    while i > 0 or j > 0:
        pa.append(a[i]); pb.append(b[j])
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            move = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
            if move == 0:
                i -= 1; j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
    pa.append(a[0]); pb.append(b[0])
    return np.array(pa[::-1]), np.array(pb[::-1])


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
    """Compute the interface normal from CDR to epitope, with the CDR backbone
    direction projected out.

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
# Scorer
# ---------------------------------------------------------------------------

class ShapeComplementarityScorer:
    """3D loop shape complementarity via depth profiles and backbone torsion.

    Component 1 — Depth profile (weight_depth, default 0.7):
        Project CDR and epitope CAs onto the backbone-corrected CDR→epitope
        interface normal.  Anti-correlated depth profiles (convex CDR into
        concave epitope) score high.  Linear interpolation to a common length
        preserves positional correspondence (N-term aligns with N-term).

    Component 2 — Torsion similarity (weight_tau, default 0.3):
        Frame-independent Frenet-Serret torsion τ of the CA trace captures 3D
        twist.  Co-correlated τ profiles score high.  Linear interpolation used
        for the same positional-correspondence reason.

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
    ) -> float:
        """Compute shape complementarity score.

        Args:
            hit_cdr_aligned: (N, 4, 3) or (N, 3) CDR hit backbone coords,
                             already superposed onto the query frame by MASTER.
            query_epitope:   (M, 4, 3) or (M, 3) query epitope backbone coords.

        Returns:
            Score in [0, 1].  Higher = better shape complementarity.
        """
        cdr_ca = self._get_ca(hit_cdr_aligned)
        epi_ca = self._get_ca(query_epitope)

        if len(cdr_ca) < 4 or len(epi_ca) < 4:
            return 0.0

        # ------------------------------------------------------------------
        # Component 1: depth profile along backbone-corrected interface normal
        # ------------------------------------------------------------------
        n_hat = _interface_normal(cdr_ca, epi_ca)
        if np.linalg.norm(n_hat) < 1e-6:
            return 0.0

        depth_cdr = cdr_ca @ n_hat    # (N,) projection along approach axis
        depth_epi = epi_ca @ n_hat    # (M,)

        # Centre profiles (remove global offset — only shape matters)
        depth_cdr = depth_cdr - depth_cdr.mean()
        depth_epi = depth_epi - depth_epi.mean()

        # Linear interpolation preserves positional correspondence;
        # DTW would incorrectly map peak to trough (same value, wrong position)
        dc_a, de_a = _interp_align(depth_cdr, depth_epi)
        depth_score = -_pearson(dc_a, de_a)   # anti-correlation = complementarity

        # ------------------------------------------------------------------
        # Component 2: backbone torsion similarity
        # ------------------------------------------------------------------
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
        return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))

    @staticmethod
    def _get_ca(coords: np.ndarray) -> np.ndarray:
        """Extract (N, 3) CA coords from (N, 4, 3) or (N, 3)."""
        if coords.ndim == 3 and coords.shape[1] == 4:
            return coords[:, 1, :]
        return coords
