"""Solvent-accessible surface area by Shrake–Rupley, and the void between two bodies.

Two primitives the interface measurement is built on, kept here because both are
general and both are the kind of thing that is easy to get subtly wrong.

:func:`shrake_rupley`
    Per-atom SASA. The original numerical method (*J Mol Biol* 79:351, 1973):
    put a sphere of radius *r + probe* around each atom, sample it evenly, and
    count the sample points no other atom's expanded sphere covers. Implemented
    here rather than taken from a dependency so that the atom radii, the probe
    and the point count are all visible and fixed in the repository — a buried
    surface area is only comparable against another computed the same way.

:func:`interface_void`
    The empty space trapped between two bodies in contact, by grid counting.
    This stands in for Laskowski's gap volume (*J Mol Graph* 13:323, 1995),
    whose published construction is a Voronoi one; the grid approximation is
    stated as such wherever it is reported, and its discretisation error is
    about 2% of the volume at the default 0.75 Å spacing.

Why the void matters at all: buried area says how much surface two bodies hide
from solvent, not how well they fit. Two shapes can bury a great deal and still
mesh badly, leaving cavities no water can reach, and that is exactly the failure
a retrieved loop is prone to — it was folded against a different antigen. Area
and void together say more than either alone.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_RADIUS",
    "VDW_RADII",
    "atom_radii",
    "interface_void",
    "shrake_rupley",
    "sphere_points",
]

#: Bondi van der Waals radii, Å (*J Phys Chem* 68:441, 1964), with the values
#: in common use for the elements proteins and their ligands are made of.
VDW_RADII: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.80,
    "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98, "SE": 1.90,
    "NA": 2.27, "MG": 1.73, "K": 2.75, "CA": 2.31, "MN": 2.05,
    "FE": 2.05, "CO": 2.00, "NI": 1.97, "CU": 1.96, "ZN": 2.01,
}

#: Used for an element not in the table above — heavier than carbon, so an
#: unrecognised atom is never treated as smaller than it is.
DEFAULT_RADIUS = 1.80


def atom_radii(elements: list[str] | tuple[str, ...]) -> np.ndarray:
    """``(N,)`` van der Waals radii for a list of element symbols."""
    return np.array(
        [VDW_RADII.get(e.strip().upper(), DEFAULT_RADIUS) for e in elements],
        dtype=np.float64,
    )


def sphere_points(n: int) -> np.ndarray:
    """``(n, 3)`` approximately even points on the unit sphere.

    The golden-spiral construction: even to within a few percent in solid angle
    for any *n*, deterministic, and without the clustering at the poles that a
    latitude–longitude grid has. Determinism matters here — a random sampling
    would make two runs of the same measurement disagree in the last digit.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
        axis=1,
    )


def shrake_rupley(
    coords: np.ndarray,
    radii: np.ndarray,
    probe: float = 1.4,
    n_points: int = 200,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """Solvent-accessible surface area per atom, in Å².

    Args:
        coords: ``(N, 3)`` heavy-atom coordinates.
        radii: ``(N,)`` van der Waals radii, from :func:`atom_radii`.
        probe: Solvent probe radius; 1.4 Å is water and the universal
            convention.
        n_points: Sample points per atom. 200 holds the per-atom error to
            roughly 1 Å²; the error is uncorrelated between atoms, so a
            whole-interface total is far more accurate than that suggests.
        indices: Compute the area of only these atoms. Every atom in *coords*
            still occludes, so each returned value is identical to the one a
            full calculation gives — this only skips atoms whose area is not
            wanted. Buried surface is a difference between two states, and an
            atom out of reach of the partner has the same area in both and
            cancels exactly, so the difference can be summed over interface
            atoms alone. That is normally a five- to twentyfold saving, and it
            is the difference between this measurement being affordable per
            candidate and not.

    Returns:
        Per-atom SASA, for every atom or for *indices* in the order given.
        Sum it over a selection to get that selection's area *in the context of
        everything passed in* — which is the point: the same atoms are measured
        with and without a binding partner, and the difference is what the
        partner buries.
    """
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got {coords.shape}")
    if len(radii) != len(coords):
        raise ValueError("one radius per atom is required")
    if len(coords) == 0:
        return np.zeros(0)

    sphere = sphere_points(n_points)
    expanded = radii + probe
    tree = cKDTree(coords)
    reach = float(expanded.max() + expanded.max())
    wanted = (
        np.arange(len(coords)) if indices is None
        else np.asarray(indices, dtype=int)
    )
    areas = np.empty(len(wanted))

    # Squared distances throughout: a point is covered when it falls inside a
    # neighbour's expanded sphere, and comparing squares decides that exactly
    # while skipping a square root over a (points x neighbours) array.
    expanded_sq = expanded ** 2
    neighbourhoods = tree.query_ball_point(coords[wanted], reach)

    for slot, (i, neighbours) in enumerate(zip(wanted, neighbourhoods, strict=True)):
        others = [j for j in neighbours if j != i]
        if others:
            test = coords[i] + sphere * expanded[i]
            delta = test[:, None, :] - coords[others][None, :, :]
            covered = (
                np.einsum("pnk,pnk->pn", delta, delta) < expanded_sq[others][None, :]
            ).any(axis=1)
            fraction = 1.0 - covered.mean()
        else:
            fraction = 1.0
        areas[slot] = 4.0 * np.pi * expanded_sq[i] * fraction

    return areas


def interface_void(
    coords_a: np.ndarray,
    radii_a: np.ndarray,
    coords_b: np.ndarray,
    radii_b: np.ndarray,
    probe: float = 1.4,
    spacing: float = 0.75,
    margin: float = 3.0,
) -> float:
    """Volume of the empty space trapped between two bodies, in Å³.

    A grid point counts towards the void when it lies

    * outside every atom of both bodies, expanded by the probe — so it is space
      a water centre could occupy rather than space the protein fills; and
    * within ``margin`` of the expanded surface of **both** bodies — so it is
      between them, not open solvent beside them.

    The second condition is what makes this a *gap* and not just a bounding-box
    emptiness, and it is why the result falls to zero for two bodies that are
    not in contact.

    Args:
        coords_a, radii_a: First body, typically the grafted loop.
        coords_b, radii_b: Second body, typically the epitope.
        probe: Solvent probe radius.
        spacing: Grid spacing in Å. Volume error goes as spacing³ against a
            surface that is itself an approximation; 0.75 Å is ~2%.
        margin: How far from each expanded surface still counts as "between".

    Returns:
        Void volume in Å³, or 0.0 when the two bodies do not come within reach
        of each other.
    """
    from scipy.spatial import cKDTree

    coords_a = np.asarray(coords_a, dtype=np.float64)
    coords_b = np.asarray(coords_b, dtype=np.float64)
    if len(coords_a) == 0 or len(coords_b) == 0:
        return 0.0

    exp_a = np.asarray(radii_a, dtype=np.float64) + probe
    exp_b = np.asarray(radii_b, dtype=np.float64) + probe

    # Only the part of B near A can trap anything, and the box is built around
    # that overlap so the grid stays small whatever the size of the antigen.
    tree_a, tree_b = cKDTree(coords_a), cKDTree(coords_b)
    reach = float(exp_a.max() + exp_b.max() + margin)
    near_b = sorted({
        j for group in tree_b.query_ball_point(coords_a, r=reach) for j in group
    })
    if not near_b:
        return 0.0

    box_points = np.concatenate([coords_a, coords_b[near_b]], axis=0)
    lo = box_points.min(axis=0) - margin
    hi = box_points.max(axis=0) + margin
    steps = np.maximum(np.ceil((hi - lo) / spacing).astype(int), 1)
    if steps.prod() > 40_000_000:                 # ~1 GB of float64; refuse
        raise MemoryError(
            f"void grid would need {steps.prod():,} points; raise `spacing` or "
            "narrow the selection"
        )

    axes = [lo[d] + spacing * np.arange(steps[d]) for d in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    # Distance from each grid point to the nearest *expanded surface* of each
    # body: negative inside, positive in the space outside it. The nearest
    # atom centre is not always the nearest surface when radii differ, so the
    # minimum is taken over the few closest atoms rather than just one.
    surf_a = _surface_distance(tree_a, exp_a, grid)
    surf_b = _surface_distance(tree_b, exp_b, grid)

    void = (surf_a > 0) & (surf_b > 0) & (surf_a < margin) & (surf_b < margin)
    return float(void.sum()) * spacing ** 3


def _surface_distance(
    tree: object, expanded: np.ndarray, points: np.ndarray, k: int = 8
) -> np.ndarray:
    """``(P,)`` signed distance from each point to a body's expanded surface."""
    k = min(k, len(expanded))
    d, idx = tree.query(points, k=k)          # type: ignore[attr-defined]
    gap: np.ndarray = np.asarray(d, dtype=np.float64) - expanded[
        np.asarray(idx, dtype=int)
    ]
    return gap if k == 1 else np.asarray(gap.min(axis=1), dtype=np.float64)
