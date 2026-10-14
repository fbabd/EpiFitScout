"""Shared geometry primitives: Kabsch superposition, RMSD, centroid helpers."""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Compute the optimal rotation matrix that maps P onto Q (Kabsch algorithm).

    Args:
        P: (N, 3) coordinate array to be rotated.
        Q: (N, 3) reference coordinate array.

    Returns:
        (3, 3) rotation matrix R such that P @ R.T minimises RMSD to Q.
    """
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"P and Q must both be (N, 3), got {P.shape} and {Q.shape}")

    # Center both sets
    p_center = P.mean(axis=0)
    q_center = Q.mean(axis=0)
    p_c = P - p_center
    q_c = Q - q_center

    # Covariance matrix
    H = p_c.T @ q_c

    # SVD
    U, _, Vt = svd(H)

    # Ensure proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])

    rotation: np.ndarray = (Vt.T @ D @ U.T)
    return rotation


def kabsch_superpose(
    P: np.ndarray, Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Superpose P onto Q; return (rotated_P, rotation, translation).

    Args:
        P: (N, 3) moving coordinates.
        Q: (N, 3) fixed reference coordinates.

    Returns:
        Tuple of (P_superposed, rotation_matrix, translation_vector).
    """
    p_center = P.mean(axis=0)
    q_center = Q.mean(axis=0)

    rotation = kabsch_rotation(P - p_center, Q - q_center)
    translation = q_center - (p_center @ rotation.T)
    p_superposed: np.ndarray = P @ rotation.T + translation
    return p_superposed, rotation, translation


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute minimum RMSD between two (N, 3) coordinate arrays after superposition.

    Args:
        P: (N, 3) coordinate array.
        Q: (N, 3) reference coordinate array.

    Returns:
        RMSD in Angstroms after optimal superposition.
    """
    if P.shape[0] != Q.shape[0]:
        raise ValueError(
            f"P and Q must have the same number of atoms, got {P.shape[0]} and {Q.shape[0]}"
        )
    p_sup, _, _ = kabsch_superpose(P, Q)
    diff = p_sup - Q
    return float(np.sqrt((diff ** 2).sum() / len(diff)))


def centroid(coords: np.ndarray) -> np.ndarray:
    """Compute mean position of a coordinate array.

    Args:
        coords: (N, 3) or (N, K, 3) coordinate array.

    Returns:
        (3,) centroid vector.
    """
    if coords.ndim == 2:
        return coords.mean(axis=0)
    elif coords.ndim == 3:
        return coords.reshape(-1, 3).mean(axis=0)
    else:
        raise ValueError(f"coords must be 2D or 3D, got {coords.ndim}D")


def ca_coords(backbone_coords: np.ndarray) -> np.ndarray:
    """Extract CA coordinates from backbone tensor.

    Args:
        backbone_coords: (N, 4, 3) array with atom order [N, CA, C, O].

    Returns:
        (N, 3) CA coordinate array.
    """
    if backbone_coords.ndim != 3 or backbone_coords.shape[1] != 4:
        raise ValueError(
            f"backbone_coords must be (N, 4, 3), got {backbone_coords.shape}"
        )
    return backbone_coords[:, 1, :]


def pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between two sets of points.

    Args:
        A: (M, 3) coordinate array.
        B: (N, 3) coordinate array.

    Returns:
        (M, N) distance matrix.
    """
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))
