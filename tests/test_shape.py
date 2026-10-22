"""The shape scorer, and the sign that was wrong in v1."""
from __future__ import annotations

import numpy as np
import pytest

from epifitscout.scoring.epitope import epitope_ca
from epifitscout.scoring.shape import (
    ShapeComplementarityScorer,
    sanity_check_sign,
)


def test_complementary_outscores_interpenetrating() -> None:
    """The regression guard for v1's inverted depth sign.

    If this fails, the scorer is rewarding loops that pass through the antigen,
    which is what v1 shipped and what made its epitope term look inert.
    """
    complementary, clashing = sanity_check_sign()
    assert complementary > clashing


def test_depth_correlation_tracks_fit_quality() -> None:
    """Score must fall monotonically as the surfaces stop matching."""
    t = np.linspace(-1.0, 1.0, 12)
    epitope = np.stack([t * 6.0, np.zeros_like(t), t ** 2 * 2.0], axis=1)
    nests = np.stack([t * 6.0, np.zeros_like(t), t ** 2 * 2.0 + 4.0], axis=1)
    flat = np.stack([t * 6.0, np.zeros_like(t), np.full_like(t, 5.0)], axis=1)
    opposed = np.stack([t * 6.0, np.zeros_like(t), -t ** 2 * 2.0 + 4.0], axis=1)

    scorer = ShapeComplementarityScorer()
    s_nest = scorer.score(nests, epitope).score
    s_flat = scorer.score(flat, epitope).score
    s_opp = scorer.score(opposed, epitope).score
    assert s_nest > s_flat > s_opp


def test_score_is_bounded() -> None:
    rng = np.random.default_rng(0)
    scorer = ShapeComplementarityScorer()
    for _ in range(20):
        loop = rng.normal(size=(10, 3)) * 5
        epi = rng.normal(size=(8, 3)) * 5 + 10
        s = scorer.score(loop, epi)
        assert 0.0 <= s.score <= 1.0


def test_too_few_residues_scores_zero() -> None:
    scorer = ShapeComplementarityScorer()
    loop = np.zeros((3, 3))
    epi = np.ones((3, 3))
    assert scorer.score(loop, epi).score == 0.0


def test_accepts_backbone_shaped_input() -> None:
    """(N, 4, 3) backbone input must be reduced to CA, not flattened."""
    rng = np.random.default_rng(1)
    bb = rng.normal(size=(8, 4, 3))
    epi = rng.normal(size=(6, 3)) + 10
    scorer = ShapeComplementarityScorer()
    assert 0.0 <= scorer.score(bb, epi).score <= 1.0


def test_epitope_picks_contacting_residues_only() -> None:
    """Residues beyond the cutoff must not enter the epitope."""
    loop = np.array([[0.0, 0.0, 0.0]])
    antigen = np.array([
        [2.0, 0.0, 0.0],      # residue 0, in contact
        [2.5, 0.0, 0.0],
        [50.0, 0.0, 0.0],     # residue 1, far away
        [50.5, 0.0, 0.0],
    ])
    resid = np.array([0, 0, 1, 1])
    is_ca = np.array([True, False, True, False])
    out = epitope_ca(loop, antigen, resid, is_ca, cutoff=4.5)
    assert len(out) == 1
    assert out[0][0] == pytest.approx(2.0)
