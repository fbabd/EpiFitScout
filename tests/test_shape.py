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


class TestTheStericCounterweight:
    """Shape alone picks the loops furthest inside the antigen.

    The depth correlation is maximised by pressing the loop into the surface,
    so it has to be paired with something that counts what that costs.
    """

    def test_a_buried_loop_is_counted(self) -> None:
        from epifitscout.scoring.collision import count_collisions
        antigen = np.zeros((1, 3))
        inside = np.array([[[0.0, 0.0, 0.5]]])     # 0.5 A away: interpenetrating
        outside = np.array([[[0.0, 0.0, 9.0]]])    # 9 A away: clear
        assert count_collisions(inside, antigen) == 1
        assert count_collisions(outside, antigen) == 0

    def test_nothing_to_hit_is_not_a_collision(self) -> None:
        from epifitscout.scoring.collision import count_collisions
        assert count_collisions(np.zeros((0, 4, 3)), np.zeros((5, 3))) == 0
        assert count_collisions(np.zeros((5, 4, 3)), np.zeros((0, 3))) == 0

    def test_the_native_cutoff_admits_real_interfaces(self) -> None:
        """2.0 A must not reject a contact a real complex would make.

        The closest any native CDR backbone comes to its antigen is 2.12 A over
        86 complexes, so a pair at 2.12 A is a contact, not a collision.
        """
        from epifitscout.scoring.collision import count_collisions
        antigen = np.zeros((1, 3))
        native_closest = np.array([[[0.0, 0.0, 2.12]]])
        assert count_collisions(native_closest, antigen) == 0

    def test_collision_outranks_shape(self) -> None:
        """A colliding candidate sorts below a clean one, whatever its score."""
        from epifitscout.pipeline.graft_pipeline import _order

        class Stub:
            def __init__(self, score, collide):
                self.shape = type("S", (), {"score": score})()
                self.n_collide = collide

        best_but_buried = Stub(0.99, 12)
        worse_but_clean = Stub(0.10, 0)
        ordered = _order([best_but_buried, worse_but_clean])
        assert ordered[0] is worse_but_clean

    def test_colliding_candidates_are_still_returned(self) -> None:
        """Ordering, not exclusion: when everything collides, output is not empty."""
        from epifitscout.pipeline.graft_pipeline import _order

        class Stub:
            def __init__(self, score, collide):
                self.shape = type("S", (), {"score": score})()
                self.n_collide = collide

        all_bad = [Stub(0.3, 5), Stub(0.8, 9), Stub(0.5, 1)]
        ordered = _order(all_bad)
        assert len(ordered) == 3
        assert [c.shape.score for c in ordered] == [0.8, 0.5, 0.3]
