"""Local shape-fit scoring: does the loop fit, where along the epitope, how far.

The property these tests pin down is the one the previous scorer got backwards:
**a loop that nests against the epitope must score above one that collides with
it.** The old whole-length profile correlation scored a perfect nest 0.000 and a
collision 1.000, monotonically inverted; see ``REPORT.md``.

They also pin the two things the old scorer could not express at all: *where*
along the epitope the match sits, and *how long* it runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from epifitscout.scoring.local_fit import (
    DEFAULT_CLASH_CUTOFF,
    LocalFit,
    LocalFitScorer,
)
from epifitscout.types.epitope import EpitopeSurface


def backbone(ca: np.ndarray) -> np.ndarray:
    """(N, 4, 3) backbone with plausible N/C/O hung off each Cα."""
    out = np.zeros((len(ca), 4, 3))
    out[:, 1, :] = ca
    out[:, 0, :] = ca + [-0.5, 0.4, 0.0]
    out[:, 2, :] = ca + [0.5, 0.4, 0.0]
    out[:, 3, :] = ca + [0.6, 1.0, 0.0]
    return out


def slab(x: np.ndarray, z: np.ndarray, *, thickness: int = 4) -> EpitopeSurface:
    """An epitope with real volume: several atom layers receding from the loop.

    A single line of atoms has no interior, so outward normals cannot be
    estimated from it; stacking layers gives the surface a body, as a real
    antigen has.
    """
    residues = []
    for i, (xi, zi) in enumerate(zip(x, z, strict=True)):
        atoms = np.array([[xi, 0.0, zi + 1.6 * k] for k in range(thickness)])
        residues.append((i + 1, "ALA", atoms))
    body = np.concatenate([a for _, _, a in residues], axis=0)
    return EpitopeSurface.from_residues("A", residues, body=body)


@pytest.fixture
def scorer() -> LocalFitScorer:
    return LocalFitScorer()


@pytest.fixture
def bulging_loop() -> np.ndarray:
    """A 12-residue loop protruding toward the epitope at its centre."""
    x = np.linspace(-6, 6, 12)
    return backbone(np.stack([x, np.zeros_like(x), 2.0 * (1 - (x / 6.0) ** 2)], axis=1))


class TestNestBeatsCollision:
    """The regression the old scorer failed."""

    def test_nest_scores_above_flat_above_collision(
        self, scorer: LocalFitScorer, bulging_loop: np.ndarray
    ) -> None:
        x = np.linspace(-6, 6, 12)
        bulge = 2.0 * (1 - (x / 6.0) ** 2)
        nest = scorer.score(bulging_loop, slab(x, 5.0 + bulge))
        flat = scorer.score(bulging_loop, slab(x, 5.0 + np.zeros_like(x)))
        collide = scorer.score(bulging_loop, slab(x, 5.0 - bulge))

        assert nest.score > flat.score > collide.score
        assert collide.n_clash > 0
        assert collide.score == 0.0
        assert not collide.fits
        assert nest.fits

    def test_a_collision_anywhere_zeroes_the_score(
        self, scorer: LocalFitScorer, bulging_loop: np.ndarray
    ) -> None:
        """One interpenetrating residue disqualifies the whole fit."""
        x = np.linspace(-6, 6, 12)
        z = 5.0 + np.zeros_like(x)
        z[6] = 2.0 + DEFAULT_CLASH_CUTOFF * 0.1   # drive one residue into the loop
        result = scorer.score(bulging_loop, slab(x, z))
        assert result.n_clash >= 1
        assert result.score == 0.0

    def test_a_loop_far_from_the_epitope_does_not_fit(
        self, scorer: LocalFitScorer, bulging_loop: np.ndarray
    ) -> None:
        x = np.linspace(-6, 6, 12)
        result = scorer.score(bulging_loop, slab(x, 60.0 + np.zeros_like(x)))
        assert result.score == 0.0
        assert result.fragment_span is None
        assert result.matched_length == 0
        assert result.min_gap > 40


class TestExtent:
    """"How long does it match" — the quantity the old scorer could not report."""

    def test_a_longer_match_scores_higher_than_a_shorter_one(
        self, scorer: LocalFitScorer
    ) -> None:
        x = np.arange(20) * 3.8 - 36.1          # realistic Cα spacing
        loop = backbone(np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1))
        # epitope present across the whole loop, versus only a third of it
        wide = slab(x, 5.0 + np.zeros_like(x))
        z_narrow = np.full_like(x, 5.0)
        z_narrow[:13] = 60.0            # withdraw most of the surface
        narrow = slab(x, z_narrow)

        long_fit, short_fit = scorer.score(loop, wide), scorer.score(loop, narrow)
        assert long_fit.matched_length > short_fit.matched_length
        assert long_fit.score > short_fit.score

    def test_reports_which_residues_matched(self, scorer: LocalFitScorer) -> None:
        # Realistic 3.8 Å Cα spacing matters here: with residues packed closer
        # together, a residue lateral to the patch still falls inside the 8 Å
        # contact cutoff and the run spreads beyond the patch.
        x = np.arange(20) * 3.8 - 36.1
        loop = backbone(np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1))
        z = np.full_like(x, 60.0)
        z[8:14] = 5.0                   # surface present only opposite residues 8-13
        result = scorer.score(loop, slab(x, z), residue_numbers=(100, 119))

        assert result.fragment_span is not None
        lo, hi = result.fragment_span
        # The run must sit on the patch (loop indices 8-13). One residue of
        # lateral spill either side is expected and correct: at 3.8 Å spacing a
        # neighbouring residue still falls inside the 8 Å contact cutoff.
        assert 7 <= lo <= 8, result.describe()
        assert 13 <= hi <= 14, result.describe()
        assert result.matched_length <= 8
        assert result.fragment_residues == (100 + lo, 100 + hi)
        assert result.epitope_residues
        assert result.fraction_matched == pytest.approx(
            result.matched_length / len(loop)
        )

    def test_fraction_matched_is_consistent(
        self, scorer: LocalFitScorer, bulging_loop: np.ndarray
    ) -> None:
        x = np.linspace(-6, 6, 12)
        r = scorer.score(bulging_loop, slab(x, 5.0 + 2.0 * (1 - (x / 6.0) ** 2)))
        assert 0.0 < r.fraction_matched <= 1.0
        assert r.matched_length == len(r.per_residue_gap) * r.fraction_matched


class TestNoCorrespondenceAssumed:
    """Each residue finds its own nearest partner, so epitope order is irrelevant.

    The previous scorer matched index-to-index, so reversing the epitope's
    residue order — the same 3-D object — changed the ranking completely
    (0/10 top-10 overlap on real hits).
    """

    def test_reversing_the_epitope_changes_nothing(
        self, scorer: LocalFitScorer, bulging_loop: np.ndarray
    ) -> None:
        x = np.linspace(-6, 6, 12)
        z = 5.0 + 2.0 * (1 - (x / 6.0) ** 2)
        forward = scorer.score(bulging_loop, slab(x, z))

        rev = slab(x[::-1], z[::-1])
        reverse = scorer.score(bulging_loop, rev)
        assert forward.score == pytest.approx(reverse.score, abs=1e-9)
        assert forward.matched_length == reverse.matched_length

    def test_score_is_invariant_to_rigid_motion_of_the_whole_complex(
        self, scorer: LocalFitScorer, bulging_loop: np.ndarray
    ) -> None:
        """Moving loop and epitope together is a change of frame, not of shape."""
        from scipy.spatial.transform import Rotation

        x = np.linspace(-6, 6, 12)
        z = 5.0 + 2.0 * (1 - (x / 6.0) ** 2)
        epi = slab(x, z)
        before = scorer.score(bulging_loop, epi)

        R = Rotation.random(random_state=5).as_matrix()
        t = np.array([120.0, -30.0, 11.0])
        moved_loop = np.einsum("nkj,ij->nki", bulging_loop, R) + t
        moved_epi = EpitopeSurface(
            chain=epi.chain, residue_numbers=epi.residue_numbers,
            residue_names=epi.residue_names,
            atom_coords=epi.atom_coords @ R.T + t,
            atom_residue=epi.atom_residue,
            body_coords=epi.body_coords @ R.T + t,
        )
        after = scorer.score(moved_loop, moved_epi)
        assert before.score == pytest.approx(after.score, abs=1e-9)


class TestMonotonicOption:
    def test_monotonic_requirement_is_accepted_and_no_looser(
        self, bulging_loop: np.ndarray
    ) -> None:
        """Requiring the loop to track one way along the surface cannot widen a run."""
        x = np.linspace(-6, 6, 12)
        epi = slab(x, 5.0 + 2.0 * (1 - (x / 6.0) ** 2))
        free = LocalFitScorer(require_monotonic=False).score(bulging_loop, epi)
        tracked = LocalFitScorer(require_monotonic=True).score(bulging_loop, epi)
        assert tracked.matched_length <= free.matched_length


class TestValidation:
    def test_rejects_wrongly_shaped_fragment(self, scorer: LocalFitScorer) -> None:
        x = np.linspace(-6, 6, 12)
        with pytest.raises(ValueError, match=r"\(N, 4, 3\)"):
            scorer.score(np.zeros((12, 3)), slab(x, np.full_like(x, 5.0)))

    @pytest.mark.parametrize("kwargs", [
        {"weight_normals": 1.5},
        {"weight_normals": -0.1},
        {"clash_cutoff": 9.0, "contact_cutoff": 8.0},
        {"gap_width": 0.0},
    ])
    def test_rejects_incoherent_settings(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            LocalFitScorer(**kwargs)

    def test_normal_term_is_off_by_default(self) -> None:
        """It is reported but unvalidated, so it must not drive the score."""
        assert LocalFitScorer()._w_norm == 0.0

    def test_describe_is_informative_either_way(
        self, scorer: LocalFitScorer, bulging_loop: np.ndarray
    ) -> None:
        x = np.linspace(-6, 6, 12)
        good = scorer.score(bulging_loop, slab(x, 5.0 + np.zeros_like(x)))
        bad = scorer.score(bulging_loop, slab(x, 60.0 + np.zeros_like(x)))
        assert "loop residues" in good.describe()
        assert "no fit" in bad.describe()
        assert isinstance(good, LocalFit)


class TestEpitopeSurface:
    def test_from_residues_indexes_atoms_correctly(self) -> None:
        residues = [(10, "ALA", np.zeros((3, 3))), (12, "GLY", np.ones((2, 3)))]
        epi = EpitopeSurface.from_residues("C", residues)
        assert epi.n_residues == 2
        assert epi.n_atoms == 5
        assert epi.residue_numbers == (10, 12)
        assert list(epi.atom_residue) == [0, 0, 0, 1, 1]
        assert epi.residue_centroids().shape == (2, 3)

    def test_body_defaults_to_the_patch_but_can_be_given(self) -> None:
        residues = [(1, "ALA", np.zeros((2, 3)))]
        assert EpitopeSurface.from_residues("A", residues).body_coords.shape == (2, 3)
        body = np.arange(30, dtype=float).reshape(10, 3)
        assert EpitopeSurface.from_residues("A", residues, body=body).body_coords.shape == (10, 3)

    def test_rejects_empty_and_malformed(self) -> None:
        with pytest.raises(ValueError, match="no residues"):
            EpitopeSurface.from_residues("A", [])
        with pytest.raises(ValueError, match=r"\(A, 3\)"):
            EpitopeSurface(chain="A", residue_numbers=(1,), residue_names=("ALA",),
                           atom_coords=np.zeros(3), atom_residue=np.zeros(1),
                           body_coords=np.zeros((1, 3)))


@pytest.mark.needs_master
class TestRealNatives:
    """On real complexes the loop must make clean, non-colliding contact."""

    @pytest.mark.parametrize("pid,hc,lo,hi,ac", [
        ("1a2y", "B", 96, 105, "C"),
        ("3gbn", "H", 93, 102, "B"),
        ("2ny7", "H", 93, 102, "G"),
    ])
    def test_native_loop_fits_its_own_epitope(
        self, pid: str, hc: str, lo: int, hi: int, ac: str, scorer: LocalFitScorer
    ) -> None:
        from pathlib import Path

        import epifitscout

        path = Path("database_builder/rcsb_cache") / f"{pid}.pdb"
        if not path.is_file():
            pytest.skip(f"{path} not cached; run prep_case.py or the benchmark first")
        qb = epifitscout.QueryBuilder(path)
        cdr = qb.get_fragment(hc, lo, hi)
        epi = qb.get_epitope_surface(ac, near=cdr, cutoff=10.0)

        assert epi.n_residues >= 5, "spatial selection should find a real patch"
        result = scorer.score(cdr.coords, epi, residue_numbers=cdr.residue_range)
        assert result.n_clash == 0, "a crystal structure must not self-collide"
        assert result.fits
        assert result.matched_length >= 3
        assert 2.5 <= result.mean_gap <= 8.0, result.describe()


class TestEpitopeSelection:
    """The epitope is the user's to define, and need not touch the template CDR."""

    @pytest.fixture
    def builder(self):
        from pathlib import Path

        import epifitscout

        path = Path("database_builder/rcsb_cache/1a2y.pdb")
        if not path.is_file():
            pytest.skip(f"{path} not cached")
        return epifitscout.QueryBuilder(path)

    def test_explicit_range(self, builder) -> None:
        epi = builder.get_epitope_surface("C", residues=(20, 28))
        assert epi.residue_numbers == tuple(range(20, 29))

    def test_explicit_discontinuous_list(self, builder) -> None:
        wanted = [18, 19, 20, 101, 102]
        epi = builder.get_epitope_surface("C", residues=wanted)
        assert set(epi.residue_numbers) == set(wanted)
        assert max(epi.residue_numbers) - min(epi.residue_numbers) > 50

    def test_an_epitope_the_cdr_does_not_touch_is_allowed(self, builder) -> None:
        """Targeting an epitope the template antibody does not engage is the
        normal case for a design search, not an error."""
        cdr = builder.get_fragment("B", 96, 105)
        far = builder.get_epitope_surface("C", residues=(60, 70))
        assert far.n_residues > 0
        reach = builder.epitope_reach(cdr, far)
        assert reach > 15, "this epitope should be well out of the loop's reach"

        near = builder.get_epitope_surface("C", residues=(20, 28))
        assert builder.epitope_reach(cdr, near) < reach

    def test_proximity_mode_still_works(self, builder) -> None:
        cdr = builder.get_fragment("B", 96, 105)
        epi = builder.get_epitope_surface("C", near=cdr, cutoff=10.0)
        assert epi.n_residues >= 5

    def test_requires_exactly_one_selector(self, builder) -> None:
        cdr = builder.get_fragment("B", 96, 105)
        with pytest.raises(ValueError, match="exactly one"):
            builder.get_epitope_surface("C")
        with pytest.raises(ValueError, match="exactly one"):
            builder.get_epitope_surface("C", residues=(20, 28), near=cdr)

    def test_absent_residues_are_reported_not_silently_dropped(
        self, builder, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            epi = builder.get_epitope_surface("C", residues=[20, 21, 9000, 9001])
        assert epi.n_residues == 2
        assert "absent" in caplog.text

    def test_all_residues_absent_is_an_error(self, builder) -> None:
        with pytest.raises(ValueError, match="None of the requested residues"):
            builder.get_epitope_surface("C", residues=[9000, 9001])


class TestCalibration:
    """The collision cutoff must not reject real structures.

    Calibrated on 86 native antibody-antigen complexes: the closest any native
    CDR backbone comes to its antigen is 2.12 Å. An earlier 2.5 Å cutoff, set
    from only five complexes, would have rejected 12% of them — a false-positive
    rate that would invalidate the filter's whole claim.
    """

    def test_collision_cutoff_sits_below_every_observed_native_contact(self) -> None:
        from epifitscout.scoring.local_fit import DEFAULT_CLASH_CUTOFF

        closest_native_contact_observed = 2.12
        assert DEFAULT_CLASH_CUTOFF < closest_native_contact_observed, (
            "the cutoff must reject no native complex"
        )
        assert DEFAULT_CLASH_CUTOFF > 1.0, "but it must still catch real overlap"

    def test_contact_cutoff_is_wide_enough_for_missing_side_chains(self) -> None:
        from epifitscout.scoring.local_fit import (
            DEFAULT_CONTACT_CUTOFF,
            DEFAULT_IDEAL_GAP,
        )

        # Native per-residue median separation is 5.89 A; the window must contain it.
        assert DEFAULT_IDEAL_GAP < 5.89 < DEFAULT_CONTACT_CUTOFF
