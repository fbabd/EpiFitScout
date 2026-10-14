"""Hits must be scored in the query's frame, not in their own crystal frame.

A MASTER match says a database loop *resembles* the query, not where it sits.
Coordinates are reloaded from the database's own crystal structure, so a hit
starts out wherever that crystal put it. Until it is laid onto the query it
cannot be compared with the query's epitope. For a time the pipeline skipped
that step and scored every hit in its crystal frame, so the same loop scored
differently in different crystals (0.33 / 0.78 / 0.72 for three crystals of one
antibody within 0.46 Å).

These tests state the property that was violated: **a hit's score depends only
on its shape, never on the rigid motion of the crystal it came from.**
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from epifitscout.master.parser import (
    _CONSISTENCY_TOL,
    _load_fragment,
    _superpose_onto_query,
    parse_match_file,
)
from epifitscout.scoring import ShapeComplementarityScorer
from epifitscout.types.fragment import Fragment
from epifitscout.types.superposition import Superposition
from tests.conftest import MINI_DB_SOURCE

CHAIN_PDB = MINI_DB_SOURCE / "pdb" / "1FDL_H_H.pdb"
SEG = (10, 19)  # a ten-residue stretch of that chain


def _rigid(coords: np.ndarray, seed: int, shift: float) -> np.ndarray:
    """Move (N, 4, 3) coordinates by a random rotation and a translation."""
    rotation = Rotation.random(random_state=seed).as_matrix()
    offset = np.array([shift, -0.6 * shift, 0.3 * shift])
    return np.einsum("nkj,ij->nki", coords, rotation) + offset


def _as_fragment(coords: np.ndarray, name: str = "Q") -> Fragment:
    return Fragment(
        pdb_id=name, chain="A", residue_range=(1, len(coords)),
        coords=coords, sequence="A" * len(coords),
    )


@pytest.fixture
def hit_fragment() -> Fragment:
    return _load_fragment(CHAIN_PDB, *SEG)


@pytest.fixture
def fake_db(tmp_path: Path) -> Path:
    """A database directory with the chain's PDB, laid out as MASTER reports it."""
    (tmp_path / "pdb").mkdir()
    (tmp_path / "pds").mkdir()
    shutil.copy2(CHAIN_PDB, tmp_path / "pdb" / CHAIN_PDB.name)
    return tmp_path


def _match_file(db: Path, rmsd: float, seg: tuple[int, int] = SEG) -> Path:
    path = db / "matches.txt"
    path.write_text(f"  {rmsd:.5f} {db / 'pds' / 'x'}.pds [({seg[0]},{seg[1]})]\n"
                    .replace("/x.pds", f"/{CHAIN_PDB.stem}.pds"))
    return path


class TestSuperposeOntoQuery:
    def test_recovers_an_exact_rigid_motion(self, hit_fragment: Fragment) -> None:
        query = _as_fragment(_rigid(hit_fragment.coords, seed=1, shift=85.0))
        placed = _superpose_onto_query(hit_fragment, query)
        assert placed is not None
        superposition, achieved = placed

        assert achieved == pytest.approx(0.0, abs=1e-4)
        np.testing.assert_allclose(
            superposition.apply(hit_fragment.coords), query.coords, atol=1e-3
        )

    def test_moves_all_four_backbone_atoms_not_just_calpha(
        self, hit_fragment: Fragment
    ) -> None:
        """The scorer reads N, CA, C and O; all must land in the query frame."""
        query = _as_fragment(_rigid(hit_fragment.coords, seed=2, shift=40.0))
        superposition, _ = _superpose_onto_query(hit_fragment, query)  # type: ignore[misc]
        moved = superposition.apply(hit_fragment.coords)
        for atom in range(4):
            np.testing.assert_allclose(moved[:, atom], query.coords[:, atom], atol=1e-3)

    def test_length_mismatch_cannot_be_placed(self, hit_fragment: Fragment) -> None:
        shorter = _as_fragment(hit_fragment.coords[:-1])
        assert _superpose_onto_query(hit_fragment, shorter) is None

    def test_result_is_a_proper_rotation(self, hit_fragment: Fragment) -> None:
        query = _as_fragment(_rigid(hit_fragment.coords, seed=3, shift=10.0))
        superposition, _ = _superpose_onto_query(hit_fragment, query)  # type: ignore[misc]
        assert np.linalg.det(superposition.rotation) == pytest.approx(1.0, abs=1e-8)


class TestScoreIsFrameIndependent:
    """The property the bug violated."""

    @pytest.fixture
    def epitope(self, hit_fragment: Fragment) -> np.ndarray:
        # An arbitrary patch near the loop, standing in for the query's epitope.
        return _rigid(hit_fragment.coords[::-1][:8], seed=9, shift=6.0)

    def test_score_does_not_depend_on_where_the_crystal_sat(
        self, hit_fragment: Fragment, epitope: np.ndarray
    ) -> None:
        query = _as_fragment(hit_fragment.coords)  # the query frame
        scorer = ShapeComplementarityScorer()

        scores = []
        for seed, shift in [(11, 0.0), (12, 85.0), (13, 400.0)]:
            # The *same* loop, as three different crystals might place it.
            crystal = _as_fragment(_rigid(hit_fragment.coords, seed, shift), "H")
            superposition, _ = _superpose_onto_query(crystal, query)  # type: ignore[misc]
            scores.append(scorer.score(superposition.apply(crystal.coords), epitope))

        assert max(scores) - min(scores) < 1e-6, scores

    def test_the_old_behaviour_was_not_frame_independent(
        self, hit_fragment: Fragment, epitope: np.ndarray
    ) -> None:
        """Proves the test above has teeth: identity placement differs by frame."""
        scorer = ShapeComplementarityScorer()
        scores = [
            scorer.score(
                Superposition.identity().apply(_rigid(hit_fragment.coords, s, shift)),
                epitope,
            )
            for s, shift in [(11, 0.0), (12, 85.0), (13, 400.0)]
        ]
        assert max(scores) - min(scores) > 1e-3, scores


class TestParseMatchFile:
    def test_hit_is_returned_in_the_query_frame(
        self, fake_db: Path, hit_fragment: Fragment
    ) -> None:
        query = _as_fragment(_rigid(hit_fragment.coords, seed=21, shift=120.0))
        hits = parse_match_file(_match_file(fake_db, 0.0), query)

        assert len(hits) == 1
        fragment, superposition, rmsd = hits[0]
        assert rmsd == 0.0
        np.testing.assert_allclose(
            superposition.apply(fragment.coords), query.coords, atol=1e-3
        )
        assert fragment.metadata["superposed_rmsd"] == pytest.approx(0.0, abs=1e-3)

    def test_superposition_is_not_the_identity(
        self, fake_db: Path, hit_fragment: Fragment
    ) -> None:
        query = _as_fragment(_rigid(hit_fragment.coords, seed=22, shift=120.0))
        _, superposition, _ = parse_match_file(_match_file(fake_db, 0.0), query)[0]
        assert np.linalg.norm(superposition.translation) > 50
        assert not np.allclose(superposition.rotation, np.eye(3), atol=1e-3)

    def test_a_hit_that_disagrees_with_masters_rmsd_is_dropped_loudly(
        self, fake_db: Path, hit_fragment: Fragment, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If reloaded residues don't reproduce MASTER's RMSD they are the wrong
        residues, and any frame computed from them would be wrong too."""
        query = _as_fragment(_rigid(hit_fragment.coords, seed=23, shift=30.0))
        claimed = _CONSISTENCY_TOL * 50  # MASTER "said" 0.5 Å; we measure ~0
        with caplog.at_level(logging.WARNING):
            hits = parse_match_file(_match_file(fake_db, claimed), query)
        assert hits == []
        assert "could not be placed" in caplog.text

    def test_length_mismatch_is_dropped(
        self, fake_db: Path, hit_fragment: Fragment
    ) -> None:
        query = _as_fragment(_rigid(hit_fragment.coords[:-2], seed=24, shift=30.0))
        assert parse_match_file(_match_file(fake_db, 0.0), query) == []

    def test_query_is_required(self, fake_db: Path) -> None:
        """No default: a caller must not be able to forget the query and silently
        get hits in their own crystal frames."""
        with pytest.raises(TypeError):
            parse_match_file(_match_file(fake_db, 0.0))  # type: ignore[call-arg]


@pytest.mark.needs_master
@pytest.mark.slow
class TestEndToEnd:
    def test_real_hits_are_placed_and_consistent_with_master(
        self, mini_db_built: Path, example_pdb: Path, master_env: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import epifitscout

        monkeypatch.setenv("EPIFITSCOUT_DB", str(mini_db_built / "pds.list"))
        qb = epifitscout.QueryBuilder(example_pdb)
        hits = epifitscout.search(
            qb.get_fragment("B", 99, 110), qb.get_fragment("C", 18, 27),
            max_workers=1, max_hits=20,
        )
        assert hits

        for hit in hits:
            achieved = hit.fragment.metadata["superposed_rmsd"]
            assert achieved == pytest.approx(hit.rmsd_cdr, abs=_CONSISTENCY_TOL)
            assert not np.allclose(hit.superposition.rotation, np.eye(3), atol=1e-6) or \
                np.linalg.norm(hit.superposition.translation) > 1e-6
