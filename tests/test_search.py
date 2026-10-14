"""End-to-end search against a real, if tiny, chain database.

These are the tests that would have caught a broken install: they go through
the public ``epifitscout.search`` entry point, with the database and binaries
resolved the same way a user's would be.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import epifitscout
from epifitscout import paths


@pytest.fixture
def query_pair(example_pdb: Path):
    """The CDR-H3 / epitope pair from the 1a2y anti-lysozyme Fab example."""
    qb = epifitscout.QueryBuilder(example_pdb)
    return qb.get_fragment("B", 99, 110), qb.get_fragment("C", 18, 27)


class TestQueryBuilder:
    def test_chains_and_ranges(self, example_pdb: Path) -> None:
        qb = epifitscout.QueryBuilder(example_pdb)
        assert set(qb.chains()) == {"A", "B", "C"}
        start, end = qb.residue_range("C")
        assert start < end

    def test_fragment_shape_and_sequence(self, example_pdb: Path) -> None:
        qb = epifitscout.QueryBuilder(example_pdb)
        frag = qb.get_fragment("B", 99, 110)
        assert frag.length == len(frag.sequence)
        # Backbone atoms per residue: N, CA, C(, O)
        assert frag.coords.ndim == 3
        assert frag.coords.shape[0] == frag.length
        assert frag.ca_coords.shape == (frag.length, 3)

    def test_unknown_chain_raises(self, example_pdb: Path) -> None:
        qb = epifitscout.QueryBuilder(example_pdb)
        with pytest.raises((KeyError, ValueError)):
            qb.get_fragment("Z", 1, 10)


class TestSetupErrors:
    """A user who has not pasted the files in yet must be told what to do."""

    def test_search_without_master(self, empty_repo: Path, query_pair) -> None:
        cdr, epitope = query_pair
        with pytest.raises(paths.MissingBinaryError, match="MASTER/bin|master-bin-v1.6"):
            epifitscout.search(cdr, epitope)

    def test_search_without_database(
        self, empty_repo: Path, query_pair, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cdr, epitope = query_pair
        fake = empty_repo / "MASTER" / "bin" / "master"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        with pytest.raises(paths.MissingDatabaseError, match="Chain database not found"):
            epifitscout.search(cdr, epitope)


@pytest.mark.needs_master
@pytest.mark.slow
class TestEndToEnd:
    @pytest.fixture(autouse=True)
    def _master(self, master_env: Path) -> None:
        """Every test here needs the real binaries resolved."""

    def test_search_returns_ranked_hits(
        self, mini_db_built: Path, query_pair, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cdr, epitope = query_pair
        monkeypatch.setenv("EPIFITSCOUT_DB", str(mini_db_built / "pds.list"))

        hits = epifitscout.search(cdr, epitope, max_workers=1, max_hits=20)

        assert hits, "a 14-chain database containing anti-lysozyme Fabs should hit"
        scores = [h.final_score for h in hits]
        assert scores == sorted(scores, reverse=True), "hits must be best-first"

        for hit in hits:
            assert 0.0 <= hit.final_score <= 1.0
            assert 0.0 <= hit.complementarity_score <= 1.0
            assert hit.rmsd_cdr >= 0.0
            assert hit.fragment.pdb_id
            assert hit.fragment.chain
            assert hit.fragment.residue_range[0] <= hit.fragment.residue_range[1]
            assert len(hit.fragment.sequence) == hit.fragment.length

    def test_rmsd_threshold_is_respected(
        self, mini_db_built: Path, query_pair, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cdr, epitope = query_pair
        monkeypatch.setenv("EPIFITSCOUT_DB", str(mini_db_built / "pds.list"))

        loose = epifitscout.search(cdr, epitope, rmsd_threshold=2.0, max_workers=1)
        tight = epifitscout.search(cdr, epitope, rmsd_threshold=0.5, max_workers=1)

        assert all(h.rmsd_cdr <= 2.0 for h in loose)
        assert all(h.rmsd_cdr <= 0.5 for h in tight)
        assert len(tight) <= len(loose)

    def test_max_hits_caps_results(
        self, mini_db_built: Path, query_pair, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cdr, epitope = query_pair
        monkeypatch.setenv("EPIFITSCOUT_DB", str(mini_db_built / "pds.list"))
        hits = epifitscout.search(cdr, epitope, max_hits=3, max_workers=1)
        assert len(hits) <= 3

    def test_sharded_and_sequential_agree(
        self, mini_db_built: Path, query_pair, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parallel sharding must not change which fragments are found.

        The merge step re-sorts by RMSD across shards; if it were wrong, results
        would silently depend on the worker count.
        """
        cdr, epitope = query_pair
        monkeypatch.setenv("EPIFITSCOUT_DB", str(mini_db_built / "pds.list"))

        sequential = epifitscout.search(cdr, epitope, max_workers=1, max_hits=50)
        sharded = epifitscout.search(cdr, epitope, max_workers=3, max_hits=50)

        def identity(hits):
            return sorted(
                (h.fragment.pdb_id, h.fragment.chain, h.fragment.residue_range)
                for h in hits
            )

        assert identity(sequential) == identity(sharded)

    def test_weights_change_the_ranking(
        self, mini_db_built: Path, query_pair, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cdr, epitope = query_pair
        monkeypatch.setenv("EPIFITSCOUT_DB", str(mini_db_built / "pds.list"))

        rmsd_only = epifitscout.search(
            cdr, epitope, weight_rmsd=1.0, weight_shape=0.0, max_workers=1
        )
        shape_only = epifitscout.search(
            cdr, epitope, weight_rmsd=0.0, weight_shape=1.0, max_workers=1
        )

        assert rmsd_only and shape_only
        # Pure-RMSD ranking must be monotonic in RMSD; the combined one need not be.
        rmsds = [h.rmsd_cdr for h in rmsd_only]
        assert rmsds == sorted(rmsds)
        assert {h.final_score for h in rmsd_only} != {h.final_score for h in shape_only}


@pytest.mark.needs_master
@pytest.mark.slow
def test_search_works_from_the_pasted_folders(
    empty_repo: Path, mini_db_built: Path, query_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user journey: copy MASTER into MASTER/bin and the dataset into data/.

    No environment variables and no keyword arguments, only files in the two
    placeholder folders, which is exactly what the README tells people to do.
    """
    import shutil

    from tests.conftest import MASTER_DIR  # noqa: PLC0415

    assert MASTER_DIR is not None
    for name in ("master", "createPDS"):
        shutil.copy2(MASTER_DIR / name, empty_repo / "MASTER" / "bin" / name)
    shutil.copytree(mini_db_built, empty_repo / "data" / "sabdab_chains.db")

    cdr, epitope = query_pair
    hits = epifitscout.search(cdr, epitope, max_workers=1, max_hits=10)
    assert hits
    assert all(h.rmsd_cdr >= 0 for h in hits)
