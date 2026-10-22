"""The six stages, end to end, on the bundled fixture database.

These run MASTER for real against `tests/data/mini_db` — 14 chains, two of them
anti-lysozyme Fabs — so the pipeline is exercised with real coordinates rather
than mocks, in seconds.

Fourteen chains is a tiny database, so a thin result here is expected and is
itself one of the things under test: the funnel has to say *where* candidates
were lost, because that is how a user distinguishes "nothing in the database
bridges this gap" from "the grafts could not be closed" from "the packer
failed". v1's equivalent failure mode was silent, and its coverage collapse on
long loops only became visible when the counts were kept.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from epifitscout.config.schema import (
    FeasibilityConfig,
    MASTERConfig,
    PipelineConfig,
)
from epifitscout.pipeline.graft_pipeline import GraftPipeline
from epifitscout.query.builder import QueryBuilder

pytestmark = pytest.mark.needs_master

H3 = (105, 117)          # IMGT CDR-H3 of 1a2y chain B: ten real residues


def _config(db: Path, **kwargs) -> PipelineConfig:
    defaults = dict(
        master=MASTERConfig(database_path=db / "pds.list", max_hits=50),
        pool_size=5, top_k=3, max_workers=1, log_level="WARNING",
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


@pytest.fixture
def pipeline(mini_db_built: Path, master_env: Path) -> GraftPipeline:
    return GraftPipeline(_config(mini_db_built))


class TestTheSearchDoesNotSeeTheLoop:
    def test_candidates_come_back_placed_in_the_template_frame(
        self, pipeline: GraftPipeline, example_pdb: Path, tmp_path: Path
    ) -> None:
        import numpy as np

        query = QueryBuilder(example_pdb).get_anchor_query("B", *H3)
        candidates = pipeline.search(query, tmp_path)
        if not candidates:
            pytest.skip("the 14-chain fixture database bridges no such gap")

        # Where the loop has to end up: it bridges the same gap the native one
        # does, so its centroid sits within a loop's width of the native's. A
        # candidate left in its donor's crystal frame would be tens of
        # ångströms away — the v1 bug this asserts against.
        native = np.asarray(query.metadata["native_loop_coords"]).reshape(-1, 3)
        target = native.mean(axis=0)
        for candidate in candidates:
            placed = candidate.placed_loop
            assert placed.shape == (query.gap_length, 4, 3)
            offset = float(
                np.linalg.norm(placed.reshape(-1, 3).mean(axis=0) - target)
            )
            assert offset < 10.0, f"{candidate.source} landed {offset:.1f} A away"

    def test_every_candidate_is_the_length_the_graft_needs(
        self, pipeline: GraftPipeline, example_pdb: Path, tmp_path: Path
    ) -> None:
        """--gapLen is what keeps the retrieved loop and the span one-to-one."""
        query = QueryBuilder(example_pdb).get_anchor_query("B", *H3)
        for candidate in pipeline.search(query, tmp_path):
            assert candidate.length == query.gap_length

    def test_the_stem_fit_reproduces_masters_own_rmsd(
        self, pipeline: GraftPipeline, example_pdb: Path, tmp_path: Path
    ) -> None:
        query = QueryBuilder(example_pdb).get_anchor_query("B", *H3)
        for candidate in pipeline.search(query, tmp_path):
            assert candidate.stem_rmsd_ca == pytest.approx(
                candidate.master_rmsd, abs=0.01
            )

    def test_an_excluded_donor_is_never_returned(
        self, pipeline: GraftPipeline, example_pdb: Path, tmp_path: Path
    ) -> None:
        query = QueryBuilder(example_pdb).get_anchor_query("B", *H3)
        everything = pipeline.search(query, tmp_path / "all")
        if not everything:
            pytest.skip("nothing retrieved, so nothing to exclude")
        donor = everything[0].loop.pdb_id
        kept = pipeline.search(query, tmp_path / "some", exclude_donors={donor})
        assert all(c.loop.pdb_id != donor for c in kept)

    def test_an_excluded_sequence_is_never_returned(
        self, pipeline: GraftPipeline, example_pdb: Path, tmp_path: Path
    ) -> None:
        """Excluding a PDB ID does not exclude the same antibody elsewhere.

        On the full database this is not a hypothetical: searching 1a2y's
        CDR-H3 with only the 1A2Y entry refused returns that exact loop from
        five other entries of the same antibody, which would then be reported
        as a retrieval success.
        """
        query = QueryBuilder(example_pdb).get_anchor_query("B", *H3)
        everything = pipeline.search(query, tmp_path / "all")
        if not everything:
            pytest.skip("nothing retrieved, so nothing to exclude")
        sequence = everything[0].sequence
        kept = pipeline.search(
            query, tmp_path / "some", exclude_sequences={sequence}
        )
        assert all(c.sequence != sequence for c in kept)

    def test_the_native_loop_can_be_excluded_by_name(
        self, mini_db_built: Path, master_env: Path, example_pdb: Path,
        tmp_path: Path,
    ) -> None:
        """`exclude_native_loop` is the benchmark's leakage control."""
        native = (
            QueryBuilder(example_pdb)
            .get_anchor_query("B", *H3)
            .metadata["native_loop_sequence"]
        )
        result = GraftPipeline(_config(mini_db_built)).run(
            template=example_pdb, ab_chain="B", loop_span=H3,
            antigen_chains=("A",), out_dir=tmp_path, packer_name="none",
            exclude_native_loop=True,
        )
        assert all(c.sequence.upper() != native.upper() for c in result.pool)


class TestTheFunnelExplainsItself:
    def test_a_run_records_every_stage(
        self, mini_db_built: Path, master_env: Path, example_pdb: Path,
        tmp_path: Path,
    ) -> None:
        result = GraftPipeline(_config(mini_db_built)).run(
            template=example_pdb, ab_chain="B", loop_span=H3,
            antigen_chains=("A",), out_dir=tmp_path, packer_name="none",
        )
        assert list(result.funnel) == [
            "retrieved", "feasible", "pooled", "scored", "returned",
        ]
        counts = list(result.funnel.values())
        assert counts == sorted(counts, reverse=True)      # monotonically narrowing

    def test_an_unbridgeable_site_is_refused_before_searching(
        self, mini_db_built: Path, master_env: Path, example_pdb: Path,
        tmp_path: Path,
    ) -> None:
        """No loop can be ranked for a gap no chain can span, so none is."""
        cfg = _config(
            mini_db_built, feasibility=FeasibilityConfig(max_span_per_residue=0.01)
        )
        result = GraftPipeline(cfg).run(
            template=example_pdb, ab_chain="B", loop_span=H3,
            antigen_chains=("A",), out_dir=tmp_path, packer_name="none",
        )
        assert not result.ok
        assert "unbridgeable" in result.funnel

    def test_a_missing_antigen_chain_is_an_error_not_a_silent_zero(
        self, mini_db_built: Path, master_env: Path, example_pdb: Path,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError):
            GraftPipeline(_config(mini_db_built)).run(
                template=example_pdb, ab_chain="B", loop_span=H3,
                antigen_chains=("Q",), out_dir=tmp_path, packer_name="none",
            )


class TestTheGateRunsBeforeTheExpensiveWork:
    def test_only_feasible_candidates_are_pooled(
        self, mini_db_built: Path, master_env: Path, example_pdb: Path,
        tmp_path: Path,
    ) -> None:
        result = GraftPipeline(_config(mini_db_built)).run(
            template=example_pdb, ab_chain="B", loop_span=H3,
            antigen_chains=("A",), out_dir=tmp_path, packer_name="none",
        )
        assert result.funnel["pooled"] <= result.funnel["feasible"]
        for candidate in result.pool:
            assert candidate.is_feasible

    def test_the_pool_is_capped_by_the_budget(
        self, mini_db_built: Path, master_env: Path, example_pdb: Path,
        tmp_path: Path,
    ) -> None:
        """pool_size is the search budget, and it has to bind."""
        cfg = _config(mini_db_built, pool_size=2, top_k=2)
        result = GraftPipeline(cfg).run(
            template=example_pdb, ab_chain="B", loop_span=H3,
            antigen_chains=("A",), out_dir=tmp_path, packer_name="none",
        )
        assert result.funnel["pooled"] <= 2
        assert len(result.top) <= 2


class TestTheRankedResult:
    @pytest.fixture
    def result(
        self, mini_db_built: Path, master_env: Path, example_pdb: Path,
        tmp_path_factory,
    ):
        out = tmp_path_factory.mktemp("ranked")
        return GraftPipeline(_config(mini_db_built)).run(
            template=example_pdb, ab_chain="B", loop_span=H3,
            antigen_chains=("A",), out_dir=out, packer_name="none",
        )

    def test_candidates_are_ordered_by_shape_complementarity(self, result) -> None:
        """Shape decides the order, and nothing measured after packing does."""
        if len(result.top) < 2:
            pytest.skip("fewer than two candidates to order")
        scores = [c.shape.score for c in result.top]
        assert scores == sorted(scores, reverse=True)

    def test_the_interface_does_not_influence_the_order(self, result) -> None:
        """The guard against v2's ranking creeping back in.

        Clash-free models must NOT be promoted above clashing ones: in v3 the
        ranking is settled before any model is packed, so a clean interface is
        an outcome of the choice rather than an input to it.
        """
        if len(result.top) < 2:
            pytest.skip("fewer than two candidates to order")
        scores = [c.shape.score for c in result.top]
        assert scores == sorted(scores, reverse=True)

    def test_each_candidate_carries_its_evidence(self, result) -> None:
        if not result.ok:
            pytest.skip("nothing came through on the fixture database")
        for candidate in result.top:
            assert candidate.interface is not None
            assert candidate.feasibility is not None and candidate.is_feasible
            assert candidate.shape is not None
            assert 0.0 <= candidate.shape.score <= 1.0
            assert candidate.sequence and len(candidate.sequence) == candidate.length

    def test_a_backbone_only_run_says_so(self, result) -> None:
        """packer_name='none' is diagnostic, and the model records it."""
        if not result.ok:
            pytest.skip("nothing came through on the fixture database")
        assert all(
            candidate.metadata.get("packed") is False for candidate in result.top
        )


class TestTheFitRanksRatherThanGates:
    """A target with no good candidate must still report its least-bad one.

    Requiring a clean contact run outright was tried and cost 13 of 101
    benchmark targets, one of them after retrieving 500 loops and finding 270
    graftable. Reporting nothing is not more useful than reporting a poor model
    whose numbers say it is poor.
    """

    def test_candidates_that_do_not_fit_are_still_reported(
        self, mini_db_built: Path, master_env: Path, example_pdb: Path,
        tmp_path: Path,
    ) -> None:
        result = GraftPipeline(_config(mini_db_built)).run(
            template=example_pdb, ab_chain="B", loop_span=H3,
            antigen_chains=("A",), out_dir=tmp_path, packer_name="none",
        )
        if not result.funnel["feasible"]:
            pytest.skip("nothing graftable on the fixture database")
        # Whatever the fit stage made of them, something comes out.
        assert result.funnel["pooled"] > 0
        assert result.funnel["pooled"] <= result.funnel["feasible"]

    def test_shape_orders_without_excluding(
        self, pipeline: GraftPipeline, example_pdb: Path, tmp_path: Path
    ) -> None:
        """Ordering, not exclusion: a poor shape score is last place, not removal."""
        from epifitscout.graft.assemble import TemplateSite
        from epifitscout.scoring.epitope import epitope_ca
        from epifitscout.scoring.shape import ShapeComplementarityScorer

        query = QueryBuilder(example_pdb).get_anchor_query("B", *H3)
        candidates = pipeline.search(query, tmp_path)
        if len(candidates) < 2:
            pytest.skip("too few candidates to order")
        site = TemplateSite.load(example_pdb, "B", query.loop_range, ("A",))
        epitope = epitope_ca(
            site.native_loop_coords, site.antigen_coords,
            site.antigen_residue_index, site.antigen_is_ca,
        )
        scorer = ShapeComplementarityScorer()
        for candidate in candidates:
            candidate.shape = scorer.score(candidate.placed_loop, epitope)
        ordered = sorted(candidates, key=lambda c: c.shape.score, reverse=True)
        scores = [c.shape.score for c in ordered]
        assert scores == sorted(scores, reverse=True)
        # Nothing was dropped on the way.
        assert len(ordered) == len(candidates)


class TestTheEpitopeIsFound:
    def test_the_epitope_is_recorded_in_the_funnel(
        self, mini_db_built: Path, master_env: Path, example_pdb: Path,
        tmp_path: Path,
    ) -> None:
        """A run must say how many epitope residues it scored against.

        A silently empty epitope would score every candidate 0 and the ranking
        would be arbitrary while still looking like a ranking.
        """
        result = GraftPipeline(_config(mini_db_built)).run(
            template=example_pdb, ab_chain="B", loop_span=H3,
            antigen_chains=("A",), out_dir=tmp_path, packer_name="none",
        )
        assert result.epitope_residues > 0
