"""FragmentSearchPipeline: MASTER chain-DB search → shape complementarity → rank."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from epifitscout.config.schema import PipelineConfig
from epifitscout.master.parser import parse_match_file
from epifitscout.master.runner import MASTERRunner
from epifitscout.ranking.ranker import HitRanker
from epifitscout.scoring import ShapeComplementarityScorer
from epifitscout.scoring.local_fit import LocalFitScorer
from epifitscout.types.epitope import EpitopeSurface
from epifitscout.types.fragment import Fragment, ScoredHit
from epifitscout.types.superposition import Superposition

logger = logging.getLogger(__name__)


class FragmentSearchPipeline:
    """3-step CDR fragment search pipeline.

    Steps:
      1. MASTER search on SAbDab CDR database → CDR-similar hits.
      2. Shape complementarity scoring (in the MASTER superposition frame).
      3. Ranking by weighted combination of shape score and RMSD.

    When ``cfg.max_workers > 1``, MASTER searches the database in parallel
    across shards for a single query — ~N× speedup over sequential search.

    Usage (full pipeline)::

        cfg = PipelineConfig(max_workers=6)
        pipeline = FragmentSearchPipeline(cfg)
        hits = pipeline.search(query_cdr, query_epitope)

    Usage (multiple queries in parallel)::

        pairs = [(cdr1, epi1), (cdr2, epi2), (cdr3, epi3)]
        all_hits = pipeline.search_many(pairs)
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))
        self._cfg = cfg
        self._runner = MASTERRunner(cfg.master, max_workers=cfg.max_workers)
        self._scorer = ShapeComplementarityScorer(
            weight_depth=cfg.scoring.weight_depth,
            weight_tau=cfg.scoring.weight_tau,
        )
        self._local_scorer = LocalFitScorer()
        self._ranker = HitRanker(cfg.ranking)

    # ------------------------------------------------------------------ #
    # Public API — full pipeline
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_cdr: Fragment,
        query_epitope: Fragment | EpitopeSurface,
        output_dir: Path | None = None,
        antibody_body: np.ndarray | None = None,
        scorer: str = "local",
    ) -> list[ScoredHit]:
        """Run the complete 3-step pipeline for a single query pair.

        When ``cfg.max_workers > 1``, the MASTER search is sharded across
        workers automatically — no change needed at the call site.

        Args:
            query_cdr: Query CDR fragment (backbone coords).
            query_epitope: The epitope. Pass an
                :class:`~epifitscout.types.epitope.EpitopeSurface` to use the
                local shape-fit scorer, which reports where along the epitope
                each loop fits and over what length; pass a ``Fragment`` to use
                the older whole-length profile scorer.
            output_dir: Directory for MASTER intermediate files. Auto-temp if None.
            antibody_body: Optional template antibody atoms, used by the
                local-fit scorer to orient the loop's outward normals.
            scorer: ``"local"`` (default) or ``"profile"``; see
                :meth:`score_complementarity`.

        Returns:
            Ranked list of ScoredHit objects.
        """
        if output_dir is not None:
            return self._run(query_cdr, query_epitope, output_dir, antibody_body, scorer)

        with tempfile.TemporaryDirectory(prefix="epifitscout_") as tmp:
            return self._run(query_cdr, query_epitope, Path(tmp), antibody_body, scorer)

    def search_many(
        self,
        queries: list[tuple[Fragment, Fragment | EpitopeSurface]],
        max_workers: int | None = None,
    ) -> list[list[ScoredHit]]:
        """Run the pipeline in parallel for multiple (cdr, epitope) pairs.

        Note: this parallelises across query pairs — each pair still uses
        the sharded MASTER runner internally if ``cfg.max_workers > 1``.
        For a single query pair, use ``search()`` with sharding instead.

        Args:
            queries: List of (query_cdr, query_epitope) pairs.
            max_workers: Parallel pipelines. Defaults to cfg.max_workers.

        Returns:
            Ranked ScoredHit lists in the same order as input queries.
        """
        workers = max_workers if max_workers is not None else self._cfg.max_workers

        def _run_one(
            args: tuple[int, Fragment, Fragment | EpitopeSurface]
        ) -> tuple[int, list[ScoredHit]]:
            idx, cdr, epitope = args
            with tempfile.TemporaryDirectory(prefix=f"epifitscout_{idx}_") as tmp:
                return idx, self._run(cdr, epitope, Path(tmp))

        results: list[list[ScoredHit]] = [[] for _ in queries]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_one, (i, cdr, epi)): i
                for i, (cdr, epi) in enumerate(queries)
            }
            for future in as_completed(futures):
                idx, hits = future.result()
                results[idx] = hits
                logger.info(
                    "Query %d/%d done: %d hits", idx + 1, len(queries), len(hits)
                )

        return results

    # ------------------------------------------------------------------ #
    # Public API — individual steps
    # ------------------------------------------------------------------ #

    def run_master(
        self,
        query_cdr: Fragment,
        output_dir: Path,
    ) -> list[tuple[Fragment, Superposition, float]]:
        """Step 1: run MASTER and return (fragment, superposition, rmsd) tuples.

        Automatically uses sharded parallel search when cfg.max_workers > 1.

        Args:
            query_cdr: Query CDR fragment.
            output_dir: Directory to write MASTER files.

        Returns:
            List of (Fragment, Superposition, rmsd) sorted by RMSD ascending.
        """
        if self._cfg.max_workers > 1:
            match_path = self._runner.run_sharded(query_cdr, output_dir)
        else:
            match_path = self._runner.run(query_cdr, output_dir)

        result: list[tuple[Fragment, Superposition, float]] = parse_match_file(
            match_path,
            query_cdr,
            rmsd_threshold=self._cfg.master.rmsd_threshold,
            max_hits=self._cfg.master.max_hits,
        )
        logger.info("Step 1 (MASTER): %d hits", len(result))
        return result

    def score_complementarity(
        self,
        hits: list[tuple[Fragment, Superposition, float]],
        query_epitope: Fragment | EpitopeSurface,
        antibody_body: np.ndarray | None = None,
        scorer: str = "local",
    ) -> list[tuple[Fragment, Superposition, float, Any]]:
        """Step 2: filter and score each hit against the epitope.

        Each hit is scored **after** its superposition is applied, so the loop
        sits at the CDR site in the antibody-antigen frame.

        Args:
            hits: Output of :meth:`run_master`.
            query_epitope: The epitope. An
                :class:`~epifitscout.types.epitope.EpitopeSurface` from
                :meth:`~epifitscout.query.builder.QueryBuilder.get_epitope_surface`
                is strongly preferred; a backbone ``Fragment`` is accepted but
                carries no side chains, so distances are approximate.
            antibody_body: Optional template antibody atoms, used to orient the
                loop's outward normals.
            scorer: ``"local"`` (default) runs
                :class:`~epifitscout.scoring.local_fit.LocalFitScorer`, which
                slides the loop along the epitope, rejects candidates that
                collide with the antigen, and reports where and how far each
                one matches. ``"profile"`` runs the superseded
                :class:`~epifitscout.scoring.shape.ShapeComplementarityScorer`;
                see its module docstring before using it.

        Returns:
            ``(fragment, superposition, score, detail)`` per hit, where *detail*
            is a :class:`~epifitscout.scoring.local_fit.LocalFit` under the
            local scorer and None under the profile scorer.

        Raises:
            ValueError: If *scorer* is not one of the two names.
        """
        if scorer not in ("local", "profile"):
            raise ValueError(f"scorer must be 'local' or 'profile', got {scorer!r}")

        result: list[tuple[Fragment, Superposition, float, Any]] = []

        if scorer == "profile":
            epi_coords = (
                query_epitope.atom_coords
                if isinstance(query_epitope, EpitopeSurface)
                else query_epitope.coords
            )
            for frag, sup, _ in hits:
                result.append(
                    (frag, sup, self._scorer.score(sup.apply(frag.coords), epi_coords), None)
                )
            logger.info("Step 2 (profile scorer, superseded): %d hits", len(result))
            return result

        surface = query_epitope
        if not isinstance(surface, EpitopeSurface):
            logger.warning(
                "Epitope given as a backbone Fragment: side chains are missing, so "
                "contact distances are approximate. Prefer "
                "QueryBuilder.get_epitope_surface()."
            )
            surface = EpitopeSurface.from_fragment(query_epitope)

        for frag, sup, _ in hits:
            fit = self._local_scorer.score(
                sup.apply(frag.coords), surface,
                residue_numbers=frag.residue_range, antibody_body=antibody_body,
            )
            result.append((frag, sup, fit.score, fit))

        n_fit = sum(1 for r in result if r[3] is not None and r[3].fits)
        n_clash = sum(1 for r in result if r[3] is not None and r[3].n_clash)
        logger.info(
            "Step 2 (local fit): %d hits, %d make clean contact, %d rejected for "
            "colliding with the antigen", len(result), n_fit, n_clash,
        )
        return result

    def rank(
        self,
        comp_hits: Sequence[
            tuple[Fragment, Superposition, float]
            | tuple[Fragment, Superposition, float, Any]
        ],
        rmsd_scores: Sequence[float],
    ) -> list[ScoredHit]:
        """Step 3: rank hits by shape score + RMSD."""
        return self._ranker.rank(comp_hits, rmsd_scores)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _run(
        self,
        query_cdr: Fragment,
        query_epitope: Fragment | EpitopeSurface,
        output_dir: Path,
        antibody_body: np.ndarray | None = None,
        scorer: str = "local",
    ) -> list[ScoredHit]:
        master_hits = self.run_master(query_cdr, output_dir)
        rmsd_scores = [r for _, _, r in master_hits]
        comp_hits = self.score_complementarity(
            master_hits, query_epitope, antibody_body=antibody_body, scorer=scorer
        )
        return self._ranker.rank(comp_hits, rmsd_scores)
