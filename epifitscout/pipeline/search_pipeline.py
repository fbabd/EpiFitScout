"""FragmentSearchPipeline: MASTER chain-DB search → shape complementarity → rank.

External systems import only this class (or its individual step methods).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from epifitscout.config.schema import PipelineConfig
from epifitscout.master.parser import parse_match_file
from epifitscout.master.runner import MASTERRunner
from epifitscout.ranking.ranker import HitRanker
from epifitscout.scoring import ShapeComplementarityScorer
from epifitscout.types.fragment import Fragment, ScoredHit
from epifitscout.types.superposition import Superposition

logger = logging.getLogger(__name__)


class FragmentSearchPipeline:
    """3-step CDR fragment search pipeline.

    Steps:
      1. MASTER search on SAbDab CDR database → CDR-similar hits.
      2. Shape complementarity scoring (in the MASTER superposition frame).
      3. Ranking by weighted combination of shape score and RMSD.

    Usage (full pipeline)::

        cfg = PipelineConfig(...)
        pipeline = FragmentSearchPipeline(cfg)
        hits = pipeline.search(query_cdr, query_epitope)

    Usage (step-by-step)::

        raw_hits  = pipeline.run_master(query_cdr, output_dir=Path("/tmp/run"))
        comp_hits = pipeline.score_complementarity(raw_hits, query_epitope)
        ranked    = pipeline.rank(comp_hits, [rmsd for _, _, rmsd in raw_hits])
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))
        self._cfg = cfg
        self._runner = MASTERRunner(cfg.master)
        self._scorer = ShapeComplementarityScorer(
            weight_depth=cfg.scoring.weight_depth,
            weight_tau=cfg.scoring.weight_tau,
        )
        self._ranker = HitRanker(cfg.ranking)

    # ------------------------------------------------------------------ #
    # Public API — full pipeline
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_cdr: Fragment,
        query_epitope: Fragment,
        output_dir: Path | None = None,
    ) -> list[ScoredHit]:
        """Run the complete 3-step pipeline.

        Args:
            query_cdr: Query CDR fragment (backbone coords).
            query_epitope: Query epitope fragment (backbone coords).
            output_dir: Directory for MASTER intermediate files. If None, a
                temporary directory is created and cleaned up automatically.

        Returns:
            Ranked list of ScoredHit objects.
        """
        if output_dir is not None:
            return self._run(query_cdr, query_epitope, output_dir)

        with tempfile.TemporaryDirectory(prefix="epifitscout_") as tmp:
            return self._run(query_cdr, query_epitope, Path(tmp))

    # ------------------------------------------------------------------ #
    # Public API — individual steps
    # ------------------------------------------------------------------ #

    def run_master(
        self,
        query_cdr: Fragment,
        output_dir: Path,
    ) -> list[tuple[Fragment, Superposition, float]]:
        """Step 1: run MASTER and return (fragment, superposition, rmsd) tuples.

        Args:
            query_cdr: Query CDR fragment.
            output_dir: Directory to write MASTER files.

        Returns:
            List of (Fragment, Superposition, rmsd) sorted by RMSD ascending.
        """
        match_path = self._runner.run(query_cdr, output_dir)
        result: list[tuple[Fragment, Superposition, float]] = parse_match_file(
            match_path,
            rmsd_threshold=self._cfg.master.rmsd_threshold,
            max_hits=self._cfg.master.max_hits,
        )
        logger.info("Step 1 (MASTER): %d hits", len(result))
        return result

    def score_complementarity(
        self,
        hits: list[tuple[Fragment, Superposition, float]],
        query_epitope: Fragment,
    ) -> list[tuple[Fragment, Superposition, float]]:
        """Step 2: score each hit for shape complementarity.

        Applies the MASTER superposition to each hit CDR and scores how well
        it complements the query epitope in the binding-pose frame.

        Args:
            hits: Output of run_master() — (Fragment, Superposition, rmsd).
            query_epitope: Query epitope fragment.

        Returns:
            List of (Fragment, Superposition, complementarity_score).
        """
        epi_coords = query_epitope.coords
        result: list[tuple[Fragment, Superposition, float]] = []

        for frag, sup, _ in hits:
            aligned = sup.apply(frag.coords)
            score = self._scorer.score(aligned, epi_coords)
            result.append((frag, sup, score))

        logger.info("Step 2 (scoring): %d hits scored", len(result))
        return result

    def rank(
        self,
        comp_hits: list[tuple[Fragment, Superposition, float]],
        rmsd_scores: list[float],
    ) -> list[ScoredHit]:
        """Step 3: rank hits by shape score + RMSD.

        Args:
            comp_hits: Output of score_complementarity().
            rmsd_scores: MASTER RMSD values aligned 1-to-1 with comp_hits.

        Returns:
            Ranked list of ScoredHit objects.
        """
        return self._ranker.rank(comp_hits, rmsd_scores)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _run(
        self,
        query_cdr: Fragment,
        query_epitope: Fragment,
        output_dir: Path,
    ) -> list[ScoredHit]:
        master_hits = self.run_master(query_cdr, output_dir)
        rmsd_scores = [r for _, _, r in master_hits]
        comp_hits = self.score_complementarity(master_hits, query_epitope)
        return self._ranker.rank(comp_hits, rmsd_scores)
