"""HitRanker: combines shape complementarity and RMSD into a final ranked list."""

from __future__ import annotations

import logging

from epifitscout.config.schema import RankingConfig
from epifitscout.types.fragment import Fragment, ScoredHit
from epifitscout.types.superposition import Superposition

logger = logging.getLogger(__name__)


class HitRanker:
    """Ranks CDR hits by combining shape complementarity score and RMSD.

    Scoring formula::

        final = weight_rmsd * (1 / (1 + rmsd_cdr))
              + weight_shape * complementarity_score

    ``1 / (1 + rmsd)`` maps RMSD ∈ [0, ∞) → [0, 1]; lower RMSD → higher score.
    ``complementarity_score`` is already in [0, 1] from the shape scorer.

    Args:
        cfg: RankingConfig with ``weight_rmsd`` and ``weight_shape``.
    """

    def __init__(self, cfg: RankingConfig) -> None:
        self._cfg = cfg

    def rank(
        self,
        hits: list[tuple[Fragment, Superposition, float]],
        rmsd_scores: list[float],
    ) -> list[ScoredHit]:
        """Build ScoredHit objects and sort by final_score descending.

        Args:
            hits: List of (Fragment, Superposition, complementarity_score) from
                  the scoring step — one tuple per hit.
            rmsd_scores: MASTER RMSD values, aligned 1-to-1 with ``hits``.

        Returns:
            List of ScoredHit sorted by final_score descending.

        Raises:
            ValueError: if ``hits`` and ``rmsd_scores`` have different lengths.
        """
        if len(hits) != len(rmsd_scores):
            raise ValueError(
                f"hits ({len(hits)}) and rmsd_scores ({len(rmsd_scores)}) "
                "must have the same length"
            )

        if not hits:
            return []

        cfg = self._cfg
        scored: list[ScoredHit] = []

        for (frag, sup, comp_score), rmsd in zip(hits, rmsd_scores):
            rmsd_component = 1.0 / (1.0 + rmsd)
            final = cfg.weight_rmsd * rmsd_component + cfg.weight_shape * comp_score
            scored.append(
                ScoredHit(
                    fragment=frag,
                    superposition=sup,
                    rmsd_cdr=rmsd,
                    complementarity_score=comp_score,
                    final_score=float(final),
                )
            )

        scored.sort(key=lambda h: h.final_score, reverse=True)
        logger.info("Ranked %d hits; top score=%.4f", len(scored), scored[0].final_score)
        return scored
