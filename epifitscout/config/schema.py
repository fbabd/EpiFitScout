"""Frozen configuration dataclasses for all pipeline stages.

Plain Python dataclasses with no third-party dependency at import time; the
YAML defaults shipped in ``epifitscout/conf/config.yaml`` are read into these
by :func:`epifitscout.config_from_yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MASTERConfig:
    """Configuration for the MASTER search step.

    ``binary_path`` and ``database_path`` default to ``None``, meaning "resolve
    at use time" via :mod:`epifitscout.paths`. They are only set explicitly to
    override that resolution — hard-coding relative defaults here would silently
    depend on the current working directory.
    """

    binary_path: Path | None = None
    database_path: Path | None = None
    rmsd_threshold: float = 2.0
    max_hits: int = 500
    timeout_seconds: int = 420
    n_threads: int = 0



@dataclass(frozen=True)
class ScoringConfig:
    """Weights for the shape complementarity score.

    S_shape = clip((weight_depth * Sd + weight_tau * S_tau + 1) / 2, 0, 1)

    Both weights should be non-negative and sum to 1.
    """

    weight_depth: float = 0.7   # depth anti-correlation component
    weight_tau: float = 0.3     # backbone torsion co-correlation component


@dataclass(frozen=True)
class RankingConfig:
    """Configuration for final hit ranking step.

    S_final = weight_rmsd * (1 / (1 + rmsd)) + weight_shape * S_shape

    Both components are in [0, 1]; weights should sum to 1.
    """

    weight_rmsd: float = 0.4
    weight_shape: float = 0.6


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration."""

    master: MASTERConfig = field(default_factory=MASTERConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    log_level: str = "INFO"
    max_workers: int = 6   # 1 = sequential; >1 = parallel search_many()

