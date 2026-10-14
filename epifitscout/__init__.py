"""epifitscout — Protein structural fragment search with epitope complementarity scoring.

Quick start::

    import epifitscout

    qb = epifitscout.QueryBuilder("my_complex.pdb")
    qb.describe()                                 # show chains + residue numbers

    query_cdr     = qb.get_fragment("H", 100, 112)
    query_epitope = qb.get_fragment("A",  50,  65)

    hits = epifitscout.search(query_cdr, query_epitope)
    for h in hits[:5]:
        print(h)

Two things are pasted into the repository before searching: the MASTER
binaries (``MASTER/bin/``) and the chain database (``data/sabdab_chains.db/``).
Run ``python check_setup.py`` to confirm both are in place.
"""

from __future__ import annotations

__version__ = "0.1.0"

from pathlib import Path
from typing import Any

import numpy as np

from epifitscout import paths
from epifitscout.config.schema import MASTERConfig, PipelineConfig, RankingConfig, ScoringConfig
from epifitscout.paths import (
    MissingBinaryError,
    MissingDatabaseError,
    SetupError,
)
from epifitscout.pipeline.search_pipeline import FragmentSearchPipeline
from epifitscout.query.builder import QueryBuilder, load_fragment_from_pdb
from epifitscout.types.epitope import EpitopeSurface
from epifitscout.types.fragment import Fragment, ScoredHit


def config_from_yaml(yaml_path: Path | str | None = None) -> PipelineConfig:
    """Load a PipelineConfig from a plain YAML file (no Hydra required).

    Expected YAML structure (all sections optional — missing keys use defaults)::

        master:
          rmsd_threshold: 1.5
          max_hits: 500
          timeout_seconds: 300
        scoring:
          weight_depth: 0.7
          weight_tau: 0.3
        ranking:
          weight_rmsd: 0.4
          weight_shape: 0.6
        max_workers: 6

    Args:
        yaml_path: Path to a YAML config file. Defaults to the config shipped
            at ``epifitscout/conf/config.yaml``.

    Returns:
        PipelineConfig with values from the file, defaults for missing keys.

    Raises:
        SetupError: If the default config is missing, which means a damaged
            checkout. This is never silently ignored, because falling back to
            built-in defaults changes ``rmsd_threshold`` and therefore the
            results.
    """
    import yaml

    path = Path(yaml_path) if yaml_path is not None else paths.default_config_path()

    with open(path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    master_raw = raw.get("master", {})
    scoring_raw = raw.get("scoring", {})
    ranking_raw = raw.get("ranking", {})

    return PipelineConfig(
        master=MASTERConfig(
            rmsd_threshold=master_raw.get("rmsd_threshold", 1.5),
            max_hits=master_raw.get("max_hits", 500),
            timeout_seconds=master_raw.get("timeout_seconds", 300),
            n_threads=master_raw.get("n_threads", 0),
        ),
        scoring=ScoringConfig(
            weight_depth=scoring_raw.get("weight_depth", 0.7),
            weight_tau=scoring_raw.get("weight_tau", 0.3),
        ),
        ranking=RankingConfig(
            weight_rmsd=ranking_raw.get("weight_rmsd", 0.4),
            weight_shape=ranking_raw.get("weight_shape", 0.6),
        ),
        log_level=raw.get("log_level", "INFO"),
        max_workers=raw.get("max_workers", 6),
    )


def _build_config(
    *,
    rmsd_threshold: float | None,
    max_hits: int | None,
    weight_depth: float | None,
    weight_tau: float | None,
    weight_rmsd: float | None,
    weight_shape: float | None,
    binary_path: Path | None,
    database_path: Path | None,
    config_path: Path | None,
    log_level: str | None,
    max_workers: int | None,
) -> PipelineConfig:
    """Build and validate a PipelineConfig, applying per-call overrides.

    Loads base values from ``epifitscout/conf/config.yaml``, then applies
    explicitly provided overrides and resolves the MASTER binary and chain
    database.

    Raises:
        SetupError: If the config, MASTER binary or database is missing. The
            message says which folder to paste what into.
    """
    base = config_from_yaml(config_path)

    bin_path = paths.find_master_binary(binary_path)
    db_path = paths.find_pds_list(database_path)

    return PipelineConfig(
        master=MASTERConfig(
            binary_path=bin_path,
            database_path=db_path,
            rmsd_threshold=(
                rmsd_threshold if rmsd_threshold is not None else base.master.rmsd_threshold
            ),
            max_hits=max_hits if max_hits is not None else base.master.max_hits,
            timeout_seconds=base.master.timeout_seconds,
            n_threads=base.master.n_threads,
        ),
        scoring=ScoringConfig(
            weight_depth=weight_depth if weight_depth is not None else base.scoring.weight_depth,
            weight_tau=weight_tau if weight_tau is not None else base.scoring.weight_tau,
        ),
        ranking=RankingConfig(
            weight_rmsd=weight_rmsd if weight_rmsd is not None else base.ranking.weight_rmsd,
            weight_shape=weight_shape if weight_shape is not None else base.ranking.weight_shape,
        ),
        log_level=log_level if log_level is not None else base.log_level,
        max_workers=max_workers if max_workers is not None else base.max_workers,
    )


def search(
    query_cdr: Fragment,
    query_epitope: Fragment | EpitopeSurface,
    *,
    rmsd_threshold: float | None = None,
    max_hits: int | None = None,
    weight_depth: float | None = None,
    weight_tau: float | None = None,
    weight_rmsd: float | None = None,
    weight_shape: float | None = None,
    binary_path: Path | None = None,
    database_path: Path | None = None,
    config_path: Path | None = None,
    log_level: str | None = None,
    max_workers: int | None = None,
    antibody_body: np.ndarray | None = None,
    scorer: str = "local",
) -> list[ScoredHit]:
    """Search for CDR-like fragments that are complementary to a target epitope.

    Runs the full 3-step pipeline: MASTER structural search → shape
    complementarity scoring → weighted ranking.

    Base configuration comes from ``epifitscout/conf/config.yaml``. Any keyword
    argument passed explicitly overrides the corresponding value.

    When ``max_workers > 1`` (from config or explicit override), MASTER searches
    the database in parallel across shards for a ~N× speedup.

    Args:
        query_cdr: Query CDR fragment (backbone coords).
        query_epitope: The epitope. Pass an
            :class:`~epifitscout.types.epitope.EpitopeSurface` from
            ``QueryBuilder.get_epitope_surface(...)``, which reads every atom
            including side chains. A backbone ``Fragment`` still works but makes
            contact distances approximate.
        rmsd_threshold: Maximum backbone RMSD (Å). Overrides config if given.
        max_hits: Maximum MASTER hits. Overrides config if given.
        weight_depth: Depth anti-correlation weight in S_shape. Overrides config.
        weight_tau: Torsion co-correlation weight in S_shape. Overrides config.
        weight_rmsd: RMSD weight in S_final. Overrides config if given.
        weight_shape: Shape weight in S_final. Overrides config if given.
        binary_path: Path to the MASTER binary. Auto-resolved when omitted.
        database_path: Path to ``pds.list``. Auto-resolved when omitted.
        config_path: Alternative YAML config file.
        log_level: Python logging level string. Overrides config if given.
        max_workers: Parallel MASTER shard processes. Overrides config if given.
        antibody_body: Optional template antibody atoms, used to orient the
            loop's outward normals.
        scorer: ``"local"`` (default) slides each grafted loop along the
            epitope, rejects candidates that collide with the antigen, and
            reports where and how far each one matches. ``"profile"`` selects
            the superseded whole-length correlation scorer.

    Returns:
        Ranked list of :class:`ScoredHit` objects, best first.

    Raises:
        MissingBinaryError: If the MASTER executables cannot be found.
        MissingDatabaseError: If the chain database is absent or incomplete.

    Example::

        hits = epifitscout.search(query_cdr, query_epitope)

        # Override one weight, rest from the config file
        hits = epifitscout.search(query_cdr, query_epitope, weight_depth=0.9)
    """
    cfg = _build_config(
        rmsd_threshold=rmsd_threshold,
        max_hits=max_hits,
        weight_depth=weight_depth,
        weight_tau=weight_tau,
        weight_rmsd=weight_rmsd,
        weight_shape=weight_shape,
        binary_path=binary_path,
        database_path=database_path,
        config_path=config_path,
        log_level=log_level,
        max_workers=max_workers,
    )
    pipeline = FragmentSearchPipeline(cfg)
    return pipeline.search(
        query_cdr, query_epitope, antibody_body=antibody_body, scorer=scorer
    )


__all__ = [
    # Top-level convenience API
    "search",
    # Query helpers
    "QueryBuilder",
    "load_fragment_from_pdb",
    # Data types
    "Fragment",
    "ScoredHit",
    "EpitopeSurface",
    # Config (for advanced / step-by-step usage)
    "config_from_yaml",
    "FragmentSearchPipeline",
    "PipelineConfig",
    "MASTERConfig",
    "ScoringConfig",
    "RankingConfig",
    # Resource resolution
    "paths",
    "SetupError",
    "MissingBinaryError",
    "MissingDatabaseError",
]
