"""epifitscout — Protein structural fragment search with epitope complementarity scoring.

Quick start::

    import epifitscout

    qb = epifitscout.QueryBuilder("my_complex.pdb")
    qb.describe()                                 # show chains + residue numbers

    query_cdr     = qb.get_fragment("H", 100, 112)
    query_epitope = qb.get_fragment("A",  50,  65)

    # Single query (uses max_workers from config for DB sharding)
    hits = epifitscout.search(query_cdr, query_epitope)
    for h in hits[:5]:
        print(h)

The ``search()`` function auto-resolves the MASTER binary and the chain database
relative to the installed package location — no manual path configuration needed.
"""

from __future__ import annotations

__version__ = "0.1.0"

from pathlib import Path
from typing import TYPE_CHECKING

from epifitscout.config.schema import MASTERConfig, PipelineConfig, RankingConfig, ScoringConfig
from epifitscout.pipeline.search_pipeline import FragmentSearchPipeline
from epifitscout.query.builder import QueryBuilder, load_fragment_from_pdb
from epifitscout.types.fragment import Fragment, ScoredHit

if TYPE_CHECKING:
    pass

# Root of the EpiFitScout deployment (one level above the package directory).
_PACKAGE_ROOT: Path = Path(__file__).resolve().parent.parent

_DEFAULT_BINARY: Path = _PACKAGE_ROOT / "MASTER" / "bin" / "master"
_DEFAULT_DB: Path = _PACKAGE_ROOT / "data" / "sabdab_chains.db" / "pds.list"
_DEFAULT_CONFIG: Path = _PACKAGE_ROOT / "conf" / "config.yaml"


def config_from_yaml(yaml_path: Path | str) -> PipelineConfig:
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
        yaml_path: Path to the YAML config file (e.g. ``conf/config.yaml``).

    Returns:
        PipelineConfig with values from the file, defaults for missing keys.
    """
    import yaml

    with open(yaml_path) as fh:
        raw: dict = yaml.safe_load(fh) or {}

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
    log_level: str | None,
    max_workers: int | None,
) -> PipelineConfig:
    """Build and validate a PipelineConfig, applying per-call overrides.

    Loads base values from conf/config.yaml if present, then applies any
    explicitly provided overrides. Validates binary and database paths.

    Raises:
        FileNotFoundError: If the MASTER binary or database list is missing.
    """
    base = (
        config_from_yaml(_DEFAULT_CONFIG)
        if _DEFAULT_CONFIG.exists()
        else PipelineConfig()
    )

    bin_path = Path(binary_path) if binary_path is not None else _DEFAULT_BINARY
    db_path = Path(database_path) if database_path is not None else _DEFAULT_DB

    if not bin_path.exists():
        raise FileNotFoundError(
            f"MASTER binary not found: {bin_path}\n"
            "Pass binary_path= explicitly or place the binary at the expected location."
        )
    if not db_path.exists():
        raise FileNotFoundError(
            f"Chain database list not found: {db_path}\n"
            "Run 'python run/build_chain_db.py' first to build the database, "
            "or pass database_path= explicitly."
        )

    return PipelineConfig(
        master=MASTERConfig(
            binary_path=bin_path,
            database_path=db_path,
            rmsd_threshold=rmsd_threshold if rmsd_threshold is not None else base.master.rmsd_threshold,
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
    query_epitope: Fragment,
    *,
    rmsd_threshold: float | None = None,
    max_hits: int | None = None,
    weight_depth: float | None = None,
    weight_tau: float | None = None,
    weight_rmsd: float | None = None,
    weight_shape: float | None = None,
    binary_path: Path | None = None,
    database_path: Path | None = None,
    log_level: str | None = None,
    max_workers: int | None = None,
) -> list[ScoredHit]:
    """Search for CDR-like fragments that are complementary to a target epitope.

    Runs the full 3-step pipeline: MASTER structural search → shape
    complementarity scoring → weighted ranking.

    Configuration is loaded automatically from ``conf/config.yaml`` (relative to
    the project root) if the file exists.  Any keyword argument passed explicitly
    overrides the corresponding value from the file.  If the file is absent,
    built-in defaults are used.

    When ``max_workers > 1`` (from config or explicit override), MASTER searches
    the database in parallel across shards for a ~N× speedup.

    Args:
        query_cdr: Query CDR fragment (backbone coords).
        query_epitope: Query epitope fragment (backbone coords).
        rmsd_threshold: Maximum backbone RMSD (Å). Overrides config if given.
        max_hits: Maximum MASTER hits. Overrides config if given.
        weight_depth: Depth anti-correlation weight in S_shape. Overrides config.
        weight_tau: Torsion co-correlation weight in S_shape. Overrides config.
        weight_rmsd: RMSD weight in S_final. Overrides config if given.
        weight_shape: Shape weight in S_final. Overrides config if given.
        binary_path: Path to MASTER binary. Defaults to bundled binary.
        database_path: Path to pds.list. Defaults to built DB in data/.
        log_level: Python logging level string. Overrides config if given.
        max_workers: Parallel MASTER shard processes. Overrides config if given.

    Returns:
        Ranked list of :class:`ScoredHit` objects, best first.

    Raises:
        FileNotFoundError: If the MASTER binary or database list file is missing.

    Example::

        # Uses conf/config.yaml automatically
        hits = epifitscout.search(query_cdr, query_epitope)

        # Override one weight, rest from config.yaml
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
        log_level=log_level,
        max_workers=max_workers,
    )
    pipeline = FragmentSearchPipeline(cfg)
    return pipeline.search(query_cdr, query_epitope)


__all__ = [
    # Top-level convenience API
    "search",
    # Query helpers
    "QueryBuilder",
    "load_fragment_from_pdb",
    # Data types
    "Fragment",
    "ScoredHit",
    # Config (for advanced / step-by-step usage)
    "config_from_yaml",
    "FragmentSearchPipeline",
    "PipelineConfig",
    "MASTERConfig",
    "ScoringConfig",
    "RankingConfig",
]
