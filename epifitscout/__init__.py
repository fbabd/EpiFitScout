"""epifitscout — retrieval and grafting of real CDR loops for a target epitope.

Quick start::

    import epifitscout

    result = epifitscout.graft("complex.pdb", "H", (95, 102), ("A",),
                               out_dir="models")
    for candidate in result.top:
        print(candidate.source, candidate.sequence, candidate.shape.describe())

The query is the two framework stems flanking the CDR, never the CDR itself, so
the loops that come back are a prediction rather than a restatement of the one
already there. Candidates are chosen by how much of the loop sits cleanly
against the epitope, counted from backbone distances.

Three things are pasted into the repository before searching: the MASTER
binaries (``MASTER/bin/``), the chain database (``data/sabdab_chains.db/``) and
FASPR (``FASPR/bin/``). Run ``epifitscout check`` to confirm all three.
"""

from __future__ import annotations

__version__ = "3.0.0"

from pathlib import Path
from typing import Any

from epifitscout import paths
from epifitscout.config.schema import (
    FeasibilityConfig,
    InterfaceConfig,
    MASTERConfig,
    PipelineConfig,
    ShapeConfig,
)
from epifitscout.graft.assemble import TemplateSite
from epifitscout.graft.feasibility import GraftFeasibility
from epifitscout.paths import (
    MissingBinaryError,
    MissingDatabaseError,
    SetupError,
)
from epifitscout.pipeline.graft_pipeline import GraftPipeline, GraftResult
from epifitscout.query.builder import QueryBuilder, load_fragment_from_pdb
from epifitscout.scoring.interface import InterfaceScore, InterfaceScorer
from epifitscout.types.anchor_query import AnchorQuery
from epifitscout.types.candidate import GraftCandidate
from epifitscout.types.fragment import Fragment


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

    def section(name: str, kind: type) -> Any:
        """Build one config section, rejecting keys it does not have.

        A silently ignored key is a silently ignored experiment: a misspelt
        ``weight_hbonds`` would leave the default in place and the run would
        look like a result for the setting that was asked for.
        """
        import dataclasses

        values = raw.get(name, {}) or {}
        known = {f.name for f in dataclasses.fields(kind)}
        unknown = set(values) - known
        if unknown:
            raise SetupError(
                f"unknown key(s) in config section '{name}': "
                f"{', '.join(sorted(unknown))}; valid keys are "
                f"{', '.join(sorted(known))}"
            )
        return kind(**values)

    return PipelineConfig(
        master=section("master", MASTERConfig),
        feasibility=section("feasibility", FeasibilityConfig),
        interface=section("interface", InterfaceConfig),
        shape=section("shape", ShapeConfig),
        pool_size=raw.get("pool_size", 50),
        top_k=raw.get("top_k", 10),
        stem_length=raw.get("stem_length", 4),
        log_level=raw.get("log_level", "INFO"),
        max_workers=raw.get("max_workers", 6),
    )


def graft(
    template: Path | str,
    ab_chain: str,
    loop_span: tuple[int, int],
    antigen_chains: tuple[str, ...] | list[str],
    *,
    out_dir: Path | str | None = None,
    exclude_donors: set[str] | frozenset[str] | None = None,
    exclude_native_loop: bool = False,
    config_path: Path | None = None,
    pool_size: int | None = None,
    top_k: int | None = None,
    packer: str = "FASPR",
    keep_models: bool = True,
    max_workers: int | None = None,
) -> GraftResult:
    """Retrieve, graft and rank real CDR loops for one antibody–antigen complex.

    The v2 entry point. The CDR's own coordinates are never used as the query —
    the two framework stems flanking it are — so the loops returned are a
    prediction rather than a restatement of the one already there. See
    :class:`~epifitscout.pipeline.graft_pipeline.GraftPipeline` for the stages.

    Args:
        template: The antibody–antigen complex to graft into.
        ab_chain: Chain carrying the CDR.
        loop_span: ``(first, last)`` residue numbers of the CDR to replace.
            State it explicitly when comparing against another method: IMGT and
            Chothia CDR-H3 differ by two residues, which silently has the two
            methods designing different windows.
        antigen_chains: Chains treated as antigen.
        out_dir: Where models and MASTER files are written. A temporary
            directory is used when omitted, and models are not kept.
        exclude_donors: PDB IDs the search must not return — the leakage
            control for a benchmark drawn from the same source as the database.
        exclude_native_loop: Also refuse any retrieved loop sharing an
            8-residue stretch with the template's own CDR. Excluding the
            target's PDB ID does not cover the same antibody deposited under
            another accession; a benchmark wants this on.
        config_path: Alternative YAML config.
        pool_size: Feasible candidates to pack and measure. This is the search
            budget; hold it equal across methods being compared.
        top_k: Candidates to return.
        packer: Side-chain packer, or ``"none"`` for backbone-only models,
            which makes the interface numbers diagnostic rather than results.
        keep_models: Keep the packed PDB files.
        max_workers: Parallel MASTER shards and packing threads.

    Returns:
        A :class:`~epifitscout.pipeline.graft_pipeline.GraftResult`, whose
        ``top`` is the ranked candidates and whose ``funnel`` records how many
        survived each stage.

    Raises:
        MissingBinaryError: If the MASTER executables cannot be found.
        MissingDatabaseError: If the chain database is absent or incomplete.

    Example::

        result = epifitscout.graft(
            "7abc.pdb", "H", (95, 102), ("A",), out_dir="models/7abc",
        )
        for candidate in result.top:
            print(candidate.source, candidate.interface.describe())
    """
    base = config_from_yaml(config_path)
    cfg = PipelineConfig(
        master=MASTERConfig(
            binary_path=paths.find_master_binary(None),
            database_path=paths.find_pds_list(None),
            rmsd_threshold=base.master.rmsd_threshold,
            max_hits=base.master.max_hits,
            timeout_seconds=base.master.timeout_seconds,
            n_threads=base.master.n_threads,
            gap_tolerance=base.master.gap_tolerance,
        ),
        feasibility=base.feasibility,
        interface=base.interface,
        shape=base.shape,
        epitope_cutoff=base.epitope_cutoff,
        pool_size=pool_size if pool_size is not None else base.pool_size,
        top_k=top_k if top_k is not None else base.top_k,
        stem_length=base.stem_length,
        log_level=base.log_level,
        max_workers=max_workers if max_workers is not None else base.max_workers,
    )
    return GraftPipeline(cfg).run(
        template=Path(template),
        ab_chain=ab_chain,
        loop_span=loop_span,
        antigen_chains=tuple(antigen_chains),
        out_dir=Path(out_dir) if out_dir is not None else None,
        exclude_donors=exclude_donors,
        exclude_native_loop=exclude_native_loop,
        packer_name=packer,
        keep_models=keep_models,
    )


__all__ = [
    # The entry point
    "graft",
    # Pipeline and its pieces, for stage-by-stage use
    "GraftPipeline",
    "GraftResult",
    "TemplateSite",
    "QueryBuilder",
    "load_fragment_from_pdb",
    # Data types
    "AnchorQuery",
    "Fragment",
    "GraftCandidate",
    "GraftFeasibility",
    "InterfaceScore",
    "InterfaceScorer",
    # Configuration
    "config_from_yaml",
    "PipelineConfig",
    "MASTERConfig",
    "FeasibilityConfig",
    "InterfaceConfig",
    "ShapeConfig",
    # Resource resolution
    "paths",
    "SetupError",
    "MissingBinaryError",
    "MissingDatabaseError",
]
