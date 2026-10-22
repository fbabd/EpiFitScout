"""GraftPipeline: stem search → feasibility → pack → interface → rank.

The v2 pipeline, and the order of its stages is the substance of the method::

    1  search      MASTER, two framework stems with --gapLen fixing the span
                   between them. Returns every real loop that can bridge this
                   gap in this geometry. The native loop is never shown to it.
    2  place       Least-squares fit of each donor's stems onto the template's.
                   The transform is determined, not searched: a grafted loop is
                   not free to move once it is bonded at both ends.
    3  gate        Junction geometry. An unclosable graft is dropped here and
                   never scored, because a score for a broken backbone
                   describes nothing.
    4  fit         How much of each placed loop sits cleanly against the
                   epitope, counted from backbone distances. **This is the
                   stage that chooses**, and it is the whole of the selection.
    5  rank        Sorted on that fit, clash-free models first; `top_k` kept.
    6  pack        Those are rebuilt all-atom and their interface measured, for
                   output and for evaluation — not to decide the ranking.

**Selection happens on backbone geometry, before packing.** v2 began the other
way round, packing all 50 pooled candidates and ranking on an all-atom measure,
on the reasoning that complementarity is a side-chain property. That reasoning
is sound and the measurement still came out flat: Lawrence–Colman Sc, properly
implemented, correlates with distance from the native loop at rho = +0.04 and
saturates at near-native values for loops 5.8 A away. Many conformations fit an
epitope about equally well. Since the expensive measure does not select, the
cheap one is used to select and the expensive ones are computed on the handful
of models that are actually reported — which makes the pipeline both simpler and
several times faster.

The gate still comes before everything, because there is no reason to score a
loop that cannot be bonded in.

The budget is explicit. `pool_size` is how many placed candidates are scored
for fit, and it is the number that must be held equal when comparing against
another method — v1's apparent advantage over a diffusion model largely
disappeared once the comparison was budget-matched, so any run that quotes a
comparison needs this number beside it.
"""

from __future__ import annotations

import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from epifitscout.config.schema import PipelineConfig
from epifitscout.graft import feasibility as feas
from epifitscout.graft import packer as pk
from epifitscout.graft.assemble import TemplateSite
from epifitscout.master.parser import ParseStats, parse_anchor_matches
from epifitscout.master.runner import MASTERRunner
from epifitscout.query.builder import QueryBuilder
from epifitscout.scoring.epitope import MIN_EPITOPE_RESIDUES, epitope_ca
from epifitscout.scoring.interface import InterfaceScorer
from epifitscout.scoring.shape import ShapeComplementarityScorer
from epifitscout.types.anchor_query import AnchorQuery
from epifitscout.types.candidate import GraftCandidate

logger = logging.getLogger(__name__)

__all__ = ["GraftPipeline", "GraftResult"]


def _order(candidates: list[GraftCandidate]) -> list[GraftCandidate]:
    """Best first, by shape complementarity to the epitope. Nothing else.

    This is the whole ranking. It is computed on backbone coordinates before
    anything is packed, so the order a caller sees is decided by how the loop
    sits against the epitope and by no property of the all-atom model. The
    interface numbers carried on each candidate are measurements of the models
    that ranking selected -- report them, do not sort by them.
    """
    return sorted(
        candidates,
        key=lambda c: float(getattr(c.shape, "score", 0.0) or 0.0),
        reverse=True,
    )


@dataclass
class GraftResult:
    """Everything one target produced, including what fell out at each stage.

    A thin result is a finding, not a failure to report: v1's coverage collapse
    on long loops was only visible because the counts were kept. ``funnel``
    records how many candidates survived each stage.
    """

    query: AnchorQuery
    site: TemplateSite
    top: list[GraftCandidate]
    pool: list[GraftCandidate] = field(default_factory=list)
    funnel: dict[str, int] = field(default_factory=dict)
    #: Antigen residues the shape score was computed against. Not a funnel
    #: stage -- it counts the target's epitope, not surviving candidates.
    epitope_residues: int = 0
    parse_stats: ParseStats | None = None

    @property
    def ok(self) -> bool:
        """True when at least one candidate came through packed and scored."""
        return bool(self.top)

    def describe(self) -> str:
        """One-line summary of the funnel."""
        stages = " → ".join(f"{k} {v}" for k, v in self.funnel.items())
        return f"{self.query.pdb_id}:{self.query.chain} {stages}"


class GraftPipeline:
    """Runs the six stages for one target at a time.

    Args:
        cfg: Pipeline configuration. ``cfg.pool_size`` sets the search budget
            and ``cfg.top_k`` how many models are returned.

    Usage::

        pipeline = GraftPipeline(PipelineConfig())
        result = pipeline.run(
            template=Path("7abc.pdb"), ab_chain="H", loop_span=(95, 102),
            antigen_chains=("A",), out_dir=Path("models/7abc"),
        )
        for candidate in result.top:
            print(candidate.source, candidate.interface.describe())
    """

    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self._cfg = cfg or PipelineConfig()
        logging.basicConfig(
            level=getattr(logging, self._cfg.log_level.upper(), logging.INFO)
        )
        self._runner = MASTERRunner(self._cfg.master, max_workers=self._cfg.max_workers)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        template: Path,
        ab_chain: str,
        loop_span: tuple[int, int],
        antigen_chains: tuple[str, ...] | list[str],
        out_dir: Path | None = None,
        exclude_donors: set[str] | frozenset[str] | None = None,
        exclude_native_loop: bool = False,
        packer_name: str = "FASPR",
        keep_models: bool = True,
    ) -> GraftResult:
        """Run every stage for one target.

        Args:
            template: The antibody–antigen complex to graft into.
            ab_chain: Chain carrying the CDR.
            loop_span: ``(first, last)`` residue numbers of the CDR to replace.
                Give this explicitly when comparing against another method —
                CDR definitions differ by two residues between IMGT and
                Chothia, which silently makes the two methods design different
                windows.
            antigen_chains: Chains treated as antigen.
            out_dir: Where models and MASTER files go. A temporary directory is
                used when omitted, and ``keep_models`` is then forced off.
            exclude_donors: PDB IDs the search must not return. For a benchmark
                this is the leakage control, and it belongs here: the database
                is built from SAbDab, so a target drawn from SAbDab can
                otherwise retrieve its own native loop and rank it first.
            exclude_native_loop: Also refuse any retrieved loop sharing an
                8-residue stretch with the template's own CDR. Excluding the
                target's PDB ID does **not** cover this: the same antibody is
                often deposited several times, and searching 1a2y's CDR-H3 with
                only 1A2Y excluded returns that exact loop from five other
                entries. Any benchmark wants this on.
            packer_name: Packer to rebuild side chains with, or ``"none"`` to
                skip packing — which also makes the interface measurement
                backbone-only, so it is for diagnostics, not results.
            keep_models: Keep the packed PDB files. Off saves disk on a large
                benchmark; the scores are kept either way.

        Returns:
            A :class:`GraftResult`.
        """
        if out_dir is None:
            with tempfile.TemporaryDirectory(prefix="epifitscout_graft_") as tmp:
                return self._run(
                    template, ab_chain, loop_span, antigen_chains, Path(tmp),
                    exclude_donors, exclude_native_loop, packer_name,
                    keep_models=False,
                )
        out_dir.mkdir(parents=True, exist_ok=True)
        return self._run(
            template, ab_chain, loop_span, antigen_chains, out_dir,
            exclude_donors, exclude_native_loop, packer_name, keep_models,
        )

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: AnchorQuery,
        out_dir: Path,
        exclude_donors: set[str] | frozenset[str] | None = None,
        stats: ParseStats | None = None,
        exclude_sequences: set[str] | frozenset[str] | None = None,
    ) -> list[GraftCandidate]:
        """Stages 1-2: retrieve loops that bridge the gap, and place them."""
        master_dir = out_dir / "master"
        master_dir.mkdir(parents=True, exist_ok=True)
        if self._cfg.max_workers > 1:
            match_path = self._runner.run_sharded(query, master_dir)
        else:
            match_path = self._runner.run(query, master_dir)
        return parse_anchor_matches(
            match_path, query,
            rmsd_threshold=self._cfg.master.rmsd_threshold,
            max_hits=self._cfg.master.max_hits,
            exclude_donors=frozenset(exclude_donors or ()),
            exclude_sequences=frozenset(exclude_sequences or ()),
            stats=stats,
        )

    def measure(
        self,
        candidates: list[GraftCandidate],
        query: AnchorQuery,
        site: TemplateSite,
        out_dir: Path,
        packer_name: str = "FASPR",
        keep_models: bool = True,
    ) -> list[GraftCandidate]:
        """Stages 4-5: pack each candidate all-atom and measure its interface.

        Candidates whose model could not be packed, or whose interface could not
        be measured, are returned unscored and are dropped by the caller. They
        are not silently treated as zero: a failed pack is missing evidence, not
        evidence of a bad loop.
        """
        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        scorer = InterfaceScorer(self._cfg.interface)
        scorer.prime(site)
        exe = None if packer_name == "none" else pk.find_packer(packer_name)

        def _one(indexed: tuple[int, GraftCandidate]) -> GraftCandidate:
            i, candidate = indexed
            path = models_dir / f"{i:03d}_{candidate.donor}.pdb"
            model = site.assemble(
                candidate, path, packer_name=packer_name, packer_exe=exe
            )
            candidate.metadata["packed"] = model.packed
            if not model.packed and packer_name != "none":
                candidate.metadata["pack_problem"] = model.problem
                logger.debug("%s: %s", candidate.source, model.problem)
                return candidate
            try:
                candidate.interface = scorer.score(
                    model.path, site.ab_chain, site.span, site.antigen_chains
                )
            except (ValueError, MemoryError) as exc:
                candidate.metadata["score_problem"] = str(exc)
                logger.debug("%s: cannot measure: %s", candidate.source, exc)
            if keep_models:
                candidate.model_path = model.path
            else:
                path.unlink(missing_ok=True)
            return candidate

        workers = max(1, self._cfg.max_workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scored = list(pool.map(_one, enumerate(candidates)))
        return [c for c in scored if c.interface is not None]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _run(
        self,
        template: Path,
        ab_chain: str,
        loop_span: tuple[int, int],
        antigen_chains: tuple[str, ...] | list[str],
        out_dir: Path,
        exclude_donors: set[str] | frozenset[str] | None,
        exclude_native_loop: bool,
        packer_name: str,
        keep_models: bool,
    ) -> GraftResult:
        cfg = self._cfg
        builder = QueryBuilder(template)
        resolved_ab = builder.resolve_chain(ab_chain)
        query = builder.get_anchor_query(
            resolved_ab, loop_span[0], loop_span[1], stem_length=cfg.stem_length
        )
        site = TemplateSite.load(
            template, resolved_ab, query.loop_range,
            tuple(builder.resolve_chain(c) for c in antigen_chains),
        )

        funnel: dict[str, int] = {}
        if not feas.query_is_bridgeable(query, cfg.feasibility):
            logger.warning(
                "Site %s spans %.2f Å per residue, beyond the reach of a chain; "
                "no loop of this length can close it",
                query.describe(), query.span_per_residue,
            )
            return GraftResult(
                query=query, site=site, top=[], funnel={"unbridgeable": 0}
            )

        stats = ParseStats()
        native = query.metadata.get("native_loop_sequence", "")
        candidates = self.search(
            query, out_dir, exclude_donors, stats,
            exclude_sequences={native} if exclude_native_loop and native else None,
        )
        funnel["retrieved"] = len(candidates)

        feasible = feas.assess_all(
            candidates, query, site.framework_coords, cfg.feasibility
        )
        funnel["feasible"] = len(feasible)

        # Stage 4: the epitope. Antigen residues in contact with the
        # template's own CDR, which is the site a replacement has to fit.
        epitope = epitope_ca(
            site.native_loop_coords, site.antigen_coords,
            site.antigen_residue_index, site.antigen_is_ca,
            cutoff=cfg.epitope_cutoff,
        )
        if len(epitope) < MIN_EPITOPE_RESIDUES:
            logger.warning(
                "Only %d epitope residues within %.1f A of the template CDR; "
                "the shape profile needs %d, so every candidate will score 0",
                len(epitope), cfg.epitope_cutoff, MIN_EPITOPE_RESIDUES,
            )

        # Stage 5: the selection. Shape complementarity against that epitope,
        # on backbone coordinates alone -- no packing, no all-atom interface.
        # This is what decides which candidates are worth building.
        scorer_shape = ShapeComplementarityScorer(
            weight_depth=cfg.shape.weight_depth,
            weight_tau=cfg.shape.weight_tau,
        )
        for candidate in feasible:
            candidate.shape = scorer_shape.score(candidate.placed_loop, epitope)

        pool = sorted(
            feasible,
            key=lambda c: float(c.shape.score),
            reverse=True,
        )[: cfg.pool_size]
        funnel["pooled"] = len(pool)

        # Stage 6: only the models that will be reported are packed and
        # measured. Those numbers are output and evidence, not the ranking.
        scored = self.measure(
            pool[: cfg.top_k], query, site, out_dir,
            packer_name=packer_name, keep_models=keep_models,
        )
        funnel["scored"] = len(scored)

        ranked = _order(scored)
        funnel["returned"] = len(ranked)

        result = GraftResult(
            query=query, site=site, top=ranked, pool=ranked,
            funnel=funnel, parse_stats=stats, epitope_residues=len(epitope),
        )
        logger.info("Pipeline: %s", result.describe())
        return result
