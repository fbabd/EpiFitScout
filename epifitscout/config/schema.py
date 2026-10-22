"""Frozen configuration dataclasses for all pipeline stages.

Plain Python dataclasses with no third-party dependency at import time; the
YAML defaults shipped in ``epifitscout/conf/config.yaml`` are read into these
by :func:`epifitscout.config_from_yaml`.

Every threshold here is a number someone can be wrong about, so each carries
where it came from. Thresholds calibrated on native complexes say so; ones set
by convention say that instead, which is a standing invitation to fit them.
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
    depend on the current working directory. Setting ``database_path`` to a
    filtered PDS list is also how a donor is excluded from a search, which is
    cheaper and safer than discarding its matches afterwards.
    """

    binary_path: Path | None = None
    database_path: Path | None = None
    rmsd_threshold: float = 2.0
    max_hits: int = 500
    #: Per-MASTER-process wall clock limit. The search is I/O bound, so the
    #: figure that matters is chains per process: 420 s was enough for ~6,000
    #: and silently too short for ~12,600, which cost 85 of 101 targets in a
    #: benchmark run before the cause was found. Set generously — a timeout
    #: here loses a whole target, and the search exits on its own when done.
    timeout_seconds: int = 1800
    n_threads: int = 0
    #: Residues of slack around the loop length in ``--gapLen``. Zero retrieves
    #: only loops of exactly the length the graft replaces, which keeps the
    #: template span and the candidate one-to-one. Non-zero changes the CDR
    #: length and needs the replaced span adjusted to match.
    gap_tolerance: int = 0


@dataclass(frozen=True)
class FeasibilityConfig:
    """When a retrieved loop can actually be bonded into the framework.

    A graft is accepted only if the two new peptide bonds it creates are real
    bonds. Ideal geometry is a C–N distance of 1.33 Å, a CA–C–N angle of 116°
    and a C–N–CA angle of 122°; the tolerances below are wide enough to pass a
    native loop re-grafted into its own framework and narrow enough to reject a
    loop whose ends point the wrong way.

    ``max_stem_rmsd`` is a second cut on the fit residual, below MASTER's own
    cutoff. It is a weak filter by design: conserved framework stems superpose
    well on every antibody, so stem RMSD bunches near the cutoff and
    discriminates poorly. Closure geometry is the gate that does the work.
    """

    #: Acceptable C–N distance at a junction, in Å (ideal 1.33).
    bond_min: float = 1.0
    bond_max: float = 1.8
    #: Permitted deviation from ideal junction bond angles, in degrees.
    angle_tolerance: float = 30.0
    #: Heavy-atom separation below which the loop clashes with its own
    #: framework. Backbone-to-backbone; the two residues flanking each junction
    #: are exempt because they are bonded to the loop.
    self_clash_cutoff: float = 2.2
    #: Reject a fit whose full-backbone stem residual exceeds this, in Å.
    max_stem_rmsd: float = 2.0
    #: Reject a loop whose span per residue exceeds this, in Å. An extended
    #: chain covers ~3.3 Å per residue, so beyond it the loop cannot close.
    max_span_per_residue: float = 3.5


@dataclass(frozen=True)
class InterfaceConfig:
    """Geometric criteria for the all-atom interface measurement.

    These are the conventional definitions, named so a reviewer can check them:
    an H-bond is a donor–acceptor heavy-atom pair within 3.5 Å whose angle at
    the acceptor exceeds 90°; a salt bridge is a charged-group pair within
    4.0 Å; a clash is a heavy-atom pair closer than the sum of their van der
    Waals radii less ``vdw_overlap``; buried area is the difference in
    solvent-accessible surface between the separated and combined partners,
    computed by Shrake–Rupley with a 1.4 Å probe.
    """

    probe_radius: float = 1.4
    sasa_points: int = 300
    #: Heavy-atom pairs closer than (r_i + r_j - vdw_overlap) count as clashing.
    vdw_overlap: float = 0.6
    #: Out to this separation two heavy atoms count as in contact, in Å.
    contact_cutoff: float = 4.5
    hbond_distance: float = 3.5
    hbond_angle: float = 90.0
    salt_bridge_distance: float = 4.0
    #: Grid spacing for the interface void (gap volume) calculation, in Å.
    #: 0.75 Å keeps the discretisation error under ~2% of the volume.
    grid_spacing: float = 0.75
    #: How far outside the union of the two surfaces the void grid extends, in Å.
    grid_margin: float = 3.0


@dataclass(frozen=True)
class ShapeConfig:
    """Weights for the epitope shape-complementarity score.

    This is the only thing that ranks candidates in v3. The two components are
    a depth-profile correlation between the loop and the epitope, and a
    backbone-torsion correlation; the defaults are v1's.

    The depth term's sign was wrong in v1 -- it rewarded interpenetration --
    and is corrected here. See :mod:`epifitscout.scoring.shape`.
    """

    #: Weight on the depth-profile correlation.
    weight_depth: float = 0.7
    #: Weight on the backbone-torsion correlation.
    weight_tau: float = 0.3


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration.

    ``pool_size`` is the search budget: how many placed, graftable candidates
    are scored for shape and kept in contention. It must be the same for any
    method being compared, since a bigger pool is a bigger search budget and
    not a better method — v1's apparent advantage over a diffusion model
    largely disappeared once that was held equal.

    Only the ``top_k`` models that are actually reported get packed all-atom and
    measured, because none of those measurements decide the ranking.
    """

    master: MASTERConfig = field(default_factory=MASTERConfig)
    feasibility: FeasibilityConfig = field(default_factory=FeasibilityConfig)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    #: Graftable candidates kept in contention after the shape score.
    pool_size: int = 50
    #: Antigen heavy atoms this close to the template CDR define the
    #: epitope, in A. Only residues genuinely in contact.
    epitope_cutoff: float = 4.5
    #: Loop backbone within this of the antigen is interpenetration, in
    #: A. Calibrated on 86 natives, whose closest approach is 2.12 A.
    collision_cutoff: float = 2.0
    #: Candidates returned after the final ranking.
    top_k: int = 10
    #: Stem length, in residues, taken on each side of the CDR for the query.
    stem_length: int = 4
    log_level: str = "INFO"
    max_workers: int = 6   # 1 = sequential; >1 = parallel search
