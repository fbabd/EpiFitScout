"""Build the all-atom model a candidate is judged on.

Order of operations, and why it is this way round
-------------------------------------------------
v1 ranked candidates on backbone geometry and packed side chains afterwards,
only for the handful it had already chosen. That inverts the dependency:
complementarity is decided by side chains, so a backbone-only score is ranking
on a proxy with little spread — which is what the v1 benchmark found, a shape
term that reordered candidates without improving any measured property.

v2 packs first. Every candidate that survives the feasibility gate is built
all-atom, with the donor's loop sequence in the template's environment, and
only then measured. Packing is the expensive step, which is why the gate runs
before it and why the pool handed to this module is bounded
(``PipelineConfig.pool_size``).

A note on what is being asked of the packer: the loop's *sequence* comes from
the donor, and the packer finds rotamers for that sequence in this new
environment. So the question each model answers is "can the donor's sequence be
accommodated here", not "what is the best sequence here" — this method retrieves
loops, it does not design them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from epifitscout.graft import packer as pk
from epifitscout.graft.relax import relax_side_chains
from epifitscout.types.candidate import GraftCandidate
from epifitscout.utils.io import detect_format

logger = logging.getLogger(__name__)

__all__ = ["AssembledModel", "TemplateSite", "write_graft"]


@dataclass(frozen=True)
class AssembledModel:
    """A written, optionally packed model and what happened on the way.

    Attributes:
        path: The PDB file written.
        packed: True when a packer rebuilt the graft's side chains. A model with
            ``packed`` False is backbone-only at the loop and must not be used
            for an interface measurement.
        problem: Why packing failed, when it did.
    """

    path: Path
    packed: bool
    problem: str | None = None


@dataclass
class TemplateSite:
    """The template complex, read once, with the CDR site located in it.

    Reading and indexing the template is the same work for every candidate, so
    it happens here once per target rather than inside the per-candidate loop.

    Attributes:
        path: Template PDB or mmCIF file.
        ab_chain: Chain carrying the CDR.
        span: ``(first, last)`` residue numbers of the loop being replaced.
        antigen_chains: Chains treated as antigen.
        framework_coords: ``(A, 3)`` antibody heavy atoms a graft must not clash
            with — the antibody *minus* the replaced span, minus the residue
            either side of it (those two are bonded to the graft and are meant
            to be in contact with it), and minus water. Ordered solvent was
            modelled around the template's own loop, so a water where the graft
            now sits is displaced, not clashed with; the packer deletes those
            waters for the same reason.
        antigen_coords: ``(A, 3)`` antigen heavy atoms.
        antigen_elements: Element symbol of each antigen atom, in the same
            order — needed for van der Waals radii when the antigen's unbound
            surface area is computed once per target.
    """

    path: Path
    ab_chain: str
    span: tuple[int, int]
    antigen_chains: tuple[str, ...]
    framework_coords: np.ndarray = field(repr=False)
    antigen_coords: np.ndarray = field(repr=False)
    antigen_elements: tuple[str, ...] = field(default=(), repr=False)
    #: ``(A,)`` residue id per antigen atom, for grouping atoms into residues.
    antigen_residue_index: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int), repr=False
    )
    #: ``(A,)`` mask marking each antigen residue's CA, which is what the shape
    #: profile is built from.
    antigen_is_ca: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=bool), repr=False
    )
    #: ``(L, 3)`` heavy atoms of the template's own CDR. Used only to locate the
    #: epitope by proximity; never passed to the search.
    native_loop_coords: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3)), repr=False
    )

    @classmethod
    def load(
        cls,
        path: Path,
        ab_chain: str,
        span: tuple[int, int],
        antigen_chains: tuple[str, ...],
    ) -> TemplateSite:
        """Read the template and index the two atom sets a graft is judged by."""
        lo, hi = span
        framework: list[np.ndarray] = []
        antigen: list[np.ndarray] = []
        antigen_elements: list[str] = []
        antigen_residue: list[int] = []
        antigen_ca: list[bool] = []
        native_loop: list[np.ndarray] = []
        residue_ids: dict[tuple[str, int, str], int] = {}
        for line in Path(path).read_text().splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if pk.element_of(line) == "H" or line[16] not in (" ", "A"):
                continue
            if line[17:20].strip() in pk.WATER_NAMES:
                continue
            chain = line[21]
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            coord = pk._coord(line)
            if chain in antigen_chains:
                antigen.append(coord)
                antigen_elements.append(pk.element_of(line))
                key = (chain, resnum, line[26])
                antigen_residue.append(
                    residue_ids.setdefault(key, len(residue_ids))
                )
                antigen_ca.append(line[12:16].strip() == "CA")
            elif chain == ab_chain and lo - 1 <= resnum <= hi + 1:
                if lo <= resnum <= hi:
                    native_loop.append(coord)
                continue        # replaced span, plus the two bonded neighbours
            else:
                framework.append(coord)

        site = cls(
            path=Path(path), ab_chain=ab_chain, span=span,
            antigen_chains=tuple(antigen_chains),
            framework_coords=np.asarray(framework, dtype=np.float64).reshape(-1, 3),
            antigen_coords=np.asarray(antigen, dtype=np.float64).reshape(-1, 3),
            antigen_elements=tuple(antigen_elements),
            antigen_residue_index=np.asarray(antigen_residue, dtype=int),
            antigen_is_ca=np.asarray(antigen_ca, dtype=bool),
            native_loop_coords=np.asarray(
                native_loop, dtype=np.float64
            ).reshape(-1, 3),
        )
        logger.info(
            "Template %s: %d framework atoms, %d antigen atoms (chains %s)",
            site.path.name, len(site.framework_coords), len(site.antigen_coords),
            ",".join(antigen_chains),
        )
        if not len(site.antigen_coords):
            raise ValueError(
                f"no atoms found for antigen chain(s) {antigen_chains} in "
                f"{site.path.name}; nothing to measure complementarity against"
            )
        return site

    # ------------------------------------------------------------------ #
    # Building models
    # ------------------------------------------------------------------ #

    def assemble(
        self,
        candidate: GraftCandidate,
        out_path: Path,
        packer_name: str = "FASPR",
        packer_exe: Path | None = None,
        relax: bool = True,
        shell: float = pk.DEFAULT_REPACK_SHELL,
    ) -> AssembledModel:
        """Graft this candidate into the template and pack its side chains.

        Args:
            candidate: A feasible, placed candidate.
            out_path: Where to write the model.
            packer_name: Packer to drive; ``"none"`` writes backbone only.
            relax: Turn clashing side chains out of the antigen after packing.
                On by default -- without it the models carry the packer's
                discrete-rotamer near-misses, which read as clashes.
            packer_exe: Resolved packer path. Looked up when omitted.
            shell: Radius around the graft that is also repacked, in Å.

        Returns:
            An :class:`AssembledModel`. Check ``packed`` before scoring.
        """
        write_graft(
            self.path, self.ab_chain, self.span,
            candidate.placed_loop, candidate.sequence, out_path,
        )
        if packer_name == "none":
            return AssembledModel(path=out_path, packed=False, problem="packing skipped")

        exe = packer_exe or pk.find_packer(packer_name)
        if exe is None:
            return AssembledModel(
                path=out_path, packed=False,
                problem=f"{packer_name} not found; see {packer_name}/README.md",
            )
        problem = pk.check_packer(exe, packer_name)
        if problem is not None:
            return AssembledModel(path=out_path, packed=False, problem=problem)

        ok = pk.run_packer(
            exe, packer_name, out_path,
            chain=self.ab_chain, span=self.span, shell=shell,
        )
        if not ok:
            return AssembledModel(
                path=out_path, packed=False,
                problem="the packer rejected this model",
            )

        # The packer chooses from discrete rotamers against a looser tolerance
        # than the scorer applies, so what it leaves is a field of near-misses.
        # Turning the chi angles of clashing side chains costs a few seconds
        # and preserves every bond length and angle exactly.
        if relax:
            lines, n_turned = relax_side_chains(
                out_path.read_text().splitlines(keepends=True),
                self.ab_chain, self.span, self.antigen_chains,
            )
            if n_turned:
                out_path.write_text("".join(lines))

        return AssembledModel(path=out_path, packed=True, problem=None)


def write_graft(
    template: Path,
    ab_chain: str,
    span: tuple[int, int],
    loop_coords: np.ndarray,
    loop_sequence: str,
    out_path: Path,
    side_chains: dict[int, list[tuple[str, np.ndarray]]] | None = None,
) -> None:
    """Write the template with its CDR replaced by a placed loop.

    Args:
        template: Template structure, PDB or mmCIF.
        ab_chain: Chain carrying the CDR.
        span: ``(first, last)`` residue numbers to replace.
        loop_coords: ``(N, 4, 3)`` backbone of the replacement loop, already in
            the template's frame.
        loop_sequence: One-letter sequence of the replacement loop.
        out_path: Destination PDB file.
        side_chains: Optional per-residue side-chain atoms to write as well,
            keyed by index into the loop. Normally left out: the packer rebuilds
            them from the sequence.

    Raises:
        ValueError: If the replaced span and the loop differ in residue count.
    """
    from Bio.PDB import PDBIO, MMCIFParser, PDBParser
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Residue import Residue

    parser = (MMCIFParser(QUIET=True) if detect_format(template) == "cif"
              else PDBParser(QUIET=True))
    structure = parser.get_structure("t", str(template))
    model = next(iter(structure))
    chain = model[ab_chain]

    lo, hi = span
    # Reuse the template's own residue identifiers for the replaced stretch.
    # Antibody CDRs are routinely numbered with insertion codes (100, 100A,
    # 100B...), so a span of 93-102 can hold more than ten residues; renumbering
    # the graft sequentially would collide with residues further along the chain.
    replaced = [r for r in chain if r.id[0] == " " and lo <= r.id[1] <= hi]
    replaced_ids = [r.id for r in replaced]
    for residue in replaced:
        chain.detach_child(residue.id)

    n_res = len(loop_coords)
    if len(replaced_ids) != n_res:
        raise ValueError(
            f"template has {len(replaced_ids)} residues in {ab_chain}/{lo}-{hi} "
            f"but the loop has {n_res}; they must match one to one"
        )

    sequence = loop_sequence or "X" * n_res
    serial = 1
    for i in range(n_res):
        resname = pk.ONE_TO_THREE.get(sequence[i].upper(), "GLY")
        residue = Residue(replaced_ids[i], resname, "")
        for j, atom_name in enumerate(pk.BACKBONE):
            residue.add(Atom(atom_name, loop_coords[i, j, :], 0.0, 1.0, " ",
                             f" {atom_name:<3}"[:4], serial, atom_name[0]))
            serial += 1
        for name, coord in (side_chains or {}).get(i, []):
            element = name[0] if name[0].isalpha() else "C"
            residue.add(Atom(name, coord, 0.0, 1.0, " ", f" {name:<3}"[:4],
                             serial, element))
            serial += 1
        chain.add(residue)

    # keep residues in sequence order so the file reads as a real chain
    chain.child_list.sort(key=lambda r: (r.id[1], r.id[2]))

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path))
