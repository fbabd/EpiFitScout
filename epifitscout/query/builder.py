"""QueryBuilder: create query Fragment objects from a PDB file.

Typical usage — prepare a CDR loop and epitope surface from a known
antibody-antigen complex, then feed them to FragmentSearchPipeline.search().

Example::

    from epifitscout.query.builder import QueryBuilder

    qb = QueryBuilder(Path("my_complex.pdb"))
    qb.describe()                          # print chains + residue ranges

    query_cdr     = qb.get_fragment("H", res_start=100, res_end=112)
    query_epitope = qb.get_fragment("A", res_start=50,  res_end=65)

    # --- or: auto-detect IMGT CDR loops on heavy chain ---
    cdr_frags = qb.get_imgt_cdrs("H")     # returns dict[str, Fragment]
    query_h3  = cdr_frags["H3"]
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

from epifitscout.types.epitope import EpitopeSurface
from epifitscout.types.fragment import Fragment
from epifitscout.utils.io import (
    detect_format,
    extract_backbone_coords_by_range,
    extract_residue_atoms,
)

logger = logging.getLogger(__name__)

# IMGT CDR core ranges
IMGT_CDR_RANGES: dict[str, tuple[int, int]] = {
    "H1": (27, 38),
    "H2": (56, 65),
    "H3": (105, 117),
    "L1": (27, 38),
    "L2": (56, 65),
    "L3": (105, 117),
}


def load_fragment_from_pdb(
    pdb_path: Path,
    chain_id: str,
    res_start: int,
    res_end: int,
    pdb_id: str | None = None,
    fmt: str | None = None,
) -> Fragment:
    """Load a backbone Fragment directly from a PDB file.

    Args:
        pdb_path: Path to the PDB file.
        chain_id: Chain identifier (single character).
        res_start: First residue number (inclusive, integer part of resseq).
        res_end: Last residue number (inclusive, integer part of resseq).
        pdb_id: Optional label; defaults to the file stem.
        fmt: ``"pdb"`` or ``"cif"``; detected from the suffix when omitted.

    Returns:
        Fragment with backbone coords (N, 4, 3).

    Raises:
        ValueError: If no residues are found in the specified range.
    """
    pdb_path = Path(pdb_path)
    label = pdb_id or pdb_path.stem.upper()
    coords, resnums, sequence = extract_backbone_coords_by_range(
        pdb_path, chain_id, res_start, res_end, fmt=fmt
    )
    return Fragment(
        pdb_id=label,
        chain=chain_id,
        residue_range=(resnums[0], resnums[-1]),
        coords=coords,
        sequence=sequence,
        metadata={"source": "query", "pdb_file": str(pdb_path)},
    )


class QueryBuilder:
    """Interactive helper for building query Fragments from a PDB file.

    Args:
        pdb_path: Path to any PDB file (complex, monomer, etc.).

    Usage::

        qb = QueryBuilder(Path("1abc.pdb"))
        qb.describe()                          # show chains + ranges
        cdr_frags = qb.get_imgt_cdrs("H")     # auto-detect CDR loops
        epitope   = qb.get_fragment("A", 45, 60)
    """

    def __init__(self, pdb_path: Path) -> None:
        self._pdb_path = Path(pdb_path)
        self._pdb_id = self._pdb_path.stem.upper()
        # RCSB serves entries too large for the PDB format as mmCIF only, and
        # those are exactly the large complexes worth searching. Detect once so
        # every reader below uses the right parser.
        self._fmt = detect_format(self._pdb_path)
        self._chain_info: dict[str, dict] | None = None

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #

    def describe(self) -> None:
        """Print a summary of all chains and residue ranges in the PDB."""
        info = self._get_chain_info()
        print(f"\nPDB: {self._pdb_path.name}")
        print(f"{'Chain':<6} {'Residues':<12} {'Range':<14} {'Sequence (first 20)'}")
        print("-" * 60)
        for chain_id, d in sorted(info.items()):
            res_range = f"{d['min_res']}–{d['max_res']}"
            print(
                f"{chain_id:<6} {d['n_residues']:<12} {res_range:<14} {d['sequence'][:20]}"
            )
        print()

    def chains(self) -> list[str]:
        """Return list of chain IDs present in the PDB."""
        return sorted(self._get_chain_info().keys())

    def resolve_chain(self, chain_id: str) -> str:
        """Map a requested chain identifier onto one present in this structure.

        mmCIF entries frequently use multi-character chain identifiers
        (``HHH``, ``LLL``) where the PDB-style name — and every annotation
        database built on it, SAbDab included — uses a single character
        (``H``, ``L``). An exact match always wins; otherwise a substitution is
        made only when exactly one candidate matches unambiguously, so the
        wrong chain is never chosen silently.

        Raises:
            ValueError: If nothing matches, or more than one thing does.
        """
        info = self._get_chain_info()
        if chain_id in info:
            return chain_id
        repeated = [c for c in info if c and set(c) == {chain_id}]
        prefixed = [c for c in info if c and c.startswith(chain_id)]
        for label, matches in (("repeated", repeated), ("prefix", prefixed)):
            if len(matches) == 1:
                logger.info(
                    "Chain '%s' not present in %s; using '%s' (%s match)",
                    chain_id, self._pdb_path.name, matches[0], label,
                )
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"Chain '{chain_id}' is ambiguous in {self._pdb_path.name}: "
                    f"{matches} all match. Name the chain exactly."
                )
        raise ValueError(
            f"Chain '{chain_id}' not found. Available: {self.chains()}"
        )

    def residue_range(self, chain_id: str) -> tuple[int, int]:
        """Return (min, max) residue number for a chain."""
        d = self._get_chain_info()[self.resolve_chain(chain_id)]
        return d["min_res"], d["max_res"]

    # ------------------------------------------------------------------ #
    # Fragment creation
    # ------------------------------------------------------------------ #

    def get_fragment(
        self,
        chain_id: str,
        res_start: int,
        res_end: int,
    ) -> Fragment:
        """Extract a backbone Fragment for a specific residue range.

        Args:
            chain_id: Chain identifier.
            res_start: First residue number (inclusive).
            res_end: Last residue number (inclusive).

        Returns:
            Fragment object ready for use as query_cdr or query_epitope.
        """
        return load_fragment_from_pdb(
            self._pdb_path, self.resolve_chain(chain_id), res_start, res_end,
            pdb_id=self._pdb_id, fmt=self._fmt,
        )

    def get_imgt_cdrs(
        self,
        chain_id: str,
        cdr_types: tuple[str, ...] | None = None,
        flank: int = 0,
    ) -> dict[str, Fragment]:
        """Auto-extract IMGT CDR loops from an IMGT-renumbered chain.

        Requires the PDB to be IMGT-renumbered (e.g. from SAbDab).
        Use flank > 0 to include flanking framework residues.

        Args:
            chain_id: Chain identifier (e.g. 'H' for heavy).
            cdr_types: CDR types to extract. Defaults to all three for the
                chain (H1/H2/H3 if chain_id is an H-type, else L1/L2/L3).
            flank: Framework residues to include on each side.

        Returns:
            Dict mapping CDR type (e.g. 'H3') to Fragment.
        """
        if cdr_types is None:
            # Infer from chain_id — try all six, keep those that extract
            cdr_types = tuple(IMGT_CDR_RANGES.keys())

        results: dict[str, Fragment] = {}
        for cdr_type, (r_start, r_end) in IMGT_CDR_RANGES.items():
            if cdr_type not in cdr_types:
                continue
            try:
                frag = load_fragment_from_pdb(
                    self._pdb_path,
                    chain_id,
                    max(1, r_start - flank),
                    r_end + flank,
                    pdb_id=self._pdb_id,
                    fmt=self._fmt,
                )
                frag.metadata["cdr_type"] = cdr_type
                frag.metadata["numbering_scheme"] = "imgt"
                results[cdr_type] = frag
                logger.debug(
                    "Extracted %s from chain %s: %d residues",
                    cdr_type, chain_id, frag.length,
                )
            except ValueError:
                logger.debug("CDR %s not found in chain %s", cdr_type, chain_id)

        if not results:
            logger.warning(
                "No CDR loops found on chain %s — is this an IMGT-renumbered structure?",
                chain_id,
            )
        return results

    def get_surface_patch(
        self,
        chain_id: str,
        center_res: int,
        radius_res: int = 8,
    ) -> Fragment:
        """Extract a residue window centred on a given residue.

        .. deprecated::
            Use :meth:`get_epitope_surface` instead. This selects by *residue
            number*, not by position in space, so the window can include
            residues nowhere near the CDR: on the benchmark complexes it
            returned 9 residues of which typically only 1-2 were within 8 Å of
            the loop. It also returns backbone atoms only, while contact
            distances need side chains.

        Args:
            chain_id: Chain identifier.
            center_res: Central residue number.
            radius_res: Half-width in residues (window = center ± radius_res).

        Returns:
            Fragment covering [center_res - radius_res, center_res + radius_res].
        """
        warnings.warn(
            "get_surface_patch selects by residue number, not by proximity, and "
            "returns backbone atoms only. Use get_epitope_surface() to define an "
            "epitope.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_fragment(
            chain_id,
            res_start=max(1, center_res - radius_res),
            res_end=center_res + radius_res,
        )

    def get_epitope_surface(
        self,
        chain_id: str,
        residues: list[int] | tuple[int, int] | None = None,
        near: Fragment | None = None,
        cutoff: float = 10.0,
    ) -> EpitopeSurface:
        """Build an epitope from the antigen chain, all atoms, side chains included.

        The epitope is normally **yours to choose**: pass the residues you want
        to target. It need not be in contact with the template's CDR — the point
        of a search is often to find a loop that *would* reach an epitope the
        template antibody does not engage.

        Args:
            chain_id: Antigen chain.
            residues: Which residues form the epitope, either an explicit list
                of residue numbers or an inclusive ``(first, last)`` range.
                Residues absent from the chain are skipped with a warning.
            near: Instead of naming residues, select everything within *cutoff*
                of this fragment. Convenient for reproducing a known interface,
                but it *presupposes* contact, so do not use it to define a
                target epitope.
            cutoff: Distance cutoff in Å, used only with *near*.

        Returns:
            The selected residues as an :class:`EpitopeSurface`.

        Raises:
            ValueError: If neither or both of *residues* and *near* are given,
                if the chain is absent, or if nothing was selected.
        """
        if (residues is None) == (near is None):
            raise ValueError("give exactly one of residues= or near=")

        chain_id = self.resolve_chain(chain_id)
        all_residues = extract_residue_atoms(self._pdb_path, chain_id, fmt=self._fmt)
        body = np.concatenate([c for _, _, c in all_residues], axis=0)

        if residues is not None:
            if isinstance(residues, tuple) and len(residues) == 2:
                lo, hi = residues
                wanted = set(range(int(lo), int(hi) + 1))
            else:
                wanted = {int(r) for r in residues}
            selected = [r for r in all_residues if r[0] in wanted]
            missing = wanted - {r[0] for r in selected}
            if missing:
                logger.warning(
                    "Epitope: %d requested residue(s) absent from chain %s: %s",
                    len(missing), chain_id,
                    ", ".join(str(m) for m in sorted(missing)[:10]),
                )
            if not selected:
                raise ValueError(
                    f"None of the requested residues exist in chain '{chain_id}' "
                    f"of {self._pdb_path.name}. Use describe() to list the "
                    "chains and their residue ranges."
                )
        else:
            from scipy.spatial import cKDTree

            assert near is not None
            tree = cKDTree(near.coords.reshape(-1, 3))
            selected = [
                r for r in all_residues
                if tree.query_ball_point(r[2], cutoff, return_length=True).any()
            ]
            if not selected:
                raise ValueError(
                    f"No residue of chain '{chain_id}' lies within {cutoff} Å of "
                    f"{near.pdb_id}:{near.chain} "
                    f"{near.residue_range[0]}-{near.residue_range[1]}. Check that "
                    "both come from the same structure and frame."
                )

        surface = EpitopeSurface.from_residues(chain_id, selected, body=body)
        logger.info(
            "Epitope: %d residues of chain %s (%d atoms)",
            surface.n_residues, chain_id, surface.n_atoms,
        )
        return surface

    def epitope_reach(self, cdr: Fragment, epitope: EpitopeSurface) -> float:
        """Closest approach, in Å, between a CDR site and an epitope.

        Worth checking before a search: if the epitope is far from the CDR site,
        no loop of that length can reach it and every candidate will score zero.
        A loop extends roughly 3 Å per residue from its anchors, so an epitope
        more than a few Å beyond the template's own loop is likely out of range.
        """
        from scipy.spatial import cKDTree

        return float(
            cKDTree(epitope.atom_coords).query(cdr.coords.reshape(-1, 3), k=1)[0].min()
        )

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _get_chain_info(self) -> dict[str, dict]:
        """Parse PDB and cache per-chain summary info."""
        if self._chain_info is not None:
            return self._chain_info

        try:
            from Bio import PDB as biopdb
            from Bio.SeqUtils import seq1
        except ImportError as e:
            raise ImportError("BioPython is required.") from e

        parser = (
            biopdb.MMCIFParser(QUIET=True) if self._fmt == "cif"
            else biopdb.PDBParser(QUIET=True)
        )
        structure = parser.get_structure("q", str(self._pdb_path))
        model = next(iter(structure))

        info: dict[str, dict] = {}
        for chain in model:
            chain_id = chain.get_id()
            resnums: list[int] = []
            seq_chars: list[str] = []
            for residue in chain:
                het, resseq, _ = residue.get_id()
                if het.strip():
                    continue
                resnums.append(resseq)
                seq_chars.append(seq1(residue.resname))
            if not resnums:
                continue
            info[chain_id] = {
                "n_residues": len(resnums),
                "min_res": min(resnums),
                "max_res": max(resnums),
                "sequence": "".join(seq_chars),
            }

        self._chain_info = info
        return info
