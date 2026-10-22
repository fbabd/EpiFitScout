"""QueryBuilder: read a complex and build the stem query for a CDR site.

Example::

    from epifitscout.query.builder import QueryBuilder

    qb = QueryBuilder(Path("my_complex.pdb"))
    qb.describe()                               # chains and residue ranges
    query = qb.get_anchor_query("H", 95, 102)    # the stems, not the loop
    print(query.describe())

:meth:`QueryBuilder.get_anchor_query` is the one that matters;
:meth:`get_fragment` and :meth:`get_imgt_cdrs` are there to find the span to
pass it when the CDR boundaries are not already known.
"""

from __future__ import annotations

import logging
from pathlib import Path

from epifitscout.types.anchor_query import AnchorQuery
from epifitscout.types.fragment import Fragment
from epifitscout.utils.io import detect_format, extract_backbone_coords_by_range

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

#: Framework residues taken on each side of a CDR to form the search query.
#: Four per side gives 12 CA atoms for the rigid-body fit and stays within the
#: conserved framework flanking CDR-H3 (…VYYC**AR** / **WGQG**TLV…).
DEFAULT_STEM_LENGTH = 4


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

    def get_anchor_query(
        self,
        chain_id: str,
        loop_start: int,
        loop_end: int,
        stem_length: int = DEFAULT_STEM_LENGTH,
    ) -> AnchorQuery:
        """Build the stem query for a CDR site: the search query v2 uses.

        The loop itself is read only to count its residues — its coordinates
        never reach the query, which is what makes the retrieved loops a
        prediction rather than a restatement of the native one. See
        :mod:`epifitscout.types.anchor_query`.

        Insertion codes are handled by position rather than by number: antibody
        CDRs are routinely numbered 111, 111A, 111B, so the residue *count* of a
        span is not ``loop_end - loop_start + 1`` and the stems have to be taken
        as "the last *n* residues before the loop", not "residues
        ``loop_start - n`` to ``loop_start - 1``".

        Args:
            chain_id: Antibody chain carrying the CDR.
            loop_start: First residue number of the loop, inclusive.
            loop_end: Last residue number of the loop, inclusive.
            stem_length: Framework residues to take on each side. Four is the
                default: enough to determine the rigid-body fit well (12 CA
                atoms over both stems) while staying inside the conserved
                framework either side of CDR-H3.

        Returns:
            An :class:`~epifitscout.types.anchor_query.AnchorQuery`.

        Raises:
            ValueError: If the loop is empty, or either flank is too short to
                supply ``stem_length`` residues.
        """
        if stem_length < 3:
            raise ValueError(f"stem_length must be at least 3, got {stem_length}")
        if loop_end < loop_start:
            raise ValueError(f"loop_end {loop_end} precedes loop_start {loop_start}")

        chain = self.resolve_chain(chain_id)
        loop_coords, loop_resnums, loop_seq = extract_backbone_coords_by_range(
            self._pdb_path, chain, loop_start, loop_end, fmt=self._fmt
        )
        gap_length = len(loop_resnums)

        # Over-reach the flanks, then take the stem residues by position: an
        # insertion code inside the window would otherwise shift the count.
        reach = stem_length + 4
        n_flank = self._slice_flank(
            chain, loop_start - reach, loop_start - 1, stem_length, side="N"
        )
        c_flank = self._slice_flank(
            chain, loop_end + 1, loop_end + reach, stem_length, side="C"
        )

        query = AnchorQuery(
            pdb_id=self._pdb_id,
            chain=chain,
            n_stem=n_flank,
            c_stem=c_flank,
            gap_length=gap_length,
            loop_range=(loop_resnums[0], loop_resnums[-1]),
            metadata={
                "pdb_file": str(self._pdb_path),
                "stem_length": stem_length,
                "native_loop_sequence": loop_seq,
                "native_loop_coords": loop_coords,
            },
        )
        logger.info("Anchor query: %s", query.describe())
        if query.span_per_residue > 3.3:
            logger.warning(
                "Span %.1f Å over %d residues (%.2f Å/res) is at or beyond the "
                "reach of an extended chain; check the loop boundaries.",
                query.span, gap_length, query.span_per_residue,
            )
        return query

    def _slice_flank(
        self, chain: str, lo: int, hi: int, want: int, side: str
    ) -> Fragment:
        """Take *want* residues from a flanking window, adjacent to the loop."""
        try:
            coords, resnums, seq = extract_backbone_coords_by_range(
                self._pdb_path, chain, lo, hi, fmt=self._fmt
            )
        except ValueError:
            resnums = []
        if len(resnums) < want:
            raise ValueError(
                f"{side}-side flank of chain {chain} holds {len(resnums)} "
                f"residues in {lo}-{hi}, fewer than the {want} the stem needs. "
                "The CDR is too close to the end of the modelled chain, or the "
                "chain has gaps there."
            )
        sl = slice(-want, None) if side == "N" else slice(0, want)
        return Fragment(
            pdb_id=self._pdb_id,
            chain=chain,
            residue_range=(resnums[sl][0], resnums[sl][-1]),
            coords=coords[sl],
            sequence=seq[sl],
            metadata={"source": "query_stem", "side": side},
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
