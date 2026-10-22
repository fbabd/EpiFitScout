"""Turn MASTER match lines into placed graft candidates.

MASTER v1.5/1.6 match line format::

    <rmsd>  <path/to/target.pds>  [(s1,e1), (s2,e2), ...]

One ``(start,end)`` pair per query segment, in the order the segments appear in
the query file. The indices are **0-based positions in the target chain**, not
PDB residue numbers, so every slice here is positional; the database's own
numbering is never trusted. PDS files live in the ``pds/`` subdirectory of the
database and the PDB coordinates in the sibling ``pdb/``, named
``{PDBID}_{CHAIN}_{TYPE}.pdb``.

Where the graft transform comes from
------------------------------------
A match line says *that* a database chain carries stems like the query's, not
*where* that chain sits: its coordinates are in its own crystal frame, often
tens of ångströms away. Placing it is the whole of stage 2, and there is exactly
one legitimate way to do it.

The retrieved loop will be bonded into the template's framework at both ends,
and the template is already in the same frame as the antigen. So the transform
that places the loop is the one that lays the donor's **stems** onto the
template's stems — nothing else is free to choose. It is computed here by
least-squares fit over both stems' backbone atoms (N, CA, C, O, both sides
weighted equally), which for three or more non-collinear atoms has a unique
optimum, and applied as a rigid body so the loop's own conformation is
untouched.

Two things this deliberately does **not** do:

*Fit on the loop.* That is v1's transform, and it requires the native loop —
the very thing a benchmark asks the method to predict. Every structural metric
downstream inherits the circularity.

*Fit to maximise epitope contact.* A loop is not free to rotate once attached to
a framework, so a placement chosen to sit well against the antigen describes a
structure that cannot exist. Complementarity is measured at the placement the
graft dictates, not searched over.

MASTER reports RMSD over CA atoms by default, so the CA-only fit is computed as
well and checked against the reported number. It is a consistency check — that
the residues reloaded from the PDB are the ones MASTER aligned — not the
transform.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from epifitscout.types.anchor_query import AnchorQuery
from epifitscout.types.candidate import GraftCandidate
from epifitscout.types.fragment import Fragment
from epifitscout.types.superposition import Superposition
from epifitscout.utils.geometry import kabsch_superpose
from epifitscout.utils.io import extract_backbone_coords_by_range

logger = logging.getLogger(__name__)

__all__ = [
    "MASTERParseError",
    "ParseStats",
    "parse_anchor_matches",
    "parse_match_file",
]

#: Length of the sequence stretch used to recognise a loop as "the same" as an
#: excluded one. Eight residues of a CDR-H3 is long enough to be specific to one
#: antibody.
LEAKAGE_KMER = 8

#: Identity to an excluded loop, over the same length, above which a candidate
#: counts as that loop. A shared stretch alone is not enough, and the gap is
#: easy to walk into: searching 1a2y's ten-residue CDR-H3 (ARERDYRLDY) with the
#: exact sequence excluded still returns 1KIQ's ARERDFRLDY — the same loop with
#: one substitution, from an affinity-maturation variant of the same antibody —
#: because a single central mutation breaks every 8-mer. At 0.8, a ten-residue
#: loop may differ from the native in at most two positions before it is
#: refused. Chance identity that high is ~10⁻⁹ for a 10-mer, so this excludes
#: variants of the answer without excluding genuinely different loops.
LEAKAGE_IDENTITY = 0.8

#: Largest tolerated gap, in Å, between the RMSD MASTER reported and the CA RMSD
#: obtained by superposing the reloaded coordinates. Observed agreement is
#: ~10⁻⁵ Å (MASTER prints five decimals), so anything beyond this means the
#: residues reloaded from the PDB are not the ones MASTER aligned.
_CONSISTENCY_TOL = 0.01

# e.g.: "  0.87428 /abs/path/to/pds/4KUZ_H_H.pds [(92,95), (106,109)]"
_MATCH_RE = re.compile(r"^\s*(?P<rmsd>\S+)\s+(?P<path>\S+)\s+\[(?P<segments>.*)\]\s*$")
_SEGMENT_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


class MASTERParseError(ValueError):
    """Raised when the MASTER output file cannot be parsed."""


@dataclass
class ParseStats:
    """Why candidates were dropped, so a thin result set can be explained.

    Each field is one reason a match did not become a candidate. Kept because a
    thin result is a finding: v1's coverage collapse on long CDR-H3 was only
    visible because the counts were retained, and "nothing bridges this gap" and
    "the entries could not be read" need telling apart.
    """

    lines: int = 0                #: match lines seen
    used: int = 0                 #: candidates produced
    over_cutoff: int = 0          #: RMSD above the threshold
    segment_count: int = 0        #: not a two-segment match
    segment_order: int = 0        #: second segment precedes the first
    missing_pdb: int = 0          #: no coordinate file for the match
    excluded: int = 0             #: donor on the exclusion list
    unreadable: int = 0           #: coordinates could not be loaded or sliced
    stem_length: int = 0          #: matched stems are the wrong length
    loop_length: int = 0          #: gap is not the length the graft needs
    excluded_sequence: int = 0    #: retrieved loop shares a stretch with an
                                  #: excluded sequence
    rmsd_disagreement: int = 0    #: reloaded residues are not the aligned ones

    def summary(self) -> str:
        dropped = {
            field: getattr(self, field)
            for field in (
                "over_cutoff", "segment_count", "segment_order", "missing_pdb",
                "excluded", "excluded_sequence", "unreadable", "stem_length",
                "loop_length", "rmsd_disagreement",
            )
            if getattr(self, field)
        }
        detail = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
        return (
            f"{self.used} candidates from {self.lines} match lines"
            + (f"; dropped: {detail}" if detail else "")
        )


# --------------------------------------------------------------------------- #
# Anchor/gap matches — the v2 path
# --------------------------------------------------------------------------- #


def parse_anchor_matches(
    match_path: Path,
    query: AnchorQuery,
    rmsd_threshold: float = 2.0,
    max_hits: int = 500,
    exclude_donors: frozenset[str] | set[str] | None = None,
    exclude_sequences: frozenset[str] | set[str] | None = None,
    stats: ParseStats | None = None,
) -> list[GraftCandidate]:
    """Read a two-segment match file into placed :class:`GraftCandidate` objects.

    Each match contributes the residues *between* its two matched stems — the
    candidate loop — together with the transform that lays the donor's stems
    onto the template's. The loop itself plays no part in that fit.

    Args:
        match_path: Path to MASTER's ``--matchOut`` file.
        query: The :class:`~epifitscout.types.anchor_query.AnchorQuery` that was
            searched. Required: without the template's stems a hit cannot be
            placed, and a loop in its donor's frame means nothing.
        rmsd_threshold: Discard matches whose stem RMSD exceeds this.
        max_hits: Stop after this many usable candidates. Matches arrive sorted
            by RMSD, so this keeps the best.
        exclude_donors: Upper-case PDB IDs to refuse, whatever they score. The
            cheaper place to exclude a donor is the target list handed to
            MASTER, which never searches it; this is the backstop for cases
            where the list could not be filtered.
        exclude_sequences: Refuse any retrieved loop that is one of these in all
            but name — sharing an ``LEAKAGE_KMER``-residue stretch with it, or
            at least ``LEAKAGE_IDENTITY`` identical to it over the same length.
            Normally the native loop. Excluding by PDB ID alone does not do
            this, and the gap it leaves is large: searching for 1a2y's CDR-H3
            with only the 1A2Y entry excluded returns that exact loop from five
            other entries of the same antibody, and excluding the sequence
            exactly still returns a one-substitution variant of it. A benchmark
            must pass this, or filter the target list by sequence before
            searching.
        stats: Optional :class:`ParseStats` to record drop reasons into.

    Returns:
        Candidates sorted by stem RMSD ascending.
    """
    st = stats if stats is not None else ParseStats()
    refuse = {d.upper() for d in (exclude_donors or ())}
    refuse_seq = tuple(s.upper() for s in (exclude_sequences or ()))
    candidates: list[GraftCandidate] = []

    query_bb = _stem_backbone(query.stem_coords)
    query_ca = query.stem_ca
    want = (query.n_stem.length, query.c_stem.length)

    for line in _match_lines(match_path):
        st.lines += 1
        parsed = _parse_line(line)
        if parsed is None:
            continue
        rmsd_val, pds_path, segments = parsed

        if rmsd_val > rmsd_threshold:
            st.over_cutoff += 1
            continue
        if len(segments) != 2:
            st.segment_count += 1
            continue

        (s1, e1), (s2, e2) = segments
        if s2 <= e1:
            # Only reachable without --gapLen, where MASTER permits any
            # ordering of disjoint segments; such a match has no graftable gap.
            st.segment_order += 1
            continue

        pdb_path = _pdb_path_for_pds(pds_path)
        if pdb_path is None or not pdb_path.exists():
            st.missing_pdb += 1
            continue

        pdb_id, chain, chain_type = _identity(pdb_path)
        if pdb_id.upper() in refuse:
            st.excluded += 1
            continue

        try:
            whole = _chain_backbone(pdb_path)
        except Exception as exc:                       # unreadable entry
            logger.debug("Cannot read %s: %s", pdb_path.name, exc)
            st.unreadable += 1
            continue

        if (e1 - s1 + 1, e2 - s2 + 1) != want:
            st.stem_length += 1
            continue
        if s2 - e1 - 1 != query.gap_length:
            st.loop_length += 1
            continue
        if e2 >= len(whole[1]):
            st.unreadable += 1
            continue

        n_stem = _slice(whole, pdb_id, chain, chain_type, s1, e1, "donor_stem_N")
        c_stem = _slice(whole, pdb_id, chain, chain_type, s2, e2, "donor_stem_C")
        loop = _slice(whole, pdb_id, chain, chain_type, e1 + 1, s2 - 1, "donor_loop")

        if refuse_seq and _is_leaked(loop.sequence, refuse_seq):
            st.excluded_sequence += 1
            continue

        hit_coords = np.concatenate([n_stem.coords, c_stem.coords], axis=0)

        # The transform in use: full-backbone least squares over both stems.
        moved_bb, rotation, translation = kabsch_superpose(
            _stem_backbone(hit_coords), query_bb
        )
        rmsd_bb = _rmsd(moved_bb, query_bb)

        # Independent CA-only fit, solely to reproduce MASTER's number.
        moved_ca, _, _ = kabsch_superpose(hit_coords[:, 1, :], query_ca)
        rmsd_ca = _rmsd(moved_ca, query_ca)
        if abs(rmsd_ca - rmsd_val) > _CONSISTENCY_TOL:
            logger.debug(
                "%s: CA fit %.5f disagrees with MASTER %.5f — reloaded residues "
                "are not the aligned ones", pdb_path.name, rmsd_ca, rmsd_val,
            )
            st.rmsd_disagreement += 1
            continue

        candidates.append(
            GraftCandidate(
                loop=loop,
                n_stem=n_stem,
                c_stem=c_stem,
                superposition=Superposition(rotation=rotation, translation=translation),
                master_rmsd=rmsd_val,
                stem_rmsd_ca=rmsd_ca,
                stem_rmsd_bb=rmsd_bb,
                metadata={
                    "donor_pdb_path": str(pdb_path),
                    "chain_type": chain_type,
                    "segments": ((s1, e1), (s2, e2)),
                },
            )
        )
        st.used += 1
        if max_hits > 0 and len(candidates) >= max_hits:
            break

    candidates.sort(key=lambda c: c.stem_rmsd_ca)
    logger.info("Parsed anchor matches: %s", st.summary())
    return candidates


# --------------------------------------------------------------------------- #
# Single-segment matches — the v1 path, kept for the ablation
# --------------------------------------------------------------------------- #


def parse_match_file(
    match_path: Path,
    query: Fragment,
    rmsd_threshold: float = 2.0,
    max_hits: int = 500,
) -> list[tuple[Fragment, Superposition, float]]:
    """Read a single-segment match file, fitting each hit onto the query loop.

    This is v1's behaviour and it is **only** appropriate when the query is a
    loop whose coordinates the method is allowed to know — an ablation, or a
    search for alternatives to a loop already in hand. Used with a native CDR it
    makes every structural metric a measurement of the query. Prefer
    :func:`parse_anchor_matches`.

    Returns:
        ``(fragment, superposition, master_rmsd)`` sorted by RMSD ascending.
    """
    results: list[tuple[Fragment, Superposition, float]] = []
    n_dropped = 0

    for line in _match_lines(match_path):
        parsed = _parse_line(line)
        if parsed is None:
            continue
        rmsd_val, pds_path, segments = parsed
        if rmsd_val > rmsd_threshold or len(segments) != 1:
            continue

        pdb_path = _pdb_path_for_pds(pds_path)
        if pdb_path is None or not pdb_path.exists():
            continue

        seg_start, seg_end = segments[0]
        pdb_id, chain, chain_type = _identity(pdb_path)
        try:
            whole = _chain_backbone(pdb_path)
        except Exception:
            continue
        if seg_end >= len(whole[1]):
            continue

        fragment = _slice(
            whole, pdb_id, chain, chain_type, seg_start, seg_end, "donor_loop"
        )
        if fragment.length != query.length:
            n_dropped += 1
            continue

        moved, rotation, translation = kabsch_superpose(
            fragment.ca_coords, query.ca_coords
        )
        achieved = _rmsd(moved, query.ca_coords)
        if abs(achieved - rmsd_val) > _CONSISTENCY_TOL:
            n_dropped += 1
            continue

        fragment.metadata["superposed_rmsd"] = achieved
        results.append(
            (fragment, Superposition(rotation=rotation, translation=translation), rmsd_val)
        )
        if max_hits > 0 and len(results) >= max_hits:
            break

    if n_dropped:
        logger.warning("Dropped %d hit(s) that could not be placed", n_dropped)
    results.sort(key=lambda r: r[2])
    logger.info("Parsed %d single-segment hits", len(results))
    return results


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _match_lines(match_path: Path) -> list[str]:
    if not match_path.exists():
        raise FileNotFoundError(f"MASTER match file not found: {match_path}")
    return match_path.read_text().splitlines()


def _parse_line(
    line: str,
) -> tuple[float, Path, list[tuple[int, int]]] | None:
    """``(rmsd, pds_path, segments)`` or None when the line is not a match."""
    m = _MATCH_RE.match(line)
    if m is None:
        return None
    try:
        rmsd_val = float(m.group("rmsd"))
    except ValueError:
        return None
    segments = [
        (int(a), int(b)) for a, b in _SEGMENT_RE.findall(m.group("segments"))
    ]
    if not segments:
        return None
    return rmsd_val, Path(m.group("path")), segments


def _is_leaked(
    sequence: str,
    excluded: tuple[str, ...],
    k: int = LEAKAGE_KMER,
    identity: float = LEAKAGE_IDENTITY,
) -> bool:
    """Whether *sequence* is one of the *excluded* loops in all but name.

    Two criteria, because each misses what the other catches. A shared
    *k*-residue stretch catches a loop embedded in a longer one, or a partial
    match. Per-position identity over the same length catches a point mutant,
    which a *k*-mer test misses entirely — a single substitution in the middle
    of a ten-residue loop breaks every 8-mer it has.

    A sequence shorter than *k* has no *k*-mer, so only the identity test
    applies to it.
    """
    query = sequence.upper()
    if not query:
        return False
    for other in excluded:
        if len(query) >= k and len(other) >= k and any(
            query[i:i + k] in other for i in range(len(query) - k + 1)
        ):
            return True
        if len(query) == len(other):
            same = sum(1 for a, b in zip(query, other, strict=True) if a == b)
            if same / len(query) >= identity:
                return True
    return False


def _identity(pdb_path: Path) -> tuple[str, str, str]:
    """``(pdb_id, chain, chain_type)`` from a ``{PDBID}_{CHAIN}_{TYPE}.pdb`` name."""
    parts = pdb_path.stem.split("_")
    pdb_id = parts[0] if parts else pdb_path.stem
    chain = parts[1] if len(parts) >= 2 else "A"
    chain_type = parts[2] if len(parts) >= 3 else ""
    return pdb_id, chain, chain_type


@lru_cache(maxsize=4096)
def _chain_backbone(
    pdb_path: Path,
) -> tuple[np.ndarray, tuple[int, ...], str]:
    """Whole-chain backbone, cached: consecutive matches often share a donor.

    Returns ``(coords, residue_numbers, sequence)``. Slicing is positional
    everywhere downstream, so the numbering is carried for labelling only.
    """
    _, chain, _ = _identity(pdb_path)
    coords, resnums, sequence = extract_backbone_coords_by_range(
        pdb_path, chain, res_start=1, res_end=10**9
    )
    return coords, tuple(int(r) for r in resnums), sequence


def _slice(
    whole: tuple[np.ndarray, tuple[int, ...], str],
    pdb_id: str,
    chain: str,
    chain_type: str,
    start: int,
    end: int,
    role: str,
) -> Fragment:
    """Positional slice ``[start, end]`` of a loaded chain, as a Fragment."""
    coords, resnums, sequence = whole
    sl = slice(start, end + 1)
    piece, nums, seq = coords[sl], resnums[sl], sequence[sl]
    if len(nums) == 0:
        raise ValueError(f"empty slice [{start}:{end}] of {pdb_id}_{chain}")
    return Fragment(
        pdb_id=pdb_id,
        chain=chain,
        residue_range=(nums[0], nums[-1]),
        coords=np.asarray(piece, dtype=np.float64),
        sequence=seq,
        metadata={
            "chain_type": chain_type,
            "role": role,
            "seg_start": start,
            "seg_end": end,
        },
    )


def _stem_backbone(coords: np.ndarray) -> np.ndarray:
    """``(4R, 3)`` flattened N, CA, C, O atoms of a ``(R, 4, 3)`` stem set."""
    return coords.reshape(-1, 3)


def _rmsd(moved: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(((moved - reference) ** 2).sum() / len(reference)))


def _pdb_path_for_pds(pds_path: Path) -> Path | None:
    """Resolve the PDB coordinate file from a PDS path.

    ``.../db/pds/4KUZ_H_H.pds`` → ``.../db/pdb/4KUZ_H_H.pdb``. Falls back to the
    same directory for databases that keep both side by side.
    """
    candidate = pds_path.parent.parent / "pdb" / f"{pds_path.stem}.pdb"
    if candidate.exists():
        return candidate
    sibling = pds_path.with_suffix(".pdb")
    return sibling if sibling.exists() else None
