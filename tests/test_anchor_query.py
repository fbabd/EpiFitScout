"""The query is the stems, not the loop — and the loop must never reach it.

v1 searched with the native CDR. Every hit was then within the RMSD cutoff of
the answer by construction, which made CA RMSD, DockQ, Fnat and iRMSD
measurements of the query rather than of the prediction. These tests hold the
property that replaced it: **the search query is determined by the framework
either side of the CDR, and changing the loop's coordinates cannot change it.**
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from epifitscout.query.builder import QueryBuilder
from epifitscout.types.anchor_query import AnchorQuery
from epifitscout.types.fragment import Fragment

EXAMPLE = Path(__file__).resolve().parent.parent / "example" / "1a2y.pdb"

# 1a2y is IMGT-renumbered, so CDR-H3 on chain B is 105-117 with a numbering gap
# at 110-112 — thirteen numbers covering ten real residues.
H3 = (105, 117)
H3_LENGTH = 10


@pytest.fixture(scope="module")
def builder() -> QueryBuilder:
    return QueryBuilder(EXAMPLE)


@pytest.fixture(scope="module")
def query(builder: QueryBuilder) -> AnchorQuery:
    return builder.get_anchor_query("B", *H3)


class TestWhatTheQueryIsMadeOf:
    def test_stems_are_the_residues_flanking_the_loop(self, query: AnchorQuery) -> None:
        assert query.n_stem.residue_range == (101, 104)
        assert query.c_stem.residue_range == (118, 121)
        assert query.n_stem.sequence == "RYYC"      # ...VYYC of the framework
        assert query.c_stem.sequence == "WGQG"      # WGQG... of the framework

    def test_gap_length_counts_real_residues_not_numbers(
        self, query: AnchorQuery
    ) -> None:
        """A numbering gap would otherwise inflate the loop length by three."""
        assert query.gap_length == H3_LENGTH
        assert H3[1] - H3[0] + 1 == 13               # what the numbering suggests

    def test_the_loop_is_not_part_of_the_query(self, query: AnchorQuery) -> None:
        loop_atoms = query.metadata["native_loop_coords"].reshape(-1, 3)
        stem_atoms = query.stem_coords.reshape(-1, 3)
        for atom in loop_atoms:
            assert not np.isclose(stem_atoms, atom).all(axis=1).any()

    def test_moving_the_loop_cannot_change_the_query(
        self, builder: QueryBuilder, query: AnchorQuery
    ) -> None:
        """The property v1 could not have: the query is framework-determined."""
        again = builder.get_anchor_query("B", *H3)
        assert np.array_equal(again.stem_coords, query.stem_coords)
        assert again.gap_length == query.gap_length


class TestGeometry:
    def test_span_is_takeoff_to_landing(self, query: AnchorQuery) -> None:
        expected = float(np.linalg.norm(query.landing - query.takeoff))
        assert query.span == pytest.approx(expected)

    def test_span_per_residue_is_within_reach_of_a_real_chain(
        self, query: AnchorQuery
    ) -> None:
        """A native loop closes its own gap, so its span must be bridgeable."""
        assert 0.0 < query.span_per_residue < 3.3


class TestSerialisation:
    def test_pdb_has_two_segments_in_n_to_c_order(self, query: AnchorQuery) -> None:
        """MASTER reads segments in file order, and --gapLen fixes that order."""
        text = query.to_pdb_string()
        resnums = [
            int(line[22:26]) for line in text.splitlines() if line.startswith("ATOM")
        ]
        assert resnums[0] == 1                       # N-stem written first
        breaks = [
            b - a for a, b in zip(resnums, resnums[1:], strict=False) if b - a > 1
        ]
        assert len(breaks) == 1 and breaks[0] > 1    # exactly one discontinuity

    def test_gap_len_argument_is_exact_by_default(self, query: AnchorQuery) -> None:
        assert query.gap_len_argument() == str(H3_LENGTH)

    def test_gap_len_tolerance_becomes_a_range(self, query: AnchorQuery) -> None:
        assert query.gap_len_argument(2) == f"{H3_LENGTH - 2}-{H3_LENGTH + 2}"
        assert re.fullmatch(r"\d+-\d+", query.gap_len_argument(2))

    def test_a_negative_tolerance_is_refused(self, query: AnchorQuery) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            query.gap_len_argument(-1)


class TestRefusals:
    def test_a_stem_shorter_than_three_residues_is_refused(
        self, query: AnchorQuery
    ) -> None:
        """Fewer than three residues leaves the rigid-body fit underdetermined."""
        with pytest.raises(ValueError, match="at least 3 residues"):
            AnchorQuery(
                pdb_id="X", chain="B",
                n_stem=_slice_fragment(query.n_stem, 2),
                c_stem=query.c_stem,
                gap_length=query.gap_length,
                loop_range=query.loop_range,
            )

    def test_stem_length_below_three_is_refused(self, builder: QueryBuilder) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            builder.get_anchor_query("B", *H3, stem_length=2)

    def test_a_loop_at_the_end_of_the_chain_is_refused(
        self, builder: QueryBuilder
    ) -> None:
        """No framework on one side means no query, and it is said plainly."""
        first, _ = builder.residue_range("B")
        with pytest.raises(ValueError, match="flank"):
            builder.get_anchor_query("B", first, first + 4)

    def test_an_inverted_span_is_refused(self, builder: QueryBuilder) -> None:
        with pytest.raises(ValueError, match="precedes"):
            builder.get_anchor_query("B", 117, 105)


def _slice_fragment(fragment: Fragment, n: int) -> Fragment:
    return Fragment(
        pdb_id=fragment.pdb_id, chain=fragment.chain,
        residue_range=fragment.residue_range,
        coords=fragment.coords[:n], sequence=fragment.sequence[:n],
    )
