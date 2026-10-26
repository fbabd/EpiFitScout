"""Recognising a retrieved loop as the native one in all but name.

Leakage in a retrieval method is not only "the search returned the target's own
PDB entry". The same antibody is deposited many times, and affinity-maturation
series put near-identical variants of one CDR in separate entries. Both were
found by running the bundled example rather than by reasoning about it:

* with only the `1A2Y` entry excluded, the top five hits for its CDR-H3 were
  that exact loop, from 1A7O, 1A7P, 1VFA, 1A7R and 1KIR;
* with the sequence `ARERDYRLDY` also excluded, the top hit became 1KIQ's
  `ARERDFRLDY` — the same loop with one substitution, which shares no 8-mer
  with it because the mutation sits in the middle.

Hence two criteria. These tests pin both, and pin that a genuinely different
loop of the same length is still allowed through.
"""

from __future__ import annotations

import pytest

from epifitscout.master.parser import (
    LEAKAGE_IDENTITY,
    LEAKAGE_KMER,
    _is_leaked,
)

NATIVE = "ARERDYRLDY"          # 1a2y CDR-H3, ten residues


class TestTheNativeLoopAndItsRelatives:
    def test_the_loop_itself(self) -> None:
        assert _is_leaked(NATIVE, (NATIVE,))

    def test_a_point_mutant(self) -> None:
        """What an 8-mer test alone misses: 1KIQ, a real case."""
        variant = "ARERDFRLDY"
        assert not any(
            variant[i:i + LEAKAGE_KMER] in NATIVE
            for i in range(len(variant) - LEAKAGE_KMER + 1)
        )
        assert _is_leaked(variant, (NATIVE,))

    def test_a_double_mutant_at_the_threshold(self) -> None:
        assert _is_leaked("ARERDFRLDA", (NATIVE,))      # 8/10 identical

    def test_the_loop_inside_a_longer_sequence(self) -> None:
        assert _is_leaked(NATIVE, ("QQ" + NATIVE + "WG",))


class TestDifferentLoopsAreAllowed:
    @pytest.mark.parametrize(
        "sequence", ["ARGLYNDYTV", "SRGGAGIMAY", "ARDSGYYLDY", "EGHYGYDGYA"]
    )
    def test_a_real_but_different_cdr(self, sequence: str) -> None:
        assert not _is_leaked(sequence, (NATIVE,))

    def test_a_loop_of_another_length_is_judged_only_by_shared_stretches(
        self,
    ) -> None:
        """Identity needs equal lengths; a longer loop is caught by its stretches."""
        assert not _is_leaked("ARERD", (NATIVE,))        # too short for a k-mer
        assert _is_leaked(NATIVE + "WGQ", (NATIVE,))     # shares every 8-mer

    def test_chance_identity_is_not_reachable(self) -> None:
        """The threshold is far above what unrelated loops hit by accident."""
        assert LEAKAGE_IDENTITY >= 0.8
        unrelated = ["ARGLYNDYTV", "SRGGAGIMAY", "ARDSGYYLDY", "EGHYGYDGYA"]
        for sequence in unrelated:
            same = sum(1 for a, b in zip(sequence, NATIVE, strict=True) if a == b)
            assert same / len(NATIVE) < LEAKAGE_IDENTITY


class TestEdges:
    def test_an_empty_sequence_is_not_leaked(self) -> None:
        assert not _is_leaked("", (NATIVE,))

    def test_nothing_excluded_means_nothing_refused(self) -> None:
        assert not _is_leaked(NATIVE, ())

    def test_comparison_ignores_case(self) -> None:
        assert _is_leaked(NATIVE.lower(), (NATIVE,))
