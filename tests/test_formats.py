"""Reading mmCIF as well as PDB, and reconciling chain identifiers.

RCSB serves an entry as mmCIF only when it is too large for the PDB format —
over 62 chains or 99,999 atoms — which is exactly the large complexes an
antibody database wants. Of the entries cached during the original database
build, 173 were mmCIF-only.

Handing an mmCIF file to BioPython's ``PDBParser`` does not fail cleanly: it
reads the columns as PDB fields and raises something unrelated, such as
``invalid literal for int() with base 10: 'ER'``. Detecting the format from the
suffix avoids that whole class of confusion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import epifitscout
from epifitscout.utils.io import detect_format, extract_residue_atoms, select_chain

CIF_SOURCES = [
    Path("../EpiFitScout/data/rcsb_cache"),
    Path("database_builder/rcsb_cache"),
]


def _a_cif() -> Path:
    for directory in CIF_SOURCES:
        if directory.is_dir():
            for path in sorted(directory.glob("*.cif")):
                return path
    pytest.skip("no cached mmCIF structure available")
    raise AssertionError


class FakeChain:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_id(self) -> str:
        return self._name


class FakeModel:
    """Minimal stand-in for a parsed BioPython model."""

    def __init__(self, names: list[str]) -> None:
        self._chains = {n: FakeChain(n) for n in names}

    def __iter__(self):
        return iter(self._chains.values())

    def __getitem__(self, key: str) -> FakeChain:
        return self._chains[key]


class TestDetectFormat:
    @pytest.mark.parametrize("name,expected", [
        ("1abc.pdb", "pdb"), ("1abc.ent", "pdb"), ("1ABC.PDB", "pdb"),
        ("1abc.cif", "cif"), ("1abc.mmcif", "cif"), ("1ABC.CIF", "cif"),
    ])
    def test_suffix_decides(self, name: str, expected: str) -> None:
        assert detect_format(Path("/tmp") / name) == expected

    def test_accepts_a_string(self) -> None:
        assert detect_format("x/y/8vqy.cif") == "cif"


class TestQueryBuilderReadsCif:
    def test_chains_and_fragment(self) -> None:
        path = _a_cif()
        qb = epifitscout.QueryBuilder(path)
        chains = qb.chains()
        assert chains, "an mmCIF entry must yield chains"

        chain = chains[0]
        lo, hi = qb.residue_range(chain)
        frag = qb.get_fragment(chain, lo, hi)
        assert frag.length > 10
        assert len(frag.sequence) == frag.length
        assert frag.coords.shape == (frag.length, 4, 3)
        # a PDB-only reader raises ValueError here; reaching this point is the test
        assert set(frag.sequence) - set("ACDEFGHIKLMNPQRSTVWYX") == set()

    def test_all_atom_read_recovers_side_chains(self) -> None:
        path = _a_cif()
        chain = epifitscout.QueryBuilder(path).chains()[0]
        residues = extract_residue_atoms(path, chain, fmt="cif")
        assert residues
        per_residue = sum(len(c) for _, _, c in residues) / len(residues)
        assert per_residue > 4.5, (
            f"only {per_residue:.1f} atoms per residue — side chains are missing"
        )

    def test_autodetect_matches_explicit_format(self) -> None:
        path = _a_cif()
        chain = epifitscout.QueryBuilder(path).chains()[0]
        assert len(extract_residue_atoms(path, chain)) == len(
            extract_residue_atoms(path, chain, fmt="cif")
        )

    def test_epitope_surface_from_a_cif_entry(self) -> None:
        path = _a_cif()
        qb = epifitscout.QueryBuilder(path)
        chains = qb.chains()
        if len(chains) < 2:
            pytest.skip("need at least two chains")
        lo, hi = qb.residue_range(chains[0])
        probe = qb.get_fragment(chains[0], lo, min(hi, lo + 9))
        for other in chains[1:]:
            try:
                epitope = qb.get_epitope_surface(other, near=probe, cutoff=12.0)
            except ValueError:
                continue
            assert epitope.n_residues > 0
            assert epitope.atom_coords.shape[1] == 3
            return
        pytest.skip("no chain lies near the probe fragment in this entry")


class TestChainIdentifierMismatch:
    """mmCIF multi-character chain IDs versus the single-character names used by
    SAbDab and every other PDB-style annotation.

    This cost 30 of 173 mmCIF-only entries (17%) in the original database build:
    SAbDab asked for chain ``H``, the file contained ``HHH``, and the entry was
    silently dropped.
    """

    def test_repeated_identifier_resolves(self) -> None:
        for pid in ("6srv", "6ss2", "6ss0"):
            path = Path("../EpiFitScout/data/rcsb_cache") / f"{pid}.cif"
            if not path.is_file():
                continue
            qb = epifitscout.QueryBuilder(path)
            assert qb.resolve_chain("H") == "HHH"
            lo, hi = qb.residue_range("H")
            assert qb.get_fragment("H", lo, min(hi, lo + 9)).length > 0
            return
        pytest.skip("none of the multi-character-chain entries are cached")

    def test_exact_match_always_wins(self, tmp_path: Path) -> None:
        """A real 'H' must never be displaced by a prefix match on 'HHH'."""
        model = FakeModel(["H", "HHH"])
        assert select_chain(model, "H", tmp_path / "x.cif").get_id() == "H"

    def test_ambiguous_match_is_refused(self, tmp_path: Path) -> None:
        """Two candidates must raise rather than pick one arbitrarily."""
        with pytest.raises(ValueError, match="ambiguous"):
            select_chain(FakeModel(["HA", "HB"]), "H", tmp_path / "x.cif")

    def test_missing_chain_lists_what_is_available(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Available"):
            select_chain(FakeModel(["A", "B"]), "Z", tmp_path / "x.cif")

    def test_query_builder_reports_available_chains(self) -> None:
        qb = epifitscout.QueryBuilder(_a_cif())
        with pytest.raises(ValueError, match="Available"):
            qb.resolve_chain("@")
