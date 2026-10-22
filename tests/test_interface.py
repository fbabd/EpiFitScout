"""The interface measurement, against cases whose answers are known in advance.

Surface area and void volume are numerical, so they are checked where analysis
gives the answer: an isolated atom's accessible surface is exactly the area of
its expanded sphere, a fully enclosed atom has none, and two bodies out of each
other's reach trap no volume between them.

The terms are also checked for the confound they exist to avoid. Burial and
contact counts reward interpenetration — in the v1 benchmark a random control
buried more surface than the native loop by being driven into the antigen — so a
clashing interface must come out with *both* a large contact count and a large
clash count, and the clash count must not be something a looser definition can
hide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from epifitscout.config.schema import InterfaceConfig
from epifitscout.scoring import sasa
from epifitscout.scoring.interface import InterfaceScorer, read_atoms


def _record(record: str, serial: int, name: str, resname: str, chain: str,
            resnum: int, xyz: tuple[float, float, float],
            element: str | None = None) -> str:
    """One PDB ATOM/HETATM line, in the fixed columns the readers index by."""
    element = element or name[0]
    return (
        f"{record:<6s}{serial:5d} {name:<4s} {resname:>3s} {chain}{resnum:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          "
        f"{element:>2s}"
    )


def _atom(serial: int, name: str, resname: str, chain: str, resnum: int,
          xyz: tuple[float, float, float], element: str | None = None) -> str:
    return _record("ATOM", serial, name, resname, chain, resnum, xyz, element)


def _het(serial: int, name: str, resname: str, chain: str, resnum: int,
         xyz: tuple[float, float, float], element: str | None = None) -> str:
    return _record("HETATM", serial, name, resname, chain, resnum, xyz, element)


class TestShrakeRupley:
    def test_a_lone_atom_has_the_area_of_its_expanded_sphere(self) -> None:
        radii = np.array([1.70])
        area = sasa.shrake_rupley(np.zeros((1, 3)), radii, probe=1.4, n_points=2000)
        assert area[0] == pytest.approx(4 * np.pi * (1.70 + 1.4) ** 2, rel=1e-9)

    def test_an_enclosed_atom_has_no_accessible_surface(self) -> None:
        """A small atom at the centre of a shell of large ones."""
        shell = sasa.sphere_points(60) * 2.0
        coords = np.vstack([np.zeros(3), shell])
        radii = np.concatenate([[1.0], np.full(len(shell), 2.5)])
        area = sasa.shrake_rupley(coords, radii, n_points=500)
        assert area[0] == pytest.approx(0.0, abs=1e-9)

    def test_touching_atoms_occlude_each_other(self) -> None:
        coords = np.array([[0.0, 0, 0], [2.5, 0, 0]])
        radii = np.array([1.70, 1.70])
        together = sasa.shrake_rupley(coords, radii, n_points=1000).sum()
        apart = 2 * sasa.shrake_rupley(
            np.zeros((1, 3)), radii[:1], n_points=1000
        ).sum()
        assert together < apart

    def test_sampling_is_deterministic(self) -> None:
        """Two runs of one measurement must not disagree in the last digit."""
        coords, radii = sasa.sphere_points(12) * 3.0, np.full(12, 1.7)
        first = sasa.shrake_rupley(coords, radii, n_points=200)
        second = sasa.shrake_rupley(coords, radii, n_points=200)
        assert np.array_equal(first, second)

    def test_an_unknown_element_is_not_treated_as_small(self) -> None:
        assert sasa.atom_radii(["ZZ"])[0] == sasa.DEFAULT_RADIUS
        assert sasa.DEFAULT_RADIUS >= sasa.VDW_RADII["C"]


class TestInterfaceVoid:
    def test_bodies_out_of_reach_trap_nothing(self) -> None:
        a = np.zeros((1, 3))
        b = np.array([[40.0, 0.0, 0.0]])
        radii = np.array([1.7])
        assert sasa.interface_void(a, radii, b, radii) == 0.0

    def test_a_gap_between_two_bodies_is_counted(self) -> None:
        """Two plates a few ångströms apart: the space between them is void."""
        grid = np.stack(np.meshgrid(
            np.arange(-6.0, 6.1, 2.0), np.arange(-6.0, 6.1, 2.0), [0.0],
            indexing="ij",
        ), axis=-1).reshape(-1, 3)
        upper = grid + np.array([0.0, 0.0, 6.0])
        radii_a = np.full(len(grid), 1.7)
        radii_b = np.full(len(upper), 1.7)
        volume = sasa.interface_void(grid, radii_a, upper, radii_b, spacing=0.5)
        assert volume > 0.0

    def test_a_wider_gap_traps_more(self) -> None:
        line = np.stack([np.arange(-8.0, 8.1, 2.0), np.zeros(9), np.zeros(9)], axis=1)
        radii = np.full(len(line), 1.7)
        close = sasa.interface_void(
            line, radii, line + np.array([0, 6.2, 0]), radii, spacing=0.5
        )
        wide = sasa.interface_void(
            line, radii, line + np.array([0, 7.6, 0]), radii, spacing=0.5
        )
        assert wide > close


class TestReadingAModel:
    def _model(self, path: Path, extra: list[str] | None = None) -> Path:
        lines = [
            _atom(1, "N", "ALA", "H", 100, (0.0, 0.0, 0.0)),
            _atom(2, "CA", "ALA", "H", 100, (1.5, 0.0, 0.0)),
            _atom(3, "N", "ALA", "H", 50, (20.0, 0.0, 0.0)),      # framework
            _atom(4, "CA", "GLY", "A", 1, (4.0, 0.0, 0.0)),       # antigen
            _het(5, "O", "HOH", "A", 2, (2.0, 0.0, 0.0)),
        ]
        path.write_text("\n".join(lines + (extra or [])) + "\nEND\n")
        return path

    def test_water_is_excluded_from_both_sides(self, tmp_path: Path) -> None:
        """Solvent would otherwise occlude the surfaces whose burial is measured."""
        model = self._model(tmp_path / "m.pdb")
        _, antigen, _ = read_atoms(model, "H", (100, 100), ("A",))
        assert len(antigen) == 1          # the GLY CA, not the water

    def test_hydrogens_are_excluded(self, tmp_path: Path) -> None:
        extra = [_atom(6, "HA", "GLY", "A", 1, (4.5, 0.5, 0.0), element="H")]
        model = self._model(tmp_path / "m.pdb", extra)
        _, antigen, _ = read_atoms(model, "H", (100, 100), ("A",))
        assert len(antigen) == 1

    def test_the_loop_mask_selects_only_the_grafted_span(self, tmp_path: Path) -> None:
        model = self._model(tmp_path / "m.pdb")
        antibody, _, mask = read_atoms(model, "H", (100, 100), ("A",))
        assert len(antibody) == 3 and mask.sum() == 2

    def test_a_glycan_on_the_antigen_is_kept(self, tmp_path: Path) -> None:
        """A sugar at the interface is part of what a loop has to fit against."""
        extra = [_het(7, "C1", "NAG", "A", 3, (5.0, 0.0, 0.0), element="C")]
        model = self._model(tmp_path / "m.pdb", extra)
        _, antigen, _ = read_atoms(model, "H", (100, 100), ("A",))
        assert len(antigen) == 2


class TestScoringRealModels:
    """On the example complex, with its own CDR-H3 in place."""

    EXAMPLE = Path(__file__).resolve().parent.parent / "example" / "1a2y.pdb"

    @pytest.fixture(scope="class")
    @classmethod
    def score(cls):
        scorer = InterfaceScorer(InterfaceConfig(sasa_points=100, grid_spacing=1.0))
        return scorer.score(cls.EXAMPLE, "B", (105, 117), ("A",))

    def test_a_native_interface_buries_surface(self, score) -> None:
        assert score.buried_area > 500.0
        assert score.loop_buried_area > 50.0

    def test_a_native_interface_does_not_clash(self, score) -> None:
        """The calibrated thresholds flag ~2% of natives; this is not one."""
        assert score.clashes == 0
        assert score.is_clean

    def test_a_native_interface_makes_contacts(self, score) -> None:
        assert score.contacts > 0
        assert 0 < score.contact_residues <= 10

    def test_terms_round_trip_through_as_dict(self, score) -> None:
        table = score.as_dict()
        assert table["buried_area"] == score.buried_area
        assert set(table) >= {"hbonds", "salt_bridges", "clashes", "gap_index"}

    def test_moving_the_antigen_away_removes_the_interface(
        self, tmp_path: Path
    ) -> None:
        """BSA must be a property of the two bodies' relative position."""
        lines = []
        for line in self.EXAMPLE.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")) and line[21] == "A":
                x = float(line[30:38]) + 80.0
                line = f"{line[:30]}{x:8.3f}{line[38:]}"
            lines.append(line)
        moved = tmp_path / "apart.pdb"
        moved.write_text("\n".join(lines) + "\n")

        scorer = InterfaceScorer(InterfaceConfig(sasa_points=100, grid_spacing=1.0))
        score = scorer.score(moved, "B", (105, 117), ("A",))
        assert score.buried_area == pytest.approx(0.0, abs=1.0)
        assert score.contacts == 0
        assert score.clashes == 0
        assert score.gap_volume == 0.0

    def test_a_missing_antigen_is_refused_not_scored_as_zero(
        self, tmp_path: Path
    ) -> None:
        scorer = InterfaceScorer(InterfaceConfig(sasa_points=50))
        with pytest.raises(ValueError, match="both sides"):
            scorer.score(self.EXAMPLE, "B", (105, 117), ("Z",))

    def test_a_missing_graft_is_refused(self) -> None:
        scorer = InterfaceScorer(InterfaceConfig(sasa_points=50))
        with pytest.raises(ValueError, match="no graft"):
            scorer.score(self.EXAMPLE, "B", (9000, 9001), ("A",))


class TestClashesAreNotHiddenByBurial:
    """The v1 confound, as a test: interpenetration must show up as clashes."""

    def test_a_driven_in_loop_reads_as_contacts_and_clashes_together(
        self, tmp_path: Path
    ) -> None:
        example = Path(__file__).resolve().parent.parent / "example" / "1a2y.pdb"
        lines = []
        for line in example.read_text().splitlines():
            if (
                line.startswith("ATOM") and line[21] == "B"
                and 105 <= int(line[22:26]) <= 117
            ):
                # push the loop 2 Å towards the antigen's centre of mass
                x = float(line[30:38]) - 2.0
                line = f"{line[:30]}{x:8.3f}{line[38:]}"
            lines.append(line)
        pushed = tmp_path / "pushed.pdb"
        pushed.write_text("\n".join(lines) + "\n")

        scorer = InterfaceScorer(InterfaceConfig(sasa_points=100, grid_spacing=1.0))
        native = scorer.score(example, "B", (105, 117), ("A",))
        driven = scorer.score(pushed, "B", (105, 117), ("A",))

        assert driven.contacts > native.contacts        # looks like a better fit
        assert driven.clashes > native.clashes          # but it is not one
        assert not driven.is_clean
