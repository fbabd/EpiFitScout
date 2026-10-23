"""The gate must pass a real graft and reject an impossible one.

A feasibility gate has two ways to be wrong and both are silent. Too strict and
it rejects loops that would have worked, which looks like a method with poor
coverage. Too loose and it admits broken backbones, whose interface scores are
measurements of nothing.

The calibration point is the native loop: put a loop back where it came from and
every junction is exactly the bond it always was, so the gate must pass it with
near-ideal geometry. Anything that fails *that* is too strict by construction.
The opposite bound comes from loops moved deliberately out of position.

Measured behaviour on a real search (1a2y CDR-H3, 600 matches from the full
25,226-chain database): 60% pass, junction bonds median 1.25 and 1.29 Å against
the ideal 1.329, worst junction angle p90 20°. The failures are real ones —
angles beyond tolerance and collisions with the antibody's own framework.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from epifitscout.build.assemble import TemplateSite
from epifitscout.config.schema import FeasibilityConfig
from epifitscout.graft import feasibility as feas
from epifitscout.query.builder import QueryBuilder
from epifitscout.types.anchor_query import AnchorQuery
from epifitscout.types.candidate import GraftCandidate
from epifitscout.types.fragment import Fragment
from epifitscout.types.superposition import Superposition

EXAMPLE = Path(__file__).resolve().parent.parent / "example" / "1a2y.pdb"
H3 = (105, 117)


@pytest.fixture(scope="module")
def query() -> AnchorQuery:
    return QueryBuilder(EXAMPLE).get_anchor_query("B", *H3)


@pytest.fixture(scope="module")
def site(query: AnchorQuery) -> TemplateSite:
    return TemplateSite.load(EXAMPLE, "B", query.loop_range, ("A",))


def _native_candidate(
    query: AnchorQuery, transform: Superposition | None = None
) -> GraftCandidate:
    """The template's own loop, offered back as a candidate for its own site."""
    coords = query.metadata["native_loop_coords"]
    loop = Fragment(
        pdb_id=query.pdb_id, chain=query.chain,
        residue_range=query.loop_range,
        coords=np.asarray(coords, dtype=np.float64),
        sequence=query.metadata["native_loop_sequence"],
    )
    return GraftCandidate(
        loop=loop, n_stem=query.n_stem, c_stem=query.c_stem,
        superposition=transform or Superposition.identity(),
        master_rmsd=0.0, stem_rmsd_ca=0.0, stem_rmsd_bb=0.0,
    )


class TestTheNativeLoopPasses:
    def test_a_loop_in_its_own_site_is_feasible(
        self, query: AnchorQuery, site: TemplateSite
    ) -> None:
        verdict = feas.assess(_native_candidate(query), query, site.framework_coords)
        assert verdict.passed, verdict.describe()

    def test_its_junction_bonds_are_peptide_bonds(
        self, query: AnchorQuery, site: TemplateSite
    ) -> None:
        verdict = feas.assess(_native_candidate(query), query, site.framework_coords)
        assert verdict.n_bond == pytest.approx(feas.IDEAL_BOND, abs=0.1)
        assert verdict.c_bond == pytest.approx(feas.IDEAL_BOND, abs=0.1)

    def test_its_junction_angles_are_near_ideal(
        self, query: AnchorQuery, site: TemplateSite
    ) -> None:
        verdict = feas.assess(_native_candidate(query), query, site.framework_coords)
        assert verdict.worst_angle < 10.0

    def test_its_junctions_are_trans(
        self, query: AnchorQuery, site: TemplateSite
    ) -> None:
        verdict = feas.assess(_native_candidate(query), query, site.framework_coords)
        assert abs(verdict.n_omega) > 150.0
        assert abs(verdict.c_omega) > 150.0

    def test_it_does_not_clash_with_its_own_framework(
        self, query: AnchorQuery, site: TemplateSite
    ) -> None:
        """If this fails, the framework atom set includes something it should not.

        Ordered water is the trap: solvent was modelled around this very loop,
        so leaving waters in makes a native loop clash with a dozen of them.
        """
        verdict = feas.assess(_native_candidate(query), query, site.framework_coords)
        assert verdict.self_clashes == 0


class TestImpossibleGraftsAreRejected:
    def test_a_displaced_loop_cannot_be_bonded(
        self, query: AnchorQuery, site: TemplateSite
    ) -> None:
        shifted = Superposition(
            rotation=np.eye(3), translation=np.array([4.0, 0.0, 0.0])
        )
        verdict = feas.assess(
            _native_candidate(query, shifted), query, site.framework_coords
        )
        assert not verdict.passed
        assert any("bond" in reason for reason in verdict.reasons)

    def test_a_rotated_loop_has_the_wrong_take_off(
        self, query: AnchorQuery, site: TemplateSite
    ) -> None:
        """A loop whose ends point the wrong way fails on angles, not distance."""
        from scipy.spatial.transform import Rotation

        centre = query.metadata["native_loop_coords"].reshape(-1, 3).mean(axis=0)
        rotation = Rotation.from_euler("z", 25, degrees=True).as_matrix()
        spun = Superposition(
            rotation=rotation, translation=centre - centre @ rotation.T
        )
        verdict = feas.assess(
            _native_candidate(query, spun), query, site.framework_coords
        )
        assert not verdict.passed

    def test_the_verdict_says_why(
        self, query: AnchorQuery, site: TemplateSite
    ) -> None:
        shifted = Superposition(
            rotation=np.eye(3), translation=np.array([0.0, 9.0, 0.0])
        )
        verdict = feas.assess(
            _native_candidate(query, shifted), query, site.framework_coords
        )
        assert verdict.reasons
        assert "feasible" not in verdict.describe()


class TestGateMechanics:
    def test_assess_all_keeps_the_verdict_on_rejects(
        self, query: AnchorQuery, site: TemplateSite
    ) -> None:
        """A rejected candidate stays inspectable; it is dropped, not erased."""
        good = _native_candidate(query)
        bad = _native_candidate(
            query,
            Superposition(rotation=np.eye(3), translation=np.array([6.0, 0.0, 0.0])),
        )
        kept = feas.assess_all([good, bad], query, site.framework_coords)
        assert len(kept) == 1
        assert bad.feasibility is not None and not bad.feasibility.passed

    def test_a_tighter_tolerance_rejects_more(
        self, query: AnchorQuery, site: TemplateSite
    ) -> None:
        candidate = _native_candidate(query)
        strict = FeasibilityConfig(bond_min=1.32, bond_max=1.33)
        assert feas.assess(candidate, query, site.framework_coords).passed
        assert not feas.assess(candidate, query, site.framework_coords, strict).passed

    def test_an_unbridgeable_site_is_identified_before_any_search(
        self, query: AnchorQuery
    ) -> None:
        assert feas.query_is_bridgeable(query)
        stretched = AnchorQuery(
            pdb_id=query.pdb_id, chain=query.chain,
            n_stem=query.n_stem, c_stem=query.c_stem,
            gap_length=1, loop_range=(1, 1),
        )
        assert not feas.query_is_bridgeable(stretched)


class TestGeometryPrimitives:
    def test_angle_of_a_right_angle(self) -> None:
        a, b, c = np.array([1.0, 0, 0]), np.zeros(3), np.array([0, 1.0, 0])
        assert feas._angle(a, b, c) == pytest.approx(90.0)

    def test_dihedral_of_a_planar_trans_arrangement(self) -> None:
        """Four atoms in a plane, the outer two on opposite sides: 180°."""
        a = np.array([-1.0, 1.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])
        d = np.array([2.0, -1.0, 0.0])
        assert abs(feas._dihedral(a, b, c, d)) == pytest.approx(180.0)

    def test_dihedral_of_a_planar_cis_arrangement(self) -> None:
        a = np.array([-1.0, 1.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])
        d = np.array([2.0, 1.0, 0.0])
        assert feas._dihedral(a, b, c, d) == pytest.approx(0.0, abs=1e-6)

    def test_dihedral_of_a_quarter_turn(self) -> None:
        a = np.array([-1.0, 1.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])
        d = np.array([2.0, 0.0, 1.0])
        assert abs(feas._dihedral(a, b, c, d)) == pytest.approx(90.0)
