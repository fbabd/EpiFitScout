"""Assembling a grafted antibody-antigen complex.

``search()`` returns ranked fragments; ``epifitscout build`` turns them into
structures. The properties worth pinning are structural, not cosmetic: the
antigen must survive untouched, the antibody must stay complete, the graft must
not introduce a chain break, and the loop must take the template's own residue
identifiers so insertion-coded CDR numbering survives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "database_builder/rcsb_cache/3gbn.pdb"


@pytest.fixture(scope="module")
def bc():
    """The tool as an ordinary package module, which is how users get it."""
    from epifitscout.tools import build_complex

    return build_complex


class TestPackerHandling:
    def test_absent_packer_is_reported_not_guessed(self, bc) -> None:
        assert bc.find_packer("definitely-not-a-real-packer-xyz") is None

    def test_known_packers_have_command_templates(self, bc) -> None:
        for name in ("FASPR", "Scwrl4"):
            assert name in bc._PACKERS
            assert any("{inp}" in part for part in bc._PACKERS[name])
            assert any("{out}" in part for part in bc._PACKERS[name])

    def test_unknown_packer_refuses_to_run(self, bc, tmp_path: Path) -> None:
        pdb = tmp_path / "x.pdb"
        pdb.write_text("ATOM\n")
        assert bc.run_packer(Path("/bin/true"), "NotAPacker", pdb) is False

    def test_env_var_locates_a_packer(self, bc, tmp_path: Path, monkeypatch) -> None:
        exe = tmp_path / "FASPR"
        exe.write_text("#!/bin/sh\n")
        monkeypatch.setenv("FASPR_BIN", str(exe))
        assert bc.find_packer("FASPR") == exe.resolve()

    def test_env_var_may_name_the_directory(self, bc, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "FASPR").write_text("#!/bin/sh\n")
        monkeypatch.setenv("FASPR_BIN", str(tmp_path))
        assert bc.find_packer("FASPR") == (tmp_path / "FASPR").resolve()

    def test_the_packers_own_folder_is_searched(self, bc, monkeypatch, tmp_path: Path) -> None:
        """FASPR lives in ``FASPR/bin/``, mirroring ``MASTER/bin/``."""
        monkeypatch.delenv("FASPR_BIN", raising=False)
        monkeypatch.setattr(bc, "REPO_ROOT", tmp_path)
        exe = tmp_path / "FASPR" / "bin" / "FASPR"
        exe.parent.mkdir(parents=True)
        exe.write_text("#!/bin/sh\n")
        assert bc.find_packer("FASPR") == exe.resolve()

    def test_the_repository_copy_is_found_before_PATH(
        self, bc, monkeypatch, tmp_path: Path
    ) -> None:
        """A system-wide packer must not shadow the one installed here."""
        monkeypatch.delenv("FASPR_BIN", raising=False)
        monkeypatch.setattr(bc, "REPO_ROOT", tmp_path)
        exe = tmp_path / "FASPR" / "bin" / "FASPR"
        exe.parent.mkdir(parents=True)
        exe.write_text("#!/bin/sh\n")
        elsewhere = tmp_path / "usr_bin"
        elsewhere.mkdir()
        other = elsewhere / "FASPR"
        other.write_text("#!/bin/sh\n")
        other.chmod(0o755)
        monkeypatch.setenv("PATH", str(elsewhere))
        assert bc.find_packer("FASPR") == exe.resolve()

    def test_symlinked_packer_resolves_to_the_real_file(
        self, bc, tmp_path: Path, monkeypatch
    ) -> None:
        """FASPR reads its rotamer library from dirname(argv[0]), so a symlink
        must be resolved or the library is sought in the wrong directory."""
        real = tmp_path / "real"
        real.mkdir()
        (real / "FASPR").write_text("#!/bin/sh\n")
        link = tmp_path / "FASPR"
        link.symlink_to(real / "FASPR")
        monkeypatch.setenv("FASPR_BIN", str(link))
        assert bc.find_packer("FASPR") == (real / "FASPR").resolve()


class TestRotamerLibraryCheck:
    """FASPR exits with an unhelpful message when its library is misplaced;
    this must be caught before a search runs, not after."""

    def test_missing_library_is_reported(self, bc, tmp_path: Path) -> None:
        exe = tmp_path / "FASPR"
        exe.write_text("#!/bin/sh\n")
        problem = bc.check_packer(exe, "FASPR")
        assert problem is not None
        assert bc.FASPR_ROTAMER_LIB in problem

    def test_complete_install_passes(self, bc, tmp_path: Path) -> None:
        exe = tmp_path / "FASPR"
        exe.write_text("#!/bin/sh\n")
        (tmp_path / bc.FASPR_ROTAMER_LIB).write_bytes(b"\0")
        assert bc.check_packer(exe, "FASPR") is None

    def test_other_packers_need_no_library(self, bc, tmp_path: Path) -> None:
        assert bc.check_packer(tmp_path / "Scwrl4", "Scwrl4") is None


def _atom(serial: int, name: str, resname: str, chain: str, num: int,
          xyz=(0.0, 0.0, 0.0), icode: str = " ", altloc: str = " ",
          record: str = "ATOM  ") -> str:
    return (f"{record}{serial:>5} {name:<4}{altloc}{resname:>3} {chain}{num:>4}{icode}   "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00\n")


class TestFasprSequence:
    """``-s`` is matched positionally against the residues FASPR parses, and
    FASPR validates nothing: a misaligned sequence silently mutates the model.
    """

    def test_reads_standard_atom_residues_in_file_order(self, bc) -> None:
        lines = [
            _atom(1, "CA", "ALA", "A", 1),
            _atom(2, "CA", "GLY", "A", 2),
            _atom(3, "CA", "HOH", "A", 3, record="HETATM"),  # heteroatom: ignored
            _atom(4, "CA", "MSE", "A", 4),                   # non-standard: ignored
            _atom(5, "CA", "SER", "B", 1),                   # chain boundary: no break
        ]
        assert [key[0] for key, _ in bc.faspr_residues(lines)] == ["A", "A", "B"]
        assert [name for _, name in bc.faspr_residues(lines)] == ["ALA", "GLY", "SER"]

    def test_alternate_locations_count_once(self, bc) -> None:
        lines = [
            _atom(1, "CA", "ALA", "A", 1, altloc="A"),
            _atom(2, "CA", "ALA", "A", 1, altloc="B"),
        ]
        assert len(bc.faspr_residues(lines)) == 1

    def test_only_the_loop_and_its_shell_are_free_to_move(self, bc) -> None:
        lines = [
            _atom(1, "CA", "ALA", "H", 93, xyz=(0.0, 0.0, 0.0)),   # the graft
            _atom(2, "CA", "LYS", "H", 94, xyz=(1.0, 0.0, 0.0)),   # the graft
            _atom(3, "CA", "SER", "B", 10, xyz=(4.0, 0.0, 0.0)),   # within 5 A
            _atom(4, "CA", "TRP", "B", 11, xyz=(40.0, 0.0, 0.0)),  # far away
        ]
        seq, n_free = bc.repack_sequence(lines, "H", (93, 94), shell=5.0)
        assert seq == "AKSw", seq
        assert n_free == 3

    def test_sequence_length_matches_the_residues_faspr_will_read(self, bc) -> None:
        if not TEMPLATE.is_file():
            pytest.skip(f"{TEMPLATE} not cached")
        lines = TEMPLATE.read_text().splitlines(keepends=True)
        seq, _ = bc.repack_sequence(lines, "H", (93, 102), shell=5.0)
        assert len(seq) == len(bc.faspr_residues(lines))

    def test_an_empty_span_raises_rather_than_packing_everything(self, bc) -> None:
        lines = [_atom(1, "CA", "ALA", "H", 93)]
        with pytest.raises(ValueError, match="no grafted residues"):
            bc.repack_sequence(lines, "H", (500, 510), shell=5.0)


class TestHeteroatomsSurvive:
    """FASPR discards every HETATM. On a glycosylated antigen those are part of
    the interface, so they are restored from the input."""

    def test_heteroatoms_are_put_back_and_serials_renumbered(self, bc) -> None:
        original = [
            _atom(1, "CA", "ALA", "A", 1),
            _atom(2, "C1", "NAG", "A", 500, record="HETATM"),
        ]
        packed = [_atom(1, "CA", "ALA", "A", 1), "END\n"]
        merged, dropped, overlapping = bc._merge_heteroatoms(packed, original)
        assert sum(x.startswith("HETATM") for x in merged) == 1
        serials = [int(x[6:11]) for x in merged if x.startswith(("ATOM", "HETATM"))]
        assert serials == [1, 2]
        assert merged[-1].startswith("END")
        assert (dropped, overlapping) == (0, [])

    def test_nothing_is_invented_when_there_were_none(self, bc) -> None:
        packed = [_atom(1, "CA", "ALA", "A", 1)]
        merged, _, _ = bc._merge_heteroatoms(packed, [_atom(1, "CA", "ALA", "A", 1)])
        assert merged == packed

    def test_a_water_in_the_grafts_way_is_displaced(self, bc) -> None:
        """Template waters were modelled around the template's own loop; where
        the graft now sits some of them overlap it, by as little as 0.5 A."""
        graft = np.zeros((1, 3))
        original = [_atom(1, "O", "HOH", "A", 300, xyz=(0.5, 0.0, 0.0), record="HETATM")]
        merged, dropped, overlapping = bc._merge_heteroatoms([], original, graft)
        assert dropped == 1
        assert not any(x.startswith("HETATM") for x in merged)
        assert overlapping == []

    def test_a_distant_water_is_kept(self, bc) -> None:
        graft = np.zeros((1, 3))
        original = [_atom(1, "O", "HOH", "A", 300, xyz=(9.0, 0.0, 0.0), record="HETATM")]
        merged, dropped, _ = bc._merge_heteroatoms([], original, graft)
        assert dropped == 0
        assert sum(x.startswith("HETATM") for x in merged) == 1

    def test_an_overlapping_ligand_is_reported_not_deleted(self, bc) -> None:
        """Removing a glycan or ligand is a modelling decision to surface, not
        one to make silently."""
        graft = np.zeros((1, 3))
        original = [_atom(1, "C1", "NAG", "A", 500, xyz=(0.5, 0.0, 0.0), record="HETATM")]
        merged, dropped, overlapping = bc._merge_heteroatoms([], original, graft)
        assert dropped == 0
        assert sum(x.startswith("HETATM") for x in merged) == 1
        assert len(overlapping) == 1 and "NAG" in overlapping[0]

    def test_a_multi_atom_water_residue_is_judged_as_a_whole(self, bc) -> None:
        """One overlapping atom condemns the residue; a ligand must not be left
        half present."""
        graft = np.zeros((1, 3))
        original = [
            _atom(1, "O", "HOH", "A", 300, xyz=(0.5, 0.0, 0.0), record="HETATM"),
            _atom(2, "O", "HOH", "A", 300, xyz=(8.0, 0.0, 0.0), record="HETATM"),
        ]
        merged, dropped, _ = bc._merge_heteroatoms([], original, graft)
        assert dropped == 1
        assert not any(x.startswith("HETATM") for x in merged)


class TestContactChemistry:
    """A single distance cut cannot separate a hydrogen bond from a clash: at
    one real interface 2.40 A Tyr-OH-to-Gln-CG is a clash while 2.65 A
    Gln-OE1-to-Gln-NE2 is a good hydrogen bond."""

    @staticmethod
    def _pdb(tmp_path: Path, pairs: list[tuple[str, str, float]]) -> Path:
        lines, serial = [], 1
        for i, (loop_atom, ag_atom, dx) in enumerate(pairs):
            lines.append(_atom(serial, loop_atom, "TYR", "H", 98 + i, xyz=(0.0, 3.0 * i, 0.0)))
            lines.append(_atom(serial + 1, ag_atom, "GLN", "B", 38 + i,
                               xyz=(dx, 3.0 * i, 0.0)))
            serial += 2
        path = tmp_path / "c.pdb"
        path.write_text("".join(lines))
        return path

    def test_a_short_polar_contact_is_a_hydrogen_bond(self, bc, tmp_path: Path) -> None:
        pdb = self._pdb(tmp_path, [("OH", "NE2", 2.65)])
        distance, clash = bc.closest_contact(pdb, "H", (98, 98), "B")
        assert distance == pytest.approx(2.65, abs=0.01)
        assert clash is None

    def test_the_same_distance_to_a_carbon_is_a_clash(self, bc, tmp_path: Path) -> None:
        pdb = self._pdb(tmp_path, [("OH", "CG", 2.65)])
        _, clash = bc.closest_contact(pdb, "H", (98, 98), "B")
        assert clash is not None and f"needs {bc.VDW_MIN:.1f}" in clash

    def test_an_impossibly_short_polar_contact_is_still_a_clash(
        self, bc, tmp_path: Path
    ) -> None:
        pdb = self._pdb(tmp_path, [("OH", "NE2", 2.0)])
        _, clash = bc.closest_contact(pdb, "H", (98, 98), "B")
        assert clash is not None and f"needs {bc.HBOND_MIN:.1f}" in clash

    def test_the_worst_clash_is_the_one_reported(self, bc, tmp_path: Path) -> None:
        pdb = self._pdb(tmp_path, [("OH", "CG", 2.7), ("OH", "CG", 2.2)])
        _, clash = bc.closest_contact(pdb, "H", (98, 99), "B")
        assert clash is not None and "2.20 A" in clash

    def test_a_comfortable_interface_reports_no_clash(self, bc, tmp_path: Path) -> None:
        pdb = self._pdb(tmp_path, [("OH", "CG", 3.6)])
        distance, clash = bc.closest_contact(pdb, "H", (98, 98), "B")
        assert clash is None and distance == pytest.approx(3.6, abs=0.01)

    def test_a_missing_chain_is_not_silently_scored(self, bc, tmp_path: Path) -> None:
        pdb = self._pdb(tmp_path, [("OH", "CG", 3.6)])
        distance, clash = bc.closest_contact(pdb, "H", (98, 98), "Z")
        assert distance != distance and clash is None  # NaN, not a number

    def test_element_falls_back_to_the_atom_name(self, bc) -> None:
        """Not every PDB writer fills columns 77-78."""
        line = _atom(1, "CA", "ALA", "A", 1)[:76] + "\n"
        assert bc._element(line) == "C"


class TestClashCalibration:
    """The thresholds come from 102 native antibody-antigen interfaces, not from
    van der Waals radii.

    Deposited structures are far closer-packed than textbook radii allow: the
    median native closest approach is 2.74 Å between polar atoms and 3.17 Å when
    a carbon is involved, but the short tail runs to 2.12 Å and 1.26 Å. A
    vdW-derived 3.0 Å carbon cut flags 12.7% of real interfaces, and a 2.5 Å
    polar cut flags 22.5% — a check that fires on a quarter of genuine
    structures tells you nothing. As set, the cuts flag 2.9% of natives on polar
    contacts and 2.0% on carbon ones, which places each between the 1st and 5th
    percentile of what real interfaces do.
    """

    #: Percentiles measured over those 102 interfaces.
    NATIVE_POLAR = {1: 2.12, 2: 2.17, 5: 2.26, 50: 2.74}
    NATIVE_CARBON = {1: 2.74, 2: 2.83, 5: 2.90, 50: 3.17}

    def test_polar_cut_admits_all_but_the_shortest_natives(self, bc) -> None:
        assert self.NATIVE_POLAR[5] > bc.HBOND_MIN > self.NATIVE_POLAR[1]

    def test_carbon_cut_admits_all_but_the_shortest_natives(self, bc) -> None:
        assert self.NATIVE_CARBON[5] > bc.VDW_MIN > self.NATIVE_CARBON[1]

    def test_a_carbon_contact_is_held_to_more_than_a_polar_one(self, bc) -> None:
        assert bc.VDW_MIN > bc.HBOND_MIN

    def test_the_median_native_interface_is_not_flagged(self, bc) -> None:
        """Whatever the cuts become, a typical real interface must pass."""
        assert bc.HBOND_MIN < self.NATIVE_POLAR[50]
        assert bc.VDW_MIN < self.NATIVE_CARBON[50]

    @pytest.mark.needs_master
    def test_the_template_passes_its_own_check(self, bc) -> None:
        """3GBN's own H3 against its own epitope: the check must not condemn the
        structure the graft is built from."""
        if not TEMPLATE.is_file():
            pytest.skip(f"{TEMPLATE} not cached")
        distance, clash = bc.closest_contact(TEMPLATE, "H", (93, 102), "B")
        assert clash is None, f"the native template was flagged: {clash}"
        assert 2.0 < distance < 4.0


class TestPackedModelIsVerified:
    """FASPR exits 0 and writes nothing when ``-s`` is the wrong length, so the
    output is checked instead of trusted."""

    def test_a_faithful_repack_is_accepted(self, bc) -> None:
        before = [_atom(1, "CA", "LYS", "H", 94), _atom(2, "CB", "LYS", "H", 94)]
        after = [_atom(1, "CA", "LYS", "H", 94),
                 _atom(2, "CB", "LYS", "H", 94, xyz=(3.0, 0.0, 0.0))]
        assert bc._verify_pack(before, after) is None

    def test_a_mutation_is_caught(self, bc) -> None:
        before = [_atom(1, "CA", "LYS", "H", 94)]
        after = [_atom(1, "CA", "ALA", "H", 94)]
        problem = bc._verify_pack(before, after)
        assert problem and "changed identity" in problem

    def test_a_dropped_residue_is_caught(self, bc) -> None:
        before = [_atom(1, "CA", "LYS", "H", 94), _atom(2, "CA", "SER", "H", 95)]
        after = [_atom(1, "CA", "LYS", "H", 94)]
        problem = bc._verify_pack(before, after)
        assert problem and "dropped" in problem

    def test_a_moved_backbone_is_caught(self, bc) -> None:
        before = [_atom(1, "CA", "LYS", "H", 94)]
        after = [_atom(1, "CA", "LYS", "H", 94, xyz=(0.0, 0.0, 2.0))]
        problem = bc._verify_pack(before, after)
        assert problem and "backbone moved" in problem


class TestGraftReport:
    def test_join_quality_is_judged_against_a_peptide_bond(
        self, bc, tmp_path: Path
    ) -> None:
        good = bc.GraftReport(1, "X:A", 0.5, 0.4, 1.33, 1.30, 0, 5, "none",
                              tmp_path / "a.pdb")
        bad = bc.GraftReport(2, "Y:B", 0.5, 0.4, 1.33, 4.20, 0, 5, "none",
                             tmp_path / "b.pdb")
        assert good.joins_cleanly and not bad.joins_cleanly
        assert "CHAIN BREAK" in bad.line()
        assert "CHAIN BREAK" not in good.line()


@pytest.mark.needs_master
@pytest.mark.slow
class TestAssembly:
    """End-to-end: build complexes and check they are structurally sound."""

    @staticmethod
    @pytest.fixture(scope="class")
    def built(tmp_path_factory, bc):
        if not TEMPLATE.is_file():
            pytest.skip(f"{TEMPLATE} not cached")
        from tests.conftest import MASTER_DIR

        if MASTER_DIR is None:
            pytest.skip("MASTER not available")
        out = tmp_path_factory.mktemp("grafts")
        assert bc.main([
            "--structure", str(TEMPLATE), "--cdr", "H", "93", "102",
            "--epitope", "B", "38", "46", "--top", "2",
            "--out-dir", str(out), "--workers", "2",
        ]) == 0
        files = sorted(out.glob("*.pdb"))
        assert files, "no complexes were written"
        return files, out

    def test_writes_complexes_and_a_summary(self, built) -> None:
        files, out = built
        assert len(files) == 2
        header = (out / "grafts.tsv").read_text().splitlines()[0].split("\t")
        for column in ("rank", "source", "rmsd", "n_gap", "c_gap", "clashes"):
            assert column in header

    def test_antigen_survives_and_antibody_stays_complete(self, built) -> None:
        from Bio.PDB import PDBParser

        files, _ = built
        p = PDBParser(QUIET=True)
        want = {c.id: len([r for r in c if r.id[0] == " "])
                for c in p.get_structure("t", TEMPLATE)[0]}
        for f in files:
            got = {c.id: len([r for r in c if r.id[0] == " "])
                   for c in p.get_structure("g", f)[0]}
            assert set(got) == set(want), f"{f.name} lost or gained a chain"
            assert got["B"] == want["B"], f"{f.name} altered the antigen"
            assert got["H"] == want["H"], f"{f.name} changed the heavy chain length"

    def test_graft_introduces_no_chain_break(self, built) -> None:
        """Deposited structures contain disordered gaps; those are not the
        graft's doing, so compare against the template rather than demand zero."""
        from Bio.PDB import PDBParser

        files, _ = built
        p = PDBParser(QUIET=True)

        def breaks(path: Path) -> set[tuple[int, int]]:
            res = [r for r in p.get_structure("x", path)[0]["H"] if r.id[0] == " "]
            return {
                (a.id[1], b.id[1])
                for a, b in zip(res, res[1:], strict=False)
                if "C" in a and "N" in b
                and np.linalg.norm(b["N"].coord - a["C"].coord) > 2.0
            }

        reference = breaks(TEMPLATE)
        for f in files:
            assert breaks(f) == reference, f"{f.name} introduced a backbone break"

    def test_insertion_coded_numbering_is_preserved(self, built) -> None:
        """Antibody CDRs are numbered 100, 100A, 100B...; renumbering the graft
        sequentially would collide with residues further along the chain."""
        from Bio.PDB import PDBParser

        files, _ = built
        p = PDBParser(QUIET=True)
        want = [r.id for r in p.get_structure("t", TEMPLATE)[0]["H"]
                if r.id[0] == " " and 93 <= r.id[1] <= 102]
        assert any(i[2].strip() for i in want), "this template should have insertion codes"
        for f in files:
            got = [r.id for r in p.get_structure("g", f)[0]["H"]
                   if r.id[0] == " " and 93 <= r.id[1] <= 102]
            assert got == want, f"{f.name} renumbered the grafted loop"
