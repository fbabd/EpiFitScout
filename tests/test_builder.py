"""Building and *updating* the chain database.

The updater is meant to be run repeatedly, months apart, by people who no longer
have the RCSB cache. Each test here pins down one way that goes wrong:

* re-downloading chains that are already on disk;
* a run over a subset truncating the database;
* a half-written PDS from a failed ``createPDS`` being trusted next time;
* a PDS left stale after its PDB changed.

RCSB is replaced by a stub that serves ``example/1a2y.pdb`` (chains A, B, C) for
any id, so nothing here touches the network.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from epifitscout.db.chain_db_builder import ChainDbBuilder
from epifitscout.db.rcsb_downloader import RcsbDownloadError, StructureFile
from epifitscout.db.sabdab_metadata import SAbDabEntry, SAbDabMetadataReader
from epifitscout.db.verify import verify_database
from tests.conftest import EXAMPLE_PDB, MASTER_DIR


class StubRcsb:
    """Stands in for RcsbDownloader; records what was asked for."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail = fail or set()

    def download(self, pdb_id: str) -> StructureFile:
        self.calls.append(pdb_id)
        if pdb_id in self.fail:
            raise RcsbDownloadError(f"simulated network failure for {pdb_id}")
        return StructureFile(path=EXAMPLE_PDB, fmt="pdb")


def entry(pdb_id: str, h: str = "B", light: str = "A") -> SAbDabEntry:
    return SAbDabEntry(pdb_id=pdb_id, hchain=h, lchain=light, antigen_chain="C")


@pytest.fixture
def create_pds() -> Path:
    if MASTER_DIR is None:
        pytest.skip("createPDS not available")
    return MASTER_DIR / "createPDS"


@pytest.fixture
def failing_create_pds(tmp_path: Path) -> Path:
    """A createPDS that writes a partial file and then dies."""
    script = tmp_path / "createPDS"
    script.write_text(
        '#!/bin/sh\nfor a in "$@"; do last="$a"; done\n'
        'echo partial > "$last"\nexit 1\n'
    )
    script.chmod(0o755)
    return script


def lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


@pytest.mark.needs_master
class TestInitialBuild:
    def test_produces_a_complete_usable_database(self, tmp_path: Path, create_pds: Path) -> None:
        db = tmp_path / "db"
        builder = ChainDbBuilder(db, StubRcsb(), create_pds, workers=2)
        builder.build_from_sabdab([entry("1A2Y"), entry("1A2Z")])

        assert sorted(lines(db / "pds.list")) == [
            "pds/1A2Y_A_L.pds", "pds/1A2Y_B_H.pds",
            "pds/1A2Z_A_L.pds", "pds/1A2Z_B_H.pds",
        ]
        assert verify_database(db).ok
        meta = json.loads((db / "metadata.json").read_text())
        assert meta["pdb/1A2Y_B_H.pdb"] == {"pdb_id": "1A2Y", "chain": "B", "chain_type": "H"}

    def test_pds_files_are_what_the_original_pipeline_produced(
        self, tmp_path: Path, create_pds: Path
    ) -> None:
        """Converting through a temporary name must not change the bytes."""
        import subprocess

        db = tmp_path / "db"
        ChainDbBuilder(db, StubRcsb(), create_pds).build_from_sabdab([entry("1A2Y")])
        direct = tmp_path / "direct.pds"
        subprocess.run(
            [str(create_pds), "--type", "target",
             "--pdb", str(db / "pdb" / "1A2Y_B_H.pdb"), "--pds", str(direct)],
            check=True, capture_output=True,
        )
        assert (db / "pds" / "1A2Y_B_H.pds").read_bytes() == direct.read_bytes()


@pytest.mark.needs_master
class TestIncrementalUpdate:
    def test_rerun_downloads_nothing_and_rebuilds_nothing(
        self, tmp_path: Path, create_pds: Path
    ) -> None:
        """An updater without the RCSB cache must not re-fetch the database."""
        db = tmp_path / "db"
        ChainDbBuilder(db, StubRcsb(), create_pds).build_from_sabdab([entry("1A2Y")])
        stamps = {p.name: p.stat().st_mtime_ns for p in (db / "pds").glob("*.pds")}

        again = StubRcsb()
        builder = ChainDbBuilder(db, again, create_pds)
        builder.build_from_sabdab([entry("1A2Y")])

        assert again.calls == []
        assert builder.report.already_present == 2
        assert builder.report.pds_built == 0
        assert {p.name: p.stat().st_mtime_ns for p in (db / "pds").glob("*.pds")} == stamps

    def test_only_new_entries_are_fetched(self, tmp_path: Path, create_pds: Path) -> None:
        db = tmp_path / "db"
        ChainDbBuilder(db, StubRcsb(), create_pds).build_from_sabdab([entry("1A2Y")])

        stub = StubRcsb()
        builder = ChainDbBuilder(db, stub, create_pds)
        builder.build_from_sabdab([entry("1A2Y"), entry("9ZZZ")])

        assert stub.calls == ["9ZZZ"]
        assert builder.report.added == 2
        assert builder.report.pds_built == 2
        assert builder.report.in_database == 4

    def test_a_subset_run_does_not_truncate_the_database(
        self, tmp_path: Path, create_pds: Path
    ) -> None:
        db = tmp_path / "db"
        ChainDbBuilder(db, StubRcsb(), create_pds).build_from_sabdab(
            [entry("1A2Y"), entry("1A2Z")]
        )
        ChainDbBuilder(db, StubRcsb(), create_pds).build_from_sabdab([entry("1A2Y")])

        assert len(lines(db / "pds.list")) == 4
        assert len(lines(db / "db.list")) == 4
        assert len(json.loads((db / "metadata.json").read_text())) == 4


@pytest.mark.needs_master
class TestPdsRefresh:
    def test_a_regenerated_pdb_refreshes_its_pds(self, tmp_path: Path, create_pds: Path) -> None:
        db = tmp_path / "db"
        ChainDbBuilder(db, StubRcsb(), create_pds).build_from_sabdab([entry("1A2Y")])
        pdb = db / "pdb" / "1A2Y_B_H.pdb"
        pds = db / "pds" / "1A2Y_B_H.pds"
        before = pds.stat().st_mtime_ns

        future = time.time() + 30
        os.utime(pdb, (future, future))  # RCSB revised the entry; PDB rewritten
        builder = ChainDbBuilder(db, StubRcsb(), create_pds)
        builder.build_from_sabdab([entry("1A2Y")])

        assert pds.stat().st_mtime_ns != before
        assert builder.report.pds_built == 1
        assert builder.report.pds_reused == 1

    def test_rebuild_pds_regenerates_everything(self, tmp_path: Path, create_pds: Path) -> None:
        """Switching MASTER versions invalidates every target."""
        db = tmp_path / "db"
        ChainDbBuilder(db, StubRcsb(), create_pds).build_from_sabdab([entry("1A2Y")])
        builder = ChainDbBuilder(db, StubRcsb(), create_pds, force_pds=True)
        builder.build_from_sabdab([entry("1A2Y")])
        assert builder.report.pds_built == 2
        assert builder.report.pds_reused == 0

    def test_an_empty_pds_is_rebuilt(self, tmp_path: Path, create_pds: Path) -> None:
        db = tmp_path / "db"
        ChainDbBuilder(db, StubRcsb(), create_pds).build_from_sabdab([entry("1A2Y")])
        (db / "pds" / "1A2Y_A_L.pds").write_bytes(b"")

        builder = ChainDbBuilder(db, None, create_pds)
        builder.sync_pds()
        assert (db / "pds" / "1A2Y_A_L.pds").stat().st_size > 0
        assert builder.report.pds_built == 1

    def test_orphaned_pds_is_removed(self, tmp_path: Path, create_pds: Path) -> None:
        db = tmp_path / "db"
        ChainDbBuilder(db, StubRcsb(), create_pds).build_from_sabdab([entry("1A2Y")])
        (db / "pds" / "OLD1_X_H.pds").write_bytes(b"left over")

        builder = ChainDbBuilder(db, None, create_pds)
        builder.sync_pds()
        assert not (db / "pds" / "OLD1_X_H.pds").exists()
        assert builder.report.orphans_removed == 1
        assert "OLD1_X_H" not in (db / "pds.list").read_text()


class TestFailureHandling:
    def test_failed_createpds_leaves_no_truncated_target(
        self, tmp_path: Path, failing_create_pds: Path
    ) -> None:
        """The dangerous case: an 8-byte PDS that later runs would have trusted."""
        db = tmp_path / "db"
        builder = ChainDbBuilder(db, StubRcsb(), failing_create_pds)
        builder.build_from_sabdab([entry("1A2Y")])

        assert list((db / "pds").glob("*")) == [], "no PDS or temp debris may remain"
        assert lines(db / "pds.list") == []
        assert builder.report.pds_failed == 2
        assert builder.report.in_database == 0

    @pytest.mark.needs_master
    def test_recovers_once_createpds_works(
        self, tmp_path: Path, failing_create_pds: Path, create_pds: Path
    ) -> None:
        db = tmp_path / "db"
        ChainDbBuilder(db, StubRcsb(), failing_create_pds).build_from_sabdab([entry("1A2Y")])

        builder = ChainDbBuilder(db, StubRcsb(), create_pds)
        builder.build_from_sabdab([entry("1A2Y")])
        assert builder.report.pds_built == 2
        for pds in (db / "pds").glob("*.pds"):
            assert pds.stat().st_size > 10_000, f"{pds.name} looks truncated"
        assert verify_database(db).ok

    def test_download_failure_is_retried_next_run(
        self, tmp_path: Path, failing_create_pds: Path
    ) -> None:
        """A network blip must not be remembered as a permanent skip."""
        db = tmp_path / "db"
        first = ChainDbBuilder(db, StubRcsb(fail={"9ZZZ"}), failing_create_pds)
        first.build_from_sabdab([entry("9ZZZ")])
        assert first.report.download_failed == 2
        assert json.loads((db / "skipped.json").read_text()) == {}

        retry = StubRcsb()
        ChainDbBuilder(db, retry, failing_create_pds).build_from_sabdab([entry("9ZZZ")])
        assert retry.calls == ["9ZZZ"]

    def test_unextractable_chain_is_remembered_and_not_refetched(
        self, tmp_path: Path, failing_create_pds: Path
    ) -> None:
        db = tmp_path / "db"
        ChainDbBuilder(db, StubRcsb(), failing_create_pds).build_from_sabdab(
            [entry("1A2Y", h="Z", light="Y")]  # chains the structure does not have
        )
        skipped = json.loads((db / "skipped.json").read_text())
        assert set(skipped) == {"1A2Y_Z_H.pdb", "1A2Y_Y_L.pdb"}

        again = StubRcsb()
        builder = ChainDbBuilder(db, again, failing_create_pds)
        builder.build_from_sabdab([entry("1A2Y", h="Z", light="Y")])
        assert again.calls == []
        assert builder.report.skipped_known == 2

        retry = StubRcsb()
        ChainDbBuilder(db, retry, failing_create_pds, retry_skipped=True).build_from_sabdab(
            [entry("1A2Y", h="Z", light="Y")]
        )
        assert retry.calls == ["1A2Y"]

    def test_refresh_only_builder_cannot_download(
        self, tmp_path: Path, failing_create_pds: Path
    ) -> None:
        builder = ChainDbBuilder(tmp_path / "db", None, failing_create_pds)
        with pytest.raises(RuntimeError, match="without a downloader"):
            builder.build_from_sabdab([entry("1A2Y")])


class TestSAbDabReader:
    HEADER = "pdb\tHchain\tLchain\tmodel\tantigen_chain\tantigen_type\n"

    def test_tab_separated_official_format(self, tmp_path: Path) -> None:
        path = tmp_path / "sabdab_summary_all.tsv"
        path.write_text(
            self.HEADER
            + "1abc\tH\tL\t0\tA\tprotein\n"
            + "1abc\tH\tL\t0\tB\tprotein\n"      # same pair, another antigen chain
            + "2xyz\tH\tNA\t0\tA\tprotein\n"      # single-domain: no light chain
            + "3qqq\tNA\tNA\t0\tA\tprotein\n"     # nothing usable
        )
        got = [(e.pdb_id, e.hchain, e.lchain) for e in SAbDabMetadataReader(path).entries()]
        assert got == [("1ABC", "H", "L"), ("2XYZ", "H", "")]

    def test_comma_separated_export(self, tmp_path: Path) -> None:
        path = tmp_path / "sabdab.csv"
        path.write_text("pdb,Hchain,Lchain,antigen_chain\n8rtw,B,A,A\n")
        entries = SAbDabMetadataReader(path).entries()
        assert [(e.pdb_id, e.hchain, e.lchain) for e in entries] == [("8RTW", "B", "A")]

    def test_wrong_file_is_an_error_not_an_empty_database(self, tmp_path: Path) -> None:
        path = tmp_path / "other.csv"
        path.write_text("foo,bar\n1,2\n")
        with pytest.raises(ValueError, match="not a SAbDab summary file"):
            SAbDabMetadataReader(path).entries()

    def test_file_with_no_usable_rows_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.tsv"
        path.write_text(self.HEADER + "1abc\tNA\tNA\t0\tA\tprotein\n")
        with pytest.raises(ValueError, match="no usable rows"):
            SAbDabMetadataReader(path).entries()


@pytest.mark.needs_master
class TestVerify:
    def test_built_database_is_ok(self, mini_db_built: Path) -> None:
        check = verify_database(mini_db_built)
        assert check.ok, check.problems
        assert check.n_listed == check.n_pdb == check.n_pds

    def test_missing_target_is_a_problem(self, tmp_path: Path, create_pds: Path) -> None:
        db = tmp_path / "db"
        ChainDbBuilder(db, StubRcsb(), create_pds).build_from_sabdab([entry("1A2Y")])
        (db / "pds" / "1A2Y_B_H.pds").unlink()
        check = verify_database(db)
        assert not check.ok
        assert any("do not exist" in p for p in check.problems)

    def test_stale_target_is_a_warning_not_a_failure(
        self, tmp_path: Path, create_pds: Path
    ) -> None:
        db = tmp_path / "db"
        ChainDbBuilder(db, StubRcsb(), create_pds).build_from_sabdab([entry("1A2Y")])
        future = time.time() + 60
        os.utime(db / "pdb" / "1A2Y_B_H.pdb", (future, future))
        check = verify_database(db)
        assert check.ok
        assert any("older than their PDB" in w for w in check.warnings)

    def test_absent_directory(self, tmp_path: Path) -> None:
        check = verify_database(tmp_path / "nothing")
        assert not check.ok
