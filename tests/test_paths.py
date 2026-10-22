"""Where EpiFitScout looks for MASTER and the database, and what it says when
they are not there.

The error messages are the interface here: pasting a file into the wrong folder
is the most likely way for a new user to get stuck, so each failure has to say
which folder, which file and which command.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from epifitscout import paths


def _make_binary(path: Path, *, executable: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755 if executable else 0o644)
    return path


class TestLocations:
    def test_defaults_are_inside_the_repository(self) -> None:
        assert paths.MASTER_BIN_DIR == paths.REPO_ROOT / "MASTER" / "bin"
        assert paths.DB_DIR == paths.REPO_ROOT / "data" / "sabdab_chains.db"

    def test_repo_root_is_the_checkout(self) -> None:
        assert (paths.REPO_ROOT / "pyproject.toml").is_file()
        assert (paths.REPO_ROOT / "epifitscout" / "paths.py").is_file()


class TestMasterBinaries:
    def test_found_in_the_placeholder_folder(self, empty_repo: Path) -> None:
        master = _make_binary(empty_repo / "MASTER" / "bin" / "master")
        create = _make_binary(empty_repo / "MASTER" / "bin" / "createPDS")
        assert paths.find_master_binary() == master.resolve()
        assert paths.find_create_pds() == create.resolve()

    def test_missing_names_url_file_and_folder(self, empty_repo: Path) -> None:
        with pytest.raises(paths.MissingBinaryError) as excinfo:
            paths.find_master_binary()
        message = str(excinfo.value)
        assert "legacy.php?sec=download&soft=MASTER" in message
        assert "master-bin-v1.6.tar.gz" in message
        assert "tar -xzf" in message
        assert str(empty_repo / "MASTER" / "bin") in message

    def test_create_pds_message_is_equally_complete(self, empty_repo: Path) -> None:
        with pytest.raises(paths.MissingBinaryError) as excinfo:
            paths.find_create_pds()
        assert "master-bin-v1.6.tar.gz" in str(excinfo.value)

    def test_not_executable_says_chmod(self, empty_repo: Path) -> None:
        """A tarball unpacked without the execute bit is a common stumble."""
        _make_binary(empty_repo / "MASTER" / "bin" / "master", executable=False)
        with pytest.raises(paths.MissingBinaryError, match="chmod"):
            paths.find_master_binary()

    def test_binary_one_folder_too_deep_is_diagnosed(self, empty_repo: Path) -> None:
        _make_binary(empty_repo / "MASTER" / "master-bin-v1.6" / "bin" / "master")
        with pytest.raises(paths.MissingBinaryError, match="must be directly inside"):
            paths.find_master_binary()

    def test_unextracted_archive_is_diagnosed(self, empty_repo: Path) -> None:
        (empty_repo / "MASTER" / "master-bin-v1.6.tar.gz").write_bytes(b"x")
        with pytest.raises(paths.MissingBinaryError, match="has not been unpacked"):
            paths.find_master_binary()

    def test_env_directory_override(
        self, empty_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elsewhere = tmp_path / "shared-master"
        master = _make_binary(elsewhere / "master")
        monkeypatch.setenv("EPIFITSCOUT_MASTER_DIR", str(elsewhere))
        assert paths.find_master_binary() == master.resolve()

    def test_explicit_path_wins(self, empty_repo: Path, tmp_path: Path) -> None:
        mine = _make_binary(tmp_path / "mine" / "master")
        assert paths.find_master_binary(mine) == mine.resolve()

    def test_explicit_bad_path_is_an_error(self, empty_repo: Path, tmp_path: Path) -> None:
        with pytest.raises(paths.MissingBinaryError):
            paths.find_master_binary(tmp_path / "nope")

    def test_create_pds_prefers_the_directory_of_master(
        self, empty_repo: Path, tmp_path: Path
    ) -> None:
        """Both binaries must come from one MASTER build, or the PDS files mismatch."""
        _make_binary(empty_repo / "MASTER" / "bin" / "createPDS")
        other = tmp_path / "other"
        master = _make_binary(other / "master")
        create = _make_binary(other / "createPDS")
        assert paths.find_create_pds(near=master) == create.resolve()


class TestDatabase:
    def test_found_when_extracted_correctly(self, empty_repo: Path) -> None:
        db = empty_repo / "data" / "sabdab_chains.db"
        db.mkdir()
        (db / "pds.list").write_text("pds/x.pds\n")
        assert paths.find_pds_list() == (db / "pds.list").resolve()

    def test_missing_points_at_the_readme_and_the_command(self, empty_repo: Path) -> None:
        with pytest.raises(paths.MissingDatabaseError) as excinfo:
            paths.find_pds_list()
        message = str(excinfo.value)
        assert "data/README.md" in message
        assert "tar -xJf" in message
        assert str(empty_repo / "data" / "sabdab_chains.db") in message

    def test_extracted_one_level_too_deep_is_diagnosed(self, empty_repo: Path) -> None:
        """Extracting into a folder you made yourself is the likeliest mistake."""
        nested = empty_repo / "data" / "my_download" / "sabdab_chains.db"
        nested.mkdir(parents=True)
        (nested / "pds.list").write_text("pds/x.pds\n")
        with pytest.raises(paths.MissingDatabaseError, match="one folder too deep"):
            paths.find_pds_list()

    def test_incomplete_directory_lists_what_it_does_hold(self, empty_repo: Path) -> None:
        db = empty_repo / "data" / "sabdab_chains.db"
        (db / "pdb").mkdir(parents=True)
        with pytest.raises(paths.MissingDatabaseError, match="no pds.list") as excinfo:
            paths.find_pds_list()
        assert "pdb" in str(excinfo.value)

    def test_env_override(
        self, empty_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shared = tmp_path / "shared.list"
        shared.write_text("pds/x.pds\n")
        monkeypatch.setenv("EPIFITSCOUT_DB", str(shared))
        assert paths.find_pds_list() == shared.resolve()

    def test_env_typo_is_an_error_not_a_fallback(
        self, empty_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = empty_repo / "data" / "sabdab_chains.db"
        db.mkdir()
        (db / "pds.list").write_text("pds/x.pds\n")
        monkeypatch.setenv("EPIFITSCOUT_DB", str(tmp_path / "typo.list"))
        with pytest.raises(paths.MissingDatabaseError, match="EPIFITSCOUT_DB"):
            paths.find_pds_list()


class TestNotACheckout:
    """A regular ``pip install`` copies the code out and loses the repository."""

    def test_errors_explain_the_editable_install(
        self, empty_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (empty_repo / "pyproject.toml").unlink()
        with pytest.raises(paths.MissingBinaryError, match="pip install -e"):
            paths.find_master_binary()
        with pytest.raises(paths.MissingDatabaseError, match="pip install -e"):
            paths.find_pds_list()

    def test_no_hint_in_a_real_checkout(self, empty_repo: Path) -> None:
        with pytest.raises(paths.MissingBinaryError) as excinfo:
            paths.find_master_binary()
        assert "pip install -e" not in str(excinfo.value)


class TestConfig:
    def test_default_config_is_present(self) -> None:
        config = paths.default_config_path()
        assert config.is_file() and config.name == "config.yaml"

    def test_missing_config_raises_not_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(paths, "__file__", str(tmp_path / "paths.py"))
        with pytest.raises(paths.SetupError, match="default configuration is missing"):
            paths.default_config_path()


def test_hierarchy() -> None:
    assert issubclass(paths.MissingBinaryError, paths.SetupError)
    assert issubclass(paths.MissingDatabaseError, paths.SetupError)
    assert os.environ.get("EPIFITSCOUT_DB") is None  # the autouse isolation works
