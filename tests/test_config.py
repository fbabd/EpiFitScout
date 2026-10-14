"""Configuration loading, and the silent-fallback bug it used to have."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import epifitscout
from epifitscout import paths
from epifitscout.config.schema import MASTERConfig, PipelineConfig


class TestPackagedDefaults:
    def test_loads_the_shipped_yaml(self) -> None:
        cfg = epifitscout.config_from_yaml()
        assert isinstance(cfg, PipelineConfig)
        assert cfg.master.rmsd_threshold > 0
        assert cfg.max_workers >= 1

    def test_shipped_yaml_values_are_actually_used(self) -> None:
        """The old code silently ignored the file when it could not find it.

        The YAML says rmsd_threshold 2.0 while the dataclass default is 1.5, so
        this assertion fails if the file is ever skipped again.
        """
        raw = yaml.safe_load(paths.default_config_path().read_text())
        cfg = epifitscout.config_from_yaml()
        assert cfg.master.rmsd_threshold == raw["master"]["rmsd_threshold"]
        assert cfg.scoring.weight_depth == raw["scoring"]["weight_depth"]
        assert cfg.ranking.weight_shape == raw["ranking"]["weight_shape"]
        assert cfg.max_workers == raw["max_workers"]


class TestExplicitConfig:
    def test_partial_file_uses_defaults_for_the_rest(self, tmp_path: Path) -> None:
        partial = tmp_path / "partial.yaml"
        partial.write_text("master:\n  rmsd_threshold: 1.25\n")
        cfg = epifitscout.config_from_yaml(partial)
        assert cfg.master.rmsd_threshold == 1.25
        assert cfg.scoring.weight_depth == 0.7

    def test_empty_file_is_all_defaults(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        cfg = epifitscout.config_from_yaml(empty)
        assert cfg.master.rmsd_threshold == 1.5


class TestSchemaDefaults:
    def test_binary_and_database_default_to_auto_resolution(self) -> None:
        """Relative string defaults made behaviour depend on the current directory."""
        cfg = MASTERConfig()
        assert cfg.binary_path is None
        assert cfg.database_path is None

    def test_pipeline_config_is_constructible_with_no_arguments(self) -> None:
        cfg = PipelineConfig()
        assert cfg.master.binary_path is None
        assert cfg.log_level == "INFO"


class TestOverridePrecedence:
    def test_keyword_overrides_beat_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Check _build_config without needing a database or binaries present."""
        config = tmp_path / "c.yaml"
        config.write_text(
            "master:\n  rmsd_threshold: 2.0\n  max_hits: 500\n"
            "scoring:\n  weight_depth: 0.7\n  weight_tau: 0.3\n"
            "ranking:\n  weight_rmsd: 0.4\n  weight_shape: 0.6\n"
            "max_workers: 6\n"
        )
        fake_bin = tmp_path / "master"
        fake_bin.write_text("#!/bin/sh\n")
        fake_bin.chmod(0o755)
        fake_db = tmp_path / "pds.list"
        fake_db.write_text("pds/x.pds\n")

        monkeypatch.setattr(paths, "find_master_binary", lambda *a, **k: fake_bin)
        monkeypatch.setattr(paths, "find_pds_list", lambda *a, **k: fake_db)
        monkeypatch.setattr(paths, "find_create_pds", lambda *a, **k: fake_bin)

        cfg = epifitscout._build_config(
            rmsd_threshold=0.8, max_hits=None, weight_depth=0.95, weight_tau=None,
            weight_rmsd=None, weight_shape=None, binary_path=None, database_path=None,
            config_path=config, log_level=None, max_workers=2,
        )
        assert cfg.master.rmsd_threshold == 0.8      # overridden
        assert cfg.master.max_hits == 500            # from file
        assert cfg.scoring.weight_depth == 0.95      # overridden
        assert cfg.scoring.weight_tau == 0.3         # from file
        assert cfg.max_workers == 2                  # overridden
