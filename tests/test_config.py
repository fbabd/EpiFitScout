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
        """The old code silently ignored the file when it could not find it."""
        raw = yaml.safe_load(paths.default_config_path().read_text())
        cfg = epifitscout.config_from_yaml()
        assert cfg.master.rmsd_threshold == raw["master"]["rmsd_threshold"]
        assert cfg.feasibility.bond_max == raw["feasibility"]["bond_max"]
        assert cfg.interface.hbond_distance == raw["interface"]["hbond_distance"]
        assert cfg.shape.weight_depth == raw["shape"]["weight_depth"]
        assert cfg.epitope_cutoff == raw["epitope_cutoff"]
        assert cfg.pool_size == raw["pool_size"]
        assert cfg.max_workers == raw["max_workers"]

    def test_a_value_in_the_file_overrides_the_dataclass(self, tmp_path: Path) -> None:
        """The teeth of the test above, now that file and dataclass agree.

        The shipped YAML and the dataclass defaults are deliberately the same,
        so matching values no longer prove the file was read. A distinctive
        value does.
        """
        config = tmp_path / "c.yaml"
        config.write_text("master:\n  rmsd_threshold: 0.37\npool_size: 7\n")
        cfg = epifitscout.config_from_yaml(config)
        assert cfg.master.rmsd_threshold == 0.37
        assert cfg.pool_size == 7

    def test_every_shipped_section_is_a_real_section(self) -> None:
        """A section the loader does not know would be silently ignored."""
        raw = yaml.safe_load(paths.default_config_path().read_text())
        sections = {k for k, v in raw.items() if isinstance(v, dict)}
        assert sections == {"master", "feasibility", "interface", "shape"}


class TestExplicitConfig:
    def test_partial_file_uses_defaults_for_the_rest(self, tmp_path: Path) -> None:
        partial = tmp_path / "partial.yaml"
        partial.write_text("master:\n  rmsd_threshold: 1.25\n")
        cfg = epifitscout.config_from_yaml(partial)
        assert cfg.master.rmsd_threshold == 1.25
        assert cfg.feasibility.bond_max == 1.8          # untouched default

    def test_empty_file_is_all_defaults(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        cfg = epifitscout.config_from_yaml(empty)
        assert cfg.master.rmsd_threshold == MASTERConfig().rmsd_threshold
        assert cfg.pool_size == PipelineConfig().pool_size

    def test_an_unknown_key_is_an_error(self, tmp_path: Path) -> None:
        """A misspelt parameter must not quietly leave the default in place.

        Silently ignoring it would make the run look like a result for the
        setting that was asked for, which is worse than failing.
        """
        bad = tmp_path / "bad.yaml"
        bad.write_text("shape:\n  weight_dpeth: 0.7\n")        # weight_depth
        with pytest.raises(paths.SetupError, match="weight_dpeth"):
            epifitscout.config_from_yaml(bad)


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
