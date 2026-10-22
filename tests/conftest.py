"""Shared fixtures.

The suite is built so the parts that do not need MASTER run anywhere — a
laptop, CI, a reviewer's machine — while the parts that drive the real binaries
are marked ``needs_master`` and skip cleanly when it is absent.

``tests/data/mini_db`` holds 14 real antibody chains (~400 kB), including two
anti-lysozyme Fabs, so a genuine end-to-end search over a real database runs in
seconds instead of needing the full ~6 GB dataset.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
MINI_DB_SOURCE = TESTS_DIR / "data" / "mini_db"
EXAMPLE_PDB = REPO_ROOT / "example" / "1a2y.pdb"

sys.path.insert(0, str(REPO_ROOT))


def _find_master_dir() -> Path | None:
    """Locate a directory holding both MASTER binaries.

    Deliberately independent of ``epifitscout.paths``, so a bug in resolution
    cannot make the tests that depend on MASTER silently skip.
    """
    candidates = []
    if env := os.environ.get("EPIFITSCOUT_MASTER_DIR"):
        candidates.append(Path(env))
    candidates.append(REPO_ROOT / "MASTER" / "bin")
    for directory in candidates:
        master, create = directory / "master", directory / "createPDS"
        if master.is_file() and create.is_file() and os.access(master, os.X_OK):
            return directory
    return None


MASTER_DIR = _find_master_dir()


def pytest_collection_modifyitems(config, items):
    """Skip ``needs_master`` tests when MASTER is not available."""
    if MASTER_DIR is not None:
        return
    skip = pytest.mark.skip(
        reason="needs the MASTER binaries: paste them into MASTER/bin or set "
               "EPIFITSCOUT_MASTER_DIR"
    )
    for item in items:
        if "needs_master" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a developer's own environment variables leaking into a test."""
    monkeypatch.delenv("EPIFITSCOUT_DB", raising=False)
    monkeypatch.delenv("EPIFITSCOUT_MASTER_DIR", raising=False)


@pytest.fixture
def master_env(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point resolution at the available MASTER build."""
    if MASTER_DIR is None:
        pytest.skip("MASTER binaries not available")
    monkeypatch.setenv("EPIFITSCOUT_MASTER_DIR", str(MASTER_DIR))
    return MASTER_DIR


@pytest.fixture
def empty_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A pristine repository layout: the placeholders exist and are empty.

    Redirects the location constants in ``epifitscout.paths`` so the
    missing-resource tests do not depend on whether the developer running them
    has pasted MASTER or the dataset into the real checkout.
    """
    from epifitscout import paths

    root = tmp_path / "repo"
    (root / "MASTER" / "bin").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "epifitscout"\n')
    monkeypatch.setattr(paths, "REPO_ROOT", root)
    monkeypatch.setattr(paths, "MASTER_BIN_DIR", root / "MASTER" / "bin")
    monkeypatch.setattr(paths, "DB_DIR", root / "data" / "sabdab_chains.db")
    return root


@pytest.fixture(scope="session")
def example_pdb() -> Path:
    if not EXAMPLE_PDB.is_file():
        pytest.skip(f"example structure missing: {EXAMPLE_PDB}")
    return EXAMPLE_PDB


@pytest.fixture
def mini_db_pdb_only(tmp_path: Path) -> Path:
    """A fresh copy of the fixture database: chain PDBs but no PDS targets."""
    dest = tmp_path / "data" / "sabdab_chains.db"
    shutil.copytree(MINI_DB_SOURCE, dest)
    return dest


@pytest.fixture(scope="session")
def mini_db_built(tmp_path_factory) -> Path:
    """The fixture database with PDS targets built once for the whole session."""
    if MASTER_DIR is None:
        pytest.skip("MASTER binaries not available")
    from epifitscout.db.chain_db_builder import ChainDbBuilder

    dest = tmp_path_factory.mktemp("mini_db_built") / "sabdab_chains.db"
    shutil.copytree(MINI_DB_SOURCE, dest)
    ChainDbBuilder(dest, None, MASTER_DIR / "createPDS", workers=2).sync_pds()
    return dest
