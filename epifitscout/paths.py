"""Where EpiFitScout looks for the two things you paste into the repository.

EpiFitScout is used from a clone of its repository. Two folders in that clone
start out holding only a README, and you fill them in::

    MASTER/bin/                    master, createPDS
    data/sabdab_chains.db/         pdb/  pds/  pds.list  db.list  metadata.json

Nothing is searched for anywhere else, so there is no hidden configuration to
debug. The paths are derived from the location of this file, which is why the
package must be installed in editable mode (``pip install -e .``): a regular
install copies the code out of the repository and leaves it unable to find the
folders above.

Two optional environment variables exist for the case where the data lives
elsewhere — a shared copy on a cluster, say:

``EPIFITSCOUT_DB``
    Full path to a ``pds.list`` file, instead of ``data/sabdab_chains.db/pds.list``.
``EPIFITSCOUT_MASTER_DIR``
    Directory holding ``master`` and ``createPDS``, instead of ``MASTER/bin``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "DB_DIR",
    "MASTER_BIN_DIR",
    "MASTER_DOWNLOAD_URL",
    "MissingBinaryError",
    "MissingDatabaseError",
    "REPO_ROOT",
    "SetupError",
    "default_config_path",
    "find_create_pds",
    "find_master_binary",
    "find_pds_list",
]

#: Root of the repository: the directory above the ``epifitscout`` package.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Folder the MASTER binaries are pasted into.
MASTER_BIN_DIR: Path = REPO_ROOT / "MASTER" / "bin"

#: Folder the chain database is extracted into.
DB_DIR: Path = REPO_ROOT / "data" / "sabdab_chains.db"

MASTER_DOWNLOAD_URL = "https://grigoryanlab.org/legacy.php?sec=download&soft=MASTER"
MASTER_ARCHIVE_NAME = "master-bin-v1.6.tar.gz"


class SetupError(RuntimeError):
    """A required piece of the installation is absent or in the wrong place."""


class MissingBinaryError(SetupError):
    """A MASTER executable is absent or not executable."""


class MissingDatabaseError(SetupError):
    """The chain database is absent or incomplete."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _is_source_checkout() -> bool:
    """True when this file sits inside a clone of the repository."""
    pyproject = REPO_ROOT / "pyproject.toml"
    try:
        return 'name = "epifitscout"' in pyproject.read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return False


def _install_hint() -> str:
    """A prominent message when the package was installed rather than cloned."""
    if _is_source_checkout():
        return ""
    return (
        "\nEpiFitScout appears to be installed without its repository "
        f"(package found in {REPO_ROOT}).\n"
        "It must be installed from a clone, in editable mode:\n"
        "    git clone <repository-url> && cd epifitscout && pip install -e .\n"
    )


def _usable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _master_dir() -> Path:
    env = os.environ.get("EPIFITSCOUT_MASTER_DIR")
    return Path(env).expanduser() if env else MASTER_BIN_DIR


def _misplaced_master_hint(name: str) -> str:
    """Spot the usual ways of pasting the MASTER files into the wrong folder."""
    base = REPO_ROOT / "MASTER"
    if not base.is_dir():
        return ""
    found = [p for p in base.rglob(name) if p.is_file() and p.parent != MASTER_BIN_DIR]
    if found:
        return (
            f"\nFound '{name}' at {found[0]}, but it must be directly inside "
            f"{MASTER_BIN_DIR}.\nMove the files there."
        )
    archives = list(base.glob("*.tar.gz"))
    if archives:
        return (
            f"\nFound {archives[0].name} but it has not been unpacked. From the "
            f"repository root run:\n    tar -xzf {archives[0]} -C MASTER/"
        )
    return ""


# --------------------------------------------------------------------------- #
# MASTER binaries
# --------------------------------------------------------------------------- #


def _locate_binary(name: str, explicit: Path | str | None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not _usable(path):
            raise MissingBinaryError(
                f"{name} was given as {path}, which is not an executable file."
            )
        return path.resolve()

    directory = _master_dir()
    candidate = directory / name
    if _usable(candidate):
        return candidate.resolve()

    if candidate.is_file():
        raise MissingBinaryError(
            f"{candidate} exists but is not executable. Run:\n"
            f"    chmod +x {directory}/*"
        )

    raise MissingBinaryError(
        f"MASTER's '{name}' was not found at {candidate}.\n\n"
        "MASTER is a separate program and is not part of this repository. Get it "
        "once:\n"
        f"  1. Download {MASTER_ARCHIVE_NAME} (the v1.6 BINARIES) from\n"
        f"     {MASTER_DOWNLOAD_URL}\n"
        "     Name, institution and email are optional; answer the arithmetic "
        "question.\n"
        "  2. From the repository root:\n"
        f"         tar -xzf {MASTER_ARCHIVE_NAME} -C MASTER/\n"
        "         chmod +x MASTER/bin/*\n"
        "     so that MASTER/bin/master and MASTER/bin/createPDS exist."
        f"{_misplaced_master_hint(name)}{_install_hint()}"
    )


def find_master_binary(explicit: Path | str | None = None) -> Path:
    """Return the ``master`` executable.

    Args:
        explicit: A path to use instead of ``MASTER/bin/master``.

    Raises:
        MissingBinaryError: If it is missing or not executable.
    """
    return _locate_binary("master", explicit)


def find_create_pds(
    explicit: Path | str | None = None, *, near: Path | None = None
) -> Path:
    """Return the ``createPDS`` executable.

    Args:
        explicit: A path to use instead of ``MASTER/bin/createPDS``.
        near: A resolved ``master`` binary; ``createPDS`` is taken from the same
            directory when it is there, so the pair always comes from one MASTER
            build.

    Raises:
        MissingBinaryError: If it is missing or not executable.
    """
    if explicit is None and near is not None and _usable(near.parent / "createPDS"):
        return (near.parent / "createPDS").resolve()
    return _locate_binary("createPDS", explicit)


# --------------------------------------------------------------------------- #
# Chain database
# --------------------------------------------------------------------------- #


def find_pds_list(explicit: Path | str | None = None) -> Path:
    """Return the ``pds.list`` file MASTER searches against.

    Args:
        explicit: A path to use instead of ``data/sabdab_chains.db/pds.list``.

    Raises:
        MissingDatabaseError: If the database is absent or incomplete.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise MissingDatabaseError(f"Database list not found: {path}")
        return path.resolve()

    env = os.environ.get("EPIFITSCOUT_DB")
    if env:
        path = Path(env).expanduser()
        if not path.is_file():
            raise MissingDatabaseError(
                f"EPIFITSCOUT_DB points at {path}, which does not exist."
            )
        return path.resolve()

    pds_list = DB_DIR / "pds.list"
    if pds_list.is_file():
        return pds_list.resolve()

    data_dir = REPO_ROOT / "data"
    nested = sorted(data_dir.glob("*/sabdab_chains.db/pds.list")) if data_dir.is_dir() else []
    if nested:
        raise MissingDatabaseError(
            f"The database was extracted one folder too deep: {nested[0].parent}\n"
            f"It must be exactly {DB_DIR}\n"
            f"Move it:  mv {nested[0].parent} {DB_DIR}"
            f"{_install_hint()}"
        )

    if DB_DIR.is_dir():
        have = ", ".join(sorted(p.name for p in DB_DIR.iterdir())[:8]) or "(empty)"
        raise MissingDatabaseError(
            f"{DB_DIR} exists but has no pds.list. It contains: {have}\n"
            "A complete database holds pdb/, pds/, pds.list, db.list and "
            "metadata.json.\n"
            "Re-extract the dataset archive, or rebuild/refresh it with "
            "database_builder/build_database.py."
            f"{_install_hint()}"
        )

    raise MissingDatabaseError(
        f"Chain database not found at {DB_DIR}\n\n"
        "Download the dataset archive (see data/README.md for the link) and, "
        "from the repository root:\n"
        "    tar -xJf <downloaded-file>.tar.xz -C data/\n"
        f"so that {DB_DIR}/pds.list exists.\n"
        "Or build it yourself: database_builder/README.md."
        f"{_install_hint()}"
    )


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def default_config_path() -> Path:
    """Return the packaged ``epifitscout/conf/config.yaml``.

    Raises:
        SetupError: If it is missing, which means a damaged checkout. This is an
            error rather than a silent fall-back to built-in defaults, because
            those differ from the shipped values and would change results.
    """
    path = Path(__file__).resolve().parent / "conf" / "config.yaml"
    if not path.is_file():
        raise SetupError(
            f"The default configuration is missing: {path}\n"
            "Restore it from the repository (git checkout -- epifitscout/conf)."
        )
    return path
