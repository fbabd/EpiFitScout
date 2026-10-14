#!/usr/bin/env python3
"""Package a built database as a single archive for people to download.

Run this after build_database.py when you want to publish a new dataset (for
example to Zenodo). The archive holds one top-level folder, ``sabdab_chains.db``,
so a user extracts it into ``data/`` and is done:

    tar -xJf epifitscout-database-<version>.tar.xz -C data/

It contains everything a search needs — chain PDBs, MASTER targets and lists —
so there is no build step for the user. The database is verified first, and the
archive is re-opened and checked after writing.

    python database_builder/package_dataset.py --version 1.0.0 \\
        --out-dir ../dataset_release

Writes into --out-dir: the archive, SHA256SUMS, and README.md (a ready-made
description for the hosting page).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from epifitscout import paths  # noqa: E402
from epifitscout.db.verify import verify_database  # noqa: E402

DB_NAME = "sabdab_chains.db"
_TOP_LEVEL_FILES = ["pds.list", "db.list", "metadata.json", "skipped.json", "BUILD_INFO.json"]


def human(n: float) -> str:
    return f"{n / 1e6:.1f} MB" if n < 1e9 else f"{n / 1e9:.2f} GB"


def say(message: str = "") -> None:
    print(message, flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def member_list(db: Path) -> list[str]:
    """Database-relative paths to ship, in a stable order."""
    members: list[str] = []
    for sub in ("pdb", "pds"):
        members += [f"{sub}/{p.name}" for p in sorted((db / sub).glob(f"*.{sub}"))]
    members += [name for name in _TOP_LEVEL_FILES if (db / name).is_file()]
    return members


def build_archive(db: Path, out: Path, members: list[str], level: int, threads: int) -> None:
    """Stream tar through multi-threaded xz; fall back to Python's lzma."""
    out.unlink(missing_ok=True)
    if shutil.which("tar") and shutil.which("xz"):
        listing = out.with_suffix(".members.txt")
        listing.write_text("".join(f"{m}\n" for m in members))
        try:
            tar = subprocess.Popen(
                ["tar", "--create", "--file", "-", "--directory", str(db),
                 "--files-from", str(listing), "--transform", f"s|^|{DB_NAME}/|",
                 "--owner", "0", "--group", "0", "--numeric-owner"],
                stdout=subprocess.PIPE,
            )
            with open(out, "wb") as sink:
                xz = subprocess.Popen(
                    ["xz", f"-{level}", f"-T{threads}", "-c"], stdin=tar.stdout, stdout=sink
                )
                assert tar.stdout is not None
                tar.stdout.close()
                xz_rc, tar_rc = xz.wait(), tar.wait()
        finally:
            listing.unlink(missing_ok=True)
        if xz_rc or tar_rc:
            out.unlink(missing_ok=True)
            raise SystemExit(f"archive build failed (tar={tar_rc}, xz={xz_rc})")
        return

    say("  (GNU tar/xz not found; using single-threaded Python lzma — this is slow)")
    with tarfile.open(out, "w:xz", preset=level) as tar_py:
        for m in members:
            info = tar_py.gettarinfo(db / m, arcname=f"{DB_NAME}/{m}")
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with open(db / m, "rb") as fh:
                tar_py.addfile(info, fh)


def verify_archive(archive: Path, expected: int) -> None:
    with tarfile.open(archive, "r:xz") as tar:
        names = [n for n in tar.getnames() if not n.endswith("/")]
    roots = {Path(n).parts[0] for n in names}
    if roots != {DB_NAME}:
        raise SystemExit(f"archive has unexpected top-level entries: {sorted(roots)}")
    if len(names) != expected:
        raise SystemExit(f"archive holds {len(names)} files, expected {expected}")


_README = """\
# EpiFitScout chain database {version}

Search database for [EpiFitScout]({homepage}): {n_chains} antibody variable-domain
chains (heavy and light) cut from PDB entries listed in SAbDab, ready to search.

## Use

From the root of an EpiFitScout clone, give `tar` the **full path** to wherever
you saved the archive (a bare filename only works if the archive is in your
current directory):

```bash
tar -xJf /path/to/{filename} -C data/
python check_setup.py
```

For example, if you saved it one level above the clone: `tar -xJf ../{filename} -C data/`.

That produces `data/sabdab_chains.db/`. Nothing else to build.

## Contents

```
sabdab_chains.db/
├── pdb/            {n_chains} chain coordinate files   ({{PDBID}}_{{CHAIN}}_{{TYPE}}.pdb)
├── pds/            {n_chains} MASTER search targets
├── pds.list        targets searched by MASTER (paths relative to this folder)
├── db.list         coordinate files, same order
├── metadata.json   per-chain annotation (pdb_id, chain, chain_type)
├── skipped.json    chains that could not be extracted, so updates do not retry them
└── BUILD_INFO.json when and from which SAbDab file this was built
```

Built from {sabdab_file} on {built}. The MASTER targets (`pds/`) were generated
with MASTER v1.6's `createPDS`; use the same MASTER version to search them.

## Provenance and licence

Coordinates come from the [RCSB PDB](https://www.rcsb.org/) (public domain); the
entries were selected with [SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/)
(Dunbar et al., *Nucleic Acids Res* 42:D1140, 2014). MASTER (Zhou & Grigoryan,
*Protein Sci* 24:508, 2015) is required to search and is **not** included.

To rebuild or update this database yourself, see `database_builder/` in the
repository.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", required=True, help="Dataset version label, e.g. 1.0.0")
    p.add_argument("--out-dir", type=Path, required=True, help="Directory to write into.")
    p.add_argument("--db-dir", type=Path, default=paths.DB_DIR,
                   help="Database to package. Default: %(default)s")
    p.add_argument("--homepage", default="https://github.com/fbabd/EpiFitScout")
    p.add_argument("--level", type=int, default=9, choices=range(1, 10),
                   help="xz level; lower is faster. Default: %(default)s")
    p.add_argument("--threads", type=int, default=0, help="xz threads; 0 = all cores.")
    args = p.parse_args(argv)

    db = args.db_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir == db or db in out_dir.parents:
        raise SystemExit("--out-dir must not be inside the database directory")

    say(f"Verifying {db}")
    check = verify_database(db)
    for w in check.warnings:
        say(f"  warning: {w}")
    if not check.ok:
        say("The database is not usable, so it will not be packaged:")
        for problem in check.problems:
            say(f"  - {problem}")
        return 1
    say(f"  OK: {check.n_listed} searchable chains")

    members = member_list(db)
    raw = sum((db / m).stat().st_size for m in members)
    say(f"Archiving {len(members)} files, {human(raw)} uncompressed (this takes a while)")

    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"epifitscout-database-v{args.version}.tar.xz"
    archive = out_dir / filename
    started = time.monotonic()
    build_archive(db, archive, members, args.level, args.threads or (os.cpu_count() or 1))
    say(f"  wrote {filename}: {human(archive.stat().st_size)} "
        f"({100 * archive.stat().st_size / max(raw, 1):.0f}% of raw) "
        f"in {time.monotonic() - started:.0f}s")

    say("Verifying the archive")
    verify_archive(archive, len(members))
    digest = sha256_of(archive)
    (out_dir / "SHA256SUMS").write_text(f"{digest}  {filename}\n")

    info = check.info or {}
    (out_dir / "README.md").write_text(_README.format(
        version=args.version, filename=filename, n_chains=check.n_listed,
        homepage=args.homepage, sabdab_file=info.get("input", "a SAbDab summary file"),
        built=info.get("updated_at", "an unrecorded date")[:10],
    ))
    (out_dir / "MANIFEST.json").write_text(json.dumps(
        {"version": args.version, "filename": filename, "sha256": digest,
         "size_bytes": archive.stat().st_size, "n_chains": check.n_listed, "build": info},
        indent=2) + "\n")

    say()
    say(f"Ready in {out_dir}:")
    for item in sorted(out_dir.iterdir()):
        say(f"  {item.name:<40} {human(item.stat().st_size)}")
    say(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
