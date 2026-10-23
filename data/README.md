# The chain database goes in this folder

EpiFitScout searches 25,226 antibody heavy and light chains from SAbDab. The
database is ~25 GB unpacked and is not in the repository, so you add it once.

## 1. Download

The packaged dataset is on Zenodo:

> **`epifitscout-database-v1.0.0.tar.xz`** — 645 MB compressed, ~25 GB unpacked

Download it together with `SHA256SUMS` from the same record.

## 2. Verify before unpacking

Do not skip this. A truncated download extracts without any complaint and then
fails much later, in the middle of a search, as a missing-target error that
looks like a bug in the tool:

```bash
sha256sum -c SHA256SUMS
```

```
epifitscout-database-v1.0.0.tar.xz: OK
```

## 3. Unpack

From the repository root:

```bash
tar -xJf epifitscout-database-v1.0.0.tar.xz -C data/
```

You should end up with:

```
data/sabdab_chains.db/
├── pdb/            25226 chain coordinate files
├── pds/            25226 MASTER search targets
├── db.list
├── pds.list
└── metadata.json
```

**Both counts must read 25226.** `tar` exiting 0 is not sufficient evidence that
it finished — check:

```bash
ls data/sabdab_chains.db/pdb | wc -l      # 25226
ls data/sabdab_chains.db/pds | wc -l      # 25226
```

## 4. Check

```bash
epifitscout check --search
```

`check` alone confirms the files are in place. `--search` additionally runs a
real query end to end, which is the only thing that proves the search targets
are usable; it takes about three minutes on an idle machine.

## Notes

**Keep this a real directory.** Symlinking it at another checkout to save disk
is a trap: when the other checkout moves or is deleted the link dangles, and the
failure presents as a missing database rather than a broken link.

**Building it yourself.** If you want a newer SAbDab snapshot or a custom set of
structures, [`database_builder/`](../database_builder/README.md) rebuilds this
layout from SAbDab and RCSB. It takes a few hours, mostly download. You do not
need it if you only want to search.
