# The chain database

`data/sabdab_chains.db/` holds 25,226 searchable antibody chains from SAbDab,
about 25 GB unpacked. It is not in the repository.

This checkout is **self-contained**: `data/` is a real directory with its own
copy. An earlier arrangement symlinked it into a neighbouring checkout to save
disk, and that is why this file exists — the neighbour was deleted, the link
dangled, and the failure presented as a missing database rather than a broken
link. Do not reintroduce it.

## Setting it up

The packaged dataset and its checksums are in `downloads/dataset_release/`.
Verify before extracting; a truncated archive extracts without complaint and
fails much later, during a search:

```bash
cd downloads/dataset_release && sha256sum -c SHA256SUMS
```

Then:

```bash
mkdir -p data
tar -xJf /path/to/epifitscout-database-v1.0.0.tar.xz -C data/
```

## Checking it

```bash
epifitscout check --search
```

`check` without `--search` verifies the files are in place; `--search` also runs
a real query end to end, which is the only thing that proves the PDS targets are
usable. Expect ~3 minutes on an idle machine. Both counts below must read
25,226 — a short `pds/` count means the extraction did not finish, and `tar`
exiting 0 is not sufficient evidence that it did:

```bash
ls data/sabdab_chains.db/pdb | wc -l      # 25226
ls data/sabdab_chains.db/pds | wc -l      # 25226
```
