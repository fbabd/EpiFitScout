# The chain database goes in this folder

The database is what EpiFitScout searches: about 25,000 antibody heavy- and
light-chain structures taken from SAbDab, already converted for MASTER.

## 1. Download

Download `epifitscout-database-v<version>.tar.xz` from Zenodo:

> **TBD — link added when the dataset is published**

About 650 MB to download, about 6 GB once extracted (extraction takes under a minute
on local disk).

## 2. Extract

From the repository root, using the **full path** to where you saved the archive:

```bash
tar -xJf /path/to/epifitscout-database-v<version>.tar.xz -C data/
```

(A bare filename only works if the archive is in your current directory; `tar: …: Cannot open: No such file or directory` means it is not.)

You should end up with:

```
data/sabdab_chains.db/
├── pdb/            chain coordinates
├── pds/            MASTER search targets
├── pds.list
├── db.list
└── metadata.json
```

The archive already contains its `sabdab_chains.db` folder, so extract into
`data/`, not into a new folder you made. If you end up with
`data/something/sabdab_chains.db`, move the inner folder up one level.

## 3. Check

```bash
python check_setup.py
```

## Keeping it current, or building your own

See [`database_builder/README.md`](../database_builder/README.md). A database
built there is written to this same location.

> Put this folder on **fast local disk**. The same search took 2 s on local
> scratch storage and 162 s from a slow network filesystem.
