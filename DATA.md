# The chain database

The chain database goes here: `sabdab_chains.db/`, ~25 GB, 25,226 searchable
antibody chains from SAbDab.

**In this checkout `data/` is a symlink to `../epifitscout_v2/data`.** The two
versions share one copy rather than duplicating 25 GB. The consequence is worth
stating plainly, because it has already bitten this project once: *if the v2
directory is deleted or moved, v3 stops working*, and the failure looks like a
missing database rather than a broken link.

To make this checkout self-contained instead:

```bash
rm data                      # the symlink, not the directory it points at
mkdir data
tar -xJf /path/to/epifitscout-database-v1.0.0.tar.xz -C data/
```

The packaged dataset and its `SHA256SUMS` are in `downloads/dataset_release/`.
Verify before extracting:

```bash
cd downloads/dataset_release && sha256sum -c SHA256SUMS
```

Check either arrangement with:

```bash
epifitscout check --search
```
