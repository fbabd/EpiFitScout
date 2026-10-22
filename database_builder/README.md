# Building and updating the chain database

The dataset EpiFitScout searches is built from two public sources:

* **SAbDab** decides *which* chains: every antibody heavy and light chain it lists.
* **RCSB PDB** supplies the *coordinates*.

`build_database.py` turns those into the folder `data/sabdab_chains.db/`. You need
this when you want a newer snapshot than the downloadable dataset, or a custom
set of structures. If you only want to search, you do not.

## Update workflow

```bash
# 1. Get a current SAbDab summary and save it here as sabdab_summary.tsv
#       https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/
#    (the file SAbDab distributes is tab-separated; a comma-separated export of
#     the same columns also works)

# 2. See what would change. Nothing is downloaded or written.
python database_builder/build_database.py --dry-run

# 3. Apply it
python database_builder/build_database.py

# 4. Confirm
python check_setup.py
```

Step 3 is safe to interrupt and re-run; finished work is kept.

## What an update does, and does not do

| Situation | Behaviour |
|---|---|
| Chain already in the database | Left alone. **Not** re-downloaded. |
| New chain | Structure downloaded, chain cut out, search target built. |
| Search target missing, empty, or older than its coordinates | Rebuilt. |
| Chain that cannot be extracted (too short, absent from the file…) | Recorded in `skipped.json` and not retried each run. `--retry-skipped` retries them. |
| Download fails (network) | Not recorded as skipped; retried next run. |
| `createPDS` fails for a chain | No partial file is kept; the chain is left out of `pds.list` and retried next run. |
| Entry no longer in SAbDab | Left in place. Obsolete structures are still valid, searchable structures. |
| Search target with no coordinates file | Deleted. |

**The search targets (`pds/`) must be refreshed whenever the coordinates change or
you change MASTER version.** The first is automatic. For the second:

```bash
python database_builder/build_database.py --rebuild-pds
```

## Options

| Flag | Purpose |
|---|---|
| `--metadata FILE` | SAbDab summary to read (default `database_builder/sabdab_summary.tsv`) |
| `--pdb-list FILE` | Build from your own list instead: lines like `1abc`, `1abc H`, `1abc H L` |
| `--output-dir DIR` | Database location (default `data/sabdab_chains.db`) |
| `--limit N` | Process only the first N entries — a quick test build |
| `--dry-run` | Report what would be added; change nothing |
| `--rebuild-pds` | Regenerate every search target |
| `--retry-skipped` | Re-attempt chains previously recorded as unextractable |
| `--workers N` | Parallel `createPDS` processes (default 4) |
| `--rcsb-cache DIR` | Where downloaded structures are kept; safe to delete afterwards |

## Cost

A first build from the full SAbDab downloads ~13,000 entries (roughly 8 GB from
RCSB, kept in `database_builder/rcsb_cache/`) and takes hours, mostly network.
**Updates are cheap**: only new entries are fetched, and the existing database
needs no cache at all.

## Publishing a dataset

To package your database as a single archive for others to download:

```bash
python database_builder/package_dataset.py --version 1.0.0 --out-dir ../dataset_release
```

It verifies the database first, writes `epifitscout-database-v1.0.0.tar.xz` (which
extracts straight into `data/`), a `SHA256SUMS` file, and a ready-made
`README.md` for the hosting page. Compression of the full dataset takes a while.

## Files in this folder

| File | Purpose |
|---|---|
| `build_database.py` | Build or update the database |
| `package_dataset.py` | Package a database as a downloadable archive |
| `sabdab_summary.tsv` | **You supply this**; not in the repository |
| `rcsb_cache/` | Created on first run; not in the repository |
