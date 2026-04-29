# EpiFitScout

Protein structural fragment search with epitope shape complementarity scoring.

Given a query CDR loop and its bound epitope, EpiFitScout searches a chain database
derived from SAbDab for structurally similar CDR fragments that are geometrically
complementary to the epitope surface.

---

## Pipeline Overview

```
SAbDab CSV  →  build_chain_db.py  →  chain DB (pds.list)
                                              │
query PDB  →  QueryBuilder  →  search()  →  ranked hits
                (CDR + epitope)
```

Three internal steps:
1. **MASTER** — exact backbone RMSD search (sliding window over full chains)
2. **Shape scoring** — depth profile anti-correlation + backbone torsion co-correlation
3. **Ranking** — weighted combination of RMSD similarity and shape complementarity

---

## Requirements

- Python ≥ 3.10, managed with `uv`
- MASTER binaries compiled for your platform (`createPDS`, `master`)
  - **Binaries are platform-specific and not included in the repository**
  - Download for Linux/macOS from http://grigoryanlab.org/master/  
  - Linux version is available in MASTER directory. Run `tar -xvzf master-bin-v1.6.tar.gz`
  - Ensure `MASTER/bin/createPDS` and `MASTER/bin/master`
  - Ensure execute permission: `chmod +x MASTER/bin/*`
- SAbDab summary CSV → `data/SAbDab/sabdab_metadata.csv`
  - Download from https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/search/

**Install dependencies:**
```bash
uv sync
```

---

## Step 1 — Build the Chain Database

The database stores full variable-domain H/L chains from SAbDab in MASTER PDS format.
Downloads PDB files from RCSB automatically (requires internet access).
Build is resumable — already-processed entries are skipped.

**Mode A — SAbDab CSV (default):**
```bash
# Full database (~13k SAbDab entries → ~27k chains)
uv run python run/build_chain_db.py

# Development / test build (fast)
uv run python run/build_chain_db.py --limit 50

# Dry run — count entries, do not build
uv run python run/build_chain_db.py --dry_run
```

**Mode B — plain PDB list file:**
```bash
uv run python run/build_chain_db.py --pdb_list my_pdbs.txt
```

PDB list format (one entry per line):
```
1abc          # extract all protein chains
1abc H        # heavy chain only
1abc H L      # heavy and light chains
# comment lines and blank lines are ignored
```

**Custom paths:**
```bash
uv run python run/build_chain_db.py \
    --create_pds /path/to/createPDS \
    --metadata   /path/to/sabdab_metadata.csv \
    --output_dir /path/to/my.db \
    --rcsb_cache /path/to/pdb_cache
```

**Output layout:**
```
data/sabdab_chains.db/
├── pdb/            ← full-chain PDB files
├── pds/            ← MASTER binary format
├── pds.list        ← paths passed to MASTER at search time
├── db.list         ← paths to PDB files
└── metadata.json   ← per-chain annotation (pdb_id, chain, chain_type)
```

---

## Step 2 — Run a Search

### Load query fragments

```python
# insert path to EpiFitScout directory if you did not install it
import sys
sys.path.insert(0, ".../EpiFitScout") 
import epifitscout


qb = epifitscout.QueryBuilder("my_complex.pdb")
qb.describe()   # print all chains and residue ranges

query_cdr = qb.get_fragment("H", 100, 110)  # heavy chain CDR-H3
query_epi = qb.get_fragment("A",  45,  60)  # antigen epitope residues
```

### Search

```python
hits = epifitscout.search(query_cdr, query_epi)
```

Configuration is loaded automatically from `conf/config.yaml`.
Any keyword argument overrides the corresponding config value.

### Inspect results

```python
print(f'Total hits: {len(hits)}')
print(f'{"Rank":<5} {"PDB:chain":<14} {"Residues":<12} {"RMSD":>6}  {"Shape":>6}  {"Score":>7}')
print('-' * 52)
for i, h in enumerate(hits[:15], 1):
    res = f'{h.fragment.residue_range[0]}-{h.fragment.residue_range[1]}'
    print(f'{i:<5} {h.fragment.pdb_id}:{h.fragment.chain:<10} '
          f'{res:<12} '
          f'{h.rmsd_cdr:>6.3f}  '
          f'{h.complementarity_score:>6.3f}  '
          f'{h.final_score:>7.4f}')
```

### Hit fields

| Field | Type | Description |
|---|---|---|
| `h.fragment.pdb_id` | `str` | PDB ID of the database hit |
| `h.fragment.chain` | `str` | Chain ID of the hit |
| `h.fragment.residue_range` | `(int, int)` | Start–end residue numbers in the hit chain |
| `h.fragment.sequence` | `str` | One-letter amino acid sequence |
| `h.fragment.metadata["chain_type"]` | `"H"` or `"L"` | Heavy or light chain |
| `h.rmsd_cdr` | `float` | Backbone RMSD to query CDR (Å) |
| `h.complementarity_score` | `float` | Shape complementarity $S_\text{shape} \in [0,1]$ |
| `h.final_score` | `float` | Combined ranking score $S_\text{final} \in [0,1]$ |

---

## Configuration

All hyperparameters are in `conf/config.yaml` and auto-loaded by `search()`.

```yaml
# MASTER search
master:
  rmsd_threshold: 2.0   # Maximum backbone RMSD (Å) to retain a hit
  max_hits: 500          # Maximum number of hits returned by MASTER
  timeout_seconds: 420   # Per-query timeout in seconds
  n_threads: 0        # 0 = disabled; set to CPU count to enable (requires OpenMP build)

# Shape complementarity score
# S_shape = clip((weight_depth * Sd + weight_tau * S_tau + 1) / 2, 0, 1)
scoring:
  weight_depth: 0.7      # Weight for depth anti-correlation component (Sd)
  weight_tau: 0.3        # Weight for backbone torsion co-correlation (S_tau)

# Final ranking score
# S_final = weight_rmsd * 1/(1+RMSD) + weight_shape * S_shape
ranking:
  weight_rmsd: 0.4       # Weight for structural similarity (RMSD) term
  weight_shape: 0.6      # Weight for shape complementarity term

max_workers: 6
```

### Per-call override

```python
hits = epifitscout.search(
    query_cdr, query_epi,
    rmsd_threshold=1.0,   # stricter backbone match
    weight_depth=0.9,     # emphasise depth profile
    weight_tau=0.1,
)
```

### Tuning guidance

| Parameter | Effect |
|---|---|
| `rmsd_threshold` ↓ | Fewer hits, higher backbone similarity to query |
| `rmsd_threshold` ↑ | More hits, more structural diversity |
| `weight_depth` ↑ | Emphasise protrusion/recession complementarity |
| `weight_tau` → 0 | Recommended for short/flat loops (H1, L1) where torsion is uninformative |
| `weight_shape` ↑ | Prioritise surface fit over backbone similarity |
| `weight_rmsd` ↑ | Stay closer to the query backbone geometry |

---

