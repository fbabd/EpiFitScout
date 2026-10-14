# EpiFitScout

Protein structural fragment search with epitope shape-complementarity scoring.

Give it a complex, mark the CDR loop and the epitope. It searches ~25,000
antibody chains from SAbDab for loops that are both structurally similar to
yours and geometrically complementary to the epitope surface — then grafts the
best ones back in and hands you finished structures.

```
query PDB ──▶ CDR + epitope ──▶ search() ──▶ ranked hits ──▶ grafted complexes
                                   │                          (epifitscout build)
                                   ├─ 1. MASTER  backbone RMSD search
                                   ├─ 2. graft   place each hit in the Ab–Ag frame
                                   ├─ 3. fit     longest clash-free contact run
                                   └─ 4. rank    weighted RMSD + shape
```

**Requires:** Linux x86-64 (MASTER's binaries are Linux-only), Python ≥ 3.10,
about 6 GB of disk.

---

## Setup

```bash
git clone https://github.com/fbabd/EpiFitScout
cd EpiFitScout
pip install -e .                  # the -e matters, see the note below
```

Then add the two things that are not in the repository. Each has its own folder
with step-by-step instructions:

| | Put it in | Instructions |
|---|---|---|
| **MASTER** binaries (`master-bin-v1.6.tar.gz`) | [`MASTER/`](MASTER/README.md) (becomes `MASTER/bin/`) | download, `tar -xzf … -C MASTER/`, `chmod +x` |
| **Dataset** (`epifitscout-database-v*.tar.xz`) | [`data/`](data/README.md) | download, `tar -xJf … -C data/` |

> ⚠️ **The dataset is not on Zenodo yet.** Until the deposit is live, build it
> with [`database_builder/`](database_builder/README.md) — it pulls from SAbDab
> and RCSB and produces the same layout, taking a few hours, mostly download.

One more is optional, and only for side-chain packing in
`epifitscout build`:

| | Put it in | Instructions |
|---|---|---|
| **FASPR** (build from source) | [`FASPR/`](FASPR/README.md) (becomes `FASPR/bin/`) | build with `g++ -O3 -o FASPR src/*.cpp`, copy it and `dun2010bbdep.bin` into `FASPR/bin/` |

Then check:

```bash
epifitscout check --search        # or: python check_setup.py --search
```

```
[ok]   master     /…/epifitscout/MASTER/bin/master
[ok]   createPDS  /…/epifitscout/MASTER/bin/createPDS
[ok]   database   /…/epifitscout/data/sabdab_chains.db
       25226 searchable chains (25226 coordinate files, 25226 search targets)
[ok]   search     20 hits in 2s; best 1A2Y:B RMSD 0.000, score 0.633
[--]   FASPR      not installed (optional; needed only for --side-chains packed)
Ready.
```

If something is in the wrong place, the FAIL line says which folder it expected
and what to run.

> **Why `pip install -e .`?** EpiFitScout finds `MASTER/bin/` and `data/` by
> their position next to the code. A regular `pip install .` copies the code
> away from them. The editable install keeps it in the clone.

---

## Quick start

One command takes a complex to grafted structures. Mark the CDR you want
replaced and the antigen chain, and the epitope is found by proximity:

```bash
epifitscout build \
    --structure 3gbn.pdb \
    --cdr H 93 102 \
    --antigen-chain B \
    --top 5 --side-chains native \
    --out-dir grafts
```

```
template 3gbn.pdb: CDR H/93-102 (14 residues), epitope B, within 10 A of the CDR: 18-20, 36, 38-39, 41-46, 49 (13 residues), reach 2.8 A
building 5 complexes

  1  6UYN:H       rmsd 0.16  score 0.249  join 1.19/1.40 A  clashes 0  contact 2.88 A  01_6UYN_H.pdb
  2  5C0S:H       rmsd 0.17  score 0.240  join 1.28/1.27 A  clashes 0  contact 2.91 A  02_5C0S_H.pdb
  3  3GBM:H       rmsd 0.18  score 0.244  join 1.22/1.36 A  clashes 0  contact 2.88 A  03_3GBM_H.pdb
  4  3GBM:I       rmsd 0.20  score 0.243  join 1.25/1.45 A  clashes 0  contact 2.86 A  04_3GBM_I.pdb
  5  8DIM:H       rmsd 0.22  score 0.244  join 1.07/1.19 A  clashes 0  contact 2.83 A  05_8DIM_H.pdb

5 complexes written to grafts
  ideal peptide bond is 1.33 A; 5/5 join cleanly
  side chains unavailable for 3: 5C0S:H, 3GBM:H, 3GBM:I
  summary: grafts/grafts.tsv
```

You get one PDB per hit — your template with its CDR replaced by that fragment,
antigen included — plus `grafts.tsv` ranking them. That is the whole tool; the
rest of this page is about controlling it.

Read the numbers as: `rmsd` to your query loop, `score` the epitope fit,
`join` how well the graft meets the framework at each end against a 1.33 Å
peptide bond, and `contact` its closest approach to the antigen.

- **Pick the epitope yourself** instead of by proximity: `--epitope B 38 46`.
  See [Defining the epitope](#defining-the-epitope).
- **Side chains**: `none`, `native` or `packed`. See
  [Building the grafted complexes](#building-the-grafted-complexes).
- **Just the ranked list, no structures**: use the Python API below.
- **Look at a fit**: `epifitscout view` exports a PyMOL session.

---

## Usage

For the ranked fragments without building structures, or to script the search:

```python
import epifitscout

qb = epifitscout.QueryBuilder("my_complex.pdb")
qb.describe()                                   # list chains and residue ranges

cdr     = qb.get_fragment("H", 100, 110)        # the CDR loop to replace
epitope = qb.get_epitope_surface("A", residues=(45, 60))   # the epitope you target

hits = epifitscout.search(cdr, epitope)

for i, h in enumerate(hits[:15], 1):
    print(f"{i:<3} {h.fragment.pdb_id}:{h.fragment.chain:<2} "
          f"RMSD {h.rmsd_cdr:.2f}  {h.local_fit.describe()}")
```

```
1   7UZ5:H  RMSD 0.21  score 0.381: loop residues 99-105 (7 of 14, 50%) against
                       epitope residues 41-46; gap 5.7 Å, normals 0.30
```

### Defining the epitope

The epitope is **yours to choose** — it does not have to be something the
template antibody already touches, since the point of a search is often to find
a loop that would reach somewhere it doesn't.

```python
qb.get_epitope_surface("A", residues=(45, 60))            # a contiguous range
qb.get_epitope_surface("A", residues=[45, 46, 47, 88, 89]) # discontinuous, as real epitopes are
qb.get_epitope_surface("A", near=cdr, cutoff=10.0)         # reproduce an existing interface
```

Check it is reachable before searching — a loop cannot stretch to an epitope on
the far side of the antigen:

```python
qb.epitope_reach(cdr, epitope)    # closest approach in Å; >15 Å is out of range
```

A worked example with figures is in
[`example/example_1a2y.ipynb`](example/example_1a2y.ipynb)
(`pip install -e ".[examples]"` for its extra dependencies).

### How hits are scored

Step 1 (MASTER) retrieves loops with a similar backbone. Step 2 then **grafts**
each one onto the CDR site, so it sits in the antibody-antigen frame where your
loop was, and judges it there:

- a candidate whose backbone comes closer than 2.0 Å to the antigen is
  **rejected**, a cutoff calibrated on 86 native complexes in which the closest
  any native CDR backbone comes to its antigen is 2.12 Å (an earlier 2.5 Å
  cutoff, set from only five complexes, rejected 12% of those natives);
- the remaining ones are scored by how long a stretch of the loop makes clean
  contact: `contact quality × (matched length / loop length)`;
- each hit reports **which** of its residues matched **which** epitope residues,
  so you can check any result yourself (`h.local_fit.describe()`, or
  `epifitscout view` to view it in PyMOL).

Step 3 ranks by `0.4 × 1/(1+RMSD) + 0.6 × shape`.

**Read the scores as a way to generate candidates, not as predicted affinity.**
The weights are not fitted to binding data. What has been measured, on 102
antibody–antigen complexes drawn from SAbDab:

| Question | Result |
|---|---|
| Does the geometric step add anything MASTER lacks? | 14.9% of 40,724 hits collide with the antigen; RMSD predicts which only weakly (median AUC 0.559, p = 0.0012) |
| Does the score prefer the real epitope? | true site beats the mean reachable decoy in **55/55** complexes (p = 5.5e-11, d = 2.71) |
| Is the native loop recovered? | median 71st percentile among non-self hits, vs 51st for the superseded scorer (p = 4.4e-07) |
| Does it track binding energy? | ρ = −0.326 (p = 0.0045) against SAbDab ΔG on 74 complexes |

So the geometry carries real information about the epitope, and more than RMSD
alone does — but a shape term is not an affinity predictor, and the correlation
with ΔG is weak. What is *not* yet measured is whether epitope-aware ranking
produces better antibodies than conformation-only ranking; that comparison is
the open question, not a settled result. See [`benchmark/`](benchmark/README.md)
and [`REPORT.md`](REPORT.md).

### Hit fields

| Field | Type | Description |
|---|---|---|
| `h.fragment.pdb_id` | `str` | PDB ID of the database hit |
| `h.fragment.chain` | `str` | Chain ID of the hit |
| `h.fragment.residue_range` | `(int, int)` | Start–end residue numbers |
| `h.fragment.sequence` | `str` | One-letter sequence |
| `h.fragment.metadata["chain_type"]` | `"H"` or `"L"` | Heavy or light chain |
| `h.rmsd_cdr` | `float` | Backbone RMSD to the query CDR (Å) |
| `h.complementarity_score` | `float` | Shape fit, in [0, 1]; 0 if the loop collides |
| `h.final_score` | `float` | Combined ranking score, in [0, 1] |
| `h.local_fit` | `LocalFit` | Where and how far it matched — see below |

`h.local_fit` carries the detail behind the score:

| Field | Meaning |
|---|---|
| `.fits` | True if a clean contact run was found and nothing collided |
| `.fragment_residues` | The matched stretch of the loop, in PDB numbering |
| `.epitope_residues` | The epitope residues it sits against |
| `.matched_length`, `.fraction_matched` | How far the match runs |
| `.mean_gap`, `.min_gap` | Separation over the run, in Å |
| `.n_clash` | Residues closer than 2.0 Å to the antigen |
| `.describe()` | One-line summary of all of the above |

### Building the grafted complexes

A search returns ranked *fragments*. To get structures — the template complex
with its CDR replaced by each hit, antigen included — use
[`epifitscout build`](epifitscout/tools/build_complex.py):

```bash
epifitscout build \
    --structure my_complex.pdb \
    --cdr H 93 102 --epitope B 38 46 \
    --top 5 --side-chains packed \
    --out-dir grafts
```

Give `--antigen-chain B` instead of `--epitope` and the epitope is taken to be
every residue of that chain within `--epitope-cutoff` (10 Å) of the CDR; the
residues chosen are printed so you can check them.

| `--side-chains` | What you get | Needs |
|---|---|---|
| `none` (default) | Backbone only at the graft | — |
| `native` | The fragment's own rotamers, moved rigidly | — |
| `packed` | Rotamers rebuilt for the new environment | FASPR, see [`FASPR/README.md`](FASPR/README.md) |

`packed` repacks the graft and the residues within `--repack-shell` Å of it
(5 Å by default) and pins the rest, so the experimental model you are grafting
into is left as deposited. `--repack-all` repacks everything.

Each output is checked and reported in `grafts.tsv`: how closely the loop joins
the framework at both ends (a peptide bond is 1.33 Å), and how close the
finished loop comes to the antigen. That last number matters — the ranking
score is computed on backbone alone, so it cannot see a side chain driven into
the antigen.

The clash check is element-aware, because one distance cut cannot tell a
hydrogen bond from a collision: two polar atoms may come within 2.2 Å, but once
a carbon is involved the contact is steric and needs 2.8 Å. At one real
interface, 2.40 Å from Tyr-OH to Gln-CG is a clash while 2.65 Å from Gln-OE1 to
Gln-NE2 is a good hydrogen bond. Offending atom pairs are named individually.

Those two cutoffs are calibrated on 102 native antibody–antigen interfaces, not
on van der Waals radii. Deposited structures pack far closer than textbook radii
allow — the median native closest approach is 2.74 Å polar and 3.17 Å with a
carbon, but the tails run to 2.12 Å and 1.26 Å. A radii-derived 3.0 Å carbon cut
flags 12.7% of real interfaces; these cuts flag about 2%, so a flag means a real
interface would not do this.

Expect flags on genuinely good models all the same. A rotamer library is
discrete, and native side chains often are not: in the bundled 3GBN example the
native Tyr-H98 sits at a non-rotameric chi1 of 84°, FASPR picks the canonical
60° one, and those 25° move the hydroxyl from 4.0 Å to 2.4 Å from the antigen.
Nothing is wrong with the graft — the contact needs a short local minimisation,
which no packer performs. That is what the flag is telling you.

Two further details of a packed model are worth knowing. FASPR discards every
heteroatom, so glycans, ligands and ions are restored afterwards from the
template — but the template's ordered waters were modelled around the
template's *own* loop, and some overlap the graft by as little as 0.5 Å. Those
waters are displaced; an overlapping glycan or ligand is kept and reported
instead, since removing one is a modelling decision to make rather than hide.

---

## Configuration

Defaults are in [`epifitscout/conf/config.yaml`](epifitscout/conf/config.yaml):

```yaml
master:
  rmsd_threshold: 2.0    # maximum backbone RMSD (Å) to keep a hit
  max_hits: 500
  timeout_seconds: 420
scoring:                 # S_shape = clip((w_depth*Sd + w_tau*S_tau + 1) / 2, 0, 1)
  weight_depth: 0.7
  weight_tau: 0.3
ranking:                 # S_final = w_rmsd * 1/(1+RMSD) + w_shape * S_shape
  weight_rmsd: 0.4
  weight_shape: 0.6
max_workers: 6           # parallel MASTER processes per query
```

Override any of them per call:

```python
hits = epifitscout.search(query_cdr, query_epitope,
                          rmsd_threshold=1.0, weight_depth=0.9, weight_tau=0.1)
```

| Change | Effect |
|---|---|
| `rmsd_threshold` ↓ | Fewer hits, closer backbone match |
| `rmsd_threshold` ↑ | More hits, more structural diversity |
| `weight_depth` ↑ | Emphasise protrusion/recession complementarity |
| `weight_tau` → 0 | Recommended for short or flat loops (H1, L1) |
| `weight_shape` ↑ | Prioritise surface fit over backbone similarity |
| `weight_rmsd` ↑ | Stay closer to the query backbone |

**Speed:** the same search took 2 s with the dataset on local disk and 162 s on a
slow network filesystem. Put `data/` on fast storage.

**Data somewhere else?** (a shared copy on a cluster, say) set
`EPIFITSCOUT_DB=/path/to/pds.list` and/or `EPIFITSCOUT_MASTER_DIR=/path/to/bin`.

---

## Keeping the dataset up to date

The shipped dataset is a snapshot. To bring it up to date, or to build your own
from a different set of structures, use [`database_builder/`](database_builder/README.md):

```bash
# 1. save a fresh SAbDab summary file as database_builder/sabdab_summary.tsv
python database_builder/build_database.py --dry-run   # what would change?
python database_builder/build_database.py             # apply the update
python check_setup.py
```

It downloads only structures that are new, and regenerates only the MASTER search
targets that are missing or out of date — so an update takes minutes, not hours.

---

## Repository layout

```
epifitscout/
├── check_setup.py          verify the setup; --search runs a real example
├── epifitscout/            the search package
│   └── conf/config.yaml
├── MASTER/bin/             ← paste the MASTER binaries here
├── data/                   ← extract the dataset here
├── FASPR/bin/              ← optional: paste FASPR here, for side-chain packing
├── tools/                  build grafted complexes, export PyMOL views
├── database_builder/       rebuild or update the dataset
├── example/                1a2y worked example
├── benchmark/              benchmark harness and results
├── tests/                  test suite (small bundled fixture database)
├── README.md   REPORT.md   (REPORT.md describes the method in detail)
```

## Development

```bash
pip install -e ".[dev]"
ruff check . && mypy
pytest                                      # ~10 s; MASTER-dependent tests skip without it
```

The suite needs neither the full dataset nor a network: `tests/data/mini_db`
holds 14 real chains (~400 kB). Tests that drive MASTER skip automatically until
`MASTER/bin/` is filled in.

## Citing

Please cite this software ([`CITATION.cff`](CITATION.cff)) together with:

* **MASTER** — Zhou J & Grigoryan G. *The design and application of structural
  motif libraries.* Protein Science 24(4):508–524, 2015.
* **SAbDab** — Dunbar J *et al.* *SAbDab: the structural antibody database.*
  Nucleic Acids Research 42:D1140–D1146, 2014.

## Licence

MIT — see [`LICENSE`](LICENSE). This covers EpiFitScout's source only. MASTER (GPL
for academic use) and SAbDab are distributed by their own authors, are not
included here, and are fetched by you from the original sources. RCSB PDB
coordinates are public domain.
