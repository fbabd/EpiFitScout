# EpiFitScout

Search ~25,000 real antibody chains for CDR loops that fit an epitope, and rank
them by how their backbone complements it.

Give it a bound antibody–antigen complex and say which CDR to replace. It finds
loops that can bridge that exact gap, places each one, and ranks them by shape
complementarity against the epitope — all on backbone coordinates, before
anything is built.

```
bound complex + CDR span + antigen chain
      │
 1. query      two framework stems either side of the CDR (4 residues each).
               The CDR's own coordinates are never in the query.
 2. search     MASTER over 25,226 SAbDab chains, --gapLen fixed to the
               loop's length → loops that can bridge that exact gap
 3. place      superpose each donor's stems onto the template's stems
 4. gate       are the two new peptide bonds real bonds?
 5. epitope    antigen residues within 4.5 Å of the template's own CDR
 6. score      shape complementarity, backbone only:
               0.7 × depth-profile correlation + 0.3 × torsion correlation
 7. collision  loop backbone atoms within 2.0 Å of antigen
 8. rank       sort by (no collision, then shape score)   ← THE TOOL ENDS HERE
      ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
 9. build      graft the chosen loop in, pack side chains (FASPR)   ] example,
10. relax      turn clashing side chains out of the antigen         ] and
11. measure    BSA, contacts, H-bonds, salt bridges, clashes, void  ] reusable
```

**Stages 1–8 are EpiFitScout.** They need MASTER and the chain database, and
they return ranked fragments:

```python
import epifitscout
result = epifitscout.search("7abc.pdb", "H", (95, 102), ("A",))
for c in result.top:
    print(c.source, c.sequence, c.shape.describe())
```

**Stages 9–11 build and measure models** from that ranking. They are a worked
example rather than a design protocol, they additionally need FASPR, and they
live in `epifitscout.build`:

```python
result = epifitscout.graft("7abc.pdb", "H", (95, 102), ("A",), out_dir="models")
```

The ranking is identical either way — building a model never reorders anything.
That is deliberate: it is what leaves the all-atom numbers usable as evidence
*about* the ranking instead of a restatement of it. See [`example/`](example/)
for what those numbers say, including where they are unflattering.

**Requires:** Linux x86-64 (MASTER's binaries are Linux-only), Python ≥ 3.10,
about 26 GB of disk for the database.

## Two things to know before trusting a result

**Shape alone is not enough, and the pair is the point.** The depth correlation
has no steric content — it is *maximised* by pressing a loop into the surface.
Ranked on shape alone, clashes doubled and 47 of 85 models carried more than
twenty. Stage 7 is the counterweight, and it orders rather than excludes, so a
target whose candidates all collide still gets an answer.

**A contact gain is not a binding gain.** Contacts and clashes are the same
geometry at two cutoffs (r = 0.98 across 85 designs). Judge a model by contact
made at *zero* clashes, not by raw contact count.

## Setup

### What is in the repository, and what is not

Three things are too large or not ours to redistribute, so you add them once.
Each has a folder with its own instructions:

| What | Where it goes | Needed for | Instructions |
|---|---|---|---|
| **Chain database** `epifitscout-database-v1.0.0.tar.xz` (645 MB → 25 GB) | `data/` | everything | [`data/README.md`](data/README.md) |
| **MASTER** binaries `master-bin-v1.6.tar.gz` (13 MB) | `MASTER/` → `MASTER/bin/` | everything | [`MASTER/README.md`](MASTER/README.md) |
| **FASPR** (build from source, ~14 MB) | `FASPR/` → `FASPR/bin/` | stages 9–11 only | [`FASPR/README.md`](FASPR/README.md) |

**FASPR is optional if you only want to search.** Stages 1–8 rank fragments on
backbone coordinates and never pack anything, so `check` reports a missing FASPR
as `[--]` rather than `[FAIL]` and still says `Ready.`

### Install

```bash
git clone https://github.com/fbabd/EpiFitScout
cd EpiFitScout
pip install -e .                  # the -e matters, see below
pip install -e ".[examples]"      # only if you will run example/
```

> **Why `pip install -e .`?** EpiFitScout finds `MASTER/bin/` and `data/` by
> their position next to the code. A regular `pip install .` copies the code
> away from them; the editable install keeps it in the clone.

### Add the three pieces

```bash
# 1. the database — verify the checksum first, a truncated archive
#    unpacks silently and fails later, mid-search
sha256sum -c SHA256SUMS
tar -xJf epifitscout-database-v1.0.0.tar.xz -C data/

# 2. MASTER
tar -xzf master-bin-v1.6.tar.gz -C MASTER/
chmod +x MASTER/bin/*

# 3. FASPR — build outside this repo, the clone shares the folder name
cd /tmp && git clone https://github.com/tommyhuangthu/FASPR.git
cd FASPR && g++ -O3 -o FASPR src/*.cpp
mkdir -p <repo>/FASPR/bin
cp FASPR dun2010bbdep.bin <repo>/FASPR/bin/
```

You should end up with:

```
data/sabdab_chains.db/{pdb,pds}/      25226 files each
MASTER/bin/{master,createPDS}
FASPR/bin/{FASPR,dun2010bbdep.bin}    both, in the same directory
```

### Check

```bash
epifitscout check --search        # or: python check_setup.py --search
```

```
[ok]   master     /…/epifitscout/MASTER/bin/master
[ok]   createPDS  /…/epifitscout/MASTER/bin/createPDS
[ok]   database   /…/epifitscout/data/sabdab_chains.db
       25226 searchable chains (25226 coordinate files, 25226 search targets)
[ok]   FASPR      /…/epifitscout/FASPR/bin/FASPR
[ok]   search      retrieved=499 feasible=317 pooled=5 returned=3 in 185s
Ready.
```

`--search` runs a real query end to end, which is the only thing that proves the
search targets are usable; it takes about three minutes. Without it the check
only confirms the files are in place. If something is misplaced, the `[FAIL]`
line names the folder it expected and what to run.

---

## Running it

### Search: rank fragments (stages 1–8)

```bash
epifitscout search \
    --structure 7abc.pdb \
    --cdr H 95 102 \
    --antigen A
```

```python
import epifitscout
result = epifitscout.search("7abc.pdb", "H", (95, 102), ("A",))
for c in result.top:
    print(c.source, c.sequence, c.shape.describe(), c.n_collide)
```

No packer needed, nothing written to disk.

### Graft: build and measure models (adds stages 9–11)

```bash
epifitscout graft \
    --structure 7abc.pdb \
    --cdr H 95 102 \
    --antigen A \
    --out-dir models
```

```python
result = epifitscout.graft("7abc.pdb", "H", (95, 102), ("A",), out_dir="models")
for c in result.top:
    print(c.source, c.sequence, c.interface.describe())
```

Needs FASPR. The order of `result.top` is the same as `search` would give —
building never reorders anything.

### What to pass

`--cdr` takes the chain and the first and last residue numbers of the loop to
replace, **as numbered in your file**. State it explicitly when comparing
against another method: IMGT and Chothia CDR-H3 differ by two residues, and left
alone that has the two methods designing different windows.

`--antigen` takes the chain(s) treated as antigen. The epitope is then found
automatically, as the antigen residues within 4.5 Å of the template's own CDR.


## Quick start

```bash
epifitscout graft \
    --structure 7abc.pdb \
    --cdr H 95 102 \
    --antigen A \
    --out-dir models
```

```
7ABC:H loop 95-102 (8 res), stems 91-94 and 103-106, span 5.4 Å (0.67 Å/res)
  retrieved 500  feasible 214  pooled 50  scored 48  returned 10

rank  donor            sequence               obj  interface
   1  8G19_H           ARDSGYYLDY           +2.31  BSA 812 Å² (loop 241), 63 contacts over 6 residues, 5 H-bonds, 1 salt bridges, gap index 1.84 Å, 0 clashes
        models/models/003_8G19_H.pdb
   2  6XKQ_B           ARDRGYSMDV           +1.88  BSA 798 Å² (loop 229), 55 contacts over 6 residues, 4 H-bonds, 1 salt bridges, gap index 2.05 Å, 0 clashes
        models/models/011_6XKQ_B.pdb
   …

7 of 10 reported models have no clash with the antigen.
```

Read the funnel left to right — it is how a thin result explains itself:

| | means |
|---|---|
| `retrieved` | loops in the database that bridge this gap in this geometry |
| `feasible` | …of those, the ones whose two new peptide bonds are real bonds |
| `pooled` | …of those, the ones packed and measured (the search budget, `--pool`) |
| `scored` | …of those, the ones that packed and measured successfully |
| `returned` | reported, best first (`--top`) |

The same thing from Python:

```python
import epifitscout

result = epifitscout.graft("7abc.pdb", "H", (95, 102), ("A",), out_dir="models")

for candidate in result.top:
    print(candidate.source, candidate.sequence, candidate.interface.describe())
```

**On `--cdr`:** state the span explicitly when comparing against another method.
IMGT and Chothia CDR-H3 differ by exactly two residues, and left alone that has
the two methods designing different windows while every metric is reported as
though they matched.

---

## Usage

### Controlling the search

| Flag | Default | What it does |
|---|---|---|
| `--pool` | 50 | Feasible candidates packed and measured. **This is the search budget** — hold it equal across methods being compared |
| `--top` | 10 | Candidates reported |
| `--max-hits` | 500 | Matches MASTER returns before the gate |
| `--rmsd-cut` | 2.0 | Stem RMSD cutoff (Å) |
| `--stem` | 4 | Framework residues per stem in the query |
| `--exclude` | — | Donor PDB IDs the search must not return |
| `--database` | — | A `pds.list` to search instead of the default |

`--pool` deserves the emphasis. A bigger pool is a bigger search budget, not a
better method: in the v1 benchmark a 4.7 standard-error advantage over a
diffusion model fell to a 0.9 s.e. null once both sides drew an equal-sized pool
and were ranked by the same objective.

### Leakage, if you are measuring anything

The chain database is built from SAbDab, so a target drawn from SAbDab can
retrieve its own native loop and rank it first. **Excluding the target's PDB ID
is not enough**, and the gap is wide. Run the bundled example with only `1A2Y`
refused and the top five hits for its CDR-H3 are that exact loop — from 1A7O,
1A7P, 1VFA, 1A7R and 1KIR, five other entries of the same antibody. Exclude the
sequence as well and the top hit becomes 1KIQ's `ARERDFRLDY`: the native
`ARERDYRLDY` with one substitution, from an affinity-maturation variant, which
shares no 8-mer with it because the mutation sits in the middle.

So there are three levels, and a benchmark wants all three:

```bash
epifitscout graft … --exclude 7ABC              # the target's own entry
epifitscout graft … --exclude-native-loop       # and variants of its own loop
epifitscout graft … --database filtered.list    # and its near-duplicate chains
```

`--exclude-native-loop` refuses any retrieved loop that shares an 8-residue
stretch with the template's CDR **or** is at least 80% identical to it over the
same length. The two criteria catch different things: the stretch catches a loop
embedded in a longer one, the identity catches a point mutant. For a
ten-residue loop, 80% allows at most two differences, and chance identity that
high is about 10⁻⁹.

Filtering the PDS list is the cheapest, since those chains are never searched.
In the v1 audit, 12 of 30 targets had their exact native CDR-H3 present in a
*different* entry; one had 81 such chains.
`graft()` exposes two controls for this: `exclude_donors`, a set of PDB IDs the
search must not return, and `exclude_native_loop=True`, which additionally
refuses any loop sharing an 8-residue stretch with the template's own CDR.
Excluding the target's own entry does not cover the same antibody deposited
under another accession, so set both.

### Scripting the stages

Each stage is usable on its own:

```python
from pathlib import Path
from epifitscout.query.builder import QueryBuilder
from epifitscout.graft.assemble import TemplateSite
from epifitscout.graft import feasibility
from epifitscout.scoring.interface import InterfaceScorer
from epifitscout.pipeline.graft_pipeline import GraftPipeline

qb = QueryBuilder(Path("7abc.pdb"))
query = qb.get_anchor_query("H", 95, 102)      # the stems, not the loop
print(query.describe())

site = TemplateSite.load(Path("7abc.pdb"), "H", query.loop_range, ("A",))

pipeline = GraftPipeline()
candidates = pipeline.search(query, Path("work"))             # stages 1-2
feasible = feasibility.assess_all(candidates, query, site.framework_coords)
scored = pipeline.measure(feasible[:50], query, site, Path("work"))   # stages 4-5
```

### What each candidate carries

| Field | Meaning |
|---|---|
| `candidate.source` | Donor entry, chain and residue range |
| `candidate.sequence` | The retrieved loop's sequence |
| `candidate.placed_loop` | `(N, 4, 3)` backbone in the template's frame |
| `candidate.stem_rmsd_bb` | Fit residual of the placement (Å) |
| `candidate.feasibility` | Junction geometry and the verdict — `.describe()` |
| `candidate.interface` | Every interface term — `.describe()`, `.as_dict()` |
| `candidate.z_scores` | Objective terms, standardised within this pool |
| `candidate.objective` | What the ranking sorted on |
| `candidate.model_path` | The packed model, if it was kept |

### Defining the epitope

The ranking measures the interface with whichever antigen chains you name, so
the "epitope" in v2 is the part of those chains the grafted loop reaches. To
score against a chosen patch instead, build an `EpitopeSurface`:

```python
qb.get_epitope_surface("A", residues=(45, 60))              # a contiguous range
qb.get_epitope_surface("A", residues=[45, 46, 47, 88, 89])  # discontinuous
qb.get_epitope_surface("A", near=cdr, cutoff=10.0)          # an existing interface
qb.epitope_reach(cdr, epitope)        # closest approach in Å; >15 Å is out of range
```

### Interface terms

Measured on the packed model, loop against antigen, with water excluded from
both sides — ordered solvent was modelled around the template's *own* loop, so a
water where the graft now sits is displaced rather than clashed with.

| Term | Definition |
|---|---|
| `buried_area` | Antibody–antigen ΔSASA, Shrake–Rupley, 1.4 Å probe. Summed over both partners, so roughly twice the one-sided "interface area" |
| `loop_buried_area` | The same difference restricted to the graft's atoms |
| `contacts`, `contact_residues` | Heavy-atom pairs within 4.5 Å, and loop residues taking part |
| `clashes` | Element-aware: polar pairs may reach 2.2 Å, anything with a carbon 2.8 Å. Calibrated on 102 native interfaces, where they flag ~2% |
| `hbonds` | Donor–acceptor heavy atoms within 3.5 Å with the angle at the acceptor above 90° |
| `salt_bridges` | Oppositely charged groups within 4.0 Å, counted once per residue pair |
| `gap_volume`, `gap_index` | Void trapped between loop and epitope, and that volume per Å² buried — Laskowski's index, by grid approximation |

`gap_index` is the term that distinguishes *meshing* from *touching*. A retrieved
loop was folded against a different antigen, so burying area while leaving
cavities is its characteristic failure, and no area or contact count detects it.

Two guards on the ranking, both added after watching it misbehave on a real run:

**Clash-free models sort first** (`clean_first`). A clash by the calibrated
threshold means a real interface would not do this, so the structure is not a
candidate and no amount of polar contact makes it one. It is a preference rather
than a gate — when nothing is clean the clashing models are still reported in
order, because reporting nothing is not more useful.

**Each standardised term is clipped to ±2 s.d.** (`clip_z`). These terms include
counts that are zero for almost every candidate, and standardising one of those
hands the lone exception a z-score of about √n. In a pool of ten where one model
had a salt bridge and nine had none, that term alone contributed +3.0 and decided
the ranking — which is how a run came back with none of its five reported models
clash-free.

**Read these as a way to generate and triage candidates, not as predicted
affinity.** The ranking weights are set by convention, not fitted;
`epifitscout.scoring.normalize.fit_weights` is the hook for fitting them against
a held-out outcome, and a flat result there would say the terms carry no
orderable signal — which is a finding, not a failure.

---

## Configuration

Defaults are in [`epifitscout/conf/config.yaml`](epifitscout/conf/config.yaml).
An unknown key in any section is an error rather than a warning: a misspelt
parameter that fell back to its default would make the run look like a result
for the setting you asked for.

```yaml
master:                    # stage 1
  rmsd_threshold: 2.0      # stem RMSD cutoff (Å)
  max_hits: 500
  gap_tolerance: 0         # 0 keeps the retrieved loop and the span one-to-one
feasibility:               # stage 3 — ideal peptide geometry is 1.33 Å, 116°, 122°
  bond_min: 1.0
  bond_max: 1.8
  angle_tolerance: 30.0
  self_clash_cutoff: 2.2
interface:                 # stage 5
  probe_radius: 1.4
  sasa_points: 200
  contact_cutoff: 4.5
  hbond_distance: 3.5
  grid_spacing: 0.75
ranking:                   # stage 6 — over z-scored terms, within each target
  weight_hbond: 1.0
  weight_salt: 1.0
  weight_clash: 1.0
  weight_gap: 1.0
  clean_first: true        # clashing models sort below clash-free ones
  clip_z: 2.0              # bound each term's influence
pool_size: 50
top_k: 10
max_workers: 6
```

Adding a term to `ranking` removes it from the independent evaluation set. That
is the trade: a term used to rank cannot also be a finding.

**Speed:** the same search took 2 s with the dataset on local disk and 162 s on a
slow network filesystem. Put `data/` on fast storage. Stage 5 is CPU-bound
(three SASA calculations per candidate), so parallelise across targets rather
than raising `max_workers` within one.

**Data somewhere else?** (a shared copy on a cluster, say) set
`EPIFITSCOUT_DB=/path/to/pds.list` and/or `EPIFITSCOUT_MASTER_DIR=/path/to/bin`.

---

## What the measurements say

Measured on 17 complexes (CDR-H3 lengths 5–21), five designs each, against the
input complex each was grafted into and scored by the identical scorer. Every
target's own PDB entry was excluded from the search and its native loop sequence
refused, so nothing here is a restatement of the answer.

### What the tool is good for

**Diverse, novel backbones.** All 85 designs come from 85 distinct donor chains,
at a mean 21% sequence identity to the loop they replace (median 19%). These are
not variations on the native loop — they are different real loops that fit the
same site.

**Sterically valid ones, on most targets.** 30 of 85 designs are clash-free, and
13 of 17 targets yield at least one, from 30 distinct donors. Those are models
worth taking further.

**Cheap to enumerate.** Stages 1–8 rank on backbone geometry with no packing, so
a target costs one MASTER search. Only the models you ask for get built.

That combination — real crystallographic conformations, low identity to the
native, sterically checked, many per target — is what makes these useful as
**starting points and as seeds for a generative or optimisation stage**, rather
than as finished designs.

### What it does not do

Mean ± SD. "Better" means better **for the antibody**: a binder wants more
buried area, more contacts, more hydrogen bonds and salt bridges, a tight pack
against the epitope (low gap index), and no interpenetration.

| | input complex | all 85 designs | the 30 clash-free | better |
|---|---|---|---|---|
| buried area (Å²) | 1784 ± 266 | 1752 ± 344 | 1634 ± 272 | higher |
| loop buried area (Å²) | 287 ± 136 | 270 ± 166 | **141 ± 77** | higher |
| contacts | 96 ± 48 | 155 ± 225 | **34 ± 28** | higher |
| contact residues | 6.1 ± 2.2 | 5.7 ± 3.2 | 3.3 ± 2.1 | higher |
| **clashes** | **0.0 ± 0.0** | **15.5 ± 38.5** | 0.0 ± 0.0 | lower |
| H-bonds | 5.5 ± 4.6 | 4.6 ± 6.6 | **1.2 ± 1.4** | higher |
| salt bridges | 0.8 ± 1.7 | 0.1 ± 0.3 | 0.0 ± 0.2 | higher |
| gap index (Å) | 4.1 ± 2.1 | 6.9 ± 17.8 | 12.7 ± 29.1 | lower |
| closest approach (Å) | 2.69 ± 0.29 | 2.29 ± 0.81 | 3.04 ± 0.63 | ≈ 2.7 |

(The clash-free column covers the 13 targets that produced one, so its input
reference is 1765 ± 271 Å² rather than 1784.)

**It does not produce a better interface than the one it replaces.** The
crystallographic input is clash-free on all 17 targets and no design matches it.

**The contact column in the middle is a trap.** Contacts and clashes are the
same geometry at two cutoffs (r = 0.98), so the 155 is largely atoms inside the
antigen — which the clash row and the 2.29 Å closest approach both confirm.

**The right-hand column is the honest one, and it is worse.** Keep only the
sterically valid designs and every term moves against them: the loop buries half
the native's area, makes a third of its contacts, and makes essentially no
hydrogen bonds. Closest approach rises *above* the native's 2.71 Å and the gap
index triples. The clash-free designs are valid because they **stand off the
epitope** — they touch less, rather than fitting better.

Asked target by target, a clash-free design beats its own input on contacts in
3 of 13 cases, all three where the native interface was unusually small (9, 26
and 56 contacts).

So a refinement stage that can drive a loop into contact is required rather
than optional, and no claim is made here about affinity — these are geometric
measurements on static structures.

Full numbers, controls and figures: [`example/`](example/).

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
├── check_setup.py              verify the setup
├── epifitscout/
│   ├── types/                  Fragment, AnchorQuery, GraftCandidate, EpitopeSurface
│   ├── query/builder.py        build the stem query from a complex
│   ├── master/                 run MASTER, parse matches, place candidates
│   ├── graft/feasibility.py    stage 4: can it be bonded in?
│   ├── scoring/                stages 5-7: epitope, shape, collision
│   ├── build/                  stages 9-11: assemble, pack, relax, measure
│   ├── pipeline/               stages 1-8, and the optional build
│   ├── tools/                  CLI entry points and the setup check
│   └── conf/config.yaml
├── MASTER/bin/                 ← paste the MASTER binaries here
├── data/                       ← extract the dataset here (data/README.md)
├── FASPR/bin/                  ← paste FASPR here
├── database_builder/           rebuild or update the dataset
├── example/                    designs vs their input complexes
├── tests/                      test suite (small bundled fixture database)
├── example/README.md           what the designs measure against the inputs
├── README.md                   the method, its results and its limits
```

## Development

```bash
pip install -e ".[dev]"
ruff check . && mypy
pytest                                      # ~60 s; MASTER-dependent tests skip without it
```

The suite needs neither the full dataset nor a network: `tests/data/mini_db`
holds 14 real chains (~400 kB). Tests that drive MASTER skip automatically until
`MASTER/bin/` is filled in.

Worth knowing what the tests pin down, because two of them are the method's
calibration: `test_feasibility.py` requires that a native loop re-grafted into
its own framework passes the gate with near-ideal geometry, and
`test_interface.py` requires that a loop driven into the antigen shows *both*
more contacts and more clashes — the confound that made a nonsense control look
best in v1.

## Citing

Please cite this software ([`CITATION.cff`](CITATION.cff)) together with:

* **MASTER** — Zhou J & Grigoryan G. *The design and application of structural
  motif libraries.* Protein Science 24(4):508–524, 2015.
* **SAbDab** — Dunbar J *et al.* *SAbDab: the structural antibody database.*
  Nucleic Acids Research 42:D1140–D1146, 2014.
* **FASPR** — Huang X, Pearce R & Zhang Y. *FASPR: an open-source tool for
  fast and accurate protein side-chain packing.* Bioinformatics 36(12):3758–3765,
  2020.

## Licence

MIT — see [`LICENSE`](LICENSE). This covers EpiFitScout's source only. MASTER (GPL
for academic use), SAbDab and FASPR are distributed by their own authors, are not
included here, and are fetched by you from the original sources. RCSB PDB
coordinates are public domain.
