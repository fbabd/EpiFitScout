# EpiFitScout v3

Retrieval and grafting of real CDR loops, ranked by how their backbone
complements the epitope — before anything is packed.

Give it a bound antibody–antigen complex and say which CDR to replace. It
searches ~25,000 antibody chains from SAbDab for loops that can bridge that
site, scores each placed loop against the epitope, and packs and measures only
the ones it is going to hand back.

```
complex ──▶ CDR site ──▶ graft() ──▶ ranked all-atom models
                            │
                            ├─ 1. search    two framework stems as a MASTER query,
                            │               --gapLen fixed to the loop's length
                            ├─ 2. place     fit each donor's stems onto the template's
                            ├─ 3. gate      are the two new peptide bonds real bonds?
                            ├─ 4. epitope   antigen residues within 4.5 Å of the
                            │               template's own CDR
                            ├─ 5. rank      shape complementarity of the placed
                            │               backbone against that epitope  ← decides
                            └─ 6. build     pack the chosen few (FASPR) and measure
                                            BSA, contacts, H-bonds, clashes, void
```

**The ranking happens before the packing.** Stage 5 sees backbone coordinates
and the epitope, nothing else. Stage 6 then builds only the `top_k` models that
will be reported, so the expensive all-atom work scales with how many models you
ask for rather than with how many loops the search returned.

**The interface numbers never rank anything.** They are measurements of the
models that stage 5 chose, which is what leaves them usable as evidence *about*
the ranking rather than a restatement of it.

**The CDR's own coordinates are never part of the query.** Only the framework
either side of it. The epitope is located by proximity to the template's CDR —
that reads the native coordinates, but only to decide which antigen residues to
score against, and nothing from it reaches MASTER.

## What changed from v2

v2 ranked on `contact_fit`, a backbone contact count against the whole antigen.
Measured against the input complexes it was grafting into, it produced models
that clashed in **every one of 17 targets** (mean 25 clashes against the input's
0) while appearing to gain contacts — an artefact, since contacts and clashes
correlate at r = 0.98 and are the same geometry at two cutoffs.

v3 replaces that selection entirely. All of v2's ranking is gone: no
`contact_fit`, no `clean_first`, no promotion of clash-free models. What ranks
is the epitope shape score, restored from v1 and **corrected**.

### The v1 sign bug

v1's depth term was negated, on the reasoning that a CDR bulge should meet an
epitope recess and so the profiles should be anti-correlated. The geometry does
not work that way: both profiles are projected onto the *same* CDR→epitope
normal, so a bulge and the recess it nests into both lie further along it, and
complementary surfaces are **correlated**. On three synthetic pairs whose answer
is not in doubt:

| | pearson(depth) | gap std |
|---|---|---|
| loop nests perfectly | +1.00 | 0.00 Å |
| flat lid over a bowl | 0.00 | 0.70 Å |
| opposite curvature | −1.00 | 1.39 Å |

Negated, that ranking is exactly inverted — v1 was scoring interpenetrating
loops highest. This is a sufficient explanation for v1's own finding that its
epitope term "does nothing measurable", and it is why that finding does not
carry over. `tests/test_shape.py` fails if the sign flips back.

**Requires:** Linux x86-64 (MASTER's binaries are Linux-only), Python ≥ 3.10,
about 6 GB of disk.

---

## Setup

```bash
git clone https://github.com/fbabd/EpiFitScout
cd EpiFitScout
pip install -e .                  # the -e matters, see the note below
```

Then add the things that are not in the repository. Each has its own folder
with step-by-step instructions:

| | Put it in | Instructions |
|---|---|---|
| **MASTER** binaries (`master-bin-v1.6.tar.gz`) | [`MASTER/`](MASTER/README.md) (becomes `MASTER/bin/`) | download, `tar -xzf … -C MASTER/`, `chmod +x` |
| **Dataset** (`epifitscout-database-v*.tar.xz`) | [`data/`](data/README.md) | download, `tar -xJf … -C data/` |
| **FASPR** (build from source) | [`FASPR/`](FASPR/README.md) (becomes `FASPR/bin/`) | build with `g++ -O3 -o FASPR src/*.cpp`, copy it and `dun2010bbdep.bin` into `FASPR/bin/` |

FASPR was optional in v1. **In v2 it is required for results**: stage 5 measures
an all-atom interface, and without a packer the models are backbone-only at the
graft, which reads as a loose interface on every term. `--packer none` still
runs, for diagnostics.

> ⚠️ **The dataset is not on Zenodo yet.** Until the deposit is live, build it
> with [`database_builder/`](database_builder/README.md) — it pulls from SAbDab
> and RCSB and produces the same layout, taking a few hours, mostly download.

Then check:

```bash
epifitscout check --search        # or: python check_setup.py --search
```

```
[ok]   master     /…/epifitscout/MASTER/bin/master
[ok]   createPDS  /…/epifitscout/MASTER/bin/createPDS
[ok]   database   /…/epifitscout/data/sabdab_chains.db
       25226 searchable chains (25226 coordinate files, 25226 search targets)
[ok]   FASPR      /…/epifitscout/FASPR/bin/FASPR
Ready.
```

If something is in the wrong place, the FAIL line says which folder it expected
and what to run.

> **Why `pip install -e .`?** EpiFitScout finds `MASTER/bin/` and `data/` by
> their position next to the code. A regular `pip install .` copies the code
> away from them. The editable install keeps it in the clone.

---

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

## What changed in v2, and why (background)

v1 searched with the CDR loop itself. Four consequences followed, and v2 is the
set of changes that remove them.

**The query was the answer.** Every hit was within the RMSD cutoff of the native
loop by construction, so CA RMSD, DockQ, Fnat and iRMSD measured the query:
`rmsd_to_native == rmsd_to_query` to four decimal places on every row of the v1
benchmark. v2 queries the two framework stems as disjoint MASTER segments with
`--gapLen` fixed to the loop's length, which asks for every real loop that can
bridge the gap and never shows the search an answer.

**Long loops returned nothing.** A single-segment query needs the whole loop to
superpose within the cutoff, and the number of real loops meeting that falls
away as the loop lengthens: 10 of 101 benchmark complexes returned nothing, all
CDR-H3 ≥ 15. A stem query needs only the stems to match, so retrieval depth is
set by framework geometry instead — a 22-residue gap that returned nothing
single-segment returns ~16,000 candidates.

**Nothing checked that a graft could be built.** v1 measured junction geometry
after ranking, on the few models it had already chosen, and printed a warning.
v2 measures both junctions before anything expensive and rejects: bond length,
both bond angles, the omega dihedral, and collisions with the antibody's own
framework. On a real search 60% pass, with junction bonds at 1.25 and 1.29 Å
against the ideal 1.329 — and a native loop put back in its own site passes with
near-ideal geometry, which is the calibration that keeps the gate from being too
strict.

**Scoring came before packing.** Complementarity is a side-chain property, and
v1 ranked on backbone-to-antigen distance with no side chains at all. Its
epitope term reordered candidates without improving any measured property and
cost sequence recovery (p = 0.016 at n = 101). v2 packs every pooled candidate
all-atom first, then measures buried area, contacts, hydrogen bonds, salt
bridges, element-aware clashes and the void volume trapped between loop and
epitope.

Two smaller changes matter as much:

**Terms are z-scored within each target's own pool.** Every interface term's
scale is set by the epitope, so raw scores are not comparable across targets. v1
compared them anyway, and a random length-matched control came out burying the
most surface of any method — it was simply driven into the antigen, which
maximises burial.

**The objective is disjoint from the evidence.** Ranking uses hydrogen bonds,
salt bridges, clashes and gap index. Buried area, contacts, sequence recovery,
CA RMSD, DockQ, Fnat and Lawrence–Colman Sc are deliberately excluded, so they
can be reported as evidence about the ranking. v1 ranked on contact with the
nominated epitope and reported epitope-F1 — contact with the intended epitope —
as a result.

### What the benchmark says

17 complexes spanning CDR-H3 lengths 5–21, five designs each, the input complex
scored by the same scorer at the same span. The comparison is against **the
bound complex being grafted into** — the one reference that needs no second
model and no matched budget, because the loop was there and it worked.

Per-target mean of five designs:

| | input complex | v2 | v3 (shape only) | v3 (shape + gate) |
|---|---|---|---|---|
| clashes | **0.0** | 25.0 | 52.2 | 21.3 |
| closest approach (Å) | **2.69** | 1.85 | 1.43 | 1.99 |
| buried area (Å²) | 1784 | 1727 | 1810 | 1741 |
| contacts | 96 | 188 | 315 | 164 |
| clash-free designs | — | 9/85 | 10/85 | **18/85** |
| targets with one | — | 6/17 | 6/17 | **8/17** |

**Shape ranking alone made it worse.** Clashes doubled against v2 (2.2 s.e.) and
closest approach fell to 1.43 Å. The depth correlation has no steric content and
is maximised by pressing the loop into the surface, so nothing opposed
interpenetration: 47 of 85 models carried more than twenty clashes. Adding the
collision counterweight recovered it — clashes 52.2 → 21.3 (3.3 s.e.), closest
approach 1.43 → 1.99 Å (4.0 s.e.).

**Against v2 the averages are a wash.** Every mean difference is under 1 s.e.
The real gain is yield: clash-free designs double, 9/85 to 18/85, and targets
with at least one usable model go 6/17 to 8/17 (7 targets improved, 1 worse, 9
tied).

**A contact gain is not a binding gain.** Contacts and clashes are the same
geometry at two cutoffs and correlate at r = 0.98 across 85 designs, so a loop
driven into the antigen scores more of both. Regressing contacts on clashes, the
intercept — contact made at *zero* clashes — is the number that means anything:

    v2           contacts = 71.6 + 4.67 × clashes
    v3 gated     contacts = 54.5 + 5.12 × clashes
    input complexes average 96 contacts at 0 clashes

**What is not solved.** Clean models are clean because they barely touch: v3's
clash-free designs average 26 contacts against the inputs' 96. Ranking cannot
fix that — it chooses among retrieved loops, and no retrieved loop in these
pools both avoids the antigen and buries like the native. That needs a
relaxation stage, not a better selector. 61% of clashes are side-chain-only at
median 0.43 Å overlap, which a backbone gate cannot see and discrete-rotamer
packing leaves behind.

**No design in any version beats the input complex on mean interface quality.**
Do not read the table as saying otherwise.

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

## The superseded v1 path

v1's search is still here, so its ablations can be rerun rather than taken on
trust: `epifitscout.search()`, `epifitscout build`, and the `scoring` and
`legacy_ranking` config sections. It queries MASTER with the native loop, so
**every structural metric it reports is a measurement of its own query.** Use it
for ablations, not for results.

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
epifitscout_v3/
├── check_setup.py              verify the setup
├── epifitscout/
│   ├── types/                  Fragment, AnchorQuery, GraftCandidate, EpitopeSurface
│   ├── query/builder.py        build the stem query from a complex
│   ├── master/                 run MASTER, parse matches, place candidates
│   ├── graft/                  feasibility gate, assembly, packing, relax
│   ├── scoring/                epitope, shape (ranks), collision, interface (reports)
│   ├── pipeline/               graft_pipeline
│   ├── tools/                  graft_cdr (v2 CLI), build_complex (v1), view
│   └── conf/config.yaml
├── MASTER/bin/                 ← paste the MASTER binaries here
├── data/                       ← extract the dataset here, see DATA.md
├── FASPR/bin/                  ← paste FASPR here
├── database_builder/           rebuild or update the dataset
├── example/                    1a2y worked example
├── tests/                      test suite (small bundled fixture database)
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
