# EpiFitScout: Epitope-Complementary CDR Fragment Retrieval

**A methods report on the retrieval pipeline, its scoring algorithms, and its use**

Status: as of 2026-10-05 · Source tree: `EpiFitScout/` · Version `0.1.0`

---

## Abstract

EpiFitScout retrieves antibody CDR fragments that are at once similar in backbone
conformation to a query loop and geometrically complementary to a nominated epitope
surface. The user supplies a bound complex and names two fragments within it — a CDR
loop and the epitope residues it contacts. The tool searches a chain database built
from SAbDab (25,226 variable-domain chains in the installed build) and returns a
ranked hypothesis list.

Retrieval proceeds in three stages. **MASTER** performs an exact backbone-RMSD search
with a sliding window over full chains, yielding fragments whose conformation matches
the query. A **shape complementarity scorer** then evaluates each hit against the query
epitope, combining the anti-correlation of interface depth profiles with the
co-correlation of Frenet–Serret backbone torsion. A **ranker** finally merges
conformational similarity and complementarity into a single score:

$$
S_{\text{shape}} = \mathrm{clip}\!\left(\frac{w_d S_d + w_\tau S_\tau + 1}{2},\; 0,\; 1\right)
$$

$$
S_{\text{final}} = w_{\text{rmsd}}\,\frac{1}{1 + \mathrm{RMSD}} \;+\; w_{\text{shape}}\, S_{\text{shape}}
$$

With default weights the ranking is 60% complementarity and 40% backbone match, and the
complementarity term is itself 70% depth and 30% torsion. The design intent is that
MASTER supplies conformational plausibility while the shape score supplies epitope
specificity, since MASTER itself never sees the epitope.

> **Status of the scoring.** Step 2 has been replaced. The original scorer correlated
> whole-length depth and torsion profiles; it scored hits in their own crystal frame,
> inverted the sign of the depth term, depended on which end of the epitope was listed
> first, and could not reject a candidate that physically collided with the antigen. It
> is superseded by a local shape fit ([§6](#6-stage--shape-complementarity-scoring)) and
> retained only to reproduce published benchmark numbers.
> [§10](#10-implementation-caveat--the-superposition-frame)
> records what was wrong, the fix, how it was validated, and — importantly — what the
> benchmark does *not* show: the corrected shape score does not single out the native
> loop, and its weights are unvalidated. Readers interpreting `complementarity_score`
> should read §10.4 and §11.

---

## 1. Introduction and motivation

The problem EpiFitScout addresses is hypothesis generation for antibody loop design:
given an epitope one wishes to bind, which CDR conformations already observed in the
PDB are plausible starting points?

Pure structural search answers only half of that. MASTER (Method of Accelerated Search
for Tertiary Ensemble Representatives) finds protein fragments whose backbone matches a
query fragment to within an RMSD cutoff. Run on a CDR loop, it returns loops of similar
shape — but it is given only the loop. **The epitope is never part of the MASTER query**,
so nothing in the search result reflects whether a hit could actually complement the
surface of interest. Ranking MASTER output by RMSD alone therefore ranks by
*resemblance to the loop you already have*, which is not the quantity of interest.

EpiFitScout's contribution is a second scoring pass that brings the epitope in. For
each MASTER hit it asks a geometric question — does this loop's surface profile
*interlock* with the epitope's? — and combines the answer with the RMSD so that final
rank reflects both conformational plausibility and epitope fit.

The complementarity model is deliberately coarse. It uses backbone α-carbons only, and
it tests two geometric signatures rather than computing buried surface area or an
energy. The two signatures are described in [§6](#6-stage-2--shape-complementarity-scoring).

---

## 2. System overview

### 2.1 Pipeline

```
                      ┌─────────────────────────────┐
  SAbDab CSV ────────▶│  build_chain_db.py          │
  (or PDB list)       │  RCSB download → createPDS  │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                           chain DB (pds.list)
                           25,226 chains
                                     │
  query PDB ──▶ QueryBuilder         │
                  │                  │
         ┌────────┴────────┐         │
         ▼                 ▼         ▼
    query_epitope      query_cdr ──▶ ① MASTER search
    (Fragment)         (Fragment)     backbone RMSD
         │                            sliding window
         │                                 │
         │                                 ▼
         │                      hits: (Fragment, Superposition, RMSD)
         │                                 │
         └────────────────▶ ② shape complementarity scoring
                              depth anti-correlation  (w=0.7)
                              torsion co-correlation  (w=0.3)
                                           │
                                           ▼
                            ③ ranking: 0.4·1/(1+RMSD) + 0.6·S_shape
                                           │
                                           ▼
                              list[ScoredHit], best first
```

### 2.2 Stage summary

| Stage | Module | Input | Output |
|---|---|---|---|
| ① Structural search | `master/runner.py`, `master/parser.py` | query CDR `Fragment` | `(Fragment, Superposition, rmsd)` per hit |
| ② Complementarity | `scoring/shape.py` | hit coords + query epitope | `S_shape` ∈ [0, 1] per hit |
| ③ Ranking | `ranking/ranker.py` | hits + RMSDs | `list[ScoredHit]` sorted by `final_score` |

Orchestration lives in `pipeline/search_pipeline.py` (`FragmentSearchPipeline`), which
exposes both the whole pipeline (`search`, `search_many`) and each stage individually
(`run_master`, `score_complementarity`, `rank`).

### 2.3 Dependencies

| Requirement | Role |
|---|---|
Python ≥ 3.10 | managed with `uv` |
| MASTER binaries `master`, `createPDS` | stage ① search and PDS conversion; Linux x86-64 only, separately licensed, pasted into `MASTER/bin/` |
| Chain database | stage ① search targets; ~6 GB, extracted into `data/` |
| SAbDab summary file | chain list for database construction (`database_builder/` only) |
| numpy | all geometry and scoring |
| scipy | `svd` in `kabsch_rotation`; `cdist` in the unused DTW path |
| BioPython | `QueryBuilder._get_chain_info` only (chain inspection) |
| PyYAML | `config_from_yaml` |

Configuration dataclasses are plain frozen dataclasses with no Hydra import at load
time (`config/schema.py`), keeping the core package light.

---

## 3. Database construction

The search target is a **chain** database, not a fragment database: full
variable-domain H/L chains are stored, and MASTER slides the query window over them at
search time. This means fragment boundaries are discovered by the search rather than
pre-enumerated.

### 3.1 Procedure

`ChainDbBuilder` (`db/chain_db_builder.py`) accepts chain assignments from either
reader, then for each chain:

1. Download the parent PDB entry from RCSB, via `RcsbDownloader` with an on-disk cache
   (`data/rcsb_cache`, 10 GB in the installed build).
2. Extract the chain's backbone with `extract_backbone_coords_by_range` and write it as
   a single-chain PDB into `pdb/`.
3. Convert that PDB to MASTER's binary PDS format by invoking `createPDS`, writing into
   `pds/`.
4. Record per-chain annotation (`pdb_id`, `chain`, `chain_type`) in `metadata.json`.

Builds are **resumable** — already-processed entries are skipped — which matters
because a full build is ~13k RCSB downloads.

### 3.2 Two input modes

| Mode | Reader | Input |
|---|---|---|
| A (default) | `SAbDabMetadataReader` | a SAbDab summary file, tab- or comma-separated (delimiter detected); yields H/L chain assignments per entry |
| B | `PdbListReader` | a plain text file, one `1abc [H [L]]` per line; bare ID means all protein chains |

```bash
python database_builder/build_database.py                  # build or update
python database_builder/build_database.py --dry-run        # what would change
python database_builder/build_database.py --limit 50       # fast dev build
python database_builder/build_database.py --pdb-list my_pdbs.txt   # mode B
python database_builder/build_database.py --rebuild-pds    # after changing MASTER
```

Paths are overridable with `--create-pds`, `--metadata`, `--output-dir`, `--rcsb-cache`.

#### Updating an existing database

Running the builder against an existing database is an *update*, and is built to be
repeated months apart by someone who no longer has the RCSB cache:

* a chain whose PDB already exists is **not re-downloaded**;
* a PDS is **rebuilt when missing, empty, or older than its PDB**, so a structure
  revised on RCSB refreshes its search target;
* each PDS is written to a temporary name and renamed on success, so a failed or
  interrupted `createPDS` cannot leave a truncated target that later runs trust;
* the list files and `metadata.json` are derived from what is on disk, so a run over
  a subset of entries never truncates the database;
* chains that cannot be extracted are remembered in `skipped.json` so they are not
  re-downloaded every run (`--retry-skipped` overrides); transient download failures
  are deliberately *not* remembered;
* a PDS with no PDB is removed.

These rules exist because the original builder did the opposite on four counts:
it re-downloaded every chain on every run, rewrote the lists from only the entries
passed in, trusted a truncated 8-byte PDS left by a failed `createPDS`, and never
refreshed a PDS whose PDB had changed. `tests/test_builder.py` pins each one.

### 3.3 Output layout and relocatability

```
data/sabdab_chains.db/
├── pdb/             full-chain PDB files   (1.5 GB — the coordinate source)
├── pds/             MASTER binary format   (4.1 GB — the search target)
├── pds.list         relative paths, passed to MASTER
├── db.list          relative paths to the PDB files
├── metadata.json    per-chain annotation
├── skipped.json     chains that cannot be extracted
└── BUILD_INFO.json  when and from which SAbDab file this was built
```

A deliberate design point: **`pds.list` and `db.list` store paths relative to the
database root** (`pds/8RTW_B_H.pds`, not an absolute path), so the database can be moved
between machines or directories. MASTER itself requires absolute paths, so
`resolve_pds_list` (`master/runner.py`) rewrites the list into a temporary file at
search time, resolving each relative entry against *the list file's own parent
directory*. The temp file is removed via `atexit`.

The split between `pdb/` and `pds/` is load-bearing for stage ①: MASTER searches the
PDS files but reports matches by path, and the parser recovers coordinates from the
sibling `pdb/` file (see [§5.4](#54-match-parsing-and-fragment-reconstruction)).

---

## 4. Query specification

### 4.1 The `Fragment` type

Every fragment — query or hit — is a `Fragment` (`types/fragment.py`):

| Field | Meaning |
|---|---|
| `pdb_id`, `chain` | provenance |
| `residue_range` | `(first, last)` inclusive, PDB residue sequence numbers |
| `coords` | **`(N, 4, 3)`** — N residues × `[N, CA, C, O]` × `[x, y, z]` |
| `sequence` | one-letter sequence |
| `metadata` | free dict; carries `cdr_type`, `numbering_scheme`, `seg_start`/`seg_end` |

Convenience properties: `length`, `ca_coords` → `(N, 3)`, `ca_centroid`, `cdr_type`.
Equality and hashing use only `(pdb_id, chain, residue_range)`, deliberately ignoring
coordinates.

The `(N, 4, 3)` backbone-only representation is the single most consequential modelling
choice in the tool: **no side chains anywhere in the pipeline.** Backbone atom order is
fixed by `BACKBONE_ATOMS = ("N", "CA", "C", "O")` in `utils/io.py`. Where a carbonyl O
is absent from the source file, `_predict_carbonyl_O` reconstructs it geometrically
rather than dropping the residue.

### 4.2 `QueryBuilder`

`QueryBuilder` (`query/builder.py`) turns a complex PDB into the two query fragments.

```python
import epifitscout

qb = epifitscout.QueryBuilder("my_complex.pdb")
qb.describe()        # chains, residue counts, ranges, first 20 aa

query_cdr     = qb.get_fragment("H", 100, 110)   # heavy-chain CDR-H3
query_epitope = qb.get_fragment("A",  45,  60)   # antigen epitope residues
```

Three selection methods:

| Method | Use |
|---|---|
| `get_fragment(chain, res_start, res_end)` | explicit residue range |
| `get_imgt_cdrs(chain, cdr_types=None, flank=0)` | auto-extract CDR loops from an **IMGT-renumbered** chain; returns `dict[str, Fragment]` keyed `H1`…`L3` |
| `get_surface_patch(chain, center_res, radius_res=8)` | a window centred on a key contact residue — convenient for defining an epitope patch |

`get_imgt_cdrs` uses fixed IMGT core ranges, and `flank > 0` widens them symmetrically
into framework:

| CDR | IMGT range | | CDR | IMGT range |
|---|---|---|---|---|
| H1 | 27–38 | | L1 | 27–38 |
| H2 | 56–65 | | L2 | 56–65 |
| H3 | 105–117 | | L3 | 105–117 |

Because these are raw IMGT numbers, `get_imgt_cdrs` is only correct on an
IMGT-renumbered structure (e.g. as distributed by SAbDab); it logs a warning when no
CDR extracts. `describe()` exists precisely so the user can confirm the numbering
before committing to a range.

---

## 5. Stage ① — MASTER backbone search

`MASTERRunner` (`master/runner.py`) is a subprocess wrapper; `parse_match_file`
(`master/parser.py`) turns its output back into `Fragment` objects.

### 5.1 Query preparation

MASTER consumes PDS, not PDB, so the query makes two hops:

1. `_write_query_pdb` serialises the query `Fragment` to `query.pdb` via
   `fragment_to_pdb_string`.
2. `_convert_to_pds` runs
   `createPDS --type query --pdb query.pdb --pds query.pds` (30 s timeout).

### 5.2 Search invocation

```
master --query query.pds \
       --targetList <resolved pds.list> \
       --rmsdCut <rmsd_threshold> \
       --matchOut matches.txt \
       [--topN <max_hits>] \
       [--nThreads <n_threads>]
```

`--topN` is added when `max_hits > 0`; `--nThreads` only when `n_threads > 0`, since
that path requires an OpenMP build of MASTER. Non-zero exit or a timeout
(`timeout_seconds`, default 420 s) raises `MASTERExecutionError` carrying MASTER's
stderr. `_validate_paths` checks the binary, `createPDS`, and the database list up
front so a missing file fails fast rather than inside the subprocess.

**What MASTER returns, and what it does not.** The match file carries one line per hit:

```
  0.87428 /abs/path/to/pds/4KUZ_H_H.pds [(0,18)]
```

— an RMSD, the matched target, and the matched segment's **start and end indices within
that chain**. It is run with `--matchOut` only. It is *not* run with `--structOut`, so
MASTER writes no superposed coordinates, and the rigid-body transform that would place
the hit onto the query is not emitted. The parser therefore computes that transform
itself, by Kabsch superposition ([§10](#10-implementation-caveat--the-superposition-frame)).

### 5.3 Sharded parallel search

With `max_workers > 1`, `MASTERRunner.__init__` pre-splits the resolved PDS list into
`max_workers` shard files by ceiling division (`_make_shards`). `run_sharded` then
launches one MASTER process per shard through a `ThreadPoolExecutor`, and
`_merge_match_files` concatenates the shard outputs, re-sorts all lines by RMSD, and
truncates to a global `max_hits`. The merged file has the same format as a single run's,
so the parser is unaffected. All shard temp files are `atexit`-unlinked.

Note this parallelises *one query over the database*. `search_many` is the orthogonal
axis — several query pairs at once, each still sharded internally.

### 5.4 Match parsing and fragment reconstruction

`parse_match_file` applies, per line:

1. Regex match on `rmsd`, `path`, `(seg_start, seg_end)`; non-matching lines skipped.
2. Discard hits with `rmsd > rmsd_threshold` (a second filter after MASTER's own
   `--rmsdCut`).
3. `_pdb_path_for_pds` maps the matched `<db>/pds/NAME.pds` → `<db>/pdb/NAME.pdb`
   (with a legacy fallback for a flat `chains/` layout). Missing PDB → hit skipped.
4. `_load_fragment` reads the **whole chain's** backbone, then slices
   `[seg_start : seg_end + 1]` to recover exactly the matched window. Identity is parsed
   from the filename convention `{PDBID}_{CHAIN}_{TYPE}.pdb`.
5. Superpose the hit's Cα trace onto the query's by Kabsch ([§10](#10-implementation-caveat--the-superposition-frame)).
   Hits whose residue count differs from the query's, or whose superposed RMSD
   disagrees with MASTER's reported value by more than 0.01 Å, are dropped and
   counted in a warning.
6. Stop at `max_hits`; finally sort ascending by RMSD.

Each hit is emitted as `(Fragment, Superposition, rmsd)`: the fragment in its own
crystal frame, the transform that lays it onto the query, and MASTER's RMSD. The RMSD
actually achieved by the superposition is in `fragment.metadata["superposed_rmsd"]`.

---

## 6. Stage ② — shape complementarity scoring

This is the tool's distinctive algorithmic content, implemented as
`ShapeComplementarityScorer` in `scoring/shape.py`. It consumes a hit's coordinates and
the query epitope's coordinates and returns a single scalar in [0, 1].

Both components operate on **CA atoms only** (`_get_ca` collapses `(N, 4, 3)` → `(N, 3)`)
and both return a value in [−1, 1] before weighting. The scorer returns `0.0`
immediately if either fragment has fewer than 4 residues.

### 6.1 Component 1 — depth-profile anti-correlation (weight 0.7)

The underlying claim is the hand-in-glove intuition made quantitative: **where the loop
protrudes toward the epitope, a complementary epitope is recessed**, and vice versa.
Measured as 1-D profiles along a common axis, complementary surfaces are therefore
*anti-correlated*.

**Step 1 — the interface normal.** Take the centroid-to-centroid direction

$$
\hat{n}_{0} = \frac{\bar{c}_{\text{epi}} - \bar{c}_{\text{cdr}}}{\lVert \bar{c}_{\text{epi}} - \bar{c}_{\text{cdr}} \rVert}
$$

**Step 2 — backbone correction.** A loop that extends partly *along* the approach
direction would contaminate the depth signal with its own lateral extent. So the CDR's
principal axis $\hat{b}$ (first right-singular vector from SVD of the centred CDR CAs)
is projected out:

$$
\hat{n} = \frac{\hat{n}_{0} - (\hat{n}_{0}\cdot\hat{b})\,\hat{b}}{\lVert \hat{n}_{0} - (\hat{n}_{0}\cdot\hat{b})\,\hat{b} \rVert}
$$

A subtle and well-commented detail: the SVD uses **the CDR alone, never CDR+epitope
combined**. In a well-separated pair the interface direction itself dominates the
combined variance, so a combined SVD would identify the approach axis as the "backbone
direction" and delete exactly the signal being measured. If the correction collapses
$\hat{n}$ (the loop runs end-on at the epitope), the code falls back to the uncorrected
$\hat{n}_0$.

**Step 3 — projection and centring.** Project each CA onto $\hat{n}$ and remove the
mean, since only profile shape matters, not absolute offset:

$$
d^{\text{cdr}}_i = \mathbf{x}^{\text{cdr}}_i \cdot \hat{n} - \overline{\mathbf{x}^{\text{cdr}} \cdot \hat{n}},
\qquad
d^{\text{epi}}_j = \mathbf{x}^{\text{epi}}_j \cdot \hat{n} - \overline{\mathbf{x}^{\text{epi}} \cdot \hat{n}}
$$

**Step 4 — length alignment.** The two profiles generally differ in length. `_interp_align`
resamples both to `max(N, M)` by linear interpolation over a normalised parameter, so
0% along one maps to 0% along the other (N-terminus to N-terminus).

**Step 5 — score.**

$$
S_d = -\,\mathrm{pearson}\!\left(d^{\text{cdr}}, d^{\text{epi}}\right)
$$

The negation is the whole point: anti-correlation is complementarity.

> **Why not DTW.** Dynamic time warping was implemented but deliberately
> *not* used by the scorer, and the reason is recorded in the module docstring: DTW
> minimises paired distance, so it aligns CDR peaks with epitope **troughs** — points of
> equal *value* — rather than points at equal *relative position*. That destroys the very
> anti-correlation the score depended on. It has since been removed, along with the
whole index-to-index correspondence it belonged to: the local fit matches each loop
residue to its own nearest epitope residue and needs no alignment at all.

### 6.2 Component 2 — backbone torsion co-correlation (weight 0.3)

The second claim is that loops maintaining contact along their length tend to share a
similar 3D twist and handedness.

**Smoothing.** `_smooth_ca` applies a 3-point moving average to the CA trace
(endpoints unchanged) to stabilise the second-difference quantities below.

**Discrete Frenet–Serret torsion.** For the smoothed trace, with segment vectors
$\Delta_i = \mathbf{x}_{i+1} - \mathbf{x}_i$:

$$
\hat{T}_i = \frac{\Delta_i}{\lVert\Delta_i\rVert}, \qquad
\hat{B}_i = \frac{\hat{T}_i \times \hat{T}_{i+1}}{\lVert \hat{T}_i \times \hat{T}_{i+1}\rVert}, \qquad
\hat{N}_i = \frac{\hat{T}_{i+1} - \hat{T}_i}{\lVert \hat{T}_{i+1} - \hat{T}_i \rVert}
$$

$$
\tau_i = -\,\frac{\left(\hat{B}_{i+1} - \hat{B}_i\right)\cdot \hat{N}_i}{\bar{s}_i}
$$

where $\bar{s}_i$ is the local mean segment length. A trace of $N$ residues yields
$N-3$ torsion values; fewer than 4 residues raises `ValueError`, caught by the scorer.
All norms are floored at `1e-8` against division by zero.

**Why torsion.** $\tau$ is **frame-independent** — invariant under rotation and
translation — so it characterises intrinsic 3D shape without reference to any shared
coordinate system. This is why it was unaffected by the frame defect recorded in
[§10](#10-implementation-caveat--the-superposition-frame).

**Planar guard.** If $\max|\tau| < 10^{-4}$ the profile is returned as exact zeros.
A near-planar loop has torsion that is numerically pure noise, and correlating noise
would yield a misleading $r$. `_pearson` independently returns `0.0` for any constant
signal (σ < 1e-8) or series shorter than 2.

**Score.** After the same linear-interpolation alignment:

$$
S_\tau = +\,\mathrm{pearson}\!\left(\tau^{\text{cdr}}, \tau^{\text{epi}}\right)
$$

Positive here: similar twist is the favourable case.

### 6.3 Combination

$$
\text{raw} = w_d S_d + w_\tau S_\tau \;\in [-1, 1], \qquad
S_{\text{shape}} = \mathrm{clip}\!\left(\frac{\text{raw}+1}{2},\, 0,\, 1\right)
$$

Defaults $w_d = 0.7$, $w_\tau = 0.3$ (`ScoringConfig`). An uninformative hit lands near
0.5, not 0 — worth remembering when reading raw scores.

---

## 6A. Stage ② as built — local shape fit

The profile scorer in §6 is superseded. What runs now answers a different, narrower
question: **where along the epitope does this loop fit, and over how long a stretch?**

### 6A.1 Inputs

Each MASTER hit is first *grafted*: the Kabsch transform of §10 lays it onto the CDR
site, so it occupies the antibody-antigen frame where the template loop was. Measured
on 1A2Y, this moves fragments from a median of 68 Å (up to 421 Å) from the epitope to
7.0 Å, against the template loop's own 6.9 Å.

The epitope is an `EpitopeSurface`: the antigen residues **the user names**, with every
atom including side chains. It need not be in contact with the template's CDR — a design
search commonly targets an epitope the template antibody does not engage. The earlier
`get_surface_patch` took a sequence window (`centre ± 8`) and selected by residue number,
which on the benchmark complexes put most of the patch 10-19 Å away with only 1-2 loop
residues in contact; it is deprecated.

Fragments carry **backbone atoms only** — the chain database discards side chains at
build time — so every distance is fragment backbone to antigen heavy atom.

### 6A.2 Method

1. For each loop residue, the distance to its nearest antigen atom, and which residue
   that is. Each residue finds its own partner, so no correspondence between loop and
   epitope is assumed; the result is unchanged by reversing the epitope's residue order.
2. Classify: **collision** below 2.5 Å, **contact** out to 8.0 Å.
3. Find the longest run of consecutive residues in contact with no collision. This is
   the "slide": the run tracks along the surface wherever it goes.
4. Score `contact_quality × (matched_length / loop_length)`, and **zero if anything
   collides** — so step 2 both filters and ranks.

### 6A.3 Calibration

Thresholds come from the five native complexes, not assertion. Over 84 native CDR
residues, backbone-to-antigen distance: closest approach ever observed **2.6 Å**,
p25 5.0 Å, median 7.7 Å; 36% within 6 Å, 52% within 8 Å. Hence collision below 2.5 Å
and contact to 8 Å, and hence the expectation that a native matches roughly half its
loop, not all of it.

### 6A.4 What it changed

| | profile scorer | local fit |
|---|---|---|
| Perfect synthetic nest | 0.150 | **0.481** |
| Synthetic collision | 0.850 | **0.000** (rejected) |
| Native shape percentile, 1A2Y / 3GBN / 3NGB / 2NY7 | 50 / 35 / — / — | **74 / 35 / 78 / 95** |
| Candidates rejected for colliding | none possible | 15-101 per case |
| 3GBN top-ranked non-self hit | 7WE9:C, collides at 2.4 Å | 7UZ5:H, 0 collisions, 7 residues matched |

On 3GBN three of the profile scorer's top ten physically collide with the antigen; on
2NY7, four of ten. `tools/visualize_fit.py` exports any of this for PyMOL.

### 6A.5 Limits

- **No side chains on the fragment**, so this measures backbone approach, not packing.
- **A distance cutoff reaches sideways**: a residue can count as in contact while beside
  the patch rather than facing it. This is exactly what the Lawrence & Colman Sc
  statistic (*J Mol Biol* 234:946, 1993) exists to address, via normal dot products
  weighted by a distance decay rather than distance alone.
- `normal_quality` is computed as a cheap stand-in for that and **reported but not used**
  (`weight_normals` defaults to 0): on the natives it ranges 0.09-0.42 and the tightest
  interface scores lowest, so the centroid-derived normals are not trustworthy. Real Sc
  needs solvent-excluded surfaces of fully-atomic structures; computing it would require
  modelling side chains onto each grafted fragment.
- **Validation remains thin**: five complexes, no binding data. The native ranks first in
  all five, but every candidate is within 2 Å RMSD of the query by construction, so they
  are near-identical loops and the discrimination available is small.

---

## 7. Stage ③ — ranking

`HitRanker.rank` (`ranking/ranker.py`) folds MASTER's RMSD together with $S_{\text{shape}}$.
RMSD is on $[0,\infty)$ and lower is better, so it is mapped to $[0,1]$ increasing:

$$
S_{\text{final}} = w_{\text{rmsd}}\cdot\frac{1}{1+\mathrm{RMSD}} \;+\; w_{\text{shape}}\cdot S_{\text{shape}}
$$

Defaults $w_{\text{rmsd}} = 0.4$, $w_{\text{shape}} = 0.6$ (`RankingConfig`). The method
raises `ValueError` if the hit list and RMSD list differ in length, and returns `[]` for
no hits; otherwise it emits `ScoredHit` objects sorted by `final_score` **descending**.

Note the transform is non-linear and compresses at the good end: RMSD 0.0 → 1.00,
0.5 → 0.67, 1.0 → 0.50, 2.0 → 0.33. Between two hits at 0.3 Å and 0.6 Å the RMSD term
differs by only 0.08 before weighting, so at a tight `rmsd_threshold` the shape term
dominates ranking in practice.

### `ScoredHit` fields

| Field | Type | Meaning |
|---|---|---|
| `fragment` | `Fragment` | the retrieved hit (`pdb_id`, `chain`, `residue_range`, coords, sequence) |
| `superposition` | `Superposition` | Kabsch transform laying the hit's Cα trace onto the query's; `apply(fragment.coords)` is the hit in the query frame ([§10](#10-implementation-caveat--the-superposition-frame)) |
| `rmsd_cdr` | `float` | backbone RMSD reported by MASTER (Å) |
| `complementarity_score` | `float` | $S_{\text{shape}}$ ∈ [0, 1] |
| `final_score` | `float` | $S_{\text{final}}$ ∈ [0, 1]; the sort key |

---

## 8. Configuration and hyperparameters

Values resolve in three layers: frozen dataclass defaults (`config/schema.py`) ←
`epifitscout/conf/config.yaml` (`config_from_yaml`) ← explicit keyword arguments to
`search()`.

Paths are not part of that cascade. `epifitscout/paths.py` derives the MASTER and
database locations from the position of the code in the repository
(`MASTER/bin/`, `data/sabdab_chains.db/`), which is why the package is installed
editable. `MASTERConfig.binary_path` and `database_path` default to `None`,
meaning "resolve at use time"; setting them, or `EPIFITSCOUT_MASTER_DIR` /
`EPIFITSCOUT_DB`, overrides the location.

| Parameter | Default | Effect of increasing |
|---|---|---|
| `master.rmsd_threshold` | 2.0 Å | More hits, looser conformational match; more candidates for shape scoring to discriminate |
| `master.max_hits` | 500 | More hits retained (`--topN`, plus the parser cap and the shard-merge cap) |
| `master.timeout_seconds` | 420 | Longer MASTER runs tolerated before `MASTERExecutionError` |
| `master.n_threads` | 0 (off) | MASTER-internal threading; **requires an OpenMP build** |
| `scoring.weight_depth` | 0.7 | Emphasise protrusion/recession complementarity |
| `scoring.weight_tau` | 0.3 | Emphasise shared 3D twist over depth interlock |
| `ranking.weight_rmsd` | 0.4 | Favour conformational similarity to the query loop |
| `ranking.weight_shape` | 0.6 | Favour epitope complementarity |
| `max_workers` | 6 | Database shards searched in parallel (≈ N× speedup); also parallel queries in `search_many` |

Within each pair, weights are documented as non-negative and summing to 1. **Neither
sum is enforced in code** — a pair summing to something other than 1 silently rescales
the score's range and breaks the [0, 1] guarantee the docstrings assert.

One inconsistency worth knowing: the fallback literals inside `config_from_yaml` differ
from the dataclass defaults (`rmsd_threshold` 1.5 vs 2.0; `timeout_seconds` 300 vs 420).
They apply only when the YAML exists but omits that key. The shipped
`conf/config.yaml` sets both explicitly, so observed behaviour is 2.0 / 420.

A missing default config raises `SetupError` rather than silently falling back to
those literals: the fallbacks differ from the shipped values, so substituting them
would change `rmsd_threshold` and therefore the results.

---

## 9. Usage

### 9.1 Prerequisites

```bash
git clone https://github.com/fbabd/EpiFitScout && cd EpiFitScout
pip install -e .        # editable: paths are resolved relative to the clone
```

Then paste two things into the repository (each folder has its own README):

* **MASTER v1.6 binaries** (`master-bin-v1.6.tar.gz`) into `MASTER/bin/` — download
  from <https://grigoryanlab.org/legacy.php?sec=download&soft=MASTER> (name,
  institution and email optional; answer the arithmetic question), then
  `tar -xzf master-bin-v1.6.tar.gz -C MASTER/ && chmod +x MASTER/bin/*`;
* **the dataset** (`epifitscout-database-v*.tar.xz`) into `data/` —
  `tar -xJf <file> -C data/`.

```bash
python check_setup.py --search     # confirm both, and run a real search
```

MASTER is Linux x86-64 only and separately licensed (GPL for academic use), so it
is neither bundled nor redistributed here.

### 9.2 Build or update the database

Only needed for a newer SAbDab snapshot or a custom entry set. See
[§3.2](#32-two-input-modes) and `database_builder/README.md`.

```bash
python database_builder/build_database.py --dry-run   # what would change
python database_builder/build_database.py             # build or update
```

The dataset archive ships **both** `pdb/` and `pds/`, so users have no build step.
This is the reverse of an earlier design that shipped coordinates only and
regenerated the targets locally: that saved download size at the cost of a
mandatory extra command. The targets are deterministic (`createPDS` output was
byte-identical across 400 sampled chains), which is what makes either choice
sound. After changing MASTER versions, regenerate with `--rebuild-pds`.

### 9.3 Run a search

```python
import epifitscout   # pip install epifitscout; no sys.path manipulation needed

qb = epifitscout.QueryBuilder("my_complex.pdb")
qb.describe()                                    # confirm chains and numbering first

query_cdr     = qb.get_fragment("H", 100, 110)
query_epitope = qb.get_fragment("A",  45,  60)

hits = epifitscout.search(query_cdr, query_epitope)
```

`search()` auto-resolves the binary and database through `epifitscout/paths.py`,
loads the packaged `conf/config.yaml`, and accepts per-call overrides:

```python
hits = epifitscout.search(query_cdr, query_epitope, weight_depth=0.9, max_workers=12)
```

A missing binary or database raises `MissingBinaryError` or
`MissingDatabaseError` (both subclasses of `SetupError`, itself a
`RuntimeError`) naming the folder and the command that fix it.

### 9.4 Inspect results

```python
print(f'{"Rank":<5} {"PDB:chain":<14} {"Residues":<12} {"RMSD":>6}  {"Shape":>6}  {"Score":>7}')
for i, h in enumerate(hits[:15], 1):
    res = f'{h.fragment.residue_range[0]}-{h.fragment.residue_range[1]}'
    print(f'{i:<5} {h.fragment.pdb_id}:{h.fragment.chain:<10} {res:<12} '
          f'{h.rmsd_cdr:>6.3f}  {h.complementarity_score:>6.3f}  {h.final_score:>7.4f}')
```

### 9.5 Stage-by-stage use

For diagnostics, drive the stages directly rather than through `search()`:

```python
from epifitscout import FragmentSearchPipeline, PipelineConfig
from pathlib import Path

pipeline = FragmentSearchPipeline(PipelineConfig(max_workers=6))
master_hits = pipeline.run_master(query_cdr, Path("run1"))        # ①
scored      = pipeline.score_complementarity(master_hits, query_epitope)  # ②
ranked      = pipeline.rank(scored, [r for _, _, r in master_hits])       # ③
```

Passing `output_dir` to `search()` likewise retains MASTER's intermediates
(`query.pdb`, `query.pds`, `matches.txt`) instead of discarding a temp directory.

### 9.6 Worked example

`example/example_1a2y.py` runs the 1A2Y Fab / hen lysozyme complex end to end;
`example/_pymol_render.py` renders the top and bottom hits for visual inspection.

---

## 10. Implementation caveat — the superposition frame

**Status: fixed in this version.** An earlier version scored every hit in its own
crystal frame. This section records what was wrong, how it was fixed, how the fix was
validated, and what it did and did not change.

### 10.1 What was wrong

The depth component of [§6.1](#61-component-1--depth-profile-anti-correlation-weight-07)
rests on a premise stated in the scorer's own docstring (`scoring/shape.py:8`):

> *"The CDR hit is already in the query frame (MASTER superposition), so the CDR and
> epitope share a common coordinate system."*

A MASTER match line says *that* a database loop resembles the query, not *where* it
sits. The earlier implementation never put it there:

| Observation | Location (earlier version) |
|---|---|
| Every hit was constructed with `Superposition.identity()` | `master/parser.py` |
| The match-line regex captures only `rmsd`, `path`, `seg_start`, `seg_end` — no transform | `master/parser.py` |
| MASTER is invoked with `--matchOut` but not `--structOut`, so no superposed coordinates are written | `master/runner.py` |
| Coordinates are reloaded from the database's original crystal PDB and sliced by index | `master/parser.py` `_load_fragment` |
| So `aligned = sup.apply(frag.coords)` was a no-op | `pipeline/search_pipeline.py` |
| `kabsch_superpose` existed but was never called | `utils/geometry.py` |

Each fragment was therefore scored **in its own crystal frame**, wherever that sat
relative to the query. Measured independently by the downstream `epifitgen` package
across 499 real 1A2Y hits, retrieved fragments sat a median of 86 Å, and up to 1005 Å,
from the design site.

**Per component.** Torsion ($S_\tau$, weight 0.3) was unaffected: Frenet–Serret torsion
is invariant under rotation and translation ([§6.2](#62-component-2--backbone-torsion-co-correlation-weight-03)).
Depth ($S_d$, weight 0.7) was compromised: its approach axis was the centroid-to-centroid
vector between two unrelated frames, an artifact of crystal placement rather than binding
geometry. The failure was quiet: mean-centring cancelled the large offset, so the
pipeline still returned a well-formed, deterministic number in [0, 1], with no exception
and no warning.

### 10.2 The fix

`parse_match_file` now takes the query fragment (a required argument, so it cannot be
forgotten) and, for each hit, Kabsch-superposes the hit's Cα trace onto the query's. The
resulting rotation and translation are stored as the hit's `Superposition`, so
`superposition.apply(fragment.coords)` gives all four backbone atoms in the query frame.

**Why Cα.** MASTER's reported RMSD is computed over Cα atoms. Measured on 500 real hits,
the Kabsch RMSD on Cα alone reproduces MASTER's value to about $10^{-5}$ Å; adding N and
C moves it away by up to 0.04 Å, and adding O by up to 0.17 Å. Superposing on Cα is
therefore both the right alignment and an exact cross-check.

**Safeguards.** A hit is dropped, and counted in a warning, if its residue count differs
from the query's (no one-to-one correspondence exists) or if its superposed RMSD
disagrees with MASTER's by more than 0.01 Å (the reloaded residues are then not the ones
MASTER aligned, and any frame computed from them would be wrong). In the benchmark
neither case occurred.

### 10.3 Validation

- **Invariance property** (`tests/test_superposition.py`): a hit's score is identical, to
  $10^{-6}$, whichever rigid motion its crystal is given. A companion test confirms the
  old identity placement *does* vary with frame, so the test is sensitive to this defect.
- **Independent implementation** (`benchmark/rescore.py`): the library's scores equal a
  separately written Kabsch computation to four decimals on all 1,337 benchmark hits.
- **Reproduction**: re-running the 2026-10-05 benchmark returns the identical hit set for
  every case, and the as-shipped scores reproduce exactly, so the before/after comparison
  below is like-for-like.
- **Same loop, different crystals**: three crystals of one antibody (2NY7:H, 1HZH:K,
  8SXI:G) scored 0.330 / 0.783 / 0.724 before the fix and 0.330 / 0.326 / 0.327 after.
  Four copies of one Fab in the 3NGB crystal scored between 0.275 and 0.645 and now agree
  to within 0.003.

### 10.4 What it changed, and what it did not establish

Native = the query's own chain, retrieved at RMSD 0. Each arrow is *before the fix → after*.

| Case | H3 | Hits | Native shape | Native rank (final) | Best other hit, shape | Native shape percentile |
|---|---|---|---|---|---|---|
| 1A2Y | 10 | 500 | 0.520 | 101 → 1 | 0.891 → 0.712 | 45% → 21% |
| 3GBN | 14 | 500 | 0.557 | 1 → 1 | 0.834 → 0.708 | 70% → 34% |
| 3NGB | 14 | 314 | 0.645 | 1 → 1 | 0.806 → 0.787 | 91% → 59% |
| 2NY7 | 20 | 23 | 0.330 | 6 → 1 | 0.783 → 0.351 | 14% → 86%* |

\*Only 23 hits, all scoring 0.32 ± 0.01 after the fix; the percentile separates
differences of about 0.004.

What it changed: scores stopped depending on crystal placement; the best-looking
non-native hits fell sharply (the apparent hits that beat the native were largely
artifacts); and the score distribution compressed (standard deviation over non-native
hits from 0.12–0.21 to 0.01–0.07).

What it did **not** establish:

- The native still scores *below the best other-structure hit on shape alone in all four
  cases*, and sits below the median in 1A2Y and 3GBN. It reaches rank 1 through the RMSD
  term, being at RMSD 0 by construction.
- The fix removes a confound. It does not show that the depth metric measures
  complementarity, and five native-versus-decoy cases would not establish that anyway.
- Retrieval coverage is unaffected and still collapses with H3 length: 26 residues
  (1MEL) returns nothing within 2.0 Å.

---

## 11. Limitations and future directions

Beyond [§10](#10-implementation-caveat--the-superposition-frame):

- **Backbone-only, CA-only.** `Fragment` carries N/CA/C/O and the scorer uses CA alone.
  Side chains — which mediate most of the chemistry and much of the steric fit of a real
  interface — are invisible. Two loops with identical CA traces and entirely different
  side-chain presentation score identically.
- **Geometry without chemistry.** There is no electrostatic, hydrophobicity,
  hydrogen-bonding, or buried-surface-area term, and no clash detection. A hit can score
  well while interpenetrating the epitope.
- **No solvent accessibility or burial.** "Depth" is a projection onto one axis, not a
  measure of how buried a residue is. True concavity is not computed.
- **Database redundancy.** SAbDab is heavily redundant and the pipeline does nothing to
  cluster it, so the top of a ranked list is readily dominated by several structures of
  the same antibody. Observed downstream on 1A2Y: the top four hits by score were the
  same antibody with the same loop sequence, all within 0.04 Å. Any application needing
  distinct hypotheses must deduplicate or diversity-select after retrieval.
- **No self-hit exclusion.** Querying with a loop whose parent entry is in the database
  retrieves it at RMSD ≈ 0 and ranks it first. Benchmarking must exclude it explicitly.
- **Planar-loop blind spot.** The $\max|\tau| < 10^{-4}$ guard correctly suppresses
  noise, but the consequence is that a planar loop contributes $S_\tau = 0$, i.e.
  mid-scale rather than "unknown" — there is no mechanism to reweight onto depth when
  torsion is uninformative.
- **Unvalidated weights, and weak discrimination.** All four weights (0.7/0.3, 0.4/0.6)
  are set by assertion, not fit to data. No benchmark relates $S_{\text{shape}}$ to any
  measured binding outcome. With the frame fixed ([§10.4](#104-what-it-changed-and-what-it-did-not-establish)),
  the native loop still does not stand out on shape alone and scores occupy a narrow band
  (standard deviation 0.01–0.07), so the shape term currently contributes little to
  ranking beyond the RMSD term. Calibrating it against a held-out set of known complexes
  is the natural next step, and is now meaningful because the depth axis is no longer an
  artifact.

---

## 12. Code map

| Module | Responsibility |
|---|---|
| `epifitscout/__init__.py` | Public API: `search()`, `config_from_yaml`, three-layer config merge |
| `paths.py` | Locations of `MASTER/bin` and `data/sabdab_chains.db`; error messages that say which folder to fix |
| `conf/config.yaml` | Shipped hyperparameters |
| `config/schema.py` | Frozen config dataclasses: `MASTERConfig`, `ScoringConfig`, `RankingConfig`, `PipelineConfig` |
| `types/fragment.py` | `Fragment` (N, 4, 3 backbone) and `ScoredHit` |
| `types/superposition.py` | `Superposition` rigid transform: `apply`, `inverse`, `identity` |
| `query/builder.py` | `QueryBuilder`, `load_fragment_from_pdb`, IMGT CDR ranges |
| `utils/io.py` | PDB backbone extraction, carbonyl-O reconstruction, `fragment_to_pdb_string` |
| `utils/geometry.py` | Kabsch rotation/superposition, RMSD, centroids |
| `master/runner.py` | `createPDS`/`master` subprocess wrapper, PDS list resolution, sharding, merge |
| `master/parser.py` | Match-file parsing, PDS→PDB path mapping, fragment reconstruction |
| `scoring/local_fit.py` | `LocalFitScorer`: per-residue contact, collision rejection, longest clean run — the scorer the pipeline uses |
| `scoring/shape.py` | `ShapeComplementarityScorer`: superseded profile correlation, kept to reproduce published numbers |
| `types/epitope.py` | `EpitopeSurface`: user-chosen antigen residues, all atoms, order-free |
| `ranking/ranker.py` | `HitRanker`: RMSD transform, weighted combination, sort |
| `pipeline/search_pipeline.py` | `FragmentSearchPipeline`: stage orchestration, `search`, `search_many` |
| `db/sabdab_metadata.py` | SAbDab CSV reader → `SAbDabEntry` |
| `db/pdb_list_reader.py` | Plain PDB-list reader (mode B) |
| `db/rcsb_downloader.py` | RCSB fetch with on-disk cache |
| `db/chain_db_builder.py` | Chain database build and incremental update: skip-existing, PDS refresh, atomic writes, orphan removal |
| `db/verify.py` | One definition of "database is usable", shared by the updater, packager and `check_setup.py` |
| `check_setup.py` | Verify MASTER and the database are in place; `--search` runs a real example |
| `database_builder/build_database.py` | Build or update the database from SAbDab + RCSB |
| `database_builder/package_dataset.py` | Package a database as a downloadable archive |
| `tests/` | Test suite; `tests/data/mini_db` is a 14-chain fixture database (~400 kB) |
| `example/example_1a2y.py` | Worked 1A2Y example |
| `example/_pymol_render.py` | PyMOL rendering of top/bottom hits |

`utils/geometry.py::kabsch_superpose` is live: the parser uses it to graft each hit
onto the CDR site (§10). The previously-dead helpers `_dtw_align`,
`place_epitope_facing_cdr`, `rmsd_no_superpose`, `save_coords` and `load_coords` have
been removed (126 lines).
`utils/geometry.py::place_epitope_facing_cdr` serves the separate case of an epitope
drawn from a different structure, where no known complex geometry is available.
