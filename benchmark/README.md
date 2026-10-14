# Benchmark

What the geometric step actually buys you, measured on **102 antibody–antigen
complexes** drawn from SAbDab. Everything here is read-only with respect to
`epifitscout/`; no library code is modified.

The benchmark is organised as four questions. Each is a thing that could come
out against the method, not a demonstration arranged to come out for it.

## Q1. Does the geometric step add anything MASTER does not have?

MASTER ranks by backbone RMSD and never sees the antigen. If its hits could
simply be filtered by RMSD, this step would be redundant.

```
Hits that cannot be placed without colliding with the antigen:
  median 7% per complex, IQR 0-21%, range 0-71%
  76 of 102 complexes have at least one such hit

Can RMSD predict which hits collide? (AUC; 0.5 = no information)
  median AUC 0.559 over 76 complexes, IQR 0.472-0.689
  Wilcoxon vs 0.5: p = 0.00119

Pooled over 40,724 hits, collision rate by RMSD band:
  0.0-0.5 A:  0.4%   0.5-1.0 A:  9.2%
  1.0-1.5 A: 16.6%   1.5-2.0 A: 20.8%

Could a tighter RMSD cut replace the check?
  cut at 0.5 A: keeps 11.6% of hits, of which 0.4% still collide
  cut at 1.0 A: keeps 32.7% of hits, of which 6.0% still collide
  cut at 1.5 A: keeps 59.0% of hits, of which 10.8% still collide
  no cut:       keeps  100% of hits, of which 14.9% collide
```

RMSD is *weakly* informative about collisions — AUC 0.559 is barely above
chance, though reliably so. The practical consequence is the last block:
tightening the RMSD cut discards many placeable hits to remove few unplaceable
ones, so it is a poor substitute for looking at the antigen.

## Q2. Does the score detect real complementarity, or just proximity?

The native loop is scored against its own epitope and against decoy patches on
the same antigen: same residue count, sharing under 25% of their residues with
the true site, and **reachable** — the loop comes as close to them as to its
real epitope. Same antibody, same antigen, same frame; only the site moves.

```
55 complexes, 225 decoy patches
mean score: true 0.537 | decoy mean 0.260 | decoy best 0.326
vs mean decoy: true wins 55/55 (100%), p = 5.54e-11, Cohen's d = 2.71
vs best decoy: true wins 54/55  (98%), p = 6.18e-11, Cohen's d = 1.98
```

The reachability constraint is the point. An earlier version of this control
placed decoys ≥25 Å away; they scored 0.000 because the loop cannot reach them,
which tests proximity and is no evidence at all. That result was discarded. Here
the decoys are genuinely in contact (mean score 0.260, not zero), so what is
being compared is fit quality.

## Q3. Is the native loop recovered?

Each complex's own H3 is removed from consideration, the search is run, and we
ask where the native loop would rank among the hits.

```
native percentile among non-self hits, local fit: median 71%, IQR 46-89%
the superseded profile scorer, same cases:        median 51%, IQR 25-74%
above the 50th percentile in 73/102 complexes
Wilcoxon vs chance (50%): p = 4.43e-07
```

The second line is the comparison that matters for the method change: the
depth-profile scorer this replaced performed at chance.

## Q4. Does the score track binding energy?

```
74 complexes with SAbDab ΔG (-15.7 to -5.0 kcal/mol)
Spearman rho = -0.326, p = 0.00454
```

Weak, and reported so the claim is not overstated in either direction. A shape
term is not an affinity predictor and is not offered as one.

## What this does and does not establish

It establishes that the geometry carries real information about the epitope
(Q2), more than RMSD alone does (Q1), and that the current scorer recovers
native loops where its predecessor did not (Q3).

**It does not establish the central claim** — that epitope-aware ranking
produces better antibodies than conformation-only ranking. That needs the same
template and the same MASTER hits ranked with and without the epitope term, with
the resulting grafts compared; it has not been run. Nothing here involves a
designed antibody being tested, in silico or otherwise.

## Reproducing

```bash
python benchmark/run_benchmark.py                 # build cases from SAbDab, search each
python benchmark/decoy_epitopes.py                # Q2 specificity control
python benchmark/analyse_benchmark.py             # statistics and figure
```

`run_benchmark.py` assembles its own cases from the SAbDab summary file, so the
case list is not hand-picked; `--require-dg` restricts it to entries carrying a
ΔG, which is how the Q4 set is built.

| File | |
|---|---|
| `run_benchmark.py` | builds cases from SAbDab and runs the pipeline on each |
| `decoy_epitopes.py` | the Q2 reachable-decoy control |
| `analyse_benchmark.py` | Q1–Q4 statistics and `results/benchmark.png` |
| `rescore.py` | scores hits three ways, to show what the superposition fix changed |
| `prep_case.py`, `run_case.py` | single-case helpers |
| `nobio.py` | BioPython-free I/O used when the environment's BioPython is shadowed |
| `results/pre_fix_v0.1.0/` | kept so the regression that prompted the rewrite stays inspectable |

Run records are committed **xz-compressed** (`benchmark.jsonl.xz`): the hit lists hold
40,724 entries and compress to 7% of their size, which is the difference between a 1 MB
and an 11 MB clone. The scripts read either form — if you re-run and produce a plain
`.jsonl`, it is picked up in preference to the archive beside it, so there is nothing to
decompress by hand.

### A data trap worth knowing about

530 rows of the SAbDab summary name an antibody chain as the antigen. Taken at
face value, those cases measure a CDR against its own light chain — 8RTW scored
0.775 that way. `run_benchmark.py` filters them. If you build cases from SAbDab
yourself, filter them too.
