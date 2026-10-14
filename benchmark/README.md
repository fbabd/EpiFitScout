# EpiFitScout literature benchmark

Scripts that run five literature antibody–antigen cases through the pipeline
and measure what the scoring does. Everything here is read-only with respect to
`epifitscout/`; no library code is modified.

Two runs are kept, both against the 25,226-chain SAbDab database:

| Run | Library | Location |
|---|---|---|
| 2026-10-05 | v0.1.0 as first written: hits scored in their own crystal frame | `results/pre_fix_v0.1.0/` |
| 2026-10-13 | after the superposition fix (`REPORT.md` §10) | `results/` |

The search results (which hits are retrieved) are identical between the two;
only the scoring differs.

## Why `nobio.py` exists

BioPython in the environment these were first run in was a broken namespace shadow: `import Bio`
succeeds but `Bio.__file__` is `None`, and `Bio.Data.IUPACData` /
`Bio.SeqUtils.seq1` both fail to import. That takes out `QueryBuilder`,
`extract_backbone_coords_by_range` and `fragment_to_pdb_string`, i.e. the whole
pipeline.

`nobio.py` provides BioPython-free replacements for the two functions the
pipeline actually calls and monkeypatches them at their import sites. Behaviour
matches the originals, including insertion-code handling, so IMGT 111A/112A
survive. It also uppercases residue names — the chain database writes them
title-case (`Glu`, not `GLU`), which otherwise yields all-`X` sequences.

BioPython works in the current environment, so `nobio` is likely no longer
needed. It is kept so the 2026-10-13 run is strictly comparable with 2026-10-05;
removing the `nobio.install()` lines runs the library code unchanged.

## Files

| File | Does |
|---|---|
| `nobio.py` | BioPython-free PDB read/write + monkeypatch installer |
| `prep_case.py` | Fetch a PDB, detect CDR-H3, pick an epitope patch, emit a case line |
| `run_case.py` | Run the shipped pipeline on one case, print ranked hits as JSON |
| `rescore.py` | Score every hit three ways: raw crystal frame (v0.1.0), an independent Kabsch implementation, and the library's own |
| `cases.txt` | The five cases, as `id heavy_chain lo hi antigen_chain elo ehi` |
| `results/results.jsonl` | Output of `run_case.py` for all five cases (post-fix) |
| `results/rescored.jsonl` | Output of `rescore.py` for the four scorable cases (post-fix) |
| `results/pre_fix_v0.1.0/` | The same two files from the 2026-10-05 run |

## Method

**CDR-H3 detection** (`prep_case.py`) uses the conserved Cys … W-G-x-G motif:
H3 is the stretch strictly between the Cys that closes framework 3 and the Trp
that opens framework 4, taking the *last* such match in the chain (H3 is the
final CDR). Validated by reproducing the bundled 1A2Y loop `ARERDYRLDY` at
author residues 96–105 exactly.

**Epitope patch** mirrors `QueryBuilder.get_surface_patch`: the antigen residue
whose CA is nearest the H3 centroid, ± 4 residues, kept contiguous. Note the
tool models an epitope as one contiguous segment, so discontinuous epitopes are
not represented — and the score is sensitive to this choice (shifting the 1A2Y
patch by one residue moved the native shape score from 0.648 to 0.520).

**Frame correction.** A MASTER match says a loop resembles the query, not where
it sits, so each hit has to be laid onto the query before it is compared with the
query's epitope. v0.1.0 skipped this. The library now Kabsch-superposes each
hit's Cα trace onto the query's (`master/parser.py`). `rescore.py` scores every
hit three ways so the effect can be isolated: `shipped` (raw crystal frame, the
old behaviour), `fixed` (its own independent Kabsch implementation) and `library`
(the library's superposition). The scorer, the weights and the ranking formula
are the same in all three, so any difference is attributable to the superposition
alone.

## Reproduce

Needs MASTER and the dataset in place (see the top-level README) and the case
structures in `database_builder/rcsb_cache/`, which `prep_case.py` downloads from RCSB.
If that download fails with `CERTIFICATE_VERIFY_FAILED`, set
`SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt` (or your system's CA bundle).

**Put the dataset on fast local disk** (or point `EPIFITSCOUT_DB` at a copy
there): each search takes about 2 s that way and about 2 minutes from a slow
network filesystem.

```bash
cd benchmark

# one case, as the tool ships
python run_case.py ../database_builder/rcsb_cache/2ny7.pdb H 93 102 G 363 371 2NY7

# all five
RC=../database_builder/rcsb_cache
while read id hc lo hi ac elo ehi; do
  python run_case.py $RC/$(echo $id | tr 'A-Z' 'a-z').pdb $hc $lo $hi $ac $elo $ehi $id
done < cases.txt > results/results.jsonl

# three-way rescoring (skip 1MEL, which returns no hits)
python rescore.py $RC/2ny7.pdb H 93 102 G 363 371 2NY7

# re-derive a case definition from scratch (downloads from RCSB if not cached)
python prep_case.py 2NY7 H G
```

`rescore.py` writes working files to `benchmark/work/`, which is git-ignored.

## Results (after the fix, 2026-10-13)

The *native* is the query's own chain, retrieved from the database at RMSD 0.
Because it is already in the query's frame, the fix cannot change its score;
what changes is how every *other* hit is scored.

| Case | Antibody / target | H3 | Hits | Native shape | Native rank (final) | Best other hit, shape | Native shape percentile |
|---|---|---|---|---|---|---|---|
| 1A2Y | HyHEL-63 / lysozyme | 10 aa | 500 | 0.520 | 101 → **1** | 0.891 → **0.712** | 45% → 21% |
| 3GBN | CR6261 / influenza HA stem | 14 aa | 500 | 0.557 | 1 → **1** | 0.834 → **0.708** | 70% → 34% |
| 3NGB | VRC01 / HIV-1 gp120 | 14 aa | 314 | 0.645 | 1 → **1** | 0.806 → **0.787** | 91% → 59% |
| 2NY7 | b12 / HIV-1 gp120 | 20 aa | 23 | 0.330 | 6 → **1** | 0.783 → **0.351** | 14% → 86%* |
| 1MEL | cAb-Lys3 nanobody / lysozyme | 26 aa | 0 | — | — | — | — |

Each "→" is *as shipped in v0.1.0* → *after the fix*. \*2NY7 has only 23 hits
and, after the fix, they all score 0.32 ± 0.01, so that percentile is separating
differences of about 0.004 and means little.

### What the fix did

1. **Scores no longer depend on where the crystal sat.** The same loop scored
   differently in different crystals:

   | Same loop, several crystals | v0.1.0 | after the fix |
   |---|---|---|
   | 2NY7:H, 1HZH:K, 8SXI:G (one antibody, H3 within 0.46 Å) | 0.330 / 0.783 / 0.724 | 0.330 / 0.326 / 0.327 |
   | 3NGB:B, J, H, E (four copies in one crystal, within 0.16 Å) | 0.645 / 0.275 / 0.585 / 0.480 | 0.645 / 0.647 / 0.647 / 0.648 |

2. **The best-looking non-native hits were artifacts.** The best other hit's shape
   score falls in every case, by up to 0.43 (2NY7). The apparent "better than the
   native" hits in the October run were largely produced by crystal placement.

3. **The score distribution compresses.** Standard deviation of the shape score
   over non-native hits goes from 0.21 / 0.15 / 0.12 / 0.16 to
   0.067 / 0.042 / 0.044 / 0.013. Scores that were being spread by frame noise
   now sit in a narrow band.

4. **The library and an independent implementation agree** to four decimals on
   all 1,337 hits, and the superposed Cα RMSD reproduces MASTER's reported RMSD
   (within the rounding of the printed values). No hit was dropped by the
   consistency check.

### What it did not establish

**The depth metric does not single out the native.** After the fix the native
still scores *below the best other-structure hit on shape alone in all four
cases* (0.520 vs 0.712; 0.557 vs 0.708; 0.645 vs 0.787; 0.330 vs 0.351), and sits
at the 21st and 34th percentile in 1A2Y and 3GBN, below the median. It reaches
rank 1 through the RMSD term, since it is at RMSD 0 by construction. Retrieval
quality is therefore not evidence that the shape score measures complementarity.

**Retrieval coverage still collapses with H3 length.** At 26 residues (1MEL)
nothing is found within 2.0 Å, and at 20 residues (2NY7) only 23 hits.

**Native-vs-decoy on five cases is not a validation.** None of this relates the
score to a measured binding outcome. The weights remain unvalidated
(`REPORT.md` §11).
