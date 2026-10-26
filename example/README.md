# Worked example: designing CDR-H3 loops and measuring what you get

EpiFitScout proper ends at stage 8 — it hands back fragments, ranked. This
directory is the other half of the question: take those fragments, build real
all-atom models from them, and measure them against the complex they were
grafted into.

The comparison is against **the input bound complex itself**. It needs no second
method and no matched budget: the loop was already there and it already worked,
so its interface is the one reference that cannot be argued with.

## What it does

```
01_select_complexes.py    pick targets and locate CDR-H3 by its conserved
                          Cys / WGxG anchors, not by deposited numbering
02_design_and_measure.py  five designs per target (stages 1-8), then build
                          and measure them (stages 9-11), plus the input
                          complex scored by the identical scorer
03_compare.py             paired comparison, the complex as the unit
04_figures.py             the two figures
05_per_target.py          one row per target, the unit that avoids skew
```

Run them in order:

```bash
pip install -e ".[examples]"

python -I 01_select_complexes.py complexes.txt cache/ candidates.tsv
python -I 02_design_and_measure.py selected.tsv cache/ results --pool 60 --top 5
python -I 03_compare.py results/designs.tsv
python -I 04_figures.py results/designs.tsv figures
python -I 05_per_target.py results/designs.tsv
```

`selected.tsv` and `results/designs.tsv` are the shipped outputs of the run
described below, so steps 3 and 4 reproduce the numbers without re-running the
search.

## The result

17 complexes, CDR-H3 lengths 5–21, five designs each, every target's own PDB
entry excluded from the search and its native loop sequence refused.

### What comes out

| | |
|---|---|
| designs, from distinct donor chains | 85 / 85 |
| mean sequence identity to the native loop | 21% (median 19%) |
| clash-free designs | 30, from 30 distinct donors |
| targets with at least one clash-free design | 13 / 17 |

The loops are novel — a fifth identical to the native on average — and they are
real crystallographic conformations rather than generated ones. Most targets
yield several that are sterically valid. That is the useful output: a diverse,
checked set of starting points to refine or to seed a generative stage.

### Measured against the complex they replace

Mean ± SD. "Better" throughout means better **for the antibody** — a binder
wants to bury more surface, make more contacts, more hydrogen bonds and more
salt bridges, pack tightly against the epitope (low gap index) and not
interpenetrate it (no clashes).

**All 85 designs, 17 targets:**

| | input complex | designs | better for a binder |
|---|---|---|---|
| buried area (Å²) | 1784 ± 266 | 1752 ± 344 | higher |
| loop buried area (Å²) | 287 ± 136 | 270 ± 166 | higher |
| contacts | 96 ± 48 | 155 ± 225 | higher |
| contact residues | 6.1 ± 2.2 | 5.7 ± 3.2 | higher |
| **clashes** | **0.0 ± 0.0** | **15.5 ± 38.5** | **lower** |
| H-bonds | 5.5 ± 4.6 | 4.6 ± 6.6 | higher |
| salt bridges | 0.8 ± 1.7 | 0.1 ± 0.3 | higher |
| gap index (Å) | 4.1 ± 2.1 | 6.9 ± 17.8 | lower |
| closest approach (Å) | 2.69 ± 0.29 | 2.29 ± 0.81 | ≈ 2.7, not lower |

The contact column reads as a win and is not one. Contacts and clashes are the
same geometry at two cutoffs (r = 0.98), so the 155 is mostly atoms *inside* the
antigen. The clash row and the closest approach row say the same thing: 2.29 Å
is below what any real interface does.

**The 30 clash-free designs, 13 targets** — the sterically valid subset:

| | input complex | clash-free designs | better for a binder |
|---|---|---|---|
| buried area (Å²) | 1765 ± 271 | 1634 ± 272 | higher |
| loop buried area (Å²) | 266 ± 145 | **141 ± 77** | higher |
| contacts | 91 ± 53 | **34 ± 28** | higher |
| contact residues | 5.7 ± 2.2 | 3.3 ± 2.1 | higher |
| clashes | 0.0 ± 0.0 | 0.0 ± 0.0 | lower |
| H-bonds | 5.7 ± 5.2 | **1.2 ± 1.4** | higher |
| salt bridges | 0.9 ± 1.9 | 0.0 ± 0.2 | higher |
| gap index (Å) | 4.4 ± 2.3 | 12.7 ± 29.1 | lower |
| closest approach (Å) | 2.71 ± 0.31 | 3.04 ± 0.63 | ≈ 2.7, not higher |

Removing the clashing models moves every term against the designs: the loop
buries about half the native's area, makes a third of its contacts, and makes
essentially no hydrogen bonds — 1.2 against 5.7.

**But read the per-target table below before concluding from this one.** These
pooled means weight targets by how many clash-free designs they happened to
produce, and one degenerate target supplied 5 of the 30. Per target, the *best*
clash-free design reaches a median 83% of its own native loop's burial — a very
different picture from "141 against 266". What survives both views is the
hydrogen-bond deficit, which is real on almost every row.

Two things to keep in mind reading the second table. It is a **selected subset**,
so it estimates nothing in an unbiased way — selecting on clashes = 0
preferentially keeps loops that stand off, which is the bias the numbers show.
And it covers **13 targets, not 17**, so the input column shifts slightly as the
four targets with no clean design drop out.

**What follows from it.** The grafted loops are usable starting points and not
interfaces: a refinement stage that can drive a loop into contact — rigid-body
and backbone, which side-chain relaxation alone cannot do — is required, not
optional.

### Per target, which is the honest unit

Pooling all 85 designs into one mean is misleading, and measurably so: `7u9p`
has a 5-residue CDR whose native loop buries 38 Å², so clash-free is trivial
there and it supplied **5 of the 30** clash-free designs, degenerate ones
included. A pooled mean is then partly a measure of which targets happened to
produce many easy models. One row per target fixes that — and a bad target shows
up as a row instead of disappearing into an average.

`05_per_target.py` prints this. Input → best clash-free design:

```
complex         H3 clean   input→loop Å²  input→contacts  input→H-bond  input→gapidx   input→minD   %nat
7u9p_H_L_A       5  5/5       38→62           9→28           2→1        11.8→8.4      2.66→2.93    165%
7uow_H_L_A      11  2/5      171→146         26→26           0→0         5.2→5.0      3.40→2.91     86%
7xst_D_E_A      11  3/5      264→232         56→67           4→5         3.6→3.9      2.43→2.88     88%
7wp8_E_G_F      12  1/5      246→107         56→9            3→0         3.2→7.7      2.74→3.63     44%
8jqm_M_N_B      12  3/5      141→128         71→33           7→1         4.8→5.3      2.90→2.89     91%
8awm_H_L_A      13  2/5      188→137         93→31           6→3         4.8→6.6      2.53→2.58     73%
8bw0_H_L_C      13  1/5      236→225         96→72           4→2         3.8→4.0      2.31→2.50     95%
8cwk_A_B_C      14  0/5             — no clash-free design —
7wn2_H_L_R      15  2/5      250→139         88→47           1→0         3.7→5.2      3.27→2.93     56%
8oxw_B_C_A      15  4/5      310→208        110→62           4→3         4.0→6.5      2.78→2.67     67%
7vng_H_L_D      16  4/5      220→183         93→56           4→1         4.0→4.9      2.60→2.76     83%
8jio_C_B_A      16  0/5             — no clash-free design —
8c3v_C_D_Y      17  1/5      282→284        115→45           8→1         3.4→3.7      2.45→2.87    101%
8hes_H_L_C      18  1/5      464→362        146→129         10→4         2.8→3.4      2.53→2.49     78%
7x95_H_L_A      19  0/5             — no clash-free design —
7yr0_C_B_A      20  0/5             — no clash-free design —
8g3p_H_L_E      21  1/5      647→124        226→18          21→1         1.7→10.0     2.62→2.81     19%

burial as a fraction of each target's own native loop:
  median 83%, mean 80%, best 165%   (13 of 17 targets)
    >=90%: 4 targets    75-90%: 4 targets    50-75%: 3 targets    <50%: 2 targets
```

**This is a better result than the pooled table suggests.** The best clash-free
design reaches a **median 83% of its own native loop's burial**, and on 8 of 13
targets it reaches 75% or more. `8c3v` matches the native outright (284 vs
282 Å²) and `8bw0` comes within 5%. The pooled "141 vs 266 Å²" understated this
because a fifth of the clash-free set came from one degenerate target.

It does not overturn the earlier conclusion, it sharpens it:

* **Burial is nearly recovered; polar contacts are not.** H-bonds drop on almost
  every row — 8g3p 21→1, 8jqm 7→1, 8c3v 8→1. The grafted loop occupies the site
  but does not re-form its chemistry, which is what you would expect when the
  donor sequence was complementary to a *different* epitope and nothing
  redesigns it.
* **Long CDRs fail hardest.** The four targets with no clash-free design at all
  are H3 lengths 14, 16, 19 and 20, and the worst row is the longest (8g3p, H3
  21, 19%). The stems fix the ends and nothing constrains the middle, so the
  unconstrained span grows with length.
* **`%nat` above 100% is not necessarily good.** `7u9p` reads 165% against a
  native loop that buries only 38 Å² — a tiny interface where almost anything
  buries more. Read that row as "the native barely engages here", not as a win.

Two other representatives are available, and both include clashing models whose
burial is inflated by interpenetration, so neither is like-for-like with the
input; the script says so when you ask for them:

| representative | median %nat | targets |
|---|---|---|
| rank-1 model | 71% | 17/17 (8 clash) |
| best clash-free | **83%** | 13/17 |
| mean of all five | 84% | 17/17 (most clash) |

```bash
python -I 05_per_target.py results/designs.tsv --representative every
```

## Two controls worth knowing about

**The packing control.** Designs are FASPR-packed while the input complex has
crystallographic side chains, so comparing them directly would confound the loop
with the packing. `02_design_and_measure.py` therefore scores the input twice —
as deposited, and FASPR-repacked over the same span. Repacking alone costs
+0.8 clashes, so packing explains about 3% of the gap rather than the gap.

**Leakage.** Excluding the target's own PDB entry is not enough: the same
antibody is often deposited under other accessions. Both controls are on
(`exclude_donors` and `exclude_native_loop`), and any benchmark built from
SAbDab needs both.

## Figures

`fig1_input_vs_designs` — paired slopes, one line per complex, input to design
mean. A paired form because the comparison is paired; a two-bar chart of means
would hide that complexes move in different directions.

`fig2_contacts_vs_clashes` — why the contact gain is not real. The designs that
out-contact the input are the ones passing through it.
