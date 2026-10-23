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
```

Run them in order:

```bash
pip install -e ".[examples]"

python -I 01_select_complexes.py complexes.txt cache/ candidates.tsv
python -I 02_design_and_measure.py selected.tsv cache/ results --pool 60 --top 5
python -I 03_compare.py results/designs.tsv
python -I 04_figures.py results/designs.tsv figures
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

**This is the table that matters, and it is the unflattering one.** Remove the
clashing models and every term moves against the designs: the loop buries about
half the native's area, makes a third of its contacts, and makes essentially no
hydrogen bonds — 1.2 against 5.7. The gap index more than doubles and closest
approach rises *above* the native's 2.71 Å.

That is the mechanism stated plainly: **the sterically valid designs are valid
because they stand off the epitope.** A binder wants to be at ~2.7 Å and in
contact; these sit further out and touch less. Clash-free is necessary, not
sufficient.

Two things to keep in mind reading the second table. It is a **selected subset**,
so it estimates nothing in an unbiased way — selecting on clashes = 0
preferentially keeps loops that stand off, which is the bias the numbers show.
And it covers **13 targets, not 17**, so the input column shifts slightly as the
four targets with no clean design drop out.

**What follows from it.** The grafted loops are usable starting points and not
interfaces: a refinement stage that can drive a loop into contact — rigid-body
and backbone, which side-chain relaxation alone cannot do — is required, not
optional.

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
