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

### What it is not

| | input complex | designs (mean of 5) | |
|---|---|---|---|
| buried area (Å²) | 1784 | 1752 | 0.7 s.e. |
| contacts | 96 | 155 | 1.4 s.e. |
| **clashes** | **0.0** | **15.5** | **2.1 s.e. against** |
| H-bonds | 5.5 | 4.6 | 0.6 s.e. |
| salt bridges | 0.8 | 0.1 | 1.8 s.e. against |

**The designs do not beat the input complex.** It is clash-free on all 17
targets and no design mean matches it. Read nothing here as "better
antibodies".

**The contact gain is not a binding gain.** Contacts and clashes are the same
geometry at two cutoffs and correlate at r = 0.98, so a loop driven into the
antigen scores more of both. Contact at *zero* clashes — the regression
intercept — is 67 against the input complexes' 96.

**Asked target by target**, does any clash-free design beat its own input on
contacts? Three of seventeen, and all three are targets whose native interface
was unusually small (9, 26 and 56 contacts). Where the native buries a real
interface — 88 to 226 contacts — the designs do not reach it.

Nothing here is a claim about affinity. These are geometric measurements on
static structures.

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
