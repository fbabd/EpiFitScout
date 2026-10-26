"""One row per target: the input complex against what the tool returned.

Pooling every design into one mean is misleading here, and demonstrably so.
Targets contribute unequal numbers of clash-free designs -- one 5-residue CDR
whose native loop buries 38 Å² supplied 5 of 30, including the degenerate ones --
so a pooled mean is partly a measure of which targets happened to produce many
easy models. Per target, each contributes once, and a bad target is visible as a
row rather than hidden in an average.

Three representatives, because they answer different questions:

``rank1``
    the model the tool puts first. What a user gets if they take one.
``best-clean``
    the best sterically valid model it found. What a shortlist is worth, which
    is the relevant question when the output is candidates to refine.
``all``
    the mean over all five. What the pool is like on average.

Burial is also reported as a percentage of the target's *own* native loop,
because an absolute Å² is not comparable between a 5-residue CDR and a
21-residue one.

Usage:
    python -I 05_per_target.py <designs.tsv> [--representative rank1|best-clean|all|every]
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

METRICS = [
    ("loop_buried_area", "loop Å²", 0),
    ("contacts", "contacts", 0),
    ("hbonds", "H-bond", 0),
    ("gap_index", "gapidx", 1),
    ("min_distance", "minD", 2),
]


def fnum(row: dict | None, key: str) -> float | None:
    if row is None:
        return None
    try:
        v = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def load(path: Path) -> tuple[dict, dict]:
    designs: dict[str, list[dict]] = defaultdict(list)
    inputs: dict[str, dict] = {}
    for r in csv.DictReader(path.open(), delimiter="\t"):
        if r["arm"] == "design":
            designs[r["complex_id"]].append(r)
        elif r["arm"] == "input_crystal":
            inputs[r["complex_id"]] = r
    return designs, inputs


def representative(designs: list[dict], mode: str) -> tuple[dict | None, int]:
    """The design a row speaks for, and how many were clash-free."""
    clean = [d for d in designs if fnum(d, "clashes") == 0]
    if mode == "rank1":
        pick = next((d for d in designs if d["rank"] == "1"), None)
    elif mode == "best-clean":
        pick = max(clean, key=lambda d: fnum(d, "loop_buried_area") or 0.0) if clean else None
    else:
        pick = None          # "all" is summarised per metric, not picked
    return pick, len(clean)


def value(designs: list[dict], pick: dict | None, mode: str, key: str) -> float | None:
    if mode != "all":
        return fnum(pick, key)
    vals = [v for d in designs if (v := fnum(d, key)) is not None]
    return st.mean(vals) if vals else None


def table(designs: dict, inputs: dict, mode: str) -> None:
    label = {"rank1": "the rank-1 model",
             "best-clean": "the best clash-free model",
             "all": "the mean of all five"}[mode]
    print(f"\n{'=' * 86}\nInput complex vs {label}, one row per target\n{'=' * 86}")
    head = f"{'complex':<15}{'H3':>3}{'clean':>6}  "
    head += "".join(f"{'input→' + name:^17}" for _, name, _ in METRICS)
    head += f"{'%nat':>6}"
    print(head)
    print("-" * len(head))

    ratios: list[float] = []
    rows_shown = 0
    for cid in sorted(designs, key=lambda c: int(designs[c][0]["h3_len"] or 0)):
        ds = designs[cid]
        ref = inputs.get(cid)
        pick, n_clean = representative(ds, mode)
        if ref is None or (mode != "all" and pick is None):
            print(f"{cid:<15}{ds[0]['h3_len']:>3}{n_clean:>3}/{len(ds)}  "
                  f"{'— no clash-free design —':>35}")
            continue
        rows_shown += 1
        cells = ""
        for key, _, dp in METRICS:
            a, b = fnum(ref, key), value(ds, pick, mode, key)
            cells += (f"{a:>7.{dp}f}→{b:<9.{dp}f}" if a is not None and b is not None
                      else f"{'—':>17}")
        nat = fnum(ref, "loop_buried_area")
        got = value(ds, pick, mode, "loop_buried_area")
        pct = f"{100 * got / nat:.0f}%" if nat and got is not None and nat > 0 else "—"
        print(f"{cid:<15}{ds[0]['h3_len']:>3}{n_clean:>3}/{len(ds)}  {cells}  {pct:>5}")
        if nat and got is not None and nat > 0:
            ratios.append(got / nat)

    if not ratios:
        return
    print("-" * len(head))
    if mode != "best-clean":
        clashing = sum(
            1 for cid, ds in designs.items()
            for d in ([x for x in ds if x["rank"] == "1"] if mode == "rank1" else ds)
            if (v := fnum(d, "clashes")) is not None and v > 0
        )
        print(f"!  {clashing} of the models summarised here clash with the "
              "antigen. Their burial is")
        print("   inflated by interpenetration, so this row is not comparable "
              "with the input.")
    print(f"targets summarised: {rows_shown}/{len(designs)}")
    print(f"burial as a fraction of each target's own native loop: "
          f"median {100 * st.median(ratios):.0f}%, "
          f"mean {100 * st.mean(ratios):.0f}%, "
          f"best {100 * max(ratios):.0f}%")
    for lo, hi, name in ((0.9, 9e9, ">=90%"), (0.75, 0.9, "75-90%"),
                         (0.5, 0.75, "50-75%"), (0.0, 0.5, "<50%")):
        n = sum(1 for r in ratios if lo <= r < hi)
        if n:
            print(f"    {name:>7}: {n} targets")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("designs")
    ap.add_argument("--representative", default="every",
                    choices=["rank1", "best-clean", "all", "every"])
    args = ap.parse_args()

    designs, inputs = load(Path(args.designs))
    if not designs:
        print("no design rows found", file=sys.stderr)
        return 1

    modes = ["rank1", "best-clean", "all"] if args.representative == "every" \
        else [args.representative]
    for mode in modes:
        table(designs, inputs, mode)

    print("\nEach arrow reads input → design. Burial is the loop's own buried "
          "area, so it\nmeasures the graft rather than the whole antibody. "
          "'clean' counts clash-free\ndesigns out of five. The percentage is "
          "the design's burial as a fraction of that\ntarget's native loop, "
          "which is the only way a 5-residue CDR and a 21-residue one\ncan be "
          "put in the same column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
