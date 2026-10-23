"""Compare the five designs per complex against the input complex.

The unit of analysis is the complex, not the model. Each complex contributes one
number per metric -- the mean over its five designs -- and that is paired against
the same metric measured on the input complex at the same span. Pairing matters
because interface size varies several-fold between complexes, so an unpaired
comparison would be dominated by which complexes happened to produce models.

The reported s.e. is of the paired difference across complexes. With 17 complexes
it is a small-sample quantity, so the table prints n alongside it and the
per-complex rows are written out in full rather than summarised away.

Usage:
    python -I compare.py <designs.tsv> [--out summary.tsv]
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
from pathlib import Path

#: ``+1`` where a larger value is the better interface, ``-1`` where smaller is.
#: ``min_distance`` is the closest approach, so more is better only up to the
#: point where the loop stops touching the antigen -- it is reported, not judged.
DIRECTION = {
    "buried_area": +1,
    "loop_buried_area": +1,
    "contacts": +1,
    "contact_residues": +1,
    "clashes": -1,
    "hbonds": +1,
    "salt_bridges": +1,
    "gap_volume": -1,
    "gap_index": -1,
    "min_distance": 0,
}

LABEL = {
    "buried_area": "buried area (A^2)",
    "loop_buried_area": "loop buried area (A^2)",
    "contacts": "contacts",
    "contact_residues": "contact residues",
    "clashes": "clashes",
    "hbonds": "H-bonds",
    "salt_bridges": "salt bridges",
    "gap_volume": "gap volume (A^3)",
    "gap_index": "gap index (A)",
    "min_distance": "closest approach (A)",
}


def load(path: Path) -> dict[str, dict]:
    per: dict[str, dict] = {}
    for r in csv.DictReader(path.open(), delimiter="\t"):
        cid = r["complex_id"]
        slot = per.setdefault(cid, {"designs": [], "h3_len": r["h3_len"]})
        if r["arm"] == "design":
            slot["designs"].append(r)
        else:
            slot[r["arm"]] = r
    return per


def fnum(row: dict | None, key: str) -> float | None:
    """Parse one cell, treating blanks and non-finite values as missing.

    ``gap_index`` is a ratio and comes back as inf when a model buries no loop
    area at all, which would otherwise poison every aggregate it enters.
    """
    if row is None:
        return None
    try:
        v = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def diversity(per: dict, selected: Path | None) -> None:
    """What the tool returns, as distinct from how well it scores.

    A ranked list of near-identical loops and a ranked list of diverse ones
    look the same in an interface table, and they are not the same output.
    Identity to the native loop also checks the leakage controls from the other
    side: if exclusion were failing, these numbers would run high.
    """
    designs = [d for s in per.values() for d in s["designs"]]
    donors = {d["donor"].split(":")[0] for d in designs}
    clean = [d for d in designs if (v := fnum(d, "clashes")) is not None and v == 0]
    print(f"\nWhat comes out ({len(designs)} designs over {len(per)} targets):")
    print(f"  distinct donor chains      {len(donors)}/{len(designs)}")
    print(f"  clash-free designs         {len(clean)}, from "
          f"{len({d['donor'].split(':')[0] for d in clean})} distinct donors")
    print(f"  targets with a clean one   "
          f"{sum(1 for s in per.values() if any(fnum(d, 'clashes') == 0 for d in s['designs']))}"
          f"/{len(per)}")

    if selected is None or not selected.exists():
        return
    native = {r["complex_id"]: r["h3_seq"]
              for r in csv.DictReader(selected.open(), delimiter="\t")}
    ids = []
    for cid, s in per.items():
        nat = native.get(cid)
        for d in s["designs"]:
            seq = d.get("h3_seq", "")
            if nat and seq and len(seq) == len(nat):
                ids.append(sum(a == b for a, b in zip(nat, seq)) / len(nat))
    if ids:
        print(f"  identity to native loop    {100 * st.mean(ids):.0f}% mean, "
              f"{100 * st.median(ids):.0f}% median, {100 * max(ids):.0f}% max")
        print("    (high values here would mean the leakage controls are leaking)")


def beats_its_own_input(per: dict, reference: str) -> None:
    """Does any clash-free design beat the complex it replaces?

    The paired means answer "on average"; this answers "ever", which is the
    question that matters when the output is a shortlist to refine rather than
    a single prediction.
    """
    wins_ct = wins_bsa = have = 0
    for s in per.values():
        ref = s.get(reference)
        clean = [d for d in s["designs"] if fnum(d, "clashes") == 0]
        if ref is None or not clean:
            continue
        have += 1
        ic, ib = fnum(ref, "contacts"), fnum(ref, "buried_area")
        bc = max((v for d in clean if (v := fnum(d, "contacts")) is not None),
                 default=None)
        bb = max((v for d in clean if (v := fnum(d, "buried_area")) is not None),
                 default=None)
        wins_ct += bool(ic is not None and bc is not None and bc > ic)
        wins_bsa += bool(ib is not None and bb is not None and bb > ib)
    print("\nTargets where a clash-free design beats its own input complex:")
    print(f"  on contacts     {wins_ct}/{have} targets that have a clean design")
    print(f"  on buried area  {wins_bsa}/{have}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("designs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selected", default="selected.tsv",
                    help="where the native CDR-H3 sequences are recorded")
    ap.add_argument("--reference", default="input_crystal",
                    choices=["input_crystal", "input_repacked"])
    args = ap.parse_args()

    per = load(Path(args.designs))
    usable = {c: s for c, s in per.items() if s["designs"] and args.reference in s}
    print(f"complexes: {len(per)} total, {len(usable)} with designs and a reference")
    dropped = sorted(set(per) - set(usable))
    for c in dropped:
        why = "no designs returned" if not per[c]["designs"] else "no reference row"
        print(f"  excluded {c}: {why}")

    # ---- the packing control -------------------------------------------
    both = [c for c, s in usable.items()
            if "input_crystal" in s and "input_repacked" in s]
    if both:
        print(f"\nPacking control, repacked minus crystal input ({len(both)} complexes):")
        for m in ("buried_area", "contacts", "clashes", "hbonds"):
            d = [rp - cr for rp, cr in (
                    (fnum(usable[c]["input_repacked"], m),
                     fnum(usable[c]["input_crystal"], m)) for c in both)
                 if rp is not None and cr is not None]
            if not d:
                continue
            print(f"  {LABEL[m]:<24} {st.mean(d):+8.2f}"
                  f"  (s.e. {st.stdev(d) / len(d) ** 0.5:.2f})" if len(d) > 1 else "")

    # ---- the comparison -------------------------------------------------
    print(f"\nDesigns (mean of five) minus input complex [{args.reference}], "
          f"paired by complex, n = {len(usable)}:\n")
    hdr = (f"{'metric':<24}{'input':>10}{'designs':>10}{'diff':>10}"
           f"{'s.e.':>8}{'s.e. units':>11}  better")
    print(hdr)
    print("-" * len(hdr))

    summary = []
    for m, sign in DIRECTION.items():
        pairs = []
        for c, s in usable.items():
            ref = fnum(s[args.reference], m)
            vals = [v for v in (fnum(d, m) for d in s["designs"]) if v is not None]
            if ref is None or not vals:
                continue
            pairs.append((c, ref, st.mean(vals)))
        if len(pairs) < 2:
            continue
        diffs = [d - r for _, r, d in pairs]
        mu = st.mean(diffs)
        se = st.stdev(diffs) / len(diffs) ** 0.5
        units = abs(mu / se) if se else float("nan")
        better = sum(1 for x in diffs if x * sign > 0) if sign else None
        verdict = "n/a" if sign == 0 else f"{better}/{len(diffs)}"
        print(f"{LABEL[m]:<24}{st.mean([r for _, r, _ in pairs]):>10.2f}"
              f"{st.mean([d for _, _, d in pairs]):>10.2f}{mu:>+10.2f}{se:>8.2f}"
              f"{units:>11.1f}  {verdict}")
        summary.append({
            "metric": m, "n": len(diffs),
            "input_mean": st.mean([r for _, r, _ in pairs]),
            "design_mean": st.mean([d for _, _, d in pairs]),
            "diff": mu, "se": se, "se_units": units,
            "better_direction": sign, "n_better": better if better is not None else "",
        })

    diversity(usable, Path(args.selected))
    beats_its_own_input(usable, args.reference)

    print("\n'better' counts complexes where the five-design mean beats the input "
          "in the metric's own direction.")
    print("Lower is better for clashes, gap volume and gap index; closest "
          "approach is reported without a direction.")

    if args.out:
        with Path(args.out).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]), delimiter="\t")
            w.writeheader()
            w.writerows(summary)
        print(f"\nwrote {args.out}")

    # ---- per-complex detail ---------------------------------------------
    print("\nPer complex (input -> design mean), n designs in brackets:\n")
    print(f"{'complex':<16}{'H3':>4}{'k':>3}  {'buried area':>22}{'contacts':>16}{'clashes':>14}")
    for c in sorted(usable, key=lambda x: int(usable[x]["h3_len"])):
        s = usable[c]
        ref = s[args.reference]
        ds = s["designs"]
        def cell(m: str, width: int = 0, ref: dict = ref, ds: list = ds) -> str:
            r = fnum(ref, m)
            vals = [v for v in (fnum(x, m) for x in ds) if v is not None]
            if r is None or not vals:
                return "-".rjust(width)
            return f"{r:.0f} -> {st.mean(vals):.0f}".rjust(width)
        print(f"{c:<16}{s['h3_len']:>4}{len(ds):>3}  {cell('buried_area', 22)}"
              f"{cell('contacts', 16)}{cell('clashes', 14)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
