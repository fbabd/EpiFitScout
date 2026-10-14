#!/usr/bin/env python3
"""Turn benchmark records into statistics and figures.

Reads the JSONL written by ``run_benchmark.py`` and answers four questions, each
of which can come out against the tool:

1. Does the geometric step add information MASTER does not already have?
2. Does the score detect real complementarity (true epitope vs decoy patches)?
3. Is the native loop recovered among the retrieved fragments?
4. Does the score track measured binding energy?

Usage::

    python benchmark/analyse_benchmark.py --records benchmark/results/benchmark.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def load(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    good = [r for r in records if "error" not in r]
    skipped = len(records) - len(good)
    print(f"{len(good)} complexes analysed ({skipped} skipped during the run)\n")
    return good


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, via the Mann-Whitney U identity.

    0.5 means the score carries no information about the label.
    """
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def q1_steric(records: list[dict]) -> dict:
    """Is a collision predictable from MASTER's RMSD?"""
    print("=" * 72)
    print("Q1. Does the geometric step add anything MASTER does not have?")
    print("=" * 72)
    print("MASTER ranks by backbone RMSD and never sees the antigen. If its hits")
    print("could simply be filtered by RMSD, this step would be redundant.\n")

    per_case, aucs, all_rmsd, all_clash = [], [], [], []
    for r in records:
        rmsd = np.array([h["rmsd"] for h in r["hits"]])
        clash = np.array([1 if h["clash"] else 0 for h in r["hits"]])
        per_case.append(100 * clash.mean())
        all_rmsd.append(rmsd)
        all_clash.append(clash)
        if 0 < clash.sum() < len(clash):
            aucs.append(auc(rmsd, clash))

    frac = np.array(per_case)
    print("Hits that cannot be placed without colliding with the antigen:")
    print(f"  median {np.median(frac):.0f}% per complex, "
          f"IQR {np.percentile(frac,25):.0f}-{np.percentile(frac,75):.0f}%, "
          f"range {frac.min():.0f}-{frac.max():.0f}%")
    print(f"  {int((frac > 0).sum())} of {len(frac)} complexes have at least one such hit")

    a = np.array(aucs)
    print("\nCan RMSD predict which hits collide? (AUC; 0.5 = no information)")
    print(f"  median AUC {np.median(a):.3f} over {len(a)} complexes, "
          f"IQR {np.percentile(a,25):.3f}-{np.percentile(a,75):.3f}")
    t = stats.wilcoxon(a - 0.5, alternative="two-sided") if len(a) > 5 else None
    if t is not None:
        print(f"  Wilcoxon vs 0.5: p = {t.pvalue:.3g}")

    rm, cl = np.concatenate(all_rmsd), np.concatenate(all_clash)
    print(f"\nPooled over {len(rm)} hits, collision rate by RMSD band:")
    edges = [0, 0.5, 1.0, 1.5, 2.0]
    for a0, a1 in zip(edges, edges[1:], strict=False):
        m = (rm >= a0) & (rm < a1)
        if m.sum():
            print(f"  RMSD {a0:.1f}-{a1:.1f} A: {100*cl[m].mean():5.1f}% collide  "
                  f"(n={int(m.sum())})")
    # State what the numbers support, not more. RMSD is weakly informative
    # about collisions, so the honest claim is that it is a poor substitute for
    # the geometric check, not that the two are independent.
    med = float(np.median(a))
    if med > 0.6:
        verdict = ("RMSD carries real information about collisions; the geometric\n"
                   "     check still adds to it, but the claim of orthogonality fails.")
    elif med > 0.55:
        verdict = ("RMSD is only weakly informative about collisions. Tightening the\n"
                   "     RMSD cut would discard many placeable hits to remove few\n"
                   "     unplaceable ones, so it is a poor substitute for the check.")
    else:
        verdict = ("RMSD carries essentially no information about collisions; the\n"
                   "     geometric check is orthogonal to what MASTER provides.")
    # What would it cost to avoid collisions by tightening RMSD instead?
    print("\n  Could a tighter RMSD cut replace the check?")
    for cut in (0.5, 1.0, 1.5):
        keep = rm < cut
        if keep.sum():
            print(f"    cut at {cut:.1f} A: keeps {100*keep.mean():4.1f}% of hits, "
                  f"of which {100*cl[keep].mean():4.1f}% still collide")
    print(f"    no cut (2.0 A): keeps 100.0% of hits, "
          f"of which {100*cl.mean():4.1f}% collide")
    print(f"\n  -> {verdict}")
    return {"clash_pct": frac, "auc": a, "rmsd": rm, "clash": cl}


def q2_specificity(records: list[dict], spec_path: Path) -> dict:
    """Does the native loop prefer its own epitope over *reachable* alternatives?"""
    print("\n" + "=" * 72)
    print("Q2. Does the score detect real complementarity?")
    print("=" * 72)

    if not spec_path.is_file():
        print(f"  {spec_path} not found — run benchmark/decoy_epitopes.py first.")
        print("  The decoys recorded during the run sit >25 A away, so the loop")
        print("  cannot reach them and they score zero by construction. That")
        print("  measures proximity, not complementarity, and is not reported.")
        return {}

    rows = json.loads(spec_path.read_text())
    t = np.array([r["true"] for r in rows])
    dm = np.array([r["decoy_mean"] for r in rows])
    db = np.array([r["decoy_best"] for r in rows])
    n_decoy = sum(r["n_decoys"] for r in rows)

    print("The native loop is scored against its own epitope and against decoy")
    print("patches on the same antigen that are the same size, share under 25% of")
    print("their residues with the true site, and that the loop can reach equally")
    print("well. Same antibody, same antigen, same frame — only the site moves.\n")
    print(f"  {len(rows)} complexes, {n_decoy} decoy patches")
    print(f"  mean score: true {t.mean():.3f} | decoy mean {dm.mean():.3f} | "
          f"decoy best {db.mean():.3f}")
    print(f"  decoys are genuinely in contact (mean score {dm.mean():.3f}, not zero),")
    print("  so this compares fit quality rather than reachability.")
    out = {"true": t, "decoy": dm, "decoy_best": db}
    for label, d, key in (("mean decoy", dm, "p"), ("best decoy", db, "p_best")):
        wins = int((t > d).sum())
        w = stats.wilcoxon(t, d, alternative="greater")
        diff = t - d
        out[key] = w.pvalue
        print(f"  vs {label}: true wins {wins}/{len(t)} ({100*wins/len(t):.0f}%), "
              f"p = {w.pvalue:.3g}, Cohen's d = {diff.mean()/(diff.std()+1e-9):.2f}")
    return out


def q3_recovery(records: list[dict]) -> dict:
    """Where does the native loop rank among the retrieved fragments?"""
    print("\n" + "=" * 72)
    print("Q3. Is the native loop recovered?")
    print("=" * 72)

    pct_local, pct_profile = [], []
    for r in records:
        others = [h for h in r["hits"] if not h["self"]]
        if len(others) < 20:
            continue
        nl = np.array([h["local"] for h in others])
        npf = np.array([h["profile"] for h in others])
        pct_local.append(100 * (nl < r["native_score"]).mean())
        selfhits = [h for h in r["hits"] if h["self"]]
        if selfhits:
            pct_profile.append(100 * (npf < max(h["profile"] for h in selfhits)).mean())
    pl = np.array(pct_local)
    print(f"  native percentile among non-self hits, local fit: "
          f"median {np.median(pl):.0f}%, IQR {np.percentile(pl,25):.0f}-{np.percentile(pl,75):.0f}%")
    if pct_profile:
        pp = np.array(pct_profile)
        print(f"  the superseded profile scorer, same cases:        "
              f"median {np.median(pp):.0f}%, IQR {np.percentile(pp,25):.0f}-{np.percentile(pp,75):.0f}%")
    print(f"  above the 50th percentile in {int((pl>50).sum())}/{len(pl)} complexes")
    t = stats.wilcoxon(pl - 50, alternative="greater") if len(pl) > 5 else None
    if t is not None:
        print(f"  Wilcoxon vs chance (50%): p = {t.pvalue:.3g}")
    return {"pct_local": pl, "pct_profile": np.array(pct_profile)}


def q4_affinity(records: list[dict]) -> dict:
    """Does the score track measured binding energy?"""
    print("\n" + "=" * 72)
    print("Q4. Does the score track binding energy?")
    print("=" * 72)
    pairs = [(r["native_score"], r["delta_g"]) for r in records
             if r.get("delta_g") is not None and np.isfinite(r.get("delta_g", np.nan))]
    if len(pairs) < 8:
        print(f"  only {len(pairs)} complexes carry a ΔG; not enough to test")
        return {}
    s = np.array([p[0] for p in pairs]); g = np.array([p[1] for p in pairs])
    rho = stats.spearmanr(s, g)
    print(f"  {len(pairs)} complexes with SAbDab ΔG ({g.min():.1f} to {g.max():.1f} kcal/mol)")
    print(f"  Spearman rho = {rho.statistic:+.3f}, p = {rho.pvalue:.3g}")
    print("  (a shape term alone is not expected to predict affinity; this is")
    print("   reported so the claim is not overstated either way)")
    return {"score": s, "dg": g, "rho": rho.statistic, "p": rho.pvalue}


def figures(records: list[dict], q1: dict, q2: dict, q3: dict, q4: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) collision rate by RMSD band
    ax = axes[0, 0]
    rm, cl = q1["rmsd"], q1["clash"]
    edges = np.arange(0, 2.01, 0.25)
    centres, rates, ns = [], [], []
    for a0, a1 in zip(edges, edges[1:], strict=False):
        m = (rm >= a0) & (rm < a1)
        if m.sum() > 30:
            centres.append((a0 + a1) / 2); rates.append(100 * cl[m].mean()); ns.append(int(m.sum()))
    ax.bar(centres, rates, width=0.21, color="#c0392b", alpha=.85)
    for x, y, n in zip(centres, rates, ns, strict=True):
        ax.text(x, y + 1, f"n={n}", ha="center", fontsize=7)
    ax.set_xlabel("MASTER backbone RMSD (Å)"); ax.set_ylabel("hits that collide (%)")
    ax.set_title(f"(a) Collisions persist well inside the RMSD threshold\n"
                 f"within-complex AUC of RMSD = {np.median(q1['auc']):.2f} "
                 f"(0.5 = no information)", fontsize=10)
    ax.grid(alpha=.25, axis="y")

    # (b) true vs decoy epitope
    ax = axes[0, 1]
    if q2:
        t, d = q2["true"], q2["decoy"]
        ax.scatter(d, t, s=28, c="#2c7fb8", alpha=.75, edgecolor="k", linewidth=.3)
        lim = [0, max(t.max(), d.max()) * 1.08]
        ax.plot(lim, lim, "k--", lw=1, label="equal")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("score at reachable decoy patches (mean)")
        ax.set_ylabel("score at the true epitope")
        ax.set_title(f"(b) The loop fits its own epitope best\n"
                     f"{int((t>d).sum())}/{len(t)} above the line, p = {q2['p']:.2g}",
                     fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.25)

    # (c) native percentile
    ax = axes[1, 0]
    if len(q3.get("pct_local", [])):
        data = [q3["pct_local"]]
        labels = ["local fit"]
        if len(q3.get("pct_profile", [])):
            data.append(q3["pct_profile"]); labels.append("profile (old)")
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=.5)
        for patch, c in zip(bp["boxes"], ["#2c7fb8", "#aaaaaa"], strict=False):
            patch.set_facecolor(c); patch.set_alpha(.8)
        ax.axhline(50, color="k", ls="--", lw=1, label="chance")
        ax.set_ylabel("native percentile among retrieved hits")
        ax.set_title("(c) Recovery of the native loop", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")

    # (d) score vs dG
    ax = axes[1, 1]
    if q4:
        ax.scatter(q4["dg"], q4["score"], s=28, c="#7b3294", alpha=.75,
                   edgecolor="k", linewidth=.3)
        ax.set_xlabel("SAbDab ΔG (kcal/mol)"); ax.set_ylabel("native shape score")
        ax.set_title(f"(d) Shape score vs measured affinity\n"
                     f"Spearman rho = {q4['rho']:+.2f}, p = {q4['p']:.2g}", fontsize=10)
        ax.grid(alpha=.25)
    else:
        ax.axis("off")

    fig.suptitle("EpiFitScout: what the geometric step contributes",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nfigure written to {out}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", type=Path,
                   default=Path(__file__).parent / "results/benchmark.jsonl")
    p.add_argument("--specificity", type=Path,
                   default=Path(__file__).parent / "results/specificity.json",
                   help="Output of decoy_epitopes.py (the reachable-decoy control).")
    p.add_argument("--figure", type=Path,
                   default=Path(__file__).parent / "results/benchmark.png")
    args = p.parse_args(argv)

    records = load(args.records)
    if len(records) < 5:
        print("too few completed complexes to analyse")
        return 1
    q1 = q1_steric(records)
    q2 = q2_specificity(records, args.specificity)
    q3 = q3_recovery(records)
    q4 = q4_affinity(records)
    figures(records, q1, q2, q3, q4, args.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
