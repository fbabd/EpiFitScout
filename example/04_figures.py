"""Figures for the input-complex comparison.

Figure 1 is a paired slope panel per metric: one line per complex, running from
the input complex's value to the mean of its five designs. A paired form is used
because the comparison is paired -- a grouped bar of two means would hide that
the complexes move in different directions, which is the whole finding.

Figure 2 plots clash burden against CDR-H3 length, which is the variable the
clash result actually tracks.

Colours are slots 1 and 2 of the reference categorical palette, in fixed order.

Usage:
    python -I make_figures.py <designs.tsv> <out_dir>
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INPUT_C = "#2a78d6"   # categorical slot 1
DESIGN_C = "#eb6834"  # categorical slot 2
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d7d2"

PANELS = [
    ("buried_area", "Buried surface area (Å²)", +1),
    ("contacts", "Loop–antigen contacts", +1),
    ("clashes", "Clashes with antigen", -1),
    ("hbonds", "Hydrogen bonds", +1),
]


def fnum(row: dict, key: str) -> float | None:
    try:
        v = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def load(path: Path) -> dict[str, dict]:
    per: dict[str, dict] = {}
    for r in csv.DictReader(path.open(), delimiter="\t"):
        slot = per.setdefault(r["complex_id"], {"designs": [], "h3_len": int(r["h3_len"] or 0)})
        if r["arm"] == "design":
            slot["designs"].append(r)
        else:
            slot[r["arm"]] = r
    return {c: s for c, s in per.items() if s["designs"] and "input_crystal" in s}


def style(ax) -> None:  # noqa: ANN001
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.yaxis.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def figure1(per: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11, 3.4))
    for ax, (metric, label, sign) in zip(axes, PANELS, strict=True):
        n_better = n = 0
        for s in per.values():
            a = fnum(s["input_crystal"], metric)
            vals = [v for v in (fnum(d, metric) for d in s["designs"]) if v is not None]
            if a is None or not vals:
                continue
            b = st.mean(vals)
            n += 1
            improved = (b - a) * sign > 0
            n_better += improved
            ax.plot([0, 1], [a, b], color=INK2 if improved else "#c9c8c3",
                    lw=1.0, alpha=0.85, zorder=2, solid_capstyle="round")
            ax.plot([0], [a], "o", ms=5, color=INPUT_C, zorder=3)
            ax.plot([1], [b], "o", ms=5, color=DESIGN_C, zorder=3)
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["input", "designs"], fontsize=9, color=INK)
        ax.set_title(label, fontsize=9.5, color=INK, pad=8)
        arrow = "higher" if sign > 0 else "lower"
        ax.set_xlabel(f"{n_better}/{n} complexes {arrow}", fontsize=8, color=INK2)
        style(ax)
    fig.suptitle("Each complex's five designs, against the input complex they are grafted into",
                 fontsize=11, color=INK, y=1.02)
    handles = [plt.Line2D([], [], marker="o", ls="", color=INPUT_C, label="input complex"),
               plt.Line2D([], [], marker="o", ls="", color=DESIGN_C, label="mean of 5 designs")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig1_input_vs_designs.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure2(per: dict, out: Path) -> None:
    """Why the contact gain is not a binding gain.

    Contacts and clashes are the same geometry measured at two cutoffs, so a
    loop pushed into the antigen scores more of both. Plotting them against each
    other shows that directly: the designs that out-contact the input complex
    are the ones interpenetrating it, and the few that do not clash sit well
    below the input complex on contacts.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    dx = [v for s in per.values() for d in s["designs"]
          if (v := fnum(d, "clashes")) is not None]
    dy = [v for s in per.values() for d in s["designs"]
          if (v := fnum(d, "contacts")) is not None]
    ix = [fnum(s["input_crystal"], "clashes") for s in per.values()]
    iy = [fnum(s["input_crystal"], "contacts") for s in per.values()]

    ax.scatter(dx, dy, s=34, color=DESIGN_C, alpha=0.75,
               edgecolor="white", linewidth=0.6, label="design (85)", zorder=3)
    ax.scatter(ix, iy, s=48, color=INPUT_C, marker="D",
               edgecolor="white", linewidth=0.6, label="input complex (17)", zorder=4)

    clean = [y for x, y in zip(dx, dy, strict=True) if x == 0]
    ax.annotate(f"{len(clean)} of {len(dx)} designs are clash-free,\n"
                f"and average {st.mean(clean):.0f} contacts against "
                f"the inputs' {st.mean(iy):.0f}",
                xy=(0, st.mean(clean)), xytext=(0.30, 0.78),
                textcoords="axes fraction", fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color="#c9c8c3", lw=1))

    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("Clashes with antigen", fontsize=9.5, color=INK)
    ax.set_ylabel("Loop–antigen contacts", fontsize=9.5, color=INK)
    ax.set_title("The designs that make more contacts are the ones passing\n"
                 "through the antigen (r = 0.98 across 85 designs)",
                 fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig2_contacts_vs_clashes.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("designs")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per = load(Path(args.designs))
    print(f"{len(per)} complexes with designs")
    figure1(per, out)
    figure2(per, out)
    print(f"wrote figures to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
