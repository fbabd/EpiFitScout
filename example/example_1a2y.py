"""Standalone script: EpiFitScout search example on 1A2Y (D1.3 anti-lysozyme).

Reproduces the analysis from example_1a2y.ipynb without interactive components.
All figures are saved serially to the test_run/ directory next to this script.

Output files
------------
test_run/
    fig_00_query_structure.png      ← full 1A2Y complex with CDR + epitope (PyMOL)
    fig_01_hit_gallery.png          ← superposed Cα backbone traces, all hits
    fig_02_profiles.png/pdf         ← depth + torsion profiles (query, top, bottom)
    fig_03_score_distribution.png   ← final-score histogram
    fig_03_score_distribution.pdf
    hits_summary.csv

Requirements
------------
- PyMOL.app at /Applications/PyMOL.app  (for fig_00 only; skip message if absent)
- All other figures: project uv environment only

Usage
-----
    cd /path/to/ProtStructDB
    uv run python example/example_1a2y.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make EpiFitScout importable when running outside its directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_EPIFITSCOUT  = _PROJECT_ROOT 

if str(_EPIFITSCOUT) not in sys.path:
    sys.path.insert(0, str(_EPIFITSCOUT))

# Use non-interactive Agg backend before importing pyplot
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd

import epifitscout
from epifitscout.scoring.shape import (
    _interface_normal,
    _interp_align,
    _torsion_profile,
    _smooth_ca,
    _pearson,
)

# ---------------------------------------------------------------------------
# Matplotlib publication defaults
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family"      : "sans-serif",
    "font.sans-serif"  : ["Arial", "DejaVu Sans"],
    "font.size"        : 9,
    "axes.linewidth"   : 0.8,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth"  : 1.5,
    "pdf.fonttype"     : 42,
    "ps.fonttype"      : 42,
})

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_CDR    = "#2166AC"   # query CDR: steel blue
C_EPI    = "#B2182B"   # epitope: dark red
C_HIT    = "#D6604D"   # top-hit profile: coral
C_EPIP   = "#F4A582"   # epitope depth profile: warm orange
C_BOT    = "#762A83"   # bottom-hit profile: purple
C_TOP5   = "#FDAE61"   # top-5 histogram bars: gold
C_REST   = "#B2DFEE"   # remaining bars: pale blue

# Colour cycle for gallery (up to 16 hits)
_GALLERY_CMAP = plt.cm.get_cmap("tab20", 20)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PYMOL_PYTHON  = "/Applications/PyMOL.app/Contents/bin/python3.10"
_RENDER_HELPER = _SCRIPT_DIR / "_pymol_render.py"

PDB_PATH = (_SCRIPT_DIR / "1a2y.pdb").resolve()
OUT = _SCRIPT_DIR / "test_run"
OUT.mkdir(parents=True, exist_ok=True)

assert PDB_PATH.exists(), (
    f"1a2y.pdb not found at {PDB_PATH}. "
    "Place the file in the ProtStructDB root directory."
)

_pymol_available = Path(_PYMOL_PYTHON).exists() and _RENDER_HELPER.exists()

print(f"EpiFitScout {epifitscout.__version__}")
print(f"PDB input   : {PDB_PATH}")
print(f"Output dir  : {OUT}")
print(f"PyMOL       : {'available' if _pymol_available else 'NOT FOUND — fig_00 will be skipped'}")
print()

# ---------------------------------------------------------------------------
# PDB writer — backbone (N, CA, C, O) from (N, 4, 3) coords array
# ---------------------------------------------------------------------------
_AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
_ATOM_NAMES = ["N", "CA", "C", "O"]


def fragment_to_pdb(
    coords: np.ndarray,
    sequence: str = "",
    chain_id: str = "A",
    res_offset: int = 1,
) -> str:
    """Convert (N, 4, 3) backbone coords to PDB-format string."""
    n   = coords.shape[0]
    seq = (sequence or "").ljust(n, "A")
    lines: list[str] = []
    serial = 1
    for ri in range(n):
        resname = _AA1_TO_3.get(seq[ri].upper(), "ALA")
        for j, aname in enumerate(_ATOM_NAMES):
            x, y, z = coords[ri, j]
            lines.append(
                f"ATOM  {serial:5d}  {aname:<3s} {resname} {chain_id}{ri + res_offset:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {aname[0]}\n"
            )
            serial += 1
    lines.append("END\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# 1 — Inspect 1A2Y chains
# ---------------------------------------------------------------------------
qb = epifitscout.QueryBuilder(PDB_PATH)
qb.describe()

# ---------------------------------------------------------------------------
# 2 — Define query CDR-H3 and epitope
# ---------------------------------------------------------------------------
query_cdr = qb.get_fragment("B", res_start=105, res_end=117)
query_cdr.metadata["cdr_type"] = "H3"
query_epi = qb.get_fragment("C", res_start=19, res_end=27)

_cdr_res = f"{query_cdr.residue_range[0]}-{query_cdr.residue_range[1]}"
_epi_res = f"{query_epi.residue_range[0]}-{query_epi.residue_range[1]}"

print(
    f"\nCDR-H3  : {query_cdr.length} residues  [{_cdr_res}]  seq = {query_cdr.sequence}"
)
print(
    f"Epitope : {query_epi.length} residues  [{_epi_res}]  seq = {query_epi.sequence}"
)
print()

# ---------------------------------------------------------------------------
# 3 — Run EpiFitScout search
# ---------------------------------------------------------------------------
print("Running EpiFitScout search ...")
hits = epifitscout.search(query_cdr, query_epi)

print(f"\nTotal ranked hits: {len(hits)}")
print(
    f'{"Rank":<5} {"PDB:chain":<14} {"Residues":<12} '
    f'{"RMSD":>6}  {"Shape":>6}  {"Score":>7}'
)
print("-" * 52)
for i, h in enumerate(hits[:15], 1):
    res = f"{h.fragment.residue_range[0]}-{h.fragment.residue_range[1]}"
    print(
        f"{i:<5} {h.fragment.pdb_id}:{h.fragment.chain:<10} "
        f"{res:<12} "
        f"{h.rmsd_cdr:>6.3f}  "
        f"{h.complementarity_score:>6.3f}  "
        f"{h.final_score:>7.4f}"
    )
print()

# ---------------------------------------------------------------------------
# Figure 0 — Full 1A2Y complex: query CDR (blue) + epitope (red)  via PyMOL
# ---------------------------------------------------------------------------
_out0 = OUT / "fig_00_query_structure.png"

if _pymol_available:
    print("Rendering query structure with PyMOL ...")
    _complex_cfg = {
        "mode"                 : "complex",
        "pdb_file"             : str(PDB_PATH),
        "width"                : 700,
        "height"               : 550,
        "output_png"           : str(_out0),
        "context_color"        : "gray80",
        "context_transparency" : 0.65,
        "orient_selection"     : "chain B+C",
        "highlights": [
            {
                "selection"           : f"chain B and resi {_cdr_res}",
                "color"               : "marine",
                "surface"             : True,
                "surface_transparency": 0.9,
            },
            {
                "selection"           : f"chain C and resi {_epi_res}",
                "color"               : "firebrick",
                "surface"             : True,
                "surface_transparency": 0.9,
            },
        ],
    }
    with tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False, prefix="epifitscout_"
    ) as _tmp:
        json.dump(_complex_cfg, _tmp)
        _tmp_path = _tmp.name

    _res0 = subprocess.run(
        [_PYMOL_PYTHON, str(_RENDER_HELPER), _tmp_path],
        capture_output=True, text=True,
    )
    Path(_tmp_path).unlink(missing_ok=True)

    if _res0.returncode != 0:
        print(f"  [WARNING] PyMOL complex render failed:\n{_res0.stderr[:400]}")
    else:
        print(f"Figure 0 saved → {_out0.name}")
else:
    print(f"  [SKIP] PyMOL not available — {_out0.name} not generated.")


# ---------------------------------------------------------------------------
# Figure 2 — Depth + torsion profiles: query CDR, top hit, bottom hit
# ---------------------------------------------------------------------------

# Shared interface normal: query CDR centroid → query epitope centroid
# (backbone-corrected via SVD on CDR Cα). All depth profiles are projected
# onto this same axis so they are directly comparable.
_qcdr_ca = query_cdr.ca_coords         # (N, 3)
_qepi_ca = query_epi.coords[:, 1, :]   # (M, 3)
n_hat    = _interface_normal(_qcdr_ca, _qepi_ca)

# ── query CDR depth + torsion ───────────────────────────────────────────────
d_qcdr   = _qcdr_ca @ n_hat;  d_qcdr -= d_qcdr.mean()
tau_qcdr = _torsion_profile(_smooth_ca(_qcdr_ca))

# ── top hit ─────────────────────────────────────────────────────────────────
top     = hits[0]
_top_ca = top.superposition.apply(top.fragment.coords)[:, 1, :]
d_top   = _top_ca @ n_hat;  d_top -= d_top.mean()
tau_top = _torsion_profile(_smooth_ca(_top_ca)) if len(_top_ca) >= 4 else np.array([0.0])

# ── bottom hit ──────────────────────────────────────────────────────────────
has_bottom = len(hits) >= 2
if has_bottom:
    bot     = hits[-1]
    _bot_ca = bot.superposition.apply(bot.fragment.coords)[:, 1, :]
    d_bot   = _bot_ca @ n_hat;  d_bot -= d_bot.mean()
    tau_bot = _torsion_profile(_smooth_ca(_bot_ca)) if len(_bot_ca) >= 4 else np.array([0.0])

# ── epitope depth (shown as reference dashed line) ──────────────────────────
d_epi = _qepi_ca @ n_hat;  d_epi -= d_epi.mean()

print(
    f"Top hit   : {top.fragment.pdb_id}:{top.fragment.chain}  "
    f"RMSD={top.rmsd_cdr:.3f}  S_shape={top.complementarity_score:.3f}  "
    f"S_final={top.final_score:.4f}"
)
if has_bottom:
    print(
        f"Bot hit   : {bot.fragment.pdb_id}:{bot.fragment.chain}  "
        f"RMSD={bot.rmsd_cdr:.3f}  S_shape={bot.complementarity_score:.3f}  "
        f"S_final={bot.final_score:.4f}"
    )

# ── resample helpers ─────────────────────────────────────────────────────────
_Nd = max(len(d_qcdr), len(d_top), len(d_bot) if has_bottom else 0)
_xd = np.linspace(0, 1, _Nd)


def _rd(arr: np.ndarray) -> np.ndarray:
    return np.interp(_xd, np.linspace(0, 1, len(arr)), arr)


_Nt = max(len(tau_qcdr), len(tau_top), len(tau_bot) if has_bottom else 0)
_xt = np.linspace(0, 1, _Nt)


def _rt(arr: np.ndarray) -> np.ndarray:
    return np.interp(_xt, np.linspace(0, 1, len(arr)), arr)


# ── plot ─────────────────────────────────────────────────────────────────────
fig2, (ax_dep, ax_tau) = plt.subplots(1, 2, figsize=(8.0, 3.2))

# — Panel A: depth profiles —
ax_dep.plot(_xd, _rd(d_qcdr), "-o", color=C_CDR,  ms=4, lw=1.8, label="Query CDR")
ax_dep.plot(_xd, _rd(d_top),  "-s", color=C_HIT,  ms=4, lw=1.8,
            label=f"Top hit  {top.fragment.pdb_id}:{top.fragment.chain}")
ax_dep.plot(_xd, _rd(d_epi),  "--", color=C_EPIP, ms=0, lw=1.4, alpha=0.85,
            label="Query epitope")
if has_bottom:
    ax_dep.plot(_xd, _rd(d_bot), "-^", color=C_BOT, ms=4, lw=1.8,
                label=f"Bottom hit  {bot.fragment.pdb_id}:{bot.fragment.chain}")
ax_dep.fill_between(_xd, _rd(d_qcdr), 0, alpha=0.10, color=C_CDR)
ax_dep.fill_between(_xd, _rd(d_top),  0, alpha=0.10, color=C_HIT)
if has_bottom:
    ax_dep.fill_between(_xd, _rd(d_bot), 0, alpha=0.08, color=C_BOT)
ax_dep.axhline(0, color="#cccccc", lw=0.8, ls="--", zorder=0)

ax_dep.text(
    0.03, 0.97,
    f"Top: $S_d$={-_pearson(*_interp_align(d_top, d_epi)):.2f}  "
    f"$S_{{\\mathrm{{shape}}}}$={top.complementarity_score:.2f}\n"
    + (
        f"Bot: $S_d$={-_pearson(*_interp_align(d_bot, d_epi)):.2f}  "
        f"$S_{{\\mathrm{{shape}}}}$={bot.complementarity_score:.2f}"
        if has_bottom else ""
    ),
    transform=ax_dep.transAxes, fontsize=7, va="top", family="monospace",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9, lw=0.7),
)
ax_dep.set_xlabel("Fractional arc-length (N→C)", fontsize=8)
ax_dep.set_ylabel(r"Depth along $\hat{n}$ (Å, centred)", fontsize=8)
ax_dep.legend(fontsize=6.5, loc="upper right", framealpha=0.85, edgecolor="#cccccc")
ax_dep.tick_params(labelsize=7)
ax_dep.set_title("A   Depth profiles", fontsize=8, loc="left", fontweight="bold")

# — Panel B: torsion profiles —
ax_tau.plot(_xt, _rt(tau_qcdr), "-o", color=C_CDR, ms=4, lw=1.8, label="Query CDR")
ax_tau.plot(_xt, _rt(tau_top),  "-s", color=C_HIT, ms=4, lw=1.8,
            label=f"Top hit  {top.fragment.pdb_id}:{top.fragment.chain}")
if has_bottom:
    ax_tau.plot(_xt, _rt(tau_bot), "-^", color=C_BOT, ms=4, lw=1.8,
                label=f"Bottom hit  {bot.fragment.pdb_id}:{bot.fragment.chain}")
ax_tau.fill_between(_xt, _rt(tau_qcdr), 0, alpha=0.10, color=C_CDR)
ax_tau.fill_between(_xt, _rt(tau_top),  0, alpha=0.10, color=C_HIT)
if has_bottom:
    ax_tau.fill_between(_xt, _rt(tau_bot), 0, alpha=0.08, color=C_BOT)
ax_tau.axhline(0, color="#cccccc", lw=0.8, ls="--", zorder=0)

_r_tau_top = _pearson(_rt(tau_top), _rt(tau_qcdr))
ax_tau.text(
    0.03, 0.97,
    f"Top vs CDR: $r_{{\\tau}}$={_r_tau_top:.2f}\n"
    + (
        f"Bot vs CDR: $r_{{\\tau}}$={_pearson(_rt(tau_bot), _rt(tau_qcdr)):.2f}"
        if has_bottom else ""
    ),
    transform=ax_tau.transAxes, fontsize=7, va="top", family="monospace",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9, lw=0.7),
)
ax_tau.set_xlabel("Fractional arc-length (N→C)", fontsize=8)
ax_tau.set_ylabel(r"Backbone torsion $\tau$ (rad/Å)", fontsize=8)
ax_tau.legend(fontsize=6.5, loc="upper right", framealpha=0.85, edgecolor="#cccccc")
ax_tau.tick_params(labelsize=7)
ax_tau.set_title("B   Backbone torsion profiles", fontsize=8, loc="left", fontweight="bold")

fig2.suptitle(
    "1A2Y CDR-H3 (query) vs top/bottom ranked hits  |  "
    "shared interface normal from query CDR centroid",
    fontsize=7.5, color="#444444", y=1.01,
)
fig2.tight_layout(pad=0.6, w_pad=1.0)
_out2_png = OUT / "fig_02_profiles.png"
_out2_pdf = OUT / "fig_02_profiles.pdf"
fig2.savefig(_out2_png, dpi=200, bbox_inches="tight")
fig2.savefig(_out2_pdf, bbox_inches="tight")
plt.close(fig2)
print(f"Figure 2 saved → {_out2_png.name}  |  {_out2_pdf.name}")

# ---------------------------------------------------------------------------
# Figure 3 — Score distribution
# ---------------------------------------------------------------------------
scores      = np.array([h.final_score for h in hits])
n_hits      = len(scores)
top5_thresh = hits[min(4, n_hits - 1)].final_score

fig3, ax3 = plt.subplots(figsize=(4.0, 3.0))

bins = np.linspace(scores.min() - 0.01, scores.max() + 0.01, max(10, min(28, n_hits + 2)))
counts, edges, patches = ax3.hist(
    scores, bins=bins, color=C_REST, edgecolor="white", linewidth=0.5, zorder=2
)
for patch, left in zip(patches, edges[:-1]):
    if left >= top5_thresh - 1e-6:
        patch.set_facecolor(C_TOP5)
        patch.set_edgecolor("white")

ax3.axvline(top5_thresh, color=C_TOP5, lw=1.4, ls="--", zorder=3)
ax3.text(
    top5_thresh + 0.003, counts.max() * 0.92,
    "Top 5", fontsize=7, color=C_TOP5, fontweight="bold", va="top",
)
ax3.annotate(
    f"{top.fragment.pdb_id}:{top.fragment.cdr_type}\n"
    f"RMSD={top.rmsd_cdr:.2f} Å\n"
    f"$S_{{\\mathrm{{shape}}}}$={top.complementarity_score:.2f}",
    xy=(scores[0], 1.0),
    xytext=(scores[0] - 0.06, counts.max() * 0.65),
    fontsize=6.5, ha="center",
    arrowprops=dict(arrowstyle="->", color="#555", lw=0.9),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9, lw=0.7),
)
ax3.text(
    0.03, 0.97,
    f"$N_{{\\mathrm{{hits}}}}$ = {n_hits}\n"
    f"$\\delta$ = 2.0 Å\n"
    f"$w_r$ = 0.4,  $w_s$ = 0.6",
    transform=ax3.transAxes, fontsize=7, va="top", family="monospace",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9, lw=0.7),
)
ax3.set_xlabel(r"$S_{\mathrm{final}}$ score", fontsize=8)
ax3.set_ylabel("Number of hits", fontsize=8)
ax3.tick_params(labelsize=7)
ax3.set_xlim(edges[0], edges[-1])
ax3.legend(
    handles=[
        Patch(facecolor=C_REST, edgecolor="white", label="All hits"),
        Patch(facecolor=C_TOP5, edgecolor="white", label="Top 5"),
    ],
    fontsize=7, loc="center left", framealpha=0.85, edgecolor="#cccccc",
)

fig3.tight_layout(pad=0.5)
_out3_png = OUT / "fig_03_score_distribution.png"
_out3_pdf = OUT / "fig_03_score_distribution.pdf"
fig3.savefig(_out3_png, dpi=200, bbox_inches="tight")
fig3.savefig(_out3_pdf, bbox_inches="tight")
plt.close(fig3)
print(f"Figure 3 saved → {_out3_png.name}  |  {_out3_pdf.name}")

# ---------------------------------------------------------------------------
# Hit summary table — saved as CSV
# ---------------------------------------------------------------------------
rows = []
for i, h in enumerate(hits[:15], 1):
    rows.append({
        "Rank"     : i,
        "PDB:chain": f"{h.fragment.pdb_id}:{h.fragment.chain}",
        "Residues" : f"{h.fragment.residue_range[0]}-{h.fragment.residue_range[1]}",
        "Sequence" : h.fragment.sequence,
        "RMSD (A)" : round(h.rmsd_cdr, 3),
        "S_shape"  : round(h.complementarity_score, 3),
        "S_final"  : round(h.final_score, 4),
    })

df = pd.DataFrame(rows).set_index("Rank")
_out_csv = OUT / "hits_summary.csv"
df.to_csv(_out_csv)
print(f"Summary   saved → {_out_csv.name}")

print()
print("All outputs:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name:<44}  {f.stat().st_size // 1024} KB")
