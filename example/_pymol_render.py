"""PyMOL headless rendering helper for EpiFitScout.

Run with the PyMOL.app bundled Python interpreter, NOT with the project uv env:

    /Applications/PyMOL.app/Contents/bin/python3.10 _pymol_render.py config.json

Two rendering modes, selected by the top-level "mode" key in the JSON config:

──────────────────────────────────────────────────────────────────────────────
Mode "complex"  —  full PDB complex with highlighted selections
──────────────────────────────────────────────────────────────────────────────
{
    "mode": "complex",
    "pdb_file": "/abs/path/to/structure.pdb",
    "width": 600,
    "height": 500,
    "output_png": "/abs/path/to/output.png",
    "context_color": "gray80",
    "context_transparency": 0.65,
    "highlights": [
        {
            "selection": "chain B and resi 105-117",
            "color": "marine",
            "surface": true,
            "surface_transparency": 0.3
        },
        ...
    ],
    "orient_selection": "chain B+C"   // optional: what to zoom/orient to
}

──────────────────────────────────────────────────────────────────────────────
Mode "gallery"  —  per-panel fragment rendering (deprecated: use matplotlib 3D)
──────────────────────────────────────────────────────────────────────────────
{
    "mode": "gallery",              // optional, "gallery" is the default
    "panel_width":  300,
    "panel_height": 250,
    "panels": [
        {
            "models": [
                {"pdb_file": "/path/query.pdb", "name": "query", "color": "firebrick"},
                {"pdb_file": "/path/hit.pdb",   "name": "hit",   "color": "marine"}
            ],
            "output_png": "/path/panel_00.png"
        },
        ...
    ]
}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _common_settings(cmd) -> None:
    """Apply shared display settings."""
    cmd.bg_color("white")
    cmd.set("antialias", 2)
    cmd.set("ray_shadows", 0)
    cmd.set("ray_opaque_background", 0)
    cmd.set("depth_cue", 0)
    cmd.set("specular", 0.15)
    cmd.set("cartoon_tube_radius", 0.35)
    cmd.set("cartoon_loop_radius", 0.35)
    cmd.set("cartoon_fancy_helices", 1)


# ── complex mode ─────────────────────────────────────────────────────────────

def render_complex(cmd, cfg: dict) -> None:
    """Render a full PDB complex with highlighted regions."""
    cmd.reinitialize()
    _common_settings(cmd)

    pdb_file    = cfg["pdb_file"]
    out_png     = cfg["output_png"]
    width       = cfg.get("width", 600)
    height      = cfg.get("height", 500)
    ctx_color   = cfg.get("context_color", "gray80")
    ctx_transp  = cfg.get("context_transparency", 0.65)
    orient_sel  = cfg.get("orient_selection", "all")

    # load full structure as grey cartoon context
    cmd.load(pdb_file, "complex")
    cmd.show_as("cartoon", "complex")
    cmd.color(ctx_color, "complex")
    cmd.set("cartoon_transparency", ctx_transp, "complex")

    # highlighted regions
    for i, h in enumerate(cfg.get("highlights", [])):
        sel_name = f"hl_{i}"
        cmd.select(sel_name, h["selection"])
        cmd.show("cartoon", sel_name)
        cmd.set("cartoon_transparency", 0.0, sel_name)
        cmd.color(h["color"], sel_name)
        if h.get("surface", False):
            cmd.show("surface", sel_name)
            cmd.set("transparency", h.get("surface_transparency", 0.3), sel_name)

    # orient camera
    cmd.orient(orient_sel)
    cmd.zoom(orient_sel, buffer=4)
    cmd.rotate("y", 10)   # slight rotation for a better angle

    cmd.ray(width, height)
    cmd.png(out_png, dpi=200)
    print(f"  complex → {out_png}", flush=True)


# ── gallery mode ─────────────────────────────────────────────────────────────

def render_gallery(cmd, cfg: dict) -> None:
    """Render per-panel fragment PDBs."""
    pw = cfg.get("panel_width", 300)
    ph = cfg.get("panel_height", 250)

    for panel in cfg["panels"]:
        cmd.reinitialize()
        _common_settings(cmd)

        for m in panel["models"]:
            cmd.load(m["pdb_file"], m["name"])
            cmd.show_as("cartoon", m["name"])
            cmd.color(m["color"], m["name"])

        cmd.orient("all")
        cmd.zoom("all", buffer=2)
        cmd.ray(pw, ph)
        cmd.png(panel["output_png"], dpi=150)
        print(f"  panel → {panel['output_png']}", flush=True)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python _pymol_render.py config.json", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    import pymol2  # type: ignore  (only available in PyMOL.app Python)

    mode = cfg.get("mode", "gallery")

    with pymol2.PyMOL() as p:
        if mode == "complex":
            render_complex(p.cmd, cfg)
        else:
            render_gallery(p.cmd, cfg)


if __name__ == "__main__":
    main()
