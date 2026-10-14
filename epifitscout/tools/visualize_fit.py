#!/usr/bin/env python3
"""Export a search result for viewing in PyMOL.

Writes the template complex, the epitope selection and the top-ranked candidate
loops as PDB files, plus a ``view.pml`` that loads and colours them. Open it
with::

    pymol <out-dir>/view.pml

The point is to make the scoring checkable by eye: a candidate the scorer likes
should sit against the epitope without passing through it.

Example::

    epifitscout view \\
        --structure database_builder/rcsb_cache/3gbn.pdb \\
        --cdr H 93 102 --epitope B 38 46 \\
        --out-dir /tmp/fit_3gbn
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

import epifitscout
from epifitscout.config.schema import MASTERConfig, PipelineConfig
from epifitscout.pipeline.search_pipeline import FragmentSearchPipeline
from epifitscout.scoring.local_fit import LocalFitScorer
from epifitscout.scoring.shape import ShapeComplementarityScorer
from epifitscout.types.fragment import Fragment
from epifitscout.types.superposition import Superposition
from epifitscout.utils.io import (
    extract_residue_atoms,
    fragment_to_pdb_string,
)

_PML = """\
# EpiFitScout: is the scoring believable?
#
#   grey surface   antigen
#   orange         the epitope you targeted
#   green          the template's own CDR loop (ground truth)
#   red            top-ranked candidate under the OLD profile scorer
#   blue           top-ranked candidate under the NEW local-fit scorer
#
# Dashes mark atom pairs closer than {clash:.1f} A, i.e. interpenetration.
# If the red loop has dashes and the blue loop does not, that is the point.

reinitialize
bg_color white
set ray_opaque_background, 1

load template.pdb, complex
load native_loop.pdb, native_loop
load old_pick.pdb, old_pick
load new_pick.pdb, new_pick

hide everything

select antigen, complex and chain {ag_chain} and polymer
select antibody, complex and chain {ab_chain} and polymer
select epitope, antigen and resi {epi_resi}

# antigen as a surface, so the epitope topography is visible
show surface, antigen
color grey80, antigen
color orange, epitope
set transparency, 0.30

show cartoon, antibody
color palecyan, antibody
set cartoon_transparency, 0.65

show sticks, native_loop or old_pick or new_pick
color green, native_loop
color red, old_pick
color marine, new_pick
set stick_radius, 0.20

# every too-close contact, per candidate
distance old_clash, old_pick, antigen, {clash:.2f}, mode=0
distance new_clash, new_pick, antigen, {clash:.2f}, mode=0
color red, old_clash
color blue, new_clash
set dash_width, 3.0
set label_size, 0

orient epitope or native_loop
turn x, -20
zoom epitope or native_loop or old_pick, 4

print "-----------------------------------------------------------------"
print "{caption_1}"
print "{caption_2}"
print "{caption_3}"
print "-----------------------------------------------------------------"
print "Toggle with:  disable old_pick   /   enable old_pick"
"""


def _write_grafted(
    frag: Fragment, superposition: Superposition, path: Path, chain: str
) -> None:
    """Write a grafted fragment, in the antibody-antigen frame, as a PDB."""
    grafted = superposition.apply(frag.coords)
    path.write_text(
        fragment_to_pdb_string(
            grafted,
            sequence=frag.sequence,
            chain_id=chain,
            start_resnum=frag.residue_range[0],
        )
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--structure", type=Path, required=True,
                   help="Template antibody-antigen complex (PDB).")
    p.add_argument("--cdr", nargs=3, metavar=("CHAIN", "FIRST", "LAST"), required=True,
                   help="The CDR loop to replace.")
    p.add_argument("--epitope", nargs=3, metavar=("CHAIN", "FIRST", "LAST"),
                   required=True, help="The epitope you are targeting.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-hits", type=int, default=500)
    p.add_argument("--rmsd-threshold", type=float, default=2.0)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)

    ab_chain, lo, hi = args.cdr[0], int(args.cdr[1]), int(args.cdr[2])
    ag_chain, elo, ehi = args.epitope[0], int(args.epitope[1]), int(args.epitope[2])
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    qb = epifitscout.QueryBuilder(args.structure)
    cdr = qb.get_fragment(ab_chain, lo, hi)
    epi_fragment = qb.get_fragment(ag_chain, elo, ehi)          # old scorer's input
    epi_surface = qb.get_epitope_surface(ag_chain, residues=(elo, ehi))
    reach = qb.epitope_reach(cdr, epi_surface)
    print(f"epitope: {epi_surface.n_residues} residues, "
          f"closest approach to the CDR site {reach:.1f} A")

    body = np.concatenate(
        [c for _, _, c in extract_residue_atoms(args.structure, ab_chain)], axis=0
    )
    cfg = PipelineConfig(
        master=MASTERConfig(rmsd_threshold=args.rmsd_threshold,
                            max_hits=args.max_hits, timeout_seconds=900),
        max_workers=args.workers, log_level="WARNING",
    )
    hits = FragmentSearchPipeline(cfg).run_master(cdr, Path(tempfile.mkdtemp()))
    print(f"{len(hits)} fragments from MASTER")

    old_scorer, new_scorer = ShapeComplementarityScorer(), LocalFitScorer()
    self_id = args.structure.stem.upper()
    scored = []
    for frag, sup, rmsd in hits:
        if frag.pdb_id.upper() == self_id:
            continue                            # exclude the query's own entry
        grafted = sup.apply(frag.coords)
        scored.append((
            frag, sup, rmsd,
            old_scorer.score(grafted, epi_fragment.coords),
            new_scorer.score(grafted, epi_surface,
                             residue_numbers=frag.residue_range,
                             antibody_body=body),
        ))
    if not scored:
        print("no non-self hits to show")
        return 1

    top_old = max(scored, key=lambda r: r[3])
    top_new = max(scored, key=lambda r: r[4].score)

    (out / "native_loop.pdb").write_text(
        fragment_to_pdb_string(cdr.coords, sequence=cdr.sequence, chain_id="N",
                               start_resnum=cdr.residue_range[0])
    )
    _write_grafted(top_old[0], top_old[1], out / "old_pick.pdb", "O")
    _write_grafted(top_new[0], top_new[1], out / "new_pick.pdb", "W")
    (out / "template.pdb").write_bytes(args.structure.read_bytes())

    clash = new_scorer.clash_cutoff
    cap1 = (f"OLD top pick {top_old[0].pdb_id}:{top_old[0].chain} - profile score "
            f"{top_old[3]:.3f}, but {top_old[4].n_clash} residue(s) collide "
            f"(closest {top_old[4].min_gap:.1f} A)")
    cap2 = (f"NEW top pick {top_new[0].pdb_id}:{top_new[0].chain} - local-fit "
            f"{top_new[4].score:.3f}, {top_new[4].n_clash} collisions, "
            f"{top_new[4].matched_length} residues matched at "
            f"{top_new[4].mean_gap:.1f} A")
    cap3 = (f"epitope {ag_chain}/{elo}-{ehi}; closest approach to the CDR site "
            f"{reach:.1f} A")

    (out / "view.pml").write_text(_PML.format(
        ag_chain=ag_chain, ab_chain=ab_chain, epi_resi=f"{elo}-{ehi}",
        clash=clash, caption_1=cap1, caption_2=cap2, caption_3=cap3,
    ))

    n_clashing_top10 = sum(
        1 for r in sorted(scored, key=lambda r: -r[3])[:10] if r[4].n_clash
    )
    print()
    print(cap1)
    print(cap2)
    print(f"of the old scorer's top 10, {n_clashing_top10} collide with the antigen")
    print(f"\nwrote {out}/ — open with:  pymol {out}/view.pml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
