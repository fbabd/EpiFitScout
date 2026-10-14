#!/usr/bin/env python3
"""Epitope-specificity control: does the loop prefer its own site over reachable ones?

``run_benchmark.py`` placed decoy patches at least 25 Å from the true epitope.
Those score zero because the loop cannot physically reach them, which tests
proximity rather than complementarity and is no evidence at all.

This recomputes the control with decoys that are **reachable**: patches of the
same residue count, on the same antigen, that the native loop comes as close to
as it does to its real epitope, but which share few residues with it. If the
score is measuring shape fit rather than mere contact, the true epitope should
still win. If it does not, the score is measuring proximity.

Usage::

    python benchmark/decoy_epitopes.py --records benchmark/results/benchmark.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analyse_benchmark import read_records  # noqa: E402

import epifitscout  # noqa: E402
from epifitscout.scoring.local_fit import LocalFitScorer  # noqa: E402
from epifitscout.types.epitope import EpitopeSurface  # noqa: E402
from epifitscout.utils.io import extract_residue_atoms  # noqa: E402

#: A decoy may share at most this fraction of its residues with the true epitope.
MAX_OVERLAP = 0.25
#: A decoy must be reachable: the loop's closest approach to it may exceed the
#: approach to the true epitope by at most this factor.
MAX_REACH_RATIO = 1.5
N_DECOYS = 5


def reachable_decoys(
    structure: Path,
    ag_chain: str,
    cdr_coords: np.ndarray,
    true_epitope: EpitopeSurface,
    rng: np.random.Generator,
) -> list[EpitopeSurface]:
    """Equal-sized patches the loop can reach, overlapping the true site little."""
    residues = extract_residue_atoms(structure, ag_chain)
    body = np.concatenate([c for _, _, c in residues], axis=0)
    centroids = np.array([c.mean(axis=0) for _, _, c in residues])
    size = true_epitope.n_residues
    true_set = set(true_epitope.residue_numbers)
    probe = cdr_coords.reshape(-1, 3)
    true_reach = float(cdist(probe, true_epitope.atom_coords).min())
    limit = max(true_reach * MAX_REACH_RATIO, true_reach + 2.0)

    out: list[EpitopeSurface] = []
    order = rng.permutation(len(residues))
    for seed in order:
        if len(out) >= N_DECOYS:
            break
        near = np.argsort(np.linalg.norm(centroids - centroids[seed], axis=1))[:size]
        if len(near) < size:
            continue
        patch = [residues[int(i)] for i in near]
        nums = {r[0] for r in patch}
        if len(nums & true_set) / size > MAX_OVERLAP:
            continue
        surface = EpitopeSurface.from_residues(ag_chain, patch, body=body)
        if float(cdist(probe, surface.atom_coords).min()) > limit:
            continue                      # the loop cannot reach it; not a fair decoy
        out.append(surface)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", type=Path,
                   default=Path(__file__).parent / "results/benchmark.jsonl")
    p.add_argument("--cache", type=Path, default=REPO_ROOT / "database_builder/rcsb_cache")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).parent / "results/specificity.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    records = read_records(args.records)
    records = [r for r in records if "error" not in r]
    rng = np.random.default_rng(args.seed)
    scorer = LocalFitScorer()

    rows = []
    for r in records:
        path = args.cache / f"{r['pdb']}.pdb"
        if not path.is_file():
            continue
        try:
            qb = epifitscout.QueryBuilder(path)
            cdr = qb.get_fragment(r["h_chain"], *r["h3_span"])
            true_epi = qb.get_epitope_surface(r["ag_chain"], near=cdr, cutoff=10.0)
            body = np.concatenate(
                [c for _, _, c in extract_residue_atoms(path, r["h_chain"])], axis=0
            )
            decoys = reachable_decoys(path, r["ag_chain"], cdr.coords, true_epi, rng)
        except Exception:
            continue
        if len(decoys) < 2:
            continue                      # antigen too small to host a fair decoy

        true_score = scorer.score(cdr.coords, true_epi,
                                  residue_numbers=tuple(r["h3_span"]),
                                  antibody_body=body).score
        decoy_scores = [
            scorer.score(cdr.coords, d, residue_numbers=tuple(r["h3_span"]),
                         antibody_body=body).score
            for d in decoys
        ]
        rows.append({
            "pdb": r["pdb"], "true": round(true_score, 4),
            "decoys": [round(x, 4) for x in decoy_scores],
            "decoy_mean": round(float(np.mean(decoy_scores)), 4),
            "decoy_best": round(float(np.max(decoy_scores)), 4),
            "n_decoys": len(decoys),
        })
        print(f"  {r['pdb']}: true {true_score:.3f}  decoys "
              f"{np.mean(decoy_scores):.3f} (best {np.max(decoy_scores):.3f}, "
              f"n={len(decoys)})")

    if len(rows) < 5:
        print(f"\nonly {len(rows)} complexes could host a fair decoy; not enough to test")
        return 1

    t = np.array([x["true"] for x in rows])
    dm = np.array([x["decoy_mean"] for x in rows])
    db = np.array([x["decoy_best"] for x in rows])

    print("\n" + "=" * 70)
    print("Epitope specificity against REACHABLE decoys")
    print("=" * 70)
    print(f"  {len(rows)} complexes, {sum(x['n_decoys'] for x in rows)} decoy patches")
    print(f"  mean score: true {t.mean():.3f} | decoy mean {dm.mean():.3f} | "
          f"decoy best {db.mean():.3f}")
    for label, d in (("mean decoy", dm), ("best decoy", db)):
        wins = int((t > d).sum())
        w = stats.wilcoxon(t, d, alternative="greater")
        diff = t - d
        print(f"  vs {label}: true wins {wins}/{len(t)} ({100*wins/len(t):.0f}%), "
              f"p = {w.pvalue:.3g}, Cohen's d = {diff.mean()/(diff.std()+1e-9):.2f}")
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
