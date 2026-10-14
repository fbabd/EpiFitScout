#!/usr/bin/env python3
"""Benchmark harness: does the geometric step pick better fragments for an epitope?

Builds its cases automatically from SAbDab so the evaluation is not a handful of
hand-picked structures, then records every hit's geometry to a JSONL file.
``analyse_benchmark.py`` turns that into statistics and figures.

Four questions are recorded, each able to come out against the tool:

1. **Does the geometric step add anything MASTER does not already have?**
   MASTER ranks by backbone RMSD and never sees the antigen. For every hit we
   record whether it can be placed at the CDR site without colliding. If
   collisions were simply the high-RMSD hits, a tighter RMSD cut would do the
   same job and this step would be redundant.

2. **Does the score detect real complementarity?** The native loop is scored
   against its own epitope and against decoy patches of the same size elsewhere
   on the same antigen. Same antibody, same antigen, same frame — only the
   epitope changes. A score that measures shape fit should prefer the real site.

3. **Is the native loop recovered?** Where does it rank among the MASTER hits on
   the shape term alone.

4. **Does the score track binding energy?** Native score against the SAbDab
   ΔG, for cases that have one.

Usage::

    python benchmark/run_benchmark.py --limit 40 --out results/benchmark.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import epifitscout  # noqa: E402
from epifitscout.config.schema import MASTERConfig, PipelineConfig  # noqa: E402
from epifitscout.db.rcsb_downloader import RcsbDownloader  # noqa: E402
from epifitscout.pipeline.search_pipeline import FragmentSearchPipeline  # noqa: E402
from epifitscout.scoring.local_fit import LocalFitScorer  # noqa: E402
from epifitscout.scoring.shape import ShapeComplementarityScorer  # noqa: E402
from epifitscout.types.epitope import EpitopeSurface  # noqa: E402
from epifitscout.utils.io import extract_residue_atoms  # noqa: E402

log = logging.getLogger("benchmark")

#: Antigen residues this close to the CDR define the true epitope.
EPITOPE_CUTOFF = 10.0
#: A decoy patch must be at least this far from the true epitope's centre.
DECOY_MIN_SEPARATION = 25.0
N_DECOYS = 3


def load_cases(
    csv_path: Path, limit: int | None, require_dg: bool = False
) -> list[dict]:
    """Pick SAbDab entries with one antigen chain, a heavy chain and an H3 sequence.

    Args:
        require_dg: Keep only entries carrying a usable binding free energy, so
            the affinity question has data to work with. The CSV is not ordered
            by annotation completeness, so an unfiltered prefix can contain none.
    """
    cases: list[dict] = []
    seen: set[str] = set()
    for row in csv.DictReader(csv_path.open()):
        pdb = row["pdb"].strip().lower()
        h = row["Hchain"].strip()
        light = row.get("Lchain", "").strip()
        ag = row["antigen_chain"].strip()
        h3 = row.get("H3", "").strip()
        if pdb in seen or not h or not h3 or len(h3) < 5:
            continue
        if not ag or ag.lower() in ("na", "nan", "none") or "|" in ag:
            continue
        if h.lower() in ("na", "nan", "none"):
            continue
        # SAbDab names an antibody chain as the antigen in ~530 rows (anti-idiotype
        # entries and annotation quirks). Scoring a CDR against its own heavy or
        # light chain is not an epitope measurement, and it scores spuriously well
        # because the two are in permanent intramolecular contact.
        if ag in (h, light):
            continue
        dg = row.get("delta_g", "").strip()
        try:
            dg_val = float(dg)
            if not -20 < dg_val < -3:
                dg_val = float("nan")
        except ValueError:
            dg_val = float("nan")
        if require_dg and not (dg_val == dg_val):      # NaN check
            continue
        seen.add(pdb)
        cases.append({"pdb": pdb, "h_chain": h, "ag_chain": ag, "h3": h3, "delta_g": dg_val})
        if limit and len(cases) >= limit:
            break
    return cases


def locate_h3(qb: epifitscout.QueryBuilder, chain: str, h3_seq: str) -> tuple[int, int] | None:
    """Find the H3 loop's residue numbers by matching the SAbDab H3 sequence.

    Using the deposited sequence rather than a motif regex avoids mis-calling the
    loop, which would silently corrupt every downstream number.
    """
    try:
        frag = qb.get_fragment(chain, 1, 9999)
    except Exception:
        return None
    seq = frag.sequence
    idx = seq.find(h3_seq)
    if idx < 0:
        return None
    resnums = _residue_numbers(qb, chain)
    if resnums is None or idx + len(h3_seq) > len(resnums):
        return None
    return int(resnums[idx]), int(resnums[idx + len(h3_seq) - 1])


def _residue_numbers(qb: epifitscout.QueryBuilder, chain: str) -> list[int] | None:
    from epifitscout.utils.io import extract_backbone_coords_by_range

    try:
        _, resnums, _ = extract_backbone_coords_by_range(
            qb._pdb_path, chain, res_start=1, res_end=9999
        )
    except Exception:
        return None
    return list(resnums)


def decoy_patches(
    structure: Path, ag_chain: str, true_epitope: EpitopeSurface, n: int,
    rng: np.random.Generator, fmt: str | None = None,
) -> list[EpitopeSurface]:
    """Surface patches of the same size, elsewhere on the same antigen.

    Matching the residue count matters: the score rewards longer matched runs, so
    a decoy with fewer residues would lose for the wrong reason.
    """
    residues = extract_residue_atoms(structure, ag_chain, fmt=fmt)
    body = np.concatenate([c for _, _, c in residues], axis=0)
    centroids = np.array([c.mean(axis=0) for _, _, c in residues])
    true_centre = true_epitope.atom_coords.mean(axis=0)
    size = true_epitope.n_residues

    far = [
        i for i, c in enumerate(centroids)
        if np.linalg.norm(c - true_centre) > DECOY_MIN_SEPARATION
    ]
    out: list[EpitopeSurface] = []
    for _ in range(n * 6):
        if len(out) >= n or not far:
            break
        seed = int(rng.choice(far))
        order = np.argsort(np.linalg.norm(centroids - centroids[seed], axis=1))
        chosen = [int(i) for i in order[:size]]
        if len(chosen) < size:
            continue
        patch = [residues[i] for i in chosen]
        surface = EpitopeSurface.from_residues(ag_chain, patch, body=body)
        if np.linalg.norm(surface.atom_coords.mean(axis=0) - true_centre) > DECOY_MIN_SEPARATION:
            out.append(surface)
    return out


def run_case(
    case: dict, cfg: PipelineConfig, downloader: RcsbDownloader, rng: np.random.Generator
) -> dict | None:
    """Run one complex end to end and return its record."""
    # mmCIF is read directly; QueryBuilder picks the parser from the suffix.
    # RCSB serves large complexes as mmCIF only, so skipping them would bias
    # the benchmark against exactly the big assemblies worth testing.
    sf = downloader.download(case["pdb"])
    qb = epifitscout.QueryBuilder(sf.path)
    span = locate_h3(qb, case["h_chain"], case["h3"])
    if span is None:
        return {"pdb": case["pdb"], "error": "H3 not located in chain"}
    lo, hi = span
    if hi - lo + 1 < 5:
        return {"pdb": case["pdb"], "error": "H3 too short"}

    cdr = qb.get_fragment(case["h_chain"], lo, hi)
    try:
        epitope = qb.get_epitope_surface(case["ag_chain"], near=cdr, cutoff=EPITOPE_CUTOFF)
    except ValueError:
        return {"pdb": case["pdb"], "error": "no antigen residue near the CDR"}
    if epitope.n_residues < 5:
        return {"pdb": case["pdb"], "error": f"epitope too small ({epitope.n_residues})"}

    body = np.concatenate(
        [c for _, _, c in extract_residue_atoms(sf.path, case["h_chain"], fmt=sf.fmt)],
        axis=0,
    )
    started = time.time()
    pipe = FragmentSearchPipeline(cfg)
    with tempfile.TemporaryDirectory(prefix="bench_") as tmp:
        hits = pipe.run_master(cdr, Path(tmp))
    if len(hits) < 20:
        return {"pdb": case["pdb"], "error": f"only {len(hits)} MASTER hits"}

    local = LocalFitScorer()
    profile = ShapeComplementarityScorer()
    self_id = case["pdb"].upper()

    rows = []
    for frag, sup, rmsd in hits:
        grafted = sup.apply(frag.coords)
        fit = local.score(grafted, epitope, residue_numbers=frag.residue_range,
                          antibody_body=body)
        rows.append({
            "pdb": frag.pdb_id, "chain": frag.chain, "rmsd": round(float(rmsd), 4),
            "local": round(fit.score, 4), "clash": int(fit.n_clash),
            "matched": fit.matched_length, "min_gap": round(fit.min_gap, 2),
            "profile": round(float(profile.score(grafted, epitope.atom_coords)), 4),
            "self": frag.pdb_id.upper() == self_id,
        })

    # the native loop against its own epitope, and against decoy patches
    native_fit = local.score(cdr.coords, epitope, residue_numbers=(lo, hi), antibody_body=body)
    decoys = decoy_patches(sf.path, case["ag_chain"], epitope, N_DECOYS, rng, fmt=sf.fmt)
    decoy_scores = [
        round(local.score(cdr.coords, d, residue_numbers=(lo, hi), antibody_body=body).score, 4)
        for d in decoys
    ]

    return {
        "pdb": case["pdb"], "h_chain": case["h_chain"], "ag_chain": case["ag_chain"],
        "h3_span": [lo, hi], "h3_len": hi - lo + 1, "delta_g": case["delta_g"],
        "epitope_residues": epitope.n_residues,
        "native_score": round(native_fit.score, 4),
        "native_matched": native_fit.matched_length,
        "native_clash": int(native_fit.n_clash),
        "decoy_scores": decoy_scores,
        "n_hits": len(rows), "seconds": round(time.time() - started, 1),
        "hits": rows,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path,
                   default=REPO_ROOT.parent / "EpiFitScout/data/SAbDab/sabdab_metadata.csv")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results/benchmark.jsonl")
    p.add_argument("--limit", type=int, default=40, help="How many complexes to attempt.")
    p.add_argument("--max-hits", type=int, default=500)
    p.add_argument("--rmsd-threshold", type=float, default=2.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--cache", type=Path,
                   default=REPO_ROOT / "database_builder/rcsb_cache")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--require-dg", action="store_true",
                   help="Only use complexes with a measured binding free energy.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    cases = load_cases(args.metadata, args.limit, require_dg=args.require_dg)
    print(f"{len(cases)} candidate complexes")

    cfg = PipelineConfig(
        master=MASTERConfig(rmsd_threshold=args.rmsd_threshold, max_hits=args.max_hits,
                            timeout_seconds=900),
        max_workers=args.workers, log_level="ERROR",
    )
    downloader = RcsbDownloader(cache_dir=args.cache)
    rng = np.random.default_rng(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = ok = 0
    with args.out.open("w") as fh:
        for case in cases:
            done += 1
            try:
                rec = run_case(case, cfg, downloader, rng)
            except Exception as exc:                      # keep going; record why
                rec = {"pdb": case["pdb"], "error": f"{type(exc).__name__}: {exc}"}
            if rec is None:
                continue
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if "error" in rec:
                print(f"  [{done}/{len(cases)}] {rec['pdb']}: skipped — {rec['error']}")
            else:
                ok += 1
                print(f"  [{done}/{len(cases)}] {rec['pdb']}: {rec['n_hits']} hits, "
                      f"H3 {rec['h3_len']} aa, epitope {rec['epitope_residues']} res, "
                      f"native {rec['native_score']:.3f}, {rec['seconds']}s")

    print(f"\n{ok} complexes completed -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
