"""Score every MASTER hit three ways, to show what the superposition fix changed.

shipped : the hit scored in its own crystal frame. This is what v0.1.0 did, and
          is reproduced here by scoring the raw coordinates directly.
fixed   : Kabsch-superpose the hit's CA trace onto the query CDR's here in this
          script, then score. An independent implementation of the fix.
library : the score from the library's own superposition (``Superposition``
          returned by ``run_master``). Since the fix this must equal ``fixed``;
          a difference would mean the library and this script disagree.

Everything else (the scorer, the weights, the ranking formula) is unchanged, so
any difference between columns is attributable to the superposition alone.

usage: rescore.py <pdb> <cdr_chain> <lo> <hi> <epi_chain> <elo> <ehi> <label>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = str(HERE.parent)                       # the EpiFitScout install
sys.path.insert(0, str(HERE))
sys.path.insert(0, ROOT)
import nobio; nobio.install()

from epifitscout.config.schema import MASTERConfig, PipelineConfig
from epifitscout.pipeline.search_pipeline import FragmentSearchPipeline
from epifitscout.scoring.shape import ShapeComplementarityScorer
from epifitscout.types.fragment import Fragment
from epifitscout.utils.geometry import kabsch_superpose

W_RMSD, W_SHAPE = 0.4, 0.6


def main():
    pdb, cch, clo, chi, ech, elo, ehi, label = sys.argv[1:9]
    cc, ck, cs = nobio._parse(pdb, cch, int(clo), int(chi))
    ec, ek, es = nobio._parse(pdb, ech, int(elo), int(ehi))

    cdr = Fragment(pdb_id=label, chain=cch, residue_range=(ck[0], ck[-1]), coords=cc, sequence=cs)
    epi = Fragment(pdb_id=label, chain=ech, residue_range=(ek[0], ek[-1]), coords=ec, sequence=es)

    # binary_path/database_path left as None: epifitscout.paths resolves the
    # MASTER build and the chain database the same way the library does for a
    # normal user, honouring MASTER/bin, data/ and EPIFITSCOUT_MASTER_DIR / EPIFITSCOUT_DB.
    cfg = PipelineConfig(master=MASTERConfig(
        rmsd_threshold=2.0, max_hits=500, timeout_seconds=1800, n_threads=0),
        max_workers=8, log_level="WARNING")

    pipe = FragmentSearchPipeline(cfg)
    workdir = HERE / "work" / label
    workdir.mkdir(parents=True, exist_ok=True)
    hits = pipe.run_master(cdr, workdir)

    scorer = ShapeComplementarityScorer()
    q_ca = cdr.coords[:, 1, :]
    self_id = label.upper()
    rows = []

    for frag, sup, rmsd in hits:
        shipped = scorer.score(frag.coords, epi.coords)       # the v0.1.0 behaviour: raw crystal frame
        library = scorer.score(sup.apply(frag.coords), epi.coords)  # the library, post-fix

        h_ca = frag.coords[:, 1, :]
        if len(h_ca) != len(q_ca):
            fixed, achieved = None, None
        else:
            _, rot, trans = kabsch_superpose(h_ca, q_ca)
            moved = np.einsum("nkj,ij->nki", frag.coords, rot) + trans
            fixed = scorer.score(moved, epi.coords)
            achieved = float(np.sqrt(((moved[:, 1, :] - q_ca) ** 2).sum() / len(q_ca)))

        rows.append({"pdb": frag.pdb_id, "chain": frag.chain,
                     "res": list(frag.residue_range), "seq": frag.sequence,
                     "rmsd": round(float(rmsd), 3),
                     "shipped": round(float(shipped), 3),
                     "library": round(float(library), 3),
                     "fixed": None if fixed is None else round(float(fixed), 3),
                     "kabsch_rmsd": None if achieved is None else round(achieved, 3),
                     "self": frag.pdb_id.upper() == self_id})

    def rank_by(key):
        ok = [r for r in rows if r[key] is not None]
        ok.sort(key=lambda r: W_RMSD / (1 + r["rmsd"]) + W_SHAPE * r[key], reverse=True)
        for i, r in enumerate(ok, 1):
            r[f"rank_{key}"] = i
        return ok

    rank_by("shipped"); rank_by("fixed"); rank_by("library")
    print(json.dumps({"label": label, "cdr_seq": cs, "cdr_len": len(cc),
                      "epi_seq": es, "n_hits": len(rows), "rows": rows}))


if __name__ == "__main__":
    main()
