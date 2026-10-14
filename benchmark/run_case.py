"""Run EpiFitScout on a case, bypassing the BioPython-dependent QueryBuilder.

usage: run_case.py <pdb> <cdr_chain> <cdr_lo> <cdr_hi> <epi_chain> <epi_lo> <epi_hi> <label>
"""
import json
import sys
import time
from pathlib import Path as _P

_HERE = _P(__file__).resolve().parent
ROOT = str(_HERE.parent)                 # the EpiFitScout install
sys.path.insert(0, str(_HERE))           # for nobio
sys.path.insert(0, ROOT)                 # for epifitscout
import nobio; nobio.install()
from epifitscout.config.schema import MASTERConfig, PipelineConfig
from epifitscout.pipeline.search_pipeline import FragmentSearchPipeline
from epifitscout.types.fragment import Fragment


def backbone(path, chain, lo, hi):
    return nobio._parse(path, chain, lo, hi)


def main():
    pdb, cch, clo, chi, ech, elo, ehi, label = sys.argv[1:9]
    cc, ck, cs = backbone(pdb, cch, int(clo), int(chi))
    ec, ek, es = backbone(pdb, ech, int(elo), int(ehi))
    if len(cc) < 4 or len(ec) < 4:
        print(json.dumps({"label": label, "error": f"too few residues cdr={len(cc)} epi={len(ec)}"}))
        return

    cdr = Fragment(pdb_id=label, chain=cch, residue_range=(ck[0], ck[-1]), coords=cc, sequence=cs)
    epi = Fragment(pdb_id=label, chain=ech, residue_range=(ek[0], ek[-1]), coords=ec, sequence=es)

    # binary_path/database_path left as None: epifitscout.paths resolves the
    # MASTER build and the chain database the same way the library does for a
    # normal user, honouring MASTER/bin, data/ and EPIFITSCOUT_MASTER_DIR / EPIFITSCOUT_DB.
    cfg = PipelineConfig(master=MASTERConfig(
        rmsd_threshold=2.0, max_hits=500, timeout_seconds=1800, n_threads=0),
        max_workers=8, log_level="WARNING")

    t0 = time.time()
    hits = FragmentSearchPipeline(cfg).search(cdr, epi)
    dt = time.time() - t0

    self_id = label.split("_")[0].upper()
    rows = [{"rank": i, "pdb": h.fragment.pdb_id, "chain": h.fragment.chain,
             "res": list(h.fragment.residue_range), "rmsd": round(h.rmsd_cdr, 3),
             "shape": round(h.complementarity_score, 3), "final": round(h.final_score, 4),
             "seq": h.fragment.sequence, "self": h.fragment.pdb_id.upper() == self_id}
            for i, h in enumerate(hits[:25], 1)]
    print(json.dumps({"label": label, "cdr_len": len(cc), "cdr_seq": cs, "epi_len": len(ec),
                      "n_hits": len(hits), "seconds": round(dt, 1), "top": rows}))


if __name__ == "__main__":
    main()
