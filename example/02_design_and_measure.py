"""Design five CDR-H3 loops per input complex and measure both against it.

For every complex in ``selected.tsv`` this runs ``graft()`` with the target's
own entry excluded and its native loop sequence refused, keeps the top five
models, and measures each with :class:`InterfaceScorer`. The same scorer then
measures the input complex at the same span, so the design rows and the input
row differ only in which loop occupies the site.

The input complex is scored twice:

``input_crystal``
    as deposited, with its crystallographic side chains.
``input_repacked``
    after FASPR repacks the same span and shell that a graft gets.

Both are reported because the designs are necessarily FASPR-packed. Comparing
them against crystal side chains would confound the loop with the packing, and
comparing against the repack alone would discard the real reference. The gap
between the two input rows is the size of that confound.

Usage:
    python -I run_analysis.py selected.tsv <cache_dir> <out_dir> [--pool N] [--top N]
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import traceback
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select

import epifitscout
from epifitscout.db.rcsb_downloader import RcsbDownloader
from epifitscout.graft.packer import find_packer, run_packer
from epifitscout.scoring.interface import InterfaceScorer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

#: Terms compared between designs and the input complex.
TERMS = [
    "buried_area", "loop_buried_area", "contacts", "contact_residues",
    "clashes", "hbonds", "salt_bridges", "gap_volume", "gap_index",
    "min_distance",
]

COLUMNS = [
    "complex_id", "arm", "rank", "donor", "h3_len", "h3_seq",
    "master_rmsd", *TERMS,
]


class _Keep(Select):
    """Keep only the named chains, drop water and alternate locations."""

    def __init__(self, chains: set[str]) -> None:
        self.chains = chains

    def accept_chain(self, chain):  # noqa: ANN001, ANN201
        return chain.id in self.chains

    def accept_residue(self, residue):  # noqa: ANN001, ANN201
        return residue.get_resname() not in ("HOH", "DOD")


def isolate(src: Path, pdb: str, chains: set[str], dest: Path) -> Path:
    """Write a PDB holding just ``chains``, so graft() sees one copy.

    Entries with several Fabs or several antigen copies would otherwise have
    the scorer burying the loop against a neighbouring copy it will never meet
    in the model.
    """
    parser = MMCIFParser(QUIET=True) if src.suffix == ".cif" else PDBParser(QUIET=True)
    model = parser.get_structure(pdb, str(src))[0]
    io = PDBIO()
    io.set_structure(model)
    io.save(str(dest), select=_Keep(chains))
    return dest


def emit(writer, fh, **row) -> None:  # noqa: ANN001
    writer.writerow({c: row.get(c, "") for c in COLUMNS})
    fh.flush()


def score_row(score, **extra) -> dict:  # noqa: ANN001
    row = {t: getattr(score, t) for t in TERMS}
    row.update(extra)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("selected")
    ap.add_argument("cache")
    ap.add_argument("out_dir")
    ap.add_argument("--pool", type=int, default=60)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--only", default=None, help="comma-separated complex_ids")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.selected), delimiter="\t"))
    if args.only:
        want = set(args.only.split(","))
        rows = [r for r in rows if r["complex_id"] in want]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(exist_ok=True)

    dl = RcsbDownloader(Path(args.cache))
    scorer = InterfaceScorer()
    faspr = find_packer("FASPR")
    if faspr is None:
        print("FASPR not found; interface numbers would be backbone-only", file=sys.stderr)
        return 2

    tsv = out_dir / "designs.tsv"
    fh = tsv.open("w", newline="")
    writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t")
    writer.writeheader()

    for n, r in enumerate(rows, 1):
        cid = r["complex_id"]
        pdb, hch, ag = r["pdb"], r["h_chain"], r["antigen"]
        span = (int(r["loop_first"]), int(r["loop_last"]))
        chains = {hch, r["l_chain"], ag} - {""}

        print(f"\n[{n}/{len(rows)}] {cid}  H3 {span[0]}-{span[1]} "
              f"({r['h3_len']}) {r['h3_seq']}", flush=True)

        try:
            sf = dl.download(pdb)
            template = isolate(sf.path, pdb, chains, work / f"{cid}.pdb")

            # --- the input complex, as deposited -------------------------
            crystal = scorer.score(template, hch, span, (ag,))
            emit(writer, fh, **score_row(
                crystal, complex_id=cid, arm="input_crystal", rank=0,
                donor=pdb, h3_len=r["h3_len"], h3_seq=r["h3_seq"]))

            # --- the input complex, repacked like a graft ----------------
            repacked = work / f"{cid}_repacked.pdb"
            shutil.copy(template, repacked)
            if run_packer(faspr, "FASPR", repacked, chain=hch, span=span):
                rp = scorer.score(repacked, hch, span, (ag,))
                emit(writer, fh, **score_row(
                    rp, complex_id=cid, arm="input_repacked", rank=0,
                    donor=pdb, h3_len=r["h3_len"], h3_seq=r["h3_seq"]))
            else:
                print("  warn: native repack failed", flush=True)

            # --- the designs ---------------------------------------------
            result = epifitscout.graft(
                template, hch, span, (ag,),
                out_dir=work / cid,
                exclude_donors={pdb.upper()},
                exclude_native_loop=True,
                pool_size=args.pool,
                top_k=args.top,
                packer="FASPR",
                keep_models=True,
            )
            print(f"  funnel: {result.funnel}", flush=True)

            for k, cand in enumerate(result.top, 1):
                if cand.interface is None:
                    continue
                emit(writer, fh, **score_row(
                    cand.interface, complex_id=cid, arm="design", rank=k,
                    donor=cand.source, h3_len=cand.length,
                    h3_seq=cand.loop.sequence, master_rmsd=cand.master_rmsd))
                print(f"    {k}. {cand.donor} {cand.loop.sequence} "
                      f"BSA {cand.interface.buried_area:.0f} "
                      f"clash {cand.interface.clashes}", flush=True)

            if not result.top:
                print("  no models returned", flush=True)

        except Exception:  # noqa: BLE001 - one bad target must not end the run
            print(f"  FAILED {cid}:\n{traceback.format_exc()}", flush=True)

    fh.close()
    print(f"\nwrote {tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
