"""Pick input complexes for the paper analysis, and locate their CDR-H3.

Reads a list of ``pdb_H_L_Ag`` identifiers, fetches each entry from RCSB, and
finds CDR-H3 by its two conserved framework anchors rather than by trusting the
deposited numbering: the Cys that closes framework 3 (IMGT 104) and the Trp of
the WGxG motif that opens framework 4 (IMGT 118). CDR-H3 is what lies strictly
between them, which is the definition ``graft()`` wants -- its four-residue
stems then land on ...VYYC and WGQG..., exactly as the query builder assumes.

A complex is kept only if every residue it needs is unambiguous: no insertion
codes and no repeated numbers across the loop and both stems, since the
pipeline addresses residues by integer number alone. Anything else is reported
with the reason, so the exclusions are auditable rather than silent.

Usage:
    python -I select_complexes.py <ids.txt> <cache_dir> <out.tsv> [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import three_to_index

from epifitscout.db.rcsb_downloader import RcsbDownloader, RcsbDownloadError

#: Residues of framework taken either side of the loop by ``graft()``.
STEM = 4

#: CDR-H3 lengths outside this range are the pathological tails -- a 2-residue
#: loop has no middle to model, and beyond 22 the stems constrain almost none
#: of it. Both ends are kept in the report so the cut is visible.
MIN_H3, MAX_H3 = 5, 22


def one_letter(resname: str) -> str:
    try:
        return "ACDEFGHIKLMNPQRSTVWY"[three_to_index(resname)]
    except (KeyError, IndexError):
        return "X"


def chain_residues(chain) -> list:
    """Standard amino-acid residues of a chain, in file order."""
    return [r for r in chain if r.id[0] == " " and one_letter(r.get_resname()) != "X"]


def find_h3_anchors(residues: list) -> tuple[int, int, str] | None:
    """Return ``(cys_index, trp_index, reason)`` for the CDR-H3 anchors.

    ``cys_index`` is the conserved Cys closing framework 3 and ``trp_index`` the
    Trp of WGxG opening framework 4, both as positions in ``residues``. Returns
    ``None`` with a reason when the pair cannot be identified.
    """
    seq = "".join(one_letter(r.get_resname()) for r in residues)

    # Framework 4 opens with W-G-x-G. Scan from the end: the C-terminal-most
    # match is the real one, since a CDR can itself contain a WG pair.
    trp = None
    for i in range(len(seq) - 4, 80, -1):
        if seq[i] == "W" and seq[i + 1] == "G" and seq[i + 3] == "G":
            trp = i
            break
    if trp is None:
        return None, "no WGxG framework-4 motif"

    # The conserved Cys is the last one before that Trp, and sits ~4-30 residues
    # upstream depending on loop length.
    cys = None
    for i in range(trp - 1, max(trp - 40, 0), -1):
        if seq[i] == "C":
            cys = i
            break
    if cys is None:
        return None, "no framework-3 Cys before WGxG"

    return (cys, trp), ""


def integer_span(residues: list, lo_i: int, hi_i: int) -> tuple[bool, str]:
    """Check that residues ``lo_i..hi_i`` are addressable by integer number."""
    window = residues[lo_i:hi_i + 1]
    for r in window:
        if r.id[2].strip():
            return False, f"insertion code at {r.id[1]}{r.id[2]}"
    nums = [r.id[1] for r in window]
    if len(set(nums)) != len(nums):
        return False, "duplicate residue numbers"
    if nums != sorted(nums):
        return False, "non-monotonic numbering"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids")
    ap.add_argument("cache")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ids = [ln.strip() for ln in Path(args.ids).read_text().split() if ln.strip()]
    if args.limit:
        ids = ids[:args.limit]

    dl = RcsbDownloader(Path(args.cache))
    rows, skipped = [], []

    for n, cid in enumerate(ids, 1):
        parts = cid.split("_")
        if len(parts) < 4:
            skipped.append((cid, "malformed id"))
            continue
        pdb, hch, lch, agch = parts[0], parts[1], parts[2], parts[3]

        try:
            sf = dl.download(pdb)
        except RcsbDownloadError as exc:
            skipped.append((cid, f"download failed: {exc}"))
            continue

        parser = MMCIFParser(QUIET=True) if sf.path.suffix == ".cif" else PDBParser(QUIET=True)
        try:
            model = parser.get_structure(pdb, str(sf.path))[0]
        except Exception as exc:  # noqa: BLE001 - a malformed entry is a skip, not a crash
            skipped.append((cid, f"parse failed: {type(exc).__name__}"))
            continue

        present = {c.id for c in model}
        missing = [c for c in (hch, agch) if c not in present]
        if missing:
            skipped.append((cid, f"chain(s) absent: {','.join(missing)}"))
            continue

        residues = chain_residues(model[hch])
        if len(residues) < 100:
            skipped.append((cid, f"heavy chain too short ({len(residues)})"))
            continue

        anchors, reason = find_h3_anchors(residues)
        if anchors is None:
            skipped.append((cid, reason))
            continue
        cys, trp = anchors

        lo_i, hi_i = cys + 1, trp - 1
        h3_len = hi_i - lo_i + 1
        if not (MIN_H3 <= h3_len <= MAX_H3):
            skipped.append((cid, f"CDR-H3 length {h3_len} outside {MIN_H3}-{MAX_H3}"))
            continue

        # The stems must be addressable too, not just the loop.
        ok, why = integer_span(residues, cys - STEM + 1, trp + STEM - 1)
        if not ok:
            skipped.append((cid, why))
            continue

        seq = "".join(one_letter(r.get_resname()) for r in residues[lo_i:hi_i + 1])
        rows.append({
            "complex_id": cid, "pdb": pdb, "h_chain": hch,
            "l_chain": lch, "antigen": agch,
            "loop_first": residues[lo_i].id[1],
            "loop_last": residues[hi_i].id[1],
            "h3_len": h3_len, "h3_seq": seq,
            "cys": residues[cys].id[1], "trp": residues[trp].id[1],
            "format": sf.path.suffix.lstrip("."),
        })
        print(f"[{n}/{len(ids)}] {cid}: H3 {residues[lo_i].id[1]}-"
              f"{residues[hi_i].id[1]} ({h3_len}) {seq}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(f"\nkept {len(rows)}, skipped {len(skipped)}")
    for cid, why in skipped:
        print(f"  skip {cid}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
