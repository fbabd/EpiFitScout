"""Prepare a literature case: fetch PDB, detect CDR-H3, pick a contiguous epitope patch.

CDR-H3 detection follows the Kabat convention used by the bundled 1A2Y example:
the loop is the stretch strictly between the conserved Cys (H92) and the Trp of
the W-G-x-G motif that opens framework 4. Verified to reproduce 1A2Y's
ARERDYRLDY exactly.

Epitope patch mirrors QueryBuilder.get_surface_patch: the antigen residue whose
CA is nearest the H3 centroid, plus/minus `radius` residues, kept contiguous.

usage: prep_case.py <PDBID> <heavy_chain> <antigen_chain> [radius]
"""
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nobio

# Shared with database_builder/, whose cache is likewise not committed.
CACHE = Path(__file__).resolve().parent.parent / "database_builder" / "rcsb_cache"
H3_RE = re.compile(r"C(.{3,70}?)WG.G")


def fetch(pdb_id):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{pdb_id.lower()}.pdb"
    if not p.exists():
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        p.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return p


def chain_seq(path, chain):
    coords, resnums, seq = nobio._parse(str(path), chain)
    return coords, resnums, seq


def find_h3(seq, resnums):
    """Return (lo, hi, subseq) author residue numbers for CDR-H3."""
    ms = list(H3_RE.finditer(seq))
    if not ms:
        return None
    m = ms[-1]                               # H3 is the LAST CDR, FR4 follows it
    i, j = m.start(1), m.end(1)              # between Cys and Trp
    return resnums[i], resnums[j - 1], seq[i:j]


def main():
    pdb_id, hchain, agchain = sys.argv[1], sys.argv[2], sys.argv[3]
    radius = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    path = fetch(pdb_id)

    hc, hres, hseq = chain_seq(path, hchain)
    h3 = find_h3(hseq, hres)
    if h3 is None:
        print(f"ERROR {pdb_id}: no CDR-H3 motif found in chain {hchain}", file=sys.stderr)
        sys.exit(2)
    lo, hi, h3seq = h3

    h3_ca = np.array([hc[k][1] for k, r in enumerate(hres) if lo <= r <= hi])
    ac, ares, aseq = chain_seq(path, agchain)
    ag_ca = ac[:, 1, :]

    d = np.linalg.norm(ag_ca[:, None, :] - h3_ca[None, :, :], axis=-1)
    centre = int(np.argmin(d.min(axis=1)))
    i0, i1 = max(0, centre - radius), min(len(ares) - 1, centre + radius)
    elo, ehi = ares[i0], ares[i1]
    contact = float(d.min())

    print(f"{pdb_id}\t{hchain}\t{lo}\t{hi}\t{agchain}\t{elo}\t{ehi}\t"
          f"{pdb_id}\t{h3seq}\t{len(h3seq)}\t{aseq[i0:i1+1]}\t{contact:.2f}\t{path}")


if __name__ == "__main__":
    main()
