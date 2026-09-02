#!/usr/bin/env python3
"""doc 96 sec 6.1 -- census of ClusteringSeparate's gate over the doc-95 sample.

Parses the SEPDBG lines written by scripts/d96_sepdbg_census.sh and answers the
question sec 5.2 raises: is JudgeSeparateDec_2's boundary route inert on
IN-TIME SBND clusters in general, or only on the owner's two events?

Splits every gate line by whether the cluster is out-of-time (`outx=1` on its
dec2 line, i.e. apparent x past the anode) because that is the one disjunct
SBND's fiducial values leave reachable.

Repro:  ./scripts/d96_sepdbg_census.sh && python3 scripts/d96_sepdbg_parse.py
"""
import collections
import glob
import os
import re

LOGD = "/home/xqian/tmp/d96/census"
RE_DEC2 = re.compile(r"SEPDBG dec2 ident=(-?\d+) len=([\d.]+) nout=(\d+) noutx=(\d+) "
                     r"nsurf=(\d+) nindep=(\d+) nfar=(\d+) outx=(\d+)")
RE_GATE = re.compile(r"SEPDBG gate ident=(-?\d+) len=([\d.]+) nblob=(\d+) dec1=(\d+) dec2=(\d+) "
                     r"nindep=(\d+) r1=([\d.e+-]+) r2=([\d.e+-]+) angbeam=([\d.]+)")

rows = []
for f in sorted(glob.glob(f"{LOGD}/*.log")):
    evt = os.path.basename(f)[:-4]
    last = None
    for ln in open(f, errors="replace"):
        m = RE_DEC2.search(ln)
        if m:
            last = m.groups()
            continue
        m = RE_GATE.search(ln)
        if m:
            ident, ln_, nblob, dec1, dec2, nindep, r1, r2, ang = m.groups()
            d = last if last and last[0] == ident else None
            rows.append(dict(evt=evt, ident=int(ident), length=float(ln_), nblob=int(nblob),
                             dec1=int(dec1), dec2=int(dec2), nindep=int(nindep),
                             r1=float(r1), angbeam=float(ang),
                             nout=int(d[2]) if d else None,
                             noutx=int(d[3]) if d else None,
                             nsurf=int(d[4]) if d else None,
                             outx=int(d[7]) if d else None))
            last = None

n = len(rows)
have = [r for r in rows if r["nout"] is not None]
intime = [r for r in have if r["outx"] == 0]
outoft = [r for r in have if r["outx"] == 1]
print(f"logs parsed: {len(glob.glob(f'{LOGD}/*.log'))};  gate lines: {n} "
      f"(clusters longer than 100 cm);  with a matching dec2 line: {len(have)}")
print(f"  in-time  (outx=0): {len(intime)}")
print(f"  out-of-time (outx=1): {len(outoft)}")

def block(tag, v):
    if not v:
        print(f"\n{tag}: none")
        return
    d2 = sum(r["dec2"] for r in v)
    d1 = sum(r["dec1"] for r in v)
    print(f"\n{tag}  (n={len(v)})")
    print(f"  dec2=1 : {d2:4d}  ({100.0*d2/len(v):.1f}%)")
    print(f"  dec1=1 : {d1:4d}  ({100.0*d1/len(v):.1f}%)")
    print(f"  nout   : {dict(sorted(collections.Counter(r['nout'] for r in v).items()))}")
    print(f"  nsurf  : {dict(sorted(collections.Counter(r['nsurf'] for r in v).items()))}")
    first = [r for r in v if (r["nout"] > 1 and r["nsurf"] > 1)
             or (r["nout"] > 2 and r["length"] > 250) or r["noutx"] > 0]
    print(f"  first conjunct of dec2 satisfied: {len(first)}  ({100.0*len(first)/len(v):.1f}%)")

block("IN-TIME clusters", intime)
block("OUT-OF-TIME clusters", outoft)

long_in = [r for r in intime if r["length"] > 250]
print(f"\nIN-TIME clusters longer than 250 cm (the through-going-cosmic regime): {len(long_in)}")
print(f"  nout histogram: {dict(sorted(collections.Counter(r['nout'] for r in long_in).items()))}")
print(f"  nsurf histogram: {dict(sorted(collections.Counter(r['nsurf'] for r in long_in).items()))}")
print(f"  dec2=1: {sum(r['dec2'] for r in long_in)} / {len(long_in)}")
