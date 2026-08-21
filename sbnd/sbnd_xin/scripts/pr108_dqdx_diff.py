#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/108 sec 9 -- diff the dQ/dx SYSTEM (WCT_DQDX_DUMP / WCP_DQDX_DUMP) near a junction.

Record: <call> dqdx <i> x y z local_dx regU regV regW cU sU bU cV sV bV cW sW bW Fdiag pos nconn [j:ou/ov/ow:Fij ...]
  c* = number of 2-D rows this position couples to (per plane), s* = sum of response/err
  values in that column, b* = (R^T data)(i) = data-weighted pull of that plane on this
  position, Fdiag = regulariser diagonal (lambda applied), pos = fitted dQ.
The call used per file is the LAST call with a position within --rj of the junction.
Usage: pr108_dqdx_diff.py --j x y z --arm LABEL=file ... [--rj 2]
"""
import argparse, numpy as np

def parse(path):
    calls = {}
    for line in open(path):
        f = line.split()
        if len(f) < 3 or f[1] != "dqdx": continue
        c = int(f[0])
        if f[2] == "call": calls.setdefault(c, {"hdr": line.strip(), "rows": []}); continue
        rec = dict(i=int(f[2]), p=np.array([float(f[3]), float(f[4]), float(f[5])]), dx=float(f[6]),
                   reg=(int(f[7]), int(f[8]), int(f[9])),
                   u=(int(f[10]), float(f[11]), float(f[12])), v=(int(f[13]), float(f[14]), float(f[15])), w=(int(f[16]), float(f[17]), float(f[18])),
                   fd=float(f[19]), pos=float(f[20]), nconn=int(f[21]), conn=f[22:])
        calls.setdefault(c, {"hdr": "", "rows": []})["rows"].append(rec)
    return calls

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--j", type=float, nargs=3, required=True); ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--rj", type=float, default=2.0)
    a = ap.parse_args(); J = np.array(a.j)
    for spec in a.arm:
        lab, f = spec.split("=", 1); C = parse(f)
        best = None
        for c in sorted(C):
            P = np.array([r["p"] for r in C[c]["rows"]]) if C[c]["rows"] else np.zeros((0, 3))
            if len(P) and np.linalg.norm(P - J, axis=1).min() <= a.rj: best = c
        if best is None: print(f"== {lab}: no call near J"); continue
        rows = C[best]["rows"]; P = np.array([r["p"] for r in rows]); d = np.linalg.norm(P - J, axis=1)
        sel = np.where(d <= a.rj)[0]; sel = sel[np.argsort(d[sel])]
        S = lambda k, t: sum(rows[i][k][t] for i in sel)
        print(f"== {lab} {C[best]['hdr']} ; near-J {len(sel)} positions: sum dQ {sum(rows[i]['pos'] for i in sel):.0f} ; "
              f"sum local_dx {sum(rows[i]['dx'] for i in sel):.2f} ; reg u/v/w {sum(rows[i]['reg'][0] for i in sel)}/{sum(rows[i]['reg'][1] for i in sel)}/{sum(rows[i]['reg'][2] for i in sel)} ; "
              f"rows U/V/W {S('u',0)}/{S('v',0)}/{S('w',0)} ; sumR U/V/W {S('u',1):.3g}/{S('v',1):.3g}/{S('w',1):.3g} ; b U/V/W {S('u',2):.3g}/{S('v',2):.3g}/{S('w',2):.3g}")
        for i in sel:
            r = rows[i]
            print(f"   d={d[i]:.2f} i={r['i']:4d} dx={r['dx']:.3f} reg={r['reg']} U({r['u'][0]},{r['u'][1]:.3g},{r['u'][2]:.3g}) V({r['v'][0]},{r['v'][1]:.3g},{r['v'][2]:.3g}) W({r['w'][0]},{r['w'][1]:.3g},{r['w'][2]:.3g}) F={r['fd']:.3g} dQ={r['pos']:.0f} conn={r['nconn']} {' '.join(r['conn'])}")

if __name__ == "__main__":
    main()
