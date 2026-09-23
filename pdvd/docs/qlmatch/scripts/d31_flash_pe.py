#!/usr/bin/env python3
"""doc qlmatch/31 -- flash-level effect of saturation_repair_mode tot: light arm B vs A (default _g31off vs _q31tot).

Per event: flash counts; flashes matched by time (|dt| < 50 ns; column 0 is ns); on matched flashes, the per-(flash, OpDet) PE ratio
B/A for railed cathode cells (flash_sat set, OpDet 4..11) and for unrailed cells, and the total-PE ratio of flashes
with a cathode rail.  Also doc 11 sec 5's smoke flash (evt298567, the flash nearest gid37's time) od7.

    cd pdvd && python3 docs/qlmatch/scripts/d31_flash_pe.py [--a _g31off --b _q31tot]
"""
import argparse
import io
import os
import sys
import tarfile

import numpy as np

PDVD = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
CATH = np.arange(4, 12)


def load(run6, evt, suf):
    p = os.path.join(PDVD, "work", f"{run6}_light{evt}{suf}", "opflash_pdvd-wct.tar.gz")
    out = {}
    with tarfile.open(p) as tf:
        names = {m.name: m for m in tf.getmembers()}
        for n, m in names.items():
            if n.endswith("_metadata.json") and "_tensor_" in n:
                import json
                nm = json.load(tf.extractfile(m))["name"]
                arr = n.replace("_metadata.json", "_array.npy")
                out[nm] = np.load(io.BytesIO(tf.extractfile(names[arr]).read()))
    return out


def q(x):
    x = np.asarray(x)
    if not len(x):
        return "n 0"
    p = np.percentile(x, [16, 50, 84, 95, 99])
    return (f"n {len(x):6d}  median {p[1]:.3f} [16,84] [{p[0]:.3f}, {p[2]:.3f}]  p95 {p[3]:.3f}  p99 {p[4]:.3f}  "
            f"moved >1% {100 * np.mean(np.abs(x - 1) > 0.01):.1f}%  <0.99 {100 * np.mean(x < 0.99):.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="_g31off")
    ap.add_argument("--b", default="_q31tot")
    a = ap.parse_args()
    evs = [l.split() for l in open(os.path.join(PDVD, "stm/events.txt")) if l.strip() and not l.startswith("#")]
    nA = nB = nm = 0
    r_rail, r_clean, r_tot, pe_a, pe_b = [], [], [], [], []
    for run, idx, evt, *_ in evs:
        run6 = f"{int(run):06d}"
        A, B = load(run6, evt, a.a), load(run6, evt, a.b)
        ta, tb = A["opflash"][:, 0], B["opflash"][:, 0]
        nA += len(ta)
        nB += len(tb)
        for i, t in enumerate(ta):
            j = int(np.argmin(np.abs(tb - t))) if len(tb) else -1
            if j < 0 or abs(tb[j] - t) > 50.0:     # times in ns
                continue
            nm += 1
            pa, pb = A["opflash"][i, 1:], B["opflash"][j, 1:]
            sat = A["flash_sat"][i] > 0
            for k in CATH:
                if pa[k] > 0:
                    (r_rail if sat[k] else r_clean).append(pb[k] / pa[k])
            if sat[CATH].any():
                r_tot.append(pb.sum() / max(pa.sum(), 1e-9))
                pe_a.append(pa.sum())
                pe_b.append(pb.sum())
        if evt == "298567":
            i = int(np.argmax(A["opflash"][:, 1:][:, 7] * (A["flash_sat"][:, 7] > 0))) if (A["flash_sat"][:, 7] > 0).any() else None
            if i is not None:
                j = int(np.argmin(np.abs(tb - ta[i])))
                print(f"# evt298567 brightest od7-railed flash t={ta[i] / 1e3:.3f} us: od7 {a.a} {A['opflash'][i, 8]:.0f} -> "
                      f"{a.b} {B['opflash'][j, 8]:.0f}; total {A['opflash'][i, 1:].sum():.0f} -> {B['opflash'][j, 1:].sum():.0f}")
    print(f"# light {a.a} (A) vs {a.b} (B), {len(evs)} events")
    print(f"flashes: A {nA}, B {nB} (B-A {nB - nA:+d}); time-matched {nm}")
    print(f"railed cathode cells   B/A PE: {q(r_rail)}")
    print(f"unrailed cathode cells B/A PE: {q(r_clean)}")
    print(f"flashes with a cathode rail, total PE B/A: {q(r_tot)}; summed PE A {sum(pe_a):.4g} B {sum(pe_b):.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
