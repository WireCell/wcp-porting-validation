#!/usr/bin/env python3
"""Full lambda characterisation for the 4-event flag-clean retune: not KS alone
(which a too-diffuse lambda can game by smearing the CDF) but the doc's criteria
together -- matcher-KS, N90 concentration ratio (pred/meas), top-3 direct-PMT
scale, full-side integral scale, dark-PMT fraction -- over an EXTENDED grid, to
see whether KS turns over or rides to the MAXPD ceiling.  Reuses build_anchors /
predict from fit_labels_multi."""
import statistics, sys, os
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from fit_labels_multi import build_anchors, calc_ks, CUR_EFF

GRID = [100, 150, 200, 250, 300, 350, 400, 500, 600, 800, 1200, 2000]


def n90(v):
    s = sum(v)
    if s <= 0: return 0
    c = n = 0
    for x in sorted(v, reverse=True):
        c += x; n += 1
        if c >= 0.9 * s: break
    return n


def _init():
    import repredict; repredict.VUVEFF = CUR_EFF
    global predict
    from repredict import predict


def one(a):
    rows = {}
    meas = [a["pe160"][i] for i in a["valid"]]
    N90m = n90(meas); o3 = sorted(range(len(meas)), key=lambda k: -meas[k])[:3]
    for lam in GRID:
        pf = predict(a["pts"], a["geom"], a["offset"], float(lam))
        prm = [0.0 if i in a["mask"] else pf[i] for i in range(160)]
        ks = calc_ks(a["pe160"], prm)
        p = [pf[i] for i in a["valid"]]; sp = sum(p)
        sp3 = sum(p[k] for k in o3)
        direct = (sum(meas[k] for k in o3) / sp3) if sp3 > 0 else 0
        full = (sum(meas) / sp) if sp > 0 else 0
        dark = (sum(p[k] for k in range(len(meas)) if meas[k] == 0) / sp) if sp > 0 else 0
        rows[lam] = (ks, (n90(p) / N90m) if N90m else 0, direct, full, dark)
    return a["apa"], rows


def med(xs): return statistics.median(xs) if xs else float("nan")


if __name__ == "__main__":
    anchors, _ = build_anchors()
    with Pool(8, initializer=_init) as pool:
        res = pool.map(one, anchors)
    print("4-event flag-clean anchors=%d (+x=%d -x=%d)"
          % (len(res), sum(1 for r in res if r[0] == 1), sum(1 for r in res if r[0] == 0)))
    print("\n  lam   medKS   N90rat  direct(+x)  full(+x)  dark%%   direct(all)")
    for lam in GRID:
        ks = [r[1][lam][0] for r in res]
        n90r = [r[1][lam][1] for r in res if r[1][lam][1] > 0]
        dpx = [r[1][lam][2] for r in res if r[0] == 1 and r[1][lam][2] > 0]
        fpx = [r[1][lam][3] for r in res if r[0] == 1 and r[1][lam][3] > 0]
        dk = [r[1][lam][4] * 100 for r in res]
        dall = [r[1][lam][2] for r in res if r[1][lam][2] > 0]
        print("  %4d  %.4f  %.2f     %.3f      %.3f     %.1f    %.3f"
              % (lam, med(ks), med(n90r), med(dpx), med(fpx), med(dk), med(dall)))
    print("\nRead: N90rat~1 => pred concentration matches data; direct~full => spread right;")
    print("direct(+x) is the norm handle. KS-min alone can ride high (over-diffuse).")
