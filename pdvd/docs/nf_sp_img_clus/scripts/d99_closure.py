#!/usr/bin/env python3
"""doc pdvd/99 sec 7 -- split the OFF -> ON plateau change into "same track, more charge" and "different tracks".

The dQ/dx plateau rows (d99_dqdx_drift.py) of the two arms are keyed by each arm's own cluster ids.  The carried
records give every new key its original record key (carried_from), so a track in both samples is joined through it.

  python3 d99_closure.py [--off p98voffq --on p98vonq] [--dq /home/xqian/tmp/p98/dq]

Prints, per volume: tracks in both samples / only OFF / only ON; the per-track ON/OFF plateau ratio (median, bootstrap
68 % interval); and the top/bottom ratio of median plateaus on the common tracks and on each full sample.
"""
import argparse
import json
import os
import random
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.normpath(os.path.join(HERE, "..", "..", "scan"))


def rows(dq, arm, carried):
    d = json.load(open(os.path.join(dq, f"dqdx_{arm}.json")))["pdvd"]
    back = {it["key"]: it["carried_from"] for it in carried}
    out = {}
    for r in d:
        orig = back.get(r["key"])
        if orig is None:
            raise SystemExit(f"{arm}: plateau row {r['key']} has no carried item")
        if orig in out:
            raise SystemExit(f"{arm}: two plateau rows carry from {orig}")
        out[orig] = r
    return out


def boot(vals, fn, n=1000, seed=99):
    rnd = random.Random(seed)
    xs = sorted(fn([rnd.choice(vals) for _ in vals]) for _ in range(n))
    return xs[int(0.16 * n)], xs[int(0.84 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", default="p98voffq")
    ap.add_argument("--on", default="p98vonq")
    ap.add_argument("--dq", default="/home/xqian/tmp/p98/dq")
    ap.add_argument("--write-common", default=None, metavar="DIR",
                    help="also write dqdx_<arm>_common.json (common tracks only) for d29_space_charge.py")
    a = ap.parse_args()
    car = {arm: json.load(open(os.path.join(SCAN, f"pdvd_stm_michel_smx9_carried_{arm}.json"))) for arm in (a.off, a.on)}
    R = {arm: rows(a.dq, arm, car[arm]) for arm in (a.off, a.on)}
    off, on = R[a.off], R[a.on]
    common = sorted(set(off) & set(on))
    print(f"plateau rows: {a.off} {len(off)}  {a.on} {len(on)}  common by original key {len(common)}")
    for side, vol in (("x>0", "top"), ("x<0", "bottom")):
        c = [k for k in common if off[k]["side"] == side]
        side_flip = [k for k in common if off[k]["side"] != on[k]["side"]]
        only_off = [k for k in off if k not in on and off[k]["side"] == side]
        only_on = [k for k in on if k not in off and on[k]["side"] == side]
        rat = [on[k]["plat"] / off[k]["plat"] for k in c]
        lo, hi = boot(rat, st.median)
        print(f"  {vol:6s} common {len(c):3d}  only {a.off} {len(only_off):3d}  only {a.on} {len(only_on):3d}  "
              f"(side changed on {len(side_flip)})")
        print(f"         per-track {a.on}/{a.off} plateau: median {st.median(rat):.4f} [{lo:.4f}, {hi:.4f}]  "
              f"p16/p84 of tracks {sorted(rat)[int(0.16*len(rat))]:.4f}/{sorted(rat)[int(0.84*len(rat))]:.4f}")
        for name, ks, src in ((f"only {a.off}", only_off, off), (f"only {a.on}", only_on, on)):
            if ks:
                print(f"         {name}: median plateau {st.median(src[k]['plat'] for k in ks):.4f} (n {len(ks)})")

    def tb(src, keys):
        t = [src[k]["plat"] for k in keys if src[k]["side"] == "x>0"]
        b = [src[k]["plat"] for k in keys if src[k]["side"] == "x<0"]
        return st.median(t), st.median(b), len(t), len(b)

    print("  top/bottom ratio of median plateaus:")
    for label, src, keys in ((f"{a.off} full", off, list(off)), (f"{a.on} full", on, list(on)),
                             (f"{a.off} common", off, common), (f"{a.on} common", on, common)):
        t, b, nt, nb = tb(src, keys)
        print(f"    {label:18s} top {t:.4f} (n {nt})  bottom {b:.4f} (n {nb})  top/bottom {t/b:.4f}")

    if a.write_common:
        # the same json layout d29_space_charge.py reads, restricted to the common tracks (pdhd rows copied unchanged)
        for arm, src in ((a.off, off), (a.on, on)):
            d = json.load(open(os.path.join(a.dq, f"dqdx_{arm}.json")))
            keep = {src[k]["key"] for k in common}
            d["pdvd"] = [r for r in d["pdvd"] if r["key"] in keep]
            p = os.path.join(a.write_common, f"dqdx_{arm}_common.json")
            json.dump(d, open(p, "w"), indent=1)
            print(f"  wrote {p} ({len(d['pdvd'])} pdvd rows)")


if __name__ == "__main__":
    main()
