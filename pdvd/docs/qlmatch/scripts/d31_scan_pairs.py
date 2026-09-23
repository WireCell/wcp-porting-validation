#!/usr/bin/env python3
"""doc qlmatch/31 -- paired hand-scan movers between the control and a ToT arm (work/ql_scores/<tag>/scores.json),
keyed on (event, cluster uid) so a retimed flash does not count as a mover; exact two-sided sign test on newly
missed vs recovered; and the movers split by whether the control flash nearest the truth time is cathode-railed.

    cd pdvd && python3 docs/qlmatch/scripts/d31_scan_pairs.py --base q31ctl --arm q31tot2 --arm q31tot2m \
        --work-root /home/xqian/tmp/p31/wroot
"""
import argparse
import json
import math
import os

PDVD = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))


def sign_p(k, n):
    if not n:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="q31ctl")
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--work-root", required=True)
    a = ap.parse_args()
    S = lambda t: json.load(open(os.path.join(PDVD, "work/ql_scores", t, "scores.json")))
    b = S(a.base)
    for arm in a.arm:
        d = S(arm)
        tb, ta = b["total"], d["total"]
        print(f"# {a.base} -> {arm}: agree {tb['agree']}->{ta['agree']}  phantom {tb['phantom']}->{ta['phantom']}  "
              f"missed {tb['missed']}->{ta['missed']}  unknown {tb['unknown']}->{ta['unknown']}  "
              f"flash-cut misses {tb['missed_flash_cut']}->{ta['missed_flash_cut']}")
        for k in ("missed_list", "phantom_list"):
            sb = {(e, x["uid"]) for e, v in b["detail"].items() for x in v[k]}
            sa = {(e, x["uid"]) for e, v in d["detail"].items() for x in v[k]}
            new, gone = sa - sb, sb - sa
            print(f"  {k[:-5]:8s}: newly {len(new):3d}, recovered {len(gone):3d}, common {len(sa & sb):3d}; "
                  f"sign test p = {sign_p(len(new), len(new) + len(gone)):.3g}")
        cnt = {"newly missed": [0, 0], "recovered": [0, 0]}
        for i in range(18):
            e = 298567 + 14 * i
            k = f"evt{e}"
            fl = json.load(open(os.path.join(a.work_root, f"039252_{i}_{a.base}", f"calib-evt{e}.json")))["flashes"]

            def railed(t):
                f = min(fl, key=lambda f: abs(f["time"] - t))
                return int(abs(f["time"] - t) < 3 and any(s > 0 for s in f["sat"][4:12]))
            mb = {m["uid"]: m["time"] for m in b["detail"][k]["missed_list"]}
            ma = {m["uid"]: m["time"] for m in d["detail"][k]["missed_list"]}
            for u in set(ma) - set(mb):
                cnt["newly missed"][railed(ma[u])] += 1
            for u in set(mb) - set(ma):
                cnt["recovered"][railed(mb[u])] += 1
        for k, (n0, n1) in cnt.items():
            print(f"  {k:12s} by flash class: cathode-railed {n1}, not railed {n0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
