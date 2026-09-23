#!/usr/bin/env python3
"""doc qlmatch/31 -- flash time map twoside light -> tot light, for ql_agree_score.py --truth-time-map.

Flash time is the PE-weighted mean of its hits' peak times (OpFlashFinder), so recovering a railed channel's PE into
one hit at the pulse peak (tot) moves a bright railed flash earlier by ~1 us -- past the scorer's 0.5 us truth join.
The physical flash is identified across the two arms by its hits on UNCHANGED channels: every hit row (channel,
start time, pe) present in both arms' ophits; each A flash maps to the B flash holding most of its shared-hit PE.
Times are taken from the two arms' calib dumps (a calib flash's "id" is its light-archive index).  Output {"events": {evt: [[t_A, t_B], ...]}} (only pairs that moved > 1 ns).

    cd pdvd && python3 docs/qlmatch/scripts/d31_time_map.py --a _g31off --b _q31tot --calib-tag q31ctl --calib-tag-b q31tot \
        --work-root /home/xqian/tmp/p31/wroot --out docs/qlmatch/d31/time_map_ctl_to_tot.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d31_flash_pe as F        # noqa: E402

RUN, EVT0, STEP, NEVT = "039252", 298567, 14, 18
ABSORB_US = 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="_g31off")
    ap.add_argument("--b", default="_q31tot")
    ap.add_argument("--calib-tag", default="q31ctl")
    ap.add_argument("--calib-tag-b", default="q31tot")
    ap.add_argument("--work-root", default="work")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, moved, unmatched, absorbed = {}, [], 0, []
    for idx in range(NEVT):
        evt = EVT0 + STEP * idx
        A, B = F.load(RUN, str(evt), a.a), F.load(RUN, str(evt), a.b)
        ha, hb = A["ophits"], B["ophits"]
        key = lambda h: (int(h[0]), round(float(h[6]), 1), round(float(h[5]), 2))
        bflash = {key(h): int(h[7]) for h in hb}
        acc = defaultdict(lambda: defaultdict(float))
        for h in ha:
            if int(h[7]) < 0:
                continue
            fb = bflash.get(key(h))
            if fb is not None and fb >= 0:
                acc[int(h[7])][fb] += h[5]
        ta, tb = A["opflash"][:, 0] / 1e3, B["opflash"][:, 0] / 1e3      # us, light frame
        # calib flash "id" = light-archive flash index; its "time" is the calib-frame time the scan truth uses
        cal = {}
        for tag, arm in ((a.calib_tag, "A"), (a.calib_tag_b, "B")):
            c = json.load(open(os.path.join(a.work_root, f"{RUN}_{idx}_{tag}", f"calib-evt{evt}.json")))
            cal[arm] = {int(f["id"]): f["time"] for f in c["flashes"]}
        pairs = []
        for fa in range(len(ta)):
            if fa not in acc:
                # no hit survived unchanged (all its PE was on railed channels, e.g. a flash split off a railed
                # pulse by twoside's fragmented hits): map it to the B flash that absorbed it -- the nearest B flash
                # starting at most ABSORB_US earlier (doc 26's late-member -> seed rule), else leave it unmapped
                cand = np.flatnonzero((tb <= ta[fa] + 1e-3) & (tb >= ta[fa] - ABSORB_US))
                if not len(cand):
                    unmatched += 1
                    continue
                fb = int(cand[np.argmax(tb[cand])])
                absorbed.append(tb[fb] - ta[fa])
            else:
                fb = max(acc[fa].items(), key=lambda kv: kv[1])[0]
            d = tb[fb] - ta[fa]
            if abs(d) > 1e-3:
                if fa in cal["A"] and fb in cal["B"]:
                    pairs.append([cal["A"][fa], cal["B"][fb]])
                moved.append(d)
        res[str(evt)] = pairs
    json.dump({"comment": "doc qlmatch/31 flash time map %s -> %s (calib frame, us)" % (a.a, a.b), "events": res},
              open(a.out, "w"), indent=0)
    m = np.array(moved)
    print(f"flashes moved: {len(m)}; dt median {np.median(m):+.3f} us, [16,84] [{np.percentile(m, 16):+.3f}, "
          f"{np.percentile(m, 84):+.3f}]; |dt| > 0.5 us: {int((np.abs(m) > 0.5).sum())}; A flashes without "
          f"shared hits and no absorbing flash: {unmatched}; mapped as absorbed into an earlier flash: {len(absorbed)} "
          f"(dt median {np.median(absorbed) if absorbed else 0:+.3f} us)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
