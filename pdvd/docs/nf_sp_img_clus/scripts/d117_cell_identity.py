#!/usr/bin/env python3
"""doc pdvd/117 -- are the regenerated SP frames (work/<evt>_d117sp) the frames p98vonq's PR chain read?

    python3 d117_cell_identity.py 039252_0 [039252_1 ...]

Reference: T_stm_michel_2d of work/<evt>_p98vonq/tracking-pr.root, every cell of every plane.  Each cell's `charge`
is the sum of frame_gauss over its 4-tick slice (doc 98 sec 2.1, verified there on the original frames), so the
regenerated frames are the same frames iff every cell re-sums to its stored charge.  Only roles 1/3/4 (muon, Michel,
gamma -- the cells the crops use) are checked: role-0 region rows repeat one charge on consecutive ticks, so their `time`
is not a slice start (on 039252_0 all 771 non-matching rows are role 0; roles 1/3/4 match 4810/4810).  Rows with
flag == 0 are not frame-measured either: on 039253_5 / 039349_27 every non-matching role-3 row is flag 0, on an
induction plane, on a channel whose whole frame row is empty (a charge the frame never held); they are counted apart.  Per event: cells, cells within
1e-4 relative, the worst relative difference.
"""
import sys
import numpy as np
import uproot
import d98_michel_crops as C

WORK = C.IMG + "/pdvd/work"


def check(evt):
    f = uproot.open(f"{WORK}/{evt}_p98vonq/tracking-pr.root")
    if "T_stm_michel_2d" not in f:
        return evt, 0, 0, float("nan")
    c = f["T_stm_michel_2d"].arrays(["apa", "channel", "time", "charge", "role", "flag"], library="np")
    n, ok, worst = 0, 0, 0.0
    for apa in np.unique(c["apa"]):
        chans, fr, tbin = C.load_gauss(f"{WORK}/{evt}_d117sp/" + C.FRAME_PFX["pdvd"] % apa)
        row = {ch: i for i, ch in enumerate(chans)}
        s = (c["apa"] == apa) & np.isin(c["role"], (1, 3, 4)) & (c["flag"] == 1)
        for ch, t, q in zip(c["channel"][s], c["time"][s], c["charge"][s]):
            if int(ch) not in row:
                continue
            t0 = int(t) - tbin
            v = float(fr[row[int(ch)], t0:t0 + C.TICKS_PER_SLICE].sum())
            d = abs(v - q) / max(abs(q), 1.0)
            n += 1; ok += d < 1e-4; worst = max(worst, d)
    return evt, n, ok, worst


if __name__ == "__main__":
    bad = 0
    for e in sys.argv[1:]:
        evt, n, ok, w = check(e)
        print(f"{evt}: cells {n}, identical (rel < 1e-4) {ok}, worst rel diff {w:.2e}  {'PASS' if n and ok == n else 'FAIL'}")
        bad += not (n and ok == n)
    print("ALL PASS" if not bad else f"{bad} event(s) FAIL")
    sys.exit(1 if bad else 0)
