#!/usr/bin/env python3
"""doc qlmatch/32 sec 2 -- which hit-finder change recovers the railed-pulse PE.  Per cathode rail run of the 18
frame-dumped run-039252 events (light arms _f31ts18 = twoside, _f31tot18 = ToT), the OpHitFinder replica
(d32_hit_emul.py, closure-exact against the C++ ophits) is run under several rules and the hit PE whose start falls in
[i-W0, j+W1) is compared with the decon_roi sum there.

Rules: short = production (static_cast<short>, wraps above 327.67 PE/tick); int = the same algorithm on an
un-wrapped integer stream; int+merge = int plus the sat_hit_merge prototype (pre 2, post W1).

    cd pdvd/docs/qlmatch/scripts && python3 d32_hit_rules.py > ../d32/hit_rules.txt
"""
import io
import sys
import tarfile

import numpy as np

import d31_flash_pe as F
import d32_hit_emul as E
from pd_mapping_audit import rail_intervals

EVTS = [298567 + 14 * i for i in range(18)]
W0, W1 = 50, 300
BINS = [(1, 20), (20, 71), (71, 141), (141, 260), (260, 5000)]


def frames(evt, suf):
    tf = tarfile.open(f"{F.PDVD}/work/039252_light{evt}{suf}/light-frames-cath-wct.tar.bz2")
    g = lambda n: np.load(io.BytesIO(tf.extractfile(f"{n}_{evt}.npy").read()))
    return g("frame_raw"), g("channels_raw"), g("frame_decon_roi"), list(g("channels_decon_roi"))


def run_rules(row, runs):
    x = np.trunc(100.0 * row.astype(np.float64)).astype(np.int64)
    out = {}
    for name in ("short", "int", "int+merge"):
        if name == "short":
            wf, subs = E.subpulses(row)
        else:
            wf = x
            subs = []
            for lo, hi in E.segments(wf):
                for p in E.sliding_window(wf[lo:hi]):
                    for s in E.split_pulse(wf[lo:hi], p):
                        s = dict(s)
                        for k in ("t_start", "t_end", "t_max"):
                            s[k] += lo
                        subs.append(s)
            if name == "int+merge":
                subs = E.merge_saturated(wf, subs, runs, 2, W1)
        out[name] = [s for s in subs if s["peak"] >= E.HIT_THRESHOLD]
    return x, out


def main():
    rows = []
    nwrap = {"_f31ts18": 0, "_f31tot18": 0}
    for evt in EVTS:
        for suf in ("_f31ts18", "_f31tot18"):
            R, CR, D, CD = frames(evt, suf)
            for r, c in enumerate(CR):
                runs = rail_intervals(R[r])
                if not runs:
                    continue
                row = D[CD.index(c)]
                nwrap[suf] += int(np.sum(np.abs(np.trunc(100.0 * row.astype(np.float64))) > 32767))
                x, hits = run_rules(row, runs)
                for i, j in runs:
                    lo, hi = max(0, i - W0), j + W1
                    roi = x[lo:hi].sum() / 100.0
                    rec = [evt, suf, c, j - i, roi, float(np.abs(x[i:j + W1]).max()) / 100.0]
                    for name in ("short", "int", "int+merge"):
                        hs = [h for h in hits[name] if lo <= h["t_start"] < hi]
                        rec += [sum(h["area"] for h in hs) / 100.0, len(hs)]
                    rows.append(rec)
    print(f"# 18 events run 039252; window [i-{W0}, j+{W1}); samples with |100*decon| > 32767 (short wraps): "
          f"twoside {nwrap['_f31ts18']}, tot {nwrap['_f31tot18']}")
    print("# per rail run: hits/roi median [16,84]; nhit median; peak PE/tick median")
    for suf, lab in (("_f31ts18", "twoside"), ("_f31tot18", "tot")):
        print(f"## {lab}")
        print(f"{'ToT':>9} {'n':>5} {'peak/tick':>9} | {'short':>24} {'int':>24} {'int+merge':>24} | nhit s/i/m")
        for b0, b1 in BINS:
            rr = [r for r in rows if r[1] == suf and b0 <= r[3] < b1 and r[4] > 0]
            if not rr:
                continue
            f = lambda k: np.array([r[k] / r[4] for r in rr])
            q = lambda a: f"{np.median(a):.3f} [{np.percentile(a, 16):.3f},{np.percentile(a, 84):.3f}]"
            nh = lambda k: int(np.median([r[k] for r in rr]))
            print(f"{b0:>4}-{b1:<4} {len(rr):5d} {np.median([r[5] for r in rr]):9.1f} | {q(f(6)):>24} {q(f(8)):>24} "
                  f"{q(f(10)):>24} | {nh(7)}/{nh(9)}/{nh(11)}")
        for t0 in (71, 141):
            rr = [r for r in rows if r[1] == suf and r[3] >= t0]
            print(f"ToT >= {t0}: summed hit PE short {sum(r[6] for r in rr):.4g}  int {sum(r[8] for r in rr):.4g}  "
                  f"int+merge {sum(r[10] for r in rr):.4g}  roi {sum(r[4] for r in rr):.4g}")


if __name__ == "__main__":
    main()
