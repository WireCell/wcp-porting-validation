#!/usr/bin/env python3
"""doc qlmatch/32 sec 2 -- C++ OpHitFinder hit PE vs decon_roi PE per cathode rail run, for the four frame-dumped
18-event light arms: _f31ts18 (twoside, short), _f31tot18 (ToT, short), _f32si18 (twoside, int_samples),
_f32ti18 (ToT, int_samples).  Also checks that the int_samples C++ hits equal the Python replica's "int" rule
(d32_hit_emul.py) channel by channel.

    cd pdvd/docs/qlmatch/scripts && python3 d32_hit_closure.py > ../d32/hit_closure.txt
"""
import io
import tarfile

import numpy as np

import d31_flash_pe as F
import d32_hit_emul as E
from pd_mapping_audit import rail_intervals

EVTS = [298567 + 14 * i for i in range(18)]
ARMS = [("_f31ts18", "twoside short"), ("_f31tot18", "tot short"), ("_f32si18", "twoside int"),
        ("_f32ti18", "tot int")]
W0, W1 = 50, 300
BINS = [(1, 20), (20, 71), (71, 141), (141, 260), (260, 5000)]


def frames(evt, suf):
    tf = tarfile.open(f"{F.PDVD}/work/039252_light{evt}{suf}/light-frames-cath-wct.tar.bz2")
    g = lambda n: np.load(io.BytesIO(tf.extractfile(f"{n}_{evt}.npy").read()))
    return g("frame_raw"), list(g("channels_raw")), g("frame_decon_roi"), list(g("channels_decon_roi")), g(
        "tickinfo_decon_roi")


def int_hits(row):
    wf = np.trunc(100.0 * row.astype(np.float64)).astype(np.int64)
    out = []
    for lo, hi in E.segments(wf):
        for p in E.sliding_window(wf[lo:hi]):
            for s in E.split_pulse(wf[lo:hi], p):
                if s["peak"] >= E.HIT_THRESHOLD:
                    out.append((s["t_start"] + lo, s["area"] / 100.0))
    return out


def main():
    res = {a: [] for a, _ in ARMS}
    nrep_bad = nrep = 0
    for evt in EVTS:
        for suf, _ in ARMS:
            R, CR, D, CD, ti = frames(evt, suf)
            H = F.load("039252", str(evt), suf)["ophits"]
            for r, c in enumerate(CR):
                row = D[CD.index(c)]
                h = H[H[:, 0] == c]
                st = np.round(h[:, 6] / 16 - ti[2]).astype(int)
                if suf.startswith("_f32"):
                    mine = int_hits(row)
                    nrep += 1
                    ok = (len(mine) == len(h) and np.array_equal(np.sort(st), np.sort([m[0] for m in mine]))
                          and np.allclose(np.sort(h[:, 5]), np.sort([m[1] for m in mine]), rtol=1e-9, atol=1e-9))
                    nrep_bad += not ok
                x = np.trunc(100.0 * row.astype(np.float64)) / 100.0
                for i, j in rail_intervals(R[r]):
                    lo, hi = max(0, i - W0), j + W1
                    sel = (st >= lo) & (st < hi)
                    res[suf].append((j - i, x[lo:hi].sum(), h[sel, 5].sum(), int(sel.sum())))
    print(f"# 18 events run 039252; C++ hits whose start is in [i-{W0}, j+{W1}) of each cathode rail run vs the "
          f"decon_roi sum there; median [16,84]")
    print(f"# int_samples C++ hits == Python replica int rule: {nrep - nrep_bad}/{nrep} channel-arms")
    for suf, lab in ARMS:
        a = np.array(res[suf])
        print(f"## {suf} ({lab})")
        for b0, b1 in BINS:
            m = (a[:, 0] >= b0) & (a[:, 0] < b1) & (a[:, 1] > 0)
            r = a[m, 2] / a[m, 1]
            print(f"  ToT {b0:>4}-{b1:<4} n {m.sum():4d}  hits/roi {np.median(r):.3f} [{np.percentile(r, 16):.3f}, "
                  f"{np.percentile(r, 84):.3f}]  nhit median {int(np.median(a[m, 3]))}")
        for t0 in (71, 141, 260):
            m = a[:, 0] >= t0
            print(f"  ToT >= {t0}: summed hit PE {a[m, 2].sum():.4g}  summed roi {a[m, 1].sum():.4g}  "
                  f"ratio {a[m, 2].sum() / a[m, 1].sum():.3f}")


if __name__ == "__main__":
    main()
