#!/usr/bin/env python3
"""doc qlmatch/31 sec 5 -- OpHitFinder PE vs decon_roi PE per cathode rail run, twoside vs tot, the 18 run-039252
events dumped with PDVD_LIGHT_FRAMES=1 (light arms _f31ts18 / _f31tot18, otherwise = _g31off / _q31tot2).

Per rail run on the raw cathode stream: decon_roi sum over [i-W0, j+W1) and the PE of the hits whose start falls in
that window (hit tick = start_time/16 - tbin, tbin from the frame's tickinfo), plus the hit count.  A hit-level loss
shows as hits/roi < 1.

    cd pdvd && python3 docs/qlmatch/scripts/d31_hit_split18.py [W0 W1] > docs/qlmatch/d31/hit_split18.txt
"""
import io
import os
import sys
import tarfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d31_flash_pe as F                                   # noqa: E402
from pd_mapping_audit import rail_intervals                # noqa: E402

PDVD = F.PDVD
EVTS = [298567 + 14 * i for i in range(18)]
W0 = int(sys.argv[1]) if len(sys.argv) > 1 else 50
W1 = int(sys.argv[2]) if len(sys.argv) > 2 else 300


def frames(evt, suf):
    tf = tarfile.open(os.path.join(PDVD, "work", f"039252_light{evt}{suf}", "light-frames-cath-wct.tar.bz2"))
    g = lambda n: np.load(io.BytesIO(tf.extractfile(f"{n}_{evt}.npy").read()))
    return {k: g(k) for k in ("frame_raw", "channels_raw", "frame_decon_roi", "channels_decon_roi", "tickinfo_raw")}


def main():
    rows = []
    for evt in EVTS:
        FA, FB = frames(evt, "_f31ts18"), frames(evt, "_f31tot18")
        off = int(FA["tickinfo_raw"][2])
        HA = F.load("039252", str(evt), "_f31ts18")["ophits"]
        HB = F.load("039252", str(evt), "_f31tot18")["ophits"]
        for r, c in enumerate(FA["channels_raw"]):
            raw = FA["frame_raw"][r]
            ra = FA["frame_decon_roi"][list(FA["channels_decon_roi"]).index(c)]
            rb = FB["frame_decon_roi"][list(FB["channels_decon_roi"]).index(c)]
            ha, hb = HA[HA[:, 0] == c], HB[HB[:, 0] == c]
            ta, tb = ha[:, 6] / 16 - off, hb[:, 6] / 16 - off
            for i, j in rail_intervals(raw):
                lo, hi = max(0, i - W0), j + W1
                sa, sb = ha[(ta >= lo) & (ta < hi)], hb[(tb >= lo) & (tb < hi)]
                rows.append((evt, c, j - i, ra[lo:hi].sum(), rb[lo:hi].sum(), sa[:, 5].sum(), sb[:, 5].sum(),
                             len(sa), len(sb)))
    r = np.array(rows, float)
    n0 = len(r)
    r = r[(r[:, 3] > 0) & (r[:, 4] > 0)]          # a ratio needs a positive ROI sum in both arms
    f = lambda x: f"{np.median(x):.3f} [{np.percentile(x, 16):.3f}, {np.percentile(x, 84):.3f}]"
    print(f"# 18 events run 039252, {n0} cathode rail runs ({n0 - len(r)} dropped: ROI sum <= 0 in an arm); "
          f"window [i-{W0}, j+{W1}); ratios median [16,84]")
    print(f"{'ToT':>9} {'n':>5} | {'hits/roi twoside':>26} {'hits/roi tot':>26} | {'nhit ts':>7} {'nhit tot':>8} | "
          f"{'roi tot/ts':>26} {'hits tot/ts':>26}")
    for lo, hi in ((1, 20), (20, 71), (71, 141), (141, 260), (260, 5000)):
        s = (r[:, 2] >= lo) & (r[:, 2] < hi)
        if not s.any():
            continue
        print(f"{lo:>4}-{hi:<4} {int(s.sum()):>5} | {f(r[s, 5] / r[s, 3]):>26} {f(r[s, 6] / r[s, 4]):>26} | "
              f"{np.median(r[s, 7]):>7.0f} {np.median(r[s, 8]):>8.0f} | {f(r[s, 4] / r[s, 3]):>26} "
              f"{f(r[s, 6] / np.maximum(r[s, 5], 1e-9)):>26}")
    for lo in (71, 141):
        s = r[:, 2] >= lo
        print(f"ToT >= {lo}: hit PE summed over runs  twoside {r[s, 5].sum():.4g}  tot {r[s, 6].sum():.4g}  "
              f"(twoside/tot {r[s, 5].sum() / r[s, 6].sum():.3f})")
        print(f"ToT >= {lo}: sum over runs  roi twoside {r[s, 3].sum():.4g}  hits twoside {r[s, 5].sum():.4g} "
              f"({r[s, 5].sum() / r[s, 3].sum():.3f})  |  roi tot {r[s, 4].sum():.4g}  hits tot {r[s, 6].sum():.4g} "
              f"({r[s, 6].sum() / r[s, 4].sum():.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
