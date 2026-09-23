#!/usr/bin/env python3
"""doc qlmatch/31 -- where ToT and twoside differ on real rail runs (cathode full streams, run 039252 file 0).

For every merged rail run (gap <= 2, the OpDecon tot_merge_gap) on the 16 cathode channels: ToT, the model depth
1/lambda(ToT), and the in-run fill area (sum of fill - pedestal over the run) of twoside (per sub-run, as OpDecon
repair_runs) and of the ToT fill (as OpDecon tot_fill_run, v2 shapes).  Pedestal = mean of the first 20 samples of
the stream (OpDecon).  Also: runs twoside leaves CLIPPED (fewer than 2 positive exit samples) and runs ToT hands back
to twoside (edge / wider than the table).  Python replica only; the C++ is pinned to the same Python by the doctest.

    cd pdvd/docs/qlmatch && python3 scripts/d31_run_census.py > d31/run_census.txt
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import saturation_recovery_study as S                               # noqa: E402
import saturation_tot_study as T                                    # noqa: E402
from pd_mapping_audit import RAWWF, RAIL, load_cathode_streams, rail_intervals   # noqa: E402

V2 = S._find_templates().replace("pdvd-spe-templates.json", "pdvd-spe-templates-v2.json")
S._find_templates = lambda: V2
GAP = 2
NPED = 20


def main():
    import uproot
    chd = S.load_channels()
    z = np.load("/home/xqian/tmp/sat_tot_v2/shape.npz")
    shp = {int(c): T.Shape(chd[int(c)], p) for c, p in zip(z["chan"], z["par"])}
    rf = sorted(glob.glob(os.path.join(RAWWF, "*_rawwf.root")))[0]
    evs = sorted(set(uproot.open(rf)["rawdump/raw_waveform"].arrays(["event"], library="np")["event"]))
    rows = []   # tot, depth, twoside_area, tot_area, ts_clipped_subruns, nsub, tot_fallback
    for ev in evs:
        (waves, _), _ = load_cathode_streams(rf, ev)
        for c, w in sorted(waves.items()):
            if c not in shp:
                continue
            w = np.asarray(w, np.float64)
            ped = w[:NPED].mean()
            runs = rail_intervals(w)
            k = 0
            while k < len(runs):
                m = k + 1
                while m < len(runs) and runs[m][0] - runs[m - 1][1] <= GAP:
                    m += 1
                i, j = runs[k][0], runs[m - 1][1]
                # twoside per sub-run (repair_runs on the group's samples only)
                ts = S.repair_runs(w[max(0, i - 8):min(len(w), j + 8)], RAIL, chd[c], ped, "twoside")
                o = i - max(0, i - 8)
                ts_area = float((ts[o:o + (j - i)] - ped).sum())
                clipped = 0
                for a, b in runs[k:m]:
                    y = w[b:b + 8] - ped
                    clipped += int((y > 0).sum() < 2)
                tot = j - i
                s = shp[c]
                fb = (i == 0) or (j == len(w)) or tot >= s.w[0]
                lam = s.level_for(tot)
                R = RAIL + 0.5 - ped
                k0 = s.u[min(len(s.u) - 1, int(np.searchsorted(s.lam, lam)))]
                kk = k0 + np.arange(tot)
                fill = (R / lam) * np.where(kk < len(s.s), s.s[np.clip(kk, 0, len(s.s) - 1)], 0.0)
                tt_area = float(np.maximum(w[i:j] - ped, fill).sum()) if not fb else ts_area
                # replica decon window PE (doc 30 methods_pe) around the run's peak, twoside vs tot_fill
                seg, s0 = T.seg_at(w, i)
                pk = i - s0 + int(np.argmax(w[i:j])) if j > i else i - s0
                o, _ = T.methods_pe(seg, chd[c], ped, pk, RAIL - 0.5, s, None, c)
                rows.append((tot, 1.0 / lam, ts_area, tt_area, clipped, m - k, int(fb), o["twoside"], o["tot_fill"]))
                k = m
    r = np.array(rows)
    print(f"# {len(r)} merged rail runs, run 039252 file 0 ({len(evs)} events), 16 cathode channels")
    print(f"# runs with >1 sub-run (merged across gaps <= {GAP}): {int((r[:, 5] > 1).sum())}; "
          f"twoside leaves >=1 sub-run CLIPPED: {int((r[:, 4] > 0).sum())}; ToT falls back to twoside: {int(r[:, 6].sum())}")
    print(f"{'ToT bin':>12} {'depth':>6} {'n':>6} {'ts clipped':>10} | ToT/twoside in-run fill area: median [16,84]  p95   | replica decon")
    for lo, hi in ((1, 5), (5, 20), (20, 40), (40, 71), (71, 108), (108, 141), (141, 194), (194, 260), (260, 2000)):
        sel = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if not sel.any():
            continue
        q = r[sel, 3] / np.maximum(r[sel, 2], 1e-9)
        p = np.percentile(q, [16, 50, 84, 95])
        print(f"{lo:>5}-{hi:<6} {np.median(r[sel, 1]):6.2f} {int(sel.sum()):>6} {int((r[sel, 4] > 0).sum()):>10} | "
              f"{p[1]:.3f} [{p[0]:.3f}, {p[2]:.3f}]  {p[3]:.3f}   | window PE tot_fill/twoside {np.median(r[sel, 8] / r[sel, 7]):.3f} "
              f"[{np.percentile(r[sel, 8] / r[sel, 7], 16):.3f}, {np.percentile(r[sel, 8] / r[sel, 7], 84):.3f}]")
    sel = r[:, 4] > 0
    if sel.any():
        q = r[sel, 3] / np.maximum(r[sel, 2], 1e-9)
        print(f"runs twoside left clipped: ToT/twoside fill area median {np.median(q):.3f}, p84 {np.percentile(q, 84):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
