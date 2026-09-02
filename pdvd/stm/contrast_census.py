#!/usr/bin/env python3
"""doc pdvd/25 sec 13.6: where the STM-tagged passes die in collect_stm_sample's cuts.
Dumps every STM=1/TGM=0/status-0 pass with its contrast, end |x|, npts, chi2; prints the
contrast histogram and the fraction whose stop end lies in the excluded near-CRP band."""
import sys, os, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_stm_sample as C
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="stm2"); ap.add_argument("--out", default=None)
a = ap.parse_args()
rows = []
for bk in C.iter_blocks(a.tag):
    v = bk["verdict"]
    if v.get("stm") != 1 or v.get("tgm") == 1 or bk["status"] != 0: continue
    rr, dq, x = bk["rr"], bk["dqdx"], bk["x"]
    near_all = rr < 2; near = near_all & (dq > 0); mid = (rr >= 20) & (rr < 40) & (dq > 0)
    endx = float(np.median(np.abs(x[near_all]))) if near_all.sum() else float("nan")
    con = float(np.median(dq[near]) / np.median(dq[mid])) if near.sum() >= 3 and mid.sum() >= 5 else float("nan")
    # contrast with the near-CRP band NOT excluded (raw), for the same end
    rows.append((bk["event"], bk["cluster"], bk["npts"], float(rr.max()), bk["chi2"], endx, con, int(near_all.sum()), int(near.sum()), int(mid.sum())))
r = np.array([(t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9]) for t in rows], dtype=float)
n = len(rows); print(f"STM-tagged accepted passes: {n}")
endx = r[:, 3]; con = r[:, 4]
print(f"  stop end |x| > {C.MAX_ABS_X} cm (near-CRP band, contrast undefined): {int(np.sum(endx > C.MAX_ABS_X))}")
print(f"  stop end within 5 cm of a CRP (|x| > 335): {int(np.sum(endx > 335))}; end in the cathode gap |x| < 8: {int(np.sum(endx < 8))}")
ok = np.isfinite(con); print(f"  contrast defined: {int(ok.sum())}")
for lo, hi in [(0, 0.8), (0.8, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99)]:
    print(f"  contrast [{lo},{hi}): {int(np.sum(ok & (con >= lo) & (con < hi)))}")
print(f"  npts >= 40 and L >= 22 cm among contrast-defined: {int(np.sum(ok & (r[:,0] >= 40) & (r[:,1] >= 22)))}")
if a.out:
    with open(a.out, "w") as f:
        f.write("event\tcluster\tnpts\tL_cm\tchi2\tend_absx_cm\tcontrast\tn_near_all\tn_near_used\tn_mid\n")
        for t in rows: f.write("\t".join(str(round(v, 3)) if isinstance(v, float) else str(v) for v in t) + "\n")
    print("wrote", a.out)
