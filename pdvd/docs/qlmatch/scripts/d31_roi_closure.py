#!/usr/bin/env python3
"""doc qlmatch/31: decon vs decon_roi per cathode rail run, twoside vs tot, evt298567.  Run from scripts/."""
import sys, tarfile, io, numpy as np
sys.path.insert(0, '.')
from pd_mapping_audit import rail_intervals
W = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work/039252_light298567"
def ld(suf):
    tf = tarfile.open(f"{W}{suf}/light-frames-cath-wct.tar.bz2")
    g = lambda n: np.load(io.BytesIO(tf.extractfile(n).read()))
    return {k: g(f"{k}_298567.npy") for k in ("frame_raw", "channels_raw", "frame_decon", "channels_decon", "frame_decon_roi", "channels_decon_roi")}
A, B = ld("_f31ts18"), ld("_f31tot18")
rows = []
for r, c in enumerate(A["channels_raw"]):
    raw = A["frame_raw"][r]
    get = lambda X, k: X[f"frame_{k}"][int(np.where(X[f"channels_{k}"] == c)[0][0])]
    da, db, ra, rb = get(A, "decon"), get(B, "decon"), get(A, "decon_roi"), get(B, "decon_roi")
    for i, j in rail_intervals(raw):
        lo, hi = max(0, i - 50), min(len(da), j + 300)
        rows.append((c, j - i, da[lo:hi].sum(), db[lo:hi].sum(), ra[lo:hi].sum(), rb[lo:hi].sum(),
                     ra[max(0, i - 2000):j + 3000].sum(), rb[max(0, i - 2000):j + 3000].sum()))
r = np.array(rows)
f = lambda x: f"{np.median(x):.3f} [{np.percentile(x,16):.3f},{np.percentile(x,84):.3f}]"
for lo, hi in ((1, 20), (20, 71), (71, 141), (141, 260)):
    s = (r[:, 1] >= lo) & (r[:, 1] < hi)
    print(f"ToT {lo}-{hi} n {s.sum()}: decon tot/ts {f(r[s,3]/r[s,2])} | roi tot/ts {f(r[s,5]/r[s,4])} | roi/decon ts {f(r[s,4]/r[s,2])} tot {f(r[s,5]/r[s,3])} | wide roi tot/ts {f(r[s,7]/r[s,6])}")
for x in r[np.argsort(-r[:, 1])][:6]: print("chan %d ToT %d decon ts %.0f tot %.0f | roi ts %.0f tot %.0f | wide roi ts %.0f tot %.0f" % tuple(x))
