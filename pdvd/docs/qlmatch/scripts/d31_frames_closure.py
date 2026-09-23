#!/usr/bin/env python3
"""doc qlmatch/31: C++ OpDecon decon (PDVD_LIGHT_FRAMES dumps _f31ts18/_f31tot18, evt298567) vs the Python replica, per cathode rail run.  Run from scripts/."""
import sys, tarfile, io, numpy as np
sys.path.insert(0, '.')
import saturation_recovery_study as S
import saturation_tot_study as T
from pd_mapping_audit import RAIL, rail_intervals
V2 = S._find_templates().replace("pdvd-spe-templates.json", "pdvd-spe-templates-v2.json"); S._find_templates = lambda: V2
chd = S.load_channels()
z = np.load("/home/xqian/tmp/sat_tot_v2/shape.npz")
shp = {int(c): T.Shape(chd[int(c)], p) for c, p in zip(z["chan"], z["par"])}
W = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work/039252_light298567"
def ld(suf):
    tf = tarfile.open(f"{W}{suf}/light-frames-cath-wct.tar.bz2")
    g = lambda n: np.load(io.BytesIO(tf.extractfile(n).read()))
    return {k: g(f"{k}_298567.npy") for k in ("frame_raw", "channels_raw", "frame_decon", "channels_decon", "tickinfo_raw", "tickinfo_decon")}
A, B = ld("_f31ts18"), ld("_f31tot18")
print("tickinfo raw", A["tickinfo_raw"], "decon", A["tickinfo_decon"], "shapes", A["frame_raw"].shape, A["frame_decon"].shape)
rows = []
for r, c in enumerate(A["channels_raw"]):
    c = int(c)
    if c not in shp: continue
    raw = A["frame_raw"][r].astype(np.float64)
    rd = int(np.where(A["channels_decon"] == c)[0][0])
    da, db = A["frame_decon"][rd], B["frame_decon"][int(np.where(B["channels_decon"] == c)[0][0])]
    ped = raw[:20].mean()
    for i, j in rail_intervals(raw):
        lo, hi = max(0, i - 50), min(len(da), j + 300)
        pa, pb = da[lo:hi].sum(), db[lo:hi].sum()
        seg, s0 = T.seg_at(raw, i)
        pk = i - s0 + int(np.argmax(raw[i:j]))
        o, _ = T.methods_pe(seg, chd[c], ped, pk, RAIL - 0.5, shp[c], None, c)
        rows.append((c, i, j - i, pa, pb, o["twoside"], o["tot_fill"]))
r = np.array(rows)
print("runs", len(r))
for lo, hi in ((1, 20), (20, 71), (71, 141), (141, 260), (260, 5000)):
    s = (r[:, 2] >= lo) & (r[:, 2] < hi)
    if not s.any(): continue
    f = lambda x: f"{np.median(x):.3f} [{np.percentile(x,16):.3f},{np.percentile(x,84):.3f}]"
    print(f"ToT {lo}-{hi} n {s.sum()}: C++ tot/ts {f(r[s,4]/r[s,3])} | replica tot/ts {f(r[s,6]/r[s,5])} | C++ts/replica-ts {f(r[s,3]/r[s,5])} | C++tot/replica-tot {f(r[s,4]/r[s,6])}")
big = r[np.argsort(-r[:, 2])][:8]
for x in big: print("chan %d i %d ToT %d  C++ ts %.0f tot %.0f | replica ts %.0f tot %.0f" % tuple(x))
