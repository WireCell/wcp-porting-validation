#!/usr/bin/env python3
"""Inspect one census pair: z-x/y-x scatter at the census T0 + past-face anatomy.

Usage: inspect_pair.py RUN IDX UID_T UID_B GID [OUT.png]
"""
import json
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cathode_tail_census as C

run, idx, uid_t, uid_b, gid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
out = sys.argv[6] if len(sys.argv) > 6 else f"pair_{run}_{idx}_{uid_t}_{uid_b}.png"

dump = glob.glob(f"{C.BASE}/{run}_{idx}_keep/calib-evt*.json")[0]
d = json.load(open(dump))
gt, gb = d["geometry"]["4"], d["geometry"]["0"]
ds = d["drift_speed"]
cby = {c["uid"]: c for c in d["clusters"]}
fby = {f["gid"]: f for f in d["flashes"]}
f = fby[gid]
t = C.cluster_info(cby[uid_t], gt)
b = C.cluster_info(cby[uid_b], gb)
Xt = t["x_raw"] + gt["sign_offset"] * f["time1"] * ds
Xb = b["x_raw"] + gb["sign_offset"] * f["time"] * ds

fig, axs = plt.subplots(1, 3, figsize=(17, 5))
for k, (ht, hb, xl) in enumerate([(t["z"], b["z"], "z [cm]"), (t["y"], b["y"], "y [cm]")]):
    a = axs[k]
    a.scatter(ht[t["tube"]], Xt[t["tube"]], s=4, c="tab:blue", label=f"top {uid_t} tube")
    a.scatter(ht[~t["tube"]], Xt[~t["tube"]], s=14, c="tab:cyan", marker="x", label="top excluded")
    a.scatter(hb[b["tube"]], Xb[b["tube"]], s=4, c="tab:green", label=f"bot {uid_b} tube")
    a.scatter(hb[~b["tube"]], Xb[~b["tube"]], s=14, c="tab:olive", marker="x", label="bot excluded")
    a.axhspan(-3, 3, color="0.5", alpha=0.35, label="cathode slab")
    a.set_xlabel(xl); a.set_ylabel("physical x [cm]")
    a.set_ylim(-60, 60)
    a.grid(alpha=0.3); a.legend(fontsize=7)
# zoom on the junction in (y,z)
a = axs[2]
a.scatter(t["y"], t["z"], s=4, c="tab:blue")
a.scatter(b["y"], b["z"], s=4, c="tab:green")
pt = t["tube"] & (Xt < gt["cathode_x"])
pb = b["tube"] & (Xb > gb["cathode_x"])
a.scatter(t["y"][pt], t["z"][pt], s=30, c="red", marker="^", label=f"top past face ({pt.sum()})")
a.scatter(b["y"][pb], b["z"][pb], s=30, c="orange", marker="v", label=f"bot past face ({pb.sum()})")
a.set_xlabel("y [cm]"); a.set_ylabel("z [cm]"); a.grid(alpha=0.3); a.legend(fontsize=7)
fig.suptitle(f"{run} idx{idx} evt{dump.split('calib-evt')[1][:-5]}  top {uid_t} + bot {uid_b} @ flash {gid} "
             f"({f['total_PE']:.0f} PE, t={f['time']:.1f} us)", fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=100)
print("wrote", out)

# past-face anatomy: clumps, drift profile
for nm, ci, X, face, sgn in (("TOP", t, Xt, gt["cathode_x"], 1), ("BOT", b, Xb, gb["cathode_x"], -1)):
    past = ci["tube"] & (sgn * (X - face) < 0)
    if not past.any():
        print(f"{nm}: none past face")
        continue
    pen = np.max(sgn * (face - X[ci["tube"]]))
    yz = np.stack([ci["y"][past], ci["z"][past]], 1)
    print(f"{nm}: {past.sum()} pts past face, pen {pen:.2f} cm, "
          f"yz spread y[{yz[:,0].min():.1f},{yz[:,0].max():.1f}] z[{yz[:,1].min():.1f},{yz[:,1].max():.1f}], "
          f"X range [{X[past].min():.2f},{X[past].max():.2f}], q frac {ci['q'][past].sum()/ci['q_tot']:.4f}")
    # contiguity in drift: sorted X gaps
    xs = np.sort(X[past])
    if len(xs) > 1:
        gaps = np.diff(xs)
        print(f"   drift gaps: max {gaps.max():.2f} cm (contiguous if small)")
    # gap between body end and past-face material
    body = ci["tube"] & ~past
    if body.any():
        near = np.min(np.abs(X[body] - face))
        print(f"   nearest body point to face: {near:.2f} cm")
