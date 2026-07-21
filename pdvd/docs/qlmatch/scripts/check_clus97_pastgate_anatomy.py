#!/usr/bin/env python3
"""Shape of top-97's 17 BODY points past the containment gate at flash-96 T0:
do they continue the track's 3-D line (real track end) or form a kink along
drift at fixed (y,z) (late/coincident charge)?

Answer: they are TWO structures --
  - 13 pts at FIXED (y=175.58, z=219.04) == the crossing point, spanning u
    338.99..342.24 (3.25 cm along drift only): the crosser's own late charge.
  - 4 pts at (y~245, z~214.5), u 345.80..346.69, 51 cm PERPENDICULAR to the
    track axis, detached by a 3.56 cm slice gap: another merged-in satellite
    (junk), NOT the real track end.
So the real track's own overshoot is 342.24 vs gate 338.91 (3.33 cm), not
346.69/7.78 cm as doc 16 originally said. See doc 16 section 10.

Repro:
    cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
    python3 docs/qlmatch/scripts/check_clus97_pastgate_anatomy.py
"""
import json
import numpy as np

DUMP = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work/039252_0_walkfloor/calib-evt298567.json"
d = json.load(open(DUMP))
cby = {c["uid"]: c for c in d["clusters"]}
fby = {f["gid"]: f for f in d["flashes"]}
gt = d["geometry"]["4"]
ds = d["drift_speed"]
f = fby[96]

c = cby[4000097]
x, y, z, q = (np.array(c[k]) for k in "xyzq")
tail = x < -228.0
body = ~tail
u = gt["s"] * (x + gt["sign_offset"] * f["time1"] * ds - gt["anode_x"])
gate = gt["u_cathode"] + 2.0

sel = body & (u > gate)          # the 17 real-body points past the gate
deep = body & (u > gt["u_cathode"] - 30.0)  # last ~30cm of body for a local line

print(f"body pts past gate: {sel.sum()}   u range {u[sel].min():.2f}..{u[sel].max():.2f}")
print(f"  x   range {x[sel].min():.2f}..{x[sel].max():.2f}  (span {x[sel].max()-x[sel].min():.2f} cm)")
print(f"  y   range {y[sel].min():.2f}..{y[sel].max():.2f}  (span {y[sel].max()-y[sel].min():.2f} cm)")
print(f"  z   range {z[sel].min():.2f}..{z[sel].max():.2f}  (span {z[sel].max()-z[sel].min():.2f} cm)")
print(f"  q: {q[sel].sum():.0f} total, {q[sel].sum()/q.sum()*100:.2f}% of cluster; "
      f"{(q[sel]==0).sum()} zero-q pts")

# local 3-D line from the last 30 cm of body INSIDE the gate
ref = deep & (u <= gate)
P = np.vstack([x[ref], y[ref], z[ref]]).T
c0 = P.mean(0)
_, _, V = np.linalg.svd(P - c0)
axis = V[0]
print(f"\nlocal body axis (last 30cm inside gate, {ref.sum()} pts): "
      f"dx/dy/dz = {axis[0]:+.3f}/{axis[1]:+.3f}/{axis[2]:+.3f}")
# perpendicular distance of each past-gate point from that line
Q = np.vstack([x[sel], y[sel], z[sel]]).T - c0
perp = np.linalg.norm(Q - np.outer(Q @ axis, axis), axis=1)
print("perp distance of the 17 pts from the local track line [cm]:")
print("  " + " ".join(f"{p:.1f}" for p in sorted(perp)))
print(f"  median {np.median(perp):.2f}  max {perp.max():.2f}")

# also: per-point u ordered, with y,z -- eyeball collinearity vs kink
o = np.argsort(u[sel])
print("\n  u      x      y      z      q")
for i in np.where(sel)[0][np.argsort(u[sel])]:
    print(f"  {u[i]:7.2f} {x[i]:7.2f} {y[i]:7.2f} {z[i]:7.2f} {q[i]:8.0f}")

# projected direction: does (y,z) keep advancing along the track as u grows?
dy = y[sel].max() - y[sel].min()
dxs = u[sel].max() - u[sel].min()
print(f"\nslope past gate: du/dy = {dxs/max(dy,1e-9):.2f}  "
      f"(body-average |du/dy| ~ {abs((u[body].max()-u[body].min())/(y[body].max()-y[body].min())):.2f})")
