#!/usr/bin/env python3
"""Measure the (channel, slice) shift that best centres the measured charge of a
cluster on the fitted track drawn over it, in a Magnify-tracking dump.

This is the check that separates a RECONSTRUCTION problem from a DISPLAY problem:
it works in raw index space (T_proj_data.channel / time_slice against
T_rec_charge.pu/pv/pw and pt), so a result of (0, 0) means the writer's
coordinates and the charge cells agree and anything visible on the GUI pad comes
from how the pad bins them.

For every charge cell within 3 index units of the track polyline it computes the
signed perpendicular offset from the nearest polyline segment, then solves the
charge-weighted least-squares problem

    min over (dch, dt) of  sum_i q_i * (s_i + dch*nx_i + dt*ny_i)^2

for the global shift.  Repeated with charge_pred instead of charge: if the two
agree, the fit's own charge prediction sits where the measurement sits.

Usage:  pr_proj_align.py <tracking-pr.root|tracking-stm.root> [cluster_id]

See docs/pr/7_magnify-tracking-projection-alignment.md.
"""
import sys
import numpy as np
import uproot

# SBND two-TPC concatenated-per-plane channel scheme (SbndPrMagnifyTrackingVisitor
# logs it as "channel scheme nch=[1984,1984,1670] base=[0,3968,7936]").
NU, NV, NW = 3968, 3968, 3340
PLANES = [("U", 0, NU, "pu"), ("V", NU, NU + NV, "pv"), ("W", NU + NV, NU + NV + NW, "pw")]


def solve(name, lo, hi, key, cells, rec):
    CH, TS, Q, QP = cells
    m = (CH >= lo) & (CH < hi)
    c, t, q, qp = CH[m], TS[m], Q[m], QP[m]
    X, PT, seg = rec[key], rec["pt"], rec["real_cluster_id"]

    # polyline: consecutive fitted points inside one PR segment
    ok = seg[:-1] == seg[1:]
    ax, ay = X[:-1][ok], PT[:-1][ok]
    bx, by = X[1:][ok], PT[1:][ok]
    vx, vy = bx - ax, by - ay
    L = np.hypot(vx, vy)
    good = L > 1e-9
    ax, ay, vx, vy, L = ax[good], ay[good], vx[good], vy[good], L[good]
    if len(L) == 0:
        print(f"{name}: no polyline")
        return

    P = np.stack([c, t], 1)[:, None, :]
    A = np.stack([ax, ay], 1)[None, :, :]
    V = np.stack([vx, vy], 1)[None, :, :]
    tt = np.clip(((P - A) * V).sum(-1) / (L ** 2)[None, :], 0, 1)
    d2 = ((P - (A + tt[..., None] * V)) ** 2).sum(-1)
    j = np.argmin(d2, 1)
    keep = np.sqrt(d2[np.arange(len(c)), j]) < 3.0

    nx, ny = -vy[j] / L[j], vx[j] / L[j]          # unit normal of the nearest segment
    s = (c - ax[j]) * nx + (t - ay[j]) * ny        # signed perpendicular offset

    out = []
    for lbl, w in (("meas", q), ("pred", qp)):
        k = keep & (w > 0)
        W, NX, NY, S = w[k], nx[k], ny[k], s[k]
        M = np.array([[np.sum(W * NX * NX), np.sum(W * NX * NY)],
                      [np.sum(W * NX * NY), np.sum(W * NY * NY)]])
        b = -np.array([np.sum(W * S * NX), np.sum(W * S * NY)])
        sol = np.linalg.solve(M, b)
        out.append(f"{lbl} N={int(k.sum())} dch={sol[0]:+.3f} dt={sol[1]:+.3f}")
    print(f"{name}: " + " | ".join(out))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    cid = int(sys.argv[2]) if len(sys.argv) > 2 else None

    f = uproot.open(path)
    p = f["T_proj_data"].arrays(library="np")
    cids = list(p["cluster_id"][0])
    r = f["T_rec_charge"].arrays(["pu", "pv", "pw", "pt", "cluster_id", "real_cluster_id"],
                                 library="np")
    if cid is None:                                  # default: the biggest fitted cluster
        vals, counts = np.unique(r["cluster_id"], return_counts=True)
        cid = int(vals[np.argmax(counts)])
    ci = cids.index(cid)
    sel = r["cluster_id"] == cid
    rec = {k: v[sel] for k, v in r.items()}
    cells = (np.asarray(p["channel"][0][ci]).astype(float),
             np.asarray(p["time_slice"][0][ci]).astype(float),
             np.clip(np.asarray(p["charge"][0][ci]).astype(float), 0, None),
             np.clip(np.asarray(p["charge_pred"][0][ci]).astype(float), 0, None))

    print(f"{path}  cluster {cid}: {int(sel.sum())} fitted points, {len(cells[0])} charge cells")
    for name, lo, hi, key in PLANES:
        solve(name, lo, hi, key, cells, rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
