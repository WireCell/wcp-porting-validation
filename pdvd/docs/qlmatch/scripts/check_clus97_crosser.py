#!/usr/bin/env python3
"""PDVD 039252 evt298567: is top cluster 97 + bottom cluster 139 a cathode-crossing
track, and why does top 97 not bundle with flash gid 96 (1423.47 us, 30494 PE)?

Repro:
    cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
    python3 docs/qlmatch/scripts/check_clus97_crosser.py

Reads the calib dump only; writes the three figures used by
docs/qlmatch/16_pdvd-clus97-crosser-evt298567.md.

Conventions (see 15_pdvd-clus34-unmatched-evt298567.md):
  - calib-dump flash "time"/"time1" are ALREADY trigger-folded (QLMatching.cxx:2885).
    Do NOT add trigger_offsets_us again.
  - the matcher uses "time1" for the TOP volume (apa 4), "time" for the BOTTOM (apa 0).
  - drift coord u = s*(x + flash_x_offset - anode_x), flash_x_offset = sign_offset*t*v.
  - physical x = x_raw + sign_offset*t*v  (a common frame: bottom [-339.91,-3],
    cathode slab [-3,+3], top [+3,+339.91]).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DUMP = "work/039252_0_walkfloor/calib-evt298567.json"
PICS = "docs/qlmatch/pics"
UID_TOP, UID_BOT, GID = 4000097, 139, 96
TAIL_X = -228.0          # raw-x split: the 15.4 cm drift gap in cluster 97
CATH_EXT1 = 2.0          # cm, runner default (ql_cathode_ext1_cm)


def load():
    d = json.load(open(DUMP))
    return (d,
            {c["uid"]: c for c in d["clusters"]},
            {f["gid"]: f for f in d["flashes"]})


def arr(c, *keys):
    return [np.array(c[k]) for k in keys]


def rfit(y, z, n=5):
    """3-sigma-clipped straight-line fit z(y). Cluster 97 carries scattered
    imaging outliers that wreck an unclipped PCA/least-squares axis."""
    keep = np.ones(len(y), bool)
    for _ in range(n):
        A = np.vstack([y[keep], np.ones(keep.sum())]).T
        m, c = np.linalg.lstsq(A, z[keep], rcond=None)[0]
        s = np.std(z[keep] - (m * y[keep] + c))
        keep = np.abs(z - (m * y + c)) < 3 * s
    return m, c, np.std(z[keep] - (m * y[keep] + c)), keep


def clumps(P, link=6.0):
    """Single-linkage grouping, used to show the tail is disconnected."""
    used = np.zeros(len(P), bool)
    out = []
    for i in range(len(P)):
        if used[i]:
            continue
        grp = [i]
        used[i] = True
        grew = True
        while grew:
            grew = False
            for j in range(len(P)):
                if used[j]:
                    continue
                if min(np.linalg.norm(P[j] - P[k]) for k in grp) < link:
                    grp.append(j)
                    used[j] = True
                    grew = True
        out.append(grp)
    return sorted(out, key=len, reverse=True)


def main():
    d, cby, fby = load()
    os.makedirs(PICS, exist_ok=True)
    ds = d["drift_speed"]
    dclk = d["trigger_offsets_us"][1] - d["trigger_offsets_us"][0]
    gt, gb = d["geometry"]["4"], d["geometry"]["0"]
    f = fby[GID]

    xt, yt, zt, qt = arr(cby[UID_TOP], "x", "y", "z", "q")
    xb, yb, zb, qb = arr(cby[UID_BOT], "x", "y", "z", "q")
    tail = xt < TAIL_X
    body = ~tail

    # physical x at the flash-96 T0
    XT = xt + gt["sign_offset"] * f["time1"] * ds
    XB = xb + gb["sign_offset"] * f["time"] * ds

    # ---- Fig 1: the crosser ------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for k, (h, v, hb, vb, xl, yl) in enumerate([
            (XT, yt, XB, yb, "physical x [cm]", "y [cm]"),
            (XT, zt, XB, zb, "physical x [cm]", "z [cm]"),
            (yt, zt, yb, zb, "y [cm]", "z [cm]")]):
        a = ax[k]
        a.scatter(hb, vb, s=3, c="tab:green", label=f"bot 139 ({len(xb)} pts)")
        a.scatter(h[body], v[body], s=3, c="tab:blue", label=f"top 97 body ({int(body.sum())} pts)")
        a.scatter(h[tail], v[tail], s=26, c="red", zorder=5,
                  label=f"top 97 merged-in tail ({int(tail.sum())} pts)")
        if k < 2:
            a.axvspan(-3, 3, color="0.5", alpha=0.4, label="cathode slab")
            a.set_xlim(-90, 130)
        a.set_xlabel(xl)
        a.set_ylabel(yl)
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="best")
    m, c, r, keep = rfit(yt[body & (qt > 0)], zt[body & (qt > 0)])
    yy = np.linspace(88, 327, 10)
    ax[2].plot(yy, m * yy + c, "k--", lw=1.2, label=f"top fit (rms {r:.1f} cm)")
    ax[2].legend(fontsize=8)
    fig.suptitle("PDVD 039252 evt298567 — top cluster 97 + bottom cluster 139 at the "
                 f"flash-{GID} T0 ({f['time']:.1f} us, {f['total_PE']:.0f} PE): one "
                 "cathode-crossing track", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{PICS}/clus97_crosser.png", dpi=110)
    print("wrote", f"{PICS}/clus97_crosser.png")

    # ---- Fig 2: the tail is ghosts ----------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    P = np.vstack([xt[tail], yt[tail], zt[tail]]).T
    cl = clumps(P)
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    ax[0].scatter(xt[body], yt[body], s=3, c="tab:blue", label="body")
    for n, g in enumerate(cl):
        ax[0].scatter(P[g, 0], P[g, 1], s=40, color=cols[n % 10],
                      label=f"clump {n+1} ({len(g)} pts)")
    ax[0].set_xlabel("raw x [cm]  (drift)")
    ax[0].set_ylabel("y [cm]")
    ax[0].set_title(f"tail = {len(cl)} disconnected clumps, not a track", fontsize=10)
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    gh_t = 100 * (qt[tail] == 0).sum() / tail.sum()
    gh_b = 100 * (qt[body] == 0).sum() / body.sum()
    ax[1].bar(["tail", "body"], [gh_t, gh_b], color=["red", "tab:blue"])
    ax[1].set_ylabel("zero-charge (ghost) points [%]")
    ax[1].set_title(f"zero-charge fraction {gh_t/gh_b:.1f}x the body's -- a red herring, "
                    "see doc 4.1", fontsize=10)
    ax[1].grid(alpha=0.3, axis="y")

    # occupancy of the tail's drift slices by OTHER top-volume clusters
    edges = np.linspace(-260, -120, 29)
    other = np.zeros(len(edges) - 1)
    for cc in d["clusters"]:
        if cc["apa"] != 4 or cc["uid"] == UID_TOP:
            continue
        h, _ = np.histogram(np.array(cc["x"]), bins=edges)
        other += h
    mine, _ = np.histogram(xt, bins=edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    ax[2].step(ctr, other, where="mid", color="0.4", label="other top-volume clusters")
    ax[2].step(ctr, mine, where="mid", color="tab:blue", label="cluster 97")
    ax[2].axvspan(-260, TAIL_X, color="red", alpha=0.15, label="tail region")
    ax[2].set_xlabel("raw x [cm]  (drift)")
    ax[2].set_ylabel("points per slice")
    ax[2].set_title("the tail lives in busy slices (crowded imaging region)", fontsize=10)
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.suptitle("PDVD 039252 evt298567 — anatomy of cluster 97's tail: real charge from "
                 "OTHER objects, merged in by the 80 cm isolated rule (doc 4)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{PICS}/clus97_tail_anatomy.png", dpi=110)
    print("wrote", f"{PICS}/clus97_tail_anatomy.png")

    # ---- Fig 3: containment budget ----------------------------------------
    u = gt["s"] * (xt + gt["sign_offset"] * f["time1"] * ds - gt["anode_x"])
    gate = gt["u_cathode"] + CATH_EXT1
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(u[body], bins=80, color="tab:blue", label="body")
    ax.hist(u[tail], bins=80, color="red", label="ghost tail")
    ax.axvline(gt["u_cathode"], color="k", lw=1.5, label=f"cathode u={gt['u_cathode']:.1f}")
    ax.axvline(gate, color="k", ls="--", lw=1.5,
               label=f"containment gate u={gate:.1f} (cathode_ext1={CATH_EXT1})")
    ax.set_xlabel("drift coordinate u [cm] at the flash-96 T0")
    ax.set_ylabel("points")
    ax.set_title("cluster 97 vs the containment gate: the tail AND the real body end "
                 "both sit outside", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{PICS}/clus97_containment.png", dpi=110)
    print("wrote", f"{PICS}/clus97_containment.png")

    # ---- numbers quoted in the doc ----------------------------------------
    print("\n--- numbers ---")
    print(f"pool(uid {UID_TOP}) contains gid {GID}: "
          f"{any(b['flash_gid'] == GID for b in d['bundles'] if b['main_cluster'] == UID_TOP)}")
    print(f"tail: {int(tail.sum())} pts, {100*qt[tail].sum()/qt.sum():.2f}% of charge, "
          f"{gh_t:.0f}% ghosts vs {gh_b:.1f}% in body, {len(cl)} clumps")
    print(f"u past gate: {int((u > gate).sum())} pts; body-only u_max={u[body].max():.2f} vs gate {gate:.2f} "
          f"-> body alone over by {u[body].max()-gate:.2f} cm")
    mb_, cb_, rb_, _ = rfit(yb, zb)
    print(f"y-z fits: top slope {m:+.4f} rms {r:.2f} | bot slope {mb_:+.4f} rms {rb_:.2f} | "
          f"angle {abs(np.degrees(np.arctan(m)-np.arctan(mb_))):.2f} deg | "
          f"dz at y=175: {abs((m*175+c)-(mb_*175+cb_)):.2f} cm")


if __name__ == "__main__":
    main()
