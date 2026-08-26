#!/usr/bin/env python3
"""doc 83 -- static case-study panels for the high/low-side MCS outliers.

Built directly from tracking-pr.root (T_rec_charge has everything needed:
per-point x/y/z, dQ/dx via Trun's scale/offset, residual range, pdg,
cluster/segment id) plus the SAME reference dQ/dx-vs-residual-range tables
the interactive PR display (pr_display/pr_display_viewer.py) plots against
(sbnd_xin/nusel_display/stm_ref_dqdx.json) -- so these panels agree with
what the live viewer would show for the same event.  To explore a case
interactively instead (click particles, toggle wire planes, etc.), re-run it
with the pr_display stage on and open the Bokeh viewer -- see doc 83 sec 2's
Repro block for the exact command.

Each panel, one event/segment per figure: LEFT = X-Z and Y-Z projections of
the WHOLE parent cluster, points coloured by real_cluster_id (their own PR
segment/fragment) so an adjacent broken-off piece is visible in the same
picture as the selected muon; RIGHT = dQ/dx vs residual range for the
SELECTED segment's own points (exactly what fed cal_kine_range and MCS),
with the muon reference curve overlaid.

Usage: mcs83_panels.py --arm ARM --out DIR EVT:SEGID:CID [EVT:SEGID:CID ...]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
STM_REF = os.path.join(HERE, "..", "nusel_display", "stm_ref_dqdx.json")


def load_ref_curve(name):
    d = json.load(open(STM_REF))[name]
    rr = d["start"] + d["step"] * np.arange(len(d["values"]))
    return rr, np.array(d["values"])


def panel(arm, evt, segid, cid, out_png, title_extra=""):
    froot = os.path.join(arm, f"pr_evt{evt}", "tracking-pr.root")
    f = uproot.open(froot)
    rc = f["T_rec_charge"].arrays(
        ["x", "y", "z", "q", "nq", "rr", "cluster_id", "real_cluster_id", "particle_id"],
        library="np")
    tr = f["Trun"].arrays(["dQdx_scale", "dQdx_offset"], library="np")
    m_cl = rc["cluster_id"] == cid
    m_seg = rc["real_cluster_id"] == segid

    fig = plt.figure(figsize=(11, 4.2))
    axxz = fig.add_subplot(1, 3, 1)
    axyz = fig.add_subplot(1, 3, 2)
    axdq = fig.add_subplot(1, 3, 3)

    rids = np.unique(rc["real_cluster_id"][m_cl])
    cmap = plt.get_cmap("tab20")
    for i, rid in enumerate(rids):
        m = m_cl & (rc["real_cluster_id"] == rid)
        if rid == segid:
            continue
        pid = np.unique(rc["particle_id"][m])
        lab = f"other rid={int(rid)} pid={list(pid)}" if m.sum() > 3 else None
        axxz.scatter(rc["x"][m], rc["z"][m], s=6, alpha=0.5, color=cmap(i % 20), label=lab)
        axyz.scatter(rc["y"][m], rc["z"][m], s=6, alpha=0.5, color=cmap(i % 20))
    axxz.scatter(rc["x"][m_seg], rc["z"][m_seg], s=10, color="black", label=f"selected seg {segid}")
    axyz.scatter(rc["y"][m_seg], rc["z"][m_seg], s=10, color="black")
    axxz.set_xlabel("x [cm]"); axxz.set_ylabel("z [cm]"); axxz.set_title("X-Z")
    axyz.set_xlabel("y [cm]"); axyz.set_ylabel("z [cm]"); axyz.set_title("Y-Z")
    axxz.legend(fontsize=6, loc="best")

    dQ = (rc["q"][m_seg] - tr["dQdx_offset"][0]) / tr["dQdx_scale"][0]
    dx = np.maximum(rc["nq"][m_seg], 1e-9)
    dqdx = dQ / dx
    rr = rc["rr"][m_seg]
    axdq.scatter(rr, dqdx, s=10, color="tab:blue", alpha=0.7, label="selected segment points")
    for name, color in [("MuonDeDx", "tab:green"), ("PionDeDx", "tab:orange"), ("ProtonDeDx", "tab:red")]:
        rr_ref, val_ref = load_ref_curve(name)
        axdq.plot(rr_ref, val_ref, "-", color=color, lw=1.2, label=name.replace("DeDx", ""))
    axdq.set_xlim(0, max(30, np.percentile(rr[rr > -1], 95) if (rr > -1).any() else 30))
    axdq.set_ylim(0, max(2e5, np.percentile(dqdx, 95) * 1.3 if len(dqdx) else 2e5))
    axdq.set_xlabel("residual range [cm]")
    axdq.set_ylabel("dQ/dx [e/cm]")
    axdq.set_title("dQ/dx vs residual range")
    axdq.legend(fontsize=6)

    fig.suptitle(f"evt{evt} seg{segid} (cluster {cid}){title_extra}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("cases", nargs="+", help="EVT:SEGID:CID[:label]")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for case in args.cases:
        parts = case.split(":")
        evt, segid, cid = int(parts[0]), int(parts[1]), int(parts[2])
        label = parts[3] if len(parts) > 3 else ""
        out_png = os.path.join(args.out, f"outlier_evt{evt}_seg{segid}.png")
        try:
            panel(args.arm, evt, segid, cid, out_png, title_extra=(" -- " + label if label else ""))
            print("wrote", out_png)
        except Exception as err:                            # noqa: BLE001
            print(f"WARN evt{evt} seg{segid}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
