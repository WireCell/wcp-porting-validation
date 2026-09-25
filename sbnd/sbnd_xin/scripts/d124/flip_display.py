#!/usr/bin/env python3
"""doc sbnd_xin/124 -- a BLINDED two-arm display of one event, for the hand scan of the numu flips.

One PNG per event, two rows labelled A and B; which arm is A is drawn per event from a seeded shuffle and
written to a sidecar map (--map) BEFORE any figure exists, so the scanner judges the pictures and only then
joins on the map (feedback_blind_the_scan_sheet).  Nothing on the image names an arm or a score.

Per row (one arm):
  1-2  z-x and z-y of the whole event (the PR job's clustering-global layer, pr_evt<E>/mabc-pr.zip):
       grey = everything; blue = the candidate's PR input (the selected main + act_in_pr clusters of T_tagger);
       red = points in this row's PR input that are NOT in the other row's (keyed by y, z, q);
       orange = when this row has NO candidate: where the other row's candidate points sit here.
  3    a +-60 cm zoom (z-y) on the neutrino vertex: shower_track-global, colour = segment
       (real_cluster_id), dark = track, light = shower; star = the neutrino vertex (vertices-global q == 15000)
  4    measured (bars) vs predicted (black ticks) PE per PMT channel of the beam-window flashes of both TPCs
       (the Q/L job's op layer, ql_evt<E>/mabc-all-apa.zip), channel index on x.
Title per row: the PR particle list (T_kine) and reco Enu -- no scores.

usage: flip_display.py --ql-a <ql_root_A> --pr-a <pr_root_A> --ql-b <ql_root_B> --pr-b <pr_root_B>
                       --map map.tsv --outdir dir [--seed 124] [--tag sample] <evt> ...
"""
import argparse, io, json, os, random, sys, zipfile
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

PDG = {11: "e", -11: "e", 13: "mu", -13: "mu", 211: "pi", -211: "pi", 2212: "p", 22: "g", 111: "pi0", 321: "K", 2112: "n", 0: "?"}
WIN = (0.2, 2.2)


def layer(zpath, suffix):
    if not os.path.exists(zpath):
        return None
    z = zipfile.ZipFile(zpath)
    n = [x for x in z.namelist() if x.endswith(suffix)]
    return json.loads(z.read(n[0])) if n else None


def arm(ql, pr, e):
    d = os.path.join(pr, "pr_evt%s" % e)
    cg = layer(os.path.join(d, "mabc-pr.zip"), "-clustering-global.json")
    st = layer(os.path.join(d, "mabc-pr.zip"), "-shower_track-global.json")
    vx = layer(os.path.join(d, "mabc-pr.zip"), "-vertices-global.json")
    op = layer(os.path.join(ql, "ql_evt%s" % e, "mabc-all-apa.zip"), "-op.json")
    f = uproot.open(os.path.join(d, "tracking-pr.root"))
    inp, pid, enu, nv = set(), "", 0.0, None
    if "T_tagger" in f:
        t = f["T_tagger"].arrays(["numu_score", "cluster_id", "act_cluster_id", "act_in_pr", "nu_x", "nu_y", "nu_z"], library="np")
        i = int(np.argmax(t["numu_score"]))
        inp = set(int(c) for c, p in zip(t["act_cluster_id"][i], t["act_in_pr"][i]) if p == 1) | {int(t["cluster_id"][i])}
        nv = (float(t["nu_x"][i]), float(t["nu_y"][i]), float(t["nu_z"][i]))
        k = f["T_kine"].arrays(["cluster_id", "kine_particle_type", "kine_energy_particle", "kine_reco_Enu"], library="np")
        for j in range(len(k["cluster_id"])):
            if int(k["cluster_id"][j]) == int(t["cluster_id"][i]):
                pid = " ".join("%s%.0f" % (PDG.get(int(a), str(int(a))), en) for a, en in zip(k["kine_particle_type"][j], k["kine_energy_particle"][j]) if en >= 5)
                enu = float(k["kine_reco_Enu"][j])
    x = np.array(cg["x"]); y = np.array(cg["y"]); z = np.array(cg["z"]); q = np.array(cg["q"]); c = np.array(cg["cluster_id"])
    key = [(round(a, 1), round(b, 1), round(w, 0)) for a, b, w in zip(y, z, q)]
    return dict(x=x, y=y, z=z, c=c, key=key, inp=inp, pid=pid, enu=enu, nv=nv, st=st, vx=vx, op=op)


def draw_row(axs, R, O, label):
    x, y, z = R["x"], R["y"], R["z"]
    inR = np.array([cc in R["inp"] for cc in R["c"]])
    okeys = set(k for k, cc in zip(O["key"], O["c"]) if cc in O["inp"])
    mykeys = set(k for k, cc in zip(R["key"], R["c"]) if cc in R["inp"])
    only = np.array([(m and k not in okeys) for m, k in zip(inR, R["key"])])
    other_here = np.array([k in okeys for k in R["key"]]) if not R["inp"] else np.zeros(len(x), bool)
    for ax, (a, b, la, lb) in zip(axs[:2], ((z, x, "z [cm]", "x [cm]"), (z, y, "z [cm]", "y [cm]"))):
        ax.scatter(a, b, s=0.3, c="0.8", linewidths=0)
        ax.scatter(a[inR & ~only], b[inR & ~only], s=1.2, c="C0", linewidths=0)
        ax.scatter(a[only], b[only], s=5, c="red", linewidths=0)
        ax.scatter(a[other_here], b[other_here], s=1.5, c="orange", linewidths=0)
        if R["nv"] is not None:
            vv = {"z [cm]": R["nv"][2], "x [cm]": R["nv"][0], "y [cm]": R["nv"][1]}
            ax.plot(vv[la], vv[lb], "*", ms=12, mfc="gold", mec="k")
        ax.axhline(0, color="k", lw=0.4, ls=":") if lb == "x [cm]" else None
        ax.set_xlim(-5, 505); ax.set_ylim(-205, 205); ax.set_xlabel(la); ax.set_ylabel(lb); ax.set_aspect("equal")
    axs[0].set_title("%s   %s   Enu %.0f MeV" % (label, R["pid"][:70] if R["inp"] else "NO CANDIDATE", R["enu"]), fontsize=9, loc="left")
    # vertex zoom
    ax = axs[2]
    st = R["st"]
    if R["nv"] is not None and st is not None and len(st["x"]):
        sx, sy, sz = np.array(st["x"]), np.array(st["y"]), np.array(st["z"])
        sq, sr = np.array(st["q"]), np.array(st["real_cluster_id"])
        vxz, vy = R["nv"][2], R["nv"][1]
        m = (np.abs(sz - vxz) < 60) & (np.abs(sy - vy) < 60)
        segs = sorted(set(sr[m]))
        cmap = plt.get_cmap("tab20")
        for i, sg in enumerate(segs):
            mm = m & (sr == sg)
            trk = sq[mm] < 1000
            col = cmap(i % 20)
            ax.scatter(sz[mm][trk], sy[mm][trk], s=3, color=col, linewidths=0)
            ax.scatter(sz[mm][~trk], sy[mm][~trk], s=3, color=col, alpha=0.3, linewidths=0)
        ax.plot(vxz, vy, "*", ms=14, mfc="gold", mec="k")
        ax.set_xlim(vxz - 60, vxz + 60); ax.set_ylim(vy - 60, vy + 60); ax.set_aspect("equal")
        ax.set_title("vertex zoom z-y (dark track, light shower)", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no vertex", ha="center", transform=ax.transAxes)
    ax.set_xlabel("z [cm]"); ax.set_ylabel("y [cm]")
    # PMT pattern of the beam-window flashes
    ax = axs[3]
    op = R["op"]
    txt = []
    if op is not None:
        for t, apa, pes, pred, tot in zip(op["op_t"], op["apa"], op["op_pes"], op["op_pes_pred"], op["op_peTotal"]):
            if WIN[0] <= float(t) < WIN[1]:
                pes = np.array(pes, float); pred = np.array(pred, float) if pred else np.zeros_like(pes)
                ch = np.arange(len(pes))
                ax.bar(ch, pes, width=1.0, color="C%d" % (int(apa) + 1), alpha=0.6)
                if pred.any():
                    ax.plot(ch, pred, "_", color="k", ms=4)
                txt.append("tpc%d t=%.3f us %.0f PE%s" % (int(apa), float(t), float(tot), " pred %.0f" % pred.sum() if pred.any() else " (no match)"))
    ax.set_yscale("symlog", linthresh=10); ax.set_xlabel("PMT channel"); ax.set_ylabel("PE")
    ax.set_title("; ".join(txt)[:110] or "no beam-window flash", fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ql-a", required=True); ap.add_argument("--pr-a", required=True)
    ap.add_argument("--ql-b", required=True); ap.add_argument("--pr-b", required=True)
    ap.add_argument("--map", required=True); ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=124); ap.add_argument("--tag", default="")
    ap.add_argument("events", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    # per-event seed (seed, tag, event): independent of how the events are batched into calls
    order = {e: random.Random("%d:%s:%s" % (a.seed, a.tag, e)).random() < 0.5 for e in a.events}   # True: A = first arm
    new = not os.path.exists(a.map)
    with open(a.map, "a") as f:
        if new:
            f.write("tag\tevent\tA\tB\n")
        for e in a.events:
            f.write("%s\t%s\t%s\t%s\n" % (a.tag, e, *(("first", "second") if order[e] else ("second", "first"))))
    for e in a.events:
        X1 = arm(a.ql_a, a.pr_a, e); X2 = arm(a.ql_b, a.pr_b, e)
        RA, RB = (X1, X2) if order[e] else (X2, X1)
        fig, axs = plt.subplots(2, 4, figsize=(22, 8.6), gridspec_kw=dict(width_ratios=[2.4, 2.4, 1.0, 1.6]))
        draw_row(axs[0], RA, RB, "A"); draw_row(axs[1], RB, RA, "B")
        fig.suptitle("%s evt %s  -- blue: candidate PR input; red: in this row's input only; orange: other row's candidate here" % (a.tag, e), fontsize=10)
        plt.tight_layout()
        out = os.path.join(a.outdir, "%s_%s.png" % (a.tag, e))
        fig.savefig(out, dpi=60); plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
