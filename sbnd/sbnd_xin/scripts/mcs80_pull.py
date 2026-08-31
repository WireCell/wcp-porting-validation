#!/usr/bin/env python3
"""doc 80 sec 9.3 -- the tune-transfer PULL TEST (run FIRST).

For each MCS-selected muon in a knob-ON arm: harvest its fitted points from
T_rec_charge (particle_id == 13 -- pdg semantics in this tree -- within the
segment's cluster), run the round-0 instrumented upstream dumper
(mcs_upstream/dumper/mcs_dump, == the port to 1e-13 in the angles) to get the
per-14cm-segment theta_xz / theta_yz / vx / distance, then:

  T_k        = KE from the upstream CSDA table at (tracklen - distance_k)
               (range truth proxy: residual range from segment midpoint)
  sigma_pred = sigma1 of pred_theta_{xz,yz}_pars(T_k) [python port of the
               tune formulas, constants verbatim from mcs.cxx:25-50]
  pull       = theta_k / sigma_pred

If the MicroBooNE tune transfers, the pull CORE width (robust MAD sigma) is
1.00 (doc 80 sec 9.3).  Sliced by T and by ivx; a resolution-term failure
grows with T, a Highland-modifier failure is flat or quartic-shaped.

Restricted to CONTAINED muons (range trustworthy): joined against the
mcs80_analysis.py TSV (isfc==1) when given via --tsv.

Usage: mcs80_pull.py --out DIR [--tsv mcs_joined.tsv] [--max-events N] ARM
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPER = os.path.join(HERE, "..", "mcs_upstream", "dumper", "mcs_dump")

# ---- upstream CSDA table (mcs.cxx:437-446), KE in MeV, rr in cm ----
RHO = 1.396
UCSDA = np.array([.9833, 1.786, 3.321, 6.598, 10.58, 30.84, 42.50, 67.32, 106.3,
                  172.5, 238.5, 493.4, 616.3, 855.2, 1202., 1758., 2297., 4359.,
                  5354., 7298.])
UKE = np.array([10., 14., 20., 30., 40., 80., 100., 140., 200., 300., 400.,
                800., 1000., 1400., 2000., 3000., 4000., 8000., 10000., 14000.])
URR = UCSDA / RHO

def ke_from_rr(rr):
    return np.interp(rr, URR, UKE)  # clamp at edges is fine for T>=10 MeV use

MMU = 105.658

# ---- tune constants (mcs.cxx:25-50, verbatim) ----
RES_S1_XZ = 0.005776
PAR_S1_XZ = [-0.449144931, 0.793132642, -1.291292240, 0.536765147, -0.084910516, 0.146304242]
RES_S1_YZ = [0.0449, 0.0206, 0.01403, 0.0131, 0.0114]
PAR_S1_YZ = [[-0.09, -0.084325217, 0.487240052, 0.395496655, -0.187184874, 0.166128734],
             [0.0, -0.575280374, 0.070151974, 0.187260875, -0.099717108, 0.160128002],
             [-0.153367057, -0.583042532, 0.983374136, -0.712652874, 0.134743902, 0.465439107],
             [-0.268993212, 0.103899779, -0.588953942, 0.282356661, -0.067930741, 0.176282668],
             [-0.2, -0.724028910, 0.660065851, -0.327141529, 0.038426745, 0.094357571]]

def sigma_h(t):
    return 13.6 * (t + MMU) / t / (t + 2 * MMU)

def quartic_decay(x, par):
    u = x / par[-1] / 1000.0
    val = sum(par[i] * u ** i for i in range(len(par) - 1))
    return 1 + val * np.exp(-u)

def sigma1_xz(t):
    return np.sqrt((sigma_h(t) * quartic_decay(t, PAR_S1_XZ)) ** 2 + RES_S1_XZ ** 2)

def sigma1_yz(t, ivx):
    return np.sqrt((sigma_h(t) * quartic_decay(t, PAR_S1_YZ[ivx])) ** 2 + RES_S1_YZ[ivx] ** 2)

def ivx_of(vx):
    v = abs(vx)
    edges = [0, 0.1, 0.2, 0.35, 0.75, 1]
    for i in range(5):
        if edges[i] <= v < edges[i + 1]:
            return i
    return 4  # |vx| >= 1: the fixed bin


def robust_sigma(x):
    if len(x) < 5:
        return np.nan
    return 1.4826 * np.median(np.abs(np.asarray(x) - np.median(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tsv", default=None, help="mcs_joined.tsv to restrict to contained muons")
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("arm")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scratch = os.path.join(args.out, "clouds")
    os.makedirs(scratch, exist_ok=True)

    contained = None
    if args.tsv:
        contained = set()
        with open(args.tsv) as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            for line in fh:
                d = dict(zip(hdr, line.rstrip("\n").split("\t")))
                if d["isfc"] == "1" and float(d["ke_mcs"]) > 0:
                    contained.add((int(d["evt"]), int(d["segid"])))

    pulls_xz, pulls_yz, ts, ivxs = [], [], [], []
    nmu = 0
    prdirs = sorted(glob.glob(os.path.join(args.arm, "pr_evt*")))
    for prdir in prdirs:
        if args.max_events and nmu >= args.max_events:
            break
        evt = int(prdir.rsplit("pr_evt", 1)[1])
        froot = os.path.join(prdir, "tracking-pr.root")
        if not os.path.exists(froot):
            continue
        try:
            f = uproot.open(froot)
            if "T_kine" not in f or "T_rec_charge" not in f:
                continue
            tk = f["T_kine"].arrays(["kine_mcs_energy", "kine_mcs_segment_id"], library="np")
            rc = f["T_rec_charge"].arrays(["x", "y", "z", "cluster_id", "particle_id"], library="np")
        except Exception as err:                           # noqa: BLE001
            print(f"WARN {prdir}: {err}", file=sys.stderr)
            continue
        for i in range(len(tk["kine_mcs_energy"])):
            if tk["kine_mcs_energy"][i] <= 0:
                continue
            segid = int(tk["kine_mcs_segment_id"][i])
            if contained is not None and (evt, segid) not in contained:
                continue
            segcl = segid // 1000
            m = (rc["cluster_id"] == segcl) & (rc["particle_id"] == 13)
            if m.sum() < 30:
                continue
            pts = np.stack([rc["x"][m], rc["y"][m], rc["z"][m]], axis=1)
            cloud = os.path.join(scratch, f"evt{evt}_seg{segid}.txt")
            with open(cloud, "w") as fh:
                fh.write("%.17g %.17g %.17g\n" % tuple(pts[0]))
                fh.write("%.17g %.17g %.17g\n" % tuple(pts[-1]))
                for p in pts:
                    fh.write("%.17g %.17g %.17g\n" % tuple(p))
            out = cloud.replace(".txt", ".json")
            r = subprocess.run([DUMPER, "--txt", cloud, out],
                               capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(out):
                continue
            try:
                d = json.load(open(out))
            except Exception:                               # noqa: BLE001
                continue
            if d.get("rr", {}).get("early_return", True):
                continue
            tracklen = d["outputs"]["mu_tracklen"]
            nmu += 1
            for k, seg in enumerate(d["segments"]["per_seg"]):
                if k == 0:
                    continue  # -1 sentinel
                bx, cy = seg["angle_projB"], seg["angle_projC"]
                if isinstance(bx, str) or isinstance(cy, str):
                    continue
                resid = max(tracklen - seg["distance"], 1.0)
                t = float(ke_from_rr(resid))
                iv = ivx_of(seg["vx"])
                pulls_xz.append(bx / sigma1_xz(t))
                pulls_yz.append(cy / sigma1_yz(t, iv))
                ts.append(t)
                ivxs.append(iv)

    pulls_xz = np.array(pulls_xz)
    pulls_yz = np.array(pulls_yz)
    ts = np.array(ts)
    ivxs = np.array(ivxs)
    summary = [f"muons used: {nmu}  angle pairs: {len(pulls_xz)}"]
    summary.append("pull core width xz = %.3f   yz = %.3f  (transfer target 1.00)"
                   % (robust_sigma(pulls_xz), robust_sigma(pulls_yz)))
    for lo, hi in [(0, 200), (200, 400), (400, 800), (800, 1500), (1500, 1e9)]:
        m = (ts >= lo) & (ts < hi)
        if m.sum() < 10:
            continue
        summary.append("  T in [%4d,%4s): N=%4d  xz=%.3f  yz=%.3f"
                       % (lo, "inf" if hi > 1e8 else int(hi), m.sum(),
                          robust_sigma(pulls_xz[m]), robust_sigma(pulls_yz[m])))
    for iv in range(5):
        m = ivxs == iv
        if m.sum() < 10:
            continue
        summary.append("  ivx=%d: N=%4d  yz=%.3f" % (iv, m.sum(), robust_sigma(pulls_yz[m])))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, p, lab in [(axes[0], pulls_xz, r"$\theta_{xz}$"), (axes[1], pulls_yz, r"$\theta_{yz}$")]:
        ax.hist(np.clip(p, -6, 6), bins=60, histtype="step", color="tab:blue")
        w = robust_sigma(p)
        x = np.linspace(-6, 6, 200)
        if len(p):
            ax.plot(x, len(p) * (12 / 60) * np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), "k--",
                    lw=1, label="unit Gaussian")
        ax.set_title(f"{lab} pull  (core width {w:.3f})")
        ax.set_xlabel(r"$\theta$ / $\sigma_{pred}$")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "mcs_pull_test.png"), dpi=150)

    with open(os.path.join(args.out, "pull_summary.txt"), "w") as fh:
        fh.write("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
