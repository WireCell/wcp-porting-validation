#!/usr/bin/env python3
"""doc 80 round 4 -- MCS vs range vs dQ/dx->dE/dx comparison + money plots.

Reads one or more knob-ON PR arms:
  - tracking-pr.root T_kine rows: kine_mcs_{energy,ambiguity,tracklen,
    range_energy,segment_id} + cluster_id + match_isFC (from the paired
    T_tagger row, same index -- pairing verified via cluster_id);
  - the driver's INFO sentinel in wct_pr_evt<ID>.log:
      mcs: source=... cluster=<cid> -> ke_MCS=... ke_range_toolkit=<MeV>
      ke_dqdx_toolkit=<MeV> ... cathode_drop=<n>/<m>
    joined on (event, cluster_id).

Populations (doc 80 sec 9.1): Part A = contained (match_isFC==1), range is
the truth proxy; Part B = exiting (match_isFC==0), visible range is a strict
LOWER bound on the true energy; crossers (cathode_drop>0) reported separately
until the sec-7.5 closure is demonstrated.

Usage: mcs80_analysis.py --out DIR ARM [ARM...]
Writes PNGs + a joined TSV + a summary text into DIR (must not pre-exist
unless --out-exist-ok).
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

MMU = 105.658  # MeV

SENT = re.compile(
    r"mcs: source=(?P<source>\S+) nseg=(?P<nseg>\d+) npoints=(?P<npoints>\d+) "
    r"len=(?P<len>[\d.]+)cm seg_id=(?P<segid>-?\d+) cluster=(?P<cid>-?\d+) -> "
    r"ke_MCS=(?P<ke>-?[\d.]+) MeV amb=(?P<amb>[\d.eE+-]+) tracklen=(?P<tracklen>-?[\d.]+)cm "
    r"ke_range=(?P<kerange>-?[\d.]+) MeV ke_range_toolkit=(?P<kert>-?[\d.]+) MeV "
    r"ke_dqdx_toolkit=(?P<kedq>-?[\d.]+) MeV \(nsegs14=(?P<nsegs14>\d+) bad_path=(?P<bad>\w+) "
    r"cathode_drop=(?P<cdrop>\d+)/(?P<cmask>\d+)\)")


def collect(arms):
    rows = []
    for arm in arms:
        for prdir in sorted(glob.glob(os.path.join(arm, "pr_evt*"))):
            evt = int(prdir.rsplit("pr_evt", 1)[1])
            froot = os.path.join(prdir, "tracking-pr.root")
            flogs = glob.glob(os.path.join(prdir, "wct_pr_evt*.log"))
            if not os.path.exists(froot) or not flogs:
                continue
            sent = {}
            with open(flogs[0], errors="replace") as fh:
                for line in fh:
                    m = SENT.search(line)
                    if m:
                        sent[int(m.group("cid"))] = m.groupdict()
            try:
                f = uproot.open(froot)
                if "T_kine" not in f:
                    continue
                tk = f["T_kine"].arrays(library="np")
                tt = f["T_tagger"].arrays(
                    ["act_cluster_id", "act_fc", "act_stm", "act_tgm", "act_lm",
                     "act_is_selected", "act_evaluated"], library="np") \
                    if "T_tagger" in f else None
            except Exception as err:                       # noqa: BLE001
                print(f"WARN {prdir}: {err}", file=sys.stderr)
                continue
            if "kine_mcs_energy" not in tk:
                continue
            n = len(tk["kine_mcs_energy"])
            for i in range(n):
                cid = int(tk["cluster_id"][i]) if "cluster_id" in tk else -1
                s = sent.get(cid)
                # the measured muon's activity: join act_* on the SEGMENT's
                # cluster (kine_mcs_segment_id // 1000)
                fc = stm = tgm = lm = sel = -1
                segcl = int(tk["kine_mcs_segment_id"][i]) // 1000
                if tt is not None and i < len(tt["act_cluster_id"]):
                    ac = tt["act_cluster_id"][i]
                    for j in range(len(ac)):
                        if int(ac[j]) == segcl and int(tt["act_evaluated"][i][j]):
                            fc = int(tt["act_fc"][i][j])
                            stm = int(tt["act_stm"][i][j])
                            tgm = int(tt["act_tgm"][i][j])
                            lm = int(tt["act_lm"][i][j])
                            sel = int(tt["act_is_selected"][i][j])
                            break
                rows.append(dict(
                    arm=os.path.basename(arm.rstrip("/")), evt=evt, row=i, cid=cid,
                    ke_mcs=float(tk["kine_mcs_energy"][i]),
                    amb=float(tk["kine_mcs_ambiguity"][i]),
                    tracklen=float(tk["kine_mcs_tracklen"][i]),
                    ke_mcs_range=float(tk["kine_mcs_range_energy"][i]),
                    segid=int(tk["kine_mcs_segment_id"][i]),
                    isfc=fc, stm=stm, tgm=tgm, lm=lm, act_sel=sel,
                    ke_range_tk=float(s["kert"]) if s else -1,
                    ke_dqdx_tk=float(s["kedq"]) if s else -1,
                    sel_len=float(s["len"]) if s else -1,
                    cdrop=int(s["cdrop"]) if s else -1,
                    cmask=int(s["cmask"]) if s else -1,
                    nseg14=int(s["nsegs14"]) if s else -1,
                ))
    return rows


def p_from_ke(ke):
    return np.sqrt(ke * ke + 2.0 * ke * MMU)


def med_mad(x):
    if len(x) == 0:
        return np.nan, np.nan
    m = np.median(x)
    return m, 1.4826 * np.median(np.abs(x - m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-exist-ok", action="store_true")
    ap.add_argument("arms", nargs="+")
    args = ap.parse_args()
    if os.path.exists(args.out) and not args.out_exist_ok:
        sys.exit(f"refusing to write into existing {args.out} (M13); use a fresh dir")
    os.makedirs(args.out, exist_ok=True)

    rows = collect(args.arms)
    ran = [r for r in rows if r["ke_mcs"] > 0]
    print(f"T_kine rows: {len(rows)}; MCS ran on {len(ran)}")

    # joined TSV (the record the doc quotes)
    keys = ["arm", "evt", "row", "cid", "segid", "ke_mcs", "amb", "tracklen",
            "ke_mcs_range", "ke_range_tk", "ke_dqdx_tk", "sel_len", "isfc",
            "stm", "tgm", "lm", "act_sel", "cdrop", "cmask", "nseg14"]
    with open(os.path.join(args.out, "mcs_joined.tsv"), "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[k]) for k in keys) + "\n")

    A = [r for r in ran if r["isfc"] == 1 and r["ke_range_tk"] > 0]      # contained
    B = [r for r in ran if r["isfc"] == 0 and r["ke_range_tk"] > 0]      # exiting
    Ax = [r for r in A if r["cdrop"] <= 0]     # non-crossers
    Ac = [r for r in A if r["cdrop"] > 0]      # cathode crossers

    summary = []
    nstm = len([r for r in A if r.get("stm") == 1])
    summary.append(f"rows={len(rows)} ran={len(ran)} contained(A)={len(A)} "
                   f"[non-cross {len(Ax)} / cross {len(Ac)}; stm-tagged {nstm}] "
                   f"exiting(B)={len(B)}")

    def resid(rs, num_key):
        return np.array([(r[num_key] - r["ke_range_tk"]) / r["ke_range_tk"] for r in rs])

    # ---- money plot 1: E_MCS vs E_range scatter (contained) ----
    fig, ax = plt.subplots(figsize=(6, 6))
    for rs, c, lab in [(Ax, "tab:blue", "contained, non-crossing"),
                       (Ac, "tab:red", "contained, cathode crosser")]:
        ax.scatter([r["ke_range_tk"] / 1000 for r in rs], [r["ke_mcs"] / 1000 for r in rs],
                   s=14, alpha=0.7, c=c, label=f"{lab} (N={len(rs)})")
    lim = ax.get_xlim()[1] if A else 2
    hi = max(1.0, lim)
    ax.plot([0, hi], [0, hi], "k--", lw=1)
    ax.set_xlabel("KE range (toolkit cal_kine_range) [GeV]")
    ax.set_ylabel("KE MCS [GeV]")
    ax.set_title("MCS vs range, stopping/contained muons")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "mcs_vs_range_scatter.png"), dpi=150)

    # ---- money plot 2: MCS momentum distribution ----
    fig, ax = plt.subplots(figsize=(6, 4))
    pk = p_from_ke(np.array([r["ke_mcs"] for r in ran])) / 1000.0
    ax.hist(pk, bins=40, range=(0, max(2.0, np.percentile(pk, 99) if len(pk) else 2.0)),
            histtype="step", color="tab:blue", label=f"all MCS muons (N={len(pk)})")
    if A:
        pa = p_from_ke(np.array([r["ke_mcs"] for r in A])) / 1000.0
        ax.hist(pa, bins=40, range=ax.get_xlim(), histtype="step", color="tab:green",
                label=f"contained (N={len(pa)})")
    ax.set_xlabel("p_MCS [GeV/c]")
    ax.set_ylabel("muons")
    ax.set_title("MCS muon momentum")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "mcs_momentum_dist.png"), dpi=150)

    # ---- money plot 3: fractional residuals ----
    fig, ax = plt.subplots(figsize=(6, 4))
    if A:
        rm = resid(A, "ke_mcs")
        rd = np.array([(r["ke_dqdx_tk"] - r["ke_range_tk"]) / r["ke_range_tk"]
                       for r in A if r["ke_dqdx_tk"] > 0])
        bins = np.linspace(-1, 1, 41)
        ax.hist(np.clip(rm, -1, 1), bins=bins, histtype="step", color="tab:blue",
                label="(MCS-range)/range  med=%.3f MAD=%.3f" % med_mad(rm))
        ax.hist(np.clip(rd, -1, 1), bins=bins, histtype="step", color="tab:orange",
                label="(dQdx-range)/range  med=%.3f MAD=%.3f" % med_mad(rd))
        summary.append("PartA MCS  bias/res = %.4f / %.4f" % med_mad(rm))
        summary.append("PartA dQdx bias/res = %.4f / %.4f" % med_mad(rd))
        # the sec-9.1 headline band (range well-measured AND >=3 MCS segments)
        Ab = [r for r in A if 100 <= r["sel_len"] <= 250]
        if Ab:
            summary.append("PartA MCS  bias/res (100-250cm, N=%d) = %.4f / %.4f"
                           % (len(Ab), *med_mad(resid(Ab, "ke_mcs"))))
            rdb = [r for r in Ab if r["ke_dqdx_tk"] > 0]
            summary.append("PartA dQdx bias/res (100-250cm, N=%d) = %.4f / %.4f"
                           % (len(rdb), *med_mad(resid(rdb, "ke_dqdx_tk"))))
        if Ax:
            summary.append("PartA non-cross MCS bias/res = %.4f / %.4f" % med_mad(resid(Ax, "ke_mcs")))
        if Ac:
            summary.append("PartA crosser  MCS bias/res = %.4f / %.4f" % med_mad(resid(Ac, "ke_mcs")))
    ax.set_xlabel("fractional residual vs range")
    ax.set_ylabel("muons")
    ax.set_title("contained muons: MCS and dQ/dx vs range")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "mcs_residuals.png"), dpi=150)

    # ---- plot 4: bias/resolution vs length ----
    fig, ax = plt.subplots(figsize=(6, 4))
    edges = [40, 80, 120, 160, 200, 260, 320, 400]
    for key, c, lab in [("ke_mcs", "tab:blue", "MCS"), ("ke_dqdx_tk", "tab:orange", "dQdx")]:
        xs, ms, es = [], [], []
        for lo, hi2 in zip(edges[:-1], edges[1:]):
            rs = [r for r in A if lo <= r["sel_len"] < hi2 and r[key] > 0]
            if len(rs) < 3:
                continue
            m, s = med_mad(resid(rs, key))
            xs.append(0.5 * (lo + hi2))
            ms.append(m)
            es.append(s)
        ax.errorbar(xs, ms, yerr=es, fmt="o-", color=c, capsize=3, label=lab)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("selected muon length [cm]")
    ax.set_ylabel("median +- MAD of (E - E_range)/E_range")
    ax.set_title("bias/resolution vs length, contained muons")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "mcs_bias_vs_length.png"), dpi=150)

    # ---- money plot 5: exiting muons, lower-bound check ----
    fig, ax = plt.subplots(figsize=(6, 4))
    if B:
        ratio = np.array([r["ke_mcs"] / r["ke_range_tk"] for r in B])
        lens = np.array([r["sel_len"] for r in B])
        ax.scatter(lens, ratio, s=14, alpha=0.7, c="tab:purple")
        ax.axhline(1, color="k", lw=1, ls="--")
        nviol = int((ratio < 1).sum())
        summary.append(f"PartB exiting: N={len(B)}  E_MCS<E_range(visible) violations={nviol} "
                       f"({100.0*nviol/len(B):.1f}%)  median ratio={np.median(ratio):.2f}")
        ax.set_title(f"exiting muons: E_MCS / E_range(visible)  ({nviol}/{len(B)} below 1)")
    ax.set_xlabel("visible track length [cm]")
    ax.set_ylabel("E_MCS / E_range(visible)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "mcs_exiting_ratio.png"), dpi=150)

    # ---- plot 6: ambiguity validation (Part C) ----
    fig, ax = plt.subplots(figsize=(6, 4))
    if A:
        ambs = np.array([r["amb"] for r in A])
        res = np.abs(resid(A, "ke_mcs"))
        qs = np.quantile(ambs, np.linspace(0, 1, 6))
        xs, ys, ns = [], [], []
        for lo, hi2 in zip(qs[:-1], qs[1:]):
            m = (ambs >= lo) & (ambs <= hi2)
            if m.sum() == 0:
                continue
            xs.append(0.5 * (lo + hi2))
            ys.append(np.median(res[m]))
            ns.append(int(m.sum()))
        ax.plot(xs, ys, "o-", color="tab:red")
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(str(n), (x, y), fontsize=7, xytext=(2, 4), textcoords="offset points")
        summary.append("PartC ambiguity quintile medians |resid|: " +
                       " ".join("%.3f" % y for y in ys))
    ax.set_xscale("log")
    ax.set_xlabel("ambiguity_MCS (quintile centers)")
    ax.set_ylabel("median |(E_MCS-E_range)/E_range|")
    ax.set_title("ambiguity vs residual (monotone = score is informative)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "mcs_ambiguity_check.png"), dpi=150)

    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
