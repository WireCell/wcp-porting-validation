#!/usr/bin/env python3
"""Does doc 55 section 7g's dQ/dx model still describe a *population* of protons?

Doc 55 fitted its free-power Modified Box to 12 muons and ONE proton, wrote the
resulting five tables into `nusel_display/stm_ref_dqdx.json`, and left section 9
item 3 open: one track cannot separate a genuine recombination statement from a
track-specific reconstruction effect.  Doc 62 supplies 12 hand-identified
protons.  This script asks the only question that matters for the model:

  hold the committed parameters FIXED -- k, p, C exactly as they sit in the
  json's `_meta.canonical_keys` -- and see whether the enlarged proton
  population lands on that curve.

Nothing is fitted here.  The refit is `fit_recombination.py --points ...
--plane rr`, reported beside these numbers in the doc; keeping the two apart is
the point, because a model that has to be refitted to follow new data has not
been tested by it.

Also reported, because both were open questions doc 55 could not close with one
track:
  * the proton/muon ratio at matched dE/dx, per bin (doc 55 section 7b, 7g.6);
  * the per-track ratio against drift time -- doc 55's single proton sat at
    1106-1251 us, the most attenuated corner of the sample, so its normalisation
    and the electron lifetime were entangled.  This population spans the drift.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 dqdx_rr_sample/proton_model_check.py \
      --points dqdx_rr_sample/sample_points_p12.tsv \
      -o dqdx_rr_sample/proton_vs_frozen_model_p12.png
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(HERE)
SBND_DQDX = ("/nfs/data/1/xqian/toolkit-dev/energy_loss/pion_travel/"
             "stopping_ave_dQ_dx_sbnd.root")
JSON = os.path.join(TOP, "nusel_display", "stm_ref_dqdx.json")
BOX_FUDGE = 0.85
DEDX_BOTH = 10.5      # above this only protons constrain a joint fit


def load_fr():
    spec = importlib.util.spec_from_file_location(
        "fr", os.path.join(HERE, "fit_recombination.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def table_curve(fr, graphs, part, name, p, C, rr=None, nsub=10):
    """convert_field.C's recipe: recombination pointwise, then 1 cm bin average."""
    rr = np.arange(60) + 0.5 if rr is None else np.asarray(rr, float)
    sub = fr.dedx_samples(graphs, part, rr, np.ones_like(rr), nsub=nsub,
                          lo_clip=0.0)
    return C * np.mean(fr.MODELS[name][4](sub, *p) * sub, axis=1) / fr.W_ION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=os.path.join(HERE,
                                                     "sample_points_p12.tsv"))
    ap.add_argument("-o", "--out", help="output PNG")
    args = ap.parse_args()

    fr = load_fr()
    graphs = fr.dedx_graphs()

    ck = json.load(open(JSON))["_meta"]["canonical_keys"]
    frozen = (ck["k"], ck["p"])
    frozen_C = ck["C"]
    print(f"frozen model from {os.path.relpath(JSON, TOP)}:")
    print(f"  {ck['model']}")
    print(f"  A = {ck['A']}, k = {frozen[0]:.6f}, p = {frozen[1]:.6f}, "
          f"C = {frozen_C:.6f}   (fitted on 12 muons + 1 proton, "
          f"chi2/ndf = {ck['chi2_per_ndf']})")

    # the shipped Box tables, as a regression on the recipe
    f = uproot.open(SBND_DQDX)
    for part in ("muon", "proton"):
        tab = np.asarray(f[part].values("y"), float)
        cur = table_curve(fr, graphs, part, "box_fixed", [], BOX_FUDGE)
        rel = np.max(np.abs(cur / tab - 1))
        print(f"  recipe vs shipped {part} table: max rel dev {rel:.2e} "
              f"({'PASS' if rel < 2e-3 else 'FAIL'})")

    # ---- the data, binned exactly as the rr-plane fit bins it ---------------
    d = fr.read_tsv(args.points)
    part, rr, dq, dx = d["particle"], d["rr"], d["dqdx"], d["dx"]
    tid = np.array([f"{int(a)}_{int(b)}" for a, b in zip(d["event"], d["block"])])
    drift = d["drift_us"]
    sub = np.zeros((len(rr), fr.NSUB))
    for p in ("muon", "proton"):
        s = part == p
        sub[s] = fr.dedx_samples(graphs, p, rr[s], dx[s])
    de = np.mean(sub, axis=1)
    keep = (rr >= fr.RR_MIN) & (de <= fr.DEDX_MAX) & (dq > 0)
    rows = fr.bin_data_rr(part[keep], rr[keep], de[keep], dq[keep], tid[keep], 0.03)

    ntrk = {p: len({t for t in tid[part == p]}) for p in ("muon", "proton")}
    print(f"\nsample: {ntrk['muon']} muons, {ntrk['proton']} protons, "
          f"{keep.sum()} points in the fit domain")

    MOD = [("shipped Box", "box_fixed", [], BOX_FUDGE, "#0b0b0b"),
           ("frozen free power", "box_p", list(frozen), frozen_C, "#2a78d6")]

    print("\n=== the enlarged sample against the two FROZEN curves "
          "(nothing fitted) ===")
    prof = {}
    for p in ("muon", "proton"):
        sel = [r for r in rows if r["part"] == p]
        cen = np.array([r["rr"] for r in sel])
        val = np.array([r["dqdx"] for r in sel])
        err = np.array([r["dqdx"] * r["sig"] for r in sel])
        nt = [r["ntrk"] for r in sel]
        deb = np.array([r["dedx"] for r in sel])
        prof[p] = (cen, val, err, nt, deb)
        cs = {lab: table_curve(fr, graphs, p, name, pp, C, rr=cen)
              for lab, name, pp, C, _ in MOD}
        print(f"\n  {p}  ({ntrk[p]} tracks)")
        print(f"  {'rr (cm)':>9s} {'dE/dx':>7s} {'ntrk':>5s} {'data ke/cm':>11s} "
              f"{'+-%':>5s} " + " ".join(f"/{lab}".rjust(20) for lab in cs))
        for i in range(len(cen)):
            tag = "  *" if deb[i] > DEDX_BOTH else ""
            print(f"  {cen[i]:9.1f} {deb[i]:7.1f} {nt[i]:5d} {val[i]/1e3:11.1f} "
                  f"{err[i]/val[i]*100:5.1f} "
                  + " ".join(f"{val[i]/cs[lab][i]:20.3f}" for lab in cs) + tag)
        print(f"  {'median':>34s} {'':>11s} {'':>5s} "
              + " ".join(f"{np.median(val/cs[lab]):20.3f}" for lab in cs))
        print(f"  {'rms of ln(ratio) about 1':>34s} {'':>11s} {'':>5s} "
              + " ".join(f"{np.sqrt(np.mean(np.log(val/cs[lab])**2))*100:19.1f}%"
                         for lab in cs))
    print(f"\n  * = above dE/dx {DEDX_BOTH} MeV/cm; on the OLD sample only the "
          f"single proton\n    lived there, which is what made those bins the "
          f"least trustworthy in doc 55 section 10.3")

    # ---- proton / muon at matched dE/dx, the model-independent test ---------
    print("\n=== proton / muon at matched dE/dx (model-independent) ===")
    print("  a recombination model is a function of dE/dx alone, so it can "
          "describe\n  both particles only if this column is flat at 1")
    # TWO error columns, and they are not interchangeable.  `+-stat` is the
    # s.e.m. of the two medians alone.  `+-tot` folds in the same 3 % per-bin
    # systematic floor `fit_recombination.bin_data` uses, which is what doc 55
    # section 7b's error column is -- quoting the statistical one against 7b's
    # would read as the error shrinking when it is only a different definition.
    FLOOR = 0.03
    EDG = fr.EDGES
    print(f"  {'dE/dx bin':>16s} {'n mu':>6s} {'n p':>6s} {'muon':>9s} "
          f"{'proton':>9s} {'p/mu':>7s} {'+-stat%':>8s} {'+-tot%':>7s}")
    pm = []
    for lo, hi in zip(EDG[:-1], EDG[1:]):
        sm = np.where((part[keep] == "muon") & (de[keep] >= lo) & (de[keep] < hi))[0]
        sp = np.where((part[keep] == "proton") & (de[keep] >= lo) & (de[keep] < hi))[0]
        if len(sm) < 4 or len(sp) < 4:
            continue
        lm, lp = np.log(dq[keep][sm]), np.log(dq[keep][sp])
        vm, vp = np.exp(np.median(lm)), np.exp(np.median(lp))
        em, ep = np.std(lm) / np.sqrt(len(sm)), np.std(lp) / np.sqrt(len(sp))
        e = np.hypot(em, ep)
        et = np.hypot(np.hypot(em, FLOOR), np.hypot(ep, FLOOR))
        pm.append(vp / vm)
        print(f"  {lo:7.1f} -{hi:7.1f} {len(sm):6d} {len(sp):6d} {vm/1e3:9.1f} "
              f"{vp/1e3:9.1f} {vp/vm:7.3f} {e*100:8.1f} {et*100:7.1f}")
    pm = np.array(pm)
    print(f"  -> median proton/muon = {np.median(pm):.3f}, spread "
          f"{np.std(np.log(pm))*100:.1f} %, {len(pm)} bins")
    print(f"  +-tot is the column comparable with doc 55 sec 7b "
          f"({FLOOR*100:.0f} % systematic floor per particle); +-stat is the\n"
          f"  s.e.m. alone.  The muon side stays floor-limited here -- it is the "
          f"same 12 tracks\n  as doc 55, with 6-76 points per bin above "
          f"4 MeV/cm against the proton's 54-141.")

    # ---- per-track ratio vs drift ------------------------------------------
    print("\n=== per-track median data / frozen free power, vs drift ===")
    print("  doc 55's one proton sat at drift 1106-1251 us, so its offset and "
          "the\n  electron lifetime could not be separated; this population "
          "spans the drift")
    for p in ("muon", "proton"):
        print(f"\n  {p}")
        print(f"  {'event':>7s} {'blk':>4s} {'n':>5s} {'drift us':>9s} "
              f"{'/shipped Box':>13s} {'/frozen power':>14s}")
        dr_, rt_ = [], []
        for t in sorted(set(tid[keep][part[keep] == p])):
            m = (tid[keep] == t) & (part[keep] == p)
            r0 = np.median(dq[keep][m] / fr.point_model("box_fixed", [], sub[keep][m],
                                                        BOX_FUDGE))
            r1 = np.median(dq[keep][m] / fr.point_model("box_p", list(frozen),
                                                        sub[keep][m], frozen_C))
            dmean = float(np.mean(drift[keep][m]))
            dr_.append(dmean)
            rt_.append(r1)
            ev, blk = t.split("_")
            print(f"  {ev:>7s} {blk:>4s} {int(m.sum()):5d} {dmean:9.0f} "
                  f"{r0:13.3f} {r1:14.3f}")
        dr_, rt_ = np.array(dr_), np.array(rt_)
        if len(dr_) >= 4:
            a, b = np.polyfit(dr_, np.log(rt_), 1)
            tau = -1.0 / a / 1e3 if a < 0 else float("inf")
            print(f"    slope over the drift: {a*1290*100:+.1f} % across "
                  f"1290 us  ->  tau = {tau:.1f} ms"
                  + ("  (the muon population gives order 10 ms, doc 55 §7e)"
                     if p == "proton" else ""))

    if not args.out:
        return

    # ---- figure ------------------------------------------------------------
    grid = np.arange(60) + 0.5
    COL = {"muon": "#2a78d6", "proton": "#e34948"}
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6))

    ax = axes[0]
    for lab, name, p, C, col in MOD:
        for pt, ls in (("muon", "--"), ("proton", "-")):
            ax.plot(grid, table_curve(fr, graphs, pt, name, p, C) / 1e3, ls,
                    color=col, lw=1.9, zorder=5, label=f"{lab}, {pt}")
    for pt, mk in (("muon", "o"), ("proton", "s")):
        cen, val, err, nt, _ = prof[pt]
        ax.errorbar(cen, val / 1e3, yerr=err / 1e3, fmt=mk, color=COL[pt],
                    ms=8.0, mew=1.4, mec="white", lw=0, elinewidth=1.3,
                    capsize=3, zorder=9,
                    label=f"{pt}, average of {ntrk[pt]} tracks")
    ax.set_xlim(0, 60)
    ax.set_ylim(40, 300)
    ax.set_yscale("log")
    ax.set_yticks([40, 60, 80, 100, 150, 200, 300])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("residual range from the stopping end  [cm]")
    ax.set_ylabel("dQ/dx  [ke/cm]")
    ax.set_title(f"{ntrk['proton']} owner-identified protons + "
                 f"{ntrk['muon']} muons vs the FROZEN curves", fontsize=10.5)
    ax.legend(fontsize=7.8, loc="upper right", ncol=2, framealpha=0.95)
    ax.grid(alpha=0.2, which="both", lw=0.6)

    ax = axes[1]
    ax.axhline(1.0, color="#a3a29b", lw=1.2, ls=":")
    for pt, mk in (("muon", "o"), ("proton", "s")):
        cen, val, err, nt, deb = prof[pt]
        for lab, name, p, C, col in MOD:
            c = table_curve(fr, graphs, pt, name, p, C, rr=cen)
            r = val / c
            ls = "--" if pt == "muon" else "-"
            ax.plot(cen, r, ls, color=col, lw=1.3, alpha=0.9, zorder=3)
            ax.errorbar(cen, r, yerr=err / c, fmt=mk, color=col, ms=6.5, mew=1.2,
                        mec="white", lw=0, elinewidth=1.0, zorder=5,
                        label=f"{pt} / {lab}   "
                              f"(rms {np.sqrt(np.mean(np.log(r)**2))*100:.1f} %)")
    ax.set_xlim(0, 60)
    ax.set_ylim(0.85, 1.25)
    ax.set_xlabel("residual range from the stopping end  [cm]")
    ax.set_ylabel("data / model")
    ax.set_title("nothing is fitted here: k, p, C are the committed json values",
                 fontsize=10.5)
    ax.legend(fontsize=7.4, loc="upper left", ncol=2, framealpha=0.95)
    ax.grid(alpha=0.2, lw=0.6)

    fig.suptitle("doc 55 §11 -- the doc-62 proton population against the "
                 "free-power model fitted WITHOUT it   "
                 f"(A = {ck['A']}, k = {frozen[0]:.4f}, p = {frozen[1]:.4f}, "
                 f"C = {frozen_C:.4f}; uncalibrated MCP2025C data)",
                 fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
