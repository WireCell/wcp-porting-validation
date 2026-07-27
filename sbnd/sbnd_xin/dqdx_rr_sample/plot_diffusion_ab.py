#!/usr/bin/env python3
"""Doc 55 section 13: the dQ/dx expectation curves against BOTH diffusion arms.

Doc 66 reverted the track fit's diffusion coefficients from DL = 6.5781 /
DT = 13.1349 to sbndcode's DL = 4.0 / DT = 8.8 cm^2/s.  Those constants enter
only the *predicted charge footprint* the fit apportions measured charge with,
so `dx` and the trajectory do not move at all (doc 55 section 12.1) and the
whole effect is in `dQ`.  The question this figure answers is the one the owner
asked: **do the shipped dQ/dx expectations still describe the data?**

Nothing is fitted.  Both arms are drawn against the SAME two frozen curves --
the shipped Modified-Box tables `TaggerCheckSTM` compares against, and the
free-power curve committed in `nusel_display/stm_ref_dqdx.json` -- so any
movement between the two marker families is diffusion and nothing else.  The
tracks are identical between the arms by construction (the new arm replays the
old arm's selected (event, block) list; see `collect_dqdx_rr_sample.py
--force-list`), so this is not a sample change either.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 dqdx_rr_sample/plot_diffusion_ab.py \
      --old dqdx_rr_sample/sample_points_p12_d66old.tsv \
      --new dqdx_rr_sample/sample_points_p12_d66newF.tsv \
      -o dqdx_rr_sample/dqdx_vs_rr_diffusion_ab.png
"""
import argparse
import importlib.util
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(HERE)
JSON = os.path.join(TOP, "nusel_display", "stm_ref_dqdx.json")
BOX_FUDGE = 0.85
DEDX_BOTH = 10.5

# House palette, unchanged from every other figure in dqdx_rr_sample/: hue
# carries the *reference curve* (ink = the shipped Box tables, blue = the
# committed free-power curve).  The diffusion arm is carried by marker fill and
# line style, never by colour alone -- the two arms are the same entities.
INK, BLUE = "#0b0b0b", "#2a78d6"
ARMS = [("old", "6.5781 / 13.1349", "--", "none", "o"),
        ("new", "4.0 / 8.8  (shipped)", "-", "full", "s")]


def load_fr():
    spec = importlib.util.spec_from_file_location(
        "fr", os.path.join(HERE, "fit_recombination.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_pmc():
    spec = importlib.util.spec_from_file_location(
        "pmc", os.path.join(HERE, "proton_model_check.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def profiles(fr, graphs, points):
    """Bin one arm exactly as the rr-plane fit bins it."""
    d = fr.read_tsv(points)
    part, rr, dq, dx = d["particle"], d["rr"], d["dqdx"], d["dx"]
    tid = np.array([f"{int(a)}_{int(b)}" for a, b in zip(d["event"], d["block"])])
    sub = np.zeros((len(rr), fr.NSUB))
    for p in ("muon", "proton"):
        s = part == p
        sub[s] = fr.dedx_samples(graphs, p, rr[s], dx[s])
    de = np.mean(sub, axis=1)
    keep = (rr >= fr.RR_MIN) & (de <= fr.DEDX_MAX) & (dq > 0)
    rows = fr.bin_data_rr(part[keep], rr[keep], de[keep], dq[keep], tid[keep], 0.03)
    out = {}
    for p in ("muon", "proton"):
        sel = [r for r in rows if r["part"] == p]
        out[p] = dict(cen=np.array([r["rr"] for r in sel]),
                      val=np.array([r["dqdx"] for r in sel]),
                      err=np.array([r["dqdx"] * r["sig"] for r in sel]),
                      dedx=np.array([r["dedx"] for r in sel]),
                      ntrk=len({t for t in tid[part == p]}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=os.path.join(
        HERE, "sample_points_p12_d66old.tsv"))
    ap.add_argument("--new", default=os.path.join(
        HERE, "sample_points_p12_d66newF.tsv"))
    ap.add_argument("-o", "--out", default=os.path.join(
        HERE, "dqdx_vs_rr_diffusion_ab.png"))
    args = ap.parse_args()

    fr = load_fr()
    pmc = load_pmc()
    graphs = fr.dedx_graphs()

    ck = json.load(open(JSON))["_meta"]["canonical_keys"]
    MOD = [("shipped Modified Box", "box_fixed", [], BOX_FUDGE, INK),
           ("committed free power", "box_p", [ck["k"], ck["p"]], ck["C"], BLUE)]

    prof = {a: profiles(fr, graphs, p)
            for a, p in (("old", args.old), ("new", args.new))}

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.6),
                             gridspec_kw=dict(height_ratios=[1.25, 1]))
    summary = {}

    for col, part in enumerate(("muon", "proton")):
        rrf = np.arange(0.5, 60, 0.25)
        curves = {lab: pmc.table_curve(fr, graphs, part, name, pp, C, rr=rrf)
                  for lab, name, pp, C, _ in MOD}

        # ---- top row: the data on the two frozen curves ---------------------
        ax = axes[0][col]
        for lab, _, _, _, col_ in MOD:
            ax.plot(rrf, curves[lab] / 1e3, "-", color=col_, lw=2.0, zorder=3,
                    label=f"{part}, {lab}")
        for arm, dl, ls, fill, mk in ARMS:
            q = prof[arm][part]
            ax.errorbar(q["cen"], q["val"] / 1e3, yerr=q["err"] / 1e3, fmt=mk,
                        ls=ls, color="#e34948", ms=6.5, lw=1.1, elinewidth=1.1,
                        capsize=2.5, fillstyle=fill, mew=1.3, alpha=0.9, zorder=5,
                        label=f"DL/DT = {dl}")
        ax.set_yscale("log")
        ax.set_xlim(0, 60)
        ax.set_ylabel("dQ/dx  [ke/cm]")
        ax.set_title(f"{part} — {prof['old'][part]['ntrk']} tracks, identical "
                     "in both arms", fontsize=10)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
        ax.grid(alpha=0.2, lw=0.6, which="both")

        # ---- bottom row: ratio to each frozen curve, both arms ---------------
        ax = axes[1][col]
        ax.axhline(1.0, ls=":", color="#a3a29b", lw=1.4, zorder=1)
        for lab, name, pp, C, col_ in MOD:
            for arm, dl, ls, fill, mk in ARMS:
                q = prof[arm][part]
                cs = pmc.table_curve(fr, graphs, part, name, pp, C, rr=q["cen"])
                r = q["val"] / cs
                summary[(part, lab, arm)] = (
                    float(np.median(r)),
                    float(np.sqrt(np.mean(np.log(r) ** 2)) * 100))
                ax.plot(q["cen"], r, mk, ls=ls, color=col_, ms=6.0, lw=1.2,
                        fillstyle=fill, mew=1.3, alpha=0.9, zorder=4,
                        label=(f"/ {lab}, {'4.0/8.8' if arm == 'new' else 'old'}"))
        # mark where only protons constrain the curves
        hi = prof["old"][part]["cen"][prof["old"][part]["dedx"] > DEDX_BOTH]
        if len(hi):
            ax.axvspan(0, hi.max(), color="#a3a29b", alpha=0.10, zorder=0)
        ax.set_xlim(0, 60)
        ax.set_ylim(0.85, 1.30)
        ax.set_xlabel("residual range from the stopping end  [cm]")
        ax.set_ylabel("data / expectation")
        ax.legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.95)
        ax.grid(alpha=0.2, lw=0.6)

        txt = "\n".join(
            f"{lab}:  {summary[(part, lab, 'old')][0]:.3f} / "
            f"{summary[(part, lab, 'old')][1]:.1f}%  ->  "
            f"{summary[(part, lab, 'new')][0]:.3f} / "
            f"{summary[(part, lab, 'new')][1]:.1f}%"
            for lab, _, _, _, _ in MOD)
        ax.text(0.985, 0.04, "median / rms of ln(ratio),  old -> new\n" + txt,
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                family="monospace",
                bbox=dict(fc="white", ec="#d5d4cf", lw=0.8, alpha=0.95))

    fig.suptitle(
        "Doc 55 §13 — the SBND dQ/dx expectations against both diffusion arms "
        "(doc 66).  Nothing is fitted: both arms are\ndrawn against the same "
        "frozen curves, on the same tracks, from the same 1000 events and the "
        "same binary.\nShaded band = the residual ranges where dE/dx > "
        f"{DEDX_BOTH} MeV/cm, which only protons constrain.",
        fontsize=9.5, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")

    print(f"\n{'particle':>8s} {'reference':>22s} {'arm':>4s} {'median':>7s} "
          f"{'rms %':>7s}")
    for (part, lab, arm), (med, rms) in summary.items():
        print(f"{part:>8s} {lab:>22s} {arm:>4s} {med:7.3f} {rms:7.1f}")


if __name__ == "__main__":
    main()
