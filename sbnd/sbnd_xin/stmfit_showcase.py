#!/usr/bin/env python3
"""Showcase plot + numbers for one STM fit: dQ/dx vs residual range.

Overlays the fitted dQ/dx of one (cluster, pass) block against the two
references the STM tagger itself uses: the muon dE/dx + recombination table
from the compiled config (nusel_display/stm_ref_dqdx.json, the exact curve
`eval_stm` tests against) and the flat 50 ke/cm MIP line.  Also prints the
binned profile and the ratio to expectation that doc 42 quotes.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 stmfit_showcase.py -r archive/stm-docs40-49/work-mcp10-stmon -e 286241 -b 80 \
      -o showcase-stmfit-286241/dqdx_286241_blk80.png
"""
import argparse
import json
import os

import numpy as np
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

BINS = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40),
        (40, 60), (60, 100), (100, 1e9)]


def ref_curve(path, key="MuonDeDxBox"):
    """The muon curve TaggerCheckSTM actually compares against.

    Pinned to the `*Box` key.  Since doc 55 sec 10 the json's canonical
    `MuonDeDx` key carries the free-power FIT, not the shipped config table;
    this script's numbers are quoted in doc 42 against the config table, so it
    must not silently follow the fit -- that is exactly the failure doc 55
    sec 4a had to write up for the uBooNE-era json.  Falls back for an older
    json that has only the canonical name.
    """
    d = json.load(open(path))
    d = d[key] if key in d else d["MuonDeDx"]
    x = d["start"] + d["step"] * np.arange(len(d["values"]))
    return x, np.array(d["values"], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--root", required=True, help="work root, e.g. archive/stm-docs40-49/work-mcp10-stmon")
    ap.add_argument("-e", "--event", required=True, help="event id")
    ap.add_argument("-b", "--block", type=int, required=True,
                    help="T_rec_charge block id = cluster_id*10 + pass")
    ap.add_argument("-o", "--out", help="output PNG (optional)")
    ap.add_argument("--ref", default="nusel_display/stm_ref_dqdx.json")
    args = ap.parse_args()

    fp = os.path.join(args.root, f"nusel_evt{args.event}", "tracking-stm.root")
    f = uproot.open(fp)
    t = f["T_rec_charge"].arrays(["x", "q", "nq", "rr", "ndf", "status", "pass",
                                  "reduced_chi2"], library="np")
    tr = f["Trun"].arrays(library="np")
    m = t["ndf"] == args.block
    if not m.any():
        raise SystemExit(f"no block {args.block} in {fp}; have "
                         f"{sorted(set(t['ndf'].tolist()))}")

    dQ = (t["q"][m] - tr["dQdx_offset"][0]) / tr["dQdx_scale"][0]
    dx = t["nq"][m]
    dqdx = np.where(dx > 0, dQ / np.maximum(dx, 1e-9), 0.0)
    rr = t["rr"][m]
    tpc = (t["x"][m] > 0).astype(int)

    print(f"{fp}  block {args.block} (cluster {args.block // 10}, pass {args.block % 10})")
    print(f"  status={t['status'][m][0]}  npts={m.sum()}  L={rr.max():.1f} cm  "
          f"TPC{'0/1' if len(set(tpc.tolist())) > 1 else tpc[0]}  "
          f"median reduced_chi2={np.median(t['reduced_chi2'][m]):.2f}")

    rx, rv = ref_curve(args.ref)
    # The muon table is tabulated on a finite rr range (0.5..59.5 cm for the
    # 60-entry MuonDeDx set).  np.interp CLAMPS outside it, so beyond rx.max()
    # the "expectation" would just be the last table value repeated -- report
    # those bins as out of domain instead of quoting a fake ratio.
    exp = np.interp(rr, rx, rv)
    rr_max_tab = rx.max()
    print(f"  muon table domain: rr {rx.min():.1f} - {rr_max_tab:.1f} cm")
    print(f"  {'rr bin (cm)':>14s} {'n':>4s} {'fit ke/cm':>10s} {'exp ke/cm':>10s} {'ratio':>7s}")
    for lo, hi in BINS:
        s = (rr >= lo) & (rr < hi)
        if not s.sum():
            continue
        fit = np.median(dqdx[s]) / 1e3
        hi_s = "inf" if hi > 1e8 else f"{hi:.0f}"
        if lo >= rr_max_tab:
            print(f"  {lo:6.0f} - {hi_s:>5s} {s.sum():4d} {fit:10.1f} "
                  f"{'(no table)':>10s} {'-':>7s}")
            continue
        ex = np.median(exp[s]) / 1e3
        flag = " *partly beyond table" if hi > rr_max_tab else ""
        print(f"  {lo:6.0f} - {hi_s:>5s} {s.sum():4d} {fit:10.1f} {ex:10.1f} "
              f"{fit/ex:7.2f}{flag}")
    far = rr > 40
    if far.sum():
        print(f"  plateau (rr>40cm): fit {np.median(dqdx[far])/1e3:.1f} ke/cm "
              f"vs flat ref 50 -> ratio {np.median(dqdx[far])/50e3:.2f}")

    if args.out:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for tpc_id, mark, col in ((0, "o", "tab:blue"), (1, "^", "tab:red")):
            s = tpc == tpc_id
            if s.sum():
                ax.plot(rr[s], dqdx[s] / 1e3, mark, ms=3.5, color=col, alpha=0.75,
                        label=f"fitted dQ/dx (TPC{tpc_id})")
        ax.plot(rx, rv / 1e3, "-", color="k", lw=1.6,
                label="muon expectation (eval_stm table)")
        ax.axhline(50, ls="--", color="gray", lw=1.2, label="flat 50 ke/cm MIP")
        ax.set_xlabel("residual range from the candidate stopping end [cm]")
        ax.set_ylabel("dQ/dx [ke/cm]")
        ax.set_xlim(0, min(rr.max() * 1.02, 200))
        ax.set_ylim(0, max(160, np.percentile(dqdx / 1e3, 99) * 1.1))
        ax.set_title(f"SBND evt {args.event}, cluster {args.block // 10} "
                     f"pass {args.block % 10} (accepted STM)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.out, dpi=140)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
