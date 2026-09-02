#!/usr/bin/env python3
"""Michel-electron candidates and their energy spectrum (doc pdvd/25 M5, ask 6).

Selection, per matched flash bundle (calib-pr-evt<ID>.json candidates[]):
  1. the bundle's main cluster is STM-tagged and NOT TGM-tagged in the PR log;
  2. the cosmic tagger's stopped-muon + Michel tests ran and fired
     (cosmict_7_filled and cosmict_flag_7, or the flag-8 pair) -- the flags
     6-8 block of NeutrinoTaggerCosmic (prototype NeutrinoID_cosmic_tagger.h
     268-588), whose michel_ele is the highest-energy shower starting at the
     main vertex;
  3. that shower is re-identified from the dump: start_vertex_id == the main
     vertex id, highest kine_best (MeV) -> michel_E.
Rows are written with the muon length, the shower length and its distance
from the main vertex; the histogram marks the Michel endpoint (52.8 MeV).
--loose keeps rows where step 2 did not fire but a shower sits at the main
vertex of an STM-tagged main (diagnostic of the flags-6-8 acceptance test).

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 stm/michel_energy.py --tag stm1 -o docs/pics/pdvd_michel_energy.png --tsv stm/michel_candidates.tsv
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pr_census import parse_log, michel_rows  # noqa: E402

MICHEL_ENDPOINT_MEV = 52.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stm1")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--tsv")
    ap.add_argument("--loose", action="store_true")
    ap.add_argument("--no-tagger", action="store_true", help="do not require the STM verdict")
    args = ap.parse_args()
    rows_out = []
    for d in sorted(glob.glob(os.path.join(PDVD, "work", f"*_{args.tag}"))):
        ev = re.sub(r"_%s$" % args.tag, "", os.path.basename(d)); run, idx = ev.split("_")
        dumps = glob.glob(os.path.join(d, "calib-pr-evt*.json"))
        if not dumps:
            continue
        v, _, _ = parse_log(os.path.join(d, f"wct_pr_{run}_{idx}.log"))
        dump = json.load(open(dumps[0])); evtno = dump.get("meta", {}).get("eventNo")
        for r in michel_rows(dump):
            verdict = v.get(r["main_cluster"], {})
            stm_ok = args.no_tagger or (verdict.get("stm") == 1 and verdict.get("tgm") != 1)
            fired = bool((r["f7_filled"] and r["f7"]) or (r["f8_filled"] and r["f8"]))
            if not stm_ok:
                continue
            if not fired and not (args.loose and r["michel_E"] is not None):
                continue
            rows_out.append(dict(event=ev, evtno=evtno, **r, stm=verdict.get("stm"), tgm=verdict.get("tgm"), fc=verdict.get("fc"), fired=int(fired)))
    E = np.array([r["michel_E"] for r in rows_out if r["michel_E"] is not None], float)
    print(f"{len(rows_out)} STM bundles with a Michel candidate ({int(sum(r['fired'] for r in rows_out))} with flags 6-8 fired); energies: n={len(E)}"
          + (f" median={np.median(E):.1f} MeV max={E.max():.1f} MeV, above endpoint: {(E > MICHEL_ENDPOINT_MEV).sum()}" if len(E) else ""))
    if args.tsv:
        keys = ["event", "evtno", "nu_index", "main_cluster", "stm", "tgm", "fc", "fired", "f7", "f8", "f7_filled", "f8_filled", "isFC",
                "michel_E", "michel_id", "michel_len", "michel_pdg", "n_showers", "n_at_main", "mvx", "mvy", "mvz", "kine_Enu"]
        with open(args.tsv, "w") as fh:
            fh.write("# PDVD Michel candidates (doc pdvd/25 M5): STM-tagged bundles whose cosmic-tagger flags 6-8 fired; michel_E = kine_best (MeV) of the highest-energy shower at the main vertex\n")
            fh.write("\t".join(keys) + "\n")
            for r in rows_out:
                fh.write("\t".join("" if r.get(k) is None else (f"{r[k]:.2f}" if isinstance(r[k], float) else str(r[k])) for k in keys) + "\n")
        print("wrote", args.tsv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    if len(E):
        ax.hist(E, bins=np.arange(0, 100.1, 4.0), color="#2a78d6", alpha=0.85, label=f"Michel candidates (n={len(E)})")
    ax.axvline(MICHEL_ENDPOINT_MEV, color="#e34948", ls="--", label="Michel endpoint 52.8 MeV")
    ax.set_xlabel("shower energy at the muon stop, kine_best [MeV]"); ax.set_ylabel("candidates"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    ax.set_title(f"PDVD Michel-electron energy ({args.tag}; uncalibrated EM scale)", fontsize=10)
    ax = axes[1]
    L = np.array([r["michel_len"] for r in rows_out if r["michel_len"] is not None], float)
    if len(L):
        ax.scatter([r["michel_len"] for r in rows_out if r["michel_E"] is not None and r["michel_len"] is not None],
                   [r["michel_E"] for r in rows_out if r["michel_E"] is not None and r["michel_len"] is not None], s=12, color="#2a78d6")
    ax.set_xlabel("shower total length [cm]"); ax.set_ylabel("kine_best [MeV]"); ax.grid(alpha=0.2); ax.set_title("energy vs shower length", fontsize=10)
    fig.tight_layout(); fig.savefig(args.out, dpi=140); print("wrote", args.out)


if __name__ == "__main__":
    main()
