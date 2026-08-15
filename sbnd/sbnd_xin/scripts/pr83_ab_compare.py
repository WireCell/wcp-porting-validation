#!/usr/bin/env python3
"""pr/83 off-vs-on A/B: per-event selection/kinematics deltas.

For each event present in BOTH arms, reads calib-pr-evt<ID>.json and compares:
  numu_score, nue_score, cosmict_flag  (tagger block)
  kine_reco_Enu                        (kine block)
  main_vertex position                 (3-D distance)
Reports every event where anything moves, plus summary counts.

Usage: pr83_ab_compare.py <arm_off> <arm_on> [--tsv out.tsv] [--vtx-cut 1.0]
"""
import argparse
import glob
import json
import math
import os


def load(path):
    d = json.load(open(path))
    t = d.get("tagger", {}) or {}
    k = d.get("kine", {}) or {}
    mv = d.get("main_vertex", {}) or {}
    return dict(
        numu=t.get("numu_score"), nue=t.get("nue_score"),
        cosmict=t.get("cosmict_flag"),
        enu=k.get("kine_reco_Enu"),
        vtx=(mv.get("x"), mv.get("y"), mv.get("z")),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_off")
    ap.add_argument("arm_on")
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--vtx-cut", type=float, default=1.0, help="cm")
    args = ap.parse_args()

    evts = sorted(
        os.path.basename(p).replace("pr_evt", "")
        for p in glob.glob(os.path.join(args.arm_off, "pr_evt*")))
    rows, n_cmp = [], 0
    n_numu = n_nue = n_cosmict = n_enu = n_vtx = n_missing = 0
    for e in evts:
        pa = os.path.join(args.arm_off, f"pr_evt{e}", f"calib-pr-evt{e}.json")
        pb = os.path.join(args.arm_on, f"pr_evt{e}", f"calib-pr-evt{e}.json")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            n_missing += 1
            continue
        a, b = load(pa), load(pb)
        n_cmp += 1
        dn = (a["numu"] is not None and b["numu"] is not None
              and abs(a["numu"] - b["numu"]) > 1e-6)
        de = (a["nue"] is not None and b["nue"] is not None
              and abs(a["nue"] - b["nue"]) > 1e-6)
        dc = a["cosmict"] != b["cosmict"]
        dE = (a["enu"] is not None and b["enu"] is not None
              and abs(a["enu"] - b["enu"]) > 0.5)  # MeV
        dv = 0.0
        if None not in a["vtx"] and None not in b["vtx"]:
            dv = math.dist(a["vtx"], b["vtx"])
        if dn: n_numu += 1
        if de: n_nue += 1
        if dc: n_cosmict += 1
        if dE: n_enu += 1
        if dv > args.vtx_cut: n_vtx += 1
        if dn or de or dc or dE or dv > args.vtx_cut:
            rows.append([e,
                         f"{a['numu']:.3f}->{b['numu']:.3f}" if dn else "=",
                         f"{a['nue']:.3f}->{b['nue']:.3f}" if de else "=",
                         f"{a['cosmict']}->{b['cosmict']}" if dc else "=",
                         f"{a['enu']:.0f}->{b['enu']:.0f}" if dE else "=",
                         f"{dv:.1f}"])
            print(f"evt {e}: numu {rows[-1][1]}  nue {rows[-1][2]}  "
                  f"cosmict {rows[-1][3]}  Enu {rows[-1][4]}  dvtx {dv:.1f} cm")
    print(f"# compared {n_cmp} events ({n_missing} missing); changed: "
          f"numu {n_numu}, nue {n_nue}, cosmict {n_cosmict}, "
          f"Enu {n_enu}, vtx>{args.vtx_cut}cm {n_vtx}")
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("event\tnumu\tnue\tcosmict\tenu\tdvtx_cm\n")
            for r in rows:
                f.write("\t".join(r) + "\n")
        print(f"# wrote {args.tsv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
