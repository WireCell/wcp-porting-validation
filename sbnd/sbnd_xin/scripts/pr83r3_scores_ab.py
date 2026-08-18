#!/usr/bin/env python3
"""pr/83 round 3 -- scores A/B between two PR arms (knob-off vs knob-on).

Joins pr_scores_table.py output for two arms on (run, subrun, event) and
reports every event whose numu_score / nue_score / kine_reco_Enu / vertex
moved beyond thresholds.  Follows pr86_movers.py's adjudication shape but
sourced from pr_scores_table.py (the pr83r3 arms carry no vf-scores.tsv).

Usage:
  pr83r3_scores_ab.py <arm_off> <arm_on> [--tsv out.tsv]
      [--numu-thr 0.05] [--nue-thr 0.05] [--enu-thr 5] [--vtx-thr 1.0]

Exit 0 always (reporting tool, not a gate).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)


def load_scores(arm):
    out = subprocess.run(
        [sys.executable, os.path.join(SX, "pr_scores_table.py"), "--root", arm],
        capture_output=True, text=True, check=True).stdout
    rows = {}
    lines = out.strip().split("\n")
    hdr = lines[0].split("\t")
    idx = {c: i for i, c in enumerate(hdr)}
    for ln in lines[1:]:
        f = ln.split("\t")
        def g(col, cast=float, dflt=None):
            try:
                return cast(f[idx[col]])
            except (KeyError, ValueError, IndexError):
                return dflt
        key = (g("run", int, 0), g("subrun", int, 0), g("event", int, 0))
        rows[key] = dict(
            numu=g("numu_score"), nue=g("nue_score"), enu=g("kine_reco_Enu_MeV"),
            vx=g("nu_x_cm"), vy=g("nu_y_cm"), vz=g("nu_z_cm"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_off")
    ap.add_argument("arm_on")
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--numu-thr", type=float, default=0.05)
    ap.add_argument("--nue-thr", type=float, default=0.05)
    ap.add_argument("--enu-thr", type=float, default=5.0)
    ap.add_argument("--vtx-thr", type=float, default=1.0)
    a = ap.parse_args()

    off = load_scores(a.arm_off)
    on = load_scores(a.arm_on)
    both = sorted(set(off) & set(on))
    only_off = sorted(set(off) - set(on))
    only_on = sorted(set(on) - set(off))
    if only_off:
        print(f"# WARNING {len(only_off)} events only in {a.arm_off}: "
              f"{[k[2] for k in only_off][:10]}", file=sys.stderr)
    if only_on:
        print(f"# WARNING {len(only_on)} events only in {a.arm_on}: "
              f"{[k[2] for k in only_on][:10]}", file=sys.stderr)

    movers = []
    for k in both:
        o, n = off[k], on[k]
        d = {}
        for col, thr in (("numu", a.numu_thr), ("nue", a.nue_thr), ("enu", a.enu_thr)):
            if o[col] is not None and n[col] is not None and abs(n[col] - o[col]) > thr:
                d[col] = (o[col], n[col])
        if all(v is not None for v in (o["vx"], o["vy"], o["vz"], n["vx"], n["vy"], n["vz"])):
            dv = ((n["vx"]-o["vx"])**2 + (n["vy"]-o["vy"])**2 + (n["vz"]-o["vz"])**2) ** 0.5
            if dv > a.vtx_thr:
                d["vtx_cm"] = (0.0, dv)
        if d:
            movers.append((k, d))

    print(f"# {a.arm_off} vs {a.arm_on}: {len(both)} joined events, "
          f"{len(movers)} movers")
    rows = []
    for (run, sub, evt), d in movers:
        parts = []
        for col, (o, n) in sorted(d.items()):
            if col == "vtx_cm":
                parts.append(f"vtx moved {n:.1f} cm")
            else:
                parts.append(f"{col} {o:.3f} -> {n:.3f}")
        print(f"evt {evt}: " + "; ".join(parts))
        rows.append([str(run), str(sub), str(evt)] +
                    [f"{d.get(c, ('', ''))[0]}\t{d.get(c, ('', ''))[1]}"
                     if c in d else "\t" for c in ("numu", "nue", "enu", "vtx_cm")])

    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("run\tsubrun\tevent\tnumu_off\tnumu_on\tnue_off\tnue_on\t"
                    "enu_off\tenu_on\tvtx_zero\tvtx_move_cm\n")
            for r in rows:
                f.write("\t".join(r) + "\n")
        print(f"# wrote {a.tsv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
