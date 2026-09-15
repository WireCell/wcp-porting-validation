#!/usr/bin/env python3
"""doc pdvd/101 sec 4.4 -- the knob arms against the baseline, read by the PRE-REGISTERED rule of sec 4.0
(/home/xqian/tmp/d101/pred.txt, frozen 2026-09-14T21:13:44 before any knob arm ran).

Input: eval_<arm>_<det>.tsv from d101_sim_fit_eval.py for the baseline and each knob arm.
Pairs events by k (same pctree, so a delta is exact: d101brep == d101b bit for bit).

Verdict stratum: grid tracks (k < 16) with theta <= 60 deg (12 tracks).  PASS on a detector when
  (1) median d(res_t_p90) < 0 and median d(dev_p90) < 0,
  (2) >= 8 of 12 tracks improve res_t_p90,
  (3) median d(dqdx_cv) <= +0.005, median d|dqdx_med - 1| <= +0.01, median d(dip_truth) <= +0.01.
theta = 80 (k 0-3) and the simulation repeats (k 16, 17) are reported, never in the verdict.

Usage: d101_knob_report.py --dir /home/xqian/tmp/d101 [--base d101b] [--arms d101kf,d101ks,d101kc] [--out FILE.tsv]
"""
import argparse, csv, os
import numpy as np

KEYS = ["res_t_med", "res_t_p90", "dev_med", "dev_p90", "x_bias", "snap_u", "snap_v", "snap_w",
        "dqdx_med", "dqdx_cv", "dip_truth", "acf1", "r_res_q", "len_ratio"]


def load(fn):
    rows = {}
    for r in csv.DictReader(open(fn), delimiter="\t"):
        rows[int(r["k"])] = {k: (float(v) if v not in ("", "nan") else np.nan) for k, v in r.items()}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/xqian/tmp/d101")
    ap.add_argument("--base", default="d101b")
    ap.add_argument("--arms", default="d101kf,d101ks,d101kc")
    ap.add_argument("--dets", default="pdvd,pdhd")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = []
    verdict = {}
    for det in a.dets.split(","):
        B = load(os.path.join(a.dir, f"eval_{a.base}_{det}.tsv"))
        for arm in a.arms.split(","):
            fn = os.path.join(a.dir, f"eval_{arm}_{det}.tsv")
            if not os.path.exists(fn):
                print(f"{det} {arm}: no {fn}"); continue
            A = load(fn)
            for stratum, sel in (("theta<=60", lambda r, k: k < 16 and r["theta"] <= 60),
                                 ("theta=80", lambda r, k: k < 16 and r["theta"] == 80),
                                 ("repeats", lambda r, k: k >= 16)):
                ks = sorted(k for k in B if k in A and sel(B[k], k))
                if not ks:
                    continue
                d = {key: np.array([A[k][key] - B[k][key] for k in ks]) for key in KEYS}
                dabs = np.array([abs(A[k]["dqdx_med"] - 1) - abs(B[k]["dqdx_med"] - 1) for k in ks])
                nimp = int(np.sum(d["res_t_p90"] < 0))
                row = dict(det=det, arm=arm, stratum=stratum, n=len(ks), n_improve_res_t_p90=nimp,
                           d_abs_dqdx_med_minus1=float(np.nanmedian(dabs)))
                for key in KEYS:
                    row["d_" + key] = float(np.nanmedian(d[key]))
                row["base_res_t_p90"] = float(np.nanmedian([B[k]["res_t_p90"] for k in ks]))
                row["base_dev_p90"] = float(np.nanmedian([B[k]["dev_p90"] for k in ks]))
                if stratum == "theta<=60":
                    c1 = row["d_res_t_p90"] < 0 and row["d_dev_p90"] < 0
                    c2 = nimp >= 8
                    c3 = (row["d_dqdx_cv"] <= 0.005 and row["d_abs_dqdx_med_minus1"] <= 0.01
                          and row["d_dip_truth"] <= 0.01)
                    row["c1"], row["c2"], row["c3"] = c1, c2, c3
                    row["PASS"] = bool(c1 and c2 and c3)
                    verdict[(det, arm)] = row["PASS"]
                out.append(row)
                print(f"{det:4s} {arm:7s} {stratum:9s} n={len(ks):2d}  d res_t p90 {row['d_res_t_p90']:+.3f} "
                      f"(base {row['base_res_t_p90']:.2f}, {nimp}/{len(ks)} improve)  d dev p90 {row['d_dev_p90']:+.3f} "
                      f"(base {row['base_dev_p90']:.2f})  d res_t med {row['d_res_t_med']:+.3f}  "
                      f"d cv {row['d_dqdx_cv']:+.4f}  d|med-1| {row['d_abs_dqdx_med_minus1']:+.4f}  "
                      f"d dip {row['d_dip_truth']:+.4f}  d r(res,q) {row['d_r_res_q']:+.3f}  "
                      f"d snap {row['d_snap_u']:+.2f}/{row['d_snap_v']:+.2f}/{row['d_snap_w']:+.2f}"
                      + (f"  c1={row['c1']} c2={row['c2']} c3={row['c3']} PASS={row['PASS']}" if 'PASS' in row else ""))
    print("\nVERDICT (sec 4.0 rule; an arm qualifies only if it PASSES on both detectors):")
    for arm in a.arms.split(","):
        v = [verdict.get((det, arm)) for det in a.dets.split(",")]
        print(f"  {arm}: " + " ".join(f"{det}={x}" for det, x in zip(a.dets.split(","), v))
              + ("  -> QUALIFIES" if all(x is True for x in v) else "  -> does not qualify"))
    if a.out and out:
        keys = list(out[0].keys()) + [k for r in out for k in r if k not in out[0]]
        keys = list(dict.fromkeys(keys))
        with open(a.out, "w") as fo:
            fo.write("\t".join(keys) + "\n")
            for r in out:
                fo.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")


if __name__ == "__main__":
    main()
