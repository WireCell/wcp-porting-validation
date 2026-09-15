#!/usr/bin/env python3
"""doc pdvd/102 sec 7 -- the retile-sampler arms against the baseline, read by the PRE-REGISTERED rule
(/home/xqian/tmp/d102/pred.txt, sha256 795adc97..., frozen 2026-09-15T05:59:07 before any knob-on arm ran).

Fork by duplication of d101_knob_report.py (untouched).  Differences: the strata and clauses of pred.txt,
and a spurious-segment clause read from tracking-pr.root.

Inputs (eval TSVs from d102_sim_fit_eval.py / d101_sim_fit_eval.py, named eval_<det>_<arm>.tsv):
  ISO stratum  : d102 set (runs 900103/900104), k 0-15 = theta {65,75,85,89}; base d102b; arms d102cs1/2/3
  repeats      : k 16, 17 (reported, never in the verdict)
  low stratum  : doc 101 set (runs 900101/900102), k < 16 with theta <= 60; base d101kf; arms d102lcs1/2/3
PASS on a detector when
  ISO: median d(res_t_p90) < 0 and median d(dev_p90) < 0; >= 10/16 tracks improve res_t_p90;
       median d(dip_truth) <= +0.01; no track's PR segment count exceeds the baseline's.
  low: median d(res_t_p90) <= +0.05 mm and median d(dip_truth) <= +0.01.
(The data-resource clause is graded separately, d102_resource_report in the doc's repro block.)

Usage: d102_knob_report.py [--dir /home/xqian/tmp/d102/eval] [--arms cs1,cs2,cs3] [--out FILE.tsv]
"""
import argparse, csv, os
import numpy as np
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
RUN_ISO = {"pdvd": 900103, "pdhd": 900104}
RUN_LOW = {"pdvd": 900101, "pdhd": 900102}
KEYS = ["res_t_med", "res_t_p90", "dev_med", "dev_p90", "x_bias", "snap_u", "snap_v", "snap_w",
        "dqdx_med", "dqdx_cv", "dip_truth", "acf1", "r_res_q", "len_ratio"]


def load(fn):
    rows = {}
    for r in csv.DictReader(open(fn), delimiter="\t"):
        rows[int(r["k"])] = {k: (float(v) if v not in ("", "nan") else np.nan) for k, v in r.items()}
    return rows


def nseg(det, run, k, arm):
    fn = f"{IMG}/{det}/work/{run}_{k}_{arm}/tracking-pr.root"
    if not os.path.exists(fn):
        return -1
    t = uproot.open(fn)["T_rec_charge"].arrays(["sub_cluster_id", "flag_vertex"], library="np")
    return len(set(t["sub_cluster_id"][t["flag_vertex"] == 0].tolist()))


def stratum_rows(det, B, A, ks):
    d = {key: np.array([A[k][key] - B[k][key] for k in ks]) for key in KEYS}
    row = dict(n=len(ks), n_improve_res_t_p90=int(np.sum(d["res_t_p90"] < 0)))
    for key in KEYS:
        row["d_" + key] = float(np.nanmedian(d[key]))
    row["base_res_t_p90"] = float(np.nanmedian([B[k]["res_t_p90"] for k in ks]))
    row["base_dev_p90"] = float(np.nanmedian([B[k]["dev_p90"] for k in ks]))
    row["per_track_d_res_t_p90"] = " ".join(f"k{k}:{A[k]['res_t_p90'] - B[k]['res_t_p90']:+.2f}" for k in ks)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/xqian/tmp/d102/eval")
    ap.add_argument("--arms", default="cs1,cs2,cs3")
    ap.add_argument("--dets", default="pdvd,pdhd")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out, verdict = [], {}
    for det in a.dets.split(","):
        Bi = load(os.path.join(a.dir, f"eval_{det}_d102b.tsv"))
        Bl = load(os.path.join(a.dir, f"eval_{det}_d101kf.tsv")) if os.path.exists(os.path.join(a.dir, f"eval_{det}_d101kf.tsv")) else None
        for s in a.arms.split(","):
            iso_arm, low_arm = f"d102{s}", f"d102l{s}"
            fi = os.path.join(a.dir, f"eval_{det}_{iso_arm}.tsv")
            if not os.path.exists(fi):
                print(f"{det} {iso_arm}: no eval"); continue
            Ai = load(fi)
            ks = sorted(k for k in Bi if k in Ai and k < 16)
            r = stratum_rows(det, Bi, Ai, ks)
            seg_b = {k: nseg(det, RUN_ISO[det], k, "d102b") for k in ks}
            seg_a = {k: nseg(det, RUN_ISO[det], k, iso_arm) for k in ks}
            gained = [k for k in ks if seg_a[k] > seg_b[k]]
            c1 = r["d_res_t_p90"] < 0 and r["d_dev_p90"] < 0
            c2 = r["n_improve_res_t_p90"] >= 10
            c3 = r["d_dip_truth"] <= 0.01
            c4 = not gained
            r.update(det=det, arm=iso_arm, stratum="ISO", c1=c1, c2=c2, c3=c3, c4_no_new_segment=c4,
                     segments_gained=" ".join(f"k{k}:{seg_b[k]}->{seg_a[k]}" for k in gained))
            out.append(r)
            print(f"{det} {iso_arm:8s} ISO n={r['n']} d res_t p90 {r['d_res_t_p90']:+.3f} (base {r['base_res_t_p90']:.2f}, "
                  f"{r['n_improve_res_t_p90']}/{r['n']} improve) d dev p90 {r['d_dev_p90']:+.3f} (base {r['base_dev_p90']:.2f}) "
                  f"d dip {r['d_dip_truth']:+.4f} d cv {r['d_dqdx_cv']:+.4f} segments gained {gained or 'none'} "
                  f"c1={c1} c2={c2} c3={c3} c4={c4}")
            print(f"      per track: {r['per_track_d_res_t_p90']}")
            rep = sorted(k for k in Bi if k in Ai and k >= 16)
            if rep:
                rr = stratum_rows(det, Bi, Ai, rep); rr.update(det=det, arm=iso_arm, stratum="repeats")
                out.append(rr)
                print(f"      repeats: {rr['per_track_d_res_t_p90']}")
            low_ok = None
            fl = os.path.join(a.dir, f"eval_{det}_{low_arm}.tsv")
            if Bl is not None and os.path.exists(fl):
                Al = load(fl)
                kl = sorted(k for k in Bl if k in Al and k < 16 and Bl[k]["theta"] <= 60)
                rl = stratum_rows(det, Bl, Al, kl)
                low_ok = rl["d_res_t_p90"] <= 0.05 and rl["d_dip_truth"] <= 0.01
                rl.update(det=det, arm=low_arm, stratum="theta<=60", low_pass=low_ok)
                out.append(rl)
                print(f"      low stratum {low_arm}: n={rl['n']} d res_t p90 {rl['d_res_t_p90']:+.3f} (base {rl['base_res_t_p90']:.2f}) "
                      f"d dip {rl['d_dip_truth']:+.4f} -> {low_ok}")
            else:
                print(f"      low stratum {low_arm}: not available")
            verdict[(det, s)] = bool(c1 and c2 and c3 and c4 and low_ok)
    print("\nVERDICT (pred.txt; a setting qualifies only if it PASSES on both detectors; resources graded separately):")
    for s in a.arms.split(","):
        v = [verdict.get((det, s)) for det in a.dets.split(",")]
        print(f"  {s}: " + " ".join(f"{det}={x}" for det, x in zip(a.dets.split(","), v))
              + ("  -> QUALIFIES (sim)" if all(x is True for x in v) else "  -> does not qualify"))
    if a.out and out:
        keys = list(dict.fromkeys(k for r in out for k in r))
        with open(a.out, "w") as fo:
            fo.write("\t".join(keys) + "\n")
            for r in out:
                fo.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")


if __name__ == "__main__":
    main()
