#!/usr/bin/env python3
"""doc pdvd/102 round 2 sec B -- are the doc 101 fit knobs still effective once the retile uses charge_stepped?

Reads the eval TSVs (d101_sim_fit_eval.py / d102_sim_fit_eval.py, eval_<det>_<arm>.tsv) of the 2x2 cells
declared in /home/xqian/tmp/d102/pred2.txt:
  ISO set (900103/900104, k 0-15):  off+stepped d102o,    on+stepped d102b,  off+cs d102ocs,  on+cs d102cs1
  low set (900101/900102, theta <= 60, k < 16): off+stepped d101koff, on+stepped d101kf, off+cs d102locs, on+cs d102lcs1
Per detector and stratum, per metric, the median over tracks of the paired knob effect (on - off) under each
sampler, and the paired sampler effect (cs - stepped) under each fit setting.

Reading declared in pred2.txt: knobs "still effective under charge_stepped" if the knob effect under cs has the
same sign as under stepped for BOTH dev_p90 and res_t_p90 and at least half its magnitude; "partly" if same sign
but less than half; "not" otherwise.

Usage: d102_interaction_report.py [--dir /home/xqian/tmp/d102/eval] [--out FILE.tsv]
"""
import argparse, csv, os
import numpy as np

METRICS = ["res_t_p90", "dev_p90", "dqdx_cv", "dip_truth", "snap_u", "snap_v", "snap_w", "res_t_med", "dev_med"]
CELLS = {"ISO": dict(off_st="d102o", on_st="d102b", off_cs="d102ocs", on_cs="d102cs1"),
         "low": dict(off_st="d101koff", on_st="d101kf", off_cs="d102locs", on_cs="d102lcs1")}


def load(fn):
    rows = {}
    for r in csv.DictReader(open(fn), delimiter="\t"):
        rows[int(r["k"])] = {k: (float(v) if v not in ("", "nan") else np.nan) for k, v in r.items()}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/xqian/tmp/d102/eval")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = []
    for det in ("pdvd", "pdhd"):
        for stratum, cells in CELLS.items():
            fns = {c: os.path.join(a.dir, f"eval_{det}_{t}.tsv") for c, t in cells.items()}
            missing = [f for f in fns.values() if not os.path.exists(f)]
            if missing:
                print(f"{det} {stratum}: missing {missing}"); continue
            T = {c: load(f) for c, f in fns.items()}
            ks = sorted(set.intersection(*[set(t) for t in T.values()]))
            ks = [k for k in ks if k < 16 and (stratum == "ISO" or T["off_st"][k]["theta"] <= 60)]
            print(f"\n{det} {stratum}: {len(ks)} tracks")
            print(f"  {'metric':10s} {'base off+st':>11s} | {'knob eff @st':>12s} {'knob eff @cs':>12s} | {'cs eff @off':>11s} {'cs eff @on':>10s}")
            eff = {}
            for m in METRICS:
                d = lambda x, y: np.array([T[x][k][m] - T[y][k][m] for k in ks])
                ke_st, ke_cs = np.nanmedian(d("on_st", "off_st")), np.nanmedian(d("on_cs", "off_cs"))
                se_off, se_on = np.nanmedian(d("off_cs", "off_st")), np.nanmedian(d("on_cs", "on_st"))
                base = np.nanmedian([T["off_st"][k][m] for k in ks])
                eff[m] = (ke_st, ke_cs)
                print(f"  {m:10s} {base:11.4f} | {ke_st:+12.4f} {ke_cs:+12.4f} | {se_off:+11.4f} {se_on:+10.4f}")
                out.append(dict(det=det, stratum=stratum, metric=m, n=len(ks), base_off_stepped=base,
                                knob_effect_stepped=ke_st, knob_effect_cs=ke_cs, cs_effect_knobs_off=se_off, cs_effect_knobs_on=se_on))
            verdicts = []
            for m in ("dev_p90", "res_t_p90"):
                st, cs = eff[m]
                if st == 0 or np.sign(st) != np.sign(cs):
                    verdicts.append("not")
                elif abs(cs) >= 0.5 * abs(st):
                    verdicts.append("still")
                else:
                    verdicts.append("partly")
            v = "not" if "not" in verdicts else ("partly" if "partly" in verdicts else "still")
            print(f"  knobs under charge_stepped (pred2 reading on dev_p90, res_t_p90 = {verdicts}): {v.upper()} EFFECTIVE; "
                  f"ratio cs/stepped dev_p90 {eff['dev_p90'][1] / eff['dev_p90'][0] if eff['dev_p90'][0] else float('nan'):.2f}, "
                  f"res_t_p90 {eff['res_t_p90'][1] / eff['res_t_p90'][0] if eff['res_t_p90'][0] else float('nan'):.2f}")
    if a.out and out:
        keys = list(out[0])
        with open(a.out, "w") as fo:
            fo.write("\t".join(keys) + "\n")
            for r in out:
                fo.write("\t".join(str(r[k]) for k in keys) + "\n")


if __name__ == "__main__":
    main()
