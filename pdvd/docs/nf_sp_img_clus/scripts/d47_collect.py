#!/usr/bin/env python3
"""doc pdvd/47 -- collect the d47_sim_transverse_profile.py outputs of every (detector, arm,
tag) into one summary table: the joint-fit c per plane (share-matched, rms-matched,
truth-centred), D_T,eff, the calibration (charge per tick vs truth, tick offset), and the
super-resolution kernel rms where --kernel was run.

Usage: d47_collect.py --root /home/xqian/tmp/xtrack --out figs/47_sim_summary.tsv [--data figs/47_data_constants.tsv]
"""
import argparse, csv, glob, os, sys
import numpy as np

PITCH = {"pdhd": (4.6691, 4.6691, 4.7915), "pdvd": (7.65, 7.65, 5.10), "sbnd": (3.0, 3.0, 3.0)}


def rd(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/xqian/tmp/xtrack")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for det in ("pdhd", "pdvd", "sbnd"):
        for fit in sorted(glob.glob(os.path.join(a.root, det, "ana", "*_fit.tsv"))):
            label = os.path.basename(fit)[:-8]
            if label.startswith("tmp"):
                continue
            arm, _, tag = label.partition("_")
            F = rd(fit); C = rd(fit.replace("_fit.tsv", "_calib.tsv"))
            K = rd(fit.replace("_fit.tsv", "_ksum.tsv")) if os.path.exists(fit.replace("_fit.tsv", "_ksum.tsv")) else []
            r = dict(det=det, arm=arm, tag=tag, n_tracks=len(C),
                     q_per_tick=np.median([float(c["q_per_tick"]) for c in C]) if C else np.nan,
                     q_ratio=np.median([float(c["q_per_tick"]) / float(c["q_per_tick_truth"]) for c in C]) if C else np.nan,
                     dtick=np.median([float(c["dtick"]) for c in C]) if C else np.nan,
                     vratio=np.median([float(c["ratio"]) for c in C]) if C else np.nan)
            for est in ("share", "rms", "truth"):
                jr = [f for f in F if f["label"] == "all" and f["est"] == est and f["plane"].endswith("(joint)")]
                if not jr:
                    continue
                r["DT_%s" % est] = float(jr[0]["DT_eff_cm2s"]); r["DTerr_%s" % est] = float(jr[0]["DT_err"])
                r["DT_model"] = float(jr[0]["DT_model_cm2s"]); r["chi2_%s" % est] = float(jr[0]["chi2"]); r["ndf_%s" % est] = int(jr[0]["ndf"])
                for f in jr:
                    p = f["plane"][0]
                    r["c%s_%s" % (p, est)] = abs(float(f["c_eff_mm"])); r["c%serr_%s" % (p, est)] = float(f["c_err"])
                    r["c%s_%s_pitch" % (p, est)] = abs(float(f["c_eff_mm"])) / PITCH[det]["UVW".index(p)]
            for k in K:
                p = k["plane"][0]
                r["ksr_%s_pitch" % p] = float(k["rms_sr_pitch"]); r["ksr_excess_%s_mm" % p] = float(k["excess_mm"])
            rows.append(r)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(a.out, "w") as fo:
        fo.write("\t".join(keys) + "\n")
        for r in rows:
            fo.write("\t".join(("%.6g" % r[k] if isinstance(r.get(k), float) else str(r.get(k, ""))) for k in keys) + "\n")
    print("%-5s %-6s %-9s %6s %6s | %7s %7s %7s | %7s %7s %7s | %6s %6s" % ("det", "arm", "tag", "ntrk", "q/q0", "cU_sh", "cV_sh", "cW_sh", "cU_rms", "cV_rms", "cW_rms", "DT_sh", "DTmod"))
    for r in rows:
        print("%-5s %-6s %-9s %6d %6.3f | %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f | %6.2f %6.2f" % (
            r["det"], r["arm"], r["tag"], r["n_tracks"], r["q_ratio"], r.get("cU_share", np.nan), r.get("cV_share", np.nan), r.get("cW_share", np.nan),
            r.get("cU_rms", np.nan), r.get("cV_rms", np.nan), r.get("cW_rms", np.nan), r.get("DT_share", np.nan), r.get("DT_model", np.nan)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
