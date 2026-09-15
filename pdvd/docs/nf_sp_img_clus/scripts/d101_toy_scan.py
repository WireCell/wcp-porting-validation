#!/usr/bin/env python3
"""doc pdvd/101 -- run the d101_toy chain over a track-angle grid, three detectors and a list of
lever variants; write one row per (detector, variant, track) and a per-(detector, variant) summary.

Grid: theta (angle to the drift axis) x phi (azimuth from the collection-wire direction) x seeds.
Each track has a random sub-pitch placement, so the lattice phase is averaged.

Usage:
  d101_toy_scan.py --out DIR [--dets pdvd,pdhd,sbnd] [--variants base,half,...] [--nseed 3]
                   [--nproc 12] [--noise 300 --thr 900] [--fluct 0]
Variants are named in VARIANTS below; 'base' is production.
"""
import argparse, json, math, os, sys, time
from dataclasses import replace, asdict
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d101_toy as T  # noqa: E402

VARIANTS = {
    "base": {},
    # A. sampling
    "step1": dict(min_step=1.0),
    "half": dict(half_pitch=True),
    "half_step1": dict(half_pitch=True, min_step=1.0),
    "xmid": dict(x_mid=True),
    # B. terminals
    "centroid": dict(term_centroid=True),
    "sep_pitch1": dict(min_sep_pitch=1.0),
    "sep0": dict(min_sep=0.0),
    # C. fit
    "assoc_p05": dict(assoc_floor_pitch=0.5),
    "assoc_p07": dict(assoc_floor_pitch=0.7),
    "assoc_p10": dict(assoc_floor_pitch=1.0),
    "cont_center": dict(assoc_cont_center=True),
    "wpow1": dict(fit_weight_pow=1.0),
    "divp1": dict(div_sigma_pitch=1.0),
    # oracles
    "oracle_truthseed": dict(seed="truth"),
    # round 2 (combinations aimed at a lever that wins on all three detectors)
    "wpow15": dict(fit_weight_pow=1.5),
    "wpow1_cc": dict(fit_weight_pow=1.0, assoc_cont_center=True),
    "wpow15_cc": dict(fit_weight_pow=1.5, assoc_cont_center=True),
    "hs1_cc": dict(half_pitch=True, min_step=1.0, assoc_cont_center=True),
    "wpow1_hs1": dict(fit_weight_pow=1.0, half_pitch=True, min_step=1.0),
    "wpow1_cc_hs1": dict(fit_weight_pow=1.0, assoc_cont_center=True, half_pitch=True, min_step=1.0),
    "wpow15_cc_hs1": dict(fit_weight_pow=1.5, assoc_cont_center=True, half_pitch=True, min_step=1.0),
    "oracle_truthseed_wpow1_cc": dict(seed="truth", fit_weight_pow=1.0, assoc_cont_center=True),
    # round 3 (sec 3.6): the sampling levers scoped to the Steiner stage only, as a PR-job sampler
    # knob would act in the C++ chain
    "hs1_steiner": dict(half_pitch=True, min_step=1.0, sampler_scope="steiner"),
    "wpow15_cc_hs1_steiner": dict(fit_weight_pow=1.5, assoc_cont_center=True, half_pitch=True, min_step=1.0,
                                  sampler_scope="steiner"),
}

GRID_THETA = (80.0, 60.0, 40.0, 20.0)
GRID_PHI = (0.0, 30.0, 60.0, 90.0)


def job(args):
    det_name, vname, theta, phi, seed, noise, thr, fluct = args
    det = T.load_detector(det_name)
    o = replace(T.Opts(), **VARIANTS[vname])
    rng = np.random.default_rng(1000 * seed + int(theta) * 7 + int(phi))
    p0, p1 = T.make_track(det, 100.0, theta, phi, rng=rng)
    t0 = time.time()
    try:
        ev = T.run_event(det, p0, p1, o, seed=seed, noise=noise, thr=thr, fluct=fluct)
        m = T.metrics(det, ev) if ev is not None else None
    except Exception as e:  # noqa: BLE001 -- recorded, not fatal
        return dict(det=det_name, variant=vname, theta=theta, phi=phi, seed=seed, error=repr(e))
    row = dict(det=det_name, variant=vname, theta=theta, phi=phi, seed=seed, wall=time.time() - t0)
    if m is None:
        row["error"] = "no fit"
    else:
        row.update(m)
        row.update(npts=ev["npts"], nterm=ev["nterm"], nblob=ev["nblob"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dets", default="pdvd,pdhd,sbnd")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--nseed", type=int, default=3)
    ap.add_argument("--nproc", type=int, default=12)
    ap.add_argument("--noise", type=float, default=300.0)
    ap.add_argument("--thr", type=float, default=900.0)
    ap.add_argument("--fluct", type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(d, v, th, ph, s, a.noise, a.thr, a.fluct) for d in a.dets.split(",") for v in a.variants.split(",")
            for th in GRID_THETA for ph in GRID_PHI for s in range(a.nseed)]
    with Pool(a.nproc) as pool:
        rows = pool.map(job, jobs, chunksize=2)
    with open(os.path.join(a.out, "rows.jsonl"), "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    keys = ["res_med", "res_p90", "res_t_med", "res_t_p90", "dev_med", "dev_p90", "x_bias", "res_u", "res_v", "res_w",
            "snap_u", "snap_v", "snap_w", "seed_med", "seed_p90",
            "dqdx_med", "dqdx_cv", "dqdx_rms", "dip", "dip_truth", "acf1", "r_res_q", "len_ratio", "nterm"]
    lines = ["det\tvariant\tok/total\t" + "\t".join(keys)]
    for d in a.dets.split(","):
        for v in a.variants.split(","):
            rr = [r for r in rows if r["det"] == d and r["variant"] == v]
            ok = [r for r in rr if "error" not in r]
            vals = []
            for k in keys:
                x = np.array([r.get(k, np.nan) for r in ok], float)
                vals.append(f"{np.nanmedian(x):.4g}" if np.isfinite(x).any() else "nan")
            lines.append(f"{d}\t{v}\t{len(ok)}/{len(rr)}\t" + "\t".join(vals))
    txt = "\n".join(lines)
    open(os.path.join(a.out, "summary.tsv"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
