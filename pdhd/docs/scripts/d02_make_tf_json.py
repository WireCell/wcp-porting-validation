#!/usr/bin/env python3
"""doc pdhd/02 -- write a PDHD TrackFitting parameter JSON variant carrying the EMPIRICAL
effective transverse smearing (joint fit of d44_sigma_fit.py --det pdhd) in place of the
configured DT / ind_sigma_u_T / ind_sigma_v_T / col_sigma_w_T.

Forked BY DUPLICATION from pdvd/docs/nf_sp_img_clus/scripts/d44_make_tf_json.py (untouched).
Two changes: the provenance text names PDHD (doc pdhd/02, arm stmwc, Wire_ind X=0.75 /
Wire_col X=10.0), and a --fix-dt mode that keeps DT at a given PHYSICAL value and fits
c^2 alone from the per-bin table (_bins.tsv): c^2 = sum_i w_i (S2_i - 2 D t_i) / sum_i w_i,
w_i = 1/err_i^2, one number per plane (the one-parameter line in c^2 at fixed D).

The canonical file is NOT touched: the variant is a copy under pdhd/stm/ selected per arm
with PDHD_PR_TLA="-A trackfitting_config=<abs path>".  Flipping production is the owner's call.

Usage:
  d02_make_tf_json.py --fit docs/figs/d02_sigma_fit.tsv --est share --src <canonical.json> --out stm/pdhd_track_fitting_d02.json
  d02_make_tf_json.py --bins docs/figs/d02_sigma_bins.tsv --est share --fix-dt 8.2017 --src ... --out stm/pdhd_track_fitting_d02fix.json
                      [--label all] [--set gaus_nsigma=6.0] [--note "..."]
"""
import argparse, csv, json, sys, datetime, os
import numpy as np

KEY = {"U": "ind_sigma_u_T", "V": "ind_sigma_v_T", "W": "col_sigma_w_T"}


def from_fit(a):
    rows = [r for r in csv.DictReader(open(a.fit), delimiter="\t")
            if r["label"] == a.label and r["est"] == a.est and r["plane"].endswith("(joint)")]
    if len(rows) != 3:
        print("need the three joint rows, got", len(rows), file=sys.stderr); sys.exit(1)
    DT = float(rows[0]["DT_json"])
    new = {"DT": float("%.4g" % DT)}
    for r in rows:
        new[KEY[r["plane"][0]]] = float("%.4g" % abs(float(r["c_eff_mm"])))
    desc = ("joint fit sigma_eff^2 = 2 DT t + c^2 over the three planes (one shared DT): DT=%.4g (%.2f +- %.2f cm2/s)"
            % (new["DT"], float(rows[0]["DT_eff_cm2s"]), float(rows[0]["DT_err"])))
    return new, desc


def from_bins(a):
    rows = [r for r in csv.DictReader(open(a.bins), delimiter="\t")
            if r["label"] == a.label and r["est"] == a.est]
    D = a.fix_dt * 1e-7          # cm2/s -> mm2/ns
    new = {"DT": float("%.4g" % D)}
    parts = []
    for p in "UVW":
        rr = [r for r in rows if r["plane"] == p]
        if not rr:
            print("no bins for plane", p, file=sys.stderr); sys.exit(1)
        t = np.array([float(r["t_us"]) * 1e3 for r in rr]); s2 = np.array([float(r["sig2_eff_mm2"]) for r in rr])
        e = np.array([float(r["sig2_err"]) for r in rr]); w = 1.0 / e**2
        c2 = float((w * (s2 - 2 * D * t)).sum() / w.sum()); c2e = float(np.sqrt(1.0 / w.sum()))
        chi2 = float((w * (s2 - 2 * D * t - c2)**2).sum())
        c = np.sqrt(max(c2, 0.0))
        new[KEY[p]] = float("%.4g" % c)
        parts.append("%s c^2=%.3f+-%.3f mm2 (c=%.3f, chi2/ndf %.1f/%d)" % (p, c2, c2e, c, chi2, len(rr) - 1))
    desc = "DT FIXED at the physical %.4g cm2/s and c fitted alone per plane from the drift bins: %s" % (a.fix_dt, "; ".join(parts))
    return new, desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit"); ap.add_argument("--bins")
    ap.add_argument("--est", required=True, choices=("rms", "share"))
    ap.add_argument("--fix-dt", type=float, default=None, help="cm2/s; with --bins")
    ap.add_argument("--label", default="all")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], help="extra key=value (double)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if (a.fix_dt is not None and a.bins is None) or (a.fix_dt is None and a.fit is None):
        print("use --fit (joint line) or --bins with --fix-dt", file=sys.stderr); return 1
    new, desc = from_bins(a) if a.fix_dt is not None else from_fit(a)
    src_tsv = a.bins if a.fix_dt is not None else a.fit
    j = json.load(open(a.src), object_pairs_hook=lambda kv: dict(kv))
    old = {k: j[k] for k in ("DT", "ind_sigma_u_T", "ind_sigma_v_T", "col_sigma_w_T")}
    prov = ("doc pdhd/02 (%s): EMPIRICAL effective transverse smearing, %s-matched estimator, %s, measured on the "
            "PDHD STM-fit arm stmwc (run 029107, 30 events, both wrapped-plane knobs on = PDHD PR production since "
            "2026-09-05; prolonged segments |dwire/dslice| < 0.436, own-centroid width, drift-resolved; "
            "pdhd/docs/scripts/d44_sigma_fit.py --det pdhd, TSV %s, label %s). Replaces the configured "
            "DT=%.4g ind_sigma_u_T=%.4g ind_sigma_v_T=%.4g col_sigma_w_T=%.4g with DT=%.4g c_u=%.4g c_v=%.4g c_w=%.4g mm. "
            "These constants describe the ctpc charge the fit is compared to under TODAY's signal processing "
            "(pdhd/sp-filters.jsonnet Wire_ind X=0.75, Wire_col X=10.0); if the SP wire filter changes they are invalid "
            "and must be re-derived from data. Do NOT re-derive them from the SP filter closed form (doc pdvd/42 sec 8.2/8.6)."
            % (datetime.date.today().isoformat(), a.est, desc, os.path.basename(src_tsv), a.label,
               old["DT"], old["ind_sigma_u_T"], old["ind_sigma_v_T"], old["col_sigma_w_T"],
               new["DT"], new["ind_sigma_u_T"], new["ind_sigma_v_T"], new["col_sigma_w_T"]))
    if a.note:
        prov += " " + a.note
    out = {}
    for k, v in j.items():
        if k == "_comment_diffusion":
            out["_comment_d02_effective_smearing"] = prov
            out["_comment_diffusion_superseded"] = v
            continue
        out[k] = new.get(k, v)
    for kv in a.set:
        k, v = kv.split("=", 1); out[k] = float(v)
        out["_comment_" + k] = "doc pdhd/02 arm knob: %s = %s (C++ default 4.0 = legacy literal)" % (k, v)
    json.dump(out, open(a.out, "w"), indent=4)
    print("%s <- %s: %s + %s" % (a.out, a.src, new, a.set))
    return 0


if __name__ == "__main__":
    sys.exit(main())
