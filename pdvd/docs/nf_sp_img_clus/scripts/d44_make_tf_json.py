#!/usr/bin/env python3
"""doc pdvd/44 -- write a TrackFitting parameter JSON variant carrying the EMPIRICAL
effective transverse smearing (joint fit of d44_sigma_fit.py) in place of the
configured DT / ind_sigma_u_T / ind_sigma_v_T / col_sigma_w_T.

The canonical file is NOT touched: the variant is a copy under pdvd/stm/ (or the
SBND campaign dir) selected per arm with -A trackfitting_config=<abs path>
(PDVD) / SBND_TRACKFIT_JSON (SBND).  Flipping production is the owner's call.

Usage:
  d44_make_tf_json.py --fit figs/44_sigma_pdvd_fit.tsv --est share --src <canonical.json> --out stm/pdvd_track_fitting_d44.json
                      [--label all] [--set gaus_nsigma=6.0] [--note "..."]
"""
import argparse, csv, json, sys, datetime, os

KEY = {"U": "ind_sigma_u_T", "V": "ind_sigma_v_T", "W": "col_sigma_w_T"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--est", required=True, choices=("rms", "share"))
    ap.add_argument("--label", default="all")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], help="extra key=value (double)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.fit), delimiter="\t")
            if r["label"] == a.label and r["est"] == a.est and r["plane"].endswith("(joint)")]
    if len(rows) != 3:
        print("need the three joint rows, got", len(rows), file=sys.stderr); return 1
    j = json.load(open(a.src), object_pairs_hook=lambda kv: dict(kv))
    old = {k: j[k] for k in ("DT", "ind_sigma_u_T", "ind_sigma_v_T", "col_sigma_w_T")}
    DT = float(rows[0]["DT_json"])
    new = {"DT": float("%.4g" % DT)}
    for r in rows:
        p = r["plane"][0]
        new[KEY[p]] = float("%.4g" % abs(float(r["c_eff_mm"])))
    prov = ("doc pdvd/44 (%s): EMPIRICAL effective transverse smearing, %s-matched estimator, joint fit "
            "sigma_eff^2 = 2 DT t + c^2 over the three planes of the doc-42 STM-fit arm (prolonged segments, "
            "own-centroid width, drift-resolved; d44_sigma_fit.py, fit TSV %s, label %s). Replaces the configured "
            "DT=%.4g ind_sigma_u_T=%.4g ind_sigma_v_T=%.4g col_sigma_w_T=%.4g with DT=%.4g (%.2f +- %.2f cm2/s) "
            "c_u=%.4g c_v=%.4g c_w=%.4g mm. These constants describe the ctpc charge the fit is compared to under "
            "TODAY's signal processing (sp-filters.jsonnet Wire_ind X=5.0); if the SP wire filter changes they are "
            "invalid and must be re-derived. Do NOT re-derive them from the SP filter closed form (doc 42 sec 8.2/8.6)."
            % (datetime.date.today().isoformat(), a.est, os.path.basename(a.fit), a.label,
               old["DT"], old["ind_sigma_u_T"], old["ind_sigma_v_T"], old["col_sigma_w_T"],
               new["DT"], float(rows[0]["DT_eff_cm2s"]), float(rows[0]["DT_err"]),
               new["ind_sigma_u_T"], new["ind_sigma_v_T"], new["col_sigma_w_T"]))
    if a.note:
        prov += " " + a.note
    out = {}
    for k, v in j.items():
        if k == "_comment_diffusion":
            out["_comment_d44_effective_smearing"] = prov
            out["_comment_diffusion_superseded"] = v
            continue
        out[k] = new.get(k, v)
    for kv in a.set:
        k, v = kv.split("=", 1); out[k] = float(v)
        out["_comment_" + k] = "doc pdvd/44 arm knob: %s = %s (C++ default 4.0 = legacy literal)" % (k, v)
    json.dump(out, open(a.out, "w"), indent=4)
    print("%s <- %s: %s + %s" % (a.out, a.src, new, a.set))
    return 0


if __name__ == "__main__":
    sys.exit(main())
