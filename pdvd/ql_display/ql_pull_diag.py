#!/usr/bin/env python
"""Per-channel pull analysis + phantom/missed taxonomy for QLMatch tuning.

Pulls: over scan-CONFIRMED long-track auto matches (objective tiers gold/high/
med), rebuild the bundle-chi2 per-channel sigma exactly as
TimingTPCBundle::per_opdet_perr + examine_bundle do (pe_err_on_pred lowpe
branch, close-to-PMT relax widening, DAPHNE-rail sat widening) and histogram
pull = (pe - pred)/sigma per channel family (cathode XA / PMT), per
predicted-PE bin, and for the saturated subset separately.
Pull RMS ~ 1 => calibrated; << 1 => error over-inflated; >> 1 => under.

Only bundles whose dumped chi2/ndf reproduces within 2% enter the pull sample
(merged/xtpc-joint bundles whose dump fields don't correspond to a single
chi2 evaluation are excluded and counted).

NOTE: quality_params in the calib dump does not record pe_err_lowpe_frac/knee;
the active values are supplied via --lowpe-frac/--lowpe-knee (PDVD runner
defaults 2.0/10.0) and VERIFIED by the chi2-reproduction guard.

Taxonomy: every scan disagreement is assigned an exclusive admission path
(priority: xtpc_cathode_rescued > xtpc_pin > xtpc_scenario1 > ladder branch
B1-B4 > other/strength-only) so the Phase-3 targets are sized. Missed
positives are diagnosed against the full candidate-bundle list: candidate
present-but-unselected (LASSO/cull territory) vs no-candidate.

Repro:
  python ql_display/ql_pull_diag.py --tag cathxa
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ql_agree_score import (  # noqa: E402
    GID_STRIDE, RUN, NEVT, OBJECTIVE_TIERS, evt_of_idx, cluster_len_cm,
    load_truth, find_truth)

PRED_BINS = [0, 2, 5, 10, 20, 50, 100, 200, 500, float("inf")]


def family_of(opdet):
    if opdet["type"] != 0:
        return "pmt"
    return "cath_xa" if abs(opdet["x"]) < 1.0 else "wall_xa"


def perr_of(qp, lowpe_frac, lowpe_knee, pred, meas_err):
    if not qp["pe_err_on_pred"]:
        return meas_err
    if lowpe_frac >= 0:
        rel = qp["pe_err_frac"] + (lowpe_frac - qp["pe_err_frac"]) * \
            math.exp(-pred / lowpe_knee)
        return math.sqrt((rel * pred) ** 2 + qp["pe_err_floor"] ** 2)
    return qp["pe_err_floor"] if pred < qp["pe_err_knee"] \
        else qp["pe_err_frac"] * pred


def bundle_channel_terms(b, flash, qp, base_mask, lowpe_frac, lowpe_knee):
    """Per-channel (pe, pred, sigma2, sat) exactly as examine_bundle;
    returns (terms, chi2, ndf)."""
    pe, pe_err, sat = flash["pe"], flash["pe_err"], flash["sat"]
    pred = b["pred_pe"]
    terms = []
    chi2, ndf = 0.0, 0
    maxc, maxj = -1.0, -1
    for j in range(len(pe)):
        if not base_mask[j]:
            continue
        if not (pe[j] < qp["pe_ndf_knee"] and pred[j] < qp["pe_ndf_knee"]):
            ndf += 1
        pr = perr_of(qp, lowpe_frac, lowpe_knee, pred[j], pe_err[j])
        denom = pe[j] + pr * pr
        if qp["chi2_relax"] and b["close_to_PMT"] and \
                pe[j] - pred[j] > qp["chi2_pmt_excess"] and \
                pe[j] > qp["chi2_pmt_ratio"] * pred[j]:
            denom += (pe[j] * qp["chi2_pmt_inflate"]) ** 2
        is_sat = qp["chi2_sat_inflate"] > 0 and sat[j] > 0
        if is_sat:
            denom += (pe[j] * qp["chi2_sat_inflate"]) ** 2
        cur = (pred[j] - pe[j]) ** 2 / denom
        chi2 += cur
        if cur > maxc:
            maxc, maxj = cur, j
        terms.append(dict(ch=j, pe=pe[j], pred=pred[j], sigma2=denom,
                          sat=bool(sat[j] > 0)))
    if qp["chi2_relax"] and maxj >= 0 and pe[maxj] == 0 and pred[maxj] > 0:
        chi2 -= maxc - 1
    return terms, chi2, ndf


def ladder_branch(b, qp):
    """Exclusive ladder branch that admitted a consistent bundle."""
    if not b["ndf"]:
        return "none"
    c2n = b["chi2"] / b["ndf"]
    ks = b["ks_dis"]
    miss = b["at_x_boundary"] or b["close_to_PMT"] or b["window_truncated"]
    if b["ndf"] >= qp["highconsist_min_ndf"] and ks < qp["hc_clean_ks"] \
            and c2n < qp["hc_clean_c2"]:
        return "B1_clean"
    if b["ndf"] >= qp["highconsist_min_ndf"] and ks < qp["hc_good_ks"] \
            and c2n < qp["hc_good_c2"]:
        return "B2_good"
    if b["two_boundary"] and b["ndf"] >= qp["highconsist_min_ndf"] \
            and ks < qp["hc_tb_ks"] and c2n < qp["hc_tb_c2"]:
        return "B3_two_boundary"
    if miss and b["ndf"] >= qp["hc_miss_min_ndf"] \
            and ks < qp["hc_miss_ks"] and c2n < qp["hc_miss_c2"]:
        return "B4_miss"
    return "none"


def admission_path(b, qp):
    if b.get("xtpc_cathode_rescued"):
        return "xtpc_cathode_rescued"
    if b.get("xtpc_pin"):
        return "xtpc_pin"
    if b.get("xtpc_scenario1"):
        return "xtpc_scenario1"
    if b.get("consistent"):
        return "ladder_" + ladder_branch(b, qp)
    return "strength_only"


def robust_stats(vals):
    if not vals:
        return dict(n=0)
    n = len(vals)
    mean = sum(vals) / n
    rms = math.sqrt(sum(v * v for v in vals) / n)
    med = sorted(vals)[n // 2]
    mad = sorted(abs(v - med) for v in vals)[n // 2] * 1.4826
    return dict(n=n, mean=round(mean, 3), rms=round(rms, 3),
                median=round(med, 3), mad=round(mad, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="cathxa")
    ap.add_argument("--work-root", default="work")
    ap.add_argument("--gold",
                    default="work/ql_labels/wfresc/labels-evt298567.json")
    ap.add_argument("--decisions-dir", default="ql_display/decisions-cathxa")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--min-len-cm", type=float, default=25.0)
    ap.add_argument("--min-npoints", type=int, default=100)
    ap.add_argument("--lowpe-frac", type=float, default=2.0)
    ap.add_argument("--lowpe-knee", type=float, default=10.0)
    ap.add_argument("--out", default=None,
                    help="write JSON summary here (default: stdout only)")
    args = ap.parse_args()

    truth = load_truth(args.gold, args.decisions_dir)

    pulls = defaultdict(list)          # (family, satkey) -> [pull]
    pulls_bin = defaultdict(list)      # (family, bin_lo) -> [pull]
    n_excluded_chi2 = 0
    n_confirmed = 0
    tax_agree = defaultdict(int)
    tax_phantom = defaultdict(int)
    missed_diag = defaultdict(int)
    missed_best = []

    for idx in range(NEVT):
        evt = evt_of_idx(idx)
        calib_path = os.path.join(args.work_root, f"{RUN}_{idx}_{args.tag}",
                                  f"calib-evt{evt}.json")
        if not os.path.exists(calib_path):
            print(f"[warn] missing {calib_path}")
            continue
        with open(calib_path) as fh:
            d = json.load(fh)
        qp = d["quality_params"]
        opdets = d["opdets"]
        base_mask = [1 if (o["active"] and not o["auto_masked"]) else 0
                     for o in opdets]
        fam = {o["ch"]: family_of(o) for o in opdets}
        clusters = {c["uid"]: c for c in d["clusters"]}
        long_uid = {u: (cluster_len_cm(c) >= args.min_len_cm
                        or c["npoints"] >= args.min_npoints)
                    for u, c in clusters.items()}
        flash_by_gid = {f["gid"]: f for f in d["flashes"]}

        entries_by_uid = defaultdict(list)
        for e in truth[evt]:
            entries_by_uid[e["uid"]].append(e)

        cand_by_uid = defaultdict(list)   # ALL bundles (candidates)
        for b in d["bundles"]:
            cand_by_uid[b["main_cluster"]].append(b)

        auto_times_by_uid = defaultdict(list)
        for b in d["bundles"]:
            if b.get("auto_selected"):
                auto_times_by_uid[b["main_cluster"]].append(
                    flash_by_gid[b["flash_gid"]]["time"])

        for b in d["bundles"]:
            if not b.get("auto_selected"):
                continue
            uid = b["main_cluster"]
            if uid not in clusters or not long_uid.get(uid):
                continue
            flash = flash_by_gid[b["flash_gid"]]
            e = find_truth(entries_by_uid, uid, flash["time"], args.tol)
            if e is None or e["conf"] not in OBJECTIVE_TIERS:
                continue
            path = admission_path(b, qp)
            if e["positive"]:
                tax_agree[path] += 1
            else:
                tax_phantom[path] += 1
            if not e["positive"]:
                continue
            # ---- pulls over confirmed matches only ----
            terms, chi2, ndf = bundle_channel_terms(
                b, flash, qp, base_mask, args.lowpe_frac, args.lowpe_knee)
            if ndf != b["ndf"] or \
                    abs(chi2 - b["chi2"]) > 0.02 * max(1.0, abs(b["chi2"])):
                n_excluded_chi2 += 1
                continue
            n_confirmed += 1
            for t in terms:
                if t["pe"] < 0.5 and t["pred"] < 0.5:
                    continue           # empty-empty channels carry no info
                pull = (t["pe"] - t["pred"]) / math.sqrt(t["sigma2"])
                key_fam = fam[t["ch"]]
                pulls[(key_fam, "sat" if t["sat"] else "nom")].append(pull)
                for lo, hi in zip(PRED_BINS[:-1], PRED_BINS[1:]):
                    if lo <= t["pred"] < hi:
                        pulls_bin[(key_fam, lo)].append(pull)
                        break

        # ---- missed-positive diagnosis ----
        for e in truth[evt]:
            if not e["positive"] or e["conf"] not in OBJECTIVE_TIERS:
                continue
            uid = e["uid"]
            if uid not in clusters or not long_uid.get(uid):
                continue
            if any(abs(t - e["time"]) <= args.tol
                   for t in auto_times_by_uid.get(uid, ())):
                continue               # covered
            cands = [b for b in cand_by_uid.get(uid, ())
                     if abs(flash_by_gid[b["flash_gid"]]["time"]
                            - e["time"]) <= args.tol]
            if not cands:
                missed_diag["no_candidate_bundle"] += 1
            else:
                best = min(cands, key=lambda b: (b["ks_dis"],
                                                 b["chi2"] / max(1, b["ndf"])))
                sel_elsewhere = bool(auto_times_by_uid.get(uid))
                key = ("candidate_unselected_cluster_matched_elsewhere"
                       if sel_elsewhere else
                       "candidate_unselected_cluster_unmatched")
                missed_diag[key] += 1
                missed_best.append(dict(
                    evt=evt, uid=uid, time=round(e["time"], 2),
                    ks=round(best["ks_dis"], 3),
                    c2n=round(best["chi2"] / max(1, best["ndf"]), 1),
                    ndf=best["ndf"], strength=round(best["strength"], 3),
                    consistent=best["consistent"],
                    elsewhere=sel_elsewhere))

    # ---------- report ----------
    out = dict(tag=args.tag, n_confirmed_bundles=n_confirmed,
               n_excluded_chi2_mismatch=n_excluded_chi2)
    print(f"# pulls over {n_confirmed} confirmed long-track matches "
          f"({n_excluded_chi2} excluded: chi2 not reproducible)\n")
    print("| family | subset | n | mean | rms | median | mad |")
    print("|---|---|---|---|---|---|---|")
    fam_stats = {}
    for key in sorted(pulls):
        s = robust_stats(pulls[key])
        fam_stats["/".join(key)] = s
        print(f"| {key[0]} | {key[1]} | {s['n']} | {s['mean']} | {s['rms']} "
              f"| {s['median']} | {s['mad']} |")
    out["family_stats"] = fam_stats

    print("\n| family | pred-PE bin | n | mean | rms | mad |")
    print("|---|---|---|---|---|---|")
    bin_stats = {}
    for key in sorted(pulls_bin, key=lambda k: (k[0], k[1])):
        s = robust_stats(pulls_bin[key])
        bin_stats[f"{key[0]}/{key[1]}"] = s
        print(f"| {key[0]} | [{key[1]}, ...) | {s['n']} | {s['mean']} "
              f"| {s['rms']} | {s['mad']} |")
    out["bin_stats"] = bin_stats

    print("\n# admission-path taxonomy (exclusive; objective long-track "
          "judged autos)\n")
    print("| path | agree | phantom | phantom% |")
    print("|---|---|---|---|")
    tax = {}
    for path in sorted(set(tax_agree) | set(tax_phantom),
                       key=lambda p: -tax_phantom.get(p, 0)):
        a, p = tax_agree.get(path, 0), tax_phantom.get(path, 0)
        pct = f"{100.0*p/(a+p):.1f}%" if a + p else "n/a"
        tax[path] = dict(agree=a, phantom=p)
        print(f"| {path} | {a} | {p} | {pct} |")
    out["taxonomy"] = tax

    print("\n# missed-positive diagnosis\n")
    print("| mode | n |")
    print("|---|---|")
    for k, v in sorted(missed_diag.items(), key=lambda kv: -kv[1]):
        print(f"| {k} | {v} |")
    out["missed_diag"] = dict(missed_diag)
    out["missed_best_candidates"] = missed_best

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
