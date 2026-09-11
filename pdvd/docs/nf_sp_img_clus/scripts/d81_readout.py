#!/usr/bin/env python3
"""doc pdvd/81 -- read the charge-based Michel energy off an ON arm.

Usage: python3 d81_readout.py --det pdvd --arm p81vq2d [--off p81voff] [--json out.json]

Per candidate with a Michel (michel_found == 1): the new estimator against the
chain's own energies, its per-plane pieces, how often the plane switch fired,
the validity census, the shared-cell fraction, and the extremes by name.  From
T_stm_michel_2d: cell counts per role, the STM footprint size, and a closure
check that the sum of the muon-only prediction over every cell equals the sum
over the role-1 footprint plus the shared Michel cells (nothing is lost between
the two tables).  Report only; nothing here decides anything.
"""
import argparse, glob, json, os, sys
import numpy as np
import uproot

def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="pdvd")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--off", default=None, help="the OFF twin: verdict branches must be identical")
    ap.add_argument("--json", default=None)
    ap.add_argument("--img", default="/nfs/data/1/xqian/toolkit-dev/wcp-porting-img")
    a = ap.parse_args()
    rows = []; cells_tot = {1: 0, 3: 0, 4: 0}; closure = []; n_files = 0; n_2d = 0
    for d in sorted(glob.glob("%s/%s/work/*_%s" % (a.img, a.det, a.arm))):
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn): continue
        n_files += 1
        F = uproot.open(fn)
        keys = set(k.split(";")[0] for k in F.keys())
        if "T_stm_michel" not in keys: continue
        T = F["T_stm_michel"].arrays(library="np")
        ev = os.path.basename(d)[: -len(a.arm) - 1]
        C = None
        if "T_stm_michel_2d" in keys:
            n_2d += 1
            C = F["T_stm_michel_2d"].arrays(library="np")
        for i in range(len(T["cluster_id"])):
            r = {k: (float(T[k][i]) if T[k].dtype.kind == "f" else int(T[k][i])) for k in T}
            r["event"] = ev; r["key"] = "%s/%d" % (ev, r["cluster_id"])
            if C is not None:
                s = C["cluster_id"] == r["cluster_id"]
                for role in (1, 3, 4):
                    n = int(((C["role"] == role) & s).sum()); r["n_cells_role%d" % role] = n; cells_tot[role] += n
                pm = C["pred_mu"][s]; rl = C["role"][s]; sh = C["shared"][s]
                tot = float(pm.sum()); part = float(pm[(rl == 1) | (sh == 1)].sum())
                r["closure_pred_mu"] = (tot, part)
                if tot > 0: closure.append(part / tot)
                r["shared_frac"] = float(pm[rl == 3].sum() / max(C["charge"][s][rl == 3].sum(), 1e-9))
            rows.append(r)
    print("files %d, with T_stm_michel_2d %d, candidates %d" % (n_files, n_2d, len(rows)))
    if not rows: sys.exit(1)
    have = [r for r in rows if "michel_q2d_valid" in r]
    print("michel_q2d branches present on %d candidates" % len(have))
    if not have: sys.exit(1)
    import collections
    print("validity: %s" % dict(collections.Counter((r["michel_q2d_valid"], r["michel_q2d_reason"]) for r in have)))
    M = [r for r in have if r.get("michel_found", 0) == 1 and r["michel_q2d_valid"] == 1]
    print("Michel-carrying valid candidates: %d" % len(M))
    if M:
        tot = np.array([r["michel_ke_total"] if "michel_ke_total" in r else r["michel_ke_best"] for r in M])
        best = np.array([r["michel_ke_best"] for r in M]); q2d = np.array([r["michel_ke_q2d"] for r in M])
        q2dt = np.array([r["michel_ke_q2d_total"] for r in M]); chg = np.array([r["michel_ke_charge"] for r in M])
        ok = tot > 0
        rat = q2dt[ok] / tot[ok]; rat_b = q2d[best > 0] / best[best > 0]
        print("michel_ke_q2d_total / michel_ke_total : n %d p10 %.3f p50 %.3f p90 %.3f mean %.3f" % (len(rat), q(rat, 10), q(rat, 50), q(rat, 90), float(rat.mean())))
        print("michel_ke_q2d / michel_ke_best        : n %d p10 %.3f p50 %.3f p90 %.3f" % (len(rat_b), q(rat_b, 10), q(rat_b, 50), q(rat_b, 90)))
        okc = chg > 0
        if okc.any(): print("michel_ke_q2d / michel_ke_charge      : p50 %.3f (the shower-pair charge estimator)" % q(q2d[okc] / chg[okc], 50))
        print("michel_ke_q2d_total MeV               : p10 %.1f p50 %.1f p90 %.1f max %.1f | > 60 MeV: %d (michel_ke_total > 60: %d)" % (
            q(q2dt, 10), q(q2dt, 50), q(q2dt, 90), float(q2dt.max()), int((q2dt > 60).sum()), int((tot > 60).sum())))
        neg = [r for r in M if r["michel_q2d"] <= 0]
        print("non-positive combined charge (energy 0): %d" % len(neg))
        dp = collections.Counter(r["michel_q2d_dropped_plane"] for r in M)
        print("plane switch: dropped plane census %s" % dict(sorted(dp.items())))
        u = np.array([r["michel_q2d_u"] for r in M]); v = np.array([r["michel_q2d_v"] for r in M]); w = np.array([r["michel_q2d_w"] for r in M])
        m = np.maximum(np.abs(u) + np.abs(v) + np.abs(w), 1e-9) / 3
        spread = (np.max([u, v, w], axis=0) - np.min([u, v, w], axis=0)) / m
        print("per-plane spread (max-min)/mean|q|     : p50 %.3f p90 %.3f" % (q(spread, 50), q(spread, 90)))
        nu = np.array([r["michel_q2d_n_u"] for r in M]); print("Michel cells per plane (u/v/w) p50    : %d / %d / %d" % (
            q(nu, 50), q(np.array([r["michel_q2d_n_v"] for r in M]), 50), q(np.array([r["michel_q2d_n_w"] for r in M]), 50)))
        mu = np.array([r["michel_q2d_mu_u"] + r["michel_q2d_mu_v"] + r["michel_q2d_mu_w"] for r in M])
        raw = np.array([r.get("michel_q2d_raw_u", 0) + r.get("michel_q2d_raw_v", 0) + r.get("michel_q2d_raw_w", 0) for r in M])
        meas = mu + raw
        f = mu[meas > 0] / meas[meas > 0]
        print("subtracted muon charge / measured Michel-cell charge: p10 %.3f p50 %.3f p90 %.3f" % (q(f, 10), q(f, 50), q(f, 90)))
        if "michel_q2d_nx_u" in M[0]:
            nx = np.array([r["michel_q2d_nx_u"] + r["michel_q2d_nx_v"] + r["michel_q2d_nx_w"] for r in M])
            nn = np.array([r["michel_q2d_n_u"] + r["michel_q2d_n_v"] + r["michel_q2d_n_w"] for r in M])
            fx = nx[nn > 0] / nn[nn > 0]
            head = u + v + w
            print("cross-shared Michel cells / all Michel cells   : p10 %.3f p50 %.3f p90 %.3f | candidates with none %d of %d" % (q(fx, 10), q(fx, 50), q(fx, 90), int((nx == 0).sum()), len(nx)))
            ok2 = np.abs(raw) > 0
            print("headline sum / raw (measured - muon) sum, 3 planes: p10 %.3f p50 %.3f p90 %.3f" % tuple(q(head[ok2] / raw[ok2], p) for p in (10, 50, 90)))
        gam = [r for r in M if r.get("michel_ke_q2d_gamma", 0) > 0]
        print("candidates with a gamma term: %d, gamma MeV p50 %.2f (chain michel_ke_gamma p50 %.2f)" % (
            len(gam), q(np.array([r["michel_ke_q2d_gamma"] for r in gam]), 50), q(np.array([r.get("michel_ke_gamma", 0) for r in gam]), 50)))
        order = np.argsort(rat) if ok.any() else []
        Mo = [M[i] for i in np.where(ok)[0]]
        print("extremes of michel_ke_q2d_total / michel_ke_total (key: q2d_total vs total, best, planes u/v/w e-, mu-subtracted, dropped):")
        for i in list(order[:5]) + list(order[-5:]):
            r = Mo[i]
            print("  %-16s %6.2f vs %6.2f (best %6.2f, len %5.1f cm, conn %d) | %8.0f %8.0f %8.0f | mu %8.0f | drop %d | cells %d/%d/%d" % (
                r["key"], r["michel_ke_q2d_total"], r.get("michel_ke_total", r["michel_ke_best"]), r["michel_ke_best"], r.get("michel_len", -1), r.get("michel_conn_type", -1),
                r["michel_q2d_u"], r["michel_q2d_v"], r["michel_q2d_w"],
                r["michel_q2d_mu_u"] + r["michel_q2d_mu_v"] + r["michel_q2d_mu_w"], r["michel_q2d_dropped_plane"],
                r["michel_q2d_n_u"], r["michel_q2d_n_v"], r["michel_q2d_n_w"]))
        big = sorted(M, key=lambda r: -r.get("michel_ke_total", r["michel_ke_best"]))[:3]
        print("the largest chain energies:")
        for r in big: print("  %-16s total %6.2f -> q2d_total %6.2f" % (r["key"], r.get("michel_ke_total", r["michel_ke_best"]), r["michel_ke_q2d_total"]))
        zero = [r for r in M if r["michel_ke_q2d_total"] <= 0]
        print("the zero-energy ones (planes u/v/w e-, mu-subtracted per plane, cells):")
        for r in zero[:12]:
            print("  %-16s best %6.2f len %5.1f conn %d | %8.0f %8.0f %8.0f | mu %8.0f %8.0f %8.0f | cells %d/%d/%d" % (
                r["key"], r["michel_ke_best"], r.get("michel_len", -1), r.get("michel_conn_type", -1), r["michel_q2d_u"], r["michel_q2d_v"], r["michel_q2d_w"],
                r["michel_q2d_mu_u"], r["michel_q2d_mu_v"], r["michel_q2d_mu_w"], r["michel_q2d_n_u"], r["michel_q2d_n_v"], r["michel_q2d_n_w"]))
    if n_2d:
        print("T_stm_michel_2d cells: role 1 (STM footprint) %d, role 3 (Michel) %d, role 4 (gamma) %d over %d files" % (cells_tot[1], cells_tot[3], cells_tot[4], n_2d))
        if closure: print("closure sum(pred_mu | role1 or shared) / sum(pred_mu | all rows): min %.4f p50 %.4f (1.000 = every predicted-muon cell is in the table)" % (min(closure), q(np.array(closure), 50)))
    if a.off:
        # verdict branches identical to the OFF twin, candidate by candidate
        V = ["is_stm", "reject_bits", "michel_found", "michel_conn_type", "michel_ke_best", "michel_ke_total", "muon_ke_best"]
        same = diff = 0
        for r in have:
            fn = "%s/%s/work/%s_%s/tracking-pr.root" % (a.img, a.det, r["event"], a.off)
            if not os.path.exists(fn): continue
            T = uproot.open(fn)["T_stm_michel"].arrays(library="np")
            s = T["cluster_id"] == r["cluster_id"]
            if s.sum() != 1: diff += 1; continue
            ok = all((k not in T) or (T[k][s][0] == r[k]) for k in V)
            same += ok; diff += (not ok)
        print("verdict branches vs %s: identical %d, differing %d" % (a.off, same, diff))
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, default=str)
        print("wrote", a.json)

if __name__ == "__main__":
    main()
