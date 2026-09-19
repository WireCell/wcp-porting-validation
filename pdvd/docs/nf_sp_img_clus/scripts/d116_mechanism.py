#!/usr/bin/env python3
"""doc pdvd/116 sec 2 -- WHICH tagger decision moved each tag between two arms, and how far the tagger's inputs moved
on the whole common candidate set.  Read-only.  Fork by duplication of the join of d107_churn.py (untouched: the
(cluster, pass index) keying of T_stm_pass / T_stm_eval and its status dictionary) generalised to any (base, arm)
pair, plus the T_stm_michel decision fields and the end geometry of d106_end_geometry.py.

Per mover (from d116_movers.py's tsv) the first decision that differs, in the tagger's order:
  CAND   the cluster is a candidate in one arm only: T_stm_pass status A -> B (0 accepted, 3 = the STM evaluation
         itself, 2 left charge, 4 mid-point tracks, 5 proton, 7 pre-fit exit, 8 no fit, -1 no row), with ks1 / ratio1
         of the deciding pass in both arms where an eval row exists;
  BITS   candidate in both, is_stm differs: the reject bits added / removed (bits_string names);
  STM    michel_found differs because is_stm was lost (the Michel is read only on an accepted stopper);
  MICHEL candidate and is_stm in both, michel_found differs: conn type, KE, length, distance, veto counters in both;
plus the geometry: |d tagger_stop|, |d stop|, muon_len B/A, plateau_med B/A, the shape-test margin ks_mu + ks_margin -
ks_flat (< 0 passes), plateau_med / mip_dqdx against the window, and the distance of each A value to its threshold
(the "near threshold" flag: a sub-step move crossed a boundary the tagger was already sitting on).

Scale check (every candidate common to both arms, no truth): paired quantiles of the B/A ratio or B-A shift of
plateau_med, ks_mu - ks_flat, ratio_mu, muon_len, michel_ke_best (Michel in both), |d stop|.

Usage: d116_mechanism.py --det pdhd --a d115hoff --b d115hp3bwp05 --movers figs/116_movers_pdhd.tsv --out figs/116_mechanism_pdhd
"""
import argparse, collections, csv, glob, os, sys
sys.dont_write_bytecode = True
import numpy as np
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
SNAME = {0: "accepted", 1: "TGM", 2: "left-charge", 3: "eval-failed", 4: "mid-point-trk", 5: "proton", 6: "default",
         7: "pre-fit-exit", 8: "no-fit", -1: "no-row"}
BITS = ["no_chain", "stop_unmatched", "no_bragg", "shape_flat", "not_muon_pid", "continuation", "stop_near_boundary",
        "vertex_hadron", "short", "profile_sparse", "plateau_off_mip", "stop_into_dead", "cluster_not_track",
        "profile_geometry", "readout_edge"]
# the production operating points (driver bags <det>/wct-pr-perevt.jsonnet; C++ defaults where the bag is silent)
BAG = {"pdhd": dict(ks_margin=-0.10, plateau_lo=0.6, plateau_hi=1.6, mip=56000.0, topo_ke_min=5.0, topo_len_min=1.5,
                    range_dis=5.0, range_ke=10.0),
       "pdvd": dict(ks_margin=-0.02, plateau_lo=0.6, plateau_hi=2.0, mip=55000.0, topo_ke_min=3.0, topo_len_min=3.0,
                    range_dis=3.0, range_ke=10.0)}
MB = ["cluster_id", "is_stm", "michel_found", "reject_bits", "pass", "has_pass", "michel_conn_type", "michel_ke_best",
      "michel_len", "michel_dis_cm", "michel_mip", "michel_kink_deg", "michel_far_len", "topology_cleared_bits",
      "stop_x", "stop_y", "stop_z", "tagger_stop_x", "tagger_stop_y", "tagger_stop_z", "entry_x", "entry_y", "entry_z",
      "muon_len", "plateau_med", "tail_med", "ks_mu", "ks_flat", "ratio_mu", "ratio_flat", "contrast", "contrast_expected",
      "bragg_valid", "n_profile_pts", "n_live_pts", "chain_coverage", "unsupported_len_cm", "n_michel_veto",
      "n_michel_range_veto", "n_stop_arms", "n_retreat", "n_split", "retreat_len", "in_fv", "stop_dis", "n_michel_segs"]
EVCOL = ["strong", "verdict", "peak_range", "offset_length", "com_range", "ks1", "ks2", "ratio1", "ratio2", "res_length", "ave_res_dqdx"]


def bits(v):
    v = int(v)
    return "STM" if v == 0 else "|".join(n for i, n in enumerate(BITS) if v & (1 << i))


def load(det, arm):
    cand, st, stp, evp = {}, {}, {}, {}
    for d in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}")):
        e = os.path.basename(d)[:-len(arm) - 1]
        try:
            t = uproot.open(f"{d}/tracking-pr.root")["T_stm_michel"].arrays(MB, library="np")
        except Exception:
            continue
        for i in range(len(t["cluster_id"])):
            cand[f"{e}/{int(t['cluster_id'][i])}"] = {c: float(t[c][i]) for c in MB if c != "cluster_id"}
        try:
            f = uproot.open(f"{d}/tracking-stm.root")
            p = f["T_stm_pass"].arrays(library="np")
            v = f["T_stm_eval"].arrays(EVCOL + ["cluster_id", "pass"], library="np")
        except Exception:
            continue
        for i in range(len(p["cluster_id"])):
            k = f"{e}/{int(p['cluster_id'][i])}"
            sv, pi = int(p["status"][i]), int(p["pass"][i])
            stp[(k, pi)] = sv
            st[k] = 0 if (st.get(k) == 0 or sv == 0) else (sv if k not in st else st[k])
        for cid in set(v["cluster_id"].astype(int)):
            k = f"{e}/{cid}"
            for pi in set(v["pass"][v["cluster_id"] == cid].astype(int)):
                m = (v["cluster_id"] == cid) & (v["pass"] == pi)
                row = {c: float(np.max(v[c][m]) if c == "verdict" else np.median(v[c][m])) for c in EVCOL}
                row["n_eval"] = int(m.sum())
                evp[(k, pi)] = row
    return cand, st, stp, evp


def deciding_eval(k, stp, evp):
    dec = next((pi for (kk, pi), sv in stp.items() if kk == k and sv == 0), None)
    if dec is None:
        ps = [pi for (kk, pi) in evp if kk == k]
        dec = max(ps) if ps else None
    return evp.get((k, dec)) if dec is not None else None


def dist(a, b, pre):
    return float(np.linalg.norm([a[pre + c] - b[pre + c] for c in "xyz"]))


def near(det, r):
    """A-arm distances to the thresholds (the tagger's own margins)."""
    B = BAG[det]
    out = {}
    if r["ks_flat"] > 0 or r["ks_mu"] > 0:
        out["ks"] = r["ks_mu"] + B["ks_margin"] - r["ks_flat"]            # < 0 passes shape_flat
    if r["plateau_med"] > 0:
        pm = r["plateau_med"] / B["mip"]
        out["plateau/mip"] = pm
        out["plateau_edge"] = min(pm - B["plateau_lo"], B["plateau_hi"] - pm)   # > 0 inside the window
    if r["michel_found"] > 0:
        out["ke-topo_min"] = r["michel_ke_best"] - B["topo_ke_min"]
        out["len-topo_min"] = r["michel_len"] - B["topo_len_min"]
    return out


def fmt(d):
    return " ".join(f"{k}={v:.3g}" for k, v in d.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--movers", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, As, Asp, Aev = load(a.det, a.a)
    Bc, Bs, Bsp, Bev = load(a.det, a.b)
    mv = list(csv.DictReader(open(a.movers), delimiter="\t"))
    L = [f"# doc pdvd/116 mechanism {a.det}: A={a.a} ({len(A)} candidates) B={a.b} ({len(Bc)} candidates); movers {len(mv)} rows",
         f"# operating point {BAG[a.det]}"]
    classes = collections.Counter()
    rows_out = []
    for r in mv:
        k, metric, cls = r["key"], r["metric"], r["class"]
        ra, rb = A.get(k), Bc.get(k)
        geo = ""
        if ra and rb:
            geo = (f"|d tagger_stop| {dist(ra, rb, 'tagger_stop_'):.1f} cm, |d stop| {dist(ra, rb, 'stop_'):.1f}, "
                   f"muon_len {ra['muon_len']:.0f}->{rb['muon_len']:.0f}, plateau_med B/A {rb['plateau_med'] / max(ra['plateau_med'], 1e-9):.3f}, "
                   f"ks margin A {near(a.det, ra).get('ks', float('nan')):+.3f} B {near(a.det, rb).get('ks', float('nan')):+.3f}, "
                   f"plateau/mip A {near(a.det, ra).get('plateau/mip', float('nan')):.2f} B {near(a.det, rb).get('plateau/mip', float('nan')):.2f}")
        if (ra is None) != (rb is None):
            sa, sb = As.get(k, -1), Bs.get(k, -1)
            ea, eb = deciding_eval(k, Asp, Aev), deciding_eval(k, Bsp, Bev)
            how = f"CAND {SNAME[sa]}->{SNAME[sb]}"
            det_ = (f"ks1 {ea['ks1'] if ea else float('nan'):.3f}->{eb['ks1'] if eb else float('nan'):.3f}, "
                    f"ratio1 {ea['ratio1'] if ea else float('nan'):.3f}->{eb['ratio1'] if eb else float('nan'):.3f}, "
                    f"res_length {ea['res_length'] if ea else float('nan'):.1f}->{eb['res_length'] if eb else float('nan'):.1f}")
            side = ra or rb
            det_ += f"; the tagged arm: muon_len {side['muon_len']:.0f}, bits {bits(side['reject_bits'])}, michel {int(side['michel_found'])} conn {int(side['michel_conn_type'])} ke {side['michel_ke_best']:.1f}"
        elif metric == "is_stm" or (ra["is_stm"] != rb["is_stm"]):
            ba, bb = int(ra["reject_bits"]), int(rb["reject_bits"])
            add = [n for i, n in enumerate(BITS) if (bb & ~ba) & (1 << i)]
            rem = [n for i, n in enumerate(BITS) if (ba & ~bb) & (1 << i)]
            how = ("BITS" if metric == "is_stm" else "STM") + " +" + "|".join(add or ["-"]) + " -" + "|".join(rem or ["-"])
            det_ = f"A {bits(ba)} -> B {bits(bb)}; topo_cleared A {int(ra['topology_cleared_bits'])} B {int(rb['topology_cleared_bits'])}; {fmt(near(a.det, ra))} | B {fmt(near(a.det, rb))}"
        else:
            side = ra if cls in ("tp_lost", "fp_gone") else rb        # the arm that HAS the Michel
            other = rb if side is ra else ra
            how = (f"MICHEL conn {int(ra['michel_conn_type'])}->{int(rb['michel_conn_type'])}"
                   + (" no-arm" if other["n_stop_arms"] == 0 else "")
                   + (" vetoed" if other["n_michel_veto"] + other["n_michel_range_veto"] > 0 else ""))
            det_ = (f"ke {ra['michel_ke_best']:.1f}->{rb['michel_ke_best']:.1f} MeV, len {ra['michel_len']:.1f}->{rb['michel_len']:.1f} cm, "
                    f"dis {ra['michel_dis_cm']:.1f}->{rb['michel_dis_cm']:.1f}, mip {ra['michel_mip']:.2f}->{rb['michel_mip']:.2f}, kink {ra['michel_kink_deg']:.0f}->{rb['michel_kink_deg']:.0f}, "
                    f"n_stop_arms {int(ra['n_stop_arms'])}->{int(rb['n_stop_arms'])}, veto {int(ra['n_michel_veto'])}/{int(ra['n_michel_range_veto'])}->{int(rb['n_michel_veto'])}/{int(rb['n_michel_range_veto'])}, "
                    f"retreat/split {int(ra['n_retreat'])}/{int(ra['n_split'])}->{int(rb['n_retreat'])}/{int(rb['n_split'])}; {fmt(near(a.det, side))}")
        classes[(metric, cls, how.split(" ")[0] + ("" if how.startswith("MICHEL") else " " + how.split(" ")[1] if how.startswith("CAND") else ""))] += 1
        L.append(f"{metric:12s} {cls:8s} {k:16s} truth {r['verdict']}/{r['michel_kind']} [{r['source']}]  {how}\n"
                 f"      {det_}\n      {geo}")
        rows_out.append((metric, cls, k, how, det_, geo))
    L.insert(2, "\n== class table (metric, mover class, first differing decision)")
    for (metric, cls, how), n in sorted(classes.items()):
        L.insert(3, f"  {metric:12s} {cls:8s} {n:3d}  {how}")
    # ---- scale check on the common candidates ----
    common = sorted(set(A) & set(Bc))
    L.append(f"\n== scale check on {len(common)} common candidates (B relative to A; quantiles p10 p50 p90)")
    def q(v):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        return f"n {len(v)} p10 {np.percentile(v, 10):.3f} p50 {np.percentile(v, 50):.3f} p90 {np.percentile(v, 90):.3f}" if len(v) else "n 0"
    L.append("  plateau_med B/A            " + q([Bc[k]["plateau_med"] / A[k]["plateau_med"] for k in common if A[k]["plateau_med"] > 0 and Bc[k]["plateau_med"] > 0]))
    L.append("  (ks_mu - ks_flat) B - A    " + q([(Bc[k]["ks_mu"] - Bc[k]["ks_flat"]) - (A[k]["ks_mu"] - A[k]["ks_flat"]) for k in common if A[k]["ks_flat"] > 0 and Bc[k]["ks_flat"] > 0]))
    L.append("  ratio_mu B - A             " + q([Bc[k]["ratio_mu"] - A[k]["ratio_mu"] for k in common if A[k]["ratio_mu"] > 0 and Bc[k]["ratio_mu"] > 0]))
    L.append("  muon_len B/A               " + q([Bc[k]["muon_len"] / A[k]["muon_len"] for k in common if A[k]["muon_len"] > 0]))
    L.append("  |d tagger_stop| cm         " + q([dist(A[k], Bc[k], "tagger_stop_") for k in common]))
    L.append("  |d stop| cm                " + q([dist(A[k], Bc[k], "stop_") for k in common]))
    L.append("  michel_ke_best B/A (both)  " + q([Bc[k]["michel_ke_best"] / A[k]["michel_ke_best"] for k in common if A[k]["michel_found"] > 0 and Bc[k]["michel_found"] > 0 and A[k]["michel_ke_best"] > 0]))
    L.append("  michel_len B - A cm (both) " + q([Bc[k]["michel_len"] - A[k]["michel_len"] for k in common if A[k]["michel_found"] > 0 and Bc[k]["michel_found"] > 0]))
    nb = collections.Counter()
    for k in common:
        if A[k]["is_stm"] == Bc[k]["is_stm"] and A[k]["michel_found"] == Bc[k]["michel_found"]:
            nb["same tags"] += 1
        else:
            nb["tag changed"] += 1
    L.append(f"  tags on the common candidates: {dict(nb)}; |d stop| p50 of the changed "
             f"{np.median([dist(A[k], Bc[k], 'stop_') for k in common if not (A[k]['is_stm'] == Bc[k]['is_stm'] and A[k]['michel_found'] == Bc[k]['michel_found'])] or [float('nan')]):.1f} cm "
             f"vs unchanged {np.median([dist(A[k], Bc[k], 'stop_') for k in common if A[k]['is_stm'] == Bc[k]['is_stm'] and A[k]['michel_found'] == Bc[k]['michel_found']] or [float('nan')]):.1f} cm")
    # candidacy churn on the whole set (doc 107 table), truth-free
    lost = sorted(set(A) - set(Bc)); gain = sorted(set(Bc) - set(A))
    L.append(f"\n== candidacy churn (truth-free): lost {len(lost)}, gained {len(gain)}")
    for nm, ks in (("lost", lost), ("gained", gain)):
        cnt = collections.Counter((As.get(k, -1), Bs.get(k, -1)) for k in ks)
        L.append(f"  {nm}: " + ", ".join(f"{SNAME[x]}->{SNAME[y]} {n}" for (x, y), n in cnt.most_common(6)))
        side = A if nm == "lost" else Bc
        L.append(f"     carrying is_stm {sum(1 for k in ks if side[k]['is_stm'] > 0)}, michel {sum(1 for k in ks if side[k]['michel_found'] > 0)}; muon_len p50 {np.median([side[k]['muon_len'] for k in ks] or [float('nan')]):.0f} cm")
    open(a.out + ".txt", "w").write("\n".join(L) + "\n")
    with open(a.out + ".tsv", "w") as f:
        f.write("metric\tclass\tkey\thow\tdetail\tgeometry\n")
        for row in rows_out:
            f.write("\t".join(row) + "\n")
    print("\n".join(l for l in L if not l.startswith("      ") and not l[:1].islower() and not l.startswith("is_stm") and not l.startswith("michel_found")))


if __name__ == "__main__":
    main()
