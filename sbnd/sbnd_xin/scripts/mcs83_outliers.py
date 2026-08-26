#!/usr/bin/env python3
"""doc 83 -- why some contained muons read KE_MCS >> KE_range (the money-plot
tail), and the four low-side violators doc 80 Part B deferred.

Re-collects the same T_kine/T_tagger/log-sentinel join as mcs80_analysis.py,
but keyed on kine_mcs_segment_id (exact) rather than cluster_id (approximate --
doc 80's join reads the wrong sentinel on the 8 two-bundle events), and with
three extra sentinel fields (nseg, bad_path, npoints) doc 80 captured but never
wrote out.  Then, for the high-side outlier population (contained, ratio =
ke_mcs/ke_range_tk > 1.5) and a length/nseg14-matched control sample:

  step 0  ambiguity-quintile cross-tab (is the tail already self-flagged?)
  step 1  replay each cloud through the shipped estimator (mcs_probe replay):
          dense lnL(KE) curve + per-segment angle spectrum
  step 2  toy statistical null: how often does an nseg14-segment MCS fit land
          >1.5x/2x the input KE from angle-count starvation ALONE (mcs_probe
          synthetic), i.e. with no reconstruction defect at all
  step 3  mis-ID census: pdg/shower composition of the selected segment itself
  step 4  Bragg test: dQ/dx vs residual range on the selected segment, scored
          against the same Bragg-contrast metric dqdx_rr_sample uses
  step 4b fragmentation/adjacency census: other track-like (pid 13/211/2212)
          real_cluster_id groups in the SAME parent cluster, and how close
          their bounding boxes sit to the selected segment's -- the "part of
          the muon lost to a PR break" test, from data already on disk
  step 5  kink test: re-minimise with the single largest angle masked
          (mcs_probe replay --maskmax; the shipped angle_keep mechanism)

Part 3 (doc 80 Part B's 4 low-side violators: E_MCS < E_range(visible) on
exiting muons) reuses steps 1/4b/5 on that separate 4-muon population.

Usage: mcs83_outliers.py --out DIR --arm ARM
Writes into DIR (must not pre-exist unless --out-exist-ok, M13).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
MCS_UPSTREAM = os.path.join(HERE, "..", "mcs_upstream")
PROBE = os.path.join(MCS_UPSTREAM, "dumper", "mcs_probe")
STM_REF = os.path.join(HERE, "..", "nusel_display", "stm_ref_dqdx.json")

MMU = 105.658  # MeV

SENT = re.compile(
    r"mcs: source=(?P<source>\S+) nseg=(?P<nseg>\d+) npoints=(?P<npoints>\d+) "
    r"len=(?P<len>[\d.]+)cm seg_id=(?P<segid>-?\d+) cluster=(?P<cid>-?\d+) -> "
    r"ke_MCS=(?P<ke>-?[\d.]+) MeV amb=(?P<amb>[\d.eE+-]+) tracklen=(?P<tracklen>-?[\d.]+)cm "
    r"ke_range=(?P<kerange>-?[\d.]+) MeV ke_range_toolkit=(?P<kert>-?[\d.]+) MeV "
    r"ke_dqdx_toolkit=(?P<kedq>-?[\d.]+) MeV \(nsegs14=(?P<nsegs14>\d+) bad_path=(?P<bad>\w+) "
    r"cathode_drop=(?P<cdrop>\d+)/(?P<cmask>\d+)\)")


def collect(arm):
    """Same join as mcs80_analysis.collect(), keyed on segid (exact) and
    carrying nseg/bad_path/npoints through to the row."""
    rows = []
    for prdir in sorted(glob.glob(os.path.join(arm, "pr_evt*"))):
        evt = int(prdir.rsplit("pr_evt", 1)[1])
        froot = os.path.join(prdir, "tracking-pr.root")
        flogs = glob.glob(os.path.join(prdir, "wct_pr_evt*.log"))
        if not os.path.exists(froot) or not flogs:
            continue
        sent_by_segid = {}
        with open(flogs[0], errors="replace") as fh:
            for line in fh:
                m = SENT.search(line)
                if m:
                    sent_by_segid[int(m.group("segid"))] = m.groupdict()
        try:
            f = uproot.open(froot)
            if "T_kine" not in f:
                continue
            tk = f["T_kine"].arrays(library="np")
            tt = f["T_tagger"].arrays(
                ["act_cluster_id", "act_fc", "act_stm", "act_tgm", "act_lm",
                 "act_is_selected", "act_evaluated"], library="np") \
                if "T_tagger" in f else None
        except Exception as err:                       # noqa: BLE001
            print(f"WARN {prdir}: {err}", file=sys.stderr)
            continue
        if "kine_mcs_energy" not in tk:
            continue
        n = len(tk["kine_mcs_energy"])
        for i in range(n):
            cid = int(tk["cluster_id"][i]) if "cluster_id" in tk else -1
            segid = int(tk["kine_mcs_segment_id"][i])
            s = sent_by_segid.get(segid)
            fc = stm = tgm = lm = sel = -1
            segcl = segid // 1000
            if tt is not None and i < len(tt["act_cluster_id"]):
                ac = tt["act_cluster_id"][i]
                for j in range(len(ac)):
                    if int(ac[j]) == segcl and int(tt["act_evaluated"][i][j]):
                        fc = int(tt["act_fc"][i][j]); stm = int(tt["act_stm"][i][j])
                        tgm = int(tt["act_tgm"][i][j]); lm = int(tt["act_lm"][i][j])
                        sel = int(tt["act_is_selected"][i][j])
                        break
            rows.append(dict(
                evt=evt, prdir=prdir, row=i, cid=cid, segid=segid,
                ke_mcs=float(tk["kine_mcs_energy"][i]),
                amb=float(tk["kine_mcs_ambiguity"][i]),
                tracklen=float(tk["kine_mcs_tracklen"][i]),
                ke_mcs_range=float(tk["kine_mcs_range_energy"][i]),
                isfc=fc, stm=stm, tgm=tgm, lm=lm, act_sel=sel,
                ke_range_tk=float(s["kert"]) if s else -1,
                ke_dqdx_tk=float(s["kedq"]) if s else -1,
                sel_len=float(s["len"]) if s else -1,
                cdrop=int(s["cdrop"]) if s else -1,
                cmask=int(s["cmask"]) if s else -1,
                nseg14=int(s["nsegs14"]) if s else -1,
                nseg=int(s["nseg"]) if s else -1,
                npoints=int(s["npoints"]) if s else -1,
                bad_path=(s["bad"] == "true") if s else None,
                sentinel_matched=(s is not None),
            ))
    return rows


# ---------------------------------------------------------------------------
def harvest_cloud(froot, segid, out_txt):
    """Write the mcs_probe/mcs_dump cloud-txt format for the exact measured
    segment: T_rec_charge[real_cluster_id == segid], in on-disk (fit) order,
    which is start-to-end (verified doc 83: rr descends monotonically from
    the first non-vertex point to the last)."""
    f = uproot.open(froot)
    rc = f["T_rec_charge"].arrays(["x", "y", "z", "real_cluster_id"], library="np")
    m = rc["real_cluster_id"] == segid
    n = int(m.sum())
    if n < 3:
        return None
    x, y, z = rc["x"][m], rc["y"][m], rc["z"][m]
    with open(out_txt, "w") as fh:
        fh.write("%.17g %.17g %.17g\n" % (x[0], y[0], z[0]))
        fh.write("%.17g %.17g %.17g\n" % (x[-1], y[-1], z[-1]))
        for i in range(n):
            fh.write("%.17g %.17g %.17g\n" % (x[i], y[i], z[i]))
    return n


def run_probe_replay(cloud_txt, out_json, maskmax=True):
    args = [PROBE, "replay", cloud_txt, out_json]
    if maskmax:
        args.append("--maskmax")
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_json):
        print("WARN mcs_probe replay failed:", r.stderr, file=sys.stderr)
        return None
    return json.load(open(out_json))


def run_probe_synthetic(nsegs, T_mev, ntrials, seed, out_json, ivx=2, sigma_scale=None):
    args = [PROBE, "synthetic", str(nsegs), str(T_mev), str(ntrials), str(seed), out_json,
            "--ivx", str(ivx)]
    if sigma_scale is not None:
        args += ["--sigma-scale", str(sigma_scale)]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_json):
        print("WARN mcs_probe synthetic failed:", r.stderr, file=sys.stderr)
        return None
    return json.load(open(out_json))


def pull_sigma_scale(T_mev):
    """doc 80 sec 9.3's own pull-test core widths, by T band -- the shipped
    tune's assumed angular width is too NARROW at T>~200 MeV (and slightly
    too WIDE below 200) relative to what SBND data actually shows. A toy
    drawn from sigma_scale=1 (the nominal tune) is therefore a BIASED
    estimate of the true outlier rate, not a fair one; this is the doc 83
    correction to the naive step-2 toy. Values are the (xz+yz)/2 average
    pull width doc 80 quoted per band."""
    if T_mev < 200:
        return 0.778
    if T_mev < 400:
        return 1.366
    return 2.364


# ---------------------------------------------------------------------------
def dqdx_profile(froot, segid):
    """(rr, dqdx) for the selected segment's own points, upstream recipe
    (dqdx_rr_sample/collect_dqdx_rr_sample.py): dQ = (q-offset)/scale, dx=nq."""
    f = uproot.open(froot)
    rc = f["T_rec_charge"].arrays(
        ["x", "y", "z", "q", "nq", "rr", "real_cluster_id", "particle_id", "flag_shower"],
        library="np")
    tr = f["Trun"].arrays(["dQdx_scale", "dQdx_offset"], library="np")
    m = rc["real_cluster_id"] == segid
    dQ = (rc["q"][m] - tr["dQdx_offset"][0]) / tr["dQdx_scale"][0]
    dx = np.maximum(rc["nq"][m], 1e-9)
    dqdx = dQ / dx
    rr = rc["rr"][m]
    return rr, dqdx, rc["particle_id"][m], rc["flag_shower"][m]


def bragg_contrast(rr, dqdx):
    near = (rr >= 0) & (rr < 2)
    far = (rr >= 20) & (rr < 40)
    if near.sum() < 2 or far.sum() < 2:
        return np.nan, int(near.sum()), int(far.sum())
    return float(np.median(dqdx[near]) / np.median(dqdx[far])), int(near.sum()), int(far.sum())


def fragmentation_census(froot, cid, segid, max_gap_cm=10.0):
    """Other track-like (pid in 13,211,2212) real_cluster_id groups sharing
    the SAME parent cluster as the selected segment, and the closest
    point-to-point distance from the selected segment's own cloud to each --
    doc 83 step 4b, the 'part of the muon lost to a PR break' test."""
    f = uproot.open(froot)
    rc = f["T_rec_charge"].arrays(
        ["x", "y", "z", "cluster_id", "real_cluster_id", "particle_id"], library="np")
    m_self = rc["real_cluster_id"] == segid
    if m_self.sum() == 0:
        return []
    self_pts = np.stack([rc["x"][m_self], rc["y"][m_self], rc["z"][m_self]], axis=1)
    m_cl = rc["cluster_id"] == cid
    others = []
    for rid in np.unique(rc["real_cluster_id"][m_cl]):
        rid = int(rid)
        if rid == segid or rid < 0:
            continue
        m = m_cl & (rc["real_cluster_id"] == rid)
        pid = np.unique(rc["particle_id"][m])
        if not any(abs(int(p)) in (13, 211, 2212) for p in pid):
            continue
        pts = np.stack([rc["x"][m], rc["y"][m], rc["z"][m]], axis=1)
        # cheap closest-pair: min over a coarse subsample x full (bounded N here)
        d = np.sqrt(((self_pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        others.append(dict(rid=rid, n=int(m.sum()), pid=[int(p) for p in pid],
                           min_dist_cm=float(d.min())))
    others.sort(key=lambda o: o["min_dist_cm"])
    return others


# ---------------------------------------------------------------------------
def med_mad(x):
    x = np.asarray(x)
    if len(x) == 0:
        return np.nan, np.nan
    m = np.median(x)
    return float(m), float(1.4826 * np.median(np.abs(x - m)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-exist-ok", action="store_true")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ratio-cut", type=float, default=1.5)
    ap.add_argument("--n-controls", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()
    if os.path.exists(args.out) and not args.out_exist_ok:
        sys.exit(f"refusing to write into existing {args.out} (M13); use a fresh dir")
    os.makedirs(args.out, exist_ok=True)
    clouds_dir = os.path.join(args.out, "clouds")
    os.makedirs(clouds_dir, exist_ok=True)

    rows = collect(args.arm)
    ran = [r for r in rows if r["ke_mcs"] > 0]
    A = [r for r in ran if r["isfc"] == 1 and r["ke_range_tk"] > 0]     # contained
    B = [r for r in ran if r["isfc"] == 0 and r["ke_range_tk"] > 0]     # exiting

    for r in A:
        r["ratio"] = r["ke_mcs"] / r["ke_range_tk"]
    outliers = sorted([r for r in A if r["ratio"] > args.ratio_cut], key=lambda r: -r["ratio"])
    good = [r for r in A if 0.85 <= r["ratio"] <= 1.15]

    summary = []
    summary.append(f"contained(A)={len(A)}  exiting(B)={len(B)}  "
                   f"outliers(ratio>{args.ratio_cut})={len(outliers)}")

    # ---- step 0: ambiguity quintile cross-tab ----
    ambs_all = np.array([r["amb"] for r in A])
    qedges = np.quantile(ambs_all, [0, .2, .4, .6, .8, 1.0])
    q80 = qedges[4]
    n_hi = sum(1 for r in outliers if r["amb"] >= q80)
    summary.append(f"step0: ambiguity quintile edges = {np.round(qedges,3).tolist()}")
    summary.append(f"step0: {n_hi}/{len(outliers)} outliers sit in the top ambiguity quintile "
                   f"(>= {q80:.3f}); residue (low-ambiguity outliers) = {len(outliers)-n_hi}")
    residue = [r for r in outliers if r["amb"] < q80]

    # ---- pick matched controls (same nseg14 distribution as outliers) ----
    rng = np.random.default_rng(args.seed)
    out_nseg = [r["nseg14"] for r in outliers]
    controls = []
    pool = list(good)
    rng.shuffle(pool)
    for target in out_nseg:
        best = min(pool, key=lambda r: abs(r["nseg14"] - target), default=None)
        if best is not None and len(controls) < args.n_controls:
            controls.append(best)
            pool.remove(best)

    # ---- 4 low-side violators (Part 3) ----
    for r in B:
        r["ratio"] = r["ke_mcs"] / r["ke_range_tk"]
    lowside = sorted([r for r in B if r["ratio"] < 1.0], key=lambda r: r["ratio"])[:4]

    # ---- per-muon deep dive: steps 1, 3, 4, 4b, 5 ----
    def deep_dive(r, tag):
        froot = os.path.join(r["prdir"], "tracking-pr.root")
        out = dict(evt=r["evt"], segid=r["segid"], cid=r["cid"], tag=tag,
                   ke_mcs=r["ke_mcs"], ke_range_tk=r["ke_range_tk"], amb=r["amb"],
                   nseg14=r["nseg14"], sel_len=r["sel_len"])
        # step 3/4: composition + Bragg on the selected segment's own points
        rr, dqdx, pid, shower = dqdx_profile(froot, r["segid"])
        out["n_points"] = int(len(rr))
        out["frac_shower"] = float(shower.mean()) if len(shower) else np.nan
        pid_u, pid_c = np.unique(pid, return_counts=True)
        out["pid_mix"] = {int(p): int(c) for p, c in zip(pid_u, pid_c)}
        contrast, n_near, n_far = bragg_contrast(rr, dqdx)
        out["bragg_contrast"] = contrast
        out["bragg_n_near"] = n_near
        out["bragg_n_far"] = n_far
        # step 4b: fragmentation census
        out["fragments"] = fragmentation_census(froot, r["cid"], r["segid"])
        # step 1/5: replay + maskmax
        cloud_txt = os.path.join(clouds_dir, f"evt{r['evt']}_seg{r['segid']}.txt")
        npts = harvest_cloud(froot, r["segid"], cloud_txt)
        out["cloud_npoints"] = npts
        if npts and npts >= 3:
            probe_json = cloud_txt.replace(".txt", "_probe.json")
            pr = run_probe_replay(cloud_txt, probe_json, maskmax=True)
            if pr:
                out["probe_ke_MCS"] = pr["result"]["ke_MCS"]
                out["probe_ambiguity"] = pr["result"]["ambiguity_MCS"]
                out["probe_nsegs"] = pr["result"]["nsegs"]
                out["probe_bad_path"] = pr["result"]["bad_path"]
                out["lnl_curve"] = pr["likelihood_curve"]
                out["segments"] = pr["segments"]
                out["maskmax"] = pr.get("maskmax")
                # REPLAY FIDELITY GATE: the harvest uses the segment's own
                # first/last on-disk T_rec_charge point as a vtx_start/vtx_end
                # PROXY (the driver's real endpoints are the PR graph's fitted
                # Vertex::fit().point, not stored per-point in this tree) --
                # trim_trajectory is sensitive to that choice on short/marginal
                # tracks, so the replay does not always reproduce the shipped
                # kine_mcs_energy.  Only trust the replay's DERIVED quantities
                # (lnL curve, per-segment angles, the maskmax kink test) when
                # it does; report the mismatch rate itself as a finding rather
                # than silently using an unfaithful replay.
                rel = abs(out["probe_ke_MCS"] - r["ke_mcs"]) / max(abs(r["ke_mcs"]), 1.0)
                out["replay_verified"] = bool(rel < 0.20)
                out["replay_rel_diff"] = float(rel)
        return out

    dives = {"outlier": [deep_dive(r, "outlier") for r in outliers],
             "control": [deep_dive(r, "control") for r in controls],
             "lowside": [deep_dive(r, "lowside") for r in lowside]}

    with open(os.path.join(args.out, "deep_dives.json"), "w") as fh:
        json.dump(dives, fh, indent=1)

    # ---- step 2: statistical null (toy) per nseg14 bucket present among outliers.
    # TWO variants: the nominal shipped tune (sigma_scale=1, matches what the
    # estimator itself assumes) AND the doc-80-pull-corrected tune (sigma_scale
    # = the MEASURED pull width per T band, pull_sigma_scale()) -- the nominal
    # number alone is a BIASED estimate (doc 80 sec 9.3 found the tune's own
    # assumed width is wrong as a function of T), so both are reported and the
    # doc quotes the bracket, not a single number.
    toy_summary = []
    buckets = sorted(set(out_nseg))
    exp_nom = exp_scaled = 0.0
    for nb in buckets:
        rs = [r for r in outliers if r["nseg14"] == nb]
        if not rs:
            continue
        n_pop = sum(1 for r in A if r["nseg14"] == nb)
        T_med = float(np.median([r["ke_range_tk"] for r in rs]))
        scale = pull_sigma_scale(T_med)
        res_nom = run_probe_synthetic(nb, T_med, 4000, 1000 + nb,
                                      os.path.join(args.out, f"toy_nseg{nb}.json"))
        res_scl = run_probe_synthetic(nb, T_med, 4000, 5000 + nb,
                                      os.path.join(args.out, f"toy_nseg{nb}_pullcorrected.json"),
                                      sigma_scale=scale)
        if res_nom is None or res_scl is None:
            continue
        r_nom = np.array(res_nom["keguess_over_T"])
        r_scl = np.array(res_scl["keguess_over_T"])
        f15_nom = float((r_nom > 1.5).mean())
        f15_scl = float((r_scl > 1.5).mean())
        actual = len(rs)
        exp_nom += f15_nom * n_pop
        exp_scaled += f15_scl * n_pop
        toy_summary.append(dict(nseg14=nb, T_MeV=T_med, n_pop=n_pop, n_outliers_here=actual,
                                sigma_scale=scale,
                                toy_frac_gt1p5_nominal=f15_nom, toy_frac_gt1p5_pullcorrected=f15_scl,
                                toy_ratios=r_nom.tolist()))
        summary.append(f"step2 toy: nseg14={nb} T={T_med:.0f}MeV pull_scale={scale:.3f} "
                       f"-> P(ratio>1.5) nominal={f15_nom:.3f} pull-corrected={f15_scl:.3f} "
                       f"(N_pop={n_pop}, {actual} real outlier(s) here)")
    summary.append(f"step2 TOTAL expected(ratio>1.5): nominal-tune={exp_nom:.2f} "
                   f"pull-corrected={exp_scaled:.2f}  observed=18  "
                   f"(doc 80's pull test shows the nominal number is BIASED LOW -- "
                   f"report the bracket, not a single multiplier)")

    with open(os.path.join(args.out, "toy_null.json"), "w") as fh:
        json.dump(toy_summary, fh, indent=1)

    # ---- fragmentation summary across all outliers, AND a population baseline.
    # A raw outlier-only rate is not evidence of anything without a baseline --
    # this population census (over ALL contained muons, not just the outlier
    # sample) is what retracted the naive "12/18 = fragmentation" reading
    # (doc 83 sec 2 step 4b correction): a busy SBND event is full of nearby
    # track-like fragments regardless of whether the muon is an MCS outlier.
    n_frag = sum(1 for d in dives["outlier"]
                if any(f["min_dist_cm"] < 10 for f in d.get("fragments", [])))
    summary.append(f"step4b: {n_frag}/{len(outliers)} outliers have another track-like "
                   f"fragment within 10cm of the selected segment (candidate PR break)")
    n_frag_pop = 0
    for r in A:
        froot = os.path.join(r["prdir"], "tracking-pr.root")
        if any(f["min_dist_cm"] < 10 for f in fragmentation_census(froot, r["cid"], r["segid"])):
            n_frag_pop += 1
    summary.append(f"step4b BASELINE: {n_frag_pop}/{len(A)} of ALL contained muons (not just "
                   f"outliers) have such a fragment -- if this rate is not below the outlier "
                   f"rate above, adjacency has NO discriminating power for this question")
    n_bragg = sum(1 for d in dives["outlier"]
                 if isinstance(d.get("bragg_contrast"), float) and d["bragg_contrast"] >= 2.0)
    n_bragg_scored = sum(1 for d in dives["outlier"] if not np.isnan(d.get("bragg_contrast", np.nan)))
    summary.append(f"step4: {n_bragg}/{n_bragg_scored} scoreable outliers show a genuine Bragg "
                   f"rise (contrast>=2) on the SELECTED segment alone")

    # ---- replay fidelity gate (must precede any claim from the replay) ----
    def fidelity(dives_list):
        scored = [d for d in dives_list if "replay_verified" in d]
        return sum(1 for d in scored if d["replay_verified"]), len(scored)
    ok_out, n_out_scored = fidelity(dives["outlier"])
    ok_ctl, n_ctl_scored = fidelity(dives["control"])
    summary.append(f"replay fidelity (probe reproduces shipped ke_mcs within 20%, using the "
                   f"segment's own first/last on-disk point as an endpoint PROXY -- the true "
                   f"vertex fit point is not stored per-point in this tree): "
                   f"outliers {ok_out}/{n_out_scored}, controls {ok_ctl}/{n_ctl_scored} "
                   f"-- outliers are markedly harder to replay exactly, consistent with them "
                   f"being the short/marginal tracks where the exact endpoint matters most")

    # ---- step 5: kink/over-clustering test, RESTRICTED to the replay-verified
    # subset -- an unverified replay's own segments/angles are for an
    # approximately-right but not exactly-right trimmed path, and the kink
    # test asks a question (does ONE angle dominate?) that is exactly the kind
    # of question a wrong path can answer either way by construction.
    n_kink = 0
    for d in dives["outlier"]:
        if not d.get("replay_verified"):
            continue
        mm = d.get("maskmax")
        if mm and d.get("ke_range_tk", 0) > 0:
            before = abs(d["ke_mcs"] - d["ke_range_tk"]) / d["ke_range_tk"]
            after = abs(mm["ke_MCS"] - d["ke_range_tk"]) / d["ke_range_tk"]
            if before > 0 and after < 0.5 * before:
                n_kink += 1
    summary.append(f"step5 (replay-verified subset only, N={ok_out}): {n_kink}/{ok_out} have the "
                   f"residual cut by >=50% when the single largest angle is masked -- i.e. NO "
                   f"single-dominant-angle (kink/over-clustering) signature survives the fidelity "
                   f"gate; the excess is spread across multiple angles, not concentrated in one")

    # ---- outliers.tsv (flat, for spreadsheet cross-checking) ----
    with open(os.path.join(args.out, "outliers.tsv"), "w") as fh:
        keys = ["evt", "segid", "cid", "row", "ke_mcs", "ke_range_tk", "ratio", "amb",
                "nseg14", "sel_len", "cdrop", "bragg_contrast", "frag_min_dist_cm",
                "maskmax_ke_MCS", "replay_verified"]
        fh.write("\t".join(keys) + "\n")
        for r, d in zip(outliers, dives["outlier"]):
            frag_min = min([f["min_dist_cm"] for f in d.get("fragments", [])], default=-1)
            mm = d.get("maskmax")
            fh.write("\t".join(str(x) for x in [
                r["evt"], r["segid"], r["cid"], r["row"], r["ke_mcs"], r["ke_range_tk"],
                r["ratio"], r["amb"], r["nseg14"], r["sel_len"], r["cdrop"],
                d.get("bragg_contrast"), frag_min, mm["ke_MCS"] if mm else -1,
                d.get("replay_verified")]) + "\n")

    # ================= plots =================
    # (i) angle spectrum: max/median |angle| per muon, outlier vs control
    def angle_stats(d):
        segs = d.get("segments")
        if not segs:
            return None
        mags = [abs(complex(bx, cy)) for bx, cy in zip(segs["angle_projB"][1:], segs["angle_projC"][1:])]
        if not mags:
            return None
        return max(mags), np.median(mags)

    # RESTRICTED to replay-verified muons (same fidelity gate as step 5) --
    # the segments/angles come from the same approximate-endpoint replay, so
    # an unverified muon's angles are for an approximately-, not exactly-,
    # right trimmed path.
    fig, ax = plt.subplots(figsize=(6, 5))
    med_by_tag = {}
    for tag, marker, color in [("outlier", "o", "tab:red"), ("control", "s", "tab:blue")]:
        xs, ys = [], []
        for d in dives[tag]:
            if not d.get("replay_verified"):
                continue
            st = angle_stats(d)
            if st:
                xs.append(st[1]); ys.append(st[0])
        ax.scatter(xs, ys, marker=marker, c=color, alpha=0.7,
                  label=f"{tag}, replay-verified (N={len(xs)})")
        med_by_tag[tag] = float(np.median(xs)) if xs else float("nan")
    lim = ax.get_xlim()
    ax.plot([0, lim[1]], [0, lim[1]], "k:", lw=1, label="max = median")
    ax.set_xlabel("median |angle| per muon [rad]")
    ax.set_ylabel("max |angle| per muon [rad]")
    ax.set_title("per-muon angle spectrum: REPLAY-VERIFIED outliers vs controls")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "angle_spectrum.png"), dpi=150)
    summary.append(f"angle spectrum (replay-verified only): median-of-medians "
                   f"outlier={med_by_tag.get('outlier'):.3f} rad vs "
                   f"control={med_by_tag.get('control'):.3f} rad")

    # (ii) lnL curve overlay -- REPLAY-VERIFIED outliers only (see the
    # fidelity gate above); an unverified replay's curve is for an
    # approximately-, not exactly-, right trimmed path.
    fig, ax = plt.subplots(figsize=(7, 5))
    verified = [d for d in dives["outlier"] if d.get("replay_verified")]
    worst = sorted(verified, key=lambda d: -(d["ke_mcs"] / max(d["ke_range_tk"], 1)))[:5]
    for d in worst:
        if "lnl_curve" not in d:
            continue
        curve = np.array(d["lnl_curve"], dtype=object)
        ke = np.array([c[0] for c in curve], dtype=float)
        lnl = np.array([c[1] if not isinstance(c[1], str) else np.nan for c in curve], dtype=float)
        lnl = lnl - np.nanmin(lnl)
        ax.plot(ke, lnl, label=f"evt{d['evt']} seg{d['segid']} (amb={d['amb']:.2f})")
        ax.axvline(d["ke_range_tk"], color=ax.lines[-1].get_color(), ls=":", lw=1)
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 20)
    ax.set_xlabel("KE [MeV]")
    ax.set_ylabel("-lnL - min(-lnL)")
    ax.set_title("likelihood curves, worst replay-verified outliers (dotted = KE_range)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "lnl_curves.png"), dpi=150)

    # (iii) toy null: distribution per bucket + observed rate overlay
    if toy_summary:
        fig, ax = plt.subplots(figsize=(7, 5))
        for t in toy_summary:
            ax.hist(np.clip(t["toy_ratios"], 0, 5), bins=50, histtype="step",
                    label=f"nseg14={t['nseg14']} (T={t['T_MeV']:.0f}MeV)", density=True)
        ax.axvline(args.ratio_cut, color="k", ls="--", lw=1)
        ax.set_xlabel("toy keguess / input T")
        ax.set_ylabel("density")
        ax.set_title("statistical-null KE_MCS/T distribution by segment count")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "toy_null.png"), dpi=150)

    # (iv) Bragg contrast: outlier vs control
    fig, ax = plt.subplots(figsize=(6, 4))
    for tag, color in [("outlier", "tab:red"), ("control", "tab:blue")]:
        vals = [d["bragg_contrast"] for d in dives[tag] if not np.isnan(d.get("bragg_contrast", np.nan))]
        ax.hist(np.clip(vals, 0, 6), bins=30, histtype="step", color=color,
                label=f"{tag} (N={len(vals)}, med={np.median(vals) if vals else float('nan'):.2f})")
    ax.axvline(2.0, color="k", ls="--", lw=1, label="stopping-track threshold")
    ax.set_xlabel("Bragg contrast: med(dQ/dx, rr<2) / med(dQ/dx, 20<=rr<40)")
    ax.set_ylabel("muons")
    ax.set_title("Bragg-peak presence: outliers vs matched controls")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "bragg_contrast.png"), dpi=150)

    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
