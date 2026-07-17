#!/usr/bin/env python
"""Census of LONG UNMATCHED clusters (Bee "non-match" long tracks) for a tag.

Companion to ql_agree_score.py: that scorer tells you HOW MANY long scan
positives an auto run missed; this tells you WHY each one is unmatched and
WHETHER a rescue-gate / precull change would recover it, without a rerun.

The target set is the scorer's per-event `missed_list` (long objective
positives with no auto counterpart within tol).  Each missed uid is placed in
one mechanism class, read straight from the calib dump `bundles[]`
(QLMatching::dump_calib, which holds only CONTAINED bundles, all metrics
valid):

  A anchored-elsewhere : uid appears in `other_clusters` of some auto_selected
      bundle near the truth time -> the track IS matched in Bee (it rides a
      group anchor); NOT a real non-match.  Excluded from the fix target.
  B no-bundle          : uid is never `main_cluster` of any dump bundle ->
      every candidate was containment-culled at every flash T0 (dump omits
      uncontained, QLMatching.cxx:3457) or the uid was never a group anchor.
      Only new C++ (deferred vis on uncontained) can reach these.
  C gate-fail          : uid HAS non-selected dump bundle(s) near the truth
      time -> rescue-reachable.  We replay the shared cluster-rescue accept()
      (ks<ks_max & chi2/ndf<c2ndf_max & ratio_lo<pred/meas<ratio_hi,
      QLMatching.cxx:2893-2901) and report which gate fails and the loosest
      candidate.  `potential_bad_match=false` == precull-pool member.
  D wrong-time-only    : uid has dump bundle(s) but none within tol of the
      truth time -> visibility / photon-model class, not a rescue-gate issue.

Offline what-if: for a set of candidate (ks_max,c2ndf_max,ratio_lo,ratio_hi)
gate sets over the PRECULL pool (contained, potential_bad_match=false), count
how many currently-missed long positives each gate set would rescue, and how
many adoptions land on truth-negative or no-verdict clusters (phantom / unknown
risk).  This predicts nm1/nm2* before the rerun; the rerun is ground truth.

Outputs work/ql_scores/<tag>/unmatched_census.{md,json}.  Read-only w.r.t.
every other record.

Repro:
  python ql_display/unmatched_census.py --tag nm0
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

# Reuse the frozen scorer's helpers so truth/length/geometry stay identical.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ql_agree_score import (  # noqa: E402
    GID_STRIDE, RUN, EVT0, EVT_STEP, NEVT, OBJECTIVE_TIERS,
    evt_of_idx, cluster_len_cm, load_truth, find_truth,
)

# Shared cluster-rescue gates currently ON in the runner (run_clus_evt.sh
# PDVD_QL_CRESCUE_* defaults) -- the baseline accept() bar.
BASE_GATES = dict(ks_max=0.25, c2ndf_max=15.0, ratio_lo=0.3, ratio_hi=3.0)

# Candidate gate sets for the what-if sweep over the PRECULL pool.  The census
# finding (nm0) overturned the naive "loosen to recover more" intuition: the
# rescue score() cannot tell a track's TRUE flash from a rival, so a loose pool
# adopts many known positives at the WRONG flash (dt >> tol).  The decisive
# metric is precision (recovered / all-truth-judged), not raw recall, so the
# sweep tracks TIGHTENING the gates from the current runner point.  ratio_lo
# stays > 0 (the precull zero-metric interlock, QLMatching.cxx:2919-2922).
SWEEP_GATES = {
    "base_.25/15/.3-3":  dict(ks_max=0.25, c2ndf_max=15.0, ratio_lo=0.3, ratio_hi=3.0),
    "t_.18/4/.5-2":      dict(ks_max=0.18, c2ndf_max=4.0,  ratio_lo=0.5, ratio_hi=2.0),
    "t_.15/3/.5-2":      dict(ks_max=0.15, c2ndf_max=3.0,  ratio_lo=0.5, ratio_hi=2.0),
    "t_.12/2/.6-1.7":    dict(ks_max=0.12, c2ndf_max=2.0,  ratio_lo=0.6, ratio_hi=1.7),
    "t_.10/2/.6-1.6":    dict(ks_max=0.10, c2ndf_max=2.0,  ratio_lo=0.6, ratio_hi=1.6),
}


def bundle_ratio(b):
    meas = b.get("total_PE", 0.0)
    return (b["total_pred_light"] / meas) if meas > 0.0 else 1e9


def bundle_c2ndf(b):
    return b["chi2"] / max(b.get("ndf", 1), 1)


def accepts(b, g):
    return (b["ks_dis"] < g["ks_max"]
            and bundle_c2ndf(b) < g["c2ndf_max"]
            and bundle_ratio(b) > g["ratio_lo"]
            and bundle_ratio(b) < g["ratio_hi"])


def rescue_score(b):
    # QLMatching.cxx:2904-2909 (lower = better).
    r = bundle_ratio(b)
    return (b["ks_dis"] * math.sqrt(max(bundle_c2ndf(b), 1.0))
            + abs(math.log(r if r > 0 else 1e-9)))


def gate_fail_reason(b, g):
    fails = []
    if not (b["ks_dis"] < g["ks_max"]):
        fails.append(f"ks {b['ks_dis']:.3f}>={g['ks_max']}")
    if not (bundle_c2ndf(b) < g["c2ndf_max"]):
        fails.append(f"c2ndf {bundle_c2ndf(b):.1f}>={g['c2ndf_max']}")
    r = bundle_ratio(b)
    if not (r > g["ratio_lo"]):
        fails.append(f"ratio {r:.2f}<={g['ratio_lo']}")
    if not (r < g["ratio_hi"]):
        fails.append(f"ratio {r:.2f}>={g['ratio_hi']}")
    return fails


def load_calib(work_root, tag, idx):
    evt = evt_of_idx(idx)
    path = os.path.join(work_root, f"{RUN}_{idx}_{tag}", f"calib-evt{evt}.json")
    with open(path) as fh:
        return json.load(fh), path


def census_event(calib, truth_entries, tol, min_len, min_npts):
    clusters = {c["uid"]: c for c in calib["clusters"]}
    flash_by_gid = {f["gid"]: f for f in calib["flashes"]}
    bundles = calib["bundles"]

    def is_long(uid):
        c = clusters.get(uid)
        return bool(c) and (cluster_len_cm(c) >= min_len or c["npoints"] >= min_npts)

    entries_by_uid = defaultdict(list)
    for e in truth_entries:
        entries_by_uid[e["uid"]].append(e)

    # main_cluster -> its dump bundles (all contained); + auto/anchor bookkeeping.
    by_main = defaultdict(list)
    for b in bundles:
        b = dict(b, _time=flash_by_gid[b["flash_gid"]]["time"])
        by_main[b["main_cluster"]].append(b)
    autos = [b for bl in by_main.values() for b in bl if b.get("auto_selected")]
    matched_main = {b["main_cluster"] for b in autos}
    # anchored-elsewhere: uid -> list of (auto anchor uid, flash time)
    anchored = defaultdict(list)
    for b in autos:
        for oc in b.get("other_clusters", ()):
            anchored[oc].append((b["main_cluster"], b["_time"]))

    # Truth-side missed long positives (mirror ql_agree_score.score_event).
    auto_times_by_uid = defaultdict(list)
    for b in autos:
        auto_times_by_uid[b["main_cluster"]].append(b["_time"])

    records = []
    for e in truth_entries:
        if not e["positive"] or e["conf"] not in OBJECTIVE_TIERS:
            continue
        uid = e["uid"]
        if uid not in clusters or not is_long(uid):
            continue
        covered = any(abs(t - e["time"]) <= tol for t in auto_times_by_uid.get(uid, ()))
        if covered:
            continue  # matched -- not a non-match
        c = clusters[uid]
        rec = dict(uid=uid, time=e["time"], conf=e["conf"],
                   len_cm=round(cluster_len_cm(c), 1), npoints=c["npoints"],
                   zc=round((max(c["z"]) + min(c["z"])) / 2, 1) if c.get("z") else None)

        # Class A: anchored under an auto bundle near the truth time.
        anc = [a for a in anchored.get(uid, ()) if abs(a[1] - e["time"]) <= tol]
        near = [b for b in by_main.get(uid, ()) if abs(b["_time"] - e["time"]) <= tol]
        if anc:
            rec.update(cls="A_anchored", detail=f"rides anchor uid {anc[0][0]}")
        elif not by_main.get(uid):
            rec.update(cls="B_nobundle",
                       detail="no contained bundle at any T0 (containment-culled)")
        elif not near:
            rec.update(cls="D_wrongtime",
                       detail=f"{len(by_main[uid])} bundle(s), none within {tol}us")
        else:
            # Class C: rescue-reachable.  Report the loosest precull candidate.
            precull = [b for b in near if not b.get("potential_bad_match")]
            pool = precull or near
            best = min(pool, key=rescue_score)
            fails_base = gate_fail_reason(best, BASE_GATES)
            rec.update(cls="C_gatefail",
                       precull_ok=bool(precull),
                       best_ks=round(best["ks_dis"], 3),
                       best_c2ndf=round(bundle_c2ndf(best), 1),
                       best_ratio=round(bundle_ratio(best), 2),
                       fails_base="; ".join(fails_base) or "PASSES base (already?)",
                       detail=f"{len(near)} near bundle(s), {len(precull)} precull")
        records.append(rec)

    return records, clusters, by_main, matched_main, entries_by_uid


def whatif_event(clusters, by_main, matched_main, entries_by_uid, tol,
                 min_len, min_npts, gates):
    """Simulate precull rescue with `gates` on top of the current matched set.
    Returns 4 lists for LONG clusters:
      recovered  = adopted at a truth-positive flash time (a genuine win)
      phantom    = adopted at a truth-negative flash time (a real phantom)
      wrongflash = cluster IS a truth positive but adopted at a DIFFERENT flash
                   (dt > tol) -- provably wrong, invisible to the frozen scorer
      unlabeled  = cluster has no scan verdict at the adopted time (rescan-only)
    """
    def is_long(uid):
        c = clusters.get(uid)
        return bool(c) and (cluster_len_cm(c) >= min_len or c["npoints"] >= min_npts)

    pos_times = defaultdict(list)
    for uid, es in entries_by_uid.items():
        for e in es:
            if e["positive"] and e["conf"] in OBJECTIVE_TIERS:
                pos_times[uid].append(e["time"])

    recovered, phantom, wrongflash, unlabeled = [], [], [], []
    for uid, bl in by_main.items():
        if uid in matched_main or not is_long(uid):
            continue
        pool = [b for b in bl if not b.get("potential_bad_match")]
        cand = [b for b in pool if accepts(b, gates)]
        if not cand:
            continue
        best = min(cand, key=rescue_score)
        item = (uid, round(best["_time"], 1))
        e = find_truth(entries_by_uid, uid, best["_time"], tol)
        if e is None:
            # No verdict at the adopted time.  If the cluster IS a known
            # positive at another flash, this adoption is provably wrong.
            wrongflash.append(item) if uid in pos_times else unlabeled.append(item)
        elif e["positive"] and e["conf"] in OBJECTIVE_TIERS:
            recovered.append(item)
        elif not e["positive"] and e["conf"] in OBJECTIVE_TIERS:
            phantom.append(item)
        else:
            unlabeled.append(item)
    return recovered, phantom, wrongflash, unlabeled


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--work-root", default="work")
    ap.add_argument("--gold", default="work/ql_labels/wfresc/labels-evt298567.json")
    ap.add_argument("--decisions-dir", default="ql_display/decisions-cathxa")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--min-len-cm", type=float, default=25.0)
    ap.add_argument("--min-npoints", type=int, default=100)
    ap.add_argument("--out-root", default="work/ql_scores")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    truth = load_truth(args.gold, args.decisions_dir)
    out_dir = os.path.join(args.out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "unmatched_census.json")) and not args.force:
        sys.exit(f"refusing to overwrite {out_dir}/unmatched_census.json (M13); "
                 "use --force or a new tag")

    all_records = {}
    cls_tot = defaultdict(int)
    whatif_tot = {name: dict(recovered=0, phantom=0, wrongflash=0, unlabeled=0)
                  for name in SWEEP_GATES}
    for idx in range(NEVT):
        evt = evt_of_idx(idx)
        try:
            calib, _ = load_calib(args.work_root, args.tag, idx)
        except FileNotFoundError:
            print(f"WARN: no calib for idx {idx} (evt {evt})", file=sys.stderr)
            continue
        recs, clusters, by_main, matched, entries_by_uid = census_event(
            calib, truth[evt], args.tol, args.min_len_cm, args.min_npoints)
        all_records[evt] = recs
        for r in recs:
            cls_tot[r["cls"]] += 1
        for name, g in SWEEP_GATES.items():
            rec, ph, wf, un = whatif_event(clusters, by_main, matched, entries_by_uid,
                                           args.tol, args.min_len_cm, args.min_npoints, g)
            whatif_tot[name]["recovered"] += len(rec)
            whatif_tot[name]["phantom"] += len(ph)
            whatif_tot[name]["wrongflash"] += len(wf)
            whatif_tot[name]["unlabeled"] += len(un)

    total_missed = sum(len(v) for v in all_records.values())
    real_nonmatch = total_missed - cls_tot.get("A_anchored", 0)

    # ---- JSON ----
    with open(os.path.join(out_dir, "unmatched_census.json"), "w") as fh:
        json.dump(dict(tag=args.tag, total_missed=total_missed,
                       real_nonmatch=real_nonmatch, class_totals=dict(cls_tot),
                       whatif=whatif_tot, per_event=all_records), fh, indent=1)

    # ---- Markdown ----
    lines = [f"# Long unmatched-cluster census — tag `{args.tag}`", "",
             f"Target = scorer `missed` (long objective positives, "
             f">= {args.min_len_cm:g} cm or >= {args.min_npoints} pts, tol "
             f"{args.tol:g} us).  Total missed **{total_missed}**; excluding "
             f"class-A anchored-elsewhere, real Bee non-matches "
             f"**{real_nonmatch}**.", "",
             "## Mechanism classes", "",
             "| class | count | meaning |", "|---|---|---|",
             f"| A_anchored | {cls_tot.get('A_anchored',0)} | rides a matched "
             "group anchor — NOT a Bee non-match |",
             f"| B_nobundle | {cls_tot.get('B_nobundle',0)} | containment-culled "
             "at every T0 — needs new C++ |",
             f"| C_gatefail | {cls_tot.get('C_gatefail',0)} | rescue-reachable "
             "candidate exists — gate/precull lever |",
             f"| D_wrongtime | {cls_tot.get('D_wrongtime',0)} | bundles exist "
             "but none near truth time — photon-model |", "",
             "## Offline what-if (precull pool, additive over current matched)",
             "",
             "recovered = adopted at a truth-positive time (real win); phantom "
             "= truth-negative time; **wrongflash** = known positive adopted at "
             "the WRONG flash (dt>tol; provably wrong, the frozen scorer counts "
             "it only as unknown); unlabeled = no verdict (rescan-only). "
             "precision = recovered / (recovered+phantom+wrongflash).", "",
             "| gate set | recovered | phantom | wrongflash | unlabeled | precision |",
             "|---|---|---|---|---|---|"]
    for name in SWEEP_GATES:
        w = whatif_tot[name]
        known = w["recovered"] + w["phantom"] + w["wrongflash"]
        prec = f"{w['recovered']/known:.2f}" if known else "n/a"
        lines.append(f"| {name} | {w['recovered']} | {w['phantom']} | "
                     f"{w['wrongflash']} | {w['unlabeled']} | {prec} |")
    lines += ["", "## Per-event missed long clusters", ""]
    for evt in sorted(all_records):
        recs = all_records[evt]
        if not recs:
            continue
        lines.append(f"### evt {evt} ({len(recs)} missed)")
        lines.append("| uid | len_cm | npts | t_us | conf | class | detail |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in sorted(recs, key=lambda x: -x["len_cm"]):
            lines.append(f"| {r['uid']} | {r['len_cm']} | {r['npoints']} | "
                         f"{r['time']:.1f} | {r['conf']} | {r['cls']} | "
                         f"{r.get('detail','')} |")
        lines.append("")
    with open(os.path.join(out_dir, "unmatched_census.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[census] tag {args.tag}: {total_missed} missed "
          f"({real_nonmatch} real non-match); classes "
          + ", ".join(f"{k}={v}" for k, v in sorted(cls_tot.items())))
    print("[what-if] " + "; ".join(
        f"{n}: +{whatif_tot[n]['recovered']} (ph {whatif_tot[n]['phantom']}, "
        f"wf {whatif_tot[n]['wrongflash']}, unl {whatif_tot[n]['unlabeled']})"
        for n in SWEEP_GATES))
    print(f"wrote {out_dir}/unmatched_census.{{md,json}}")


if __name__ == "__main__":
    main()
