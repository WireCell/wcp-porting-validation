#!/usr/bin/env python
"""Explain each long non-matched cluster of a tag against the scan record.

Companion to unmatched_census.py: the census places every scorer-missed long
positive in a mechanism class (A/B/C/D); this script joins each one with the
HAND/AI SCAN verdict that says where the cluster belongs (gold labels carry
the flash; AI decision lines carry flash + free-text reason) and with what the
matcher ACTUALLY did in this tag's dumps, then assigns a refined failure stage:

  WRONG_FLASH      cluster IS auto-matched, at a different flash (Bee shows it
                   matched; the scorer counts it missed).  Sub-noted whether
                   the truth-time candidate would pass the rescue gates --
                   rescue never runs on matched clusters.
  ANCHORED_ELSEW   not auto itself but rides another cluster's auto bundle
                   (other_clusters) at a non-truth time.
  CULLED_BADMATCH  every truth-time candidate carries potential_bad_match=true
                   (culled by cull_inconsistent; outside every rescue pool).
  PASSES_UNADOPTED cluster is fully unmatched, its truth-time candidate passes
                   the relaxed rescue gates, yet no tier adopted it -- the
                   rescue's best-score pick for this cluster sits at another
                   flash (per-cluster ranking loss) or the tier skipped it.
  GATE_NEAR_MISS   best truth-time candidate fails the relaxed gates by < 20%
                   relative on its closest gate.
  GATE_FAR_FAIL    fails by >= 20% -- light metrics genuinely dislike the pair.
  WRONGTIME_*      class D split: _FLASH_CUT (no admitted flash within tol at
                   all) vs _NO_BUNDLE (flash admitted but this cluster has no
                   contained bundle there).

Also censuses the Bee "non-match" population itself: every LONG cluster with
no auto bundle and no anchor ride, split by scan verdict (objective positive /
low-conf-only / scanner rejected all / no verdict) -- answering "did the scans
match most of the non-matched long tracks?".

Gate sets replayed (runner defaults after nm3+nm4b adoption, run_clus_evt.sh):
  tight   ks<0.15  chi2/ndf<3   0.5<ratio<2.0   (precull additive tier, nm3)
  relaxed ks<0.25  chi2/ndf<15  0.3<ratio<3.0   (second-chance tier, nm4b,
                                                 length >= 50 cm only)

Scan gid caveat: AI decision lines were recorded on the cathxa dumps; flash
gids renumber between tags, so joins here use flash TIME (bit-stable across
matching-only reprocesses) + uid, never gid (see remap_scan_state.py).

Outputs work/ql_scores/<tag>/nonmatch_explain.{md,json} -- NEW files beside
the tag's census; refuses to overwrite (M13).  Read-only otherwise.

Repro:
  python ql_display/nonmatch_explain.py --tag nm4b
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ql_agree_score import (  # noqa: E402
    GID_STRIDE, RUN, EVT0, EVT_STEP, NEVT, OBJECTIVE_TIERS,
    evt_of_idx, cluster_len_cm, load_truth,
)
from unmatched_census import (  # noqa: E402
    bundle_ratio, bundle_c2ndf, rescue_score, gate_fail_reason,
    load_calib, census_event,
)

TIGHT_GATES = dict(ks_max=0.15, c2ndf_max=3.0, ratio_lo=0.5, ratio_hi=2.0)
RELAX_GATES = dict(ks_max=0.25, c2ndf_max=15.0, ratio_lo=0.3, ratio_hi=3.0)
RELAX_MIN_LEN_CM = 50.0     # nm4b relaxed tier length gate
NEAR_MISS_REL = 0.20        # closest-gate relative excess below this = near miss
RIVAL_WINDOW_US = 5.0


def gate_margin(b, g):
    """Smallest relative excess over the failed gate(s); 0 if it passes."""
    r = bundle_ratio(b)
    excess = []
    if not (b["ks_dis"] < g["ks_max"]):
        excess.append((b["ks_dis"] - g["ks_max"]) / g["ks_max"])
    c2 = bundle_c2ndf(b)
    if not (c2 < g["c2ndf_max"]):
        excess.append((c2 - g["c2ndf_max"]) / g["c2ndf_max"])
    if not (r > g["ratio_lo"]):
        excess.append((g["ratio_lo"] - r) / g["ratio_lo"])
    if not (r < g["ratio_hi"]):
        excess.append((r - g["ratio_hi"]) / g["ratio_hi"])
    return min(excess) if excess else 0.0


def passes(b, g):
    return gate_margin(b, g) == 0.0


def load_scan_reasons(gold_path, decisions_dir):
    """(evt, uid) -> list of positive scan lines with time/reason/conf/verdict."""
    scan = defaultdict(list)
    with open(gold_path) as fh:
        lab = json.load(fh)
    evt = int(lab["event"].replace("evt", ""))
    for e in lab["matches"]:
        uid = e["apa"] * GID_STRIDE + e["cluster_idents"][0]
        scan[(evt, uid)].append(dict(
            time=e["flash_time_us"], verdict="gold-match", conf="gold",
            reason="(owner gold scan, no reason text recorded)"))
    for path in sorted(glob.glob(os.path.join(decisions_dir,
                                              "decisions-evt*.jsonl"))):
        with open(path) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                d = json.loads(ln)
                if d["verdict"] not in ("keep", "add"):
                    continue
                devt = int(d["event"].replace("evt", ""))
                scan[(devt, d["main_cluster_uid"])].append(dict(
                    time=d["flash_time_us"], verdict=d["verdict"],
                    conf=d.get("confidence", "med"),
                    reason=d.get("reason", "")))
    return scan


def flash_info(calib, time, tol):
    """Admitted flash nearest `time` within tol, with PE/sat/cov summary."""
    best, best_dt = None, tol
    for f in calib["flashes"]:
        dt = abs(f["time"] - time)
        if dt <= best_dt:
            best, best_dt = f, dt
    if best is None:
        return None
    return dict(gid=best["gid"], time=best["time"],
                total_PE=round(best.get("total_PE", 0.0), 1),
                nsat=sum(1 for s in best.get("sat", []) if s > 0),
                nuncov=sum(1 for c in best.get("cov", []) if c < 1.0))


def explain_event(calib, truth_entries, scan, evt, tol, min_len, min_npts):
    recs, clusters, by_main, matched_main, entries_by_uid = census_event(
        calib, truth_entries, tol, min_len, min_npts)
    flash_by_gid = {f["gid"]: f for f in calib["flashes"]}

    autos = [b for bl in by_main.values() for b in bl if b.get("auto_selected")]
    anchored_any = defaultdict(list)   # uid -> (anchor uid, time)
    for b in autos:
        for oc in b.get("other_clusters", ()):
            anchored_any[oc].append((b["main_cluster"], b["_time"]))

    out = []
    for rec in recs:
        uid, t = rec["uid"], rec["time"]
        near = [b for b in by_main.get(uid, ()) if abs(b["_time"] - t) <= tol]
        precull = [b for b in near if not b.get("potential_bad_match")]
        best = min(precull or near, key=rescue_score) if near else None

        r = dict(rec)   # census fields: uid,time,conf,len_cm,npoints,zc,cls,...
        # ---- scan side ----
        cand = sorted(scan.get((evt, uid), ()),
                      key=lambda s: abs(s["time"] - t))
        s = cand[0] if cand and abs(cand[0]["time"] - t) <= tol else None
        r["scan"] = s or dict(verdict="?", conf=rec["conf"],
                              reason="(no positive scan line joined -- "
                                     "check truth assembly)")
        # ---- truth flash as admitted in THIS run ----
        r["truth_flash"] = flash_info(calib, t, tol)
        r["rival_flashes_5us"] = sum(
            1 for f in calib["flashes"]
            if tol < abs(f["time"] - t) <= RIVAL_WINDOW_US)

        # ---- matcher outcome ----
        my_autos = [b for b in by_main.get(uid, ()) if b.get("auto_selected")]
        anc = anchored_any.get(uid, ())
        if my_autos:
            wb = min(my_autos, key=lambda b: abs(b["_time"] - t))
            wf = flash_by_gid[wb["flash_gid"]]
            r["outcome"] = "wrong_flash"
            r["wrong"] = dict(
                time=round(wb["_time"], 1), dt_us=round(wb["_time"] - t, 1),
                flash_PE=round(wf.get("total_PE", 0.0), 1),
                ks=round(wb["ks_dis"], 3),
                c2ndf=round(bundle_c2ndf(wb), 1),
                ratio=round(bundle_ratio(wb), 2),
                strength=round(wb.get("strength", 0.0), 3),
                relaxed_adopt=bool(wb.get("cluster_rescue_relaxed")))
        elif anc:
            a = min(anc, key=lambda x: abs(x[1] - t))
            r["outcome"] = "anchored_wrong_time"
            r["wrong"] = dict(anchor_uid=a[0], time=round(a[1], 1),
                              dt_us=round(a[1] - t, 1))
        else:
            r["outcome"] = "unmatched"   # the true Bee non-match population

        # ---- best truth-time candidate ----
        if best is not None:
            r["cand"] = dict(
                ks=round(best["ks_dis"], 3),
                c2ndf=round(bundle_c2ndf(best), 1),
                ratio=round(bundle_ratio(best), 2),
                strength=round(best.get("strength", 0.0), 3),
                in_precull=not best.get("potential_bad_match"),
                n_near=len(near), n_precull=len(precull),
                fails_tight="; ".join(gate_fail_reason(best, TIGHT_GATES))
                            or "passes",
                fails_relaxed="; ".join(gate_fail_reason(best, RELAX_GATES))
                              or "passes",
                margin_relaxed=round(gate_margin(best, RELAX_GATES), 2),
                flags=[k for k in ("at_cathode", "at_x_boundary",
                                   "close_to_PMT", "two_boundary",
                                   "window_truncated", "xtpc_pin",
                                   "xtpc_scenario1", "xtpc_consistent",
                                   "xtpc_cathode_rescued")
                       if best.get(k)])
            # where would the per-cluster rescue ranking actually land?
            pool = [b for b in by_main.get(uid, ())
                    if not b.get("potential_bad_match")]
            if pool:
                gbest = min(pool, key=rescue_score)
                r["cand"]["cluster_best_time"] = round(gbest["_time"], 1)
                r["cand"]["cluster_best_is_truth"] = (
                    abs(gbest["_time"] - t) <= tol)

        # ---- refined stage ----
        if r["outcome"] == "wrong_flash":
            stage = "WRONG_FLASH"
        elif r["outcome"] == "anchored_wrong_time":
            stage = "ANCHORED_ELSEW"
        elif rec["cls"] == "D_wrongtime":
            stage = ("WRONGTIME_NO_BUNDLE" if r["truth_flash"]
                     else "WRONGTIME_FLASH_CUT")
        elif rec["cls"] == "B_nobundle":
            stage = "CONTAINMENT"
        elif near and not precull:
            stage = "CULLED_BADMATCH"
        elif best is not None and passes(best, RELAX_GATES):
            stage = "PASSES_UNADOPTED"
        elif best is not None and gate_margin(best, RELAX_GATES) < NEAR_MISS_REL:
            stage = "GATE_NEAR_MISS"
        else:
            stage = "GATE_FAR_FAIL"
        # relaxed tier is length-gated; flag short-for-relaxed cases
        if stage in ("GATE_NEAR_MISS", "GATE_FAR_FAIL", "PASSES_UNADOPTED") \
                and rec["len_cm"] < RELAX_MIN_LEN_CM:
            r["below_relax_minlen"] = True
        r["stage"] = stage
        out.append(r)

    # ---- Bee non-match population (matcher-side, scan-verdict split) ----
    bee_matched = set(matched_main) | set(anchored_any)
    pop = dict(n_long=0, scan_pos=0, scan_low_only=0, scan_neg_only=0,
               scan_none=0, members=[])
    for uid, c in clusters.items():
        if uid in bee_matched:
            continue
        if not (cluster_len_cm(c) >= min_len or c["npoints"] >= min_npts):
            continue
        pop["n_long"] += 1
        es = entries_by_uid.get(uid, ())
        if any(e["positive"] and e["conf"] in OBJECTIVE_TIERS for e in es):
            cat = "scan_pos"
        elif any(e["positive"] for e in es):
            cat = "scan_low_only"
        elif es:
            cat = "scan_neg_only"
        else:
            cat = "scan_none"
        pop[cat] += 1
        pop["members"].append(dict(uid=uid,
                                   len_cm=round(cluster_len_cm(c), 1),
                                   npoints=c["npoints"], cat=cat))
    return out, pop


def short(txt, n=90):
    txt = (txt or "").replace("|", "/").replace("\n", " ")
    return txt if len(txt) <= n else txt[:n - 1] + "…"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--work-root", default="work")
    ap.add_argument("--gold",
                    default="work/ql_labels/wfresc/labels-evt298567.json")
    ap.add_argument("--decisions-dir", default="ql_display/decisions-cathxa")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--min-len-cm", type=float, default=25.0)
    ap.add_argument("--min-npoints", type=int, default=100)
    ap.add_argument("--out-root", default="work/ql_scores")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.tag)
    out_json = os.path.join(out_dir, "nonmatch_explain.json")
    if os.path.exists(out_json) and not args.force:
        sys.exit(f"refusing to overwrite {out_json} (M13); use --force")
    os.makedirs(out_dir, exist_ok=True)

    truth = load_truth(args.gold, args.decisions_dir)
    scan = load_scan_reasons(args.gold, args.decisions_dir)

    per_event, pops = {}, {}
    stage_tot, outcome_tot = defaultdict(int), defaultdict(int)
    unjoined = 0
    for idx in range(NEVT):
        evt = evt_of_idx(idx)
        try:
            calib, _ = load_calib(args.work_root, args.tag, idx)
        except FileNotFoundError:
            print(f"WARN: no calib for idx {idx} (evt {evt})", file=sys.stderr)
            continue
        recs, pop = explain_event(calib, truth[evt], scan, evt, args.tol,
                                  args.min_len_cm, args.min_npoints)
        per_event[evt], pops[evt] = recs, pop
        for r in recs:
            stage_tot[r["stage"]] += 1
            outcome_tot[r["outcome"]] += 1
            if r["scan"]["verdict"] == "?":
                unjoined += 1

    total = sum(len(v) for v in per_event.values())

    # ---- sanity gates vs the tag's census ----
    census_path = os.path.join(out_dir, "unmatched_census.json")
    checks = [f"total missed here {total}"]
    if os.path.exists(census_path):
        with open(census_path) as fh:
            cj = json.load(fh)
        ok = (cj["total_missed"] == total)
        checks.append(f"census total {cj['total_missed']} "
                      f"{'== PASS' if ok else '!= FAIL'}")
        if not ok:
            print("FATAL: totals disagree with unmatched_census.json",
                  file=sys.stderr)
    checks.append(f"outcome split {dict(outcome_tot)} "
                  f"(sums to {sum(outcome_tot.values())})")
    checks.append(f"unjoined scan lines {unjoined} (expect 0)")

    with open(out_json, "w") as fh:
        json.dump(dict(tag=args.tag, tol=args.tol,
                       tight_gates=TIGHT_GATES, relax_gates=RELAX_GATES,
                       relax_min_len_cm=RELAX_MIN_LEN_CM,
                       total_missed=total, stage_totals=dict(stage_tot),
                       outcome_totals=dict(outcome_tot),
                       sanity=checks, per_event=per_event,
                       bee_population=pops), fh, indent=1)

    # ---- markdown ----
    L = [f"# Non-match explanations — tag `{args.tag}` vs scan record", "",
         f"Missed long objective positives: **{total}**.  "
         f"Sanity: {'; '.join(checks)}.", "",
         "## Where they actually are (matcher outcome)", "",
         "| outcome | n | meaning |", "|---|---|---|",
         f"| unmatched | {outcome_tot.get('unmatched', 0)} | true Bee "
         "non-match: no auto bundle, no anchor ride |",
         f"| wrong_flash | {outcome_tot.get('wrong_flash', 0)} | auto-matched "
         "at a different flash (matched in Bee, wrong T0) |",
         f"| anchored_wrong_time | {outcome_tot.get('anchored_wrong_time', 0)}"
         " | rides an anchor at a non-truth time |", "",
         "## Failure stages", "", "| stage | n |", "|---|---|"]
    for k in sorted(stage_tot, key=lambda k: -stage_tot[k]):
        L.append(f"| {k} | {stage_tot[k]} |")

    L += ["", "## Bee non-match population vs scans (all long unmatched "
          "clusters, matcher side)", "",
          "| evt | long unmatched | scan matched it | low-conf only | "
          "scanner rejected all | no verdict |", "|---|---|---|---|---|---|"]
    tp = defaultdict(int)
    for evt in sorted(pops):
        p = pops[evt]
        L.append(f"| {evt} | {p['n_long']} | {p['scan_pos']} | "
                 f"{p['scan_low_only']} | {p['scan_neg_only']} | "
                 f"{p['scan_none']} |")
        for k in ("n_long", "scan_pos", "scan_low_only", "scan_neg_only",
                  "scan_none"):
            tp[k] += p[k]
    L.append(f"| **all** | {tp['n_long']} | {tp['scan_pos']} | "
             f"{tp['scan_low_only']} | {tp['scan_neg_only']} | "
             f"{tp['scan_none']} |")

    L += ["", "## Per-cluster explanations", ""]
    for evt in sorted(per_event):
        recs = per_event[evt]
        if not recs:
            continue
        L.append(f"### evt {evt} ({len(recs)} missed)")
        L.append("| uid | len | conf | truth t_us | flashPE | rivals±5us | "
                 "outcome | stage | best cand (ks/c2n/ratio/str) | "
                 "fails relaxed | scan says |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(recs, key=lambda x: -x["len_cm"]):
            tf = r.get("truth_flash")
            cd = r.get("cand")
            wr = r.get("wrong")
            outc = r["outcome"]
            if wr and outc == "wrong_flash":
                outc = f"wrong_flash dt={wr['dt_us']}us PE={wr['flash_PE']}"
            elif wr:
                outc = f"anchored uid{wr['anchor_uid']} dt={wr['dt_us']}us"
            L.append(
                f"| {r['uid']} | {r['len_cm']} | {r['conf']} | "
                f"{r['time']:.1f} | {tf['total_PE'] if tf else 'CUT'} | "
                f"{r['rival_flashes_5us']} | {outc} | {r['stage']} | "
                + (f"{cd['ks']}/{cd['c2ndf']}/{cd['ratio']}/{cd['strength']}"
                   if cd else "-") + " | "
                + (cd["fails_relaxed"] if cd else "-") + " | "
                + f"{short(r['scan'].get('reason'))} |")
        L.append("")
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "nonmatch_explain.md"), "w") as fh:
        fh.write(text)

    print(f"[explain] tag {args.tag}: {total} missed; stages "
          + ", ".join(f"{k}={v}" for k, v in sorted(stage_tot.items())))
    print("[sanity] " + "; ".join(checks))
    print(f"wrote {out_dir}/nonmatch_explain.{{md,json}}")


if __name__ == "__main__":
    main()
