#!/usr/bin/env python
"""Score automatic QLMatch output against the frozen 039252 scan reference.

Reference truth (fixed for the whole tuning campaign, doc 19):
  - gold hand scan  work/ql_labels/wfresc/labels-evt298567.json
      "matches" (incl. human-added) => positive, "rejected_auto" => negative;
      confidence tier "gold".
  - AI scan         ql_display/decisions-cathxa/decisions-evt*.jsonl (17 evts)
      verdict keep/add => positive, reject => negative; per-line confidence.

Objective tiers: gold + high + med.  Low-confidence verdicts are excluded from
the objective (user decision 2026-07-17) but reported informationally.

Join key is (event, flash TIME within --tol us, main-cluster uid) -- NEVER
flash_gid, which renumbers whenever flash admission changes (see
remap_scan_state.py).  uid = apa*1e6 + ident, stable across matching-only
reprocesses of the same clustering (_keep tarballs).

Long-track filter: main cluster bbox diagonal >= --min-len-cm (25) OR
npoints >= --min-npoints (100), computed from the SCORED run's calib clusters.

Metrics per event and overall (objective tiers, long tracks only):
  agree    = judged autos the scanner kept        (kept-auto agreement)
  phantom  = judged autos the scanner rejected
  unknown  = long autos with no scan verdict (new bundles; informational)
  missed   = scan positives with no auto counterpart in this run
             (subclassified flash-cut when no flash survives near the time)
Flag splits (xtpc_pin, two_boundary, ...) are tallied over judged autos.

Outputs to work/ql_scores/<tag>/scores.{json,md}.  Refuses to overwrite an
existing score dir without --force (M13: records are append-only; new run =>
new tag).

Repro (baseline):
  python ql_display/ql_agree_score.py --tag cathxa --self-check
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

GID_STRIDE = 1000000  # uid = apa*stride + ident (QLMatching kFlashGidStride)
RUN = "039252"
EVT0, EVT_STEP, NEVT = 298567, 14, 18
OBJECTIVE_TIERS = ("gold", "high", "med")
FLAG_KEYS = [
    "xtpc_pin", "xtpc_scenario1", "xtpc_cathode_rescued", "xtpc_consistent",
    "consistent", "two_boundary", "at_cathode", "at_x_boundary",
    "close_to_PMT", "window_truncated",
]


def evt_of_idx(idx):
    return EVT0 + EVT_STEP * idx


def cluster_len_cm(c):
    if not c.get("x"):
        return 0.0
    dx = max(c["x"]) - min(c["x"])
    dy = max(c["y"]) - min(c["y"])
    dz = max(c["z"]) - min(c["z"])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def load_truth(gold_path, decisions_dir):
    """truth[evt] = list of dicts {time, uid, positive, conf}."""
    truth = defaultdict(list)
    with open(gold_path) as fh:
        lab = json.load(fh)
    evt = int(lab["event"].replace("evt", ""))
    for e in lab["matches"]:
        uid = e["apa"] * GID_STRIDE + e["cluster_idents"][0]
        truth[evt].append(dict(time=e["flash_time_us"], uid=uid,
                               positive=True, conf="gold"))
    for e in lab["rejected_auto"]:
        uid = e["apa"] * GID_STRIDE + e["cluster_idents"][0]
        truth[evt].append(dict(time=e["flash_time_us"], uid=uid,
                               positive=False, conf="gold"))
    for path in sorted(glob.glob(os.path.join(decisions_dir, "decisions-evt*.jsonl"))):
        with open(path) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                d = json.loads(ln)
                devt = int(d["event"].replace("evt", ""))
                truth[devt].append(dict(
                    time=d["flash_time_us"], uid=d["main_cluster_uid"],
                    positive=(d["verdict"] in ("keep", "add")),
                    conf=d.get("confidence", "med")))
    return truth


def apply_time_map(truth, map_path, tol):
    """Translate truth flash times across a tail-merge reprocess (doc 26).

    map_path: tmerge_time_map.py output {events: {evt: [[t_old, t_new], ...]}}.
    A truth entry recorded at a merged-away LATE member's time is moved to the
    surviving seed's time.  If the same (uid, seed time) then carries both a
    positive and a negative verdict (scanner kept the late member and rejected
    the fast one, or vice versa), the positive wins -- post-merge there is one
    flash and the scanner did endorse this cluster-flash pairing; the dropped
    negative is reported, never silent.  No-op when map_path is None.
    """
    if not map_path:
        return truth
    with open(map_path) as fh:
        m = json.load(fh)
    n_moved = 0
    for evt, entries in truth.items():
        pairs = m["events"].get(str(evt), [])
        if not pairs:
            continue
        for e in entries:
            for t_old, t_new in pairs:
                if abs(e["time"] - t_old) <= tol:
                    e["time"] = t_new
                    n_moved += 1
                    break
        # positive-wins collision resolution at translated times
        drop = []
        for i, e in enumerate(entries):
            if e["positive"]:
                continue
            for p in entries:
                if (p["positive"] and p["uid"] == e["uid"]
                        and abs(p["time"] - e["time"]) <= tol):
                    drop.append(i)
                    print(f"[time-map] evt{evt}: dropping negative verdict "
                          f"uid={e['uid']} t={e['time']:.2f} colliding with a "
                          f"positive after merge translation")
                    break
        for i in reversed(drop):
            entries.pop(i)
    print(f"[time-map] translated {n_moved} truth times using {map_path}")
    return truth


CELL_CM = 3.0  # y/z occupancy cell for the geometric uid map


def _cells(c):
    """Set of occupied (y,z) grid cells of a calib cluster (v-independent)."""
    return {(int(y // CELL_CM), int(z // CELL_CM))
            for y, z in zip(c["y"], c["z"])}


def build_uid_map(old_calib, new_calib, min_frac=0.5):
    """old cluster uid -> new cluster uid by y/z cell overlap (doc 26 step 5).

    Drift-velocity changes move cluster x positions, altering clustering
    merge decisions and renumbering idents wholesale; y/z shapes are
    v-independent, so an old cluster maps to the same-apa new cluster
    covering the largest fraction (>= min_frac) of its occupied 3 cm y/z
    cells.  Unmapped old uids are left out (callers count them).
    """
    with open(old_calib) as fh:
        old = json.load(fh)["clusters"]
    with open(new_calib) as fh:
        new = json.load(fh)["clusters"]
    new_by_apa = defaultdict(list)
    for c in new:
        new_by_apa[c["apa"]].append((c["uid"], _cells(c)))
    m = {}
    for c in old:
        cells = _cells(c)
        if not cells:
            continue
        best_uid, best_frac = None, min_frac
        for uid, ncells in new_by_apa.get(c["apa"], ()):
            frac = len(cells & ncells) / len(cells)
            if frac > best_frac or (frac == best_frac and best_uid is None):
                best_uid, best_frac = uid, frac
        if best_uid is not None:
            m[c["uid"]] = best_uid
    return m


def find_truth(entries_by_uid, uid, time, tol):
    """Best truth entry for (uid, time) within tol, else None."""
    best, best_dt = None, tol
    for e in entries_by_uid.get(uid, ()):  # noqa: B905
        dt = abs(e["time"] - time)
        if dt <= best_dt:
            best, best_dt = e, dt
    return best


def score_event(calib_path, truth_entries, tol, min_len, min_npts,
                objective_tiers):
    with open(calib_path) as fh:
        calib = json.load(fh)
    clusters = {c["uid"]: c for c in calib["clusters"]}
    long_uid = {}
    for uid, c in clusters.items():
        long_uid[uid] = (cluster_len_cm(c) >= min_len
                         or c["npoints"] >= min_npts)
    flash_by_gid = {f["gid"]: f for f in calib["flashes"]}
    flash_times = [f["time"] for f in calib["flashes"]]
    sat_gids = {f["gid"] for f in calib["flashes"]
                if any(s > 0 for s in f.get("sat", []))}

    entries_by_uid = defaultdict(list)
    for e in truth_entries:
        entries_by_uid[e["uid"]].append(e)

    autos = [b for b in calib["bundles"] if b.get("auto_selected")]
    auto_times_by_uid = defaultdict(list)
    for b in autos:
        auto_times_by_uid[b["main_cluster"]].append(
            flash_by_gid[b["flash_gid"]]["time"])

    r = dict(agree=0, phantom=0, unknown=0, low_agree=0, low_phantom=0,
             missed=0, covered=0, missed_flash_cut=0,
             pos_short=0, pos_cluster_missing=0,
             flag_judged=defaultdict(int), flag_phantom=defaultdict(int),
             phantom_list=[], missed_list=[], unknown_list=[])

    # ---- auto side: judge every long auto bundle against the scan ----
    for b in autos:
        uid = b["main_cluster"]
        if uid not in clusters or not long_uid.get(uid):
            continue
        t = flash_by_gid[b["flash_gid"]]["time"]
        e = find_truth(entries_by_uid, uid, t, tol)
        if e is None:
            r["unknown"] += 1
            r["unknown_list"].append(dict(uid=uid, time=t))
            continue
        objective = e["conf"] in objective_tiers
        if e["positive"]:
            r["agree" if objective else "low_agree"] += 1
        else:
            r["phantom" if objective else "low_phantom"] += 1
            if objective:
                r["phantom_list"].append(dict(
                    uid=uid, time=t, conf=e["conf"],
                    ks=b["ks_dis"], chi2=b["chi2"], ndf=b["ndf"],
                    strength=b["strength"],
                    flags={k: bool(b.get(k)) for k in FLAG_KEYS},
                    sat_flash=b["flash_gid"] in sat_gids))
        if objective:
            for k in FLAG_KEYS:
                if b.get(k):
                    r["flag_judged"][k] += 1
                    if not e["positive"]:
                        r["flag_phantom"][k] += 1
            if b["flash_gid"] in sat_gids:
                r["flag_judged"]["sat_flash"] += 1
                if not e["positive"]:
                    r["flag_phantom"]["sat_flash"] += 1

    # ---- truth side: every objective positive must be covered ----
    for e in truth_entries:
        if not e["positive"] or e["conf"] not in objective_tiers:
            continue
        uid = e["uid"]
        if uid not in clusters:
            r["pos_cluster_missing"] += 1
            continue
        if not long_uid.get(uid):
            r["pos_short"] += 1
            continue
        covered = any(abs(t - e["time"]) <= tol
                      for t in auto_times_by_uid.get(uid, ()))
        if covered:
            r["covered"] += 1
        else:
            r["missed"] += 1
            flash_cut = not any(abs(t - e["time"]) <= tol
                                for t in flash_times)
            if flash_cut:
                r["missed_flash_cut"] += 1
            r["missed_list"].append(dict(uid=uid, time=e["time"],
                                         conf=e["conf"],
                                         flash_cut=flash_cut))
    return r


def self_check(work_root, tag, truth, tol):
    """Reproduce the documented headline: every keep/reject decision must join
    to a cathxa auto bundle; agree = keep/judged (no filters, all tiers, AI
    events only).  Known value 860/1279 = 67.2%."""
    keep = judged = fail = 0
    for idx in range(1, NEVT):
        evt = evt_of_idx(idx)
        calib = os.path.join(work_root, f"{RUN}_{idx}_{tag}",
                             f"calib-evt{evt}.json")
        with open(calib) as fh:
            d = json.load(fh)
        flash_by_gid = {f["gid"]: f for f in d["flashes"]}
        auto_keys = defaultdict(list)
        for b in d["bundles"]:
            if b.get("auto_selected"):
                auto_keys[b["main_cluster"]].append(
                    flash_by_gid[b["flash_gid"]]["time"])
        for e in truth[evt]:
            if e["conf"] == "gold":
                continue
            # keep/reject verdicts judged an auto bundle
            is_reject = not e["positive"]
            joined = any(abs(t - e["time"]) <= tol
                         for t in auto_keys.get(e["uid"], ()))
            if joined:
                judged += 1
                if e["positive"]:
                    keep += 1
            elif is_reject:
                fail += 1  # a reject verdict MUST join to an auto bundle
    return keep, judged, fail


def fmt_pct(num, den):
    return f"{100.0*num/den:.1f}%" if den else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="work/039252_<idx>_<tag>")
    ap.add_argument("--work-root", default="work")
    ap.add_argument("--gold",
                    default="work/ql_labels/wfresc/labels-evt298567.json")
    ap.add_argument("--decisions-dir", default="ql_display/decisions-cathxa")
    ap.add_argument("--tol", type=float, default=0.5,
                    help="flash-time match tolerance (us)")
    ap.add_argument("--min-len-cm", type=float, default=25.0)
    ap.add_argument("--min-npoints", type=int, default=100)
    ap.add_argument("--out-root", default="work/ql_scores")
    ap.add_argument("--self-check", action="store_true",
                    help="also reproduce the unfiltered 67.2%% baseline join")
    ap.add_argument("--truth-time-map", default=None,
                    help="tmerge_time_map.py JSON: translate truth flash "
                         "times across a tail-merge reprocess (doc 26)")
    ap.add_argument("--truth-uid-map-tag", default=None,
                    help="OLD calib tag (e.g. tm0k): per event, remap truth "
                         "cluster uids onto the scored run's clustering by "
                         "y/z cell overlap (doc 26 step 5 -- drift-velocity "
                         "changes renumber cluster idents). Default off = "
                         "byte-identical scoring.")
    ap.add_argument("--truth-time-shift", type=float, default=0.0,
                    help="us ADDED to every truth time after the map: scores "
                         "a run whose trigger-offset frame differs from the "
                         "truth's (doc 26 step 5: offset-0 sweep vs truth "
                         "recorded at PDVD_QL_EXTRA_OFFSET_US=13.507 => "
                         "-13.507). Default 0 = byte-identical scoring.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    truth = load_truth(args.gold, args.decisions_dir)
    truth = apply_time_map(truth, args.truth_time_map, args.tol)
    if args.truth_time_shift:
        for entries in truth.values():
            for e in entries:
                e["time"] += args.truth_time_shift
        print(f"[time-shift] all truth times {args.truth_time_shift:+.3f} us")

    out_dir = os.path.join(args.out_root, args.tag)
    if os.path.exists(os.path.join(out_dir, "scores.json")) and not args.force:
        sys.exit(f"refusing to overwrite {out_dir}/scores.json (M13); "
                 "use --force or a new tag")
    os.makedirs(out_dir, exist_ok=True)

    if args.self_check:
        keep, judged, fail = self_check(args.work_root, args.tag, truth,
                                        args.tol)
        print(f"[self-check] joined keep/reject verdicts: {judged} "
              f"(expect 1279 on cathxa), keep {keep} (expect 860), "
              f"agree {fmt_pct(keep, judged)} (expect 67.2%), "
              f"unjoined rejects {fail} (expect 0)")

    rows, total = [], defaultdict(int)
    tot_fj, tot_fp = defaultdict(int), defaultdict(int)
    detail = {}
    for idx in range(NEVT):
        evt = evt_of_idx(idx)
        calib = os.path.join(args.work_root, f"{RUN}_{idx}_{args.tag}",
                             f"calib-evt{evt}.json")
        if not os.path.exists(calib):
            print(f"[warn] missing {calib}; skipping evt{evt}")
            continue
        truth_evt = truth[evt]
        if args.truth_uid_map_tag:
            old_calib = os.path.join(args.work_root,
                                     f"{RUN}_{idx}_{args.truth_uid_map_tag}",
                                     f"calib-evt{evt}.json")
            umap = build_uid_map(old_calib, calib)
            unmapped = sum(1 for e in truth_evt if e["uid"] not in umap)
            truth_evt = [dict(e, uid=umap[e["uid"]])
                         for e in truth_evt if e["uid"] in umap]
            print(f"[uid-map] evt{evt}: {len(umap)} clusters mapped, "
                  f"{unmapped} truth entries unmapped")
        r = score_event(calib, truth_evt, args.tol, args.min_len_cm,
                        args.min_npoints, OBJECTIVE_TIERS)
        judged = r["agree"] + r["phantom"]
        npos = r["covered"] + r["missed"]
        rows.append((evt, r["agree"], r["phantom"],
                     fmt_pct(r["agree"], judged), r["unknown"],
                     r["missed"], npos, fmt_pct(r["missed"], npos),
                     r["missed_flash_cut"],
                     r["low_agree"], r["low_phantom"]))
        for k in ("agree", "phantom", "unknown", "missed", "covered",
                  "missed_flash_cut", "pos_short", "pos_cluster_missing",
                  "low_agree", "low_phantom"):
            total[k] += r[k]
        for k, v in r["flag_judged"].items():
            tot_fj[k] += v
        for k, v in r["flag_phantom"].items():
            tot_fp[k] += v
        detail[f"evt{evt}"] = {k: r[k] for k in
                               ("phantom_list", "missed_list", "unknown_list")}

    judged = total["agree"] + total["phantom"]
    npos = total["covered"] + total["missed"]
    md = []
    md.append(f"# QLMatch agreement — tag `{args.tag}` vs frozen scan truth")
    md.append("")
    md.append(f"Objective tiers {OBJECTIVE_TIERS}, long tracks "
              f"(>= {args.min_len_cm} cm or >= {args.min_npoints} pts), "
              f"tol {args.tol} us.")
    md.append("")
    md.append("| evt | agree | phantom | agree% | unknown | missed | npos "
              "| missed% | miss-flashcut | low-agree | low-phantom |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        md.append("| " + " | ".join(str(x) for x in row) + " |")
    md.append(f"| **all** | {total['agree']} | {total['phantom']} | "
              f"**{fmt_pct(total['agree'], judged)}** | {total['unknown']} | "
              f"{total['missed']} | {npos} | "
              f"**{fmt_pct(total['missed'], npos)}** | "
              f"{total['missed_flash_cut']} | {total['low_agree']} | "
              f"{total['low_phantom']} |")
    md.append("")
    md.append(f"Positives excluded: short {total['pos_short']}, "
              f"cluster-missing {total['pos_cluster_missing']}.")
    md.append("")
    md.append("| flag | judged | phantom | phantom% |")
    md.append("|---|---|---|---|")
    for k in FLAG_KEYS + ["sat_flash"]:
        if tot_fj.get(k):
            md.append(f"| {k} | {tot_fj[k]} | {tot_fp.get(k, 0)} | "
                      f"{fmt_pct(tot_fp.get(k, 0), tot_fj[k])} |")
    text = "\n".join(md) + "\n"
    print(text)

    with open(os.path.join(out_dir, "scores.md"), "w") as fh:
        fh.write(text)
    with open(os.path.join(out_dir, "scores.json"), "w") as fh:
        json.dump(dict(tag=args.tag, tol=args.tol,
                       min_len_cm=args.min_len_cm,
                       min_npoints=args.min_npoints,
                       objective_tiers=OBJECTIVE_TIERS,
                       total=dict(total), per_event=rows,
                       flag_judged=dict(tot_fj), flag_phantom=dict(tot_fp),
                       detail=detail), fh, indent=1)
    print(f"[out] {out_dir}/scores.md  {out_dir}/scores.json")


if __name__ == "__main__":
    main()
