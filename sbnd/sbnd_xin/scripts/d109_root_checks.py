#!/usr/bin/env python3
"""doc 109: content checks on a knob-ON arm's tracking-pr.root files.

Each check is a statement the new branches must satisfy BY CONSTRUCTION; a
nonzero failure count is a bug in the writer, not physics.  The census part
explains every event that has no T_tagger row.

Usage:
  scripts/d109_root_checks.py <label> [--samples nuecc48 ncpi0 mcp1k] [--jobs 24]

Checks (per T_tagger row unless stated):
  C1  exactly one roster entry with act_is_selected=1 and act_role<=1, and its
      act_cluster_id == sel_cluster_id
  C2  vertex_moved_cluster == (cluster_id != sel_cluster_id)
  C3  exactly one act_is_final=1, on the entry whose act_cluster_id == cluster_id
  C4  has_vertex==0 => nu_x=nu_y=nu_z=0; T_kine.has_vertex == T_tagger.has_vertex
  C5  run/subrun/event on T_tagger, T_kine, T_bundle, T_flash == Trun
  C6  T_bundle rows with nu_index>=0 == T_tagger rows: same (nu_index, gid),
      sel_cluster_id, final_cluster_id == cluster_id; reason in {0,1} iff nu_index>=0
  C7  Trun.nu_n_candidates == T_tagger entries; Trun.nu_n_bundles == T_bundle entries
  C8  T_cluster tgm/stm/fc/lm == act_tgm/stm/fc/lm for role-0/1 roster entries
  C9  T_cluster.beam_flash == (matched_flash_gid>=0 and cluster_t0_us in window)
  C10 T_flash gids unique; T_cluster.flash_tpc == T_flash.tpc of its gid;
      T_tagger flash_time_us/flash_pe/flash_tpc/flash_group == T_flash of its gid
  C11 T_flash rows sharing a flash_group: different TPCs, time span < flash_pair_dt_us
      (for pairs); group id == min gid
  C12 Trun provenance strings present and non-empty (wct_version, op_config_sha256)

sbnd_xin/docs/109 rev 3 (only run when the rev-3 branches are present, so this
script still grades a doc-109-only arm):
  C14 T_rec_charge and T_proj_data exist in EVERY file, candidate or not
  C15 every T_rec_charge row has cluster_id >= 0, and nu_index in [0, n_tagger)
  C16 the rows of candidate i carry exactly cluster_id == T_tagger.cluster_id[i]
      -- i.e. the points JOIN to the row that owns them, which is what the
      pre-rev-3 file could not do on a vertex-moved or demoted-main candidate
  C17 point_cluster_id == round(ndf) on every row (same quantity, honest name),
      and every point_cluster_id is a cluster T_cluster knows
"""
import argparse
import collections
import glob
import os
from multiprocessing import Pool

import numpy as np
import uproot

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REASON = {0: "selected main", 1: "selected demoted", 2: "all cosmic", 3: "length floor",
          4: "stm_only", 5: "no eligible",
          # sbnd_xin/docs/109 rev 3: a candidate WAS selected, then dropped because another
          # bundle of the same flash_group (one physical flash seen by both TPCs) kept a
          # longer one.  Only ever appears with nu_dedup_flash_group on.
          6: "dedup flash group"}


def arr(t, names):
    return {n: t[n].array(library="np") for n in names}


def check_event(path):
    c = collections.Counter()
    notes = []
    stats = collections.Counter()

    def bad(key, msg):
        c[key] += 1
        if len(notes) < 4:
            notes.append(f"{key}: {msg}")

    f = uproot.open(path)
    names = {k.split(";")[0] for k in f.keys()}

    # ---- sbnd_xin/docs/109 rev 3 ----------------------------------------- #
    # rev3 = the rec_charge_provenance knob was on for this arm.  Detected from
    # the file so a doc-109-only arm still grades C1-C12 and skips C14-C17.
    rev3 = "T_rec_charge" in names and "point_cluster_id" in {k.split(";")[0] for k in f["T_rec_charge"].keys()}
    if rev3:
        stats["rev3 files"] += 1
        # C14: the tree set no longer varies for a non-semantic reason.
        for tn in ("T_rec_charge", "T_proj_data"):
            if tn not in names:
                bad("C14 missing tree", tn)
        stats["T_rec_charge present"] += int("T_rec_charge" in names)
        stats["T_proj_data present"] += int("T_proj_data" in names)

    tr = f["Trun"]
    run = int(tr["runNo"].array(library="np")[0])
    sub = int(tr["subRunNo"].array(library="np")[0])
    evt = int(tr["eventNo"].array(library="np")[0])
    trk = set(tr.keys())
    if "nu_census_filled" not in trk:
        bad("C0 no census branches", path)
        return c, stats, notes
    R = {k: tr[k].array(library="np")[0] for k in trk if k.startswith("nu_") or k.endswith("_us")}
    stats["events"] += 1
    stats["census filled"] += int(R["nu_census_filled"])
    lo, hi, pair_dt = float(R["beam_window_low_us"]), float(R["beam_window_high_us"]), float(R["flash_pair_dt_us"])

    # C12
    for k in ("wct_version", "op_config_sha256"):
        if k not in trk:
            bad("C12 provenance missing", k)
        else:
            v = tr[k].array(library="np")[0]
            if not str(v):
                bad("C12 provenance empty", k)

    B = arr(f["T_bundle"], ["run", "subrun", "event", "gid", "flash_tpc", "flash_time_us", "flash_group",
                            "reason", "nu_index", "sel_cluster_id", "final_cluster_id", "n_main", "n_companion",
                            "n_demoted", "n_rej_cosmic", "n_rej_floor", "n_rej_stm_only"])
    F = arr(f["T_flash"], ["run", "subrun", "event", "gid", "tpc", "time_us", "pe", "in_window",
                           "flash_group", "nu_index", "n_matched_main"])
    C = arr(f["T_cluster"], ["cluster_id", "tgm", "stm", "fc", "lm", "beam_flash", "matched_flash_gid",
                             "flash_tpc", "cluster_t0_us", "flash_id", "flash_time_us"])
    for nm, T in (("T_bundle", B), ("T_flash", F)):
        if len(T["run"]) and not (np.all(T["run"] == run) and np.all(T["subrun"] == sub) and np.all(T["event"] == evt)):
            bad("C5 rse", nm)

    # C10 flash table
    fg = list(F["gid"])
    if len(set(fg)) != len(fg):
        bad("C10 dup flash gid", str(fg))
    fidx = {int(g): i for i, g in enumerate(fg)}
    for i, g in enumerate(C["matched_flash_gid"]):
        if g >= 0:
            j = fidx.get(int(g))
            if j is None:
                bad("C10 cluster gid not in T_flash", int(g))
            elif int(C["flash_tpc"][i]) != int(F["tpc"][j]):
                bad("C10 T_cluster flash_tpc", int(g))
            stats["cluster gid rows"] += 1
            stats["flash_tpc == gid//1e6"] += int(int(C["flash_tpc"][i]) == int(g) // 1000000)
    # C11 groups
    groups = collections.defaultdict(list)
    for i, g in enumerate(F["flash_group"]):
        groups[int(g)].append(i)
    for gid, members in groups.items():
        if gid != min(int(F["gid"][i]) for i in members):
            bad("C11 group id != min gid", gid)
        if len(members) > 1:
            tpcs = [int(F["tpc"][i]) for i in members]
            if len(members) == 2 and tpcs[0] == tpcs[1]:
                bad("C11 same-TPC pair", gid)
            if len(members) == 2 and abs(F["time_us"][members[0]] - F["time_us"][members[1]]) >= pair_dt + 1e-4:
                bad("C11 pair dt", gid)
            stats[f"flash groups of {len(members)}"] += 1
    stats["flashes"] += len(fg)
    stats["flashes in window"] += int(np.sum(F["in_window"]))
    nwin_groups = {int(F["flash_group"][i]) for i in range(len(fg)) if F["in_window"][i]}
    stats[f"in-window flash groups per event={min(len(nwin_groups), 3)}"] += 1

    # C9 beam_flash
    for i in range(len(C["cluster_id"])):
        g, t0 = int(C["matched_flash_gid"][i]), float(C["cluster_t0_us"][i])
        exp = int(g >= 0 and lo <= t0 < hi)
        if int(C["beam_flash"][i]) != exp:
            bad("C9 beam_flash", (int(C["cluster_id"][i]), g, t0))
        stats["T_cluster rows"] += 1
        stats["T_cluster beam_flash=1"] += int(C["beam_flash"][i] == 1)
        stats["T_cluster tgm=1"] += int(C["tgm"][i] == 1)
        stats["T_cluster stm=1"] += int(C["stm"][i] == 1)
        stats["T_cluster fc=1"] += int(C["fc"][i] == 1)

    # C7 / census reasons
    ntag = f["T_tagger"].num_entries if "T_tagger" in names else 0
    if int(R["nu_n_candidates"]) != ntag:
        bad("C7 nu_n_candidates", (int(R["nu_n_candidates"]), ntag))
    if int(R["nu_n_bundles"]) != len(B["gid"]):
        bad("C7 nu_n_bundles", (int(R["nu_n_bundles"]), len(B["gid"])))
    for r in B["reason"]:
        stats[f"T_bundle reason {int(r)} {REASON.get(int(r))}"] += 1
    if ntag == 0:
        stats["events with no T_tagger"] += 1
        if len(B["gid"]) == 0:
            if int(R["nu_n_in_window_main"]) == 0 and int(R["nu_n_in_window_demoted"]) == 0:
                why = "no in-window main/demoted cluster"
            elif int(R["nu_n_in_window_nogid"]) > 0:
                why = "in-window clusters all without matched flash"
            else:
                why = "UNEXPLAINED (no bundles)"
                bad("C8x unexplained no-row event", path)
        else:
            rs = sorted({int(r) for r in B["reason"]})
            why = "bundles rejected: " + "+".join(REASON[r] for r in rs)
            if any(r in (0, 1) for r in rs):
                bad("C6 selected bundle but no T_tagger", path)
        stats[f"no-row why: {why}"] += 1
        stats["no-row events with in-window flash(es)"] += int(np.sum(F["in_window"]) > 0)
        stats["no-row events with nogid clusters"] += int(R["nu_n_in_window_nogid"] > 0)
        if rev3 and "T_rec_charge" in names:
            n = f["T_rec_charge"].num_entries
            stats["no-row events: T_rec_charge rows"] += n
            if n:
                bad("C15 rows with no candidate", (path, n))
        return c, stats, notes

    T = arr(f["T_tagger"], ["run", "subrun", "event", "cluster_id", "matched_flash_gid", "nu_index",
                            "sel_cluster_id", "vertex_moved_cluster", "has_vertex", "flash_time_us",
                            "flash_pe", "flash_tpc", "flash_group", "nu_x", "nu_y", "nu_z"])
    ACT = arr(f["T_tagger"], ["act_cluster_id", "act_is_selected", "act_role", "act_is_final",
                              "act_in_pr", "act_tgm", "act_stm", "act_fc", "act_lm"])
    K = arr(f["T_kine"], ["run", "subrun", "event", "has_vertex", "nu_index", "cluster_id"])
    if not (np.all(T["run"] == run) and np.all(T["subrun"] == sub) and np.all(T["event"] == evt)):
        bad("C5 rse", "T_tagger")
    if not (np.all(K["run"] == run) and np.all(K["subrun"] == sub) and np.all(K["event"] == evt)):
        bad("C5 rse", "T_kine")
    cl_row = {int(cid): i for i, cid in enumerate(C["cluster_id"])}
    bundle_by_nu = {int(n): i for i, n in enumerate(B["nu_index"]) if n >= 0}
    for i in range(ntag):
        stats["rows"] += 1
        cid, sel = int(T["cluster_id"][i]), int(T["sel_cluster_id"][i])
        ids, iss, role, fin = ACT["act_cluster_id"][i], ACT["act_is_selected"][i], ACT["act_role"][i], ACT["act_is_final"][i]
        n = len(ids)
        if not (len(iss) == len(role) == len(fin) == n):
            bad("C1 roster lengths", i)
            continue
        js = [j for j in range(n) if iss[j] == 1 and role[j] <= 1]
        if len(js) != 1 or int(ids[js[0]]) != sel:
            bad("C1 selected entry", (i, sel, list(ids), list(iss)))
        if int(T["vertex_moved_cluster"][i]) != int(cid != sel):
            bad("C2 vertex_moved_cluster", (i, cid, sel))
        stats["rows vertex_moved_cluster=1"] += int(cid != sel)
        jf = [j for j in range(n) if fin[j] == 1]
        if len(jf) != 1 or int(ids[jf[0]]) != cid:
            bad("C3 act_is_final", (i, cid, list(ids), list(fin)))
        for r in role:
            stats[f"roster role {int(r)}"] += 1
        # C13: each cluster once in the roster; act_in_pr marks the selected
        # activity plus every kept companion (T_bundle.n_companion).
        if len(set(int(x) for x in ids)) != n:
            bad("C13 duplicate roster cluster", (i, list(ids)))
        inpr = ACT["act_in_pr"][i]
        b = bundle_by_nu.get(int(T["nu_index"][i]))
        if b is not None and int(np.sum(inpr)) != 1 + int(B["n_companion"][b]):
            bad("C13 act_in_pr count", (i, int(np.sum(inpr)), int(B["n_companion"][b])))
        if any(int(inpr[j]) != 1 for j in range(n) if iss[j] == 1 and role[j] <= 1):
            bad("C13 selected entry not in_pr", i)
        stats["roster role-1 entries also in PR pass"] += sum(1 for j in range(n) if role[j] == 1 and inpr[j] == 1)
        hv = int(T["has_vertex"][i])
        stats[f"rows has_vertex={hv}"] += 1
        if hv == 0 and not (T["nu_x"][i] == 0 and T["nu_y"][i] == 0 and T["nu_z"][i] == 0):
            bad("C4 no vertex but nu_xyz", i)
        if hv == 1 and T["nu_x"][i] == 0 and T["nu_y"][i] == 0 and T["nu_z"][i] == 0:
            stats["rows has_vertex=1 at (0,0,0)"] += 1
        if int(K["has_vertex"][i]) != hv:
            bad("C4 T_kine has_vertex", i)
        # C6
        b = bundle_by_nu.get(int(T["nu_index"][i]))
        if b is None or int(B["gid"][b]) != int(T["matched_flash_gid"][i]):
            bad("C6 bundle row", i)
        else:
            if int(B["sel_cluster_id"][b]) != sel or int(B["final_cluster_id"][b]) != cid:
                bad("C6 bundle ids", (i, int(B["sel_cluster_id"][b]), int(B["final_cluster_id"][b])))
            if int(B["reason"][b]) not in (0, 1):
                bad("C6 bundle reason", i)
        # C10 row flash
        j = fidx.get(int(T["matched_flash_gid"][i]))
        if j is None:
            bad("C10 row gid not in T_flash", i)
        else:
            if not (abs(T["flash_time_us"][i] - F["time_us"][j]) < 1e-4 and abs(T["flash_pe"][i] - F["pe"][j]) < 1e-2
                    and int(T["flash_tpc"][i]) == int(F["tpc"][j]) and int(T["flash_group"][i]) == int(F["flash_group"][j])):
                bad("C10 row flash fields", i)
            if int(F["nu_index"][j]) != int(T["nu_index"][i]):
                bad("C10 T_flash nu_index", i)
        # C8 flags of role-0/1 roster entries vs T_cluster
        for j in range(n):
            if role[j] > 1:
                continue
            k = cl_row.get(int(ids[j]))
            if k is None:
                bad("C8 roster cluster not in T_cluster", int(ids[j]))
                continue
            for fl in ("tgm", "stm", "fc", "lm"):
                if int(C[fl][k]) != int(ACT["act_" + fl][i][j]):
                    bad(f"C8 {fl}", (int(ids[j]), int(C[fl][k]), int(ACT['act_' + fl][i][j])))
    for i in range(len(K["nu_index"])):
        if int(K["nu_index"][i]) != int(T["nu_index"][i]) or int(K["cluster_id"][i]) != int(T["cluster_id"][i]):
            bad("C4 T_kine pairing", i)
    # ---- sbnd_xin/docs/109 rev 3: C15 / C16 / C17 ------------------------ #
    if rev3 and "T_rec_charge" in names:
        RC = arr(f["T_rec_charge"], ["cluster_id", "nu_index", "point_cluster_id", "ndf"])
        nrc = len(RC["cluster_id"])
        stats["T_rec_charge rows"] += nrc
        if nrc:
            # C15: no -1 survives, and nu_index names a real candidate.
            nneg = int(np.sum(RC["cluster_id"] < 0))
            if nneg:
                bad("C15 cluster_id still -1", (path, nneg))
            bad_ni = [int(x) for x in RC["nu_index"] if not (0 <= int(x) < ntag)]
            if bad_ni:
                bad("C15 nu_index out of range", (path, sorted(set(bad_ni))[:4], ntag))
            # C17: point_cluster_id is what ndf carries, and it is a real cluster.
            nd = np.rint(RC["ndf"]).astype(int)
            nmis = int(np.sum(nd != RC["point_cluster_id"]))
            if nmis:
                bad("C17 point_cluster_id != round(ndf)", (path, nmis))
            for pcid in sorted({int(x) for x in RC["point_cluster_id"]}):
                if pcid not in cl_row:
                    bad("C17 point cluster not in T_cluster", (path, pcid))
        # C16: the join.  Every candidate's rows carry exactly its cluster_id.
        for i in range(ntag):
            sel = RC["nu_index"] == i if nrc else np.zeros(0, dtype=bool)
            ids = sorted({int(x) for x in RC["cluster_id"][sel]}) if nrc else []
            want = int(T["cluster_id"][i])
            if not ids:
                # A candidate with no vertex writes no points; that is the
                # placeholder row, and has_vertex already says so.
                stats["candidates with 0 T_rec_charge rows"] += 1
                if int(T["has_vertex"][i]) == 1:
                    bad("C16 vertex but no points", (path, i, want))
                continue
            stats["candidates with T_rec_charge rows"] += 1
            if ids != [want]:
                bad("C16 points do not join to the row", (path, i, want, ids))
            if int(T["vertex_moved_cluster"][i]) == 1:
                stats["vertex-moved rows whose points join"] += int(ids == [want])
            # Honest accounting: cluster_id is now a per-candidate constant, so
            # C16 is a fallback check (it fails only if TaggerInfo::cluster_id
            # was unset and the old flag scan ran).  point_cluster_id is what
            # says where the points actually came from -- in particular whether
            # the candidate's OWN main contributed any (the "route B" shape
            # behind part of the old -1: main flagged, but absent from the
            # candidate's graph, so the rows came only from companions).
            own = {int(x) for x in RC["point_cluster_id"][sel]}
            stats["candidates whose main contributes points"] += int(want in own)
            stats["candidates whose points are ALL from companions"] += int(want not in own)

    if ntag == 2:
        g = [int(x) for x in T["flash_group"]]
        stats["two-row events"] += 1
        stats["two-row events sharing a flash_group"] += int(g[0] == g[1])
        stats["two-row events sharing a group, a row without vertex"] += int(g[0] == g[1] and 0 in [int(x) for x in T["has_vertex"]])
    return c, stats, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--samples", nargs="+", default=["nuecc48", "ncpi0", "mcp1k"])
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    fails = collections.Counter()
    allnotes = []
    for s in a.samples:
        files = sorted(glob.glob(f"{SX}/work-{s}-{a.label}/pr_evt*/tracking-pr.root"))
        stats = collections.Counter()
        with Pool(a.jobs) as p:
            for c, st, notes in p.imap(check_event, files):
                fails.update(c)
                stats.update(st)
                allnotes += notes
        print(f"== {s} ({a.label}): {len(files)} files")
        for k in sorted(stats):
            print(f"  {k:62s} {stats[k]}")
    print("== check failures (0 expected everywhere)")
    for k in sorted(fails):
        print(f"  {k:62s} {fails[k]}")
    if not fails:
        print("  none")
    for n in allnotes[:20]:
        print("  note:", n)
    print("=== OVERALL:", "PASS" if not fails else "FAIL")


if __name__ == "__main__":
    main()
