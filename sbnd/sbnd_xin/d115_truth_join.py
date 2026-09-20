#!/usr/bin/env python3
"""doc sbnd_xin/115: join the round-3 truth table with the tracking-pr tagger/PR output of OUR
standalone chain, and write the doc-107 analysis tables.

Fork of d107_truth_tagger_consistency.py (CLAUDE.md M10 -- that script is the doc-107 record and
stays byte-untouched).  Same CAND_COLS / TRUTH_COLS output schema, so d107_selection.py and
d107_tables.py read these products unmodified and the two rounds stay directly comparable.
Five deliberate differences, each forced by a real change:

  1. TRUTH SOURCE.  Doc 107's sample was produced by the LArSoft-embedded 1-step chain, which
     runs wclsTensorSetLabeler and therefore writes a truth layer into Bee's 0-mc.json.  Our
     standalone chain has no such stage and writes no truth layer at all.  Truth here comes from
     scripts/d115/build_truth.py's truth_base.tsv (the sample's GENIE TSV + Edep read out of the
     reco1 files), joined on (run, subrun, event).

  2. RSE.  Doc 107 parsed run/subrun/event out of the FILENAME (r<run>_s<subrun>_e<event>).
     Doc 109 put real run/subrun/event branches on T_tagger/T_kine/T_bundle/T_flash, so we read
     them.  That is what lets this campaign use one out_root per reco1 file -- SBND MC event
     numbers repeat across files, so pr_evt<ID> is not a unique key, but the branches are.

  3. act_role <= 1.  Doc 109 sec 2.1 APPENDED companions to the act_* roster as role 2 (companion)
     and 3 (companion dropped as cosmic).  Doc 107's roster was mains + demoted mains only, and
     it says to keep that meaning by filtering role <= 1.  Without the filter n_act, n_vetoed,
     bundle_has_vetoed and cid_relation all silently change meaning between the two rounds.

  4. REAL BRANCHES FOR THE VERTEX AND THE SELECTED ACTIVITY.  vertex_default now comes from
     has_vertex (1 - has_vertex), not from doc 107's "nu_x/y/z == (0,0,0)" heuristic; the
     selected activity from sel_cluster_id and vertex_moved_cluster, not re-derived from
     act_is_selected.  Column names and semantics are preserved, so the consumers do not change.

  5. NO CANDIDATE.  T_tagger and T_kine are written only when a candidate exists (10 trees vs 8),
     so a file without them is "no candidate", not an error.

Association is by nearest true vertex, as in doc 107.  The charge-based match doc 107 sec 7
recommends needs Bee's truth_trackid_labeled point set, which the standalone chain does not
produce -- out of scope for this round.  As an independent cross-check this script also records
flash_time_us, so the doc can report the doc-108 sec 3.5 time association
(flash_time_us - 0.136 us ~ true nu time) beside the vertex one.

Usage:
  python3 d115_truth_join.py --arm work-r3cv-d115pr --truth products/d115/cv/truth_base.tsv \\
                             --out products/d115/cv
  python3 d115_truth_join.py --arm work-r3off-d115pr --out products/d115/off   # no truth (data)
"""
import argparse
import collections
import glob
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
import uproot

# Doc 107's schemas, unchanged.
CAND_COLS = ["run", "subrun", "event", "row", "nu_index", "gid", "cluster_id",
             "sel_cluster_id", "cid_relation", "n_act", "sel_demoted", "sel_fc",
             "sel_length_cm", "bundle_has_vetoed", "n_vetoed", "nu_x", "nu_y", "nu_z",
             "vertex_default", "reco_Enu", "numu_score", "nue_score", "neutrino_type",
             "nt_cosmic", "nt_numucc", "nt_nc", "nt_nue", "cosmict_flag",
             "n_truth", "t_idx", "t_flav", "t_mode", "t_ccnc", "t_Etot", "t_Edep", "t_T",
             "t_dist_cm", "mc_reco_Enu"]
TRUTH_COLS = ["run", "subrun", "event", "idx", "flav", "mode", "ccnc", "Etot", "Edep",
              "T", "vx", "vy", "vz", "event_has_candidate", "n_candidates",
              "min_cand_dist_cm"]
# d115 additions, appended so the doc-107 consumers keep working on the first 38 columns.
EXTRA_COLS = ["flash_time_us", "flash_pe", "flash_tpc", "flash_group",
              "has_vertex", "vertex_moved_cluster", "br_filled", "mip_filled",
              "n_act_all_roles", "n_companion"]

TAGGER_BR = ["run", "subrun", "event", "cluster_id", "sel_cluster_id", "vertex_moved_cluster",
             "has_vertex", "matched_flash_gid", "nu_index", "neutrino_type",
             "nu_x", "nu_y", "nu_z", "numu_score", "nue_score", "cosmict_flag",
             "br_filled", "mip_filled", "flash_time_us", "flash_pe", "flash_tpc", "flash_group",
             "act_cluster_id", "act_role", "act_length_cm", "act_is_selected", "act_is_demoted",
             "act_tgm", "act_stm", "act_fc", "act_lm"]

TRUTH = {}          # (run, subrun, event) -> [interaction dicts]; filled in main, shared by fork


def load_truth(path):
    """truth_base.tsv -> {(run, subrun, event): [dict, ...]}."""
    out = collections.defaultdict(list)
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ci = {k: i for i, k in enumerate(hdr)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < len(hdr):
                continue
            key = tuple(int(f[ci[k]]) for k in ("run", "subrun", "event"))
            out[key].append(dict(idx=int(f[ci["idx"]]), flav=f[ci["flav"]], mode=f[ci["mode"]],
                                 ccnc=f[ci["ccnc"]], Etot=float(f[ci["Etot"]]),
                                 Edep=float(f[ci["Edep"]]), T=float(f[ci["T"]]),
                                 vx=float(f[ci["vx"]]), vy=float(f[ci["vy"]]),
                                 vz=float(f[ci["vz"]])))
    return out


def one(path):
    """One pr_evt<ID>/tracking-pr.root -> (counter, candidate rows, truth rows)."""
    C = collections.Counter()
    cand_rows, truth_rows = [], []
    try:
        F = uproot.open(path)
    except Exception as exc:                     # a truncated file is a finding, not a crash
        C["unreadable_root"] += 1
        return C, [], [], [(path, repr(exc))]
    keys = {k.split(";")[0] for k in F.keys()}

    # RSE: from T_tagger when there is a candidate, else from Trun (always present).
    has_tagger = "T_tagger" in keys and F["T_tagger"].num_entries > 0
    if "Trun" not in keys:
        C["no_Trun"] += 1
        return C, [], [], [(path, "no Trun")]
    tr = F["Trun"].arrays(["runNo", "subRunNo", "eventNo"], library="np")
    run, sub, evt = int(tr["runNo"][0]), int(tr["subRunNo"][0]), int(tr["eventNo"][0])
    # d115 V8 -- the doc-109 provenance strings, counted so the summary shows one value per arm.
    # A second value here means events were produced by different binaries or configs, which
    # would invalidate every number in the round; it must be visible, not assumed away.
    prov = F["Trun"].arrays(["wct_version", "toolkit_git", "op_config_sha256",
                             "numu_xgboost_xml", "nue_xgboost_xml", "dl_weights"], library="np")
    for k in prov:
        v = prov[k][0]
        C["prov %s = %s" % (k, v.decode() if isinstance(v, bytes) else v)] += 1
    truth = TRUTH.get((run, sub, evt), [])
    C["events"] += 1
    C["events_with_truth=%d" % int(bool(truth))] += 1
    C["has_T_tagger=%d" % int(has_tagger)] += 1

    cand_vtx = []
    if has_tagger:
        t = F["T_tagger"]
        a = t.arrays(TAGGER_BR, library="np")
        kine = F["T_kine"].arrays(["kine_reco_Enu"], library="np") if "T_kine" in keys else None
        # T_tagger's own RSE must agree with Trun's -- doc 109 says they are equal; assert it
        # rather than assume, because everything downstream joins on it.
        if int(a["run"][0]) != run or int(a["subrun"][0]) != sub or int(a["event"][0]) != evt:
            C["rse_mismatch_Trun_vs_Ttagger"] += 1
        for i in range(t.num_entries):
            ac_all = a["act_cluster_id"][i]
            role = a["act_role"][i]
            # ---- difference 3: doc-107 roster == mains + demoted mains == role <= 1 ----------
            keep = role <= 1
            ac = ac_all[keep]
            C["activities"] += len(ac)
            C["activities_all_roles"] += len(ac_all)
            C["companions"] += int((role >= 2).sum())

            veto = ((a["act_tgm"][i] == 1) | (a["act_stm"][i] == 1) | (a["act_lm"][i] > 0))[keep]
            sel_cid = int(a["sel_cluster_id"][i])
            cid = int(a["cluster_id"][i])
            # index of the selected activity inside the filtered roster
            jj = np.where(keep & (a["act_is_selected"][i] == 1))[0]
            if len(jj) == 0:
                jj = np.where(keep & (ac_all == sel_cid))[0]
            if len(jj) == 1:
                j = int(jj[0])
                jf = int(np.where(np.where(keep)[0] == j)[0][0])
                sel_dem = int(a["act_is_demoted"][i][j])
                sel_fc = int(a["act_fc"][i][j])
                sel_len = float(a["act_length_cm"][i][j])
                others = np.arange(len(ac)) != jf
                n_vet = int(veto[others].sum())
            else:
                C["selected_activity_not_resolved"] += 1
                sel_dem, sel_fc, sel_len = -1, -1, -1.0
                n_vet = int(veto.sum())

            # ---- difference 4: the real branches, not the (0,0,0) heuristic -----------------
            hv = int(a["has_vertex"][i])
            vmoved = int(a["vertex_moved_cluster"][i])
            vdef = 1 - hv
            rel = ("same" if not vmoved else
                   "in_act_not_selected" if cid in ac else "not_in_act")
            C["cluster_id vs selected: %s" % rel] += 1

            nt = int(a["neutrino_type"][i])
            v = (float(a["nu_x"][i]), float(a["nu_y"][i]), float(a["nu_z"][i]))
            best, bd = None, None
            if truth and not vdef:
                best = min(truth, key=lambda x: math.dist(v, (x["vx"], x["vy"], x["vz"])))
                bd = math.dist(v, (best["vx"], best["vy"], best["vz"]))
                cand_vtx.append(v)
            row = dict(run=run, subrun=sub, event=evt, row=i, nu_index=int(a["nu_index"][i]),
                       gid=int(a["matched_flash_gid"][i]), cluster_id=cid,
                       sel_cluster_id=sel_cid, cid_relation=rel, n_act=len(ac),
                       sel_demoted=sel_dem, sel_fc=sel_fc, sel_length_cm="%.1f" % sel_len,
                       bundle_has_vetoed=int(n_vet > 0), n_vetoed=n_vet,
                       nu_x="%.2f" % v[0], nu_y="%.2f" % v[1], nu_z="%.2f" % v[2],
                       vertex_default=vdef,
                       reco_Enu=("%.1f" % float(kine["kine_reco_Enu"][i])) if kine is not None else "-1",
                       numu_score="%.4f" % float(a["numu_score"][i]),
                       nue_score="%.4f" % float(a["nue_score"][i]), neutrino_type=nt,
                       nt_cosmic=(nt >> 1) & 1, nt_numucc=(nt >> 2) & 1, nt_nc=(nt >> 3) & 1,
                       nt_nue=(nt >> 5) & 1, cosmict_flag=int(a["cosmict_flag"][i]),
                       n_truth=len(truth),
                       t_idx=best["idx"] if best else -1,
                       t_flav=best["flav"] if best else "",
                       t_mode=best["mode"] if best else "",
                       t_ccnc=best["ccnc"] if best else "",
                       t_Etot=best["Etot"] if best else -1,
                       t_Edep=best["Edep"] if best else -1,
                       t_T=best["T"] if best else -1,
                       t_dist_cm="%.2f" % bd if best else -1,
                       mc_reco_Enu="",           # doc-107 column with no counterpart here
                       flash_time_us="%.4f" % float(a["flash_time_us"][i]),
                       flash_pe="%.1f" % float(a["flash_pe"][i]),
                       flash_tpc=int(a["flash_tpc"][i]), flash_group=int(a["flash_group"][i]),
                       has_vertex=hv, vertex_moved_cluster=vmoved,
                       br_filled=int(float(a["br_filled"][i])),
                       mip_filled=int(float(a["mip_filled"][i])),
                       n_act_all_roles=len(ac_all), n_companion=int((role >= 2).sum()))
            cand_rows.append(row)
        C["candidate_rows"] += t.num_entries

    for x in truth:
        d = min((math.dist(cv, (x["vx"], x["vy"], x["vz"])) for cv in cand_vtx), default=-1.0)
        truth_rows.append(dict(run=run, subrun=sub, event=evt, idx=x["idx"], flav=x["flav"],
                               mode=x["mode"], ccnc=x["ccnc"], Etot=x["Etot"], Edep=x["Edep"],
                               T=x["T"], vx=x["vx"], vy=x["vy"], vz=x["vz"],
                               event_has_candidate=int(has_tagger),
                               n_candidates=len(cand_rows), min_cand_dist_cm="%.2f" % d))
    return C, cand_rows, truth_rows, []


def _init(truth_map):
    global TRUTH
    TRUTH = truth_map


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True,
                    help="stage-B arm root; globs <arm>/f*/pr_evt*/ and <arm>/pr_evt*/")
    ap.add_argument("--truth", default="", help="truth_base.tsv (omit for data: no truth)")
    ap.add_argument("--out", required=True, help="output dir for candidates/truth/summary")
    ap.add_argument("--jobs", type=int, default=24)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.arm, "f*", "pr_evt*", "tracking-pr.root"))
                   + glob.glob(os.path.join(args.arm, "pr_evt*", "tracking-pr.root")))
    if not files:
        sys.exit("ERROR: no tracking-pr.root under %s" % args.arm)
    truth_map = load_truth(args.truth) if args.truth else {}
    print("arm %s: %d tracking-pr.root; truth events %d" % (args.arm, len(files), len(truth_map)))

    C = collections.Counter()
    cand, truth_rows, errs = [], [], []
    with Pool(args.jobs, initializer=_init, initargs=(truth_map,)) as pool:
        for c, cr, tr, er in pool.imap_unordered(one, files, chunksize=8):
            C.update(c); cand += cr; truth_rows += tr; errs += er

    cand.sort(key=lambda r: (r["run"], r["subrun"], r["event"], r["row"]))
    truth_rows.sort(key=lambda r: (r["run"], r["subrun"], r["event"], r["idx"]))

    os.makedirs(args.out, exist_ok=True)
    cols = CAND_COLS + EXTRA_COLS
    with open(os.path.join(args.out, "candidates.tsv"), "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in cand:
            fh.write("\t".join(str(r.get(k, "")) for k in cols) + "\n")
    if truth_map:
        with open(os.path.join(args.out, "truth.tsv"), "w") as fh:
            fh.write("\t".join(TRUTH_COLS) + "\n")
            for r in truth_rows:
                fh.write("\t".join(str(r[k]) for k in TRUTH_COLS) + "\n")

    # Join integrity (d115 V7): every candidate event must be in the truth table.
    if truth_map:
        cev = {(r["run"], r["subrun"], r["event"]) for r in cand}
        orphan = sorted(cev - set(truth_map))
        C["candidate_events_not_in_truth"] = len(orphan)
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write("arm %s\ntruth %s\nfiles %d\n\n" % (args.arm, args.truth or "(none)", len(files)))
        for k in sorted(C):
            fh.write("%-48s %d\n" % (k, C[k]))
        if errs:
            fh.write("\nERRORS (%d):\n" % len(errs))
            for p, e in errs[:50]:
                fh.write("  %s : %s\n" % (p, e))
    print("events %d  candidate rows %d  truth interactions %d  errors %d"
          % (C["events"], len(cand), len(truth_rows), len(errs)))
    print("roster: role<=1 activities %d, companions dropped by the filter %d"
          % (C["activities"], C["companions"]))
    if truth_map:
        print("candidate events not in the truth table: %d" % C["candidate_events_not_in_truth"])
    print("-> %s/{candidates.tsv%s,summary.txt}" % (args.out, ",truth.tsv" if truth_map else ""))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
