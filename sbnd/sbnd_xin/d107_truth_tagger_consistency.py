#!/usr/bin/env python3
"""doc 107: join the Bee mc.json truth with the tracking-pr tagger/PR output of
the lar+WCT SBND MC production, and census the cosmic-tagger vs neutrino-PR
consistency.  Read-only on the sample.

Inputs (per event tag r<run>_s<subrun>_e<event>):
  <run>/tracking-pr/tracking-pr_<tag>.root  T_tagger (+act_*), T_kine
  <run>/bee/bee_<tag>.zip::data/0/0-mc.json  truth + reco particle-flow forest

mc.json node classification (see doc 107 sec 2):
  truth interaction  id in [9000000, 9100000), text
                     "<n> <flav> <mode> <CC|NC> Etot E MeV Edep D MeV T t us"
  reco summary       id 19999999, text "reco nu <Enu> MeV numu <s> nue <s>"
  no-candidate       text "no reco neutrino candidate (<why>)"

Outputs (in --out, which must not exist yet):
  candidates.tsv  one row per T_tagger row
  truth.tsv       one row per truth interaction
  summary.txt     census counts

Usage:
  python3 d107_truth_tagger_consistency.py --run sbnd_mc_data --out products/d107
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import sys
import zipfile
from multiprocessing import Pool

import numpy as np
import uproot

TRUTH_RE = re.compile(
    r"^(\d+) (\S+) (\S+) (CC|NC|\?\?) Etot ([-\d.]+) MeV Edep ([-\d.]+) MeV T ([-\d.]+) us$")
RECO_RE = re.compile(r"^reco nu\s+([-\d.]+) MeV\s+numu ([-\d.]+)\s+nue ([-\d.]+)")
TAG_RE = re.compile(r"r(\d+)_s(\d+)_e(\d+)")

TAGGER_BR = ["cluster_id", "matched_flash_gid", "nu_index", "neutrino_type",
             "nu_x", "nu_y", "nu_z", "numu_score", "nue_score", "cosmict_flag",
             "act_cluster_id", "act_length_cm", "act_is_selected", "act_is_demoted",
             "act_tgm", "act_stm", "act_fc", "act_lm"]

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


def parse_mc(zpath):
    """Return (truth list, reco summaries, markers, other) from mc.json."""
    truth, reco, marker, other = [], [], [], []
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith("-mc.json")]
        if not names:
            return None
        nodes = json.loads(z.read(names[0]))
    for n in nodes:
        t, nid = n.get("text", ""), n.get("id", -1)
        m = TRUTH_RE.match(t)
        if m and 9000000 <= nid < 9100000:
            s = n["data"]["start"]
            truth.append(dict(idx=int(m.group(1)), flav=m.group(2), mode=m.group(3),
                              ccnc=m.group(4), Etot=float(m.group(5)),
                              Edep=float(m.group(6)), T=float(m.group(7)),
                              vx=s[0], vy=s[1], vz=s[2]))
            continue
        m = RECO_RE.match(t)
        if m and nid == 19999999:
            reco.append(dict(Enu=float(m.group(1)), numu=float(m.group(2)),
                             nue=float(m.group(3))))
            continue
        if t.startswith("no reco neutrino candidate"):
            marker.append(t)
            continue
        other.append(t)
    return truth, reco, marker, other


def one(args):
    run_dir, tag = args
    C = collections.Counter()
    run, sub, evt = (int(x) for x in TAG_RE.search(tag).groups())
    cand_rows, truth_rows = [], []

    mc = parse_mc(os.path.join(run_dir, "bee", f"bee_{tag}.zip"))
    if mc is None:
        C["mcjson_missing"] += 1
        truth, reco, marker, other = [], [], [], []
    else:
        truth, reco, marker, other = mc
    C[f"mc_reco_nodes={len(reco)}"] += 1
    C[f"mc_markers={len(marker)}"] += 1
    C["mc_other_top_nodes"] += len(other)
    C[f"mc_truth_interactions={min(len(truth), 5)}{'+' if len(truth) >= 5 else ''}"] += 1

    F = uproot.open(os.path.join(run_dir, "tracking-pr", f"tracking-pr_{tag}.root"))
    keys = {k.split(";")[0] for k in F.keys()}
    has_tagger = "T_tagger" in keys and F["T_tagger"].num_entries > 0
    C[f"has_T_tagger={int(has_tagger)} mc_reco_node={int(len(reco) > 0)}"] += 1

    cand_vtx = []
    if has_tagger:
        t = F["T_tagger"]
        a = t.arrays(TAGGER_BR, library="np")
        k = F["T_kine"].arrays(["kine_reco_Enu"], library="np")
        for i in range(t.num_entries):
            sel = a["act_is_selected"][i] == 1
            ac = a["act_cluster_id"][i]
            nsel = int(sel.sum())
            C[f"nsel={nsel}"] += 1
            j = int(np.argmax(sel)) if nsel == 1 else -1
            veto = ((a["act_tgm"][i] == 1) | (a["act_stm"][i] == 1) | (a["act_lm"][i] > 0))
            if j >= 0:
                C[f"selected tgm={int(a['act_tgm'][i][j])} stm={int(a['act_stm'][i][j])} "
                  f"lm>0={int(a['act_lm'][i][j] > 0)}"] += 1
                others = np.arange(len(ac)) != j
                n_vet = int(veto[others].sum())
                sel_cid = int(ac[j])
                sel_dem = int(a["act_is_demoted"][i][j])
                sel_fc = int(a["act_fc"][i][j])
                sel_len = float(a["act_length_cm"][i][j])
            else:
                n_vet, sel_cid, sel_dem, sel_fc, sel_len = int(veto.sum()), -1, -1, -1, -1.0
            cid = int(a["cluster_id"][i])
            rel = ("same" if cid == sel_cid else
                   "in_act_not_selected" if cid in ac else "not_in_act")
            C[f"cluster_id vs selected: {rel}"] += 1
            for v in a["act_lm"][i]:
                C[f"act_lm_value={int(v)}"] += 1
            for fl in ("tgm", "stm", "fc"):
                C[f"act_{fl}_ones"] += int((a[f"act_{fl}"][i] == 1).sum())
            C["activities"] += len(ac)
            nt = int(a["neutrino_type"][i])
            v = (float(a["nu_x"][i]), float(a["nu_y"][i]), float(a["nu_z"][i]))
            vdef = int(v == (0.0, 0.0, 0.0))
            best = None
            if truth and not vdef:
                best = min(truth, key=lambda tr: math.dist(v, (tr["vx"], tr["vy"], tr["vz"])))
                bd = math.dist(v, (best["vx"], best["vy"], best["vz"]))
                cand_vtx.append(v)
            row = dict(run=run, subrun=sub, event=evt, row=i, nu_index=int(a["nu_index"][i]),
                       gid=int(a["matched_flash_gid"][i]), cluster_id=cid,
                       sel_cluster_id=sel_cid, cid_relation=rel, n_act=len(ac),
                       sel_demoted=sel_dem, sel_fc=sel_fc, sel_length_cm=f"{sel_len:.1f}",
                       bundle_has_vetoed=int(n_vet > 0), n_vetoed=n_vet,
                       nu_x=f"{v[0]:.2f}", nu_y=f"{v[1]:.2f}", nu_z=f"{v[2]:.2f}",
                       vertex_default=vdef, reco_Enu=f"{float(k['kine_reco_Enu'][i]):.1f}",
                       numu_score=f"{float(a['numu_score'][i]):.4f}",
                       nue_score=f"{float(a['nue_score'][i]):.4f}", neutrino_type=nt,
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
                       t_dist_cm=f"{bd:.2f}" if best else -1,
                       mc_reco_Enu=reco[0]["Enu"] if reco else -1)
            cand_rows.append(row)
        C["candidate_rows"] += t.num_entries
        # the mc.json summary node reports the primary slot (nu_index 0)
        if reco:
            p = [r for r in cand_rows if r["nu_index"] == 0] or cand_rows
            ok = abs(float(p[0]["reco_Enu"]) - reco[0]["Enu"]) < 0.11
            C[f"mc_reco_Enu matches T_kine primary={int(ok)}"] += 1

    for tr in truth:
        d = min((math.dist(cv, (tr["vx"], tr["vy"], tr["vz"])) for cv in cand_vtx),
                default=-1.0)
        truth_rows.append(dict(run=run, subrun=sub, event=evt, idx=tr["idx"], flav=tr["flav"],
                               mode=tr["mode"], ccnc=tr["ccnc"], Etot=tr["Etot"],
                               Edep=tr["Edep"], T=tr["T"], vx=tr["vx"], vy=tr["vy"],
                               vz=tr["vz"], event_has_candidate=int(has_tagger),
                               n_candidates=len(cand_rows), min_cand_dist_cm=f"{d:.2f}"))
    return C, cand_rows, truth_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="production run dir (tracking-pr/, bee/)")
    ap.add_argument("--out", required=True, help="new output dir (must not exist)")
    ap.add_argument("--jobs", type=int, default=24)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"refusing to write into existing {args.out} (new run => new dir)")
    run_dir = os.path.realpath(args.run)
    tags = sorted(TAG_RE.search(os.path.basename(p)).group(0)
                  for p in glob.glob(os.path.join(run_dir, "tracking-pr", "tracking-pr_*.root")))
    total = collections.Counter()
    cands, truths = [], []
    with Pool(args.jobs) as pool:
        for C, cr, tr in pool.imap_unordered(one, [(run_dir, t) for t in tags], chunksize=20):
            total.update(C)
            cands += cr
            truths += tr
    total["events"] = len(tags)
    key = lambda r: (r["run"], r["subrun"], r["event"], r.get("row", r.get("idx")))
    cands.sort(key=key)
    truths.sort(key=key)
    os.makedirs(args.out)
    with open(os.path.join(args.out, "candidates.tsv"), "w") as f:
        f.write("\t".join(CAND_COLS) + "\n")
        for r in cands:
            f.write("\t".join(str(r[c]) for c in CAND_COLS) + "\n")
    with open(os.path.join(args.out, "truth.tsv"), "w") as f:
        f.write("\t".join(TRUTH_COLS) + "\n")
        for r in truths:
            f.write("\t".join(str(r[c]) for c in TRUTH_COLS) + "\n")
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(f"run_dir {run_dir}\n")
        for k, v in sorted(total.items()):
            f.write(f"{v:8d}  {k}\n")
    print(f"events {len(tags)} candidate rows {len(cands)} truth interactions {len(truths)}")


if __name__ == "__main__":
    main()
