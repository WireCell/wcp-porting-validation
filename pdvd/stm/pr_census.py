#!/usr/bin/env python3
"""Firing census of the PDVD PR chain (doc pdvd/25 M3).

Per event (work/<RUN6>_<idx>_<tag>): from wct_pr_<RUN6>_<idx>.log the
TGM/STM/FC verdict lines (one per evaluated main cluster), the STM pre-fit
exits and guard lines; from calib-pr-evt<ID>.json the per-bundle candidates
(nu_per_bundle) with their cosmic-tagger flags 6-8 (stopped-muon + Michel
tests) and the shower at the main vertex.

Log formats (TaggerCheck{TGM,STM,FC}.cxx):
  visit: TaggerCheckTGM: cluster N -> TGM=true|false
  visit: TaggerCheckSTM: cluster N -> STM=1|0 TGM=1|0
  visit: TaggerCheckFC:  cluster N -> FC=true|false
(spdlog may split a line; we anchor on the tagger token.)

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 stm/pr_census.py --tag stm1                # table to stdout + stm/census_<tag>.tsv
  python3 stm/pr_census.py --tag m1on --events 039252_0
"""
import argparse
import collections
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.dirname(HERE)
RE_TGM = re.compile(r"TaggerCheckTGM: cluster (\d+) [^A-Za-z]*TGM=(true|false)")
RE_STM = re.compile(r"TaggerCheckSTM: cluster (\d+) [^A-Za-z]*STM=(\d) TGM=(\d)")
RE_FC = re.compile(r"TaggerCheckFC: cluster (\d+) [^A-Za-z]*FC=(true|false|\d)")
RE_NOFIT = re.compile(r"cluster (\d+) no STM fit: (.*)")
RE_GUARD = re.compile(r"(guard[a-z_]*|entry_rise|vertex_hadron|descent|deficit|second_track|cathode_guard|proton_muon|accept_guard)[^\n]{0,80}", re.I)


def parse_log(path):
    v = collections.defaultdict(dict)
    nofit = collections.Counter(); guards = collections.Counter()
    if not os.path.exists(path):
        return v, nofit, guards
    for line in open(path, errors="replace"):
        if "TaggerCheck" in line or "no STM fit" in line or "guard" in line:
            m = RE_TGM.search(line)
            if m: v[int(m.group(1))]["tgm_stage"] = int(m.group(2) == "true"); continue
            m = RE_STM.search(line)
            if m: v[int(m.group(1))].update(stm=int(m.group(2)), tgm=int(m.group(3))); continue
            m = RE_FC.search(line)
            if m: v[int(m.group(1))]["fc"] = int(m.group(2) in ("true", "1")); continue
            m = RE_NOFIT.search(line)
            if m: nofit[m.group(2).strip()[:60]] += 1; continue
            if "guard" in line and "TaggerCheckSTM" in line:
                m = RE_GUARD.search(line)
                if m: guards[m.group(0)[:50]] += 1
    return v, nofit, guards


def michel_rows(dump):
    """One row per candidate (bundle) of a calib-pr dump: the cosmic-tagger
    flags 6-8 and the highest-energy shower starting at the main vertex."""
    rows = []
    cands = dump.get("candidates") or [dict(dump, nu_index=-1)]
    for c in cands:
        tg = c.get("tagger", {}) or {}
        mv = c.get("main_vertex") or {}
        vids = [vv["id"] for vv in (c.get("vertices") or dump.get("vertices") or []) if vv.get("is_main")]
        showers = c.get("showers") or []
        at_main = [s for s in showers if vids and s.get("start_vertex_id") in vids]
        best = max(at_main, key=lambda s: (s.get("kine_best") or 0.0)) if at_main else None
        rows.append(dict(nu_index=c.get("nu_index", -1), main_cluster=mv.get("cluster_id"),
                         mvx=mv.get("x"), mvy=mv.get("y"), mvz=mv.get("z"),
                         f6=tg.get("cosmict_flag_6"), f7=tg.get("cosmict_flag_7"), f8=tg.get("cosmict_flag_8"),
                         f6_filled=tg.get("cosmict_6_filled"), f7_filled=tg.get("cosmict_7_filled"), f8_filled=tg.get("cosmict_8_filled"),
                         cosmict=tg.get("cosmict_flag"), isFC=tg.get("match_isFC"),
                         n_showers=len(showers), n_at_main=len(at_main),
                         michel_E=(best or {}).get("kine_best"), michel_id=(best or {}).get("id"),
                         michel_len=(best or {}).get("total_length"), michel_pdg=(best or {}).get("particle_id"),
                         kine_Enu=(c.get("kine") or {}).get("kine_reco_Enu")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stm1")
    ap.add_argument("--events", help="comma list of <RUN6>_<idx>")
    ap.add_argument("--out")
    args = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(PDVD, "work", f"*_{args.tag}")))
    if args.events:
        want = set(args.events.split(","))
        dirs = [d for d in dirs if re.sub(r"_%s$" % args.tag, "", os.path.basename(d)) in want]
    out = args.out or os.path.join(HERE, f"census_{args.tag}.tsv")
    tot = collections.Counter(); allnofit = collections.Counter(); allguards = collections.Counter()
    with open(out, "w") as fh:
        fh.write("event\tevtno\tn_eval\tn_tgm\tn_stm\tn_fc\tn_cand\tn_f7\tn_f8\tn_f7_filled\tn_michel_E\tmichel_E_list\n")
        for d in dirs:
            ev = re.sub(r"_%s$" % args.tag, "", os.path.basename(d))
            run, idx = ev.split("_")
            log = os.path.join(d, f"wct_pr_{run}_{idx}.log")
            if not os.path.exists(log):
                continue
            v, nofit, guards = parse_log(log)
            allnofit.update(nofit); allguards.update(guards)
            n_eval = len(v); n_tgm = sum(1 for r in v.values() if r.get("tgm") == 1 or r.get("tgm_stage") == 1)
            n_stm = sum(1 for r in v.values() if r.get("stm") == 1); n_fc = sum(1 for r in v.values() if r.get("fc") == 1)
            dumps = glob.glob(os.path.join(d, "calib-pr-evt*.json"))
            rows = []; evtno = ""
            if dumps:
                dump = json.load(open(dumps[0])); evtno = dump.get("meta", {}).get("eventNo", "")
                rows = michel_rows(dump)
            n_cand = len(rows); n_f7 = sum(1 for r in rows if r["f7"]); n_f8 = sum(1 for r in rows if r["f8"])
            n_f7f = sum(1 for r in rows if r["f7_filled"]); mE = [round(r["michel_E"], 1) for r in rows if r["michel_E"] is not None]
            fh.write(f"{ev}\t{evtno}\t{n_eval}\t{n_tgm}\t{n_stm}\t{n_fc}\t{n_cand}\t{n_f7}\t{n_f8}\t{n_f7f}\t{len(mE)}\t{','.join(map(str, mE))}\n")
            tot.update(events=1, n_eval=n_eval, n_tgm=n_tgm, n_stm=n_stm, n_fc=n_fc, n_cand=n_cand, n_f7=n_f7, n_f8=n_f8, n_f7_filled=n_f7f, n_michel=len(mE))
    print("totals:", dict(tot))
    print("STM pre-fit exits:", dict(allnofit.most_common(12)))
    print("guard lines:", dict(allguards.most_common(12)))
    print("wrote", out)


if __name__ == "__main__":
    main()
