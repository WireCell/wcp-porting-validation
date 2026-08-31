#!/usr/bin/env python3
"""doc pr/122 -- recognition census over the SHOWER_SEED / shower_topo /
SHOWER_PID probe streams (both scan manifests).

Three questions, one per diagnosed case:

  (a) SBND 18255-54332 seg 16014: which segments become in_main_cluster
      shower ROOTS on the kShowerTopology flag alone, with what features
      (len, straightness, median dQ/dx, classifier branch), and does any
      feature cut separate the mis-flagged straight tracks from real EM
      stems without collateral on label-approved showers.
  (b) SBND 18255-166870 node 85045: multi-segment showers whose REPORTED
      pdg (the calculate_kinematics start-segment copy) is a muon -- the
      update_particle_type vote only ever writes 11, so trace VOTE/GUARD/
      COPY to name the write that leaves 13.
  (c) SBND 18255-235435: events where seeding aborts (no main vertex) or
      no shower is ever created.

Labels: marks/verdicts from em_labels/emscan-0827 (98-manifest, owner) and
em_labels/emscan-0828-agent5 (141-manifest, model); events are disjoint so
both dirs are tried per event.  Node matching per em117_score.match_shower
(charge-weighted), with membership charge taken from the arm's own
SHOWER_CONTENT probe lines (non-lossy, no sidecar needed).

Repro:
  ./scripts/pr122_recog_census.py --tsv docs/pr/pr122-seed-census.tsv \
      'work-pr121r1-dbgA-*' 'work-pr121r1-dbg141-*'
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
KV = re.compile(r"(\w+)=([^\s()]+)")
TOPO = re.compile(r"shower_topo dbg: seg (\d+) gidx (\d+) L ([\d.]+)cm .* "
                  r"max_spread ([\d.]+)cm .* lsl ([\d.]+)cm tel ([\d.]+)cm .* branch (\S+)")
LABEL_DIRS = [os.path.join(SX, "em_labels", t)
              for t in ("emscan-0827", "emscan-0828-agent5")]


def parse_kv(line):
    d = {}
    for k, v in KV.findall(line):
        v2 = v.rstrip("cm")
        try:
            d[k] = float(v2) if "." in v2 else int(v2)
        except ValueError:
            d[k] = v
    return d


def load_labels(ev):
    for ld in LABEL_DIRS:
        p = os.path.join(ld, "labels-evt%d.json" % ev)
        if os.path.exists(p):
            em = json.load(open(p)).get("em") or {}
            marks = em.get("marks_by_shower") or {}
            detail = em.get("marks_detail") or {}
            out = {}
            for shw, mm in marks.items():
                ins = {int(s) for s, v in mm.items() if v == "in"}
                outs = {int(s) for s, v in mm.items() if v == "out"}
                members = {int(x) for x in (detail.get(shw, {}).get("members") or ())}
                out[int(shw)] = ((members | ins) - outs, ins, outs)
            return {"verdict": em.get("verdict"), "marks": out, "tag": os.path.basename(ld)}
    return None


def collect(roots):
    out = {}
    for root in roots:
        for log in sorted(glob.glob(os.path.join(root, "pr_evt*", "stdout.log"))):
            ev = int(os.path.basename(os.path.dirname(log))[len("pr_evt"):])
            rec = out.setdefault(ev, {"arm": os.path.basename(root), "seeds": [],
                                      "aborts": 0, "topo": {}, "pid": {}, "content": {},
                                      "content_members": {}})
            cur_content = None
            for line in open(log, errors="replace"):
                if line.startswith("SHOWER_SEED site=in_main_cluster ABORT"):
                    rec["aborts"] += 1
                elif line.startswith("SHOWER_SEED site=in_main_cluster "):
                    rec["seeds"].append(parse_kv(line))
                elif "shower_topo dbg: seg " in line and " branch " in line:
                    m = TOPO.search(line)
                    if m:
                        sid = int(m.group(1))
                        rec["topo"][sid] = dict(L=float(m.group(3)), max_spread=float(m.group(4)),
                                                lsl=float(m.group(5)), tel=float(m.group(6)),
                                                branch=m.group(7))
                elif line.startswith("SHOWER_PID "):
                    d = parse_kv(line)
                    kind = line.split()[1]
                    sh = d.get("shower_id")
                    e = rec["pid"].setdefault(sh, {"mems": [], "votes": [], "copies": [],
                                                   "guards": []})
                    if kind == "VOTE_MEM":
                        e["mems"].append(d)
                    elif kind == "VOTE":
                        e["votes"].append(d)
                    elif kind == "GUARD":
                        e["guards"].append(d)
                    elif kind == "COPY":
                        e["copies"].append(d)
                elif line.startswith("SHOWER_CONTENT shower_id=") and "node_id=" in line:
                    d = parse_kv(line)
                    cur_content = d["node_id"]
                    rec["content"][cur_content] = d
                elif line.startswith("SHOWER_CONTENT   shower_id=") and " seg=" in line:
                    d = parse_kv(line)
                    if cur_content is not None:
                        rec["content_members"].setdefault(cur_content, {})[d["seg"]] = d.get("dQ", 0.0)
    return out


def match_node(target, members):
    best, best_q = None, 0.0
    for node in sorted(members):
        inter = target & set(members[node])
        if not inter:
            continue
        qi = sum(members[node][s] for s in inter)
        if qi > best_q:
            best, best_q = node, qi
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+")
    ap.add_argument("--tsv")
    ap.add_argument("--pid-tsv")
    args = ap.parse_args()
    data = collect([r for g in args.globs for r in sorted(glob.glob(g))])

    # ---- (a) seed census -------------------------------------------------
    seed_rows = []
    for ev in sorted(data):
        rec = data[ev]
        lab = load_labels(ev)
        # label node -> matched reco node (for GOOD/BAD classing of seeds)
        matched = {}
        if lab:
            for key, (target, ins, outs) in lab["marks"].items():
                node = match_node(target, rec["content_members"])
                if node is not None:
                    matched[node] = key
        for s in rec["seeds"]:
            seg = s.get("seg")
            row = dict(event=ev, arm=rec["arm"], **s)
            t = rec["topo"].get(seg % 1000 if seg >= 1000 else seg)
            if t:
                row.update({("topo_" + k): v for k, v in t.items()})
            row["verdict"] = lab["verdict"] if lab else ""
            row["label_tag"] = lab["tag"] if lab else ""
            # is this seed the start of a shower matching a labeled target?
            row["seed_node_labeled"] = ""
            for node, key in matched.items():
                if seg in rec["content_members"].get(node, {}):
                    row["seed_node_labeled"] = key
                    break
            # direct mark class of the seg
            cls = ""
            if lab:
                for key, (target, ins, outs) in lab["marks"].items():
                    if seg in outs:
                        cls = "OUT"
                    elif seg in target and not cls:
                        cls = "TARGET"
            row["seg_mark"] = cls
            seed_rows.append(row)

    topo_only = [r for r in seed_rows
                 if r.get("topo") == 1 and r.get("traj") == 0 and r.get("pdg11") == 0
                 and r.get("long_muon") == 0]
    print("SEED lines: %d accepted roots over %d events; topo-ONLY-admitted: %d"
          % (len(seed_rows), len(data), len(topo_only)))
    for r in sorted(topo_only, key=lambda r: -r.get("len_cm", 0)):
        print("  topo-only evt%(event)d seg=%(seg)s pdg=%(pdg)s len=%(len_cm)s "
              "straight=%(straight)s med_mip=%(med_dqdx_mip)s branch=%(topo_branch)s "
              "verdict=%(verdict)s mark=%(seg_mark)s labeled=%(seed_node_labeled)s"
              % {**{"topo_branch": ""}, **r})

    # ---- (b) reported-muon showers --------------------------------------
    pid_rows = []
    for ev in sorted(data):
        rec = data[ev]
        for sh, e in rec["pid"].items():
            if not e["copies"]:
                continue
            last = e["copies"][-1]
            if abs(last.get("pdg", 0)) != 13 or last.get("nseg", 1) <= 1:
                continue
            n_track = sum(1 for m in e["mems"] if m.get("counts_track"))
            n_mem = len({m.get("seg") for m in e["mems"]})
            fired = e["votes"][-1].get("fired") if e["votes"] else ""
            pid_rows.append(dict(event=ev, arm=rec["arm"], shower_id=sh,
                                 nseg=last.get("nseg"), start_seg=last.get("start_seg"),
                                 vote_fired=fired, n_mem_lines=n_mem, n_counts_track=n_track,
                                 vote_skips=sum(1 for v in e["votes"] if not v.get("fired"))))
    print("\nreported-mu multi-seg showers: %d" % len(pid_rows))
    for r in pid_rows:
        print("  mu-shower evt%(event)d shower_id=%(shower_id)s nseg=%(nseg)s "
              "start_seg=%(start_seg)s vote_fired=%(vote_fired)s "
              "counts_track=%(n_counts_track)s/%(n_mem_lines)s" % r)

    # ---- (c) assembly ----------------------------------------------------
    for ev in sorted(data):
        rec = data[ev]
        if rec["aborts"] or not rec["content"]:
            print("ASSEMBLY evt%d aborts=%d final_showers=%d" %
                  (ev, rec["aborts"], len(rec["content"])))

    def dump(rows, path):
        if not path or not rows:
            return
        keys = sorted({k for r in rows for k in r})
        with open(path, "w") as fh:
            fh.write("\t".join(keys) + "\n")
            for r in rows:
                fh.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
        print("wrote %s (%d rows)" % (path, len(rows)))
    dump(seed_rows, args.tsv)
    dump(pid_rows, args.pid_tsv)


if __name__ == "__main__":
    main()
