#!/usr/bin/env python3
"""doc pr/119 -- offline census of the byte-neutral membership probe.

Forked from scripts/pr118_probe_census.py.  Two structural changes:

  1. The unit is the (shower, cluster-group) from the EXPEL_GROUP probe lines,
     not the shower pair: pr/119 measures which member groups a shower
     wrongly HOLDS (the 29 hand-scan OUT marks), the mirror of pr/118's
     which fragments it wrongly LACKS.
  2. The truth join is charge-weighted cross-run matching (the
     em117_score.py match_shower recipe), not exact label-key equality:
     evt142421's label key 7010 matches reco node 108104 (43 members), so
     the pr/118-style exact join would return UNKNOWN on the single largest
     OUT-mark event.

Member truth classes (per (shower, member)):
  OUT     marked "out" of the matched scanned shower (wrongly held)
  IN      in the scanned shower's target = (scan members | ins) - outs
  OTHER   member of a matched scanned shower but in neither set (drift)
  HOLD    member of an unmatched shower in a scanned event (expel = churn)
  UNKNOWN unscanned event

Group rollup: OUT if out-charge fraction >= 0.5, IN if <= 0.05 with IN
charge present, MIXED otherwise (predicate failure -- reported prominently).

The evaluator models the Phase-B pass: non-anchor groups with
cluster != root_cluster, not touching the main vertex, len >= min_len, and
NOT retained by the pr/118-continuity test (T1 touching-aligned or T2
bright-aligned-stub at the FROZEN production numerics) -- the retention
guard is what protects the pr/118 merges (423981/281485/469665) from being
undone.

Repro:
  ./scripts/pr119_expel_census.py --prepdir em_display/emprep-119dbgA \
      --groups-tsv docs/pr/pr119-expel-groups.tsv \
      work-pr119r1-dbgA-mcp1k work-pr119r1-dbgA-mcp2k \
      work-pr119r1-dbgA-ncpi0 work-pr119r1-dbgA-nuecc48
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
LABEL_DIR = os.path.join(SX, "em_labels", "emscan-0827")

KV = re.compile(r"(\w+)=([^\s()]+)")


def parse_line(line):
    d = {}
    for k, v in KV.findall(line):
        v = v.rstrip("cm")
        try:
            d[k] = float(v) if ("." in v or "e" in v) else int(v)
        except ValueError:
            d[k] = v
    return d


def collect(roots):
    """event -> {'shower': [..], 'group': [..], 'member': [..]}"""
    out = {}
    for root in roots:
        for log in sorted(glob.glob(os.path.join(root, "pr_evt*", "stdout.log"))):
            ev = int(os.path.basename(os.path.dirname(log))[len("pr_evt"):])
            rec = out.setdefault(ev, {"shower": [], "group": [], "member": []})
            with open(log, errors="replace") as fh:
                for line in fh:
                    if line.startswith("EXPEL_SHOWER "):
                        rec["shower"].append(parse_line(line))
                    elif line.startswith("EXPEL_GROUP "):
                        rec["group"].append(parse_line(line))
                    elif line.startswith("EXPEL_MEMBER "):
                        rec["member"].append(parse_line(line))
    return out


def load_labels(ev):
    """-> None (unscanned) or {label_key: (target, ins, outs)} using
    marks_detail members when present (the score_shower target recipe)."""
    p = os.path.join(LABEL_DIR, "labels-evt%d.json" % ev)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        em = json.load(fh).get("em") or {}
    marks = em.get("marks_by_shower") or {}
    detail = em.get("marks_detail") or {}
    out = {}
    for shw, mm in marks.items():
        ins = {int(s) for s, v in mm.items() if v == "in"}
        outs = {int(s) for s, v in mm.items() if v == "out"}
        members = {int(x) for x in (detail.get(shw, {}).get("members") or ())}
        out[int(shw)] = ((members | ins) - outs, ins, outs)
    return out


def load_sidecar(prepdir, ev):
    """node -> {seg -> dQ} from the arm's own emprep sidecar."""
    p = os.path.join(prepdir, "emprep-evt%d.json" % ev)
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        prep = json.load(fh)
    out = {}
    for node, e in (prep.get("showers") or {}).items():
        out[int(node)] = {int(m["seg"]): float(m.get("dQ") or 0.0)
                          for m in (e.get("members") or ())}
    return out


def match_label_to_node(target, sidecar, label_key):
    """em117_score.match_shower recipe: reco node with the largest
    charge-weighted overlap with the label target (ties -> smaller node);
    zero overlap everywhere -> the exact key if present, else None."""
    best, best_q = None, 0.0
    for node in sorted(sidecar):
        inter = target & set(sidecar[node])
        if not inter:
            continue
        qi = sum(sidecar[node][s] for s in inter)
        if qi > best_q:
            best, best_q = node, qi
    if best is None:
        return label_key if label_key in sidecar else None
    return best


GRID_MIN_LEN = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)

# pr/118 shower_merge_relax_continuity production numerics (FROZEN -- these
# are already validated; the retention guard reuses them, adding nothing).
CONT_T1_GAP = 1.0
CONT_GAP = 8.0
CONT_AXIS = 7.5
CONT_FRAC = 1.0
CONT_QMED = 5000.0


def axis_of(g):
    aa = [a for a in (g.get("ax30_ang", -1), g.get("ax100_ang", -1)) if a >= 0]
    return min(aa) if aa else -1.0


def cont_retained(g):
    """Would the pr/118 continuity test claim this group as a legitimate
    aligned attachment?  T1 touching-aligned (any len) or T2
    bright-aligned-stub; nstep==0 (touching) counts qfrac as 1.0."""
    ax = axis_of(g)
    gap = g.get("gap_exact", -1)
    if gap < 0 or ax < 0:
        return False
    if gap <= CONT_T1_GAP and ax < CONT_AXIS:
        return True
    qfrac = 1.0 if g.get("nstep", 0) == 0 else g.get("conn_qfrac", -1)
    qmed = g.get("conn_qmed", -1)
    return (gap <= CONT_GAP and ax < CONT_AXIS
            and (qfrac >= CONT_FRAC or (g.get("nstep", 0) == 0))
            and (qmed > CONT_QMED or g.get("nstep", 0) == 0))


def fires(g, min_len, use_guard):
    """The Phase-B pass predicate on one probed group."""
    if g.get("anchor") == 1:
        return False
    if g.get("cluster") == g.get("root_cluster"):
        return False                       # same-cluster bridge: strand guard
    if g.get("touches_main_vtx") == 1:
        return False
    if g.get("len", 0.0) < min_len:
        return False
    if use_guard and cont_retained(g):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepdir", required=True,
                    help="the dbg arm's own sidecars (em_display/emprep-<tag>)")
    ap.add_argument("--groups-tsv", default=None)
    ap.add_argument("roots", nargs="+")
    args = ap.parse_args()

    roots = [r if os.path.isabs(r) else os.path.join(SX, r) for r in args.roots]
    for r in roots:
        if not os.path.isdir(r):
            sys.exit("no such arm root: %s" % r)
    prepdir = args.prepdir if os.path.isabs(args.prepdir) else os.path.join(SX, args.prepdir)

    events = collect(roots)
    print("events with probe lines: %d (showers %d, groups %d, members %d)\n"
          % (len(events),
             sum(len(r["shower"]) for r in events.values()),
             sum(len(r["group"]) for r in events.values()),
             sum(len(r["member"]) for r in events.values())))

    groups = []            # flat group rows with truth attached
    mem_cls_count = {"OUT": 0, "WRONGOWNER": 0, "IN": 0, "OTHER": 0, "HOLD": 0, "UNKNOWN": 0}
    out_events = set()
    for ev, rec in sorted(events.items()):
        labels = load_labels(ev)
        sidecar = load_sidecar(prepdir, ev)
        # matched reco node -> [(label, target, ins, outs), ...].  A LIST:
        # two scanned showers can match the same reco node (evt76346 labels
        # 14059 and 40030 both land on reco 14059 -- the wrong-owner merge
        # class), and a dict silently kept only the last label's marks.
        node_truth = {}
        if labels:
            for lk, (target, ins, outs) in sorted(labels.items()):
                node = match_label_to_node(target, sidecar, lk)
                if node is not None:
                    node_truth.setdefault(node, []).append((lk, target, ins, outs))
                if outs:
                    out_events.add(ev)
        # root_cluster + node_id per probed shower_id
        sh_info = {s["shower_id"]: s for s in rec["shower"]}
        # member truth per (shower_id, seg)
        members_by_grp = {}
        for m in rec["member"]:
            sid = m["shower_id"]
            node = sh_info.get(sid, {}).get("node_id", -1)
            if labels is None:
                cls = "UNKNOWN"
            elif node in node_truth:
                is_out = any(m["seg"] in outs for _, _, _, outs in node_truth[node])
                is_in = any(m["seg"] in target for _, target, _, _ in node_truth[node])
                if is_out and is_in:
                    cls = "WRONGOWNER"   # marked out of one scanned shower, in
                                         # another, both merged into this node:
                                         # the doc 118 sec 7 hardest class
                elif is_out:
                    cls = "OUT"
                elif is_in:
                    cls = "IN"
                else:
                    cls = "OTHER"
            else:
                cls = "HOLD"
            mem_cls_count[cls] += 1
            m["cls"] = cls
            members_by_grp.setdefault((sid, m["grp"]), []).append(m)
        for g in rec["group"]:
            g["event"] = ev
            sh = sh_info.get(g["shower_id"], {})
            g["root_cluster"] = sh.get("root_cluster", -2)
            g["node_id"] = sh.get("node_id", -1)
            g["sh_pdg"] = sh.get("pdg", 0)
            mem = members_by_grp.get((g["shower_id"], g["grp"]), [])
            q_out = sum(m["dQ"] for m in mem if m["cls"] == "OUT")
            q_wo = sum(m["dQ"] for m in mem if m["cls"] == "WRONGOWNER")
            q_in = sum(m["dQ"] for m in mem if m["cls"] in ("IN", "OTHER"))
            q_tot = sum(m["dQ"] for m in mem) or 1.0
            g["q_out"] = q_out
            g["q_wo"] = q_wo
            g["q_in"] = q_in
            g["out_qfrac"] = q_out / q_tot
            if labels is None:
                g["truth"] = "UNKNOWN"
            elif sh.get("node_id", -1) in node_truth or any(
                    m["cls"] in ("OUT", "IN", "OTHER") for m in mem):
                if q_wo / q_tot >= 0.5:
                    g["truth"] = "WRONGOWNER"
                elif g["out_qfrac"] >= 0.5:
                    g["truth"] = "OUT"
                elif g["out_qfrac"] <= 0.05 and q_in > 0:
                    g["truth"] = "IN"
                else:
                    g["truth"] = "MIXED"
            else:
                g["truth"] = "HOLD"
            g["out_segs"] = sorted(m["seg"] for m in mem if m["cls"] == "OUT")
            groups.append(g)

    print("member truth classes: %s" % mem_cls_count)
    print("events with OUT marks seen by the probe: %s\n" % sorted(out_events))

    if args.groups_tsv:
        cols = ["event", "truth", "out_qfrac", "q_out", "q_wo", "q_in", "shower_id",
                "node_id", "sh_pdg", "grp", "cluster", "root_cluster", "anchor",
                "nseg", "len", "dQ", "qfrac", "med_dqdx_mip", "max_seglen",
                "n_track_pid", "nsh_holding", "touches_main_vtx", "nlinks",
                "gap_exact", "nstep", "cont_frac", "conn_qmed", "conn_qfrac",
                "walked", "ax30_ang", "ax100_ang", "grp_ang", "dis_start",
                "dis_main", "anchor_vtx", "anchor_dis", "conn_new", "out_segs"]
        path = args.groups_tsv if os.path.isabs(args.groups_tsv) \
            else os.path.join(SX, args.groups_tsv)
        with open(path, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for g in groups:
                fh.write("\t".join(str(g.get(c, "")) for c in cols) + "\n")
        print("groups tsv: %s (%d rows)\n" % (path, len(groups)))

    # candidate universe = what the pass would even look at
    cand = [g for g in groups if g.get("anchor") == 0]
    foreign = [g for g in cand if g.get("cluster") != g.get("root_cluster")]
    bridge = [g for g in cand if g.get("cluster") == g.get("root_cluster")]
    print("== candidate universe ==")
    print("non-anchor groups: %d  (foreign-cluster %d, same-cluster bridge %d)"
          % (len(cand), len(foreign), len(bridge)))
    for cls in ("OUT", "WRONGOWNER", "MIXED", "IN", "HOLD", "UNKNOWN"):
        sel = [g for g in foreign if g["truth"] == cls]
        print("  foreign %-8s n=%d  (events %s)"
              % (cls, len(sel),
                 sorted({g["event"] for g in sel}) if cls in ("OUT", "WRONGOWNER", "MIXED") else ""))
    print("  bridge truth: %s" % {c: sum(1 for g in bridge if g["truth"] == c)
                                  for c in ("OUT", "WRONGOWNER", "MIXED", "IN", "HOLD", "UNKNOWN")})
    print()

    # total OUT charge visible to the probe, per event (coverage denominator)
    print("== OUT coverage by event (foreign groups only) ==")
    ev_out = {}
    for g in groups:
        ev_out.setdefault(g["event"], [0.0, 0.0])
        ev_out[g["event"]][0] += g.get("q_out", 0.0)
    for g in foreign:
        ev_out[g["event"]][1] += g.get("q_out", 0.0)
    for ev, (tot, infor) in sorted(ev_out.items()):
        if tot > 0:
            print("  evt%-8d q_out_total=%.0f in_foreign_groups=%.0f (%.0f%%)"
                  % (ev, tot, infor, 100.0 * infor / tot))
    print()

    print("== operating grid: expel foreign non-mv groups, len >= min_len ==")
    print("%-8s %-6s | fired OUT / WO / MIXED / IN / HOLD / UNKNOWN | OUT ev covered | IN+HOLD fired events" % ("min_len", "guard"))
    for use_guard in (True, False):
        for min_len in GRID_MIN_LEN:
            fired = [g for g in foreign if fires(g, min_len, use_guard)]
            cnt = {c: sum(1 for g in fired if g["truth"] == c)
                   for c in ("OUT", "WRONGOWNER", "MIXED", "IN", "HOLD", "UNKNOWN")}
            cov = sorted({g["event"] for g in fired if g["truth"] == "OUT"})
            churn = sorted({g["event"] for g in fired if g["truth"] in ("IN", "HOLD")})
            print("%-8.1f %-6s | %d / %d / %d / %d / %d / %d | %s | %s"
                  % (min_len, "on" if use_guard else "off",
                     cnt["OUT"], cnt["WRONGOWNER"], cnt["MIXED"], cnt["IN"], cnt["HOLD"],
                     cnt["UNKNOWN"], cov, churn))
    print()

    # retention check: the pr/118 merge products must be retained
    print("== pr/118 merge-product retention check (423981, 281485, 469665) ==")
    for ev in (423981, 281485, 469665):
        for g in foreign:
            if g["event"] != ev:
                continue
            print("  evt%-8d shower=%d grp=%d cluster=%d len=%.1f gap=%.2f axis=%.2f "
                  "qfrac=%.2f qmed=%.0f nstep=%d retained=%d truth=%s"
                  % (ev, g["shower_id"], g["grp"], g["cluster"], g.get("len", -1),
                     g.get("gap_exact", -1), axis_of(g),
                     g.get("conn_qfrac", -1), g.get("conn_qmed", -1),
                     g.get("nstep", -1), int(cont_retained(g)), g["truth"]))
    print()

    # the 54332 track-feature table (dQdx discriminator reserve)
    print("== OUT/MIXED group features (track-vs-gamma reserve) ==")
    for g in sorted(foreign, key=lambda g: (g["event"], g["shower_id"], g["grp"])):
        if g["truth"] not in ("OUT", "WRONGOWNER", "MIXED"):
            continue
        print("  evt%-8d shower=%d grp=%d cluster=%d nseg=%d len=%.1f q_out=%.0f "
              "out_qfrac=%.2f dqdx_mip=%.2f maxseg=%.1f ntrkpid=%d nlinks=%d "
              "gap=%.2f axis=%.2f grp_ang=%.1f dis_main=%.1f nsh=%d retained=%d out_segs=%s"
              % (g["event"], g["shower_id"], g["grp"], g["cluster"], g["nseg"],
                 g.get("len", -1), g.get("q_out", 0), g.get("out_qfrac", -1),
                 g.get("med_dqdx_mip", -1), g.get("max_seglen", -1),
                 g.get("n_track_pid", -1), g.get("nlinks", -1),
                 g.get("gap_exact", -1), axis_of(g), g.get("grp_ang", -1),
                 g.get("dis_main", -1), g.get("nsh_holding", -1),
                 int(cont_retained(g)), g.get("out_segs")))
    print()


if __name__ == "__main__":
    main()
