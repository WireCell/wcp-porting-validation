#!/usr/bin/env python3
"""doc pr/136 #1 -- WHY can no existing seat reach the missing charge?

THE QUESTION THIS ANSWERS.  pr/130 item 4 part 10 left one live lead: four of
seven target showers emit NO `SHOWER_XCLUS` lines at all while carrying 79% of
the missing charge, i.e. they are never enumerated as cross-cluster absorbers.
The charter (doc pr/136 sec 7 #1) ranks "instrument the enumeration itself" as
the next step.  This is that instrument, at population scale: every hand-marked
shower, every segment the scanner says it should hold and does not, classified
by WHICH mechanism made the segment unreachable.

THE VERDICT CASCADE.  EVIDENCE FIRST: if the tape carries a line for the pair,
that line IS the verdict -- no inference can overrule the shower's own record.
Only when the tape is silent do we infer why it never got there:

  OWNED              the tape says another shower already held the segment when
                     the loop ran, so it was dropped before any geometry was
                     computed (NeutrinoShowerClustering.cxx:2411, :3877).
  REJECT             the tape says the shower computed the geometry and the cone
                     refused -- a threshold question, and the line carries the
                     miss margin.  THIS PLUS `OWNED` IS THE WHOLE OF WHAT ANY
                     PREDICATE CHANGE AT AN EXISTING SEAT CAN REACH.
  --- the tape is silent below here ---
  SAME_CLUSTER       the segment is in the matched shower's own cluster.  Every
                     cross-cluster cone begins
                     `if (seg1->cluster() == shower->start_segment()->cluster())
                     continue;` (:2412, :3888), so no cross-cluster seat is even
                     the right question -- growth here is the graph walk's job
                     (pr/130 part 8's BLOCKED domain).
  MAIN_CLUSTER_SKIP  the segment is in the main cluster and the shower is not.
                     Both cone seats skip main-cluster segments at :2405 / :3871.
                     NOTE `shower_absorb_unreachable_main = true` in SBND
                     production (wct-pr-perevt.jsonnet:2747), so the escape set
                     m_absorb_unreachable_main_segs already exempts the
                     graph-UNREACHABLE ones; what lands here is a main-cluster
                     segment the main-vertex walk did reach and therefore claims.
  NO_SEAT            the matched shower never ran a cross-cluster cone loop AT
                     ALL.  Only two seats have one -- pass 4 of
                     shower_clustering_with_nv_from_vertices (:2396) and sub-pass
                     A of shower_clustering_in_other_clusters (:3868).  A shower
                     built anywhere else (in_main_cluster, connecting_to_main_vertex,
                     in_other_clusters_B, examine_shower_1, conn3_unreachable,
                     examine_showers_retarget) never enumerates one cross-cluster
                     candidate.  THIS IS STRUCTURAL: no threshold, ordering rule
                     or tie-break at any existing seat can reach these segments.
  OWNED              had a seat; the tape says another shower already held the
                     segment when the loop ran, so it was dropped before any
                     geometry (:2411, :3877).
  REJECT             had a seat, computed the geometry, and the cone refused --
                     a threshold question, and the tape carries the miss margin.
  ABSENT             had a seat and left no tape line for the pair.  Residual:
                     read it as "the enumeration is narrower than the cascade
                     above predicts" and chase it by hand.

JOINING THE TAPE TO THE FINAL SHOWERS.  A tape line's `shower=` is the start
segment's display id AT THE TIME OF THE PASS.  A shower that re-roots later
(examine_showers_retarget) changes id, so the naive key join silently loses
lines.  Both joins are computed: the tape id is also mapped through
"which shower finally owns that start segment", and the disagreement rate is
printed rather than hidden.

READ-ONLY over em_labels/ (M13), the dumps and the arm logs.

    scripts/pr136_xclus_enum.py --manifest98 em117-136f086probe98-manifest.tsv \\
        --prepdir98 emprep-136f086 --arm 'work-pr136-f086probe-*' \\
        --tsv docs/pr/pr136-xclus-enum.tsv
"""
import argparse, collections, csv, glob, importlib.util, json, os, re, sys

SD = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(SD)
ED = os.path.join(SX, "em_display")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load("em117_score", os.path.join(ED, "em117_score.py"))

RE_OWN = re.compile(r"SHOWER_XCLUS OWNED site=(\S+) shower=(-?\d+) seg=(-?\d+) owner=(-?\d+)")
RE_REJ = re.compile(r"SHOWER_XCLUS REJECT site=(\S+) shower=(-?\d+) seg=(-?\d+) "
                    r"angle_v1=([-\d.]+) angle_v2=([-\d.]+) pair_dis_cm=([-\d.]+)"
                    r"(?: tmp_dis_cm=([-\d.]+) close_dis_cm=([-\d.]+))?")

# the only two seats with a cross-cluster candidate loop
CONE_SEATS = {"from_vertices", "in_other_clusters_A"}
ORDER = ["OWNED", "REJECT", "SAME_CLUSTER", "MAIN_CLUSTER_SKIP", "NO_SEAT", "ABSENT"]


def read_tape(logs):
    own, rej = {}, {}
    for log in logs:
        try:
            fh = open(log, errors="replace")
        except OSError:
            continue
        with fh:
            for ln in fh:
                if "SHOWER_XCLUS" not in ln:
                    continue
                m = RE_OWN.search(ln)
                if m:
                    own[(int(m.group(2)), int(m.group(3)))] = (m.group(1), int(m.group(4)))
                    continue
                m = RE_REJ.search(ln)
                if m:
                    rej.setdefault((int(m.group(2)), int(m.group(3))), []).append(m.groups())
    return own, rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest98", default="em117-136f086probe98-manifest.tsv")
    ap.add_argument("--manifest141", default="em114c-136f086probe141-manifest.tsv")
    ap.add_argument("--tag98", default="emscan-0827")
    ap.add_argument("--tag141", default="emscan-0828-agent5")
    ap.add_argument("--prepdir", default="emprep-136f086")
    ap.add_argument("--arm", default="work-pr136-f086probe-*",
                    help="glob of arm roots holding the stdout.log probe tape")
    ap.add_argument("--tsv", default="docs/pr/pr136-xclus-enum.tsv")
    ap.add_argument("--expanded-join", action="store_true",
                    help="attribute a predecessor shower id's tape line to the "
                         "final shower (see the note at the joinfix branch)")
    a = ap.parse_args()

    prep_dir = a.prepdir if os.path.isabs(a.prepdir) else os.path.join(ED, a.prepdir)
    arm_roots = sorted(glob.glob(os.path.join(SX, a.arm)))

    rows = []
    n_ev = n_shw = 0
    joinmiss = joinfix = 0
    for man, tag, setname in ((a.manifest98, a.tag98, "98"),
                              (a.manifest141, a.tag141, "141")):
        mp = man if os.path.isabs(man) else os.path.join(ED, man)
        if not os.path.exists(mp):
            print("[warn] no manifest %s -- set skipped" % mp); continue
        labels = S.load_labels(tag)
        for ev, mrow in sorted(S.load_manifest(mp).items()):
            rec = labels.get(ev)
            if not rec or not ((rec.get("em") or {}).get("marks_detail")):
                continue
            dpath = mrow["dump"] if os.path.isabs(mrow["dump"]) else os.path.join(SX, mrow["dump"])
            if not os.path.exists(dpath):
                continue
            dump = S.load_dump(dpath)
            prep = S.load_prep(ev, prep_dir)
            actual, seginfo, _ = S.digest_dump(dump, prep)
            _, srows = S.score_event(rec, dump, prep, cross_run=True)
            if not srows:
                continue
            n_ev += 1

            # segment -> is_main_cluster and cluster, straight from the dump
            main_seg, seg_cluster = {}, {}
            for s in dump.get("segments") or ():
                main_seg[int(s["id"])] = bool(s.get("is_main_cluster"))
                seg_cluster[int(s["id"])] = int(s["cluster_id"])
            # final owner of every segment
            owner_of = {}
            for node, segs in actual.items():
                for sid in segs:
                    owner_of[sid] = node
            # walk sites per shower node
            sites = collections.defaultdict(set)
            for w in ((prep or {}).get("walks") or ()):
                sites[int(w["shower"])].add(w["site"])

            own, rej = read_tape([os.path.join(r, "pr_evt%d" % ev, "stdout.log")
                                  for r in arm_roots])
            # tape id -> final shower node (a re-rooted shower renamed itself)
            def tape_keys(node):
                ks = {node}
                for tid in {k[0] for k in own} | {k[0] for k in rej}:
                    if owner_of.get(tid) == node:
                        ks.add(tid)
                return ks

            for r in srows:
                n_shw += 1
                node = r["matched"]
                if node < 0:
                    continue
                # STRICT join: the tape's `shower=` is the start segment at PASS
                # time, `node` is the start segment at END time.  Use the strict
                # pair as the verdict and report separately how many rows the
                # re-root map would have added -- an id-remap must never be able
                # to credit shower A's evaluation to shower B silently.
                keys = tape_keys(node)
                had_seat = bool(sites.get(node, set()) & CONE_SEATS)
                own_cluster = seg_cluster.get(node, -999)   # node IS a segment id
                for sid in r["miss"]:
                    q = seginfo.get(sid, {}).get("charge", 0.0)
                    o = own.get((node, sid))
                    j = rej.get((node, sid))
                    if not o and not j and any((k, sid) in own or (k, sid) in rej
                                               for k in keys if k != node):
                        joinfix += 1
                        # --expanded-join: credit a PREDECESSOR id's evaluation to
                        # the final shower.  Off by default because an id remap
                        # must not silently attribute shower A's decision to B;
                        # on, it bounds how much of NO_SEAT is really "a
                        # predecessor looked and refused".
                        if a.expanded_join:
                            o = next((own[(k, sid)] for k in keys if (k, sid) in own), None)
                            j = next((rej[(k, sid)] for k in keys if (k, sid) in rej), None)
                    if o:
                        v = "OWNED"
                    elif j:
                        v = "REJECT"
                    elif seg_cluster.get(sid, -1) == own_cluster:
                        v = "SAME_CLUSTER"
                    elif main_seg.get(sid) and not main_seg.get(node, False):
                        v = "MAIN_CLUSTER_SKIP"
                    elif not had_seat:
                        v = "NO_SEAT"
                    else:
                        v = "ABSENT"
                    if v == "ABSENT":
                        joinmiss += 1
                    rows.append(dict(
                        set=setname, sample=mrow.get("sample", ""), event=ev,
                        shower=r["shower"], matched=node,
                        sites="|".join(sorted(sites.get(node, set()))) or "-",
                        had_cone_seat=int(had_seat),
                        conn=next((sh.get("start_connection_type") for sh in
                                   (dump.get("showers") or ()) if int(sh["id"]) == node), -1),
                        seg=sid, seg_cluster=seg_cluster.get(sid, -1),
                        seg_is_main=int(bool(main_seg.get(sid))),
                        q=round(q, 1), verdict=v,
                        cur_owner=owner_of.get(sid, -1),
                        detail=("owner=%d site=%s" % (o[1], o[0])) if o else
                               ("site=%s angle_v2=%s dis=%s" % (j[0][0], j[0][4], j[0][5])) if j else ""))

    # ------------------------------------------------------------- report
    print("WHY THE MISSING CHARGE IS UNREACHABLE  (doc pr/136 #1)")
    print("  events with marks %d;  scored showers %d;  missed segments %d"
          % (n_ev, n_shw, len(rows)))
    print("  prepdir %s;  arms %s" % (a.prepdir, a.arm))
    qtot = sum(r["q"] for r in rows) or 1.0
    print("\n%-18s %6s %7s %8s %8s" % ("verdict", "n_seg", "n_shw", "q", "q_share"))
    byv = collections.defaultdict(list)
    for r in rows:
        byv[r["verdict"]].append(r)
    for v in ORDER:
        rs = byv.get(v) or []
        if not rs:
            continue
        print("%-18s %6d %7d %8.3e %7.1f%%"
              % (v, len(rs), len({(x["event"], x["matched"]) for x in rs}),
                 sum(x["q"] for x in rs), 100.0 * sum(x["q"] for x in rs) / qtot))
    print("%-18s %6d %7d %8.3e %7.1f%%" % ("TOTAL", len(rows),
          len({(x["event"], x["matched"]) for x in rows}), qtot, 100.0))

    print("\nNO_SEAT broken down by the seat the shower WAS built at")
    bys = collections.Counter()
    qs = collections.Counter()
    for r in byv.get("NO_SEAT") or ():
        bys[r["sites"]] += 1
        qs[r["sites"]] += r["q"]
    for k, n in bys.most_common():
        print("  %-52s %4d seg  %8.3e  (%.1f%% of all q_miss)"
              % (k, n, qs[k], 100.0 * qs[k] / qtot))

    print("\nby the matched shower's start_connection_type")
    bc, qc = collections.Counter(), collections.Counter()
    for r in rows:
        bc[(r["conn"], r["verdict"])] += 1
        qc[(r["conn"], r["verdict"])] += r["q"]
    for conn in sorted({k[0] for k in bc}):
        tot = sum(qc[k] for k in qc if k[0] == conn)
        parts = ", ".join("%s %.0f%%" % (v, 100.0 * qc[(conn, v)] / tot)
                          for v in ORDER if qc.get((conn, v)))
        print("  conn=%-3d q=%8.3e (%4.1f%% of all)   %s"
              % (conn, tot, 100.0 * tot / qtot, parts))

    print("\ntape-join health: %d rows where the STRICT (node,seg) join is silent but a\n"
          "  predecessor id of the same final shower has a tape line (NOT counted as\n"
          "  evidence above); %d ABSENT rows -- a seat with no tape line either way."
          % (joinfix, joinmiss))

    print("\nTOP 15 missed segments by charge")
    print("  %-8s %-8s %-8s %-9s %-18s %s"
          % ("event", "shower", "seg", "q", "verdict", "detail/sites"))
    for r in sorted(rows, key=lambda x: -x["q"])[:15]:
        print("  %-8d %-8d %-8d %-9.3e %-18s %s"
              % (r["event"], r["matched"], r["seg"], r["q"], r["verdict"],
                 r["detail"] or r["sites"]))

    if rows:
        o = a.tsv if os.path.isabs(a.tsv) else os.path.join(SX, a.tsv)
        with open(o, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("\nwrote %s (%d rows)" % (o, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
