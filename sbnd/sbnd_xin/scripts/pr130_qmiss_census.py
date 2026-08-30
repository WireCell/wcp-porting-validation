#!/usr/bin/env python3
"""pr/130 item 4 -- mechanism census of the q_miss (under-clustering) pool.

The mirror of pr130_qextra_rank.py.  For every hand-marked shower row that
carries q_miss, ask of each MISSING segment: what actually happened to it?

Four mutually-exclusive classes, in decreasing order of "we can fix this":

  DECLINED   an absorber considered the segment and refused it -- the sidecar
             `absorb` tape carries a walk_exclude record.  This is OUR OWN
             guard family over-firing and is the only class a threshold can
             reach.
  SIBLING    the segment is held by the reconstructed shower that another
             LABELLED row in this event matched to, i.e. it is that row's
             `extra`.  Then the charge is DOUBLE-COUNTED across the two rows
             (A's miss and B's extra are the same charge) -- the pr/121
             ex1-dedup shape and the 98-set's 16.3%.  Same definition as
             pr130_qextra98.py:mirror_qmiss so the two docs reconcile.
  SPLIT      the segment is the ROOT of its own reconstructed shower and no
             larger object holds it.  The reco built the piece but never
             merged it into the target -- this is under-clustering proper,
             the class a merge-side knob could actually reach.
  STOLEN     some other, unlabelled reconstructed shower holds the segment.
             Mis-partition into an object nobody scanned.
  ADMITTED   an absorb record exists (direct / walk_add) but no reco shower
             holds the segment now -- absorbed then later shed.
  UNTOUCHED  no reco shower holds it and no absorb record names it.  NOTE the
             tape records ADMISSIONS and F12 excludes only -- it does not
             record a candidate weighed and dropped on distance or angle, so
             tape-absence proves "nothing admitted or F12-excluded it", NOT
             "nothing ever considered it".

For SPLIT and STOLEN the segment is already held by some reconstructed object,
so `q_miss` counts it as lost only relative to the LABELLED shower.  Whether it
is lost to the neutrino candidate is a different question, and the pr/128
precedent answers it on main-cluster membership (105074: the candidate's own
cluster is the candidate's energy, vertex reachability not required).  Two
columns carry that: `is_main_cluster` from the dump segment, and `held_conn`,
the holding shower's connectivity class (conn-4 = cluster >80 cm from the
candidate, NeutrinoShowerClustering.cxx:3733).

Rows whose own ROOT segment is in the miss list are reported separately as
REROOT: the reconstruction never built that shower and the scorer's
`--cross-run` matcher paired two different objects, so the row's q_miss is a
seeding failure that no admission-time predicate can address.

Charge weights come from em117_score itself (imported, not reimplemented) so
every total reconciles with pr130-{98,141}-score-prod.tsv by construction.

Repro:
  cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  scripts/pr130_qmiss_census.py > docs/pr/pr130-qmiss-census.txt

READ-ONLY over em_labels/, the calib dumps and emprep-pr130q{98,141}/ (M13).
"""
import csv
import os
import sys
from collections import defaultdict

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SX, "em_display"))
import em117_score as S                                    # noqa: E402

# owner-adjudicated rows: an event the owner has already ruled on cannot
# motivate a fresh hand-scan (pr130-qmiss-refresh.md "Adjudicated rows").
ADJUDICATED = {318769, 415278, 283515, 179369}

SETS = [
    ("98", "emscan-0827", "em117-pr130q98-manifest.tsv", "emprep-pr130q98"),
    ("141", "emscan-0828-agent5", "em114c-pr130q141-manifest.tsv", "emprep-pr130q141"),
]


def targets_of(rec):
    """eventNo -> {labelled shower id: target segment set}, the scorer's own
    (members | ins) - outs."""
    out = {}
    for shw, det in ((rec.get("em") or {}).get("marks_detail") or {}).items():
        marked = det.get("marked") or {}
        if not marked:
            continue
        members = set(int(x) for x in (det.get("members") or ()))
        ins = set(int(s) for s, m in marked.items() if m.get("kind") == "in")
        outs = set(int(s) for s, m in marked.items() if m.get("kind") == "out")
        out[int(shw)] = (members | ins) - outs
    return out


def census_set(label, tag, manifest, prepdir):
    man = S.load_manifest(os.path.join(SX, "em_display", manifest))
    labs = S.load_labels(tag)
    prep_root = os.path.join(SX, "em_display", prepdir)

    seg_rows = []      # one per (row, missing segment)
    reroot = []        # rows whose root segment is itself missing
    nomatch = []
    for ev in sorted(labs):
        m = man.get(ev)
        if not m or not m.get("dump"):
            continue
        dump = S.load_dump(m["dump"])
        if not dump:
            continue
        prep = S.load_prep(ev, prep_root)
        actual, seginfo, _ = S.digest_dump(dump, prep)
        # pr/128 precedent: main-cluster membership decides whether charge is
        # the candidate's, independent of which node holds it.
        mainc = {int(x["id"]): int(bool(x.get("is_main_cluster")))
                 for x in (dump.get("segments") or ())}
        conn = {int(k): v.get("conn") for k, v in ((prep or {}).get("showers") or {}).items()}

        holder = defaultdict(set)                  # seg -> reco showers holding it
        for node, segs in actual.items():
            for sid in segs:
                holder[sid].add(node)
        tape = (prep or {}).get("absorb") or {}
        tgts = targets_of(labs[ev])

        md = (labs[ev].get("em") or {}).get("marks_detail") or {}
        # pass 1: every labelled row's matched node, and what it holds as
        # `extra` -- the SIBLING test needs the whole event, so it cannot be
        # done inside the per-row loop.
        scored = {}
        for shw, det in sorted(md.items(), key=lambda kv: int(kv[0])):
            rr0 = S.score_shower(int(shw), det, actual, seginfo, cross_run=True)
            if rr0:
                scored[int(shw)] = rr0
        sib_extra = {}                     # seg -> [labelled shower ids]
        for shw0, r0 in scored.items():
            for sg in r0["extra"]:
                sib_extra.setdefault(sg, []).append(shw0)

        for shw, det in sorted(md.items(), key=lambda kv: int(kv[0])):
            r = scored.get(int(shw))
            if not r or r["q_miss"] <= 0:
                continue
            r["event"] = ev
            r["adj"] = ev in ADJUDICATED
            if r["matched"] < 0:
                nomatch.append(r)
                continue
            if int(shw) in set(r["miss"]):
                reroot.append(r)
                continue
            held_cls = set(seginfo.get(s, {}).get("cluster", -1)
                           for s in actual.get(r["matched"], ()))
            for sid in r["miss"]:
                info = seginfo.get(sid, {})
                recs = tape.get(str(sid)) or tape.get(sid) or []
                hold = sorted(holder.get(sid, ()))
                sib = sorted(o for o in sib_extra.get(sid, ())
                             if o != int(shw))
                tgt_sib = sorted(o for o, t in tgts.items()
                                 if o != int(shw) and sid in t)
                if any(x.get("how") == "walk_exclude" for x in recs):
                    cls = "DECLINED"
                elif sib:
                    cls = "SIBLING"
                elif hold == [sid]:
                    cls = "SPLIT"
                elif hold:
                    cls = "STOLEN"
                elif recs:
                    cls = "ADMITTED"
                else:
                    cls = "UNTOUCHED"
                # how big is the object that holds it, if any
                hn = hq = 0
                for nnode in hold:
                    segs = actual.get(nnode, ())
                    if len(segs) > hn:
                        hn = len(segs)
                        hq = sum(seginfo.get(x, {}).get("charge", 0.0) for x in segs)
                seg_rows.append(dict(
                    set=label, event=ev, shower=int(shw), matched=r["matched"],
                    seg=sid, cls=cls, adj=r["adj"],
                    q=info.get("charge", 0.0), length=info.get("length", 0.0),
                    is_main_cluster=mainc.get(sid, -1),
                    held_conn=",".join(str(conn.get(x)) for x in hold) or "",
                    pdg=info.get("pdg"), cluster=info.get("cluster", -1),
                    same_cluster=int(info.get("cluster", -1) in held_cls),
                    held_by=",".join(str(x) for x in hold),
                    held_nseg=hn, held_q=hq,
                    sib_labelled=",".join(str(x) for x in sib),
                    tgt_sib=",".join(str(x) for x in tgt_sib),
                    sites=",".join(sorted(set(x.get("site", "?") for x in recs))),
                    hows=",".join(sorted(set(x.get("how", "?") for x in recs))),
                ))
    return seg_rows, reroot, nomatch


def pct(a, b):
    return (100.0 * a / b) if b else 0.0


def main():
    all_rows = []
    print("=" * 78)
    print("pr/130 item 4 -- q_miss mechanism census")
    print("=" * 78)
    for label, tag, manifest, prepdir in SETS:
        seg_rows, reroot, nomatch = census_set(label, tag, manifest, prepdir)
        all_rows += seg_rows
        kept = [r for r in seg_rows if not r["adj"]]
        kr = [r for r in reroot if not r["adj"]]
        kn = [r for r in nomatch if not r["adj"]]
        q_seg = sum(r["q"] for r in kept)
        q_rr = sum(r["q_miss"] for r in kr)
        q_nm = sum(r["q_miss"] for r in kn)
        tot = q_seg + q_rr + q_nm
        print("\n--- set %s  (adjudicated events removed) ---" % label)
        print("kept q_miss total  %.4e   over %d segment(s) in %d addressable row(s)"
              % (tot, len(kept), len(set((r["event"], r["shower"]) for r in kept))))
        print("  %-11s %2d row(s)  %.4e  %5.1f%%   %s"
              % ("REROOT", len(kr), q_rr, pct(q_rr, tot),
                 ",".join(str(x["event"]) for x in kr)))
        if kn:
            print("  %-11s %2d row(s)  %.4e  %5.1f%%   %s"
                  % ("NOMATCH", len(kn), q_nm, pct(q_nm, tot),
                     ",".join(str(x["event"]) for x in kn)))
        by = defaultdict(list)
        for r in kept:
            by[r["cls"]].append(r)
        for cls in ("DECLINED", "SIBLING", "SPLIT", "STOLEN", "ADMITTED", "UNTOUCHED"):
            rr = by.get(cls, [])
            q = sum(x["q"] for x in rr)
            print("  %-11s %2d seg(s)  %.4e  %5.1f%%" % (cls, len(rr), q, pct(q, tot)))
        # geometry of the addressable classes
        for cls in ("SPLIT", "STOLEN", "UNTOUCHED"):
            rr = by.get(cls, [])
            if not rr:
                continue
            sc = [x for x in rr if x["same_cluster"]]
            print("      %-9s same-cluster-as-reco %d/%d seg, %.3e of %.3e"
                  % (cls, len(sc), len(rr), sum(x["q"] for x in sc), sum(x["q"] for x in rr)))
        # Is the charge lost to the CANDIDATE, or only to the labelled shower?
        frag = [x for x in kept if x["cls"] in ("SPLIT", "STOLEN", "UNTOUCHED")]
        if frag:
            qf = sum(x["q"] for x in frag)
            mc = [x for x in frag if x["is_main_cluster"] == 1]
            print("  donor location (pr/128 metric): of the %d fragment seg / %.3e"
                  % (len(frag), qf))
            print("      in the MAIN CLUSTER          %3d seg  %.4e  %5.1f%%"
                  % (len(mc), sum(x["q"] for x in mc),
                     pct(sum(x["q"] for x in mc), qf)))
            for c in ("1", "2", "3", "4", "None", ""):
                cc = [x for x in frag if x["held_conn"] == c]
                if cc:
                    print("      holder conn=%-4s            %3d seg  %.4e  %5.1f%%"
                          % (c or "(none)", len(cc), sum(x["q"] for x in cc),
                             pct(sum(x["q"] for x in cc), qf)))

    # ---- per-segment table -------------------------------------------------
    print("\n" + "=" * 78)
    print("per-segment detail (kept rows only, by charge)")
    print("=" * 78)
    hdr = ("set", "event", "shower", "matched", "seg", "cls", "q", "len",
           "pdg", "cls#", "same", "main", "conn", "held_by", "sites")
    print("%-4s %-8s %-7s %-8s %-7s %-11s %10s %6s %5s %5s %4s %4s %5s %-10s %s" % hdr)
    for r in sorted((x for x in all_rows if not x["adj"]),
                    key=lambda x: -x["q"]):
        print("%-4s %-8d %-7d %-8d %-7d %-11s %10.3e %6.1f %5s %5d %4d %4s %5s %-10s %s"
              % (r["set"], r["event"], r["shower"], r["matched"], r["seg"],
                 r["cls"], r["q"], r["length"], r["pdg"], r["cluster"],
                 r["same_cluster"], r["is_main_cluster"], r["held_conn"] or "-",
                 r["held_by"] or "-", r["sites"] or "-"))

    out = os.path.join(SX, "docs", "pr", "pr130-qmiss-census.tsv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print("\nwrote %s (%d rows, adjudicated included and flagged)" % (out, len(all_rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
