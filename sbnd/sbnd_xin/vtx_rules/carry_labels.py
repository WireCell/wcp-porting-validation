#!/usr/bin/env python3
"""doc pr/82 sec 4.1 -- carry the 473 hand-scan labels onto a new arm.

THE ARGUMENT.  The owner's question this round was "pattern recognition has
advanced, so must we re-scan?".  The answer is no for ~95% of the set, and the
reason is a property of how a label is defined, not an empirical accident:

    vtx_io.load_labels() defines a label's truth as the rank-1 pick's (x,y,z)
    (vtx_io.py:79), and correct() as Euclidean distance <= TOL (vtx_io.py:86).

Neither definition mentions an arm.  The human answered the question "where is
the neutrino vertex in this event", and that answer is a point in the detector.
A reconstruction change moves the *graph*; it does not move the vertex.  So a
label carries forward whenever the new arm still puts a vertex at the point the
human clicked -- which is a measurement, made here, not an assumption.

WHY POSITION AND NOT vertex_id.  Joining on the old `vertex_id` is tempting and
is WRONG in both directions, measured on this exact data (doc pr/82 sec 4.1):

  * 7 of 473 events keep the point but renumber the id -- an id-join calls
    those "lost" and sends them to a human who has nothing to decide.
  * evt283040 keeps id 2000 while that id now names a point 117.5 cm away -- an
    id-join calls that "clean" and silently carries a label to the wrong place.
    That single event is the whole case against the id-join.

So: join on position, and report the id change as provenance rather than as a
verdict.  Every carried label records the old id, the new id, and the distance,
so an id-join audit remains possible after the fact.

WHAT IS AND IS NOT COPIED.  `truth` -- the human's answer -- is copied verbatim
and is the only thing that is.  Everything arm-dependent (`arm`, `source`,
`main_vertex`, `route`, the `dl_*` operating point, and every scoreboard column
on the pick) is RE-READ from the new dump, because a label that claimed the old
arm's route while pointing at the new arm's dump would corrupt every downstream
join.  This is the same field set pr_display_viewer.py's on_vscan_save writes,
built the same way, so a carried label and a hand-made one are indistinguishable
to every consumer.

M13.  Output goes to a FRESH tag and the script refuses to overwrite an existing
label file.  vertex_labels/vtxscan-prod0813*/ is a historical record of scans
taken against a now-deleted arm and is never modified.

TAG REGISTRATION -- deliberately NOT done.  The new tags are not added to
vtx_io.TAGS.  The carried labels cover the SAME events as the prod0813 tags, so
a default that held both would hand every unfiltered consumer (baselines.py,
selfscan.py score, build_dataset.py) ~922 labels with duplicate event keys and
quietly wrong denominators.  vtx_io.load_labels() already takes an explicit tag
list; callers that want this epoch pass vtx_io.TAGS_HARV3.

Repro:
  python3 vtx_rules/carry_labels.py --dry-run          # report only
  python3 vtx_rules/carry_labels.py --write            # write the fresh tags
  python3 vtx_rules/carry_labels.py --delta-list /home/xqian/tmp/pr82/delta.txt

doc pr/100 addition: --arms lets a later epoch carry a DIFFERENT tag set onto a
DIFFERENT arm set without editing this file.  Omitting it reproduces the pr/82
ARMS dict above byte-for-byte -- this is the closure test, not a courtesy.
`{sample}` in the arm or newtag half is resolved per-label from the label's own
`arm` field (substring match against nuecc48/ncpi0/mcp1k/mcp2k), because
vtxscan-harv3-delta is one tag spanning several samples' arms (pr/82 sec 4's
"the one tag spanning two arms" already forced this same resolution on
build_dataset.py's --harvest-roots):
  python3 vtx_rules/carry_labels.py --write \
      --arms vtxscan-harv3-nuecc48=work-vtx100-base-nuecc48:vtxscan-v100-nuecc48 \
             vtxscan-harv3-delta=work-vtx100-base-{sample}:vtxscan-v100-delta \
             vtxscan-mcp2k=work-vtx100-base-mcp2k:vtxscan-v100-mcp2k
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vtx_io  # noqa: E402

# old scan tag -> (new arm, new label tag).  The fourth production tag,
# vtxscan-prod0813-mc, is deliberately absent: its two arms (work-r1qlmc /
# work-r2mc-prod0813) are still live and were NOT re-run with harvest this
# round, so there is nothing to carry it onto.  Adding it here without an arm
# would silently drop its 8 labels into the "no dump" bucket.
ARMS = {
    "vtxscan-prod0813":       ("work-nuecc48-harv3", "vtxscan-harv3-nuecc48"),
    "vtxscan-prod0813-ncpi0": ("work-ncpi0-harv3",   "vtxscan-harv3-ncpi0"),
    "vtxscan-prod0813-mcp1k": ("work-mcp1k-harv3",   "vtxscan-harv3-mcp1k"),
}

SAMPLES = ("nuecc48", "ncpi0", "mcp1k", "mcp2k")


def parse_arms_arg(pairs):
    """['oldtag=newarm:newtag', ...] -> {oldtag: (newarm, newtag)}.

    Replaces ARMS wholesale (never merges with it) -- a later epoch's tag set
    is not a superset of pr/82's, and merging would let a stale default entry
    silently ride along into a new carry.
    """
    out = {}
    for p in pairs:
        oldtag, rest = p.split("=", 1)
        newarm, newtag = rest.split(":", 1)
        out[oldtag] = (newarm, newtag)
    return out


def resolve_sample(template, lab):
    """Fill a `{sample}` placeholder from the label's OWN `arm` field.

    Only vtxscan-harv3-delta needs this: doc pr/82 sec 4 built it as one tag
    spanning several samples' arms (nuecc48/mcp1k), the same fact that forced
    build_dataset.py's --harvest-roots to key on `@arm`.  Templates without
    `{sample}` pass through untouched -- every pr/82 default entry does.
    """
    if "{sample}" not in template:
        return template
    a = lab.get("arm") or ""
    for s in SAMPLES:
        if s in a:
            return template.format(sample=s)
    raise ValueError("cannot resolve {sample}: tag=%s evt=%s arm=%r matches none of %s"
                      % (lab["tag"], lab["eventNo"], a, SAMPLES))


def dump_path(arm, evt):
    return os.path.join(vtx_io.BASE, arm, "pr_evt%d" % evt,
                        "calib-pr-evt%d.json" % evt)


def find_row(dump, new_vid, old_vid, new_xyz, tol=vtx_io.TOL):
    """The scoreboard row describing this pick, and how it was found.

    Not a plain `rows[vertex_id]` lookup, because `improve_vertex` REFITS the
    chosen vertex after the reranker has scored it and can renumber it in the
    process (the pr/80 sec 11 F2 warning).  evt277276 is the worked example: the
    board scored vertex 8002, the final graph carries 8012 at the same place,
    and an id-only lookup finds nothing.  So try, in order:

      new-vid      the final vertex is itself a scored row      (372/449)
      old-vid      the row the ORIGINAL label was built from    (+5)
      nearest-row  a row within `tol` of the final position     (+9)

    and otherwise return None.  None is a real answer, not a lookup failure:
    63 of the 449 carried labels sit on a vertex the current arm's reranker
    never scored (55 of those in the MAIN cluster).  Writing a zero or a
    fabricated row there would turn an admission gap into apparent data.
    """
    rows = ((dump.get("vertex_scoreboard") or {}).get("rows") or [])
    byid = {r["vertex_id"]: r for r in rows}
    if new_vid in byid:
        return byid[new_vid], "new-vid"
    if old_vid is not None and old_vid in byid:
        return byid[old_vid], "old-vid"
    cand = [(vtx_io.dist(new_xyz, vtx_io.xyz(r)), r) for r in rows]
    cand = [(d, r) for d, r in cand if d is not None and d <= tol]
    if cand:
        return min(cand, key=lambda t: t[0])[1], "nearest-row"
    return None, "no-row"


def build_pick(vertex, row, dump, old_pick):
    """The rank-1 pick for the carried label, on the NEW arm.

    Mirrors pr_display_viewer.py's on_vscan_save so a carried label is
    structurally identical to a hand-made one.  `kind` is preserved from the
    old pick: a "manual" pick stays manual even though we can now name a
    vertex_id for it -- promoting it would overstate what the human did.
    """
    x, y, z = vtx_io.vertex_xyz(vertex)
    mv = vtx_io.xyz(dump.get("main_vertex"))
    pick = {
        "rank": 1,
        "kind": old_pick.get("kind", "candidate"),
        "vertex_id": vertex["id"],
        "cluster_id": vertex.get("cluster_id"),
        "x": x, "y": y, "z": z,
        "dis_to_main": vtx_io.dist((x, y, z), mv),
    }
    if pick["kind"] == "candidate":
        pick["degree"] = vertex.get("degree")
        pick["is_main"] = (vertex.get("cluster_id")
                           == (dump.get("main_vertex") or {}).get("cluster_id"))
        pick["main_candidate"] = pick["is_main"]
        if row is not None:
            # Only meaningful when the board actually scored this vertex --
            # copying a 0.0 default as though it were a score is how a tuning
            # fit silently learns from a placeholder.
            if row.get("dl_snapped"):
                pick["dl_score"] = row.get("dl_score")
                pick["snap_dis"] = row.get("snap_dis")
                pick["rerank_total"] = row.get("total")
            pick["trad_score"] = row.get("trad_score") if row.get("trad_scored") else None
            pick["dl_winner"] = row.get("dl_winner")
            pick["trad_winner"] = row.get("trad_winner")
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="actually write the carried labels (default: report only)")
    ap.add_argument("--dry-run", action="store_true", help="explicit no-op default")
    ap.add_argument("--tol", type=float, default=vtx_io.TOL)
    ap.add_argument("--tol-loose", type=float, default=vtx_io.TOL_LOOSE)
    ap.add_argument("--delta-list", default=None,
                    help="write the non-carried events' NEW-arm calib paths here, "
                         "worst first -- the input selfscan --dumps and the "
                         "pr_display viewer both want")
    ap.add_argument("--tsv", default=None, help="per-event result TSV")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="oldtag=newarm:newtag ... ; REPLACES the pr/82 ARMS "
                         "dict above (never merges).  Omit to reproduce the "
                         "pr/82 carry exactly.  newarm/newtag may contain "
                         "{sample}, resolved per-label from its own arm field.")
    args = ap.parse_args()

    arms_map = parse_arms_arg(args.arms) if args.arms else ARMS

    now = datetime.datetime.now(datetime.timezone.utc).replace(
        microsecond=0).isoformat()
    carried, delta, missing = [], [], []

    for lab in vtx_io.load_labels(tags=sorted(arms_map)):
        arm_t, newtag_t = arms_map[lab["tag"]]
        arm, newtag = resolve_sample(arm_t, lab), resolve_sample(newtag_t, lab)
        evt = lab["eventNo"]
        p = dump_path(arm, evt)
        if not os.path.isfile(p):
            missing.append((lab["tag"], evt))
            continue
        with open(p) as fh:
            dump = json.load(fh)

        cands = [(vtx_io.dist(lab["truth"], vtx_io.vertex_xyz(v)), v)
                 for v in dump.get("vertices", [])]
        cands = [(d, v) for d, v in cands if d is not None]
        if not cands:
            missing.append((lab["tag"], evt))
            continue
        dmin, vbest = min(cands, key=lambda t: t[0])

        rec = dict(tag=lab["tag"], evt=evt, arm=arm, newtag=newtag, dist=dmin,
                   old_vid=lab["truth_vid"], new_vid=vbest["id"], dump=p,
                   label=lab, vertex=vbest)
        if dmin <= args.tol:
            carried.append(rec)
        else:
            rec["bucket"] = "loose" if dmin <= args.tol_loose else "broken"
            delta.append(rec)

    # ---------------------------------------------------------------- report
    vid_changed = sum(1 for r in carried
                      if r["old_vid"] is not None and r["old_vid"] != r["new_vid"])
    exact0 = sum(1 for r in carried if r["dist"] == 0.0)
    print("labels considered : %d" % (len(carried) + len(delta) + len(missing)))
    print("carried (<=%.1f cm): %d   of which %d bit-identical, %d changed vertex_id"
          % (args.tol, len(carried), exact0, vid_changed))
    print("delta   (> %.1f cm): %d   (%d loose, %d broken)"
          % (args.tol, len(delta),
             sum(1 for r in delta if r["bucket"] == "loose"),
             sum(1 for r in delta if r["bucket"] == "broken")))
    if missing:
        print("!! no dump / no vertices: %d  %s" % (len(missing), missing[:10]))

    if args.tsv:
        with open(args.tsv, "w") as fh:
            fh.write("evt\ttag\tnewtag\tverdict\tdist_cm\told_vid\tnew_vid\n")
            for r in sorted(carried + delta, key=lambda r: (r["tag"], r["evt"])):
                fh.write("%d\t%s\t%s\t%s\t%.4f\t%s\t%s\n"
                         % (r["evt"], r["tag"], r["newtag"],
                            r.get("bucket", "carried"), r["dist"],
                            r["old_vid"], r["new_vid"]))

    if args.delta_list:
        with open(args.delta_list, "w") as fh:
            for r in sorted(delta, key=lambda r: -r["dist"]):
                fh.write("%s\n" % r["dump"])
        print("delta list -> %s (%d, worst first)" % (args.delta_list, len(delta)))

    if not args.write:
        print("\n(dry run -- nothing written; pass --write)")
        return 0

    # ----------------------------------------------------------------- write
    written = 0
    joins = {}
    for r in carried:
        outdir = os.path.join(vtx_io.LABELS_ROOT, r["newtag"])
        os.makedirs(outdir, exist_ok=True)
        outp = os.path.join(outdir, "labels-evt%d.json" % r["evt"])
        if os.path.exists(outp):
            # M13: a fresh tag should be empty; a collision means this ran
            # before or the tag is not fresh.  Refuse rather than merge.
            print("REFUSE (exists): %s" % outp)
            return 1

        lab, dump = r["label"], None
        with open(r["dump"]) as fh:
            dump = json.load(fh)
        sb = dump.get("vertex_scoreboard") or {}
        meta = dump.get("meta") or {}
        row, how = find_row(dump, r["new_vid"], lab["truth_vid"],
                            vtx_io.vertex_xyz(r["vertex"]))
        pick = build_pick(r["vertex"], row, dump, (lab.get("picks") or [{}])[0])
        joins[how] = joins.get(how, 0) + 1

        doc = {
            "arm": r["arm"],
            "confidence": lab.get("confidence"),
            "dl_best_score": sb.get("dl_best_score"),
            "dl_min_accept_score": sb.get("dl_min_accept_score"),
            "dl_score_scale": sb.get("dl_score_scale"),
            "event": "evt%d" % r["evt"],
            "eventNo": meta.get("eventNo", r["evt"]),
            # doc pr/100: carried straight through, not re-derived -- the
            # human/AI distinction describes who made the ORIGINAL pick, not
            # this carry.  Without this the pr/88 ai-scanner tag's 299 labels
            # silently read back as "human" (scn_vtx.io.load_label defaults
            # a missing key to 'human'), erasing the split entirely.
            "label_source": lab.get("label_source") or "human",
            "main_vertex": dump.get("main_vertex"),
            "not_a_candidate": lab.get("not_a_candidate", False),
            "picks": [pick],
            "route": sb.get("route"),
            "runNo": meta.get("runNo", lab.get("runNo")),
            "subRunNo": meta.get("subRunNo", lab.get("subRunNo")),
            "saved_utc": now,
            "scan_tag": r["newtag"],
            "scoreboard_present": bool(sb.get("filled")),
            "source": os.path.realpath(r["dump"]),
            # Provenance: everything needed to audit or undo this carry.
            "carried_from": {
                "tag": lab["tag"],
                "arm": lab.get("arm"),
                "path": lab["path"],
                "vertex_id": lab["truth_vid"],
                "truth": list(lab["truth"]),
                "saved_utc": lab.get("saved_utc"),
                "reanchor_dist_cm": r["dist"],
                "vertex_id_changed": (lab["truth_vid"] is not None
                                      and lab["truth_vid"] != r["new_vid"]),
                # How the pick's scoreboard columns were sourced -- "no-row"
                # means the current arm's reranker never scored this vertex, so
                # the dl_*/trad_* fields are absent rather than zero.
                "row_join": how,
                "carried_by": "vtx_rules/carry_labels.py (doc pr/82 sec 4.1)",
                "carried_utc": now,
            },
        }
        tmp = outp + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        os.replace(tmp, outp)
        written += 1

    print("\nwrote %d carried labels into %s"
          % (written, ", ".join(sorted({r["newtag"] for r in carried}))))
    print("scoreboard-row join: %s" % sorted(joins.items()))
    if joins.get("no-row"):
        print("  note: %d picks sit on a vertex the current reranker never "
              "scored -- dl_*/trad_* absent by design, not zeroed"
              % joins["no-row"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
