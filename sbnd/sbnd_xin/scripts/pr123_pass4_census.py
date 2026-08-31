#!/usr/bin/env python3
"""doc pr/123 -- pass4_angle over-reach census over the PASS4_GEOM probe
stream (fork of pr122_recog_census.py: same label loading, same
charge-weighted node matching from SHOWER_CONTENT; new stream, new question).

The owner's over-reach line (decision 2026-08-28): a pass4_angle-accepted
member counts as over-reach when EITHER it is detached from the contiguous
shower body (gap: closest approach to the rest of the shower over ~25 cm) OR
it is track-like (pdg 13/211/2212 or MIP-flat dQ/dx) beyond the body.

For every SHOWER_ABSORB PASS4_GEOM line (one per accepted segment, with the
at-absorb-time geometry: pair_dis to the start vertex, front_dis to the start
segment front, body_dis to the shower body, angle_v1/v2, disjunct tier), this
script joins the segment against the labels (OUT mark = labeled-bad absorb,
TARGET = labeled-good member, else unlabeled) and reports:

  1. the labeled scatter (body_dis x track-likeness per class),
  2. a guard sweep: for candidate rules, how many OUT / TARGET / unlabeled
     absorbs each would decline (kill the bad, count the collateral).

Repro:
  ./scripts/pr123_pass4_census.py --tsv docs/pr/pr123-pass4-census.tsv \
      'work-pr123r1-dbgA-*' 'work-pr123r1-dbg141-*'
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
LABEL_DIRS = [os.path.join(SX, "em_labels", t)
              for t in ("emscan-0827", "emscan-0828-agent5")]
TRACK_PDG = {13, 211, 2212}


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
            rec = out.setdefault(ev, {"arm": os.path.basename(root), "geoms": [],
                                      "content_members": {}})
            cur_content = None
            for line in open(log, errors="replace"):
                if line.startswith("SHOWER_ABSORB PASS4_GEOM "):
                    rec["geoms"].append(parse_kv(line))
                elif line.startswith("SHOWER_CONTENT shower_id=") and "node_id=" in line:
                    cur_content = parse_kv(line)["node_id"]
                elif line.startswith("SHOWER_CONTENT   shower_id=") and " seg=" in line:
                    d = parse_kv(line)
                    if cur_content is not None:
                        rec["content_members"].setdefault(cur_content, {})[d["seg"]] = d.get("dQ", 0.0)
    return out


def track_like(row, mip_hi=1.3):
    if int(row.get("pdg", 0)) in TRACK_PDG:
        return True
    med = row.get("med_dqdx_mip", -1.0)
    return 0.0 < med < mip_hi


def snap(r):
    # chain-immune gap (probe v2); fall back to current-body gap on v1 logs
    v = r.get("snap_dis_cm", -1.0)
    return v if v >= 0 else r["body_dis_cm"]


RULES = {
    # name: fn(row) -> would-decline
    "gap15": lambda r: r["body_dis_cm"] > 15.0,
    "gap25": lambda r: r["body_dis_cm"] > 25.0,
    "gap40": lambda r: r["body_dis_cm"] > 40.0,
    "snap15": lambda r: snap(r) > 15.0,
    "snap20": lambda r: snap(r) > 20.0,
    "snap25": lambda r: snap(r) > 25.0,
    "snap30": lambda r: snap(r) > 30.0,
    "snap40": lambda r: snap(r) > 40.0,
    "trk_far": lambda r: track_like(r) and r["pair_dis_cm"] > r["cur_len_cm"],
    "trk_gap5": lambda r: track_like(r) and r["body_dis_cm"] > 5.0,
    "trk_long": lambda r: track_like(r) and r["len_cm"] > 20.0,
    "trk_long25": lambda r: track_like(r) and r["len_cm"] > 25.0,
    "snap25_or_trklong": lambda r: snap(r) > 25.0 or (track_like(r) and r["len_cm"] > 20.0),
    "snap30_or_trklong": lambda r: snap(r) > 30.0 or (track_like(r) and r["len_cm"] > 20.0),
    "gap25_or_trkfar": lambda r: r["body_dis_cm"] > 25.0 or (track_like(r) and r["pair_dis_cm"] > r["cur_len_cm"]),
    "gap25_or_trkgap5": lambda r: r["body_dis_cm"] > 25.0 or (track_like(r) and r["body_dis_cm"] > 5.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+")
    ap.add_argument("--tsv")
    args = ap.parse_args()
    data = collect([r for g in args.globs for r in sorted(glob.glob(g))])

    rows = []
    for ev in sorted(data):
        rec = data[ev]
        lab = load_labels(ev)
        for g in rec["geoms"]:
            seg = g.get("seg")
            row = dict(event=ev, arm=rec["arm"], **g)
            row["verdict"] = (lab or {}).get("verdict") or ""
            row["label_tag"] = (lab or {}).get("tag") or ""
            cls, mshw = "", ""
            if lab:
                for key, (target, ins, outs) in lab["marks"].items():
                    if seg in outs:
                        cls, mshw = "OUT", key
                        break
                    if seg in target and not cls:
                        cls, mshw = "TARGET", key
            row["seg_mark"] = cls
            row["marked_shower"] = mshw
            row["trk"] = int(track_like(row))
            rows.append(row)

    n_lab = sum(1 for r in rows if r["label_tag"])
    print("PASS4_GEOM: %d accepted absorbs over %d events (%d rows in labeled events)"
          % (len(rows), len(data), n_lab))
    for cls in ("OUT", "TARGET", ""):
        sel = [r for r in rows if r["seg_mark"] == cls and (cls or r["label_tag"])]
        if not sel:
            continue
        import statistics as st
        bd = [r["body_dis_cm"] for r in sel]
        sd = [snap(r) for r in sel]
        print("  class=%-7s n=%3d body_dis med=%6.1f p90=%6.1f  snap_dis med=%6.1f p90=%6.1f  track_like=%d/%d"
              % (cls or "(none)", len(sel), st.median(bd),
                 sorted(bd)[int(0.9 * (len(bd) - 1))],
                 st.median(sd), sorted(sd)[int(0.9 * (len(sd) - 1))],
                 sum(r["trk"] for r in sel), len(sel)))

    print("\nGuard sweep (declines: OUT=want-high, TARGET=collateral, unlabeled=exposure):")
    for name, fn in RULES.items():
        ko = sum(1 for r in rows if r["seg_mark"] == "OUT" and fn(r))
        kt = sum(1 for r in rows if r["seg_mark"] == "TARGET" and fn(r))
        ku = sum(1 for r in rows if not r["seg_mark"] and fn(r))
        no = sum(1 for r in rows if r["seg_mark"] == "OUT")
        nt = sum(1 for r in rows if r["seg_mark"] == "TARGET")
        print("  %-18s OUT %2d/%2d  TARGET %2d/%2d  unlabeled %3d" % (name, ko, no, kt, nt, ku))

    print("\nOUT-marked absorbs, full geometry:")
    for r in sorted([r for r in rows if r["seg_mark"] == "OUT"],
                    key=lambda r: (r["event"], r["seg"])):
        r = dict(r, snapv=snap(r))
        print("  evt%(event)d seg=%(seg)s pdg=%(pdg)s len=%(len_cm).1f med_mip=%(med_dqdx_mip).2f "
              "pair=%(pair_dis_cm).1f front=%(front_dis_cm).1f body=%(body_dis_cm).1f snap=%(snapv).1f "
              "a1=%(angle_v1).1f a2=%(angle_v2).1f tier=%(tier)s cur_len=%(cur_len_cm).1f "
              "divert=%(divert)s trk=%(trk)d" % r)

    if args.tsv and rows:
        cols = ["event", "arm", "seg", "pdg", "len_cm", "med_dqdx_mip", "cur", "cur_nseg",
                "cur_len_cm", "owner", "divert", "pair_dis_cm", "front_dis_cm", "body_dis_cm",
                "snap_dis_cm", "angle_v1", "angle_v2", "tier", "trk", "seg_mark", "marked_shower",
                "verdict", "label_tag"]
        with open(args.tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
        print("\nTSV: %s (%d rows)" % (args.tsv, len(rows)))


if __name__ == "__main__":
    main()
