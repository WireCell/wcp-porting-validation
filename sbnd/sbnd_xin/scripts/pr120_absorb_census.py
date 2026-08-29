#!/usr/bin/env python3
"""doc pr/120 -- admission census: which pass admitted each shower member,
at what angle to the FINAL shower axis, and does the admission-time frame
already separate the hand-scan OUT marks.

Reuses pr119_expel_census's truth machinery (charge-weighted cross-run label
matching with the WRONGOWNER-aware member classes).  New here:

  1. Per-(shower, member) FINAL scan-equivalent angle, computed offline the
     way the scan display did (em_display_viewer.seg_vs_shower): axis = the
     content probe's dir15, vector = shower start -> closest dump point of
     the member to the start.  This is the frame in which the stem_backfill /
     pass3_cone / examine_shower_1_tmp OUT marks sit at 113-148 deg.
  2. absorbed_by per member from the SHOWER_ABSORB probe lines (viewer
     convention: last record per segment wins).
  3. The P120_STEM / P120_P3CONE admission-frame lines (site's own angle +
     scan-equivalent angle at admission time) -- decides whether a guard can
     sit AT the admission site or must audit late.
  4. The straight-long-electron walk census (evt54332's hole: the
     absorb_track_guard exempts pdg==11, ADD lines carry straight=/len_cm=).

Repro:
  ./scripts/pr120_absorb_census.py --prepdir em_display/emprep-120dbgA \
      --members-tsv docs/pr/pr120-absorb-members.tsv \
      work-pr120r1-dbgA-mcp1k work-pr120r1-dbgA-mcp2k \
      work-pr120r1-dbgA-ncpi0 work-pr120r1-dbgA-nuecc48
"""
import argparse
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pr119_expel_census as p119   # load_labels / load_sidecar / match_label_to_node

KV = re.compile(r"(\w+)=([^\s()]+)")
TUP = re.compile(r"(\w+)=\(([-\d.e+]+),([-\d.e+]+),([-\d.e+]+)\)")

GUARD_SITES = ("stem_backfill", "pass3_cone", "examine_shower_1_tmp")


def parse_kv(line):
    d = {}
    for k, v in KV.findall(line):
        v = v.rstrip("cm")
        try:
            d[k] = float(v) if ("." in v or "e" in v) else int(v)
        except ValueError:
            d[k] = v
    for k, x, y, z in TUP.findall(line):
        d[k] = (float(x), float(y), float(z))
    return d


def collect(roots):
    """event -> {'content': {node_id: {dir15, start, shower_id}},
                 'absorb': {seg: last record}, 'p120': [lines],
                 'adds': [walk ADD records with bound site]}"""
    out = {}
    for root in roots:
        for log in sorted(glob.glob(os.path.join(root, "pr_evt*", "stdout.log"))):
            ev = int(os.path.basename(os.path.dirname(log))[len("pr_evt"):])
            rec = out.setdefault(ev, {"content": {}, "absorb": {}, "p120": [],
                                      "adds": [], "dumpdir": os.path.dirname(log)})
            last_site = None
            with open(log, errors="replace") as fh:
                for line in fh:
                    if line.startswith("SHOWER_CONTENT shower_id="):
                        d = parse_kv(line)
                        if "node_id" in d:
                            rec["content"][d["node_id"]] = d
                    elif line.startswith("SHOWER_ABSORB P120_"):
                        d = parse_kv(line)
                        d["tag"] = line.split()[1]
                        rec["p120"].append(d)
                    elif line.startswith("SHOWER_ABSORB DIRECT "):
                        d = parse_kv(line)
                        rec["absorb"][d["seg"]] = {"site": d.get("site"), "how": "direct",
                                                   "shower_start_seg": d.get("shower_start_seg")}
                    elif line.startswith("SHOWER_ABSORB site="):
                        last_site = parse_kv(line).get("site")
                    elif line.startswith("SHOWER_ABSORB ADD "):
                        d = parse_kv(line)
                        d["site"] = last_site
                        rec["adds"].append(d)
                        rec["absorb"][d["seg"]] = {"site": last_site, "how": "walk_add",
                                                   "shower_start_seg": d.get("shower_start_seg"),
                                                   "len_cm": d.get("len_cm"),
                                                   "straight": d.get("straight"),
                                                   "pdg": d.get("pdg")}
    return out


def load_dump_points(dumpdir, ev):
    p = os.path.join(dumpdir, "calib-pr-evt%d.json" % ev)
    if not os.path.exists(p):
        return {}, {}
    with open(p) as fh:
        d = json.load(fh)
    pts = {}
    for s in d.get("segments") or ():
        # dump segment "id" is already the display id (cluster_id*1000 + seg)
        pts[int(s["id"])] = [(pp["x"], pp["y"], pp["z"]) for pp in (s.get("points") or ())]
    starts = {int(s["id"]): (s["start"]["x"], s["start"]["y"], s["start"]["z"])
              for s in (d.get("showers") or ()) if s.get("start")}
    return pts, starts


def ang_deg(ax, v):
    ma = math.sqrt(sum(a * a for a in ax))
    mv = math.sqrt(sum(a * a for a in v))
    if ma < 1e-3 or mv < 1e-6:
        return -1.0
    c = sum(a * b for a, b in zip(ax, v)) / (ma * mv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def final_angle(member_pts, start, dir15):
    """viewer seg_vs_shower: closest member point to start; angle vs dir15."""
    if not member_pts or start is None or dir15 is None:
        return -1.0, -1.0
    best, bp = None, None
    for p in member_pts:
        d2 = sum((a - b) ** 2 for a, b in zip(p, start))
        if best is None or d2 < best:
            best, bp = d2, p
    v = tuple(a - b for a, b in zip(bp, start))
    return ang_deg(dir15, v), math.sqrt(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepdir", required=True)
    ap.add_argument("--members-tsv", default=None)
    ap.add_argument("roots", nargs="+")
    args = ap.parse_args()

    roots = [r if os.path.isabs(r) else os.path.join(SX, r) for r in args.roots]
    for r in roots:
        if not os.path.isdir(r):
            sys.exit("no such arm root: %s" % r)
    prepdir = args.prepdir if os.path.isabs(args.prepdir) else os.path.join(SX, args.prepdir)

    events = collect(roots)
    print("events: %d (content headers %d, absorb records %d, p120 lines %d, walk adds %d)\n"
          % (len(events),
             sum(len(r["content"]) for r in events.values()),
             sum(len(r["absorb"]) for r in events.values()),
             sum(len(r["p120"]) for r in events.values()),
             sum(len(r["adds"]) for r in events.values())))

    members = []          # flat rows
    for ev, rec in sorted(events.items()):
        labels = p119.load_labels(ev)
        sidecar = p119.load_sidecar(prepdir, ev)
        node_truth = {}
        if labels:
            for lk, (target, ins, outs) in sorted(labels.items()):
                node = p119.match_label_to_node(target, sidecar, lk)
                if node is not None:
                    node_truth.setdefault(node, []).append((lk, target, ins, outs))
        pts, starts = load_dump_points(rec["dumpdir"], ev)
        for node, mem in sorted(sidecar.items()):
            hdr = rec["content"].get(node)
            dir15 = hdr.get("dir15") if hdr else None
            start = starts.get(node) or (hdr.get("start") if hdr else None)
            for seg, dq in sorted(mem.items()):
                fa, fd = final_angle(pts.get(seg), start, dir15)
                if labels is None:
                    cls = "UNKNOWN"
                elif node in node_truth:
                    is_out = any(seg in outs for _, _, _, outs in node_truth[node])
                    is_in = any(seg in target for _, target, _, _ in node_truth[node])
                    cls = ("WRONGOWNER" if (is_out and is_in) else
                           "OUT" if is_out else "IN" if is_in else "OTHER")
                else:
                    cls = "HOLD"
                ab = rec["absorb"].get(seg) or {}
                members.append({
                    "event": ev, "node": node, "seg": seg, "dQ": dq, "cls": cls,
                    "site": ab.get("site") or "", "how": ab.get("how") or "",
                    "final_ang": round(fa, 2), "final_dist": round(fd, 2),
                })

    print("member truth classes: %s\n"
          % {c: sum(1 for m in members if m["cls"] == c)
             for c in ("OUT", "WRONGOWNER", "IN", "OTHER", "HOLD", "UNKNOWN")})

    if args.members_tsv:
        cols = ["event", "node", "seg", "cls", "site", "how", "final_ang",
                "final_dist", "dQ"]
        path = args.members_tsv if os.path.isabs(args.members_tsv) \
            else os.path.join(SX, args.members_tsv)
        with open(path, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for m in members:
                fh.write("\t".join(str(m[c]) for c in cols) + "\n")
        print("members tsv: %s (%d rows)\n" % (path, len(members)))

    # 1. validation + per-site final-angle distributions
    print("== final-angle distribution per admitting site (quartiles) ==")
    by_site = {}
    for m in members:
        if m["final_ang"] < 0:
            continue
        by_site.setdefault((m["site"] or "(none)", m["cls"]), []).append(m["final_ang"])
    for (site, cls), a in sorted(by_site.items()):
        a = sorted(a)
        q = lambda f: a[min(len(a) - 1, int(f * len(a)))]
        print("  %-28s %-10s n=%-4d min=%6.1f med=%6.1f max=%6.1f  >110: %d"
              % (site, cls, len(a), a[0], q(.5), a[-1],
                 sum(1 for x in a if x > 110)))
    print()

    # 2. the guard predicate sweep
    print("== guard sweep: site in %s AND final_ang > theta ==" % (GUARD_SITES,))
    print("%-6s | fired OUT/WO/IN/OTHER/HOLD | OUT events | IN+HOLD events" % "theta")
    for theta in (100, 105, 110, 120, 130, 140):
        fired = [m for m in members
                 if m["site"] in GUARD_SITES and m["final_ang"] > theta]
        cnt = {c: sum(1 for m in fired if m["cls"] == c)
               for c in ("OUT", "WRONGOWNER", "IN", "OTHER", "HOLD")}
        oev = sorted({m["event"] for m in fired if m["cls"] in ("OUT", "WRONGOWNER")})
        cev = sorted({m["event"] for m in fired if m["cls"] in ("IN", "HOLD")})
        print("%-6d | %d / %d / %d / %d / %d | %s | %s"
              % (theta, cnt["OUT"], cnt["WRONGOWNER"], cnt["IN"], cnt["OTHER"],
                 cnt["HOLD"], oev, cev))
    print()

    # 3. admission-frame vs final-frame (P120 lines)
    print("== admission-frame lines (P120_STEM / P120_P3CONE) ==")
    for tag in ("P120_STEM", "P120_P3CONE"):
        rows = []
        for ev, rec in sorted(events.items()):
            for d in rec["p120"]:
                if d["tag"] != tag:
                    continue
                d["event"] = ev
                mm = next((m for m in members
                           if m["event"] == ev and m["seg"] == d.get("seg")), None)
                d["final_ang"] = mm["final_ang"] if mm else -1
                d["cls"] = mm["cls"] if mm else "?"
                rows.append(d)
        print("  %s: %d lines" % (tag, len(rows)))
        a15 = sorted(d.get("ang15", -1) for d in rows if d.get("ang15", -1) >= 0)
        if a15:
            q = lambda f: a15[min(len(a15) - 1, int(f * len(a15)))]
            print("    admission ang15 quartiles: min=%.1f q25=%.1f med=%.1f q75=%.1f max=%.1f"
                  % (a15[0], q(.25), q(.5), q(.75), a15[-1]))
        for d in rows:
            if d["cls"] in ("OUT", "WRONGOWNER") or (d.get("ang15", -1) > 110):
                print("    evt%-8d seg=%-8s cls=%-10s adm_ang15=%-7s adm_ang60=%-7s "
                      "site_ang=%-7s ok=%s final_ang=%s"
                      % (d["event"], d.get("seg"), d["cls"], d.get("ang15"),
                         d.get("ang60"), d.get("site_ang", ""), d.get("ok", ""),
                         d.get("final_ang")))
    print()

    # 4. straight-long-electron walk census (the evt54332 hole)
    print("== walk ADD straight-long pdg==11 census (absorb_track_guard exemption) ==")
    for minlen in (10.0, 20.0, 30.0):
        sel = []
        for ev, rec in sorted(events.items()):
            for d in rec["adds"]:
                if d.get("pdg") == 11 and d.get("straight") == 1 \
                        and (d.get("len_cm") or 0) >= minlen:
                    mm = next((m for m in members
                               if m["event"] == ev and m["seg"] == d.get("seg")), None)
                    sel.append((ev, d.get("site"), d.get("seg"), d.get("len_cm"),
                                mm["cls"] if mm else "?"))
        cnt = {}
        for _, site, _, _, cls in sel:
            cnt[cls] = cnt.get(cls, 0) + 1
        print("  len>=%2.0fcm: n=%d  truth=%s" % (minlen, len(sel), cnt))
        if minlen == 20.0:
            for ev, site, seg, ln, cls in sel:
                print("    evt%-8d site=%-24s seg=%-8s len=%.1f cls=%s"
                      % (ev, site, seg, ln, cls))
    print()


if __name__ == "__main__":
    main()
