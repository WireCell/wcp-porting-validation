#!/usr/bin/env python3
"""doc pr/91 round 1 -- turn the three env-gated shower probes into tables.

Reads `stdout.log` of a pr_evt<ID> dir produced with

    WCT_SHOWER_CONTENT_DEBUG=1 WCT_SHOWER_MERGE_DEBUG=1 \
    WCT_SHOWER_ENDPOINT_DEBUG=1 ./run_pr_chain_batch.sh ...

and prints, per event:

  inventory  one block per shower: header + every member segment with its own
             length / charge / energy share, cluster subtotals, and the
             `ORPHAN_VTX` lines (view vertices no member segment touches).
             This is the only NON-LOSSY membership source: the calib dump's
             `segment.shower_id` join keeps one shower per segment
             (PrDisplayDump.cxx:432), so an overlapped shower reads as empty.

  endpoint   the last farthest-vertex search per shower, with the winner and
             whether a member segment touches it.  A winner with
             touched_by_member=0 is an end point outside the shower's own
             charge -- the F1 defect.

  gates      every shower-merge decision, one line per candidate pair, with the
             clause that rejected it.  SKIP_PASS lines mark a pass that did not
             run at all.

Repro:
  scripts/pr91_shower_content.py work-pr91r1-dbg-mc [--only 169626]
"""
import glob
import os
import re
import sys
from collections import defaultdict


def kv(line):
    """Parse `k=v` tokens; values stay strings except obvious numbers."""
    out = {}
    for m in re.finditer(r"(\w+)=(\([^)]*\)|[^\s]+)", line):
        k, v = m.group(1), m.group(2)
        if v.startswith("("):
            out[k] = v
            continue
        try:
            out[k] = float(v) if ("." in v or "nan" in v) else int(v)
        except ValueError:
            out[k] = v
    return out


def parse(path):
    showers, members, orphans = {}, defaultdict(list), defaultdict(list)
    endpoints, gates = {}, []
    cur_ep = None
    for line in open(path, errors="replace"):
        if line.startswith("SHOWER_CONTENT shower_id="):
            d = kv(line)
            showers[d["shower_id"]] = d
        elif line.startswith("SHOWER_CONTENT   shower_id="):
            d = kv(line)
            if "ORPHAN_VTX" in d:
                orphans[d["shower_id"]].append(d)
            else:
                members[d["shower_id"]].append(d)
        elif line.startswith("SHOWER_ENDPOINT tag=add_shower"):
            d = kv(line)
            d["raw"] = line.strip()
            gates.append(("add_shower", d))
        elif line.startswith("SHOWER_ENDPOINT tag="):
            cur_ep = kv(line)
            cur_ep["cands"] = []
            endpoints[cur_ep["shower_id"]] = cur_ep   # last one wins
        elif line.startswith("SHOWER_ENDPOINT   ") and cur_ep is not None:
            d = kv(line)
            if "WINNER" in line:
                cur_ep["winner"] = d
            else:
                cur_ep["cands"].append(d)
        elif line.startswith("SHOWER_MERGE "):
            d = kv(line)
            d["raw"] = line.strip()
            gates.append((d.get("tag", "?"), d))
    return showers, members, orphans, endpoints, gates


def report(evt, path):
    showers, members, orphans, endpoints, gates = parse(path)
    print(f"================ evt {evt}   ({path})")
    print("---- inventory")
    for sid in sorted(showers):
        s = showers[sid]
        print(f"  shower_id={sid} node={s['node_id']} conn={s['conn']} pdg={s['pdg']} "
              f"nseg={s['nseg']} ncls={s['ncls']} kine_charge={s['kine_charge']}MeV "
              f"len={s['len']}cm start_vtx={s['start_vtx']}")
        print(f"      start={s['start']} end={s['end']} dir15={s['dir15']}")
        per_cluster = defaultdict(lambda: [0, 0.0, 0.0])
        for m in members[sid]:
            per_cluster[m["cluster"]][0] += 1
            per_cluster[m["cluster"]][1] += m["len"]
            per_cluster[m["cluster"]][2] += m["E_est"]
            print(f"      seg={m['seg']:>7} cl={m['cluster']:>3} len={m['len']:7.3f} "
                  f"pdg={m['pdg']:>5} E_est={m['E_est']:8.3f} flags={m['flags']} "
                  f"v0={m['v0']} v1={m['v1']}")
        if len(per_cluster) > 1:
            tot = " ".join(f"cl{c}:{n}seg/{L:.1f}cm/{E:.1f}MeV"
                           for c, (n, L, E) in sorted(per_cluster.items()))
            print(f"      by cluster: {tot}")
        for o in orphans[sid]:
            flag = "" if o["is_start_vtx"] else "   <-- NOT the start vertex (F1 suspect)"
            print(f"      ORPHAN_VTX={o['ORPHAN_VTX']} cl={o['cluster']} "
                  f"dis_from_start={o['dis_from_start']:.3f}cm "
                  f"is_start_vtx={o['is_start_vtx']}{flag}")

    print("---- endpoint provenance (last calculate_kinematics per shower)")
    for sid in sorted(endpoints):
        e = endpoints[sid]
        w = e.get("winner")
        if not w:
            continue
        bad = " <-- END POINT OUTSIDE OWN CHARGE" if w["touched_by_member"] == 0 else ""
        print(f"  shower_id={sid} tag={e['tag']} winner_vtx={w['vtx']} dis={w['dis']}cm "
              f"touched_by_member={w['touched_by_member']}{bad}")

    print("---- merge gates")
    for tag, g in gates:
        if tag == "add_shower":
            print("  " + g["raw"].replace("SHOWER_ENDPOINT tag=", ""))
            continue
        if "ENTER" in g.get("raw", ""):
            print(f"  {tag:18s} ENTER flag_skip={g.get('flag_skip')} "
                  f"nshowers={g.get('nshowers')}")
            continue
        v = g.get("verdict", "SKIP_PASS")
        if v in ("len_lt_3cm", "pdg_not_11", "is_the_parent_class"):
            continue
        bits = " ".join(f"{k}={g[k]}" for k in
                        ("cand_node", "cand_sid", "cand_conn", "cand_ke", "min_dis",
                         "angle", "angle1", "dis", "cand_seg", "reason", "flag_skip")
                        if k in g)
        print(f"  {tag:18s} {bits} verdict={v}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = sys.argv[1]
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    for d in sorted(glob.glob(os.path.join(root, "pr_evt*"))):
        evt = os.path.basename(d).replace("pr_evt", "")
        if only and evt != only:
            continue
        log = os.path.join(d, "stdout.log")
        if os.path.exists(log):
            report(evt, log)


if __name__ == "__main__":
    main()
