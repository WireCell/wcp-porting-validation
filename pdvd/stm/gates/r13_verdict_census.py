#!/usr/bin/env python3
"""doc pdvd/26 round 2, item 1: how many cosmic-tagger verdicts did the
fetch_channel_from_anode lower-bound fix (toolkit 3e1854a8, doc 25 sec 13.4
item 12) move on the 120-event PDVD `_keep` manifest?

Arm PRE  = work/*_d25r12fast   (the last arm on the pre-fix binary; same
                                protect_graph_name flavor as production now)
Arm POST = work/*_d25r13fix    (post-fix binary + the doc 26 loop guard, which
                                does not touch tagger verdicts)

Compares every `visit: TaggerCheck{STM,TGM,FC}: cluster N -> ...` line per
(event, tagger, cluster) present in BOTH logs (039349/32 crashed pre-fix, so
its log is truncated -- only the common clusters count), and marks whether a
flipped cluster is in the doc 25 dQ/dx sample (stm/sample_index.tsv, keyed by
event + cluster).

Usage: python3 stm/gates/r13_verdict_census.py [pre_tag post_tag]
Writes stm/gates/r13_verdict_census.tsv (one row per flip) and prints totals.
"""
import glob
import os
import re
import sys
from collections import Counter, defaultdict

PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRE = sys.argv[1] if len(sys.argv) > 2 else "d25r12fast"
POST = sys.argv[2] if len(sys.argv) > 2 else "d25r13fix"
RE = re.compile(r"visit: TaggerCheck(STM|TGM|FC): cluster (\d+) → (.*)$")


def verdicts(d):
    out = {}
    logs = glob.glob(f"{d}/wct_pr_*.log")
    if not logs:
        return None
    for line in open(logs[0], errors="replace"):
        m = RE.search(line.rstrip())
        if m:
            out[(m.group(1), int(m.group(2)))] = m.group(3).strip()
    return out


def sample_clusters():
    s = set()
    p = f"{PDVD}/stm/sample_index.tsv"
    if not os.path.exists(p):
        return s
    for line in open(p):
        if line.startswith("#") or line.startswith("particle"):
            continue
        f = line.split("\t")
        if len(f) > 4:
            s.add((f[1], int(f[4])))
    return s


def main():
    sample = sample_clusters()
    rows, per_tagger, direction = [], Counter(), Counter()
    n_events = n_common = 0
    events_hit = set()
    compared = Counter()
    for dpost in sorted(glob.glob(f"{PDVD}/work/*_{POST}")):
        ev = os.path.basename(dpost)[: -len(POST) - 1]
        dpre = f"{PDVD}/work/{ev}_{PRE}"
        vpre, vpost = verdicts(dpre), verdicts(dpost)
        if vpre is None or vpost is None:
            continue
        n_events += 1
        for key in sorted(set(vpre) & set(vpost)):
            compared[key[0]] += 1
            if vpre[key] != vpost[key]:
                tagger, cid = key
                per_tagger[tagger] += 1
                direction[(tagger, vpre[key], vpost[key])] += 1
                events_hit.add(ev)
                rows.append((ev, tagger, cid, vpre[key], vpost[key], "yes" if (ev, cid) in sample else "no"))
    out = os.path.join(PDVD, "stm", "gates", "r13_verdict_census.tsv")
    with open(out, "w") as f:
        f.write(f"# pre={PRE} post={POST} events_compared={n_events}\n")
        f.write("event\ttagger\tcluster\tpre\tpost\tin_dqdx_sample\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"events compared: {n_events}; verdict lines compared per tagger: {dict(compared)}")
    print(f"flips: {dict(per_tagger)} on {len(events_hit)} event(s)")
    for (t, a, b), n in sorted(direction.items()):
        print(f"  {t}: '{a}' -> '{b}': {n}")
    ins = [r for r in rows if r[5] == "yes"]
    print(f"flipped clusters that are in the dQ/dx sample (stm/sample_index.tsv, {len(sample)} entries): {len(ins)}")
    for r in ins:
        print("  ", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
