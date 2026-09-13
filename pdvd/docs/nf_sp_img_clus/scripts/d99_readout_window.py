#!/usr/bin/env python3
"""doc pdvd/99 sec 4.1 -- what the clustering readout window does to STM tagging.

p98voff ran clustering with the SP frames in its input dir (run_clus_evt.sh then reads readout_window_ticks = 6400 from
them); production (p96vprod) and the production-staged arms (p98voffq, p98vonq) ran with the 10000 fallback.

  cd pdvd && python3 docs/nf_sp_img_clus/scripts/d99_readout_window.py > docs/nf_sp_img_clus/d99/readout_window_effect_p98voff.txt

Per arm: events; readout_edge_guard FIRINGS (the "readout_edge_guard: cluster N rejected" lines, not the configure line);
STM candidates summed over "CheckSTM_Michel: N candidate(s)"; readout_window_ticks from the clustering pctree-evt*.tlas.
Then, on the record matched onto p98voff: items whose object matches cleanly (ok) and had a candidate on p96vprod but have
none on p98voff.
"""
import glob
import json
import re
import sys

ARMS = ("p96vprod", "p98voff", "p98voffq", "p98vonq")
MATCH_OFF = "/home/xqian/tmp/p98/carry/match_p98voff.json"
MATCH_ID = "/home/xqian/tmp/p98/g4/g4a_identity.json"


def items(path):
    m = json.load(open(path))
    return {it["key"] if "key" in it else it["old_key"]: it for it in (m if isinstance(m, list) else m.values())}


def field(it, k):
    return (it.get("match") or it).get(k)


def main():
    print("# doc pdvd/99 sec 4.1 -- readout window: frames in the clustering input dir (p98voff) vs production staging")
    ex = {}
    for arm in ARMS:
        dirs = sorted(glob.glob(f"work/*_{arm}"))
        evals = fire = cand = 0
        win = {}
        for d in dirs:
            for lg in glob.glob(f"{d}/wct_pr_*.log"):
                for line in open(lg, errors="replace"):
                    if "readout_edge_guard: cluster" in line:
                        evals += 1
                        # a firing logs twice: a debug evaluation with early/late=true and an Info "rejected" line;
                        # count the Info line only
                        if "rejected" in line:
                            fire += 1
                            ex.setdefault(arm, line.strip())
                    m = re.search(r"CheckSTM_Michel: (\d+) candidate\(s\)", line)
                    if m:
                        cand += int(m.group(1))
            for t in glob.glob(f"{d}/pctree-evt*.tlas"):
                m = re.search(r"readout_window_ticks\D{0,4}(\d+)", open(t).read())
                k = m.group(1) if m else "none"
                win[k] = win.get(k, 0) + 1
        print(f"{arm:9s} events {len(dirs):3d}  readout_edge_guard log lines {evals:4d}, of which firing {fire:4d}  "
              f"STM candidates {cand:4d}  "
              f"readout_window_ticks {dict(sorted(win.items()))}")
    for arm, line in ex.items():
        print(f"example firing {arm}: {line[:220]}")
    off, ident = items(MATCH_OFF), items(MATCH_ID)
    keys = sorted(off)
    if len(keys) != len(ident):
        sys.exit(f"record size differs: {len(keys)} vs {len(ident)}")
    ok = [k for k in keys if field(off[k], "status") == "ok"]
    lost = [k for k in ok if field(ident[k], "candidate") and not field(off[k], "candidate")]
    print(f"record on p98voff: {len(keys)} items, ok {len(ok)}; ok with a candidate on p96vprod and none on p98voff: {len(lost)}")


if __name__ == "__main__":
    main()
