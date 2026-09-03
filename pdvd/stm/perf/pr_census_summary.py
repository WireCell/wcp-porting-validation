#!/usr/bin/env python3
"""doc pdvd/28: summarise the env-gated censuses of one PR log.

  WCT_RELAXED_EDGE_TIMING=1  -> 'OC28TIMING strict ...' one line per strict-connector call
  WCT_DQDX_SOLVE_CENSUS=1    -> 'DQDX_SOLVE single|multi n3d= n2d=u/v/w nnzA= iters= err= ms='
  WCT_TF_COVERAGE_CENSUS=1   -> 'TF_COV_CENSUS index rebuilt ... | since last: calls= visited= box_pass= covered='
                                'TF_COV_CENSUS at destruction: calls= ...'
Usage: python3 stm/perf/pr_census_summary.py work/<run>_<idx>_<tag>/wct_pr_*.log
"""
import re
import sys
from collections import defaultdict

RE_KV = re.compile(r"(\w+)=([-0-9.e+]+)")


def kv(line):
    return {k: float(v) for k, v in RE_KV.findall(line)}


def main(path):
    conn, solve, cov = [], defaultdict(lambda: [0, 0.0, 0.0, 0, 0]), [0, 0, 0, 0, 0]
    for line in open(path, errors="replace"):
        if "OC28TIMING strict" in line:
            d = kv(line[line.index("OC28TIMING"):])
            conn.append(d)
        elif "DQDX_SOLVE" in line:
            kind = "multi" if "DQDX_SOLVE multi" in line else "single"
            d = kv(line[line.index("DQDX_SOLVE"):])
            s = solve[kind]
            s[0] += 1
            s[1] += d.get("ms", 0)
            s[2] = max(s[2], d.get("ms", 0))
            s[3] += int(d.get("iters", 0))
            s[4] = max(s[4], int(d.get("n3d", 0)))
        elif "TF_COV_CENSUS" in line:
            d = kv(line[line.index("TF_COV_CENSUS"):])
            for i, k in enumerate(("calls", "visited", "box_pass", "covered")):
                cov[i] += int(d.get(k, 0))
            cov[4] += 1
    print(path)
    if conn:
        # the line carries repeated keys (n=, ms=); parse positionally instead
        agg = defaultdict(float)
        for c_line in open(path, errors="replace"):
            if "OC28TIMING strict" not in c_line:
                continue
            m = re.search(r"total_ms=([0-9.]+) closest n=(\d+) ms=([0-9.]+) dir1 n=(\d+) ms=([0-9.]+) dir2 n=(\d+) ms=([0-9.]+) ghost n=(\d+) steps=(\d+) ms=([0-9.]+)", c_line)
            if not m:
                continue
            v = [float(x) for x in m.groups()]
            for k, x in zip(("total", "n_closest", "t_closest", "n_dir1", "t_dir1", "n_dir2", "t_dir2", "n_ghost", "ghost_steps", "t_ghost"), v):
                agg[k] += x
        print("  (positional) total %.1f s | closest %d walks %.1f s | dir1 %d %.1f s | dir2 %d %.1f s | ghost %d tests, %d steps, %.1f s" % (
            agg["total"] / 1000, agg["n_closest"], agg["t_closest"] / 1000, agg["n_dir1"], agg["t_dir1"] / 1000,
            agg["n_dir2"], agg["t_dir2"] / 1000, agg["n_ghost"], agg["ghost_steps"], agg["t_ghost"] / 1000))
        big = max(conn, key=lambda c: c["total_ms"])
        print("  largest call: ncomp=%d total %.1f s" % (big["ncomp"], big["total_ms"] / 1000))
    for kind, s in sorted(solve.items()):
        print("dQ/dx solver %-6s: %d calls, %.1f s, max %.0f ms, %d iterations total, max n3d %d" % (kind, s[0], s[1] / 1000, s[2], s[3], s[4]))
    if cov[4]:
        print("coverage index: %d rebuild/destruction records; calls=%d visited=%d (%.1f per call) box_pass=%d covered=%d" % (
            cov[4], cov[0], cov[1], cov[1] / max(cov[0], 1), cov[2], cov[3]))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
