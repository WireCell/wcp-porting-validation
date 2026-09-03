#!/usr/bin/env python3
"""doc pdvd/25 sec 13.11 round 12: lazy-flavor byte-identity + ncomp census
across the full 120-event PDVD `_keep` manifest (clus/docs/connect-graph-
strict-perf-round1.md sec 5a).  Compares work/<run>_<evt>_d25r12eager vs
work/<run>_<evt>_d25r12fast: mabc-pr.zip member-content hash, calib-pr-
evt*.json hash (minus the vertex_scoreboard.dual_chain timer field, per
feedback_calib_dump_cmp_timer_field, stripped recursively).  Also parses
every d25r12fast wct_pr log for the 'connect_graph_relaxed_strict:
... ncomp=... lazy=' census line to report which clusters actually crossed
busy_num_threshold and took the lazy path.

Usage: python3 stm/gates/r12_lazy_gate.py
Both arms must already exist (run_pr_evt.sh -s d25r12eager|d25r12fast
-stm-fit <run> all for run in 39252 39253 39349; d25r12fast needs
PDVD_PR_TLA="-S protect_graph_name='relaxed_strict_img_2d_rescue_long_wtrack_fast'").
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.join(os.path.dirname(PDVD), "abtest", "hash_archive.py")
OUT_DIR = os.path.join(PDVD, "stm", "gates")


def h_zip(p):
    return subprocess.check_output(["python3", H, p]).split()[0].decode()


def _strip_dual_chain_timer(obj):
    if isinstance(obj, dict):
        vs = obj.get("vertex_scoreboard")
        if isinstance(vs, dict):
            vs.pop("dual_chain", None)
        for v in obj.values():
            _strip_dual_chain_timer(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_dual_chain_timer(v)


def h_json(p):
    d = json.load(open(p))
    _strip_dual_chain_timer(d)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


CENSUS_RE = re.compile(
    r"nblobs=(\d+) npoints=(\d+) ncomp=(\d+) threshold=(\d+) lazy=(true|false)"
)


def main():
    eager_dirs = sorted(glob.glob(f"{PDVD}/work/*_d25r12eager"))
    tot = {"same": 0, "diff": 0, "missing": 0}
    lines = []
    census = []
    for ed in eager_dirs:
        base = os.path.basename(ed)[: -len("_d25r12eager")]
        fd = f"{PDVD}/work/{base}_d25r12fast"
        if not os.path.isdir(fd):
            tot["missing"] += 1
            lines.append(f"MISSING {base} fast-arm-dir-absent")
            continue
        for kind, pat, fn in (
            ("bee", "mabc-pr.zip", h_zip),
            ("calib", "calib-pr-evt*.json", h_json),
        ):
            a = glob.glob(f"{ed}/{pat}")
            b = glob.glob(f"{fd}/{pat}")
            if not a or not b:
                tot["missing"] += 1
                lines.append(f"MISSING {base} {kind}")
                continue
            ha, hb = fn(a[0]), fn(b[0])
            k = "same" if ha == hb else "diff"
            tot[k] += 1
            lines.append(f"{k.upper()} {base} {kind} {ha[:16]} {hb[:16]}")
        logs = glob.glob(f"{fd}/wct_pr_*.log")
        if logs:
            text = open(logs[0], errors="replace").read()
            for m in CENSUS_RE.finditer(text):
                nblobs, npoints, ncomp, threshold, lazy = m.groups()
                census.append((base, int(nblobs), int(npoints), int(ncomp), int(threshold), lazy))

    out = os.path.join(OUT_DIR, "r12_compare.txt")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("R12 gate:", tot, "->", out)

    print(f"\ncensus lines parsed: {len(census)} (fast arm only -- print fires whenever `fast` param is non-null)")
    over = [c for c in census if c[3] > c[4]]
    print(f"clusters with ncomp > threshold (actually walked lazy): {len(over)}")
    for c in over:
        print(f"  {c[0]}: nblobs={c[1]} npoints={c[2]} ncomp={c[3]} threshold={c[4]} lazy={c[5]}")
    census_out = os.path.join(OUT_DIR, "r12_census.tsv")
    with open(census_out, "w") as fh:
        fh.write("event\tnblobs\tnpoints\tncomp\tthreshold\tlazy\n")
        for c in census:
            fh.write("\t".join(str(x) for x in c) + "\n")
    print(f"full census -> {census_out}")
    return 0 if tot["diff"] == 0 and tot["missing"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
