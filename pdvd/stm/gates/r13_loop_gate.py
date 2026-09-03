#!/usr/bin/env python3
"""doc pdvd/26: examine_partial_identical_segments degenerate-split guard --
byte-identity + fire census across the full 120-event PDVD `_keep` manifest.

Arm A (baseline): work/<run>_<evt>_d25r13base -- the tree at 4776d637 WITHOUT
the guard (pinned /home/xqian/tmp/doc25r13/lib_base), 118 events: 039349/14
and /53 are deliberately absent because that binary never terminates on them
(the d25r12eager arm, 3e1854a8-era, was killed on them at 2891 s / 2649 s).
Arm B (fix):      work/<run>_<evt>_d25r13fix  -- same inputs, the guard
(pinned /home/xqian/tmp/doc25r13/lib).
NOTE: d25r12eager is NOT a valid Bee-zip baseline for the guard -- toolkit
ab0762c6 (08:30, doc pdvd/30) fixed the stm_fit Bee layer's real_cluster_id
(hard-coded 0 before), so every mabc-pr.zip differs between the 07:47 and any
later binary; pass it as arm A to see exactly that (calib dumps still match).

Per event: mabc-pr.zip member-content hash (abtest/hash_archive.py), calib-pr-
evt*.json hash minus the vertex_scoreboard.dual_chain timer (recursively, per
feedback_calib_dump_cmp_timer_field), and the count of the guard's own DEBUG
line ('degenerate split skipped') in arm B's wct_pr log.  The claim under test:
the guard fires ONLY on the two never-terminating events, and every other
event is byte-identical.

Usage: python3 stm/gates/r13_loop_gate.py [armA_tag armB_tag]
Writes stm/gates/r13_compare.txt and stm/gates/r13_census.tsv.
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
TAG_A = sys.argv[1] if len(sys.argv) > 2 else "d25r13base"
TAG_B = sys.argv[2] if len(sys.argv) > 2 else "d25r13fix"


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


def wall_s(d):
    for p in glob.glob(f"{d}/pr_resource_*.txt"):
        m = re.search(r"wall_s=(\d+)", open(p).read())
        if m:
            return int(m.group(1))
    return -1


def main():
    # Iterate over arm B (the fix arm holds every event); an event absent from
    # arm A is reported as such -- for d25r13base those are exactly 14 and 53.
    dirs_b = sorted(glob.glob(f"{PDVD}/work/*_{TAG_B}"))
    tot = {"same": 0, "diff": 0, "missing": 0, "absent_in_A": 0, "no_dump": 0}
    lines, census = [], []
    for db in dirs_b:
        ev = os.path.basename(db)[: -len(TAG_B) - 1]
        da = f"{PDVD}/work/{ev}_{TAG_A}"
        logs_b = glob.glob(f"{db}/wct_pr_*.log")
        fires = 0
        if logs_b:
            fires = sum(1 for l in open(logs_b[0], errors="replace") if "degenerate split skipped" in l)
        wa, wb = wall_s(da), wall_s(db)
        census.append((ev, fires, wa, wb))
        verdicts = []
        if not glob.glob(f"{da}/mabc-pr.zip"):
            tot["absent_in_A"] += 1
            lines.append(f"{ev} fires={fires} wall {wa}s -> {wb}s  ABSENT in arm A (never terminated there)")
            continue
        for name, hf in (("mabc-pr.zip", h_zip), ("calib", h_json)):
            pa = glob.glob(f"{da}/{name}" if name != "calib" else f"{da}/calib-pr-evt*.json")
            pb = glob.glob(f"{db}/{name}" if name != "calib" else f"{db}/calib-pr-evt*.json")
            if not pa and not pb:
                # zero STM tags => no calib dump at all (doc 25 sec 13.10); same on both sides
                verdicts.append(f"{name}:absent-both")
                tot["no_dump"] += 1
                continue
            if not pa or not pb:
                verdicts.append(f"{name}:MISSING")
                tot["missing"] += 1
                continue
            same = hf(pa[0]) == hf(pb[0])
            tot["same" if same else "diff"] += 1
            verdicts.append(f"{name}:{'same' if same else 'DIFF'}")
        lines.append(f"{ev} fires={fires} wall {wa}s -> {wb}s  " + " ".join(verdicts))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "r13_compare.txt"), "w") as f:
        f.write(f"# arm A={TAG_A} arm B={TAG_B}\n")
        f.write("\n".join(lines) + "\n")
        f.write(f"# TOTAL same={tot['same']} diff={tot['diff']} missing={tot['missing']} "
                f"absent_in_A={tot['absent_in_A']} no_dump_both={tot['no_dump']}\n")
    with open(os.path.join(OUT_DIR, "r13_census.tsv"), "w") as f:
        f.write("event\tguard_fires\twall_s_A\twall_s_B\n")
        for ev, n, wa, wb in census:
            f.write(f"{ev}\t{n}\t{wa}\t{wb}\n")
    fired = [(ev, n) for ev, n, _, _ in census if n]
    print(f"events_B={len(dirs_b)} same={tot['same']} diff={tot['diff']} missing={tot['missing']} "
          f"absent_in_A={tot['absent_in_A']} no_dump_both={tot['no_dump']}")
    print(f"guard fired on {len(fired)} event(s): {fired}")
    for l in lines:
        if "DIFF" in l or "MISSING" in l or "ABSENT" in l or ("fires=" in l and "fires=0" not in l):
            print(l)
    return 0 if tot["diff"] == 0 and tot["missing"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
