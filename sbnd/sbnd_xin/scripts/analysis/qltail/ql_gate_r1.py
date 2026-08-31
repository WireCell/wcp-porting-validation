#!/usr/bin/env python3
"""doc 78 round 1 QL byte-identity gate.

Compares every archive of every manifest event between the base arm and the
round-1 arm.  Base side is base-ql/ except for the 18 events rerun in
base-ql-fix/ (4 rc!=0 + 14 binary-contaminated by the 02:29:26 rebuild --
see doc 78).  Covers mabc-all-apa.zip AND the two per-face zips (the
outputs of the per-face ExamineBundles instances that ql_arm_compare.py
does not hash) plus pctree and both opflash tars.

usage: ql_gate_r1.py <scratch-qltail-dir> [arm-name (default r1-ql)] [base-name (default composite base-ql/base-ql-fix)]
exit 0 = every archive of every event identical.
"""

# Provenance: this lived only in the doc-78 scratch arm ($S/qltail) and was
# rescued into the repo on 2026-08-25 when that arm was deleted.  Neither its
# two input lists nor the arms it compares survived that deletion, so it is
# kept here as the gate DEFINITION, not as a runnable one-liner:
#   ql_manifest.txt   -- the list is gone, but doc 78 line 172 records how it
#                        was built: 30 mcp1k tails + 56 mcp2k tails + the
#                        first 100 regular mcp1k = 186 events.  Rebuildable.
#   base_fix_events.txt -- the 18 events rerun into base-ql-fix (4 rc!=0 + 14
#                        contaminated by the 02:29:26 rebuild); doc 78 sec 7.
#                        Not recorded event-by-event anywhere: if the gate is
#                        re-run, regenerate both arms cleanly and drop the
#                        base-ql/base-ql-fix split entirely.
# Results it produced are in doc 78 line 172 (185/186 on the 6-archive
# variant) and doc 79 (186/186, knob ON).
import subprocess, sys, os
from concurrent.futures import ProcessPoolExecutor

AB = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest/hash_archive.py"
QT = sys.argv[1]
ARM = sys.argv[2] if len(sys.argv) > 2 else "r1-ql"
FIX = set(open(f"{QT}/base_fix_events.txt").read().split())
EVTS = sorted(open(f"{QT}/ql_manifest.txt").read().split(), key=int)

def h(p):
    r = subprocess.run(["python3", AB, p], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return r.stdout.split()[0]

def one(ev):
    base_root = f"{QT}/base-ql-fix" if ev in FIX else f"{QT}/base-ql"
    bad = []
    for f in [f"mabc-all-apa.zip", "mabc-apa0-face0.zip", "mabc-apa1-face0.zip",
              f"pctree-evt{ev}.tar.gz", "opflash_apa0.tar.gz", "opflash_apa1.tar.gz"]:
        a = h(f"{base_root}/ql_evt{ev}/{f}")
        b = h(f"{QT}/{ARM}/ql_evt{ev}/{f}")
        if a is None or b is None or a != b:
            bad.append((f, a, b))
    return ev, bad

nbad = 0
with ProcessPoolExecutor(8) as ex:
    for ev, bad in ex.map(one, EVTS):
        if bad:
            nbad += 1
            for f, a, b in bad:
                print(f"FAIL {ev} {f} base={a} arm={b}")
print(f"{'PASS' if nbad==0 else 'FAIL'}: {len(EVTS)-nbad}/{len(EVTS)} events identical "
      f"({len(FIX)} from base-ql-fix), 6 archives each")
sys.exit(0 if nbad == 0 else 1)
