#!/usr/bin/env python3
"""Materialise a standing witness arm for the 31 pr127/d97 sentinels.

WHY.  Production (work-*-d97fvpr2) is 30 PASS / 1 FAIL: 105074 (pr/128 class
B) asserts a PF node production deliberately no longer produces, and that
re-anchor is an OPEN owner call.  The only arms on disk where the suite is
31/31 are work-*-d97off2pr (prod-2026-09-03) and work-*-prod0901b -- and the
2026-09-02 retire round releases BOTH.  Deleting them would leave the open
sentinel with no passing arm to adjudicate against, which is exactly what doc
91's interlock 8 exists to refuse.

So before that round deletes anything, the 28 distinct sentinel events are
copied out of work-*-d97off2pr into work-sent97-<sample>/ (~70 MB), and the
suite must report 31 PASS / 0 FAIL against the WITNESS ALONE.  If it does not,
the witness is not a substitute and the source arm has to stay.

Same role work-sent130-* played before doc 91 folded it into production.
Reads only; copies, never moves.
"""
import os, shutil, subprocess, sys, glob

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
os.chdir(ROOT)
SRC = "d97off2pr"                       # the 31/31 arm being retired
SAMPLES = ("mcp1k", "mcp2k", "ncpi0", "nuecc48")

evs = set()
for ln in subprocess.run([sys.executable, "scripts/d97_sentinels.py", "--list"],
                         capture_output=True, text=True).stdout.splitlines():
    if ln.strip() and ln.split()[0].isdigit():
        evs.add(ln.split()[0])
print(f"{len(evs)} distinct sentinel events")

made = miss = 0
for ev in sorted(evs, key=int):
    for s in SAMPLES:
        src = f"work-{s}-{SRC}/pr_evt{ev}"
        if os.path.isdir(src):
            dst = f"work-sent97-{s}/pr_evt{ev}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copytree(src, dst, symlinks=True)
            made += 1
            break
    else:
        print(f"  MISSING in {SRC}: {ev}")
        miss += 1
print(f"copied {made} event dirs, {miss} missing")

# carry the arm-level tables too -- some sentinels read nusel-events.tsv
for s in SAMPLES:
    if os.path.isdir(f"work-sent97-{s}"):
        for t in ("nusel-events.tsv", "nusel-table.tsv"):
            p = f"work-{s}-{SRC}/{t}"
            if os.path.exists(p):
                shutil.copy2(p, f"work-sent97-{s}/{t}")
sys.exit(1 if miss else 0)
