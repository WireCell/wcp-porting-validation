#!/usr/bin/env python3
"""Cleanup ROUND I 2026-09-21 -- planner for the ORPHANED DRIVER LOGS.  Doc sbnd_xin/121.

  usage:  python3 plan_orphanlogs_20260925.py           # plan + files_orphanlogs_<tree>_<STAMP>.txt
          PLAN_SUFFIX=confirm python3 plan_orphanlogs_20260925.py   # the driver's re-plan

THE CLASS THIS ROUND DISCOVERED.  `run_pr_evt.sh` writes its per-event driver log as
`<tree>/work/.batch_pr_<run6>_<evt>[_<arm>].log` -- BESIDE the event directory, not inside
it.  Nine rounds of directory-level retirement have therefore never touched one: the
planner enumerates directories (`arm_token()` returns None for a file), and the leading dot
hides them from `du dir/*`, which is why no round's survey ever reported them either.
Measured 2026-09-21: 33495 files, 30.70 GiB, of which 27.39 GiB name an event directory
that no longer exists -- debris from rounds A-H (`d25r12eager`, `d28r2fb`, `h26q2dprod`,
`h28prod`, `h100a`, `h100b` ...).

OWNER RULING 2026-09-21, asked with the measurement in hand: release the ORPHANS ONLY --
"Yes, orphans only".  The 3.31 GiB whose event directory still exists stays, because that
is the debug trail of an arm we are keeping.

WHY THIS IS NOT THE FILE-LEVEL RELEASE THE OWNER DECLINED.  Round D sec 14 removed
`frames.tar.bz2`/`icluster.npz` from INSIDE kept arms, which cost those arms their
re-runnability (PDVD imaging is no longer re-runnable from disk).  Nothing of that kind is
possible here: an orphan log's run was removed rounds ago, so there is no product left to
make un-re-runnable.  The re-runnability cost of this release is exactly zero, and that is
the reason it was worth asking separately.

THE RECORD IS A MANIFEST, NOT A TAR -- deliberately, and this is the one honest
compromise.  archive_records' HEAVY list carries a non-heavy file bodily into the
`.tar.zst`, and `.log` is not heavy, so a naive freeze would have compressed 27 GiB of
logs into the record layer and freed a fraction of what it claims.  Round D sec 14's
file-level release set the precedent for heavy content: a manifest (path, size, mtime,
SHA-256) and a links file, no tar.  So the record here says exactly which logs existed,
how big each was and what it hashed to -- not what they said.  Stated in doc 121 sec 5
rather than buried here.

This script RETIRES NOTHING.  It prints a plan and writes the file lists.
"""
import os, re, sys, json, time, subprocess, collections

R      = "/home/xqian/toolkit-dev/wcp-porting-img"
STAMP  = "20260925"
SUFFIX = ('.' + os.environ["PLAN_SUFFIX"]) if os.environ.get("PLAN_SUFFIX") else ""
HERE   = os.path.dirname(os.path.abspath(__file__))

# `.batch_pr_<run6>_<evt>[_<arm>].log` and nothing else.  Anchored at both ends, no
# path separator admitted, so a name can never address anything but a sibling of work/.
LOG = re.compile(r"^\.batch_pr_((\d{6})_(\d+)(?:_([A-Za-z0-9][A-Za-z0-9._-]*))?)\.log$")

TREES = {"pdvd": f"{R}/pdvd/work", "pdhd": f"{R}/pdhd/work"}

fails = []
def check(tree, n, ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'}  GATE O{n}: {msg}")
    if not ok: fails.append((tree, n))

def protected_lines():
    """Same union of all three PROTECTED.txt files the directory planner honours."""
    names, prefixes = set(), set()
    for p in (f"{R}/sbnd/sbnd_xin/scripts/retire/PROTECTED.txt",
              f"{R}/pdhd/scripts/retire/PROTECTED.txt",
              f"{R}/pdvd/scripts/retire/PROTECTED.txt"):
        if not os.path.exists(p): continue
        for line in open(p):
            line = line.strip()
            if not line or line.startswith("#"): continue
            for tok in line.split("\t")[0].split():
                (prefixes if tok.endswith("*") else names).add(tok.rstrip("*"))
    return names, prefixes

PROT_N, PROT_P = protected_lines()

def kept_arms(tree):
    """The arm tags THIS ROUND keeps, read from the directory planner's own keep list --
    never re-derived here, so the two planners cannot disagree about what is kept."""
    p = os.path.join(HERE, f"keep_{tree}_{STAMP}.txt")
    if not os.path.exists(p):
        sys.exit(f"missing {p} -- run plan_{STAMP}.py first")
    out = set()
    for line in open(p):
        d = os.path.basename(line.strip())
        if not d: continue
        m = re.match(r"^(\d{6})_(\d+)(?:_(.+))?$", d)
        if m: out.add(m.group(3) or "(bare)")
    return out

def plan_tree(tree, work):
    print(f"\n{'='*78}\n== {tree}   ({work})\n{'='*78}")
    KEPT  = kept_arms(tree)
    now   = time.time()
    dirs  = {d for d in os.listdir(work) if os.path.isdir(os.path.join(work, d))}

    cand, malformed, live_dir, kept_arm, prot, hot, notfile = [], [], [], [], [], [], []
    bytes_by_arm = collections.Counter()
    for f in os.listdir(work):
        if not f.startswith(".batch_pr_"): continue
        fp = os.path.join(work, f)
        m = LOG.match(f)
        if not m:
            malformed.append(f); continue
        target, arm = m.group(1), (m.group(4) or "(bare)")
        if os.path.islink(fp) or not os.path.isfile(fp):
            notfile.append(f); continue
        if target in dirs:                       # O1: the run is still on disk
            live_dir.append(f); continue
        if arm in KEPT:                          # O2: value-first, the ARM survives
            kept_arm.append(f); continue
        if arm in PROT_N or any(arm.startswith(x) for x in PROT_P):
            prot.append(f); continue             # O3: PROTECTED names it
        try: st = os.stat(fp)
        except OSError: notfile.append(f); continue
        if now - st.st_mtime < 3600:             # O4: nothing written in the last hour
            hot.append(f); continue
        cand.append(f); bytes_by_arm[arm] += st.st_size

    tot = sum(bytes_by_arm.values())
    print(f"  universe {len([f for f in os.listdir(work) if f.startswith('.batch_pr_')])} logs")
    print(f"  RELEASE  {len(cand)} logs = {tot/2**30:.2f} GiB")
    print(f"  held: {len(live_dir)} event dir alive | {len(kept_arm)} arm kept | "
          f"{len(prot)} PROTECTED | {len(hot)} written <1 h | {len(malformed)} malformed name | "
          f"{len(notfile)} not a regular file")
    print(f"  {'--- RELEASE ---':<28}{'logs':>7}{'GiB':>9}")
    per = collections.Counter()
    for f in cand: per[LOG.match(f).group(4) or "(bare)"] += 1
    for arm, n in sorted(bytes_by_arm.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {arm:<28}{per[arm]:>7}{arm and bytes_by_arm[arm]/2**30:>9.2f}")

    # ---- O5: every release name re-derives to a target that is REALLY absent
    bad = [f for f in cand if os.path.exists(os.path.join(work, LOG.match(f).group(1)))]
    check(tree, 5, not bad, f"no release log's event dir exists ({bad[:3] or 'clear'})")
    # ---- O6: no release name escapes work/ (belt and braces over the anchored regex)
    esc = [f for f in cand if os.path.dirname(os.path.realpath(os.path.join(work, f)))
                              != os.path.realpath(work)]
    check(tree, 6, not esc, f"every target resolves inside work/ ({esc[:3] or 'clear'})")
    # ---- O7: a malformed name is never released, and we say how many there were
    check(tree, 7, True, f"malformed names excluded, not guessed at ({len(malformed)})")
    # ---- O8: the kept-arm filter actually fired (a silent 0 would mean KEPT was empty)
    check(tree, 8, bool(KEPT), f"keep list loaded: {len(KEPT)} kept arm tags")

    out = os.path.join(HERE, f"files_orphanlogs_{tree}_{STAMP}{SUFFIX}.txt")
    with open(out, "w") as fh:
        for f in sorted(cand): fh.write(os.path.join(work, f) + "\n")
    print(f"  list: {out}  ({len(cand)} lines, {tot/2**30:.2f} GiB)")
    return tot

grand = 0
for t, w in TREES.items():
    grand += plan_tree(t, w)
print(f"\n{'='*78}\nGRAND TOTAL orphan logs {grand/2**30:.2f} GiB")
if fails:
    print(f"GATES FAILED: {fails}")
    sys.exit(1)
print("This script retired nothing.  Review the lists, then run the driver (CONFIRM=yes).")
