#!/bin/bash
# Cleanup ROUND K 2026-09-25 (doc sbnd_xin/125) -- FILE-level delete driver.
# DRY RUN unless CONFIRM=yes.
#
#   ./retire_files_20260925.sh
#   CONFIRM=yes ./retire_files_20260925.sh
#
# Owner, 2026-09-25 ("Both (~90 GB)"):
#   sbnd-icluster-split  56408 files  67.65 GiB + 24068 links   evt*/icluster copies, d123base x7 + d123lgop x4
#   sbnd-base-qlevt      48510 files  23.24 GiB + 32340 links   d123base x7 ql_evt*/ (reco1-flash Q/L output)
# g*/ is never touched (plan_files gate F8).
#
# ORDER: the ~/tmp sweep FIRST (it releases ~/tmp/d124/nudge, whose 648 links resolve onto set-A
# targets and make gate F2 fail -- measured at plan time), then plan_files_20260925.py --hash, then this.
# Fork of retire_files_20260921b.sh (round J).  What round K adds: a set carries a LINKS list (the
# ql_evt*/ symlinks that point at its own targets, gate F9) removed after the files, and the per-event
# dirs the release empties (evt<N>/, and base's ql_evt<N>/) are rmdir'ed -- rmdir only, never rm -r, so
# a dir holding anything this round did not list stays.
# Exit codes:
#   2   no plan list for a set
#   3   N of M targets already gone (stale list; re-plan)
#   4   a target is a symlink or not a regular file / a link-target is not a symlink
#   5   a target or link is outside its set's allow-list, under an input/record/reference tree, or under g<N>/
#   6   the record is missing -- run plan_files_20260925.py --hash first
#   10  the confirm-time re-plan fails a gate (F1-F9)
#   11  the re-planned lists differ from the plan-time lists (something moved)
#   14  record gate: a target has no manifest row carrying its own size AND sha256, or a link no row
#   16  the broken-symlink count of a tree ROSE after the delete
set -u
D="$(cd "$(dirname "$0")" && pwd)"
R=/home/xqian/toolkit-dev/wcp-porting-img
STAMP=20260925
SETS="sbnd-icluster-split sbnd-base-qlevt"
CONFIRM=${CONFIRM:-no}
REC=${REC_OVERRIDE:-$R/sbnd/sbnd_xin/archive/records/cleanup-${STAMP}}

for s in $SETS; do
  [ -s "$D/files_${s}_${STAMP}.txt" ] \
    || { echo "REFUSING: no plan list for $s -- run plan_files_${STAMP}.py"; exit 2; }
done

for t in pdvd/work pdhd/work sbnd/sbnd_xin; do
  k=$(echo "$t" | tr / _)
  find "$R/$t" -xtype l 2>/dev/null | wc -l > "$D/prebroken_files_${k}_${STAMP}.txt"
done

if [ "$CONFIRM" = yes ]; then
  echo "== re-plan at confirm time"
  PLAN_SUFFIX=confirm python3 "$D/plan_files_${STAMP}.py" \
      > "$D/plan_files_${STAMP}.confirm.out" 2>&1 \
    || { echo "REFUSING: a gate fails on re-plan ($D/plan_files_${STAMP}.confirm.out)"; exit 10; }
  for s in $SETS; do
    for x in "" ".links"; do
      cmp -s "$D/files_${s}${x}_${STAMP}.txt" "$D/files_${s}${x}_${STAMP}.confirm.txt" \
        || { echo "REFUSING: the $s$x list moved between plan and confirm"; exit 11; }
    done
  done
  echo "   OK: F1-F9 PASS, lists unchanged"
fi

python3 - "$D" "$STAMP" "$REC" "$SETS" "$CONFIRM" "$R" <<'PY' || exit $?
import os, re, sys
D, STAMP, REC, SETS, CONFIRM, R = sys.argv[1:7]
S = f"{R}/sbnd/sbnd_xin"
BASE = [f"{S}/work-{s}-d123base" for s in ("mcp1k","mcp2k","ncpi0","nuecc48","r3cv","r3nue","r3off")]
LGOP = [f"{S}/work-{s}-d123lgop" for s in ("mcp1k","mcp2k","r3cv","r3off")]
ALLOW = {"sbnd-icluster-split": BASE + LGOP, "sbnd-base-qlevt": BASE}
FORBIDDEN = ("input_files_reco1", "xin-round3-samples", "input_data_", "/archive/",
             "/products/", "/ref/", "/docs/", "/scripts/")
GRP = re.compile(r"^g\d+$")
def outside(p, allow):
    ap = os.path.join(os.path.realpath(os.path.dirname(p)), os.path.basename(p))
    return (any(k in ap for k in FORBIDDEN) or not any(ap.startswith(os.path.realpath(a) + "/") for a in allow)
            or any(GRP.match(x) for x in os.path.relpath(ap, S).split("/")))
rc = 0
for s in SETS.split():
    files = [l.strip() for l in open(f"{D}/files_{s}_{STAMP}.txt") if l.strip()]
    lf = f"{D}/files_{s}.links_{STAMP}.txt"
    links = [l.strip() for l in open(lf) if l.strip()] if os.path.exists(lf) else []
    gone = [f for f in files + links if not os.path.lexists(f)]
    if gone:
        print(f"REFUSING: {len(gone)} of {len(files)+len(links)} {s} targets already gone -- re-plan"); sys.exit(3)
    bad = [f for f in files if os.path.islink(f) or not os.path.isfile(f)] + [l for l in links if not os.path.islink(l)]
    if bad:
        print(f"REFUSING: {len(bad)} {s} targets have the wrong type, e.g. {bad[0]}"); sys.exit(4)
    off = [f for f in files + links if outside(f, ALLOW[s])]
    if off:
        print(f"REFUSING: {len(off)} {s} targets are outside the allow-list / under g<N>/, e.g. {off[0]}"); sys.exit(5)
    man, lnk = f"{REC}/{s}.manifest.tsv", f"{REC}/{s}.links.tsv"
    if not (os.path.exists(man) and os.path.exists(lnk)):
        print(f"REFUSING: no record for {s} ({man}) -- run plan_files_{STAMP}.py --hash first"); sys.exit(6)
    rows = {}
    for line in open(man):
        if line.startswith("#"): continue
        p, size, digest = line.rstrip("\n").split("\t")
        rows[os.path.join(R, p)] = (int(size), digest)
    lrows = {os.path.join(R, l.split("\t")[0]) for l in open(lnk) if not l.startswith("#")}
    missing = [f for f in files if f not in rows or rows[f][0] != os.path.getsize(f) or len(rows[f][1]) != 64]
    missing += [l for l in links if l not in lrows]
    if missing:
        print(f"REFUSING: {len(missing)} {s} targets lack a record row, e.g. {missing[0]}"); sys.exit(14)
    tot = sum(os.path.getsize(f) for f in files)
    print(f"== {s}: {len(files)} files, {tot/2**30:.2f} GiB + {len(links)} links, frozen 1:1 in "
          f"{os.path.basename(man)} / {os.path.basename(lnk)}")
    if CONFIRM != "yes":
        print(f"   DRY RUN -- nothing removed.  e.g. {os.path.relpath(files[0], R)}")
        continue
    n = nl = nd = 0
    for f in files:
        try: os.unlink(f); n += 1
        except OSError as e: print(f"   FAILED {f}: {e}"); rc = 1
    for l in links:
        try: os.unlink(l); nl += 1
        except OSError as e: print(f"   FAILED {l}: {e}"); rc = 1
    for d in sorted({os.path.dirname(x) for x in files + links}, key=len, reverse=True):
        try: os.rmdir(d); nd += 1          # only if now EMPTY
        except OSError: pass
    print(f"   removed {n}/{len(files)} files, {nl}/{len(links)} links, {nd} emptied per-event dirs")
sys.exit(rc)
PY
rc=$?

if [ "$CONFIRM" = yes ]; then
  for t in pdvd/work pdhd/work sbnd/sbnd_xin; do
    k=$(echo "$t" | tr / _)
    before=$(cat "$D/prebroken_files_${k}_${STAMP}.txt")
    after=$(find "$R/$t" -xtype l 2>/dev/null | wc -l)
    echo "== $t broken symlinks: $before -> $after"
    [ "$after" -le "$before" ] || { echo "REFUSING (after the fact): $t broken-symlink count ROSE"; exit 16; }
  done
fi
echo "done (CONFIRM=$CONFIRM) rc=$rc"
exit $rc
