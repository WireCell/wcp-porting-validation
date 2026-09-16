#!/bin/bash
# Cleanup ROUND D follow-up 2026-09-16 (doc sbnd_xin/111 sec 14) -- FILE-level delete driver.
# DRY RUN unless CONFIRM=yes.
#
#   ./retire_files_20260916b.sh
#   CONFIRM=yes ./retire_files_20260916b.sh
#
# Owner, 2026-09-16: "We can remove the 40 G SP frames inside p98von, and remove the sbnd's final
# icluster npz files."  Two sets, planned and frozen by plan_files_20260916b.py:
#   pdvd-p98von-frames   pdvd/work/*_p98von/protodune-sp-dnnroi-frames-anode*.tar.bz2
#   sbnd-d102m-icluster  sbnd_xin/work-mcp{1,2}k-d102m/{evt,g}<N>/icluster-apa*.npz  + the in-arm
#                        ql_evt<N>/icluster-*.npz symlinks onto them (they would dangle)
# ORDER: plan_files_20260916b.py --hash (manifest) -> this.  Gates, each with its own exit code:
#   10  the confirm-time re-plan fails a gate (F1-F5: regular single-name files, no stray symlink
#       onto a target, in-arm links resolve, nothing holds a target open, pvdimg inode-disjoint)
#   11  the re-planned lists differ from the plan-time lists (something moved)
#   14  a target has no manifest row with its size, or the frozen links file differs from the list
#   15  the product the target was an INPUT to is not complete: every p98von event keeps 16 non-empty
#       clusters-apa archives and pvdimg 16; every ql_evt<N> of the two arms keeps a non-empty
#       pctree-evt<N>.tar.gz
#   16  the broken-symlink count of the tree rose after the delete
set -u
D="$(cd "$(dirname "$0")" && pwd)"
R=/home/xqian/toolkit-dev/wcp-porting-img
STAMP=20260916b
SETS="pdvd-p98von-frames sbnd-d102m-icluster"
CONFIRM=${CONFIRM:-no}
REC=${REC_OVERRIDE:-$R/sbnd/sbnd_xin/archive/records/cleanup-${STAMP}}

for s in $SETS; do [ -s "$D/files_${s}_${STAMP}.txt" ] || { echo "REFUSING: no plan list for $s -- run plan_files_${STAMP}.py"; exit 2; }; done

if [ "$CONFIRM" = yes ]; then
  echo "== re-plan at confirm time"
  PLAN_SUFFIX=confirm python3 "$D/plan_files_${STAMP}.py" > "$D/plan_files_${STAMP}.confirm.out" 2>&1 \
    || { echo "REFUSING: a gate fails on re-plan ($D/plan_files_${STAMP}.confirm.out)"; exit 10; }
  for s in $SETS; do
    for k in files links; do
      cmp -s "$D/${k}_${s}_${STAMP}.txt" "$D/${k}_${s}_${STAMP}.confirm.txt" \
        || { echo "REFUSING: ${k}_${s} moved between plan and confirm"; exit 11; }
    done
  done
  echo "   OK: F1-F5 PASS, lists unchanged"
fi

python3 - "$D" "$STAMP" "$REC" "$SETS" <<'PY' || exit $?
import os, sys, glob, re
D, STAMP, REC, SETS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split()
R = "/home/xqian/toolkit-dev/wcp-porting-img"
for s in SETS:
    files = [l.strip() for l in open(f"{D}/files_{s}_{STAMP}.txt") if l.strip()]
    man = f"{REC}/{s}.manifest.tsv"
    rows = {}
    if os.path.exists(man):
        for l in open(man):
            p, sz, h = l.rstrip("\n").split("\t"); rows[p] = int(sz)
    miss = [p for p in files if rows.get(p) != (os.lstat(p).st_size if os.path.lexists(p) else -1)]
    if miss:
        print(f"REFUSING: {len(miss)} of {len(files)} {s} targets have no manifest row with their size in {REC}"); sys.exit(14)
    lk = f"{REC}/{s}.links.txt"
    if not os.path.exists(lk) or open(lk).read() != open(f"{D}/links_{s}_{STAMP}.txt").read():
        print(f"REFUSING: the frozen links file of {s} differs from the list"); sys.exit(14)
    print(f"   record gate {s}: {len(files)}/{len(files)} manifest rows, links file identical")
# completeness of the products these files were inputs to
bad = []
for ev in sorted(glob.glob(f"{R}/pdvd/work/*_p98von")):
    base = ev[:-len("_p98von")]
    for arm in (ev, base + "_pvdimg"):
        arc = [a for a in glob.glob(f"{arm}/clusters-apa-anode*-ms-*.tar.gz") if os.path.getsize(a) > 0]
        if len(arc) != 16: bad.append(arm)
for arm in ("work-mcp1k-d102m", "work-mcp2k-d102m"):
    for q in glob.glob(f"{R}/sbnd/sbnd_xin/{arm}/ql_evt*"):
        n = re.sub(r"^ql_evt", "", os.path.basename(q))
        pc = f"{q}/pctree-evt{n}.tar.gz"
        if not (os.path.isfile(pc) and os.path.getsize(pc) > 0): bad.append(q)
if bad:
    print(f"REFUSING: {len(bad)} product dir(s) incomplete, e.g. {bad[0]}"); sys.exit(15)
print(f"   product gate: 120 p98von + 120 pvdimg events have 16 imaging archives; every ql_evt of mcp1k/mcp2k d102m has its pctree")
PY

before_pdvd=$(find "$R/pdvd/work" -xtype l 2>/dev/null | wc -l)
before_sbnd=$(find "$R/sbnd/sbnd_xin" -xtype l 2>/dev/null | wc -l)
echo "   broken symlinks before: pdvd $before_pdvd, sbnd_xin $before_sbnd"
if [ "$CONFIRM" != yes ]; then
  for s in $SETS; do echo "   DRY RUN $s: $(wc -l < "$D/files_${s}_${STAMP}.txt") files, $(grep -c . "$D/links_${s}_${STAMP}.txt") links"; done
  echo "DRY RUN.  Re-run with CONFIRM=yes to execute."; exit 0
fi
for s in $SETS; do
  echo "== DELETING $s"
  cut -f1 "$D/links_${s}_${STAMP}.txt" | while read -r l; do [ -n "$l" ] && [ -L "$l" ] && rm -f "$l"; done
  xargs -a "$D/files_${s}_${STAMP}.txt" -d '\n' rm -f
  echo "   files rc=$?"
done
# the sbnd evt<N>/ and g<N>/ dirs that held only npz are now empty: rmdir only, never rm -r
n=0; for d in "$R"/sbnd/sbnd_xin/work-mcp1k-d102m/{evt,g}[0-9]* "$R"/sbnd/sbnd_xin/work-mcp2k-d102m/{evt,g}[0-9]*; do
  [ -d "$d" ] && rmdir "$d" 2>/dev/null && n=$((n+1)); done
echo "   removed $n empty evt/g dirs"
after_pdvd=$(find "$R/pdvd/work" -xtype l 2>/dev/null | wc -l)
after_sbnd=$(find "$R/sbnd/sbnd_xin" -xtype l 2>/dev/null | wc -l)
echo "   broken symlinks after: pdvd $after_pdvd, sbnd_xin $after_sbnd"
[ "$after_pdvd" -le "$before_pdvd" ] && [ "$after_sbnd" -le "$before_sbnd" ] || { echo "FAIL: broken symlinks rose"; exit 16; }
for s in $SETS; do left=$(while read -r p; do [ -e "$p" ] && echo "$p"; done < "$D/files_${s}_${STAMP}.txt" | wc -l); echo "   $s left on disk: $left"; done
echo "POST-STATE: free $(df -h /home/xqian | awk 'NR==2{print $4}')"
