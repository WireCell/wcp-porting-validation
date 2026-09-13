#!/bin/bash
# ~/tmp sweep, cleanup ROUND B 2026-09-12 (doc 105 sec 13).  DRY RUN unless CONFIRM=yes.
#
#   ./sweep_tmp_20260912b.sh <tier>        tier = 1 or 2
#
# The owner, after round A executed: "These can be cleaned up: ... mg10 and the
# wt-merge/wt-premerge worktrees (~8 GiB)".  Round A had held all three because
# mg10 -- the master-merge validation -- never got a committed doc.  That is also
# why tier 1 FREEZES before it removes: ~/tmp/mg10/RESULTS.md, its logs and its
# scripts are the only record that validation left (M13).
set -u
T=/home/xqian/tmp
R=/home/xqian/toolkit-dev/wcp-porting-img
TK=/home/xqian/toolkit-dev/toolkit
D="$(cd "$(dirname "$0")" && pwd)"
STAMP=20260912b
REC="$R/sbnd/sbnd_xin/archive/records/cleanup-${STAMP}/tmp-tier1"
TIER=${1:?tier (1 or 2)}
CONFIRM=${CONFIRM:-no}

newest_age_min() {
  python3 - "$1" <<'PY'
import os, sys, time
m = 0
for cur, s, fs in os.walk(sys.argv[1]):
    s[:] = [x for x in s if not os.path.islink(os.path.join(cur, x))]
    for f in fs:
        try: m = max(m, os.lstat(os.path.join(cur, f)).st_mtime)
        except OSError: pass
print(int((time.time() - m) / 60) if m else 99999)
PY
}
cwd_inside() {
  for p in /proc/[0-9]*; do
    c=$(readlink "$p/cwd" 2>/dev/null) || continue
    case "$c/" in "$1"/*) echo "${p#/proc/}"; return 0;; esac
  done
  return 1
}
freeze() {   # $1 = tag, $2 = dir: manifest (SHA-256 per file) + tar.zst of the non-heavy layer
  RETIRE_OUT="$(dirname "$REC")" python3 - "$1" "$2" <<'PY'
import sys, os, importlib.util
here = "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/scripts/retire"
spec = importlib.util.spec_from_file_location("ar", f"{here}/archive_records_20260912b.py")
ar = importlib.util.module_from_spec(spec); spec.loader.exec_module(ar)
tag, src = sys.argv[1], sys.argv[2]
r = ar.archive_one(("tmp-tier1", tag, src))
print(f"   froze {tag}: {r[1]} files hashed, {r[2]} carried, {r[3]} dropped, {r[4]} links, {r[5]/2**20:.1f} MiB")
man = os.path.join(ar.OUT, "tmp-tier1", tag + ".manifest.tsv")
assert os.path.getsize(man) > 0 or r[1] == 0, "empty manifest"
PY
}

case "$TIER" in
 1)
  echo "== TIER 1: the merge validation -- ~/tmp/mg10 and the two toolkit worktrees"
  GO_WT=()
  for w in $T/wt-merge $T/wt-premerge; do
    [ -d "$w" ] || { echo "   (absent) $w"; continue; }
    git -C "$TK" worktree list --porcelain | grep -qx "worktree $w" || { echo "   REFUSING $w: not a registered toolkit worktree"; exit 5; }
    h=$(git -C "$w" rev-parse HEAD)
    git -C "$TK" merge-base --is-ancestor "$h" apply-pointcloud || {
      echo "   REFUSING $w: HEAD $h is not on apply-pointcloud -- removing it could lose a commit"; exit 6; }
    n=$(git -C "$w" status --porcelain --untracked-files=no | wc -l)
    [ "$n" = 0 ] || { echo "   REFUSING $w: $n modified tracked file(s)"; exit 7; }
    u=$(git -C "$w" status --porcelain | grep '^??' | grep -v '^?? build.rc$' | wc -l)
    [ "$u" = 0 ] || { echo "   REFUSING $w: untracked content other than build.rc"; exit 8; }
    a=$(newest_age_min "$w"); [ "$a" -ge 60 ] || { echo "   REFUSING $w: written $a min ago"; exit 9; }
    if pid=$(cwd_inside "$w"); then echo "   REFUSING $w: process $pid has its cwd inside"; exit 10; fi
    echo "   $(du -sh "$w" | cut -f1) | HEAD ${h:0:8} on apply-pointcloud | idle $a min | $w"
    GO_WT+=("$w")
  done
  M=$T/mg10
  if [ -d "$M" ]; then
    a=$(newest_age_min "$M"); [ "$a" -ge 60 ] || { echo "   REFUSING $M: written $a min ago"; exit 9; }
    if pid=$(cwd_inside "$M"); then echo "   REFUSING $M: process $pid has its cwd inside"; exit 10; fi
    # nothing outside mg10 itself may name it or the worktrees (cleanup docs excepted)
    ext=$(grep -rlI -E 'tmp/mg10|tmp/wt-merge|tmp/wt-premerge' $R/pdvd/docs $R/pdhd/docs $R/sbnd/sbnd_xin/docs \
          $R/pdvd/scripts $R/pdhd/scripts $R/sbnd/sbnd_xin/scripts $TK/clus/docs 2>/dev/null | grep -v -E '/retire/|cleanup' | wc -l)
    [ "$ext" = 0 ] || { echo "   REFUSING: $ext file(s) outside ~/tmp/mg10 name it or the worktrees"; exit 11; }
    echo "   $(du -sh "$M" | cut -f1) | idle $a min | $M  (RESULTS.md, logs, scripts frozen first)"
  fi
  if [ "$CONFIRM" = yes ]; then
    mkdir -p "$REC"
    # A FAILED FREEZE ABORTS BEFORE ANY REMOVAL.  Measured 2026-09-12: the first
    # negative control pointed REC at NOSUCH- and proved nothing, because
    # archive_one() makedirs its own target -- so "a manifest exists" cannot tell a
    # good freeze from a freeze into the wrong place.  What is tested now is the
    # freeze's own exit status, and nothing is removed unless it is 0.
    if [ -d "$M" ]; then
      freeze mg10 "$M" || { echo "   REFUSING: the freeze of $M failed -- nothing removed"; exit 13; }
    fi
    for w in "${GO_WT[@]}"; do
      b=$(basename "$w"); mkdir -p "$REC/$b"
      cp -p "$w/build.rc" "$REC/$b/build.rc" 2>/dev/null
      git -C "$w" rev-parse HEAD > "$REC/$b/HEAD"
      git -C "$TK" worktree remove --force "$w" && echo "   removed worktree $w (build.rc + HEAD frozen)" \
        || { echo "   git refused $w -- stopping"; exit 12; }
    done
    if [ -d "$M" ]; then
      [ -s "$REC/mg10.manifest.tsv" ] || { echo "   REFUSING to remove $M: its manifest is missing"; exit 13; }
      rm -rf "$M" && echo "   removed $M"
    fi
  else
    echo "   would freeze ~/tmp/mg10 + build.rc/HEAD into $REC, then 'git worktree remove --force' x${#GO_WT[@]} and rm ~/tmp/mg10"
  fi
  ;;
 2)
  echo "== TIER 2: top-level pins of the arms round B released (value-first; run AFTER the work release)"
  python3 "$D/pins_of_released_arms_${STAMP}.py" | sed 's/^/   /'
  mapfile -t P < <(python3 "$D/pins_of_released_arms_${STAMP}.py" --list)
  GO=()
  for d in "${P[@]}"; do
    kb=$(find "$d" -type f ! -name '*.so*' -printf '%k\n' 2>/dev/null | awk '{s+=$1}END{print s+0}')
    if [ "${kb:-0}" -ge 2048 ]; then echo "   SKIP $d: ${kb} KiB of non-.so content -- a full snapshot, kept"; continue; fi
    GO+=("$d")
  done
  [ "${#GO[@]}" -gt 0 ] || { echo "   nothing releasable"; exit 0; }
  if [ "$CONFIRM" = yes ]; then rm -rf "${GO[@]}"; echo "   removed ${#GO[@]} pin dir(s)"
  else printf '   would remove: %s\n' "${GO[@]}"; fi
  ;;
 *) echo "tiers 1 and 2"; exit 1;;
esac
[ "$CONFIRM" = yes ] || echo -e "\nDRY RUN.  Re-run with CONFIRM=yes to execute."
