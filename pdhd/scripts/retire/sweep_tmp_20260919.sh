#!/bin/bash
# ~/tmp sweep, cleanup ROUND D 2026-09-16 (doc sbnd_xin/111).  DRY RUN unless CONFIRM=yes.
#
#   ./sweep_tmp_20260919.sh
#   CONFIRM=yes ./sweep_tmp_20260919.sh
#
# Fork of sweep_tmp_20260913.sh.  Round D changes: the permanent pins follow production
# (libpin_d102, d102m-libsnap, pdhdstm_libpin); the held prefixes are the live peer's
# (d111 d112 d113 d103 d108); the census's fifth class (closed-round scratch) is in the
# tier; and the registered git worktrees the census lists (tmp_worktrees_20260919.txt)
# are removed with `git worktree remove` after the rm, only if clean and their HEAD is
# on the pushed remote head -- never rm -rf, which would leave a stale registration.
#
# --- round C's header follows ---
# ~/tmp sweep, cleanup ROUND C 2026-09-13 (doc 106).
# ORDER: plan_20260913.py (tier1_*) -> tmp_census_20260913.py (tmp_tier_20260913.txt)
# -> the work release -> this.  The census reads "alive" from the work tier files,
# so it can run before the work release; this sweep re-runs it at confirm time.
#
# What it removes: the census's FREE units -- dead-round pins, the prep dirs of
# released arms, idle session scratchpads, ~/tmp/h28's scratch and old harness
# leftovers.  Each is frozen first (archive_tmp_20260913.py: SHA-256 manifest of
# every file + the small text layer), and NOTHING is removed unless that freeze
# exits 0 and wrote a fresh manifest for every unit (round B's lesson: its first
# negative control passed because the archiver makedirs its own target).
set -u
T=/home/xqian/tmp
R=/home/xqian/toolkit-dev/wcp-porting-img
D=/home/xqian/toolkit-dev/wcp-porting-img/pdhd/scripts/retire
STAMP=20260919
# CENSUS_SUFFIX (round D review): the owner's pre-sweep re-census writes tmp_tier_${STAMP}.<suffix>.txt,
# so the committed plan-time tier is never overwritten; this sweep then acts on the suffixed file.
CS=${CENSUS_SUFFIX:+.$CENSUS_SUFFIX}
TF="$D/tmp_tier_${STAMP}${CS}.txt"
CONFIRM=${CONFIRM:-no}
export RETIRE_OUT=${RETIRE_OUT:-$R/sbnd/sbnd_xin/archive/records/cleanup-${STAMP}/tmp}
THIS=87534ba0-ada7-4811-b7af-02fbd8996b32
PERM="$T/d102/libpin_d102 $T/d102m-libsnap $T/pdhdstm_libpin"

[ -s "$TF" ] || { echo "REFUSING: $TF missing or empty -- run tmp_census_${STAMP}.py first"; exit 2; }

# ---- the WORK release must already have happened.  The census computes "alive"
# as disk minus tier1_*, i.e. the FUTURE state, so without this gate a sweep run
# before (or instead of) retire_20260913.sh would remove the pins and preps of arms
# that are still on disk -- the value-first failure this whole test exists to stop.
if [ "${REQUIRE_WORK:-yes}" = yes ]; then
  for t in sbnd pdvd pdhd; do
    left=$(while read -r p; do [ -e "$p" ] && echo "$p"; done < "$D/tier1_${t}_${STAMP}.txt" | wc -l)
    [ "$left" = 0 ] || { echo "REFUSING: $left $t work dir(s) from tier1_${t}_${STAMP}.txt are still on disk -- run retire_${STAMP}.sh first"; exit 4; }
  done
  echo "== work release done: every tier1 dir of sbnd/pdvd/pdhd is gone"
fi

# ---- INTERLOCK A: re-census at confirm time; the unit list must not have moved
if [ "$CONFIRM" = yes ] && [ "${RECENSUS:-yes}" = yes ]; then
  python3 "$D/tmp_census_${STAMP}.py" --list > "$D/tmp_tier_${STAMP}.confirm.txt" 2> "$D/tmp_census_${STAMP}.confirm.err" \
    || { echo "REFUSING: the re-census failed ($D/tmp_census_${STAMP}.confirm.err)"; exit 10; }
  if ! cmp -s "$TF" "$D/tmp_tier_${STAMP}.confirm.txt"; then
    echo "REFUSING: the ~/tmp unit list moved between census and confirm -- a live writer, or a"
    echo "work release that has not happened yet.  diff:"; diff "$TF" "$D/tmp_tier_${STAMP}.confirm.txt" | head -8
    exit 11
  fi
  echo "== INTERLOCK A: re-census unchanged ($(wc -l < "$TF") units)"
fi

# ---- per-unit guards
n=0; miss=0
while read -r p; do
  [ -n "$p" ] || continue
  n=$((n+1))
  case "$p" in "$T"/?*) ;; *) echo "REFUSING: not under ~/tmp: $p"; exit 5;; esac
  if [ ! -e "$p" ] && [ ! -L "$p" ]; then miss=$((miss+1)); continue; fi
  case "$p/" in "$T"/d111*|"$T"/d112*|"$T"/d113*|"$T"/d103*|"$T"/d108*|*"/$THIS/"*) echo "REFUSING: live unit $p"; exit 6;; esac
  for k in $PERM; do
    case "$k/" in "$p/"*) echo "REFUSING: $p is or contains the permanent pin $k"; exit 6;; esac
  done
done < "$TF"
[ "$miss" = 0 ] || { echo "REFUSING: $miss of $n units are already gone -- a stale list; re-census"; exit 3; }
for k in $PERM; do [ -f "$k/libWireCellClus.so" ] || { echo "REFUSING: permanent pin $k is not intact"; exit 7; }; done
python3 - "$TF" "$THIS" <<'PY' || exit 9
import os, sys, time
tf, this = sys.argv[1], sys.argv[2]
units = [l.strip() for l in open(tf) if l.strip()]
now, hot, cwd = time.time(), [], []
for u in units:
    m = 0
    for cur, subs, fs in (os.walk(u) if os.path.isdir(u) and not os.path.islink(u) else [(os.path.dirname(u), [], [os.path.basename(u)])]):
        subs[:] = [s for s in subs if not os.path.islink(os.path.join(cur, s))]
        for f in fs:
            try: m = max(m, os.lstat(os.path.join(cur, f)).st_mtime)
            except OSError: pass
    if now - m < 3600: hot.append(u)
for pid in filter(str.isdigit, os.listdir("/proc")):
    try: c = os.readlink(f"/proc/{pid}/cwd") + "/"
    except OSError: continue
    cwd += [(pid, u) for u in units if c.startswith(u.rstrip("/") + "/")]
if hot: print(f"REFUSING: {len(hot)} unit(s) written in the last hour, e.g. {hot[0]}")
if cwd: print(f"REFUSING: process {cwd[0][0]} has its cwd inside {cwd[0][1]}")
print(f"   guards: {len(units)} units, none written in the last hour, no process cwd inside" if not (hot or cwd) else "")
sys.exit(1 if hot or cwd else 0)
PY
echo "   $n units | permanent pins intact: $PERM"

if [ "$CONFIRM" = yes ]; then
  start=$(date +%s)
  python3 "$D/archive_tmp_${STAMP}.py" "$TF" > "$D/archive_tmp_${STAMP}.log" 2>&1
  rc=$?
  [ "$rc" = 0 ] || { echo "REFUSING: the freeze failed (rc=$rc; $D/archive_tmp_${STAMP}.log) -- nothing removed"; exit 13; }
  python3 - "$TF" "$RETIRE_OUT" "$start" <<'PY' || { echo "REFUSING: a unit has no fresh manifest -- nothing removed"; exit 13; }
import os, sys
tf, out, start = sys.argv[1], sys.argv[2], float(sys.argv[3])
T = "/home/xqian/tmp"
units = [l.strip() for l in open(tf) if l.strip()]
def ok(u):
    m = os.path.join(out, os.path.relpath(u, T).replace("/", "__") + ".manifest.tsv")
    return os.path.exists(m) and os.path.getmtime(m) >= start - 1
bad = [u for u in units if not ok(u)]
print(f"   record gate: {len(units) - len(bad)}/{len(units)} units have a manifest written by this freeze in {out}")
sys.exit(1 if bad else 0)
PY
  echo "   REMOVING..."
  xargs -a "$TF" -d '\n' rm -rf
  echo "   done, rc=$?"
  # pin containers left empty (d42_libpin/, d47_libpin/, ...): rmdir only, never rm -r
  while read -r p; do
    c=$(dirname "$p")
    case "$c" in "$T"/*_libpin|"$T"/*-libsnap) rmdir "$c" 2>/dev/null && echo "   removed empty container $c";; esac
  done < "$TF"
  # registered git worktrees: clean + HEAD on the pushed remote head -> git worktree remove
  REMOTE=$(cat "$D/remote_head_${STAMP}.txt")
  while read -r w; do
    [ -n "$w" ] || continue
    case "$w/" in "$T"/d111*|"$T"/d112*|"$T"/d113*|"$T"/d103*|"$T"/d108*) echo "   WT held, skipped: $w"; continue;; esac
    if [ -n "$(git -C "$w" status --porcelain 2>/dev/null)" ]; then echo "   WT dirty, skipped: $w"; continue; fi
    h=$(git -C "$w" rev-parse HEAD)
    if ! git -C "$R" merge-base --is-ancestor "$h" "$REMOTE"; then echo "   WT HEAD $h not on the remote, skipped: $w"; continue; fi
    git -C "$R" worktree remove "$w" && echo "   WT removed: $w (HEAD ${h:0:8} on the remote)"
    c=$(dirname "$w"); rmdir "$c" 2>/dev/null && echo "   removed empty container $c"
  done < "$D/tmp_worktrees_${STAMP}${CS}.txt"
else
  echo "   DRY RUN -- would freeze $n units into $RETIRE_OUT, then remove them.  First 3:"
  head -3 "$TF" | sed 's/^/      /'
fi

echo
echo "POST-STATE: ~/tmp $(du -sh $T 2>/dev/null | cut -f1) | free $(df -h /home/xqian | awk 'NR==2{print $4}')"
for k in $PERM; do echo "   $k: $([ -f "$k/libWireCellClus.so" ] && echo intact || echo MISSING)"; done
[ "$CONFIRM" = yes ] || echo -e "\nDRY RUN.  Re-run with CONFIRM=yes to execute."
