#!/bin/bash
# ~/tmp sweep, cleanup round 2026-09-05.  DRY RUN unless CONFIRM=yes.
#
# Doc 100's rule, refined.  That round removed only whole lib*/ dirs and build
# trees and never a *.log/.txt/.md/.json/.zip, because a libsnap is regenerable
# from the commit its doc records and a gate log is not.  Measuring the scratch
# dirs here showed records and bulk are cleanly separable (doc25gate is 8.86 GiB
# of bulk and 0.00 GiB of records), so tier 3 drops ONLY non-record files and
# every log/txt/md/json/tsv survives in place.
#
# LIVENESS IS FROM ps, NEVER FROM AGE (doc 100: an 18 GiB scratchpad untouched
# for two days belonged to a running session).  Verified at write time:
#   PID 1314142  claude --resume 7117f9b1...  -> 17.46 GiB scratchpad, KEPT
#   PID 2711374  claude --resume b48747db...  ->  0.51 GiB scratchpad, KEPT
#   PID 1440435  peer session reading pdvd d45* -> ~/tmp/d45, d45_libpin KEPT
set -u
T=/home/xqian/tmp
CONFIRM=${CONFIRM:-no}
# Whole-dir removals refuse on an already-gone target: that is the signature of
# pointing at a previous round's list (doc 100 catch 2), and it is also what a
# second CONFIRM=yes run of this script would look like.
run() {
  for p in "$@"; do
    if [ ! -e "$p" ]; then
      echo "   REFUSING: $p is already gone -- re-check which round's list this is."
      exit 3
    fi
  done
  if [ "$CONFIRM" = yes ]; then rm -rf "$@"; else echo "   would remove: $*"; fi
}

echo "=== TIER 1: build trees (regenerable, no records) ==="
for d in "$T/claude-25225/cmrel1"; do [ -e "$d" ] && { du -sh "$d"; run "$d"; }; done

echo "=== TIER 2: pinned binaries of CLOSED rounds ==="
# A pin goes with its arms.  These six back rounds whose arms are gone or whose
# docs record the commit to rebuild from.  WARNING (M1 shape): a missing
# LD_LIBRARY_PATH entry is SILENTLY ignored, so a script that still names one of
# these will fall back to live local/lib with no error.  The scripts that name
# each pin are listed by the report below -- those arms become re-readable but
# NOT re-runnable.
for d in pinlib d31r6lib d39r2_libpin prod0901b-libsnap d99r2-libsnap d100regen-libsnap; do
  [ -e "$T/$d" ] && { du -sh "$T/$d"; run "$T/$d"; }
done

echo "=== TIER 3: embedded libsnaps (*.so) in CLOSED-round scratch dirs ==="
# NARROWED after characterising the bulk at file granularity.  The first draft
# deleted everything that was not a *.log/.txt/.md/.json/.tsv, and that was
# wrong twice over:
#   * d44sp is 64 .npy frames that doc 44's REPRO BLOCK names as an INPUT
#     (line 41, --frames /home/xqian/tmp/d44sp/039252_16).  A file is a record
#     by FUNCTION, not by extension.  d44sp is dropped from this tier entirely.
#   * doc37's 4.90 GiB is a long tail of tiny install-tree files (wav, xml, xz,
#     typed, ...) plus .h/.cxx/.py/.sh sources, none of it safely classified.
# So this tier now removes ONLY *.so -- the libsnap class doc 100 sanctioned,
# regenerable from the commit each doc records.  22.91 GiB, zero ambiguity.
for d in doc25gate doc25r13 doc37 d39 doc25r12; do
  [ -d "$T/$d" ] || continue
  n=$(find "$T/$d" -type f -name '*.so*' | wc -l)
  kb=$(find "$T/$d" -type f -name '*.so*' -printf '%k\n' | awk '{s+=$1}END{print s+0}')
  other=$(find "$T/$d" -type f ! -name '*.so*' | wc -l)
  printf '   %-12s %5d .so %8.2f GiB   (%d non-.so files untouched)\n' \
         "$d" "$n" "$(echo "$kb/1048576"|bc -l)" "$other"
  [ "$CONFIRM" = yes ] && find "$T/$d" -type f -name '*.so*' -delete
done

echo "=== TIER 4: scratchpads of DEAD sessions (live ones verified from ps) ==="
for d in "$T"/claude-25225/-home-*/*/; do
  id=$(basename "$d")
  case "$id" in
    7117f9b1-c512-4158-8c4b-eb56f686d997|b48747db-62fd-47bf-b819-eb159eebd55f|217ba691-e575-45a1-bb28-4e01394618bf)
      echo "   KEEP (live): $id"; continue;;
  esac
  pgrep -af "claude" | grep -q "$id" && { echo "   KEEP (ps says live): $id"; continue; }
  du -sh "$d"; run "$d"
done

echo
echo "NOT touched, deliberately -- owner call (see the doc):"
echo "   d41_libpin 7.78G  d42_libpin 3.34G  d44_libpin 2.22G  pdhdstm_libpin 1.11G"
echo "   d44sp 0.97G -- doc 44's repro block names it as an INPUT (--frames)"
echo "   doc37 non-.so 2.7G -- an install tree, not classifiable by extension"
echo "   -- docs 41/42/44 shipped constants to PDVD production on 2026-09-05 and"
echo "      pdhdstm_libpin backs the STM hand-scan arms protected this round."
echo "   d45, d45_libpin -- LIVE peer round (PID 1440435)"
[ "$CONFIRM" = yes ] || echo -e "\nDRY RUN.  Re-run with CONFIRM=yes to execute."
