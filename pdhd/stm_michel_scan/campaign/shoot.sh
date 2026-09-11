#!/usr/bin/env bash
# doc pdhd/18 -- shoot every item of a scan sheet with scan_harness.py, N processes.
#
# Ported from the doc pdvd/55 tranche-2 scratch copy (never committed).  Changes:
# the sheet, prep dir, det and flags are arguments; --blind and --hide-selection
# are ON (doc pdhd/18); the item list comes from the sheet, not a scratch file.
#
# Each process gets a THROWAWAY tag and its own private blank labeldir, so no
# process in this phase has a write path to any real label file.  The stagger is
# not politeness: App.__init__ takes free_port(5300), so simultaneous launches can
# bind-race onto one port.  On a WebGL backend keep NPROC <= 2 (doc pdvd/55 sec
# 13.2: five lost the context); on Canvas2D (an unreachable $DISPLAY, doc pdvd/69
# sec 8.3) there is no context to lose and 4 is fine.
#
# Usage: shoot.sh ROUND_DIR DET SHEET PREPDIR [NPROC]
#   writes ROUND_DIR/shots/<event>_<cluster>/, ROUND_DIR/logs/shoot_<n>.log,
#   ROUND_DIR/chunk_<n>.txt, and ROUND_DIR/SHOTS_DONE when every process has exited.
set -u
R=$1; DET=$2; SHEET=$3; PREP=$4; N=${5:-4}
H=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$R/logs" "$R/shots"
grep -v '^#' "$SHEET" | awk -F'\t' 'NR>1{print $3"/"$4}' > "$R/items_all.txt"
split -n r/$N -d -a 1 "$R/items_all.txt" "$R/chunk_"
for n in $(seq 0 $((N-1))); do
  mv "$R/chunk_$n" "$R/chunk_$n.txt"
  mkdir -p "$R/blank_$n"
  echo '{"labels": {}}' > "$R/blank_$n/labels.json"
  nohup python3 -u "$H/scan_harness.py" shots \
      --det "$DET" --tag shots18 --labeldir "$R/blank_$n" \
      --prepdir "$PREP" --manifest "$SHEET" --blind --hide-selection \
      --items-file "$R/chunk_$n.txt" --out "$R/shots_p$n" \
      > "$R/logs/shoot_$n.log" 2>&1 &
  echo "launched chunk $n ($(wc -l < "$R/chunk_$n.txt") items) pid $!"
  sleep 25
done
wait
# per-process out dirs keep each process's _webgl_lost.txt; fold the item dirs
for n in $(seq 0 $((N-1))); do
  for d in "$R/shots_p$n"/*/; do mv "$d" "$R/shots/"; done
  cp "$R/shots_p$n/_webgl_lost.txt" "$R/logs/webgl_lost_p$n.txt" 2>/dev/null
done
touch "$R/SHOTS_DONE"
echo "ALL SHOTS DONE"
