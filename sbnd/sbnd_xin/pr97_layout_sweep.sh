#!/bin/bash
# doc pr/97 sec 5 -- the 18255-178410 crash reproducer.
#
# Same event, same inputs (work-img-mcp2k/evt178410, symlinked), same binary,
# ONE Q/L job at a time, every run under `setarch x86_64 -R` (run_ql_batch.sh
# always does).  The ONLY thing that varies between runs is the byte length of
# one extra exported variable -- i.e. the address-space layout.  ~1 layout in
# 6-10 dies with rc=139 and a 3.5x peak-RSS excursion; the rest run in ~680 MB.
#
# Usage:  ./pr97_layout_sweep.sh <root-prefix> [pad ...]
#   e.g.  ./pr97_layout_sweep.sh work-pr97c-pad 0 8 16 24 32 48 64 96 128 192
#
# One FRESH root per padding (M13).  Refuses to reuse one.  Concurrency is
# capped at 5 (M5) -- the point is one job per layout, not throughput.
set -u
S=$(cd "$(dirname "$0")" && pwd)
cd "$S"
PREFIX=${1:?usage: pr97_layout_sweep.sh <root-prefix> [pad ...]}
shift
PADS=${*:-0 8 16 24 32 48 64 96 128 192}
for n in $PADS; do
    R=$S/$PREFIX$n
    [ -e "$R" ] && { echo "REFUSE existing $R"; continue; }
    pad=$(head -c "$n" /dev/zero | tr '\0' 'a')
    ( PR97PAD="$pad" ROOT=$R ./run_ql_batch.sh -j 1 178410 > "/home/xqian/tmp/${PREFIX}$n.log" 2>&1
      echo "pad=$n $(cat "$R"/.status/* 2>/dev/null)" ) &
    while [ "$(jobs -r | wc -l)" -ge 5 ]; do sleep 5; done
done
wait
echo "sweep done: $PREFIX{$PADS}"
