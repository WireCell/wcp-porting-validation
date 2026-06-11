#!/bin/bash
# Sample VmRSS/VmHWM of a process at 0.5 s into a TSV until it exits.
# Usage: ./rss_sample.sh <pid> <out.tsv>
PID=${1:?pid}
OUT=${2:?out.tsv}
echo -e "t_s\tvmrss_kb\tvmhwm_kb" > "$OUT"
T0=$(date +%s.%N)
while [ -d "/proc/$PID" ]; do
    NOW=$(date +%s.%N)
    RSS=$(awk '/VmRSS/{print $2}' "/proc/$PID/status" 2>/dev/null)
    HWM=$(awk '/VmHWM/{print $2}' "/proc/$PID/status" 2>/dev/null)
    [ -n "$RSS" ] && printf '%.1f\t%s\t%s\n' "$(echo "$NOW - $T0" | bc)" "$RSS" "$HWM" >> "$OUT"
    sleep 0.5
done
