#!/bin/bash
# Aggregate stage timing across a sweep label's wct logs.
#
# Usage: ./parse_timing.sh <label>
# Sections: MABC per-pass (by class), TaggerCheckNeutrino sub-stages,
# and the [timer] per-node totals.
set -eu
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
QLPORT=$(dirname "$SCRIPTS")
ABTEST=$(cd "$QLPORT/../abtest" && pwd)

LABEL=${1:?usage: parse_timing.sh <label>}
LOGS=("$SCRIPTS/sweep/$LABEL"/*/wct_5384_*.log)
echo "== ${#LOGS[@]} logs, label=$LABEL =="

echo "-- MABC passes (by class) --"
"$ABTEST/parse_mabc_perf.sh" "${LOGS[@]}" -c

echo "-- TaggerCheckNeutrino sub-stages --"
grep -h "TaggerCheckNeutrino timing" "${LOGS[@]}" 2>/dev/null \
  | sed -E 's/.*TaggerCheckNeutrino timing: (.*) took ([0-9.]+) ms.*/\2\t\1/' \
  | awk -F'\t' '{a[$2]+=$1; tot+=$1}
         END {for (k in a) printf "%10.0f ms  %5.1f%%  %s\n", a[k], 100*a[k]/tot, k;
              printf "%10.0f ms  TOTAL\n", tot}' \
  | sort -rn

echo "-- [timer] node totals --"
grep -h "\[ timer  \] Timer:" "${LOGS[@]}" 2>/dev/null \
  | sed -E 's/.*Timer: (.*) : ([0-9.]+) sec.*/\2\t\1/' \
  | awk -F'\t' '{a[$2]+=$1} END {for (k in a) printf "%10.1f s  %s\n", a[k], k}' \
  | sort -rn
