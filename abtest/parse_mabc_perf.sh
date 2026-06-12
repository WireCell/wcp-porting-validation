#!/bin/bash
# Aggregate "MABC timing:" per-pass lines from clustering logs.
# Usage: ./parse_mabc_perf.sh <wct_clus_*.log...> [-c]
#   -c : aggregate by pass CLASS (strip the :instance suffix)
BYCLASS=false
files=()
for a in "$@"; do
    [ "$a" = "-c" ] && BYCLASS=true || files+=("$a")
done
grep -h "MABC timing" "${files[@]}" 2>/dev/null \
  | sed -E 's/.*MABC timing: (.*) took ([0-9.]+) ms.*/\2 \1/' \
  | { if $BYCLASS; then sed -E 's/ ([A-Za-z0-9]+):.*/ \1/'; else cat; fi; } \
  | awk '{a[$2]+=$1; tot+=$1}
         END {for (k in a) printf "%10.0f ms  %5.1f%%  %s\n", a[k], 100*a[k]/tot, k;
              printf "%10.0f ms  TOTAL\n", tot}' \
  | sort -rn
