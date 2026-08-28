#!/bin/bash
# doc pr/114 round 7 -- stage 2 for the COVERAGE extension: the 141 mcp1k+mcp2k
# events with em_max >= 100 MeV that the pr/114 scan sample never contained
# (docs/pr/pr114c-beam-em100-unscanned.index.txt, from
# scripts/pr114c_unscanned_em.py).
#
# A FORK of run_em114_probe.sh, not a generalisation of it: that script is the
# reproduction record of the 98-event scan now on port 5017 and stays
# byte-identical (CLAUDE.md sec 2 "fork by duplication").  Everything about the
# probes -- what they emit, why they cannot change the reconstruction, and why
# the display still reads the prod0825 dumps -- is documented there and is
# unchanged here.
#
# Fresh arms (M13): work-em114c-<arm>.  Nothing writes into work-em114-* or
# work-em114b-*.
#
# Usage:
#   ./em_display/run_em114c_probe.sh [arm ...]     # default: mcp1k mcp2k
#
# Output: work-em114c-<arm>/pr_evt<ID>/stdout.log
set -u

SX=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$SX"

INDEX=${EM114C_INDEX:-docs/pr/pr114c-beam-em100-unscanned.index.txt}

ARMS=("$@")
[ "${#ARMS[@]}" -eq 0 ] && ARMS=(mcp1k mcp2k)

events_for() {
    local arm=$1
    awk -F'\t' -v a="$arm" '!/^#/ && NF>4 && $1==a {print $5}' "$INDEX" \
        | sort -n | tr '\n' ' '
}

nskip=0
nrun=0
for arm in "${ARMS[@]}"; do
    evts=$(events_for "$arm")
    if [ -z "$evts" ]; then
        echo "[$arm] no events in $INDEX -- skipped"
        continue
    fi
    n=$(echo "$evts" | wc -w)
    out="work-em114c-$arm"
    if [ -d "$out" ]; then
        echo "[$arm] $out already exists with $(ls -d "$out"/pr_evt* 2>/dev/null | wc -l) events -- SKIPPED, nothing re-run"
        echo "       (M13: a fresh arm per run.  Move it aside to genuinely redo it.)"
        nskip=$((nskip+1))
        continue
    fi
    nrun=$((nrun+1))
    echo "[$arm] $n events -> $out"
    WCT_SHOWER_CONTENT_DEBUG=1 \
    WCT_SHOWER_ABSORB_DEBUG=1 \
    WCT_SHOWER_MERGE_DEBUG=1 \
    PR_EXTRA_STAGES=pr_display \
    PR_JOBS=${PR_JOBS:-6} \
        ./run_pr_chain_batch.sh "work-$arm-grp0825" "$out" data $evts \
        > "/home/xqian/tmp/em114c-$arm.log" 2>&1
    echo "[$arm] rc=$? log=/home/xqian/tmp/em114c-$arm.log"
done

echo
if [ "$nrun" -eq 0 ] && [ "$nskip" -gt 0 ]; then
    echo "NOTHING WAS RUN: all $nskip arm(s) already exist.  A no-op, not a success."
else
    echo "ran $nrun arm(s), skipped $nskip.  Next:"
    echo "  python em_display/prep_em_scan.py --sample-index $INDEX \\"
    echo "      --out em_display/em114c-manifest.tsv --prepdir em_display/emprep-c \\"
    echo "      --parse-probes work-em114c-mcp1k work-em114c-mcp2k"
fi
