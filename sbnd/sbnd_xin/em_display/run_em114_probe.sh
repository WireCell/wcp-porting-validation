#!/bin/bash
# doc pr/114 -- stage 2: re-run the scan sample with the shower probes ON so the
# display can show WHY each segment is in the shower it is in.
#
# Nothing here changes the reconstruction.  The three probes are env-gated and
# stderr-only, and the code that emits them says so at the definition site
# ("Log/stderr only: no effect on emitted bytes", NeutrinoShowerClustering.cxx
# :109 / :126, PRShower.cxx :73-74):
#
#   WCT_SHOWER_CONTENT_DEBUG  the NON-LOSSY membership of every PR::Shower, with
#                             each member's length, dQ, dQ/dx and its fraction of
#                             the shower's kine_charge, plus start/end and the
#                             dir15/dir100 axes.  This is the only faithful
#                             source: the calib dump's segments[].shower_id join
#                             keeps ONE shower per segment, so an overlapped
#                             shower comes back looking empty (measured: 11 of
#                             1081 showers over the 67 curated events).
#   WCT_SHOWER_ABSORB_DEBUG   the site= tag naming WHICH pass absorbed each
#                             segment, plus per-segment ADD/EXCLUDE from the
#                             flood-fill walk.
#   WCT_SHOWER_MERGE_DEBUG    every quantity in every shower-merge condition
#                             together with its verdict, and a reason line when
#                             a whole pass is skipped.
#
# PROVEN, not assumed (doc pr/114 sec 3): a 3-event probe run
# (work-em114-probe3: ncpi0 21073, 84229, 463565) reproduces the prod0825 dumps
# on EVERY physics field -- showers, segments, kine, tagger and main_vertex all
# byte-equal -- across both the library drift (prod0825 ran at the lib installed
# 2026-08-25 02:04, this runs at 16:34, the MCS commits in between) and the mode
# change (prod0825 was PR_GROUP_SIZE=16, this is per-event).  The ONLY field that
# moves is showers[].shower_id, the per-PROCESS sequential counter, which is not
# a physics quantity -- which is exactly why everything joins on the stable
# `id` / `node_id` (cluster_id*1000 + segment index) instead.
#
# So the display reads the PROD0825 dumps (the canonical products, untouched)
# and takes only the probe text from here.
#
# Usage:
#   ./em_display/run_em114_probe.sh [arm ...]      # default: all four
#
# Output: work-em114-<arm>/pr_evt<ID>/stdout.log  <- the probes land HERE, not in
# wct_pr_evt<ID>.log.  wire-cell's -l flags route only the spdlog logger; a raw
# fprintf(stderr) follows the subshell redirect in run_pr_chain_batch.sh:1642.
set -u

SX=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$SX"

ARMS=("$@")
[ "${#ARMS[@]}" -eq 0 ] && ARMS=(ncpi0 nuecc48 mcp1k mcp2k)

# The scan sample: the whole corrected NCpi0 list (46) plus the curated nueCC48
# arm (48) = 94 events.  Both lists are doc pr/113 round 2 -- round 1's NCpi0
# list was half this size because of the falsy-zero pio_id bug.
events_for() {
    local arm=$1
    /nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/python - "$arm" <<'PY'
import sys
arm = sys.argv[1]
def rows(p):
    out = []
    for ln in open(p):
        if ln.startswith('#'):
            continue
        f = ln.split()
        if len(f) > 4:
            out.append((f[0], f[4]))
    return out
sel = set(rows('docs/pr/pr113-ncpi0.index.txt'))
sel |= {r for r in rows('docs/pr/pr113-nuecc.index.txt') if r[0] == 'nuecc48'}
print(' '.join(sorted(e for a, e in sel if a == arm)))
PY
}

for arm in "${ARMS[@]}"; do
    evts=$(events_for "$arm")
    if [ -z "$evts" ]; then
        echo "[$arm] no events in the scan sample -- skipped"
        continue
    fi
    n=$(echo "$evts" | wc -w)
    out="work-em114-$arm"
    if [ -d "$out" ]; then
        echo "[$arm] $out exists -- skipped (M13: a fresh arm per run; remove it deliberately or pick a new tag)"
        continue
    fi
    echo "[$arm] $n events -> $out"
    WCT_SHOWER_CONTENT_DEBUG=1 \
    WCT_SHOWER_ABSORB_DEBUG=1 \
    WCT_SHOWER_MERGE_DEBUG=1 \
    PR_EXTRA_STAGES=pr_display \
    PR_JOBS=${PR_JOBS:-6} \
        ./run_pr_chain_batch.sh "work-$arm-grp0825" "$out" data $evts \
        > "/home/xqian/tmp/em114-$arm.log" 2>&1
    echo "[$arm] rc=$? log=/home/xqian/tmp/em114-$arm.log"
done
