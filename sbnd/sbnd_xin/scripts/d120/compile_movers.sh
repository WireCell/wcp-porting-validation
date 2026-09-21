#!/bin/bash
# doc sbnd_xin/120 round A: compile every relocation candidate the way its
# runner does, from the WORK-DIR path (which is what run_pr_evt.sh and friends
# invoke -- they `cd` into the work dir and name the file bare).
#
# usage: compile_movers.sh <outdir>
#
# Run once before the move and once after.  The two output dirs must be
# byte-identical: that is the whole gate.  Compiling from the work-dir path both
# times is deliberate -- after the move that path is a 1-line re-export shim, so
# this measures exactly what production will get.
set -u
OUT=${1:?}; mkdir -p "$OUT"
TK=/nfs/data/1/xqian/toolkit-dev
W=$TK/local/bin/wcsonnet
WD=$TK/wcp-porting-img
export WIRECELL_PATH=$TK/toolkit/cfg:$TK/wire-cell-data

# The PR jobs are compiled twice: bare (every default) and at the runner's own
# TLA set (the production branch).  A file with no unqualified import cannot
# change meaning on a move, but the PR job is the one that matters, so it gets
# the second point regardless.
PDHD_PR_TLA=(-A input=pctree.tar.gz -A output_dir=out -S run=27305 -S subrun=0
             -S event=150 -S trigger_offset_us=0 -S readout_window_ticks=6000)
PDVD_PR_TLA=(-A input=pctree.tar.gz -A output_dir=out -S run=27305 -S subrun=0
             -S event=150 -S drift_speed_bot_mmus=1.48073 -S drift_speed_top_mmus=1.48073
             -S trigger_offset_bot_us=0 -S trigger_offset_top_us=0
             -S readout_window_ticks=10000 -S stepped_center_fallback=false)
LIGHT_TLA=(-A input_file=light.root -A output_dir=out)

one () {                                  # one <det> <stem> <tag> [tlas...]
    local det=$1 stem=$2 tag=$3; shift 3
    ( cd "$WD/$det" && $W "$@" "$stem.jsonnet" ) \
        > "$OUT/${det}_${stem}${tag}.json" 2> "$OUT/${det}_${stem}${tag}.err"
    printf '%-6s %-28s %-6s rc=%s %9s bytes\n' \
           "$det" "$stem" "${tag:-bare}" "$?" "$(stat -c%s "$OUT/${det}_${stem}${tag}.json")"
}

for s in wct-pr-perevt wct-clustering wct-img-all wct-sp-to-magnify; do one pdhd "$s" ""; done
for s in wct-light-reco wct-light-fullstream-reco wct-light-convert; do
    one pdhd "$s" "" "${LIGHT_TLA[@]}"; done
# wct-light-allpd-reco takes two input files and no `input_file`
one pdhd wct-light-allpd-reco "" -A snip_file=snip.root -A fs_file=fs.root -A output_dir=out
for s in wct-pr-perevt wct-clustering wct-nf-sp wct-nf-sp-dnnroi wct-sp-to-magnify wct-img-all; do
    one pdvd "$s" ""; done
one pdvd wct-light-reco "" "${LIGHT_TLA[@]}"

one pdhd wct-pr-perevt ".prod" "${PDHD_PR_TLA[@]}"
one pdvd wct-pr-perevt ".prod" "${PDVD_PR_TLA[@]}"
