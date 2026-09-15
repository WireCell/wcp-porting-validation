#!/bin/bash
# doc pdvd/102: compile the PR job for a compiled-config proof.  Same args as run_pr_evt.sh -nu -stm-fit, dummy I/O.
# usage: compile.sh <pdhd|pdvd> <label> [extra wcsonnet args...]
set -u
DET=$1; LAB=$2; shift 2
B=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH="$B/toolkit/cfg:$B/wire-cell-data"
export LD_LIBRARY_PATH=/home/xqian/tmp/d102/libpin_d102
O=/home/xqian/tmp/d102/cfg
PIPE='["switch_scope","flag_mains","unmerge_assoc","steiner","fiducialutils","tagger_check_tgm","tagger_check_stm","tagger_check_fc","protect_bundle","steiner_refresh","check_stm_michel","tracking_visitor","pr_display","stm_magnify"]'
if [ "$DET" = pdhd ]; then
  DARGS=(-S run=28084 -S subrun=0 -S event=74576 -S trigger_offset_us=0 -S readout_window_ticks=6000)
else
  DARGS=(-S run=39253 -S subrun=0 -S event=8 -S drift_speed_bot_mmus=1.48073 -S drift_speed_top_mmus=1.48073 -S trigger_offset_bot_us=0 -S trigger_offset_top_us=0 -S readout_window_ticks=10000 -S stepped_center_fallback=false)
fi
(cd $B/wcp-porting-img/$DET && $B/local/bin/wcsonnet -A input=$O/dummy.tar.gz -A output_dir=$O "${DARGS[@]}" -S "pipeline_names=$PIPE" "$@" -o $O/${LAB}_$DET.json wct-pr-perevt.jsonnet) > $O/${LAB}_$DET.log 2>&1
rc=$?
echo "$DET $LAB rc=$rc md5=$(md5sum $O/${LAB}_$DET.json 2>/dev/null | cut -c1-12)"
exit $rc
