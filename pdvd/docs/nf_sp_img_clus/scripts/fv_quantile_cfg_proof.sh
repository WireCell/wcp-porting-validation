#!/bin/bash
# doc pdvd/43: compiled-config byte-identity proof for the curved_fv_profile knob.
#   Usage: fv_quantile_cfg_proof.sh [scratch dir] [toolkit ref, default HEAD] [wcp-porting-img ref, default HEAD]
# Compiles pdvd/wct-pr-perevt.jsonnet with fixed TLAs from (a) a reference tree --
# `git archive <ref> cfg` of the toolkit + `git show <ref>:pdvd/wct-pr-perevt.jsonnet`
# of wcp-porting-img -- and (b) the working trees, knob OFF and ON (d50), and cmp's
# them; then compiles ON with profile p80 / p90 to show the new corners appear.
set -u
D=${1:-/home/xqian/tmp/doc43}
TKREF=${2:-HEAD}
WPREF=${3:-HEAD}
mkdir -p "$D/refcfg"
git -C /home/xqian/toolkit-dev/toolkit archive "$TKREF" cfg | tar -x -C "$D/refcfg"
git -C /nfs/data/1/xqian/toolkit-dev/wcp-porting-img show "$WPREF:pdvd/wct-pr-perevt.jsonnet" > "$D/refcfg/wct-pr-perevt.jsonnet"
WCD=/home/xqian/toolkit-dev/wire-cell-data
TLAS=(-A input=/x/pctree-evt298567.tar.gz -A output_dir=/x/out -S run=39252 -S subrun=0 -S event=298567
      -S drift_speed_bot_mmus=1.48073 -S drift_speed_top_mmus=1.48073 -S trigger_offset_bot_us=-2503.820999
      -S trigger_offset_top_us=-2483.677 -S readout_window_ticks=10000 -S stepped_center_fallback=false
      -S 'pipeline_names=["switch_scope","flag_mains","unmerge_assoc","steiner","fiducialutils","tagger_check_tgm","tagger_check_stm","tagger_check_fc","protect_bundle","steiner_refresh","pr_display"]')
comp () { # <label> <cfgroot> <driver> [extra tlas]
  local lab=$1 root=$2 drv=$3; shift 3
  ( cd "$(dirname "$drv")" && WIRECELL_PATH="$root:$WCD" wcsonnet "${TLAS[@]}" "$@" -o "$D/cfgproof_$lab.json" "$(basename "$drv")" ) > "$D/cfgproof_$lab.err" 2>&1
  echo "$lab rc=$? size=$(stat -c %s "$D/cfgproof_$lab.json" 2>/dev/null)"
}
NEW=/home/xqian/toolkit-dev/toolkit/cfg
NEWDRV=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/wct-pr-perevt.jsonnet
REF=$D/refcfg/cfg
REFDRV=$D/refcfg/wct-pr-perevt.jsonnet
comp ref_off "$REF" "$REFDRV" &
comp ref_on  "$REF" "$REFDRV" -S curved_fv=true &
comp new_off "$NEW" "$NEWDRV" &
comp new_on  "$NEW" "$NEWDRV" -S curved_fv=true &
comp new_p80 "$NEW" "$NEWDRV" -S curved_fv=true -A curved_fv_profile=p80 &
comp new_p90 "$NEW" "$NEWDRV" -S curved_fv=true -A curved_fv_profile=p90 &
wait
cmp "$D/cfgproof_ref_off.json" "$D/cfgproof_new_off.json" && echo "OFF: byte-identical to HEAD"
cmp "$D/cfgproof_ref_on.json"  "$D/cfgproof_new_on.json"  && echo "ON d50: byte-identical to HEAD's curved_fv=true"
cmp -s "$D/cfgproof_new_on.json" "$D/cfgproof_new_p90.json" || echo "ON p90 differs from ON d50 (expected)"
