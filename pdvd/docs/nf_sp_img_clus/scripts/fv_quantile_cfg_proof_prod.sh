#!/bin/bash
# doc pdvd/43 sec 8: the production flip's equivalence proofs.
#  (1) working tree, NO TLA        == HEAD tree with the d43p90c5 arm's TLAs   (production = the graded arm)
#  (2) working tree, curved_fv=false == HEAD tree, no TLA                      (the doc-35 flat point still reachable)
set -u
D=/home/xqian/tmp/doc43; WCD=/home/xqian/toolkit-dev/wire-cell-data
mkdir -p $D/refcfg_prod && git -C /home/xqian/toolkit-dev/toolkit archive HEAD cfg | tar -x -C $D/refcfg_prod
git -C /nfs/data/1/xqian/toolkit-dev/wcp-porting-img show HEAD:pdvd/wct-pr-perevt.jsonnet > $D/refcfg_prod/wct-pr-perevt.jsonnet
TLAS=(-A input=/x/pctree-evt298567.tar.gz -A output_dir=/x/out -S run=39252 -S subrun=0 -S event=298567
      -S drift_speed_bot_mmus=1.48073 -S drift_speed_top_mmus=1.48073 -S trigger_offset_bot_us=-2503.820999
      -S trigger_offset_top_us=-2483.677 -S readout_window_ticks=10000 -S stepped_center_fallback=false
      -S 'pipeline_names=["switch_scope","flag_mains","unmerge_assoc","steiner","fiducialutils","tagger_check_tgm","tagger_check_stm","tagger_check_fc","protect_bundle","steiner_refresh","pr_display"]')
comp () { local lab=$1 root=$2 drv=$3; shift 3
  ( cd "$(dirname "$drv")" && WIRECELL_PATH="$root:$WCD" wcsonnet "${TLAS[@]}" "$@" -o "$D/prodproof_$lab.json" "$(basename "$drv")" ) > "$D/prodproof_$lab.err" 2>&1
  echo "$lab rc=$? size=$(stat -c %s "$D/prodproof_$lab.json" 2>/dev/null)"; }
NEW=/home/xqian/toolkit-dev/toolkit/cfg; NEWDRV=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/wct-pr-perevt.jsonnet
REF=$D/refcfg_prod/cfg; REFDRV=$D/refcfg_prod/wct-pr-perevt.jsonnet
comp new_prod "$NEW" "$NEWDRV" &
comp ref_p90c5 "$REF" "$REFDRV" -S curved_fv=true -A curved_fv_profile=p90 -S curved_fv_margin_y=5 -S curved_fv_margin_z=5 &
comp new_flat "$NEW" "$NEWDRV" -S curved_fv=false &
comp ref_off "$REF" "$REFDRV" &
wait
cmp $D/prodproof_new_prod.json $D/prodproof_ref_p90c5.json && echo "(1) production == d43p90c5 arm config: byte-identical"
cmp $D/prodproof_new_flat.json $D/prodproof_ref_off.json && echo "(2) curved_fv=false == doc-35 flat production config: byte-identical"
cmp -s $D/prodproof_new_prod.json $D/prodproof_ref_off.json || echo "(3) production differs from the old flat config (expected)"
