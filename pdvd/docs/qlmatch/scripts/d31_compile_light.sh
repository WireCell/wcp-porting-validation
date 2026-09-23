#!/bin/bash
# compile the PDVD light job exactly as run_light_evt.sh does, for evt298567 with _keep's offsets.
# usage: compile_light.sh OUT.json SUFFIX [extra -S args...]
out=$1; suf=$2; shift 2
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
wcsonnet -A input_file=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/input_data_light/np02vd_raw_run039252_1176_df-s03-d3_dw_0_20250830T054542_rawwf.root \
  -A output_dir=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work/039252_light298567$suf \
  -S run=39252 -S event=298567 -S offset_bot_us=-2517.327999 -S offset_top_us=-2497.184 \
  -S veto_saturation=false -S flag_saturation=true -S saturation_repair=true -S emit_coverage=true -S spe_v2=true -S overflow_to_rail=true \
  "$@" -o "$out" /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/wct-light-reco.jsonnet
