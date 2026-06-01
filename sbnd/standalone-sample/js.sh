#!/bin/bash
# Render wcls-sim-drift-depoflux-nf-sp.jsonnet (sbndcode SBND detsim path)
# to JSON (+ PDF dot graph) using ext-vars sourced from
# standalone-sample/standard_detsim_sbnd-dump.fcl.
#
# Usage:  ./js.sh [json|pdf|all]  [extra-dotify-args]
#         ./js.sh all
#
# Output: wcls-sim-drift-depoflux-nf-sp.json
#         wcls-sim-drift-depoflux-nf-sp.pdf

set -e

mode=${1:-all}

TARGET_JSONNET=/exp/sbnd/app/users/yuhw/sbndcode/sbndcode/WireCell/cfg/pgrapher/experiment/sbnd/wcls-sim-drift-depoflux-nf-sp.jsonnet
base_name=$(basename "$TARGET_JSONNET" .jsonnet)

# Build -J args from WIRECELL_PATH in reverse so the first dir on the path
# wins lookup (mirrors how wire-cell itself searches).
IFS=':' read -ra cfg_dirs <<< "$WIRECELL_PATH"
J_ARGS=""
for (( idx=${#cfg_dirs[@]}-1; idx>=0; idx-- )); do
    J_ARGS="$J_ARGS -J ${cfg_dirs[$idx]}"
done

if [[ $mode == "json" || $mode == "all" ]]; then
    # ext-str: standard_detsim_sbnd-dump.fcl `params:` block (string args).
    # ext-code: structs:/strings that are numeric or boolean (parsed as code).
    jsonnet \
      --ext-str dnnroi_model_p0="DNN_ROI/plane0.ts" \
      --ext-str dnnroi_model_p1="DNN_ROI/plane1.ts" \
      --ext-str epoch="perfect" \
      --ext-str inputTag="ionandscint:priorSCE" \
      --ext-str reality="sim" \
      --ext-str roi="both" \
      --ext-str save_simdigits="false" \
      --ext-str save_track_id="true" \
      --ext-str signal_output_form="sparse" \
      --ext-str wc_device="cpu" \
      --ext-code DL=4 \
      --ext-code DT=8.8 \
      --ext-code driftSpeed=1.563 \
      --ext-code enableLowROIThresholds=true \
      --ext-code lifetime=35 \
      --ext-code nchunks=2 \
      --ext-code nticks=3427 \
      --ext-code tick0_time=-205 \
      --ext-code tick_per_slice=4 \
      $J_ARGS \
      "$TARGET_JSONNET" \
      -o ${base_name}.json
fi

if [[ $mode == "pdf" || $mode == "all" ]]; then
    wirecell-pgraph dotify --jpath -1 ${2} ${base_name}.json ${base_name}.pdf
fi
