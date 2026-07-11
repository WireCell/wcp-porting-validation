#!/bin/bash
# gperftools CPU-profile (or jemalloc heap-profile) one event of the PDHD
# NF+SP+DNNROI chain (production settings of run_nf_sp_dnnroi_evt.sh:
# fp32 6-ch DNN model, L1SP on in dnn mode, all anodes present).
# Usage: ./profile_nfsp.sh <run> <evt> [out.prof]
# Pre-compiles the cfg with wcsonnet (SIGPROF kills gojsonnet GC otherwise)
# and points sp_prefix at scratch so profiling never clobbers work/ archives.
# Env overrides:
#   PROFLIB  - LD_PRELOAD lib (default libtcmalloc_and_profiler)
#   OUTDIR   - where wire-cell writes outputs (default scratch)
#   JEHEAP   - if set to a dir, do jemalloc sampled heap profiling instead
#              of a CPU profile (MALLOC_CONF prof; analyze with jeprof)
#   ANODES   - jsonnet list override, e.g. '[0]' (default: all present)
set -e
PDHD_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}
export LD_LIBRARY_PATH=${WCT_BASE}/libtorch-shim/lib:${WCT_BASE}/local/lib:${LD_LIBRARY_PATH}

RUN=${1:?run} EVT=${2:?event}
RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//'); [ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")
OUT=${3:-/home/xqian/tmp/nfsp_pdhd_${RUN_PADDED}_${EVT}.prof}

# Locate the event input dir (same heuristic as run_nf_sp_dnnroi_evt.sh).
EVTDIR=""
for base in "$PDHD_DIR"/input_data "$PDHD_DIR"/input_data_*; do
    [ -d "$base" ] || continue
    for rname in "run${RUN}" "run${RUN_PADDED}" "run${RUN_STRIPPED}"; do
        for ename in "evt${EVT}" "evt_${EVT}"; do
            cand="$base/$rname/$ename"
            if [ -d "$cand" ] && ls "$cand"/protodunehd-orig-frames-anode*.tar.bz2 >/dev/null 2>&1; then
                EVTDIR="$cand"; break 3
            fi
        done
    done
done
[ -n "$EVTDIR" ] || { echo "[err] no input dir for run=$RUN evt=$EVT" >&2; exit 2; }
echo "Event dir: $EVTDIR"

# Gain + coherent grouping from the input-root name (as in the driver).
INPUT_ROOT_NAME="${EVTDIR#$PDHD_DIR/}"; INPUT_ROOT_NAME="${INPUT_ROOT_NAME%%/*}"
ELEC_GAIN="14"
[[ "$INPUT_ROOT_NAME" =~ ^input_data_([0-9p]+)_ ]] && ELEC_GAIN="${BASH_REMATCH[1]//p/.}"
PREFLIP_TLA=()
case "$INPUT_ROOT_NAME" in
    *_old_coh_grouping) PREFLIP_TLA=(-S coh_groups_preflip=true) ;;
esac

if [ -z "$ANODES" ]; then
    ANODE_LIST=()
    for ai in 0 1 2 3; do
        [ -f "$EVTDIR/protodunehd-orig-frames-anode${ai}.tar.bz2" ] && ANODE_LIST+=("$ai")
    done
    ANODES="[$(IFS=,; echo "${ANODE_LIST[*]}")]"
fi
echo "elecGain: $ELEC_GAIN  anodes: $ANODES"

OUTDIR=${OUTDIR:-/home/xqian/tmp/prof_nfsp_pdhd_${RUN_PADDED}_${EVT}}
mkdir -p "$OUTDIR"

CFG=$OUTDIR/.wct-nf-sp-dnnroi.json
wcsonnet \
    -V "elecGain=${ELEC_GAIN}" \
    -A "orig_prefix=${EVTDIR}/protodunehd-orig-frames" \
    -A "sp_prefix=${OUTDIR}/protodunehd-sp-dnnroi-frames" \
    -A "reality=data" \
    -S "anode_indices=${ANODES}" \
    -A "dnnroi_model=dnnroi/pdhd/pipe_distill_transformer_6ch.ts" \
    -A "dnnroi_device=cpu" \
    -S "dnnroi_nchan=6" \
    -S "use_l1sp_dnn=true" \
    -A "l1sp_pd_mode=dnn" \
    -S "dnnroi_mask_thresh=0.2" \
    -S "apa0_w_roi_tune=true" \
    "${PREFLIP_TLA[@]}" \
    -o "$CFG" \
    "$PDHD_DIR/wct-nf-sp-dnnroi.jsonnet"

# Background VmRSS/VmHWM sampler for peak-vs-time attribution.
RSS_CSV=$OUTDIR/rss_${RUN_PADDED}_${EVT}.csv
sample_rss() {
    local pid=$1
    echo "t_s,vmrss_kb,vmhwm_kb" > "$RSS_CSV"
    local t0=$(date +%s.%N)
    while kill -0 "$pid" 2>/dev/null; do
        if [ -r /proc/$pid/status ]; then
            awk -v t="$(echo "$(date +%s.%N) - $t0" | bc)" \
                '/^VmRSS:/{r=$2} /^VmHWM:/{h=$2} END{printf "%.1f,%s,%s\n", t, r, h}' \
                /proc/$pid/status >> "$RSS_CSV" 2>/dev/null
        fi
        sleep 0.5
    done
}

if [ -n "$JEHEAP" ]; then
    mkdir -p "$JEHEAP"
    cd "$JEHEAP"
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    MALLOC_CONF=prof:true,lg_prof_sample:19,lg_prof_interval:33,prof_final:true \
    wire-cell -l stderr -L info -c "$CFG" &
    WC_PID=$!
    OUT="$JEHEAP (jeprof.*.heap; analyze with /home/xqian/tmp/jeprof)"
else
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so.4 \
    CPUPROFILE="$OUT" CPUPROFILE_FREQUENCY=1000 \
    wire-cell -l stderr -L info -c "$CFG" &
    WC_PID=$!
fi
sample_rss $WC_PID &
SAMPLER_PID=$!
RC=0; wait $WC_PID || RC=$?
wait $SAMPLER_PID 2>/dev/null || true
[ $RC -eq 0 ] || { echo "[err] wire-cell rc=$RC" >&2; exit $RC; }

echo "profile -> $OUT"
echo "rss trace -> $RSS_CSV"
echo "view: google-pprof --text $(which wire-cell) $OUT | head -40"
