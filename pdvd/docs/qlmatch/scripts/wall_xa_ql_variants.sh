#!/bin/bash
# Doc 25 §8: wall-XA-in-QLMatching evaluation variants — matching-only reruns
# of 039252 idx 0..17 from the canonical _keep clustering, _whfix light.
#   wx0 = walls masked (light-fix control)
#   wx1 = walls in, measured_pe_scale calibration + wall pe_err family 1.5/3.0
#   wx2 = walls in, uncalibrated (ablation)
#   wx3 = walls in, calibrated, errors 3.0/5.0 (error-saturation test)
# Fresh tags only (M13); reruns skip events whose calib dump already exists.
set -u
PDVD=$(cd "$(dirname "$0")/../../.." && pwd)
LOGDIR=${WALLXA_QL_LOGDIR:-/home/xqian/tmp/wallxa_ql_logs}
mkdir -p "$LOGDIR"
# 1/median(meas_whfix/exp) per live wall channel (wall_xa_ql_calib.py)
SCALE='[0.787, 0.751, 1.0, 0.393, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.585, 1.0, 1.0, 1.0, 1.0, 1.0, 0.518, 0.457, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]'

run_one() {
    local tag=$1 idx=$2; shift 2
    local dir="$PDVD/work/039252_${idx}_${tag}"
    if ls "$dir"/calib-evt*.json >/dev/null 2>&1; then
        echo "skip $tag idx$idx (calib exists)"; return 0
    fi
    [ -d "$dir" ] || "$PDVD/scripts/stage_ql_tag.sh" 39252 "$idx" "$tag" || return 1
    ( cd "$PDVD" && env "$@" ./run_clus_evt.sh -s "$tag" -calib 39252 "$idx" \
        > "$LOGDIR/${tag}_idx${idx}.log" 2>&1 )
    echo "$tag idx$idx rc=$?"
}

variant() {
    local tag=$1; shift
    local idx
    for idx in $(seq 0 17); do
        run_one "$tag" "$idx" "$@" &
        while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n; done
    done
    wait
    echo "=== variant $tag: $(ls $PDVD/work/039252_*_${tag}/calib-evt*.json 2>/dev/null | grep -vc group)/18 calib dumps ==="
}

variant wx0 PDVD_LIGHT_SUFFIX=_whfix

variant wx1 PDVD_LIGHT_SUFFIX=_whfix PDVD_QL_MASK_WALL_XA=0 PDVD_QL_WALL_FLAGS=1 \
    PDVD_QL_PEERR_WALL_FRAC=1.5 PDVD_QL_PEERR_WALL_LOWPE_FRAC=3.0 \
    PDVD_QL_PEERR_WALL_LOWPE_KNEE=10 PDVD_QL_MEASURED_PE_SCALE="$SCALE"

variant wx2 PDVD_LIGHT_SUFFIX=_whfix PDVD_QL_MASK_WALL_XA=0 PDVD_QL_WALL_FLAGS=1

variant wx3 PDVD_LIGHT_SUFFIX=_whfix PDVD_QL_MASK_WALL_XA=0 PDVD_QL_WALL_FLAGS=1 \
    PDVD_QL_PEERR_WALL_FRAC=3.0 PDVD_QL_PEERR_WALL_LOWPE_FRAC=5.0 \
    PDVD_QL_PEERR_WALL_LOWPE_KNEE=10 PDVD_QL_MEASURED_PE_SCALE="$SCALE"

echo ALL DONE
