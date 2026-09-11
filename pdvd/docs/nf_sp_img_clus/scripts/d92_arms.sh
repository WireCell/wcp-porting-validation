#!/usr/bin/env bash
# doc pdvd/92 -- the arms for the wide Bragg-peak read (doc 91 sec 10 item 1).
#
# Fork by duplication (CLAUDE.md M10) of d90_arms.sh.  The differences: this round DOES change C++
# (bragg_wide_anchor_cm / _rise_min / _tail_max in CheckSTM_Michel.cxx), so
#   - the pin is a FRESH full local/lib snapshot taken after this round's wcbuild (libpin_p92), and
#   - an OFF gate is required on BOTH detectors: the new binary with the knob absent must be
#     byte-identical to production.  p92voff has no TLA at all, so the compiled config is
#     production's and the only difference from p90vprod is the binary.
#     p92hoff is the same on PDHD against p88hoff (PDHD's config has not moved since; PDHD stays OFF).
#
# Every arm is BARE production plus its TLA (no d53 survey bag).  The two ON arms are doc 91's
# pre-registered operating points; d92_twin.py predicts both item by item before they run.
#
# Usage:  WAVE=1 ./d92_arms.sh    (p92voff p92hoff p92v13 p92v15)
set -uo pipefail
X=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
PIN=${PIN:-/home/xqian/tmp/p92/libpin_p92}
JOBS=${JOBS:-5}
LOGD=${LOGD:-/home/xqian/tmp/p92}
W13="bragg_wide_anchor_cm:8.0,bragg_wide_anchor_rise_min:1.3,bragg_wide_anchor_tail_max:0.8"
W15="bragg_wide_anchor_cm:8.0,bragg_wide_anchor_rise_min:1.5,bragg_wide_anchor_tail_max:0.8"

if [ ! -s "$PIN/libWireCellClus.so" ]; then
    echo "REFUSING: no pin at $PIN -- snapshot local/lib first (a peer rebuild would void every arm)" >&2
    exit 2
fi

run () {  # run <arm> <det> <src> <extra-keys-or-empty>
    local arm=$1 det=$2 src=$3 extra=$4 tla=""
    [ -n "$extra" ] && tla="-S stm_michel_extra={$extra}" || tla=""
    echo "[$(date +%H:%M:%S)] $arm ($det) ${extra:-<no TLA: the OFF gate>}"
    ARM=$arm DET=$det SRC=$src JOBS=$JOBS PIN=$PIN PR_TLA="$tla" \
        LOGD=$LOGD/arm_$arm "$X/d53_run_arms.sh" > "$LOGD/arm_${arm}.log" 2>&1
    echo "[$(date +%H:%M:%S)] $arm rc=$?"
}

WAVE=${WAVE:-1}
if [ "$WAVE" = 1 ]; then
    run p92voff pdvd d16vnu "" &
    run p92hoff pdhd d16hnu "" &
    run p92v13  pdvd d16vnu "$W13" &
    run p92v15  pdvd d16vnu "$W15" &
    wait
fi
touch "$LOGD/DONE_wave$WAVE"
echo "[$(date +%H:%M:%S)] wave $WAVE done"
