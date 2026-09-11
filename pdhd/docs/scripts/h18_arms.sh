#!/usr/bin/env bash
# doc pdhd/18 -- the two PR arms behind the PDHD STM+Michel hand scan (tag smx18).
#
# Fork by duplication (CLAUDE.md M10) of pdvd/docs/nf_sp_img_clus/scripts/d82_arms.sh.
# The ONLY differences are the arm names, the TLAs and the log dir.
#
#   h18s  DISPLAY arm: PDHD production + the scan SURVEY bag (exactly p82hoff's
#         bag: survey + publish_other_arms, no segment_census) + max_candidates 64.
#         The scanner looks at this arm.
#   h18b  GRADING arm for the cap extras: bare PDHD production + max_candidates 64.
#
# max_candidates 64 is the owner's scope choice for this round (2026-09-11): PDHD
# production keeps the C++ default 8, which truncates the candidate set on 7 of
# the 61 events (doc pdvd/79), so the extras are scanned now rather than after a
# PDHD flip.  Gate (doc pdvd/79's test): on every SHARED candidate h18s must be
# bit-identical to p82hoff and h18b to p82bhoff -- the cap only ADDS candidates.
#
# Compiled-config proof (M6), /home/xqian/tmp/h18/cfgproof: p82hoff vs h18s and
# bare vs h18b each differ by exactly one line, "max_candidates": 64.
#
# The pin is a FULL local/lib snapshot (feedback_shared_tree_binary_pin):
# libpin_p82 = toolkit 082376c5, libWireCellClus md5 5f2c3ede, the binary that
# ran p82hoff / p82bhoff.
#
# Usage:  ./h18_arms.sh
set -uo pipefail
X=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
PIN=${PIN:-/home/xqian/tmp/p82/libpin_p82}
JOBS=${JOBS:-8}
LOGD=${LOGD:-/home/xqian/tmp/h18}
SURVEY='survey_enable:true,survey_radius_cm:60.0,survey_max_len_cm:25.0,publish_other_arms:true'

run () {  # run <arm> <det> <src> <extra-keys-or-empty>
    local arm=$1 det=$2 src=$3 extra=$4 tla=""
    [ -n "$extra" ] && tla="-S stm_michel_extra={$extra}" || tla=""
    echo "[$(date +%H:%M:%S)] $arm ($det) $extra"
    ARM=$arm DET=$det SRC=$src JOBS=$JOBS PIN=$PIN PR_TLA="$tla" \
        LOGD=$LOGD/arm_$arm "$X/d53_run_arms.sh" > "$LOGD/arm_${arm}.log" 2>&1
    echo "[$(date +%H:%M:%S)] $arm rc=$?"
}

run h18s pdhd d16hnu "$SURVEY,max_candidates:64" &
run h18b pdhd d16hnu "max_candidates:64" &
wait
touch "$LOGD/DONE_arms"
echo "[$(date +%H:%M:%S)] arms done"
