#!/bin/bash
# doc sbnd_xin/123 round 3 (MC): a hit-flash arm on every per-file sub-root of a round-3 baseline
# (mc_base.sh), through scripts/d123/hits_arm.sh.  beam-off is one root of groups.
#
# Usage: [JOBS=n] scripts/d123/mc_hits.sh <cv|nuecc|off> <hits|nosplit> [extra hits_arm.sh args]
# Resumable: a per-file sub-root with a pctree for every baseline ql_evt is skipped (counting the
# ql_evt DIRS is not enough: a Q/L killed mid-way leaves the dirs without their pctree).
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
S=${1:?usage: mc_hits.sh <cv|nuecc|off> <hits|nosplit>}; ARM=${2:?}; shift 2
J=${JOBS:-6}
case "$S" in
    cv)    B=$SX/work-r3cv-d123base;  O=$SX/work-r3cv-d123$ARM;  REALITY=sim;  PERFILE=1 ;;
    nuecc) B=$SX/work-r3nue-d123base; O=$SX/work-r3nue-d123$ARM; REALITY=sim;  PERFILE=1 ;;
    off)   B=$SX/work-r3off-d123base; O=$SX/work-r3off-d123$ARM; REALITY=data; PERFILE=0 ;;
    *) echo "unknown sample: $S" >&2; exit 2 ;;
esac
case "$ARM" in
    hits)    EXTRA=(--ref) ;;
    nosplit) EXTRA=(--ff '{"pulse_split":false}') ;;
    # doc 123 round 6 (sec 18): production hit flashes, the Q/L job with one knob file (hits_arm.sh QLTLA)
    lg)      EXTRA=(); export QLTLA=$SX/scripts/d123/tla/xtpc_lg.txt ;;     # scenario-1 crosser flags gated on light
    nr)      EXTRA=(); export QLTLA=$SX/scripts/d123/tla/rescue_off.txt ;;  # the whole cathode-bundle rescue OFF
    lgop)    EXTRA=(); export QLTLA=$SX/scripts/d123/tla/xtpc_lgop.txt      # the gate + its over-prediction ceiling (needs the
             export PIN=${PIN:-$HOME/tmp/d123-libpin-r6} ;;                 # r6 pin: libWireCellMatch with xtpc_sc1_overpred_max)
    *) echo "unknown arm: $ARM (hits|nosplit|lg|nr|lgop)" >&2; exit 2 ;;
esac
[ -d "$B" ] || { echo "ERROR: no baseline $B" >&2; exit 1; }
LOGD=$HOME/tmp/d123-mc-$S-$ARM; mkdir -p "$LOGD"
echo "=== d123 MC $ARM arm: sample=$S base=$B out=$O jobs=$J"
t0=$(date +%s)
if [ "$PERFILE" = 1 ]; then
    mkdir -p "$O"; touch "$O/.d123_hits_arm"
    ls -d "$B"/f[0-9]* | xargs -n1 basename | xargs -P "$J" -I{} bash -c \
        'nb=$(ls -d "$1/{}"/ql_evt* 2>/dev/null | wc -l); no=$(ls "$2/{}"/ql_evt*/pctree-evt*.tar.gz 2>/dev/null | wc -l); if [ "$nb" -gt 0 ] && [ "$no" -eq "$nb" ]; then echo "[{}] complete -- skipped"; exit 0; fi; JOBS=1 "$0/scripts/d123/hits_arm.sh" "$1/{}" "$2/{}" "$3" "${@:5}" > "$4/{}.log" 2>&1; echo "[{}] rc=$? ql_evt=$(ls -d "$2/{}"/ql_evt* 2>/dev/null | wc -l)"' \
        "$SX" "$B" "$O" "$REALITY" "$LOGD" "${EXTRA[@]}" "$@"
else
    JOBS=$J "$SX/scripts/d123/hits_arm.sh" "$B" "$O" "$REALITY" "${EXTRA[@]}" "$@" > "$LOGD/all.log" 2>&1
    echo "[all] rc=$? ql_evt=$(ls -d "$O"/ql_evt* 2>/dev/null | wc -l)"
fi
echo "=== $ARM arm $S finished in $(( $(date +%s) - t0 )) s; ql_evt dirs: $(ls -d "$O"/f*/ql_evt* "$O"/ql_evt* 2>/dev/null | wc -l)"
