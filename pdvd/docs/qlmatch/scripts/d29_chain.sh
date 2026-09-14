#!/bin/bash
# doc qlmatch/29 -- run the pre-registered arms (d29/prereg.md sec 1-2) one after another, each: a compile-only config
# proof on proof tag p101cfg idx 0 (same env as the arm) -> d29_arms.sh -> ql_agree_score.py into a NEW work/ql_scores tag.
#
#   setsid nohup bash d29_chain.sh step0 > /home/xqian/tmp/p29/chain_step0.log 2>&1 < /dev/null &
#   setsid nohup bash d29_chain.sh levers > /home/xqian/tmp/p29/chain_levers.log 2>&1 < /dev/null &
set -u
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
S=$PDVD/docs/qlmatch/scripts
V6=protodunevd-wires-larsoft-v6.json.bz2
CFG=/home/xqian/tmp/p29/cfg; mkdir -p "$CFG"
arm() {   # ARM SRC WIRES TLA ENVS
    local ARM=$1 SRC=$2 WIRES=$3 TLA=$4 ENVS=$5 rc
    echo "=== $ARM start $(date -Is) src=$SRC wires='$WIRES' tla='$TLA' envs='$ENVS'"
    (cd "$PDVD" && env -u PDVD_READOUT_NTICKS -u PDVD_QTOL -u PDVD_CLUS_WIRES ${WIRES:+PDVD_CLUS_WIRES=$WIRES} \
        PDVD_CLUS_TLA="$TLA" $ENVS PDVD_ALLOW_STALE_GEOMETRY=1 PDVD_CLUS_COMPILE_ONLY=1 PDVD_LIGHT_SUFFIX=_keep \
        ./run_clus_evt.sh -s p101cfg -calib -save-pctree 39252 0) > "$CFG/compile_$ARM.log" 2>&1
    cp -p "$PDVD/work/039252_0_p101cfg/.wct-clus.json" "$CFG/$ARM.json"
    ARM=$ARM SRC=$SRC WIRES=$WIRES TLA=$TLA ENVS=$ENVS bash "$S/d29_arms.sh"; rc=$?
    echo "=== $ARM arms rc=$rc"
    if [ -e "$PDVD/work/ql_scores/$ARM" ]; then echo "REFUSE score $ARM exists"; return; fi
    (cd "$PDVD" && python3 ql_display/ql_agree_score.py --tag "$ARM" --truth-uid-map-tag keep) > "/home/xqian/tmp/p29/score_$ARM.log" 2>&1
    echo "=== $ARM score rc=$? $(grep '^| \*\*all' "$PDVD/work/ql_scores/$ARM/scores.md" 2>/dev/null)"
}
case ${1:?step0|levers|combo} in
  step0)
    arm q29base pvdimg "" "" ""
    arm q29v6   keep   "$V6" "" ""
    arm q29v6nw keep   "$V6" "-S wrapped_channel_charge=false" ""
    arm q29v7nw pvdimg "" "-S wrapped_channel_charge=false" ""
    # tm0k's scoring mode (no map): keep's clustering is the truth's
    if [ ! -e "$PDVD/work/ql_scores/q29v6_nomap" ]; then
        (cd "$PDVD" && python3 ql_display/ql_agree_score.py --tag q29v6 --out-root work/ql_scores_nomap) \
            > /home/xqian/tmp/p29/score_q29v6_nomap.log 2>&1
        echo "=== q29v6 no-map score rc=$? $(grep '^| \*\*all' "$PDVD/work/ql_scores_nomap/q29v6/scores.md" 2>/dev/null)"
    fi ;;
  levers)
    arm q29l1 pvdimg "" "" "PDVD_QL_LASSO_LAMBDA=0.1"
    arm q29l2 pvdimg "" "" "PDVD_QL_LASSO_LAMBDA=0.15"
    arm q29l3 pvdimg "" "" "PDVD_QL_LASSO_LAMBDA=0.3"
    arm q29l4 pvdimg "" "" "PDVD_QL_STRENGTH_CUTOFF=0.02"
    arm q29l5 pvdimg "" "" "PDVD_QL_BKG_WEIGHT=0.3"
    arm q29l6 pvdimg "" "" "PDVD_QL_LASSO_BWEIGHT=0.1"
    arm q29l7 pvdimg "" "" "PDVD_QL_LASSO_BWEIGHT=0.4"
    arm q29l8 pvdimg "" "" "PDVD_QL_HC_CLEAN_C2N=35 PDVD_QL_HC_GOOD_C2N=35 PDVD_QL_HC_TB_C2N=35 PDVD_QL_HC_MISS_C2N=60"
    arm q29l9 pvdimg "" "" "PDVD_QL_PIN_MIN_STRENGTH=" ;;
  combo)
    arm q29c1 pvdimg "${COMBO_WIRES:-}" "${COMBO_TLA:-}" "${COMBO_ENVS:?set COMBO_ENVS}" ;;
esac
echo "=== chain $1 finished $(date -Is)"
