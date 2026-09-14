#!/bin/bash
# doc qlmatch/29 -- clustering-only Q/L arms on the 18 hand-scanned events of run 039252 (fork of
# pdvd/docs/nf_sp_img_clus/scripts/d100r3_arms.sh; that script stays as committed).
#
#   ARM=q29v6   SRC=keep   WIRES=protodunevd-wires-larsoft-v6.json.bz2                 ./d29_arms.sh
#   ARM=q29v6nw SRC=keep   WIRES=protodunevd-wires-larsoft-v6.json.bz2 TLA="-S wrapped_channel_charge=false" ./d29_arms.sh
#   ARM=q29v7nw SRC=pvdimg TLA="-S wrapped_channel_charge=false"                        ./d29_arms.sh
#   ARM=q29base SRC=pvdimg                                                                ./d29_arms.sh   (control == p100flip)
#   ARM=q29l1   SRC=pvdimg ENVS="PDVD_QL_LASSO_LAMBDA=0.1"                                ./d29_arms.sh   (a lever)
#
# Per event: scripts/stage_ql_tag.sh <run> <idx> ARM SRC (imaging archives + provenance from SRC), then run_clus_evt.sh
# -calib -save-pctree with PDVD_LIGHT_SUFFIX=_keep, PDVD_CLUS_WIRES=$WIRES (unset when empty), PDVD_CLUS_TLA=$TLA and the
# extra env assignments in ENVS; PDVD_READOUT_NTICKS / PDVD_QTOL / PDVD_ALLOW_STALE_GEOMETRY unset (the imaging-provenance
# guard stays armed).  setarch -R, pin prepended, completion = calib dump + pctree.  No PR (Q/L output only).  Existing arm
# or log dirs are refused (M13).
set -u
ARM=${ARM:?set ARM}; SRC=${SRC:?set SRC}; WIRES=${WIRES:-}; TLA=${TLA:-}; ENVS=${ENVS:-}
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
PIN=${PIN:-/home/xqian/tmp/p100/libpin_p100b}
LOGD=/home/xqian/tmp/p29/arm_$ARM
[ -e "$LOGD" ] && { echo "REFUSE existing log dir $LOGD" >&2; exit 2; }
mkdir -p "$LOGD"
[ -s "$PIN/libWireCellClus.so" ] || { echo "no pin $PIN" >&2; exit 2; }
export LD_LIBRARY_PATH="$PIN:${LD_LIBRARY_PATH:-}"
export ARM SRC WIRES TLA ENVS PDVD LOGD
PIN_MD5=$(md5sum < "$PIN/libWireCellClus.so" | cut -c1-12)
echo "[$ARM] src $SRC wires '${WIRES}' tla '${TLA}' envs '${ENVS}' start $(date -Is) pin Clus $PIN_MD5" | tee -a "$LOGD/arm.log"
one() {   # idx
    local idx=$1 e d rc
    e=039252_$idx; d=$PDVD/work/${e}_$ARM
    [ -e "$d" ] && { echo "REFUSE $d"; return 0; }
    (cd "$PDVD" && ./scripts/stage_ql_tag.sh 39252 "$idx" "$ARM" "$SRC") > "$LOGD/stage_${e}.log" 2>&1 || { echo "FAIL stage $e"; return 0; }
    (cd "$PDVD" && env -u PDVD_READOUT_NTICKS -u PDVD_QTOL -u PDVD_ALLOW_STALE_GEOMETRY -u PDVD_CLUS_WIRES \
        ${WIRES:+PDVD_CLUS_WIRES=$WIRES} PDVD_CLUS_TLA="$TLA" $ENVS PDVD_LIGHT_SUFFIX=_keep \
        setarch x86_64 -R ./run_clus_evt.sh -s "$ARM" -calib -save-pctree 39252 "$idx") > "$LOGD/clus_${e}.log" 2>&1; rc=$?
    if ls "$d"/calib-evt*.json >/dev/null 2>&1 && ls "$d"/pctree-evt*.tar.gz >/dev/null 2>&1; then echo "ok clus rc=$rc $e"; else echo "FAIL clus rc=$rc $e"; fi
}
export -f one
seq 0 17 | xargs -P 12 -I{} bash -c 'one {}' >> "$LOGD/clus.status" 2>&1
echo "[$ARM] done $(date -Is): ok $(grep -c '^ok ' "$LOGD/clus.status") FAIL $(grep -c '^FAIL' "$LOGD/clus.status") REFUSE $(grep -c '^REFUSE' "$LOGD/clus.status") pin Clus $(md5sum < "$PIN/libWireCellClus.so" | cut -c1-12) loadavg $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$LOGD/arm.log"
