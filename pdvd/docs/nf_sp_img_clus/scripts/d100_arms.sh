#!/bin/bash
# doc pdvd/100 -- the gain round's arms (fork of d99rw_arms.sh; that script stays untouched).
#
#   WAVE=pr   ARM=p100c   SRC=p99rwon PR_TLA="-S stm_recomb_C=0.8630" ./d100_arms.sh
#       PR only on the source arm's clustering: its pctree is linked (readlink -f) and its .tlas COPIED verbatim (the
#       source already ran each event's real window), then run_pr_evt.sh -nu -stm-fit with PDVD_PR_TLA="$PR_TLA".  The
#       only thing that moves is what the TLA changes; cluster ids are the source's, so its record keys join directly.
#   WAVE=flip ARM=p100flip ./d100_arms.sh
#       production as shipped, no overrides: stm/run_campaign.sh's staging (scripts/stage_ql_tag.sh, its default source
#       tag), run_clus_evt.sh -calib -save-pctree with PDVD_LIGHT_SUFFIX=_keep and nothing else set (the window from
#       readout_window_ticks.txt), then run_pr_evt.sh -nu -stm-fit with PDVD_PR_TLA empty.
#
# Per event, setarch -R, libpin_p96 prepended, completion judged by output (pctree + tlas; tracking-pr.root +
# "CheckSTM_Michel: N candidate(s)").  Existing arm or log dirs are refused (M13).
set -u
WAVE=${WAVE:?set WAVE pr|flip}; ARM=${ARM:?set ARM}
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
PIN=${PIN:-/home/xqian/tmp/p96/libpin_p96}   # round 2: the exemption's binary is libpin_p100b
LOGD=/home/xqian/tmp/p100/arm_$ARM
[ -e "$LOGD" ] && { echo "REFUSE existing log dir $LOGD" >&2; exit 2; }
mkdir -p "$LOGD"
[ -s "$PIN/libWireCellClus.so" ] || { echo "no pin $PIN" >&2; exit 2; }
export LD_LIBRARY_PATH="$PIN:${LD_LIBRARY_PATH:-}"
[ "$(pgrep -c '^wire-cell')" = 0 ] || echo "[$ARM] note: $(pgrep -c '^wire-cell') wire-cell job(s) already live"
if [ -n "${EVENTS:-}" ]; then EVS=$EVENTS; else
    EVS=$(grep -v '^#' "$PDVD/stm/events.txt" | awk 'NF>=2 {printf "%06d_%s\n", $1, $2}'); fi
export ARM PDVD LOGD WAVE SRC="${SRC:-}" PR_TLA="${PR_TLA:-}"
PIN_MD5=$(md5sum < "$PIN/libWireCellClus.so" | cut -c1-12)
echo "[$ARM] wave $WAVE src ${SRC:-production staging} PR_TLA '${PR_TLA}' start $(date -Is) pin Clus $PIN_MD5 events $(echo $EVS | wc -w)" | tee -a "$LOGD/arm.log"

clus_ok() { ls "$1"/pctree-evt*.tar.gz >/dev/null 2>&1 && ls "$1"/pctree-evt*.tlas >/dev/null 2>&1; }
pr_ok()   { [ -s "$1/tracking-pr.root" ] && grep -q "CheckSTM_Michel: .* candidate(s)" "$1"/wct_pr_*.log 2>/dev/null; }
export -f clus_ok pr_ok

stage_link() {   # one event: link the source pctree, copy its tlas verbatim
    local e=$1 s=$PDVD/work/${1}_$SRC d=$PDVD/work/${1}_$ARM pc tl
    pc=$(ls "$s"/pctree-evt*.tar.gz 2>/dev/null | head -1); tl=$(ls "$s"/pctree-evt*.tlas 2>/dev/null | head -1)
    [ -n "$pc" ] && [ -n "$tl" ] || { echo "NOSRC $e"; return 1; }
    [ -e "$d" ] && { echo "REFUSE $d"; return 1; }
    mkdir -p "$d"
    ln -s "$(readlink -f "$pc")" "$d/$(basename "$pc")"
    cp -p "$(readlink -f "$tl")" "$d/$(basename "$tl")"
    cmp -s "$(readlink -f "$tl")" "$d/$(basename "$tl")" || { echo "BADTLAS $e"; return 1; }
    echo "staged $e"
}

one() {   # stage run idx
    local st=$1 run=$2 idx=$3 e d rc
    e=$(printf %06d "$run")_$idx; d=$PDVD/work/${e}_$ARM
    case $st in
      stage)
        [ -e "$d" ] && { echo "REFUSE $d"; return 0; }
        (cd "$PDVD" && ./scripts/stage_ql_tag.sh "$run" "$idx" "$ARM") > "$LOGD/stage_${e}.log" 2>&1; rc=$?
        [ "$rc" = 0 ] && echo "staged $e" || echo "FAIL stage rc=$rc $e" ;;
      clus)
        clus_ok "$d" && { echo "skip clus $e"; return 0; }
        (cd "$PDVD" && env -u PDVD_READOUT_NTICKS -u PDVD_QTOL PDVD_LIGHT_SUFFIX=_keep setarch x86_64 -R \
            ./run_clus_evt.sh -s "$ARM" -calib -save-pctree "$run" "$idx") > "$LOGD/clus_${e}.log" 2>&1; rc=$?
        clus_ok "$d" && echo "ok clus rc=$rc $e" || echo "FAIL clus rc=$rc $e" ;;
      pr)
        clus_ok "$d" || { echo "NOCLUS $e"; return 0; }
        pr_ok "$d" && { echo "skip pr $e"; return 0; }
        (cd "$PDVD" && env PDVD_LIGHT_SUFFIX=_keep PDVD_PR_TLA="$PR_TLA" setarch x86_64 -R ./run_pr_evt.sh -nu -stm-fit -s "$ARM" "$run" "$idx") \
            > "$LOGD/pr_${e}.log" 2>&1; rc=$?
        pr_ok "$d" && echo "ok pr rc=$rc $e" || echo "FAIL pr rc=$rc $e" ;;
    esac
}
export -f one

run_stage() {   # stage jobs
    echo "$EVS" | tr ' ' '\n' | awk 'NF {split($1,a,"_"); print a[1]+0, a[2]}' \
        | xargs -P "$2" -n 2 bash -c "one $1 \"\$@\"" _ >> "$LOGD/$1.status" 2>&1
    echo "[$ARM] $1 done $(date -Is): ok $(grep -c '^ok \|^skip \|^staged' "$LOGD/$1.status") FAIL $(grep -c '^FAIL' "$LOGD/$1.status") no-input $(grep -c '^NO\|^REFUSE' "$LOGD/$1.status") loadavg $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$LOGD/arm.log"
}

case $WAVE in
  pr)
    [ -n "$SRC" ] || { echo "WAVE=pr needs SRC" >&2; exit 2; }
    for e in $EVS; do stage_link "$e"; done > "$LOGD/stage.status" 2>&1
    echo "[$ARM] staged $(grep -c '^staged' "$LOGD/stage.status"), problems $(grep -vc '^staged' "$LOGD/stage.status")" | tee -a "$LOGD/arm.log"
    grep -vq '^staged' "$LOGD/stage.status" && { echo "[$ARM] staging problems, not running" | tee -a "$LOGD/arm.log"; exit 1; }
    run_stage pr 8 ;;
  flip)
    [ -z "${PR_TLA}" ] || { echo "WAVE=flip runs production with no overrides; PR_TLA must be empty" >&2; exit 2; }
    run_stage stage 8
    grep -q '^FAIL\|^REFUSE' "$LOGD/stage.status" && { echo "[$ARM] staging problems, not running" | tee -a "$LOGD/arm.log"; exit 1; }
    run_stage clus 12
    run_stage pr 8 ;;
  *) echo "WAVE must be pr|flip" >&2; exit 2 ;;
esac
echo "[$ARM] finished $(date -Is) pin Clus $(md5sum < "$PIN/libWireCellClus.so" | cut -c1-12) (start $PIN_MD5)" | tee -a "$LOGD/arm.log"
