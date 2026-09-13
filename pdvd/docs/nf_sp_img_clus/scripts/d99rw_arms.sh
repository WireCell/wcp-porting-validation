#!/bin/bash
# doc pdvd/99 sec 4.4 -- give clustering and PR the readout window each event's SP frames really have
# (d99/frame_nticks_120evt.txt: runs 039252 and 039253 10000 ticks, run 039349 6400), on production and on the latest
# configuration.  Production staging never holds frames, so every production event ran at the runner's 10000 fallback
# (sec 4.1); only run 039349 is wrong by that.
#
#   WAVE=guard ARM=p99rgprod SRC=p96vprod ./d99rw_arms.sh
#   WAVE=guard ARM=p99rgon   SRC=p98vonq  ./d99rw_arms.sh
#       PR only.  The source arm's pctree is linked (readlink -f) and its .tlas COPIED with readout_window_ticks set to
#       the event's frame length.  Clustering, flash matching and cluster ids stay the source's, so the only thing that
#       moves is what PR does with the window (TaggerCheckSTM readout_edge_guard).  The record keys join directly.
#   WAVE=full  ARM=p99rwprod SRC=d27fresh ./d99rw_arms.sh
#   WAVE=full  ARM=p99rwon   SRC=p98von   ./d99rw_arms.sh
#       clustering + PR staged like production (d99_stage_q.sh: imaging archives only) with PDVD_READOUT_NTICKS set per
#       event (run_clus_evt.sh, default unset = the frame-or-10000 rule), so QLMatching's window-truncation flag moves too
#       and cluster ids may change (carry the record with d99_match.py).
#   WAVE=full  ARM=p99rwnul  SRC=p98von EVENTS="039349_7" NOWIN=1 ./d99rw_arms.sh
#       the knob unset: must equal p98vonq.
#
# Gates: on the 36 events whose frames are 10000 ticks the window does not change, so every arm must reproduce its
# source byte for byte there (d99rw_identity.py).  Per event, setarch -R, libpin_p96 prepended, completion judged by
# output (pctree + tlas; tracking-pr.root + "CheckSTM_Michel: N candidate(s)").  Existing arm dirs are refused (M13).
set -u
WAVE=${WAVE:?set WAVE guard|full}; ARM=${ARM:?set ARM}; SRC=${SRC:?set SRC}
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
X=$PDVD/docs/nf_sp_img_clus/scripts
NT_TABLE=$PDVD/docs/nf_sp_img_clus/d99/frame_nticks_120evt.txt
PIN=/home/xqian/tmp/p96/libpin_p96
LOGD=/home/xqian/tmp/p99rw/arm_$ARM
[ -e "$LOGD" ] && { echo "REFUSE existing log dir $LOGD" >&2; exit 2; }
mkdir -p "$LOGD"
[ -s "$PIN/libWireCellClus.so" ] || { echo "no pin $PIN" >&2; exit 2; }
export LD_LIBRARY_PATH="$PIN:${LD_LIBRARY_PATH:-}"
[ "$(pgrep -c '^wire-cell')" = 0 ] || echo "[$ARM] note: $(pgrep -c '^wire-cell') wire-cell job(s) already live"
if [ -n "${EVENTS:-}" ]; then EVS=$EVENTS; else
    EVS=$(grep -v '^#' "$PDVD/stm/events.txt" | awk 'NF>=2 {printf "%06d_%s\n", $1, $2}'); fi
export ARM SRC PDVD NT_TABLE LOGD WAVE NOWIN="${NOWIN:-0}"
PIN_MD5=$(md5sum < "$PIN/libWireCellClus.so" | cut -c1-12)
echo "[$ARM] wave $WAVE src $SRC start $(date -Is) pin Clus $PIN_MD5 events $(echo $EVS | wc -w)" | tee -a "$LOGD/arm.log"

clus_ok() { ls "$1"/pctree-evt*.tar.gz >/dev/null 2>&1 && ls "$1"/pctree-evt*.tlas >/dev/null 2>&1; }
pr_ok()   { [ -s "$1/tracking-pr.root" ] && grep -q "CheckSTM_Michel: .* candidate(s)" "$1"/wct_pr_*.log 2>/dev/null; }
export -f clus_ok pr_ok

stage_guard() {   # one event: link the source pctree, copy its tlas with the event's window
    local e=$1 s=$PDVD/work/${1}_$SRC d=$PDVD/work/${1}_$ARM nt pc tl n
    nt=$(awk -v e="$e" '$1==e {print $2}' "$NT_TABLE"); [ -n "$nt" ] || { echo "NONT $e"; return 1; }
    pc=$(ls "$s"/pctree-evt*.tar.gz 2>/dev/null | head -1); tl=$(ls "$s"/pctree-evt*.tlas 2>/dev/null | head -1)
    [ -n "$pc" ] && [ -n "$tl" ] || { echo "NOSRC $e"; return 1; }
    [ -e "$d" ] && { echo "REFUSE $d"; return 1; }
    mkdir -p "$d"
    ln -s "$(readlink -f "$pc")" "$d/$(basename "$pc")"
    sed "s/^readout_window_ticks=.*/readout_window_ticks=$nt/" "$(readlink -f "$tl")" > "$d/$(basename "$tl")"
    n=$(diff "$(readlink -f "$tl")" "$d/$(basename "$tl")" | grep -c '^>')
    grep -q "^readout_window_ticks=$nt$" "$d/$(basename "$tl")" || { echo "BADTLAS $e"; return 1; }
    echo "staged $e nt=$nt tlas_lines_changed=$n"
}
export -f stage_guard

one() {   # stage run idx
    local st=$1 run=$2 idx=$3 e d rc nt
    e=$(printf %06d "$run")_$idx; d=$PDVD/work/${e}_$ARM
    case $st in
      clus)
        clus_ok "$d" && { echo "skip clus $e"; return 0; }
        nt=$(awk -v e="$e" '$1==e {print $2}' "$NT_TABLE")
        if [ "$NOWIN" = 1 ]; then
            (cd "$PDVD" && env PDVD_LIGHT_SUFFIX=_keep setarch x86_64 -R ./run_clus_evt.sh -s "$ARM" -calib -save-pctree "$run" "$idx") \
                > "$LOGD/clus_${e}.log" 2>&1; rc=$?
        else
            [ -n "$nt" ] || { echo "NONT $e"; return 0; }
            (cd "$PDVD" && env PDVD_LIGHT_SUFFIX=_keep PDVD_READOUT_NTICKS="$nt" setarch x86_64 -R ./run_clus_evt.sh -s "$ARM" -calib -save-pctree "$run" "$idx") \
                > "$LOGD/clus_${e}.log" 2>&1; rc=$?
        fi
        clus_ok "$d" && echo "ok clus rc=$rc $e nt=$([ "$NOWIN" = 1 ] && echo unset || echo "$nt")" || echo "FAIL clus rc=$rc $e" ;;
      pr)
        clus_ok "$d" || { echo "NOCLUS $e"; return 0; }
        pr_ok "$d" && { echo "skip pr $e"; return 0; }
        (cd "$PDVD" && env PDVD_LIGHT_SUFFIX=_keep PDVD_PR_TLA="" setarch x86_64 -R ./run_pr_evt.sh -nu -stm-fit -s "$ARM" "$run" "$idx") \
            > "$LOGD/pr_${e}.log" 2>&1; rc=$?
        pr_ok "$d" && echo "ok pr rc=$rc $e" || echo "FAIL pr rc=$rc $e" ;;
    esac
}
export -f one

run_stage() {   # stage jobs
    echo "$EVS" | tr ' ' '\n' | awk 'NF {split($1,a,"_"); print a[1]+0, a[2]}' \
        | xargs -P "$2" -n 2 bash -c "one $1 \"\$@\"" _ >> "$LOGD/$1.status" 2>&1
    echo "[$ARM] $1 done $(date -Is): ok $(grep -c '^ok \|^skip ' "$LOGD/$1.status") FAIL $(grep -c '^FAIL' "$LOGD/$1.status") no-input $(grep -c '^NO' "$LOGD/$1.status") loadavg $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$LOGD/arm.log"
}

case $WAVE in
  guard)
    for e in $EVS; do stage_guard "$e"; done > "$LOGD/stage.status" 2>&1
    echo "[$ARM] staged $(grep -c '^staged' "$LOGD/stage.status"), problems $(grep -vc '^staged' "$LOGD/stage.status")" | tee -a "$LOGD/arm.log"
    grep -vq '^staged' "$LOGD/stage.status" && { echo "[$ARM] staging problems, not running" | tee -a "$LOGD/arm.log"; exit 1; }
    run_stage pr 8 ;;
  full)
    (cd "$X" && SRC=$SRC DST=$ARM EVENTS="$EVS" ./d99_stage_q.sh) > "$LOGD/stage.status" 2>&1 || { cat "$LOGD/stage.status"; exit 1; }
    tail -1 "$LOGD/stage.status" | tee -a "$LOGD/arm.log"
    run_stage clus 12
    run_stage pr 8 ;;
  *) echo "WAVE must be guard|full" >&2; exit 2 ;;
esac
echo "[$ARM] finished $(date -Is) pin Clus $(md5sum < "$PIN/libWireCellClus.so" | cut -c1-12) (start $PIN_MD5)" | tee -a "$LOGD/arm.log"
