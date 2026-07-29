#!/usr/bin/env bash
# Doc 66 sec 10: one dQ/dx-vs-residual-range figure per STM verdict flip, with the
# OLD (DL/DT 6.5781/13.1349) and NEW (4.0/8.8) fit of the SAME bundle overlaid.
#
# Both arms ran the same binary over the same Q/L pctrees and differ only in the
# runtime TrackFitting JSON, and every one of these bundles has identical
# npts/kink/exit_L/left_L in the two arms -- so the two traces in each figure
# differ ONLY in the charge apportioned to each fitted point.  That is what makes
# these plots readable as "what the diffusion change did to dQ/dx".
#
# Block id = cluster_id*10 + pass (stmfit_particle_overlay.py).  The pass listed
# per bundle is the DECIDING one -- the pass whose status differs between the
# arms, read from `persist_stm_fit: cluster N stmfit pass=P status=S` in each
# arm's .log_<evt>.log.  315849:10 is the only two-pass case (pass 0 exits
# status=2 in both arms; pass 1 is where the verdict changes) -> block 101.
#
# Usage: ./d66_flip_plots.sh [outdir]        (default pics/d66)
set -u
cd "$(dirname "$0")"
OUT=${1:-pics/d66}
mkdir -p "$OUT"

OLD=work-stmcamp-d66old
NEW=work-stmcamp-d66new

# evt : main : block : flip : classification-vs-doc-62-owner-truth
BUNDLES="
281632:8:80:STM->nu:REGRESSION_owner-STM
283463:14:140:STM->nu:REGRESSION_owner-STM
315849:10:101:STM->nu:REGRESSION_owner-STM
319809:20:200:nu->STM:REGRESSION_owner-notSTM
321107:13:130:STM->nu:FIX_owner-notSTM
58345:7:70:STM->nu:unadjudicated
58755:21:210:nu->STM:unadjudicated
63163:6:60:STM->nu:unadjudicated
289295:15:150:nu->STM:unadjudicated
317543:15:150:nu->STM:unadjudicated
390864:16:160:nu->STM:unadjudicated
"

LOG=$OUT/overlay-numbers.txt
: > "$LOG"
fail=0

for row in $BUNDLES; do
    IFS=: read -r evt main blk flip cls <<< "$row"
    png=$OUT/d66_evt${evt}_main${main}.png
    {
        echo
        echo "############################################################"
        echo "# evt $evt main $main  block $blk  $flip  [$cls]"
        echo "############################################################"
    } >> "$LOG"
    python3 stmfit_particle_overlay.py -o "$png" \
        "$OLD:$evt:$blk:evt $evt main $main  OLD 6.5781/13.1349" \
        "$NEW:$evt:$blk:evt $evt main $main  NEW 4.0/8.8" \
        >> "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        printf "%-14s FAILED rc=%s\n" "$evt:$main" "$rc"; fail=$((fail+1))
    else
        printf "%-14s %-10s %-26s -> %s\n" "$evt:$main" "$flip" "$cls" "$png"
    fi
done

echo
echo "figures: $(ls "$OUT"/d66_evt*.png 2>/dev/null | wc -l)/11 in $OUT   failures: $fail"
echo "per-bundle ratio tables: $LOG"
