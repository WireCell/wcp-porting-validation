#!/usr/bin/env bash
# doc pdhd/28 -- one arm on libpin_h28 (the wire-lookup build); writes DONE_<arm> with rc and pin md5 before/after.
# Fork by duplication of pdhd/docs/scan/h25/run_arm.sh: DET is a parameter, the pin and log dir are doc 28's.
# Usage: bash run_arm28.sh <pdhd|pdvd> <arm> [tla_file]
# Arms of this round: h28off h28wl h28prod (pdhd), p97voff p97vwl (pdvd).
DET=$1; ARM=$2; TLA_FILE=${3:-}
PIN=/home/xqian/tmp/h28/libpin_h28
S=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
case "$DET" in pdhd) SRC=d16hnu; JOBS=8 ;; pdvd) SRC=d16vnu; JOBS=6 ;; *) echo "DET?"; exit 2 ;; esac
[ -s $PIN/libWireCellClus.so ] || { echo "REFUSING: no pin at $PIN"; exit 2; }
if [ -n "$(ls -d /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/$DET/work/*_$ARM 2>/dev/null)" ]; then
    echo "REFUSING: $DET/work/*_$ARM exists (M13)"; exit 2
fi
TLA=""; [ -n "$TLA_FILE" ] && TLA=$(cat "$TLA_FILE")
M0=$(md5sum $PIN/libWireCellClus.so | cut -c1-12)
ARM=$ARM DET=$DET SRC=$SRC JOBS=$JOBS PIN=$PIN LOGD=/home/xqian/tmp/h28/arm_$ARM PR_TLA="$TLA" bash $S > /home/xqian/tmp/h28/arm_$ARM.log 2>&1
rc=$?
M1=$(md5sum $PIN/libWireCellClus.so | cut -c1-12)
n=$(ls /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/$DET/work/*_$ARM/tracking-pr.root 2>/dev/null | wc -l)
echo "arm=$ARM det=$DET rc=$rc events_with_output=$n md5_before=$M0 md5_after=$M1 tla=${TLA:-<none>} finished=$(date '+%F %T')" > /home/xqian/tmp/h28/DONE_$ARM
