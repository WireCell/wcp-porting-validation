#!/usr/bin/env bash
# doc pdhd/25 -- one arm on libpin_p96; writes DONE_<arm> with rc and pin md5 before/after.
ARM=$1; TLA_FILE=${2:-}
PIN=/home/xqian/tmp/p96/libpin_p96
S=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
TLA=""; [ -n "$TLA_FILE" ] && TLA=$(cat "$TLA_FILE")
M0=$(md5sum $PIN/libWireCellClus.so | cut -c1-12)
ARM=$ARM DET=pdhd SRC=d16hnu JOBS=8 PIN=$PIN LOGD=/home/xqian/tmp/h25/arm_$ARM PR_TLA="$TLA" bash $S > /home/xqian/tmp/h25/arm_$ARM.log 2>&1
rc=$?
M1=$(md5sum $PIN/libWireCellClus.so | cut -c1-12)
echo "arm=$ARM rc=$rc md5_before=$M0 md5_after=$M1 tla=${TLA:-<none>} finished=$(date '+%F %T')" > /home/xqian/tmp/h25/DONE_$ARM
