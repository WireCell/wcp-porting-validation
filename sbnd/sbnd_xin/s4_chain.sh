#!/bin/bash
# doc pr/20 II S4, relaunch.  The first ON arm (work-mcp1kall-cathA12on) was
# taken while cfg/pgrapher/common/clus.jsonnet was being edited for B0 -- the
# arm started 21:48:14, the B0 body edit landed before its signature (6 events
# died 21:49:46-47 on "Unknown variable: cathode_x"), and the revert only
# happened at 21:53:02.  So a large slice of that arm compiled its config with
# the COMPLETE B0 config state present.  That state is inert in the compiled
# JSON (proven for the Q/L job: identical to the clean-tree compile), but the
# arm is not single-provenance and is therefore retired rather than argued for.
#
# This run takes ON and OFF back-to-back from ONE unchanging tree:
#   toolkit aff0ffde + the A1/A2 line only, libWireCellClus.so 525a7c21...
# NOTHING may edit the toolkit tree until "S4-chain DONE" appears.
set -u
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
L=/home/xqian/tmp/pr20exec

cd "$SX" || exit 1
md5sum /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCellClus.so  > $L/binary_on2.txt
git -C /nfs/data/1/xqian/toolkit-dev/toolkit status --porcelain --untracked-files=no >> $L/binary_on2.txt

echo "=== [on2] mcp1k 1000 events, 8 jobs === $(date -Is)"
TAG=cathA12on2 ./run_full1k_nusel.sh 1000 8 > $L/s4_mcp1k_on2.log 2>&1
echo "mcp1k-on2 rc=$?"
echo "on2 fails: $(grep -l 'rc=1' $SX/work-mcp1kall-cathA12on2/.status/* 2>/dev/null | wc -l)"

md5sum /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCellClus.so >> $L/binary_on2.txt

echo "=== [off] handing over to s4_off.sh === $(date -Is)"
$L/s4_off.sh > $L/s4_off_driver.log 2>&1
echo "off rc=$?"
echo "S4-chain DONE $(date -Is)"
