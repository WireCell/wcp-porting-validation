#!/bin/bash
# doc pr/20 II S4: the OFF arm.  Reverts the A1/A2 config line for the duration,
# runs all three OFF sweeps, restores it.  Same binary as the ON arm (recorded).
set -u
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
L=/home/xqian/tmp/pr20exec
CFG=$TK/cfg/pgrapher/experiment/sbnd/clus.jsonnet

# ATOMIC config swap.  `cp -f` truncates and rewrites in place, so a wire-cell
# job that compiles its config mid-write sees a TORN file.  That is exactly what
# killed 6/1000 events in the first ON arm ("Unknown variable: cathode_x",
# cfg/pgrapher/common/clus.jsonnet:446, all six inside a 1.6 s window).
# mv within one filesystem is a rename: a reader sees the whole old file or the
# whole new one, never a mixture.
swap() { cp -f "$1" "$CFG.tmp$$" && mv -f "$CFG.tmp$$" "$CFG"; }

restore() { swap $L/clus_A12.jsonnet; echo "[cfg] A1/A2 restored"; }
trap restore EXIT

swap $L/clus_OFF.jsonnet
echo "[cfg] reverted to HEAD (A1/A2 off) at $(date -Is)"
md5sum /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCellClus.so > $L/binary_off.txt

cd "$SX"
echo "=== [off] 11-event Bee arm ==="
TAG=cathA12beeoff ENTRIES="454 185 191 366 316 496 314 164 249 263 809" \
    CATHODE_CONNECT_DEBUG=1 CC_FEATURE_DUMP=1 ./run_full1k_nusel.sh 1000 6 \
    > $L/s4_bee_off.log 2>&1
echo "bee rc=$?"

echo "=== [off] mcp1k 1000 events, 8 jobs ==="
TAG=cathA12off ./run_full1k_nusel.sh 1000 8 > $L/s4_mcp1k_off.log 2>&1
echo "mcp1k rc=$?"

echo "=== [off] nueCC48, 6 jobs ==="
# run_nusel_evt.sh aborts unless imaging is already staged in the work root
# ("ERROR: missing icluster-apa0-active.npz -- run ./run_img_evt.sh 1 first"),
# which is why the first ON attempt returned rc=123 with 0 completions.
# s4_nuecc48.sh seeds it from our own oc19on arm, then runs.
$L/s4_nuecc48.sh cathA12off > $L/s4_nuecc48_off.log 2>&1
echo "nuecc48 rc=$?"

md5sum /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCellClus.so >> $L/binary_off.txt
echo "S4-off DONE"
