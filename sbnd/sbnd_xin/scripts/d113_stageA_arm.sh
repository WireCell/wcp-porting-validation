#!/bin/bash
# doc sbnd_xin/113 -- stage-A (imaging + clustering/QL) arm on selected groups of a production sample,
# re-using the production reco1 dump (frames-dnn.tar.bz2 + opflash tarballs of work-<s>-d102m/g<K>, our
# own products) so the art file is not re-read.  Fresh out_root work-<s>-d113<tag> (M13).
#
#   scripts/d113_stageA_arm.sh <sample> <tag> <groups: 0,3,7 | all> [IMG_EXTRA_TLA file]
#
# Env: JOBS (default 8, <= 32 procs in flight), KEEP_ICLUSTER (default 1: keep the imaging npz so the blob
# set can be read per event), FROM (default img), TO (default ql), PIN (a lib dir to prepend; default none =
# the installed local/lib).  The runner precompiles the imaging config; with the TLA file set, IMG_EXTRA_TLA
# appends its lines (doc 113 knobs), else the compiled JSON is byte-identical to production.
set -euo pipefail
SX=/home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
s=$1; tag=$2; groups=$3; tla=${4:-}
src=$SX/work-$s-d102m
out=$SX/work-$s-d113$tag
[ -d "$src" ] || { echo "no source arm $src" >&2; exit 1; }
if [ -e "$out" ] && [ ! -f "$out/.d113_arm" ]; then echo "refusing to touch existing $out (M13)" >&2; exit 1; fi
mkdir -p "$out"; touch "$out/.chain_group" "$out/.d113_arm"; echo data > "$out/.lineage_reality"
input=$(python3 -c "import json,sys; c=json.load(open('$src/g0/.wct-cfg-dump.json')); print([n for n in c if n.get('type')=='SBNDReco1FrameSource'][0]['data']['filename'])")
[ -r "$input" ] || { echo "reco1 input not readable: $input" >&2; exit 1; }
if [ "$groups" = all ]; then
    groups=$(ls -d "$src"/g[0-9]* | sed 's#.*/g##' | sort -n | paste -sd,)
fi
gbase=0
for k in ${groups//,/ }; do
    g="$out/g$k"; mkdir -p "$g"
    for f in frames-dnn.tar.bz2 opflash_apa0.tar.gz opflash_apa1.tar.gz .wct-cfg-dump.json; do
        [ -e "$g/$f" ] || ln -s "$src/g$k/$f" "$g/$f"
    done
done
# mcp2k part2 groups (63..125) were produced with --gbase 63 from a second art file; the reco1 dump is
# skipped here (frames exist), and the group index only names the directory, so run them by directory name.
export SBND_QL_KEEP_ICLUSTER=${KEEP_ICLUSTER:-1}
export SBND_MAX_JOBS=${JOBS:-8}
[ -n "${PIN:-}" ] && export LD_LIBRARY_PATH=$PIN:${LD_LIBRARY_PATH:-}
[ -n "$tla" ] && export IMG_EXTRA_TLA=$(readlink -f "$tla")
[ -n "${QLTLA:-}" ] && export QL_EXTRA_TLA=$(readlink -f "$QLTLA")
echo "arm $out groups=$groups input=$input tla=${tla:-none} jobs=$SBND_MAX_JOBS keep_icluster=$SBND_QL_KEEP_ICLUSTER"
md5sum /home/xqian/toolkit-dev/local/lib/libWireCellImg.so /home/xqian/toolkit-dev/local/lib/libWireCellClus.so > "$out/.libs.md5.start"
cd "$SX"
./run_chain_group.sh "$input" "$out" data --size 16 --layout perevt --groups "$groups" --from "${FROM:-img}" --to "${TO:-ql}" > "$out/.run.log" 2>&1
rc=$?
md5sum /home/xqian/toolkit-dev/local/lib/libWireCellImg.so /home/xqian/toolkit-dev/local/lib/libWireCellClus.so > "$out/.libs.md5.end"
cmp -s "$out/.libs.md5.start" "$out/.libs.md5.end" || echo "WARNING: library changed during the arm" >&2
echo "rc=$rc" | tee -a "$out/.run.log"
exit $rc
