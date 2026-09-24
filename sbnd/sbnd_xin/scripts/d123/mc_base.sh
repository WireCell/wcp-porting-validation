#!/bin/bash
# doc sbnd_xin/123 round 3 (MC): the BASELINE stage A (reco1 flashes, knob absent) on a round-3 sample,
# at the d123 lib pin, KEEPING the imaging npz so hit-flash arms can share it (mc_hits.sh).
# Layout as scripts/d115/stageA.sh: one out_root per reco1 file for cv/nuecc (event numbers repeat
# across files), one root of 16-event groups for beam-off.
#
# Usage: [JOBS=n] scripts/d123/mc_base.sh <cv|nuecc|off>
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
S=${1:?usage: [JOBS=n] mc_base.sh <cv|nuecc|off>}
J=${JOBS:-6}
P=${PIN:-$HOME/tmp/d123-libpin}
export LD_LIBRARY_PATH=$P:$P/reco1/lib:${LD_LIBRARY_PATH:-}
export SBND_QL_KEEP_ICLUSTER=1
case "$S" in
    cv)    OUT=$SX/work-r3cv-d123base;  REALITY=sim;  MCF_STR=--mc; GSIZE=1000; PERFILE=1 ;;
    nuecc) OUT=$SX/work-r3nue-d123base; REALITY=sim;  MCF_STR=--mc; GSIZE=1000; PERFILE=1 ;;
    off)   OUT=$SX/work-r3off-d123base; REALITY=data; MCF_STR='';   GSIZE=16;   PERFILE=0 ;;
    *) echo "unknown sample: $S (cv|nuecc|off)" >&2; exit 2 ;;
esac
LST=$SX/products/d115/$S/files.lst
[ -s "$LST" ] || { echo "ERROR: no $LST" >&2; exit 1; }
LOGD=$HOME/tmp/d123-mc-$S-base; mkdir -p "$LOGD" "$OUT"
# run_chain_group.sh refuses a non-empty out_root it did not create (M13) unless it carries its marker;
# the per-file layout creates f<i>/ itself, the single-root layout writes into $OUT directly.
touch "$OUT/.chain_group"; echo "$REALITY" > "$OUT/.lineage_reality"
md5sum $P/libWireCellClus.so $P/libWireCellMatch.so $P/libWireCellFlash.so $P/reco1/lib/libWireCellSBNDReco1.so > "$OUT/.libs.md5.start"
echo "=== d123 MC baseline: sample=$S files=$(wc -l < "$LST") jobs=$J reality=$REALITY pin=$P ($(cat "$P/TOOLKIT_HEAD"))"
t0=$(date +%s)
one() {  # one <fileidx> <path>
    local i=$1 f=$2 g="$OUTT/f$1"
    if [ -s "$g/g0/events.txt" ] \
       && [ "$(wc -l < "$g/g0/events.txt")" -eq "$(ls -d "$g"/ql_evt* 2>/dev/null | wc -l)" ] \
       && [ "$(ls -d "$g"/ql_evt* 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[f$i] already complete -- skipped"; return 0
    fi
    SBND_MAX_JOBS=1 "$SXX/run_chain_group.sh" "$f" "$g" "$REAL" $MCF_STR --size "$GSZ" --layout perevt > "$LOGDD/f$i.log" 2>&1
    echo "[f$i] rc=$? evt=$(ls -d "$g"/ql_evt* 2>/dev/null | wc -l)"
    return 0
}
export -f one
export SXX=$SX OUTT=$OUT LOGDD=$LOGD REAL=$REALITY GSZ=$GSIZE MCF_STR
if [ "$PERFILE" = 1 ]; then
    awk -F'\t' '{print $1"\t"$2}' "$LST" \
      | xargs -P "$J" -I{} bash -c 'IFS=$(printf "\t"); set -- {}; one "$1" "$2"'
else
    f=$(awk -F'\t' 'NR==1{print $2; exit}' "$LST")
    SBND_MAX_JOBS=$J "$SX/run_chain_group.sh" "$f" "$OUT" "$REALITY" --size "$GSIZE" --layout perevt > "$LOGD/all.log" 2>&1
    echo "[all] rc=$? evt=$(ls -d "$OUT"/ql_evt* 2>/dev/null | wc -l)"
fi
md5sum $P/libWireCellClus.so $P/libWireCellMatch.so $P/libWireCellFlash.so $P/reco1/lib/libWireCellSBNDReco1.so > "$OUT/.libs.md5.end"
echo "=== stage A $S finished in $(( $(date +%s) - t0 )) s; ql_evt dirs: $(ls -d "$OUT"/f*/ql_evt* "$OUT"/ql_evt* 2>/dev/null | wc -l) (expect $(( $(wc -l < "$SX/products/d115/$S/file_rse.tsv") - 1 )))"
