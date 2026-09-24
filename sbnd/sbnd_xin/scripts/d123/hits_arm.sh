#!/bin/bash
# doc sbnd_xin/123 round 0 -- a HIT-FLASH arm that shares a baseline arm's imaging.
#
#   scripts/d123/hits_arm.sh <base_root> <hits_root> <data|sim> [--ff JSON] [--groups 0,3|all]
#                            [--ref] [--hit-time rise|peak|start] [--no-ql]
#
# For every g<K> of <base_root> (a run_chain_group.sh --layout perevt arm that KEPT its
# icluster npz):
#   1. re-run ONLY the flash part of the reco1 dump with flash_source=hits (wct-reco1-dump.jsonnet,
#      same file / entry range / caf offset mode / frameshift product as the baseline, read from
#      its compiled .wct-cfg-dump.json) into an EMPTY <hits_root>/g<K>  ->  opflash_apa{0,1}.tar.gz
#      (+ reco1flash_apa{0,1}.tar.gz with --ref);
#   2. check the event-id set equals the baseline's events.txt;
#   3. symlink the baseline's frames-dnn.tar.bz2 + icluster-apa*-{active,masked}.npz, COPY events.txt
#      (never symlink anything the runner writes: it would write through into the baseline, M13);
#   4. run_chain_group.sh --from ql --layout perevt on the group  ->  the SAME Q/L job on the same
#      imaging, only the flash file differs.
# Post-checks per group: rse.json identical to the baseline's (run/subrun/event pass through the
# finder), the compiled Q/L config identical after path normalisation, the pinned libraries
# unchanged during the arm.
#
# Env: JOBS (concurrent groups, default 8; each wire-cell is multi-threaded, CLAUDE.md M5)
#      PIN  (lib dir prepended to LD_LIBRARY_PATH; default ~/tmp/d123-libpin; '' = installed local/lib)
#      QLTLA (a file of --tla-code lines appended to the Q/L job, run_chain_group.sh QL_EXTRA_TLA)
set -uo pipefail
SX=$(cd "$(dirname "$0")/../.." && pwd -P)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
TK=$WCT_BASE/toolkit
AB=$SX/../../abtest
SBND_RECO1=${SBND_RECO1:-${WCT_BASE}/wire-cell-sbnd-reco1/install}
export WIRECELL_PATH=$TK/cfg:$WCT_BASE/wire-cell-data:$WCT_BASE/wire-cell-data/sbnd/photodet:${SBND_RECO1}/share/wirecell
PIN=${PIN-$HOME/tmp/d123-libpin}
export LD_LIBRARY_PATH=${PIN:+$PIN:}${PIN:+$PIN/reco1/lib:}${SBND_RECO1}/lib:${LD_LIBRARY_PATH:-}
JOBS=${JOBS:-8}

[ $# -ge 3 ] || { sed -n '2,24p' "$0"; exit 1; }
BASE=$(readlink -f "$1"); OUT=$2; REALITY=$3; shift 3
case "$REALITY" in data|sim) ;; *) echo "ERROR: reality must be data|sim" >&2; exit 1;; esac
FF=''; GLIST=all; REF=0; HIT_TIME=''; RUN_QL=1
while [ $# -gt 0 ]; do
    case "$1" in
        --ff) FF=$2; shift 2;;
        --groups) GLIST=$2; shift 2;;
        --ref) REF=1; shift;;
        --hit-time) HIT_TIME=$2; shift 2;;
        --no-ql) RUN_QL=0; shift;;
        *) echo "ERROR: unknown option $1" >&2; exit 1;;
    esac
done
[ -d "$BASE" ] || { echo "ERROR: no baseline arm $BASE" >&2; exit 1; }
case "$OUT" in /*) ;; *) OUT=$SX/$OUT;; esac
if [ -e "$OUT" ] && [ ! -f "$OUT/.d123_hits_arm" ]; then
    echo "ERROR: refusing to touch existing $OUT (M13)" >&2; exit 1
fi
mkdir -p "$OUT"; touch "$OUT/.chain_group" "$OUT/.d123_hits_arm"; echo "$REALITY" > "$OUT/.lineage_reality"
echo "$BASE" > "$OUT/.d123_base_root"
if [ "$GLIST" = all ]; then
    GLIST=$(ls -d "$BASE"/g[0-9]* | sed 's#.*/g##' | sort -n | paste -sd,)
fi
LIBS="$TK/../local/lib/libWireCellClus.so $TK/../local/lib/libWireCellMatch.so $TK/../local/lib/libWireCellFlash.so ${SBND_RECO1}/lib/libWireCellSBNDReco1.so"
[ -n "$PIN" ] && LIBS="$PIN/libWireCellClus.so $PIN/libWireCellMatch.so $PIN/libWireCellFlash.so $PIN/reco1/lib/libWireCellSBNDReco1.so"
md5sum $LIBS > "$OUT/.libs.md5.start"
echo "hits arm $OUT  base=$BASE  reality=$REALITY  groups=$GLIST  ff=${FF:-none}  ref=$REF  jobs=$JOBS  pin=${PIN:-none}"

# ---- 1-3. the flash dump per group, JOBS at a time ----
dump_group() {
    local K=$1 g="$OUT/g$K" b="$BASE/g$K"
    [ -s "$b/events.txt" ] || { echo "[g$K] baseline has no events.txt" >&2; return 1; }
    [ -s "$b/.wct-cfg-dump.json" ] || { echo "[g$K] baseline has no .wct-cfg-dump.json" >&2; return 1; }
    if [ -s "$g/opflash_apa0.tar.gz" ] && [ -s "$g/opflash_apa1.tar.gz" ]; then
        echo "[g$K] flash dump exists -- skipped"
    else
        mkdir -p "$g"
        # the baseline's dump parameters, from its compiled config
        local P; P=$(python3 - "$b/.wct-cfg-dump.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
fs = [n for n in c if n.get('type') == 'SBNDReco1OpFlashSource' and n.get('name') == 'tpc0'][0]['data']
print(fs['filename'], fs.get('entry_begin', 0), fs.get('entry_count', -1),
      fs.get('caf_offset_mode', 'none'), fs.get('frameshift_product', ''))
EOF
        ) || return 1
        local FILE BEG CNT CAF FSP; read -r FILE BEG CNT CAF FSP <<< "$P"
        case "$FILE" in /*) ;; *) FILE=$SX/$FILE;; esac
        [ -r "$FILE" ] || { echo "[g$K] reco1 file not readable: $FILE" >&2; return 1; }
        local -a TLA=(--tla-str "input=$FILE" --tla-str "output_dir=$g"
                      --tla-str "caf_offset_mode=$CAF" --tla-str "caf_offset_override=0")
        [ -n "$FSP" ] && TLA+=(--tla-str "frameshift_product=$FSP")
        TLA+=(--tla-str "entry=-1" --tla-str "entry_begin=$BEG" --tla-str "entry_count=$CNT"
              --tla-str "flash_source=hits" --tla-str "with_frames=false")
        [ -n "$HIT_TIME" ] && TLA+=(--tla-str "hit_time=$HIT_TIME")
        [ "$REF" = 1 ] && TLA+=(--tla-str "reco1_reference=true")
        [ -n "$FF" ] && TLA+=(--tla-code "ff=$FF")
        wcsonnet "${TLA[@]}" -o "$g/.wct-cfg-dump.json" "$SX/wct-reco1-dump.jsonnet" > "$g/.wct-cfg-dump.json.log" 2>&1 \
            || { echo "[g$K] wcsonnet FAILED (see $g/.wct-cfg-dump.json.log)" >&2; return 1; }
        ( cd "$SX" && wire-cell -l stderr -l "$g/wct_dump.log:info" -L info -c "$g/.wct-cfg-dump.json" > "$g/dump.stdout" 2>&1 ) \
            || { rm -f "$g"/opflash_apa*.tar.gz; echo "[g$K] hit-flash dump FAILED (see $g/wct_dump.log)" >&2; return 1; }
    fi
    # 2. the event-id set must be the baseline's
    local ids; ids=$(tar tzf "$g/opflash_apa0.tar.gz" | sed -n 's/^opflash_tensorset_\([0-9]*\)_metadata\.json$/\1/p' | sort -u)
    if [ "$ids" != "$(sort -u "$b/events.txt")" ]; then
        echo "[g$K] event-id set differs from the baseline events.txt" >&2; return 1
    fi
    # 3. shared inputs: links for what the runner only READS, a copy for what it rewrites
    local f
    for f in frames-dnn.tar.bz2 icluster-apa0-active.npz icluster-apa0-masked.npz icluster-apa1-active.npz icluster-apa1-masked.npz; do
        [ -e "$b/$f" ] || { echo "[g$K] baseline lacks $f (imaging not kept?)" >&2; return 1; }
        [ -e "$g/$f" ] || ln -s "$b/$f" "$g/$f"
    done
    [ -e "$g/events.txt" ] || cp "$b/events.txt" "$g/events.txt"
    echo "[g$K] flash dump ok ($(echo "$ids" | wc -l) events)"
}
fail=0; n=0
for K in ${GLIST//,/ }; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || fail=1; done
    dump_group "$K" > "$OUT/.dump_g$K.log" 2>&1 &
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n 2>/dev/null || fail=1; done
cat "$OUT"/.dump_g*.log
[ "$fail" = 0 ] || { echo "ERROR: a flash dump failed -- not running Q/L" >&2; exit 1; }
[ "$RUN_QL" = 1 ] || { echo "dumps done (--no-ql)"; exit 0; }

# ---- 4. the Q/L job, unchanged, on the shared imaging ----
export SBND_QL_KEEP_ICLUSTER=1
export SBND_MAX_JOBS=$JOBS
[ -n "${QLTLA:-}" ] && export QL_EXTRA_TLA=$(readlink -f "$QLTLA")
FILE0=$(python3 -c "import json,sys; c=json.load(open(sys.argv[1])); print([n for n in c if n.get('type')=='SBNDReco1OpHitSource'][0]['data']['filename'])" "$OUT/g${GLIST%%,*}/.wct-cfg-dump.json")
cd "$SX"
./run_chain_group.sh "$FILE0" "$OUT" "$REALITY" --size 16 --layout perevt --groups "$GLIST" --from ql > "$OUT/.run.log" 2>&1
rc=$?
md5sum $LIBS > "$OUT/.libs.md5.end"
cmp -s "$OUT/.libs.md5.start" "$OUT/.libs.md5.end" || echo "WARNING: a library changed during the arm" >&2

# ---- post-checks ----
bad=0
for K in ${GLIST//,/ }; do
    g="$OUT/g$K"; b="$BASE/g$K"
    cmp -s "$g/rse.json" "$b/rse.json" || { echo "[g$K] rse.json differs from the baseline" >&2; bad=1; }
    if [ -s "$g/.wct-cfg-ql.json" ] && [ -s "$b/.wct-cfg-ql.json" ]; then
        # a root may appear absolute or relative to sbnd_xin/ (run_chain_group.sh keeps the form it was given)
        if ! diff -q <(sed "s#$OUT#ROOT#g; s#$(basename "$OUT")#ROOT#g" "$g/.wct-cfg-ql.json") \
                     <(sed "s#$BASE#ROOT#g; s#$(basename "$BASE")#ROOT#g" "$b/.wct-cfg-ql.json") > /dev/null; then
            [ -n "${QLTLA:-}" ] || { echo "[g$K] compiled Q/L config differs from the baseline beyond the root path" >&2; bad=1; }
        fi
    fi
done
echo "rc=$rc postchecks_bad=$bad" | tee -a "$OUT/.run.log"
[ "$rc" = 0 ] && [ "$bad" = 0 ]
