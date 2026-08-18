#!/usr/bin/env bash
# Q/L-ONLY batch over an explicit EVENT-ID list, across the mcp1k and mcp2k
# staged samples.  docs/73 round 2: the cathode bundle rescue runs inside the
# Q/L job, and the whole doc-72 search reads only ql_evt<ID>/mabc-all-apa.zip,
# so an A/B on this component needs no PR chain at all.  Written for that round
# (owner: "evaluate based on QL match result, no need to run PR chain").
#
# Differences from run_full1k_nusel{,_2k}.sh, which this is modelled on:
#   * takes EVENT IDS, not entry numbers, and resolves the entry (and hence the
#     staged input dir) from each sample's entry_event_map.tsv -- the doc-72
#     products are keyed by event;
#   * spans BOTH samples in one call, picking the sample that owns each event;
#   * stops after run_ql_evt.sh.  No nusel, no taggers, no nusel-table.tsv.
#
# Imaging is SYMLINKED from IMGBASE (M11: never regenerated, never borrowed);
# ROOT is always a fresh root (M13: never write into an existing arm).
#
# Usage:
#   ROOT=$PWD/work-cbr2-on ./run_ql_batch.sh [-j N] <evt> [<evt> ...]
#   ROOT=... ./run_ql_batch.sh -j 6 -f events.txt
# Env:
#   ROOT          output root (REQUIRED)
#   IMG1K/IMG2K   imaging roots (default work-img-mcp1k / work-img-mcp2k)
#   any SBND_*    passed through to run_ql_evt.sh (SBND_RESCUE_* etc.)
#   QL_EXTRA      extra flags appended to the run_ql_evt.sh command line,
#                 word-split on purpose.  The one that matters here is
#                 -save-pctree: docs/73 round 2 was Q/L-only and never wrote
#                 pctree-evt<ID>.tar.gz, which is the ONLY file run_pr_chain_batch.sh
#                 needs beyond what this script already produces (verified by
#                 diffing an arm against the known-good work-mcp1k-cb0805).
#   CATHODE_RESCUE_DEBUG=1  [cbrsel]/[cbrx] tracer -> $ROOT/.log_e<entry>.log
# Internal:
#   ./run_ql_batch.sh --worker <sample> <entry> <evt>
set -u
cd "$(dirname "$0")"
SBND_DIR=$PWD
IMG1K=${IMG1K:-$SBND_DIR/work-img-mcp1k}
IMG2K=${IMG2K:-$SBND_DIR/work-img-mcp2k}
S1K=$SBND_DIR/input_files_reco1/staged-mcp2025c-1000evt
S2K=$SBND_DIR/input_files_reco1/staged-mcp2025c-2nd-2000evt
TC=$(cd ../../abtest && pwd)/timecmd.py

# --- one event -------------------------------------------------------------
if [ "${1:-}" = "--worker" ]; then
    sample=$2; i=$3; evt=$4
    case $sample in
        1k) STAGE=$S1K; IMGBASE=$IMG1K ;;
        2k) STAGE=$S2K; IMGBASE=$IMG2K ;;
    esac
    st=$ROOT/.status/$sample-$i
    if [ ! -d "$IMGBASE/evt$evt" ]; then
        echo "rc=91 sample=$sample entry=$i evt=$evt no-imaging" > "$st"; exit 0
    fi
    ln -sfn "$IMGBASE/evt$evt" "$ROOT/evt$evt"
    cwd=$ROOT/.cwd/$sample-$i
    mkdir -p "$cwd"
    (
        cd "$cwd" || exit 92
        SBND_INPUT_DIR=$STAGE/e$i SBND_WORK_ROOT=$ROOT \
            setarch x86_64 -R python3 "$TC" "$ROOT/.time_e$i.meta" \
            "$SBND_DIR/run_ql_evt.sh" data 1 ${QL_EXTRA:-}
    ) > "$ROOT/.log_e$i.log" 2>&1
    rc=$?
    # The rescue announces every move it makes; record the count so a firing
    # census never has to re-open the logs.  Keep the 'rescue round' substring
    # -- scripts/analysis/cathode/cbr_sweep_compare.py greps for exactly that.
    nfire=$(grep -ac "rescue round" "$ROOT/ql_evt$evt/wct_ql_evt$evt.log" 2>/dev/null); : "${nfire:=0}"
    printf 'rc=%s sample=%s entry=%s evt=%s %s fired=%s\n' "$rc" "$sample" "$i" "$evt" \
        "$(tr '\n' ' ' < "$ROOT/.time_e$i.meta" 2>/dev/null)" "$nfire" > "$st"
    rmdir "$cwd" 2>/dev/null
    exit 0
fi

# --- batch driver ----------------------------------------------------------
J=6
evts=()
while [ $# -gt 0 ]; do
    case "$1" in
        -j) J=$2; shift 2 ;;
        -f) while read -r _e; do [ -n "$_e" ] && evts+=("$_e"); done < "$2"; shift 2 ;;
        *)  evts+=("$1"); shift ;;
    esac
done
[ -n "${ROOT:-}" ] || { echo "REFUSE: set ROOT explicitly (M13)" >&2; exit 2; }
[ ${#evts[@]} -gt 0 ] || { echo "no events given" >&2; exit 2; }
mkdir -p "$ROOT/.status" "$ROOT/.cwd"

# event -> (sample, entry), from each sample's own map.  An event present in
# neither map is reported, not silently skipped.
resolve() {
    local e=$1 i
    i=$(awk -F'\t' -v e="$e" '$4==e {print $1; exit}' "$S1K/entry_event_map.tsv")
    if [ -n "$i" ]; then echo "1k $i"; return; fi
    i=$(awk -F'\t' -v e="$e" '$4==e {print $1; exit}' "$S2K/entry_event_map.tsv")
    if [ -n "$i" ]; then echo "2k $i"; return; fi
    echo ""
}

echo "=== docs/73 Q/L-only batch ==="
echo "  root=$ROOT  jobs=$J  events=${#evts[@]}"
echo "  toolkit HEAD=$(git -C "$SBND_DIR/../../../toolkit" rev-parse --short HEAD 2>/dev/null)"
echo "  libWireCellClus.so mtime=$(stat -c %y /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCellClus.so 2>/dev/null)"
env | grep -E '^(SBND_RESCUE|SBND_CATHODE|CATHODE_RESCUE)' | sed 's/^/  /'
echo "  started $(date -Is)"
t0=$(date +%s)

work=()
for e in "${evts[@]}"; do
    r=$(resolve "$e")
    if [ -z "$r" ]; then echo "  WARN evt $e in neither sample map -- skipped"; continue; fi
    work+=("$r $e")
done
[ ${#work[@]} -gt 0 ] || { echo "  no resolvable events -- nothing to do"; exit 3; }
printf '%s\n' "${work[@]}" | \
    xargs -P "$J" -n 3 bash -c 'exec "$0" --worker "$1" "$2" "$3"' "$SBND_DIR/run_ql_batch.sh"

echo "  finished $(date -Is) ($(( $(date +%s) - t0 ))s)"
nok=$(cat "$ROOT"/.status/* 2>/dev/null | grep -c '^rc=0 ')
echo "  rc=0 on $nok/${#work[@]}"
grep -h . "$ROOT"/.status/* 2>/dev/null | grep -v 'fired=0' | sed 's/^/  FIRED /'
