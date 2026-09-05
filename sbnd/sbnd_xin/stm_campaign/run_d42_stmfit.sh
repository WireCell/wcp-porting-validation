#!/usr/bin/env bash
# doc pdvd/42: the SBND reference arm for the STM trajectory + dQ/dx fit
# validation -- tracking-stm.root (T_rec_charge / T_stm_pass / T_stm_eval /
# T_proj_data) at TODAY's SBND production operating point, over the 72
# owner-adjudicated STM-baseline events (doc 62, scan-d59k/stm-baseline.tsv)
# plus the 30 doc-55 dQ/dx-sample events (dqdx_rr_sample/d55ton_events.txt):
# 99 distinct data events.
#
# Fork by duplication of run_round.sh (doc 63), with three deliberate changes:
#   * pctree/imaging source is work-mcp1k-d97fv (work-mcp1kall-d59k was retired
#     2026-09-04, doc 100); every input is symlinked read-only (M13);
#   * NO tagger flags: since doc 68 the production operating point lives in
#     cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet, and a bare
#     run_nusel_evt.sh IS production.  Only -stm-fit is added (the dump; doc 41
#     shows it is verdict-neutral);
#   * the binary is pinned through LD_LIBRARY_PATH (env D42_LIBPIN), because a
#     peer's wcbuild can swap local/lib mid-arm.
#
# Usage: ./stm_campaign/run_d42_stmfit.sh <root-label> [extra run_nusel_evt.sh flags...]
#   root-label   e.g. d42fit -> work root work-stmcamp-<label>  (REFUSED if it exists)
# Env: STM_EVENTS  space-separated event list override (default: the 99 above)
#      NJOBS       parallel jobs (default 16; owner licence 2026-09-03 is ~32 wire-cell processes)
#      D42_LIBPIN  directory of pinned libWireCell*.so (+ *.pcm) -> LD_LIBRARY_PATH
#      SRC_ROOT    pctree/imaging source (default work-mcp1k-d97fv)
#      D42_NO_STMFIT=1  drop -stm-fit (bare production chain; the old-vs-new binary identity gate)
set -u
cd "$(dirname "$0")/.."
SBND_DIR=$PWD
LABEL=${1:?usage: run_d42_stmfit.sh <root-label> [flags...]}; shift
ROOT=$SBND_DIR/work-stmcamp-$LABEL
SRC=$SBND_DIR/${SRC_ROOT:-work-mcp1k-d97fv}
STAGE=$SBND_DIR/input_files_reco1/staged-mcp2025c-1000evt
MAP=$STAGE/entry_event_map.tsv
NJOBS=${NJOBS:-16}
FLAGS="-stm-fit $*"; [ "${D42_NO_STMFIT:-0}" = 1 ] && FLAGS="$*"   # D42_NO_STMFIT=1: the bare production chain (identity gate arms)

if [ -e "$ROOT" ]; then
    echo "REFUSING: $ROOT exists (new arm => new label, M13)" >&2
    exit 1
fi
[ -d "$SRC" ] || { echo "ERROR: source root $SRC missing" >&2; exit 1; }
mkdir -p "$ROOT/.status"

if [ -z "${STM_EVENTS:-}" ]; then
    STM_EVENTS=$(cat <(grep -v '^#' "$SBND_DIR/scan-d59k/stm-baseline.tsv" | awk -F'\t' 'NR>1{print $2}') \
                     <(grep -v '^#' "$SBND_DIR/dqdx_rr_sample/d55ton_events.txt") | sort -un)
fi

entry_of() { awk -F'\t' -v e="$1" '$4==e {print $1; exit}' "$MAP"; }

run_one() {
    local evt=$1
    local entry; entry=$(entry_of "$evt")
    if [ -z "$entry" ]; then echo "rc=90 evt=$evt no-entry" > "$ROOT/.status/$evt"; return; fi
    if [ ! -d "$SRC/evt$evt" ] || [ ! -s "$SRC/ql_evt$evt/pctree-evt$evt.tar.gz" ]; then
        echo "rc=91 evt=$evt missing-source-input" > "$ROOT/.status/$evt"; return
    fi
    ln -sfn "$(readlink -f "$SRC/evt$evt")" "$ROOT/evt$evt"
    ln -sfn "$SRC/ql_evt$evt" "$ROOT/ql_evt$evt"
    local cwd=$ROOT/.cwd/$evt
    mkdir -p "$cwd"
    (
        cd "$cwd" || exit 92
        if [ -n "${D42_LIBPIN:-}" ]; then export LD_LIBRARY_PATH="$D42_LIBPIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
        SBND_INPUT_DIR=$STAGE/e$entry SBND_WORK_ROOT=$ROOT \
            setarch x86_64 -R "$SBND_DIR/run_nusel_evt.sh" data $FLAGS 1
    ) > "$ROOT/.log_$evt.log" 2>&1
    echo "rc=$? evt=$evt" > "$ROOT/.status/$evt"
}

{
    echo "label $LABEL: $(echo "$STM_EVENTS" | wc -w) events, flags: $FLAGS, src $SRC, libpin ${D42_LIBPIN:-<none>}"
    echo "toolkit $(git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse --short HEAD)  wcp-porting-img $(git -C "$SBND_DIR" rev-parse --short HEAD)  $(date -Is)"
    [ -n "${D42_LIBPIN:-}" ] && md5sum "$D42_LIBPIN"/libWireCellClus.so "$D42_LIBPIN"/libWireCellRoot.so
} | tee "$ROOT/.arm_info.txt"
n=0
for evt in $STM_EVENTS; do
    run_one "$evt" &
    n=$((n+1))
    if [ $((n % NJOBS)) -eq 0 ]; then wait; fi
done
wait

ok=$(grep -l '^rc=0' "$ROOT"/.status/* 2>/dev/null | wc -l)
tot=$(ls "$ROOT"/.status/ | wc -l)
roots=$(ls "$ROOT"/nusel_evt*/tracking-stm.root 2>/dev/null | wc -l)
echo "label $LABEL done: $ok/$tot rc=0, $roots tracking-stm.root; loadavg $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$ROOT/.arm_info.txt"
grep -L '^rc=0' "$ROOT"/.status/* 2>/dev/null | while read -r f; do cat "$f"; done
