#!/bin/bash
# doc pr/11: batch driver for the FULL 13-stage SBND PR chain -- the one that
# is missing today.  Every existing runner (run_nusel_evt.sh, run_pr_evt.sh,
# run_full1k_nusel.sh) stops at tagger_check_fc; this one appends
# tagger_check_neutrino -> numu_bdt_scorer -> nue_bdt_scorer -> tracking_visitor
# -> tagger_output, so it is the first driver that can produce numu_score /
# nue_score / T_kine at population scale.
#
# Runs on an EXISTING Q/L pctree root (never re-runs Q/L or imaging -- CLAUDE.md
# M11): fork-by-duplication of run_nusel_evt.sh's process_event() (M10 -- that
# production script stays byte-untouched), with:
#   - the DL (SCN) neutrino vertex ON (the production default since e3d46c91;
#     no dl_weights override, unlike run_nusel_evt.sh which always forces
#     geometric).  Needs libpython RTLD_GLOBAL preloaded or the SCN import
#     fails and the DL vertex SILENTLY falls back to geometric (a WARN
#     "DL vertex failed: ..." is the only sign -- checked per event, see below).
#   - OMP_NUM_THREADS=MKL_NUM_THREADS=1: the DL inference is multithreaded: a
#     30-event probe showed 5.7s wall / 5.6s core pinned vs 5.3s wall / 9.5s
#     core unpinned on the same event.  Pinned so wall time is a comparable
#     single-thread latency number across every event and arm.
#   - -stm-fit (stm_magnify / tracking-stm.root) is OMITTED: doc pr/3 confirms
#     save_stm_fit only gates a diagnostic dump, never the STM verdict itself,
#     and skipping it avoids a second UPDATE-mode ROOT writer per event.
#   - RSE (run/subrun/event) is read from the Q/L job's own opflash metadata
#     (opflash_tensorset_<EVT>_metadata.json), exactly like run_nusel_evt.sh --
#     no external per-sample RSE file needed, and it is correct even for
#     samples spanning multiple runs (nueCC48) or reusing event numbers across
#     samples (MC evt 12 appears in both round1-qlmatch and round2-patrec).
#
# Usage:
#   ./run_pr_chain_batch.sh <ql_root> <out_root> <data|sim> [evt ...]
#     ql_root   dir containing ql_evt<ID>/{pctree-evt<ID>.tar.gz,
#               opflash_apa0.tar.gz, mabc-all-apa.zip} -- e.g. work-mcp1kall-d59k
#     out_root  FRESH dir for pr_evt<ID>/ outputs (refuses to reuse a non-empty
#               one that was not created by a prior run of this script -- M13)
#     data|sim  reality TLA
#     evt ...   optional explicit event-id subset; default = every ql_evt<ID>
#               found under ql_root
#
# Env: PR_JOBS (default 6, M5), SBND_WCT_LOGLEVEL (default debug -- needed for
#      the MABC timing / TaggerCheckNeutrino timing substage lines, doc pr/11
#      sec 3/5; perf=true is already on for SBND).
#
# Per-event output: out_root/pr_evt<ID>/{wct_pr_evt<ID>.log, stdout.log, rc.txt,
#   .time.meta (timecmd.py), mabc-pr.zip, tracking-pr.root,
#   pctree-pr-evt<ID>.tar.gz, nusel-evt<ID>.tsv}.
# Batch: out_root/nusel-table.tsv + nusel-events.tsv (merged, same shape as
#   run_nusel_evt.sh all).  Read the SCORES with pr_scores_table.py.
set -u

SX=$(cd "$(dirname "$0")" && pwd -P)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
TK=$WCT_BASE/toolkit
export WIRECELL_PATH=$TK/cfg:$WCT_BASE/wire-cell-data:$WCT_BASE/wire-cell-data/sbnd/photodet:${WIRECELL_PATH:-}
export PYTHONPATH=$TK/pyutil/python:$WCT_BASE/local/python:$WCT_BASE/wire-cell-python:${PYTHONPATH:-}
AB=$SX/../../abtest

SBND_DIR=$SX  # required by _runlib.sh (unused otherwise here: QLROOT/OUTROOT
              # are explicit args, not SBND_WORK_ROOT-derived).
. "$SX/_runlib.sh"

usage() {
    cat <<EOF
Usage: $0 <ql_root> <out_root> <data|sim> [evt ...]
  ql_root   dir with ql_evt<ID>/{pctree,opflash,mabc-all-apa.zip}
  out_root  fresh output dir (pr_evt<ID>/ per event)
  data|sim  reality TLA
  evt ...   optional event-id subset (default: every ql_evt<ID> in ql_root)
Env: PR_JOBS (default 6), SBND_WCT_LOGLEVEL (default debug)
EOF
}

[ $# -ge 3 ] || { usage; exit 1; }
QLROOT=$(cd "$1" 2>/dev/null && pwd -P) || { echo "ERROR: no such ql_root: $1" >&2; exit 1; }
OUTROOT=$2
REALITY=$3
shift 3

case "$REALITY" in
    data|sim) ;;
    *) echo "ERROR: reality must be data|sim, got '$REALITY'" >&2; exit 1 ;;
esac

mkdir -p "$OUTROOT"
OUTROOT=$(cd "$OUTROOT" && pwd -P)

if [ $# -ge 1 ]; then
    EVENT_IDS=("$@")
else
    mapfile -t EVENT_IDS < <(ls -d "$QLROOT"/ql_evt*/ 2>/dev/null | sed -E 's#.*/ql_evt([0-9]+)/?$#\1#' | sort -n)
fi
[ "${#EVENT_IDS[@]}" -gt 0 ] || { echo "ERROR: no events found under $QLROOT" >&2; exit 1; }

JSONNET="$SX/wct-pr-perevt.jsonnet"
[ -f "$JSONNET" ] || { echo "ERROR: missing jsonnet: $JSONNET" >&2; exit 1; }

# doc 68: the LAr set and the TrackFitting parameter file come from the job's
# own defaults (cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet); only an
# SBND_TRACKFIT_JSON override is named, for the doc-66 diffusion A/B.
TFJSON="${SBND_TRACKFIT_JSON:-}"
TFJSON_TLA=()
[ -n "$TFJSON" ] && TFJSON_TLA=(--tla-str "trackfitting_config=$TFJSON")

# Production NUF pipeline + the 5 neutrino-PR stages (doc pr/2-3; ordering
# matters -- BDTs after tagger_check_neutrino, nue after numu, tagger_output
# after tracking_visitor because it opens tracking-pr.root in UPDATE mode).
PIPELINE="switch_scope,unmerge_bundle,unmerge_assoc,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_fc,tagger_check_neutrino,numu_bdt_scorer,nue_bdt_scorer,tracking_visitor,tagger_output"

# The embedded interpreter needs libpython loaded RTLD_GLOBAL for the SCN
# (DL vertex) import to succeed -- same idiom as run_pr3_evt_dl.sh / M4.
PYLIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.11.so.1.0
[ -r "$PYLIB" ] || { echo "ERROR: libpython not found: $PYLIB" >&2; exit 1; }

process_event() {
    local EVT_ID=$1
    local QLDIR="$QLROOT/ql_evt${EVT_ID}"
    local PCT="$QLDIR/pctree-evt${EVT_ID}.tar.gz"
    local PRDIR="$OUTROOT/pr_evt${EVT_ID}"
    local LOG="$PRDIR/wct_pr_evt${EVT_ID}.log"

    [ -s "$PCT" ] || { echo "ERROR: [evt $EVT_ID] no pctree: $PCT" >&2; return 1; }

    rm -rf "$PRDIR"; mkdir -p "$PRDIR"

    # RSE from the Q/L job's own opflash metadata (same source run_nusel_evt.sh
    # uses; correct across every sample this driver targets -- verified for
    # data/MCP2025C, nueCC48, and both MC roots, doc pr/11 sec 1).
    local RUN_NO=0 SUBRUN_NO=0 _md
    _md=$(tar xzOf "$QLDIR/opflash_apa0.tar.gz" "opflash_tensorset_${EVT_ID}_metadata.json" 2>/dev/null) || _md=''
    if [ -n "$_md" ]; then
        local _rse
        _rse=$(printf '%s' "$_md" | python3 -c \
            'import json,sys; d=json.load(sys.stdin); print(int(d.get("run",0)), int(d.get("subrun",0)))' \
            2>/dev/null) && [ -n "$_rse" ] && read -r RUN_NO SUBRUN_NO <<< "$_rse"
    fi

    echo "[evt $EVT_ID] rse=($RUN_NO, $SUBRUN_NO, $EVT_ID) pipeline=($PIPELINE) reality=$REALITY dl=on"

    (
        cd "$PRDIR" || exit 1
        export LD_PRELOAD="$PYLIB"
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
        # PR_TIMEOUT (seconds, default 3600) bounds a single event.  A pattern-
        # recognition hang is real: SBND MCP2025C evt 352365 spun at 100% CPU with
        # byte-flat RSS for 8h17m in shower_clustering_with_nv_from_vertices and
        # stalled the whole 1000-event batch on its last slot (doc pr/11 sec 6).
        # `timeout` sits INSIDE timecmd.py so .time.meta is still written (rc=124).
        setarch x86_64 -R python3 "$AB/timecmd.py" "$PRDIR/.time.meta" \
        timeout --signal=TERM --kill-after=60 "${PR_TIMEOUT:-3600}" \
        wire-cell \
            -l stderr -l "${LOG}:${SBND_WCT_LOGLEVEL:-debug}" -L "${SBND_WCT_LOGLEVEL:-debug}" \
            --tla-str  "input=$PCT" \
            --tla-code "anode_indices=[0,1]" \
            --tla-str  "output_dir=$PRDIR" \
            --tla-code "run=${RUN_NO}" --tla-code "subrun=${SUBRUN_NO}" --tla-code "event=${EVT_ID}" \
            --tla-str  "reality=$REALITY" \
            `# doc 68: the LAr set, the beam window and every tgm_*/stm_* knob` \
            `# spelled out here were byte-for-byte the job's own defaults, so` \
            `# they are gone.  PIPELINE stays explicit -- this chain adds the` \
            `# neutrino taggers + BDT scorers on top of the default list.` \
            --tla-code "pipeline_names=[$(echo "$PIPELINE" | sed "s/[^,]\+/'&'/g")]" \
            "${TFJSON_TLA[@]}" \
            --tla-str  "save_tensors=$PRDIR/pctree-pr-evt${EVT_ID}.tar.gz" \
            -c "$JSONNET"
        echo "rc=$?" > "$PRDIR/rc.txt"
    ) > "$PRDIR/stdout.log" 2>&1
    rm -f "$PRDIR/trash-pr.tar.gz"

    local rc; rc=$(sed -n 's/^rc=//p' "$PRDIR/rc.txt" 2>/dev/null); rc=${rc:-1}

    # DL-engagement proof (WARN visible at debug level -- doc pr/11 sec 1):
    # absent = DL ran without a hard failure this event (it may still have
    # legitimately deferred to the traditional vertex on a low rerank score).
    if grep -q "DL vertex failed" "$LOG" 2>/dev/null; then
        echo "[evt $EVT_ID] WARN: DL vertex failed this event (see $LOG)" >&2
    fi

    # Only extract when wire-cell actually succeeded.  On a crash the zip/prtree
    # are zero-byte and nusel_extract.py dies with a BadZipFile traceback that
    # buries the real cause at the tail of stdout.log -- exactly what happened to
    # the 73 doc-pr/11 failures, where SIGABRT/SIGTERM all read as "rc=250
    # BadZipFile".  rc here is wire-cell's own (timecmd.py encodes a fatal signal
    # N as 256-N: 250 = SIGABRT, 241 = SIGTERM, 124 = PR_TIMEOUT).
    if [ "$rc" != 0 ]; then
        echo "[evt $EVT_ID] wire-cell rc=$rc -- skipping nusel_extract (no usable outputs)" \
            >> "$PRDIR/stdout.log"
        echo "[evt $EVT_ID] rc=$rc  -> $PRDIR"
        return 1
    fi

    # Per-bundle label table -- same nusel_extract.py production call
    # (unmodified script; --prtree uses the save_tensors dump above, so labels
    # come from the authoritative flag_TGM/STM/FC, not the log fallback).
    python3 "$SX/nusel_extract.py" \
        --pctree "$PCT" --prbee "$PRDIR/mabc-pr.zip" --prlog "$LOG" \
        --prtree "$PRDIR/pctree-pr-evt${EVT_ID}.tar.gz" \
        --qlbee "$QLDIR/mabc-all-apa.zip" \
        --beam-window "0.2,2.2" \
        --run "$RUN_NO" --subrun "$SUBRUN_NO" \
        --out "$PRDIR/nusel-evt${EVT_ID}.tsv" 2>>"$PRDIR/stdout.log"

    echo "[evt $EVT_ID] rc=$rc  -> $PRDIR"
    [ "$rc" = 0 ]
}

batch_init
BATCH_MAX=${PR_JOBS:-6}
echo "ql_root=$QLROOT out_root=$OUTROOT reality=$REALITY events=${#EVENT_IDS[@]} jobs=$BATCH_MAX"
for evt in "${EVENT_IDS[@]}"; do
    _blog="$OUTROOT/.batch_pr_evt${evt}.log"
    batch_wait_slot
    ( process_event "$evt" ) > "$_blog" 2>&1 &
    BATCH_PIDS[$!]=$evt
    echo "  [start] evt=$evt  log: $_blog"
done
batch_drain
batch_summary

# Merge the per-event nusel tables (label/TGM/STM/FC/LM census).
_tsvs=()
for evt in "${EVENT_IDS[@]}"; do
    _t="$OUTROOT/pr_evt${evt}/nusel-evt${evt}.tsv"
    [ -s "$_t" ] && _tsvs+=("$_t")
done
if [ "${#_tsvs[@]}" -gt 0 ]; then
    python3 "$SX/nusel_extract.py" --merge "${_tsvs[@]}" \
        --out "$OUTROOT/nusel-table.tsv" \
        --events-out "$OUTROOT/nusel-events.tsv"
    echo "merged -> $OUTROOT/nusel-table.tsv + nusel-events.tsv"
fi

echo "loadavg: $(cat /proc/loadavg)"
