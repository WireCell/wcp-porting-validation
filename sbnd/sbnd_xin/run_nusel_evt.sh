#!/bin/bash
# Per-event SBND neutrino-selection chain — standalone, self-contained.  -h for help.
# Usage: ./run_nusel_evt.sh [mc|data] [-N n] [-bw l,h] <idx|all>
#        ./run_nusel_evt.sh [mc|data] [-N n]           # list available events
#
# Chain (see docs/23_nusel-tgm-stm-chain.md):
#   1. Q/L matching with the persisted point-cloud tree
#      (./run_ql_evt.sh <mode> -save-pctree <idx>; skipped when
#      work/ql_evt<ID>/pctree-evt<ID>.tar.gz already exists);
#   2. the PR tagger job on the loaded tree:
#      switch_scope -> steiner -> fiducialutils -> tagger_check_tgm ->
#      tagger_check_stm -> tagger_check_fc
#      (wct-pr-perevt.jsonnet, beam window on cluster_t0;
#       check_neutrino_candidate ON by default -- see -no-nucand);
#   3. the per-bundle label table (nusel_extract.py): one row per matched
#      bundle (= main cluster) with flash time / PE, bundle size, TGM/STM/FC
#      verdicts and a label, plus a row per beam-window flash that matched
#      no bundle.
#
# Honors SBND_WORK_ROOT / SBND_INPUT_DIR / SBND_SAMPLE like run_ql_evt.sh, so
# it runs directly on reprocessing trees, e.g. the MCP2025C reco1 sample:
#   SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
#   SBND_WORK_ROOT=$PWD/work-mcp10 ./run_nusel_evt.sh data all
#
# Prerequisite: ./run_img_evt.sh <mode> <idx> (per-event imaging npz).
# Output: work/nusel_evt<ID>/{wct_nusel_evt<ID>.log, mabc-pr.zip, nusel-evt<ID>.tsv}
#         'all' also merges: work/nusel-table.tsv (per bundle) and
#         work/nusel-events.tsv (per event).

set -e

# pwd -P: sbnd_xin is reachable through a symlink (toolkit/sbnd_xin); the
# PR jsonnet's relative import '../particle_dataset.jsonnet' only resolves
# from the REAL location (wcp-porting-img/sbnd/), so canonicalize here.
SBND_DIR=$(cd "$(dirname "$0")" && pwd -P)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WCT_BASE}/wire-cell-data/sbnd/photodet:${WIRECELL_PATH}

. "$SBND_DIR/_runlib.sh"

JSONNET="$SBND_DIR/wct-pr-perevt.jsonnet"
# Same LAr TLAs as run_ql_evt.sh / run_pr_evt.sh (identical anode/params objects).
DL=6.2; DT=9.8; LIFETIME=6; DRIFTSPEED=1.563
# The tagger pipeline.  fiducialutils MUST precede the taggers (they silently
# no-op without it); TGM before STM (STM skips TGM-flagged mains).
# tagger_check_fc is LAST: it evaluates every in-scope main regardless of the
# TGM/STM verdicts (so position does not change its coverage), and running it
# after them keeps their inputs free of the PCA/hough/steiner-boundary caches
# cluster_fc_check populates.  Verified: TGM/STM verdicts on the 10-event
# MCP2025C sample are identical with and without it (docs/25_fc-flag.md).
PIPELINE="switch_scope,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_fc"

usage() {
    cat <<EOF
Per-event SBND neutrino selection: Q/L bundles -> TGM/STM taggers -> label table.

Usage: $(basename "$0") [mc|data] [-N n] [-bw l,h] [-save-pr-tree] <idx|all>
       $(basename "$0") [mc|data] [-N n]            # list available events

  mc|data       input set (default mc)
  idx           1-based event index (same numbering as run_ql_evt.sh);
                'all' runs every event (parallel, cap nproc / SBND_MAX_JOBS)
  -bw l,h       beam window [l,h) in us on cluster_t0 (= matched flash time,
                trigger-referenced).  Default ${BEAM_WINDOW}.  Used for the TGM
                beam protection AND the table's in_beam / label columns.
  -nucand       enable the ported check_neutrino_candidate veto in
                tagger_check_tgm: in-beam-window bundles may then be tagged
                TGM when the Dijkstra path-topology veto clears them.
                ON BY DEFAULT for this chain since doc 26.
  -no-nucand    disable it (pre-doc-26 conservative never-tag-in-beam).
                Env: SBND_TGM_NUCAND=0.
                NB: the C++/jsonnet default stays FALSE -- only this runner
                opts in, so every other config and detector is unaffected.
  -save-pr-tree also re-save the post-PR tree to
                work/nusel_evt<ID>/pctree-pr-evt<ID>.tar.gz (NB: tagger flags
                set on only some clusters do NOT survive re-serialization; the
                table always takes verdicts from the log)

Table: one row per QUALIFYING BUNDLE = a cluster that is (a) flagged
flag_main_cluster and (b) in scope (passes switch_scope's active-volume
filter) -- exactly the population the taggers evaluate.  Main-flagged
out-of-volume shards are reported on stderr, not tabulated.  Flashes are
deduplicated across APAs first (one physical flash is reconstructed once per
TPC), so a beam flash seen in both TPCs counts once.
  run subrun event main_id flash_gid flash_apa flash_grp flash_time_us
  flash_pe flash_pe_grp in_beam n_bundle npts_main npts_bundle len_main_cm
  tgm stm fc label
  fc: fully-contained verdict (TaggerCheckFC).  Orthogonal to the cosmic
      taggers -- it does NOT enter 'label', matching the prototype where FC is
      an independent eval variable, not a veto.
  label: TGM | STM | nu-candidate (in-window, untagged) | not-tagged | no-bundle

Requires per-event imaging (./run_img_evt.sh) in \$SBND_WORK_ROOT/evt<ID>/.
The Q/L step reruns automatically when the pctree tarball is missing.
EOF
    sbnd_common_help
}

MODE=mc
BEAM_WINDOW="0.2,2.2"
SAVEPRT=""
# check_neutrino_candidate veto in tagger_check_tgm (in-beam-window bundles may
# then be tagged TGM).  DEFAULT ON for this chain as of doc 26 -- the knob-off
# path left in-beam bundles untaggable by construction, which is not the
# behavior we want from the selection.  The C++ and jsonnet defaults are still
# false; this runner passes tgm_neutrino_candidate=true explicitly, so nothing
# outside this chain changes.  -no-nucand / SBND_TGM_NUCAND=0 restores it.
NUCAND="${SBND_TGM_NUCAND:-1}"
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -N) SBND_SAMPLE="$2"; shift 2 ;;
        -N*) SBND_SAMPLE="${1#-N}"; shift ;;
        mc|data) MODE="$1"; shift ;;
        -bw) BEAM_WINDOW="$2"; shift 2 ;;
        -save-pr-tree|--save-pr-tree) SAVEPRT=1; shift ;;
        -nucand|--nucand) NUCAND=1; shift ;;
        -no-nucand|--no-nucand) NUCAND=0; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

case "$MODE" in
    mc)   REALITY=sim ;;
    data) REALITY=data ;;
esac

sbnd_check_sample "$MODE" || exit 1
[ -f "$JSONNET" ] || { echo "ERROR: missing jsonnet: $JSONNET" >&2; exit 1; }

load_events "$MODE" || exit 1
EVENT_IDS=("${SBND_EVENTS[@]}")

if [ $# -eq 0 ]; then
    echo "Sample: input-${SBND_SAMPLE}evt-$MODE   (${#EVENT_IDS[@]} events)"
    echo "Events for mode '$MODE' (idx -> EVT_ID):"
    for i in "${!EVENT_IDS[@]}"; do printf "  %2d -> %s\n" $((i + 1)) "${EVENT_IDS[$i]}"; done
    exit 0
fi

process_event() {
    local IDX=$1
    local EVT_ID="${EVENT_IDS[$((IDX - 1))]}"
    [ -n "$EVT_ID" ] || { echo "ERROR: invalid idx $IDX (1..${#EVENT_IDS[@]})" >&2; return 1; }

    local QLDIR="$SBND_WORK_ROOT/ql_evt${EVT_ID}"
    local PCT="$QLDIR/pctree-evt${EVT_ID}.tar.gz"
    local NUDIR="$SBND_WORK_ROOT/nusel_evt${EVT_ID}"
    local LOG="$NUDIR/wct_nusel_evt${EVT_ID}.log"

    # 1. Q/L matching with the persisted tree (idempotent: reuse an existing
    # pctree; run_ql_evt.sh is deterministic so a rerun reproduces the same
    # matching, only adding the tarball).
    if [ ! -s "$PCT" ]; then
        echo "[evt $EVT_ID] pctree missing — running Q/L step (-save-pctree)"
        "$SBND_DIR/run_ql_evt.sh" "$MODE" -save-pctree "$IDX" || return 1
        [ -s "$PCT" ] || { echo "ERROR: Q/L step did not produce $PCT" >&2; return 1; }
    fi

    rm -rf "$NUDIR"; mkdir -p "$NUDIR"

    # Run/subrun for the table + Bee labels, from the reco1 opflash metadata
    # (same fallback semantics as run_ql_evt.sh: absent keys -> 0/0).
    local RUN_NO=0 SUBRUN_NO=0 _md
    _md=$(tar xzOf "$QLDIR/opflash_apa0.tar.gz" "opflash_tensorset_${EVT_ID}_metadata.json" 2>/dev/null) || _md=''
    if [ -n "$_md" ]; then
        local _rse
        _rse=$(printf '%s' "$_md" | python3 -c \
            'import json,sys; d=json.load(sys.stdin); print(int(d.get("run",0)), int(d.get("subrun",0)))' \
            2>/dev/null) && [ -n "$_rse" ] && read -r RUN_NO SUBRUN_NO <<< "$_rse"
    fi

    local SAVEPRT_TLA=""
    [ -n "$SAVEPRT" ] && SAVEPRT_TLA="$NUDIR/pctree-pr-evt${EVT_ID}.tar.gz"

    # 2. PR tagger job.  Run from NUDIR so the dump-mode TensorFileSink's
    # trash-pr.tar.gz (if any) lands here, not in the source tree.
    echo "[evt $EVT_ID] rse=($RUN_NO, $SUBRUN_NO, $EVT_ID) taggers ($PIPELINE, bw=[$BEAM_WINDOW] us)"
    rm -f "$LOG"
    (
        cd "$NUDIR"
        wire-cell \
            -l stderr -l "${LOG}:debug" -L debug \
            --tla-str  "input=$PCT" \
            --tla-code "anode_indices=[0,1]" \
            --tla-str  "output_dir=$NUDIR" \
            --tla-code "run=${RUN_NO}" --tla-code "subrun=${SUBRUN_NO}" --tla-code "event=${EVT_ID}" \
            --tla-str  "reality=$REALITY" \
            --tla-code "DL=$DL" --tla-code "DT=$DT" \
            --tla-code "lifetime=$LIFETIME" --tla-code "driftSpeed=$DRIFTSPEED" \
            --tla-code "pipeline_names=[$(echo "$PIPELINE" | sed "s/[^,]\+/'&'/g")]" \
            --tla-str  "trackfitting_config=$SBND_DIR/sbnd_track_fitting.json" \
            --tla-str  "save_tensors=$SAVEPRT_TLA" \
            --tla-str  "dl_weights=" \
            --tla-code "beam_window_us=[$BEAM_WINDOW]" \
            --tla-code "tgm_neutrino_candidate=$([ "$NUCAND" = 1 ] && echo true || echo false)" \
            -c "$JSONNET"
    ) || return 1
    rm -f "$NUDIR/trash-pr.tar.gz"

    # 3. Per-bundle label table.
    python3 "$SBND_DIR/nusel_extract.py" \
        --pctree "$PCT" --prbee "$NUDIR/mabc-pr.zip" --prlog "$LOG" \
        --qlbee "$QLDIR/mabc-all-apa.zip" \
        --beam-window "$BEAM_WINDOW" \
        --run "$RUN_NO" --subrun "$SUBRUN_NO" \
        --out "$NUDIR/nusel-evt${EVT_ID}.tsv" || return 1

    echo "[evt $EVT_ID] table -> $NUDIR/nusel-evt${EVT_ID}.tsv"
    column -t "$NUDIR/nusel-evt${EVT_ID}.tsv" | sed 's/^/  /'
}

mkdir -p "$SBND_WORK_ROOT"
IDX="$1"
if [ "$IDX" = "all" ]; then
    batch_init
    echo "Mode $MODE: ${#EVENT_IDS[@]} events. Parallel jobs: $BATCH_MAX"
    for i in $(seq 1 "${#EVENT_IDS[@]}"); do
        _evtid="${EVENT_IDS[$((i - 1))]}"
        _blog="$SBND_WORK_ROOT/.batch_nusel_evt${_evtid}.log"
        batch_wait_slot
        ( process_event "$i" ) > "$_blog" 2>&1 &
        BATCH_PIDS[$!]=$_evtid
        echo "  [start] idx=$i evt=$_evtid  log: $_blog"
    done
    batch_drain
    batch_summary || exit 1

    # Merge the per-event tables (only those produced by this sample's events).
    _tsvs=()
    for _evtid in "${EVENT_IDS[@]}"; do
        _t="$SBND_WORK_ROOT/nusel_evt${_evtid}/nusel-evt${_evtid}.tsv"
        [ -s "$_t" ] && _tsvs+=("$_t")
    done
    if [ "${#_tsvs[@]}" -gt 0 ]; then
        python3 "$SBND_DIR/nusel_extract.py" --merge "${_tsvs[@]}" \
            --out "$SBND_WORK_ROOT/nusel-table.tsv" \
            --events-out "$SBND_WORK_ROOT/nusel-events.tsv"
        echo
        echo "===== per-bundle table: $SBND_WORK_ROOT/nusel-table.tsv ====="
        column -t "$SBND_WORK_ROOT/nusel-table.tsv"
        echo
        echo "===== per-event summary: $SBND_WORK_ROOT/nusel-events.tsv ====="
        column -t "$SBND_WORK_ROOT/nusel-events.tsv"
    fi
else
    process_event "$IDX"
fi
