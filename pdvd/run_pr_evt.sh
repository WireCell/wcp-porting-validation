#!/bin/bash
# Run the PDVD pattern-recognition (PR) tail for one event (doc pdvd/25).
# Usage: ./run_pr_evt.sh [-s sel_tag] [-stm-fit] [-stm|-stmlean|-unmerge|-nu|-nu-legacy|-empty] [-pipe a,b,c] <run> <evt|all> [subrun]
#
# DEFAULT MODE IS -stm (doc pdvd/39, owner 2026-09-04): the chain stops after
# the cosmic taggers.  -nu appends the PR tail, which since doc pdvd/48 (owner
# 2026-09-05) is the STM + MICHEL stage (check_stm_michel): every STM-tagged
# main is reconstructed as entry -> muon body (+ delta rays) -> Bragg stop ->
# Michel e- (+ nearby dots), the particle flow is rooted at the ENTRY point,
# and a reject verdict flags tagged clusters that are not stopped muons
# (T_stm_michel in tracking-pr.root; 'CheckSTM_Michel: cluster' in the log).
# It restores the track_fit / shower_track / vertices / mc Bee layers.
# -nu-legacy runs the pre-doc-48 neutrino PR tail (tagger_check_neutrino).
#
# Input : work/<RUN6>_<EVT>[_<TAG>]/pctree-evt<ID>.tar.gz + pctree-evt<ID>.tlas
#         (both written by run_clus_evt.sh -save-pctree; the .tlas sidecar carries
#         the drift speeds / trigger offsets / readout window the Q/L job used, so
#         the PR job rebuilds DetectorVolumes identically).
# Output: work/<RUN6>_<EVT>[_<TAG>]/mabc-pr.zip            Bee: clustering + dead + stm_fit
#                                                          + stm/steiner_graph/steiner_terminals (doc pdvd/39 round 3:
#                                                            the clusters the STM tagger TAGGED; stm_tagged is gone)
#                                                          + track_fit/shower_track/vertices/mc ONLY with -nu / -nu-legacy
#
# -unmerge (doc pdvd/39 round 2) adds the unmerge_assoc stage, which undoes the
# clustering-stage isolated grouping so the taggers see individual objects
# instead of a main body plus detached clumps.  REQUIRES a pctree from
# run_clus_evt.sh -save-assoc; the runner refuses otherwise (the stage is
# silently inert without the provenance).  PDVD_ALLOW_NO_ASSOC=1 overrides.
#         work/<RUN6>_<EVT>[_<TAG>]/calib-pr-evt<ID>.json  PrDisplayDump (segments, showers, kine, tagger flags)
#         work/<RUN6>_<EVT>[_<TAG>]/tracking-stm.root      STM fits (T_rec_charge/T_stm_pass/T_stm_eval), with -stm-fit
#         work/<RUN6>_<EVT>[_<TAG>]/tracking-pr.root       PR fits (T_rec_charge) + T_stm_michel/T_stm_michel_pts (-nu);
#                                                          + T_tagger/T_kine with -nu-legacy
#         work/<RUN6>_<EVT>[_<TAG>]/wct_pr_<RUN6>_<EVT>.log
# Verdicts: grep 'TaggerCheckSTM: cluster' / 'TaggerCheckTGM: cluster' in the log.
#
#   -nu      the PDVD PR chain: switch_scope, flag_mains, unmerge_assoc, steiner, fiducialutils,
#            tagger_check_tgm, tagger_check_stm, tagger_check_fc, protect_bundle,
#            steiner_refresh, check_stm_michel, tracking_visitor, pr_display   (doc pdvd/48)
#   -nu-legacy  the pre-doc-48 chain with tagger_check_neutrino + tagger_output in place
#            of check_stm_michel.  The notes below describe THAT tail.
#            Since 2026-09-02 the -nu chain runs the per-bundle PR ONLY on
#            STM-tagged bundles (nu_per_bundle_stm_only=true in
#            wct-pr-perevt.jsonnet; doc 25 sec 13.10) -- the PDVD working mode.
#            PDVD_PR_TLA="-S nu_per_bundle_stm_only=false" turns that one knob
#            off; since 2026-09-02 it is NO LONGER enough to reproduce the
#            stm1/stm2/stm3 arms -- see protect_stm_only_bundles below, both
#            must be passed together.
#            Since 2026-09-02 protect_bundle is gated the same way
#            (protect_stm_only_bundles=true; doc 25 sec 13.11): only a bundle
#            holding an STM-tagged cluster is opened for splitting.  This is
#            the mirror of the knob above and is what makes the chain usable --
#            039252/8's ClusteringProtectBundle goes 1726.8 s -> 146.2 ms and
#            the event 1821 s -> 74 s.  It is NOT byte-identical: cosmic
#            bundles keep their over-clustered shape in mabc-pr.zip and
#            calib-pr-evt*.json, while every STM bundle is split exactly as
#            before.  PDVD_PR_TLA="-S protect_stm_only_bundles=false" restores
#            the every-bundle stage.  To reproduce the stm1/stm2/stm3 arms pass
#            BOTH: PDVD_PR_TLA="-S nu_per_bundle_stm_only=false -S protect_stm_only_bundles=false"
#   -stm     cosmic taggers only (stops after steiner_refresh) + pr_display
#   -empty   pipeline_names=[] : the M2 round-trip identity gate
#   -pipe    explicit comma-separated pipeline list
#   -stm-fit append stm_magnify (tracking-stm.root); save_stm_fit is ON by default
#            in wct-pr-perevt.jsonnet, this only adds the ROOT writer
#   PDVD_PR_TLA="-S key=val ..."   extra wcsonnet args (knob overrides)
#   PDVD_PR_COMPILE_ONLY=1         write the compiled JSON and stop
#   PDVD_KEEP_CFG=1                keep .wct-pr<TAG>.json
#   PDVD_MAX_JOBS=N                parallel cap for 'evt all' (default 6)
#   PDVD_PR_RELEASE=1              add release_post_nu after tagger_check_neutrino (doc pdvd/28)
#   TCMALLOC_RELEASE_RATE=N        passed through to the preloaded tcmalloc (doc pdvd/28 M4)
#   PDVD_LOG_LEVEL=trace           file-sink level (default debug); PDVD_LOG_LOGGERS
#                                  the -L value (default = sink level), doc pdvd/28
#   PDVD_RESMON=off                disable the RSS sampler
#   WCT_TCMALLOC=off               drop the tcmalloc preload
set -o pipefail
PDVD_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH="$WCT_BASE/toolkit/cfg:$WCT_BASE/wire-cell-data${WIRECELL_PATH:+:$WIRECELL_PATH}"
_PRELOADS=""
if [ "${WCT_TCMALLOC:-on}" != "off" ] && [ -f /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 ]; then
    _PRELOADS="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
fi
# doc pdvd/28 sec 27 (2026-09-04): the DL (SCN) vertex is PDVD production now.
# Its embedded interpreter needs libpython loaded RTLD_GLOBAL or the SCN import
# fails and TaggerCheckNeutrino silently falls back to the geometric vertex
# after one WARN ("DL vertex failed") -- same idiom as sbnd_xin/run_pr_chain_batch.sh.
# WCT_PYLIB=off drops it (only sensible with dl_weights='').
if [ "${WCT_PYLIB:-on}" != "off" ]; then
    PYLIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.11.so.1.0
    [ -r "$PYLIB" ] || { echo "ERROR: libpython not found: $PYLIB (WCT_PYLIB=off to skip)" >&2; exit 1; }
    _PRELOADS="${_PRELOADS:+$_PRELOADS:}$PYLIB"
fi
WC_PRELOAD="${_PRELOADS:+LD_PRELOAD=$_PRELOADS}"
# torch inside the SCN call would otherwise spawn a thread per core in every job
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
. "$PDVD_DIR/_runlib.sh"

SEL_TAG=""
MODE=stm    # doc pdvd/39: cosmic taggers only; -nu restores the PR tail
STM_FIT=0
PIPE_EXPLICIT=""
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -stm-fit|--stm-fit) STM_FIT=1; shift ;;
        -stm) MODE=stm; shift ;;
        -stmlean) MODE=stmlean; shift ;;   # must precede the -s* catch-all
        -unmerge) MODE=unmerge; shift ;;   # doc pdvd/39 r2; before the -s* catch-all
        -nounmerge) MODE=merged; shift ;;    # pre-flip merged chain (owner 2026-09-04)
        -nounmerge-nu) MODE=numerged; shift ;;  # pre-flip merged NEUTRINO chain
        -nu) MODE=nu; shift ;;
        -nu-legacy) MODE=nulegacy; shift ;;   # doc pdvd/48: the pre-replacement neutrino PR tail
        -empty) MODE=empty; shift ;;
        -pipe) PIPE_EXPLICIT="$2"; shift 2 ;;
        -s) SEL_TAG="$2"; shift 2 ;;
        -s*) SEL_TAG="${1#-s}"; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"
if [ $# -lt 2 ]; then
    sed -n '2,32p' "$0"; echo; echo "Runs with pctree inputs:"; ls -d "$PDVD_DIR"/work/*/pctree-evt*.tar.gz 2>/dev/null | sed 's#.*/work/##' | head -20; exit 1
fi
RUN=$1; EVT=$2; SUBRUN_ARG=${3:-}

# doc pdvd/48 (owner 2026-09-05): -nu now runs the STM + Michel stage
# (check_stm_michel) in place of the neutrino PR tail; tagger_output is dropped
# with it (T_tagger/T_kine carried only neutrino BDT features).  The legacy
# neutrino tail is preserved VERBATIM as PIPE_NU_LEGACY behind -nu-legacy for
# the A/B gates (same precedent as PIPE_NU_MERGED).
PIPE_NU="switch_scope,flag_mains,unmerge_assoc,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_fc,protect_bundle,steiner_refresh,check_stm_michel,tracking_visitor,pr_display"
PIPE_NU_LEGACY="switch_scope,flag_mains,unmerge_assoc,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_fc,protect_bundle,steiner_refresh,tagger_check_neutrino,tracking_visitor,tagger_output,pr_display"
# PDVD PRODUCTION (owner decision 2026-09-04): unmerge_assoc is in the DEFAULT
# chain now, in both PIPE_STM and PIPE_NU, so -nu does not silently lose it.
# Pre-flip chain = -nounmerge (PIPE_STM_MERGED / PIPE_NU_MERGED below).
PIPE_STM="switch_scope,flag_mains,unmerge_assoc,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_fc,protect_bundle,steiner_refresh,pr_display"
# The pre-flip (merged) chains, kept verbatim for A/B archaeology.
PIPE_STM_MERGED="switch_scope,flag_mains,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_fc,protect_bundle,steiner_refresh,pr_display"
PIPE_NU_MERGED="switch_scope,flag_mains,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_fc,protect_bundle,steiner_refresh,tagger_check_neutrino,tracking_visitor,tagger_output,pr_display"
# doc pdvd/39: same stages, but tagger_check_tgm moved AHEAD of steiner so the
# Steiner build can skip TGM-flagged clusters (pair with
# PDVD_PR_TLA="-S 'steiner_skip_flags=[\"TGM\"]'").  Held out of the default
# until the sec.5 A/B shows the TGM and STM verdict sets are unchanged.
PIPE_STM_LEAN="switch_scope,flag_mains,fiducialutils,tagger_check_tgm,steiner,tagger_check_stm,tagger_check_fc,protect_bundle,steiner_refresh,pr_display"
# doc pdvd/39 round 2: PIPE_STM plus unmerge_assoc, which undoes the isolated
# GROUPING clustering_isolated performed at the clus stage (a main cluster
# physically merged with the small clusters near but NOT connected to it).
# Placed after flag_mains -- so the split-off fragments leave the main's Steiner
# build and STM fit without being promoted to mains -- and before steiner,
# because separate() does not carry node-local PCs.
# REQUIRES a pctree written by run_clus_evt.sh -save-assoc; without the
# provenance arrays the visitor is inert (it warns and splits nothing).
# Held out of the default until the doc pdvd/39 sec 12 A/B is adjudicated.
# -unmerge is now the default; the name is kept so old commands keep working.
PIPE_STM_UNMERGE="$PIPE_STM"
case "$MODE" in
    nu) PIPE="$PIPE_NU" ;;
    nulegacy) PIPE="$PIPE_NU_LEGACY" ;;
    stm) PIPE="$PIPE_STM" ;;
    numerged) PIPE="$PIPE_NU_MERGED" ;;
    merged) PIPE="$PIPE_STM_MERGED" ;;
    stmlean) PIPE="$PIPE_STM_LEAN" ;;
    unmerge) PIPE="$PIPE_STM_UNMERGE" ;;
    empty) PIPE="" ;;
esac
[ -n "$PIPE_EXPLICIT" ] && PIPE="$PIPE_EXPLICIT"
# PDVD_PR_RELEASE=1 (doc pdvd/28): insert the ClusteringReleaseCaches visitor
# right after tagger_check_neutrino (graphs + fitter scratch dropped before the
# writers; output-neutral, lowers the writer-stage RSS).  Default off.
if [ "${PDVD_PR_RELEASE:-0}" = 1 ] && [ -n "$PIPE" ]; then
    case ",$PIPE," in
        *,tagger_check_neutrino,*) PIPE=$(echo "$PIPE" | sed 's/tagger_check_neutrino/tagger_check_neutrino,release_post_nu/') ;;
        *) PIPE="$PIPE,release_post_nu" ;;
    esac
fi
if [ "$STM_FIT" = 1 ] && [ -n "$PIPE" ]; then PIPE="$PIPE,stm_magnify"; fi
PIPE_JSON="[$(echo "$PIPE" | sed -e 's/,/","/g' -e 's/^/"/' -e 's/$/"/' -e 's/^""$//')]"

process_event() {
    local RUN=$1 EVT=$2
    local RUN_STRIPPED=$((10#$RUN))
    local RUN_PADDED
    RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")
    local WORKDIR="$PDVD_DIR/work/${RUN_PADDED}_${EVT}${SEL_TAG:+_$SEL_TAG}"
    local PCTREE TLAS EVENT_NO
    PCTREE=$(ls "$WORKDIR"/pctree-evt*.tar.gz 2>/dev/null | head -1)
    if [ -z "$PCTREE" ]; then
        echo "[skip] run=$RUN evt=$EVT: no pctree-evt*.tar.gz in $WORKDIR (run_clus_evt.sh -save-pctree first)" >&2
        return 2
    fi
    EVENT_NO=$(basename "$PCTREE" | sed -E 's/pctree-evt([0-9]+)\.tar\.gz/\1/')
    TLAS="$WORKDIR/pctree-evt${EVENT_NO}.tlas"
    # PDVD_PR_SKIP_DONE=1: resume a batch -- skip an event whose PR outputs exist
    if [ "${PDVD_PR_SKIP_DONE:-0}" = 1 ] && [ -s "$WORKDIR/calib-pr-evt${EVENT_NO}.json" ] && [ -s "$WORKDIR/tracking-stm.root" ] && [ -s "$WORKDIR/mabc-pr.zip" ]; then
        echo "[skip-done] run=$RUN evt=$EVT: PR outputs present"
        return 0
    fi
    if [ ! -f "$TLAS" ]; then
        echo "ERROR: missing $TLAS (the Q/L job's TLA sidecar) -- rerun run_clus_evt.sh -save-pctree" >&2
        return 1
    fi
    local T_BOT T_TOP NT DS_BOT DS_TOP SCF QL
    T_BOT=$(awk -F= '$1=="trigger_offset_bot_us"{print $2}' "$TLAS")
    T_TOP=$(awk -F= '$1=="trigger_offset_top_us"{print $2}' "$TLAS")
    NT=$(awk -F= '$1=="readout_window_ticks"{print $2}' "$TLAS")
    DS_BOT=$(awk -F= '$1=="drift_speed_bot_mmus"{print $2}' "$TLAS")
    DS_TOP=$(awk -F= '$1=="drift_speed_top_mmus"{print $2}' "$TLAS")
    SCF=$(awk -F= '$1=="stepped_center_fallback"{print $2}' "$TLAS")
    QL=$(awk -F= '$1=="qlmatch"{print $2}' "$TLAS")
    # subrun: the value the Q/L job stamped (sidecar) unless given on the command line
    local SUBRUN=${SUBRUN_ARG:-$(awk -F= '$1=="subrun"{print $2}' "$TLAS")}
    SUBRUN=${SUBRUN:-0}
    if [ "$QL" != 1 ]; then
        echo "[skip] run=$RUN evt=$EVT: the pctree was written WITHOUT Q/L matching (qlmatch=$QL); no matched bundles to tag" >&2
        return 2
    fi
    local TAG_SUFFIX=""
    local LOG="$WORKDIR/wct_pr_${RUN_PADDED}_${EVT}.log"
    local CFG_JSON="$WORKDIR/.wct-pr${SEL_TAG:+_$SEL_TAG}.json"
    rm -f "$LOG"   # spdlog appends; one run = one log
    echo "PR: run=$RUN evt=$EVT art_event=$EVENT_NO work=$WORKDIR pipeline=[$PIPE]"
    echo "    tlas: v_bot=$DS_BOT v_top=$DS_TOP trig_bot=$T_BOT trig_top=$T_TOP nticks=$NT"
    # shellcheck disable=SC2086
    (cd "$PDVD_DIR" && wcsonnet \
        -A "input=${PCTREE}" \
        -A "output_dir=${WORKDIR}" \
        -S "run=${RUN_STRIPPED}" \
        -S "subrun=${SUBRUN}" \
        -S "event=${EVENT_NO}" \
        -S "drift_speed_bot_mmus=${DS_BOT:-1.48073}" \
        -S "drift_speed_top_mmus=${DS_TOP:-1.48073}" \
        -S "trigger_offset_bot_us=${T_BOT:-0}" \
        -S "trigger_offset_top_us=${T_TOP:-0}" \
        -S "readout_window_ticks=${NT:-10000}" \
        -S "stepped_center_fallback=${SCF:-false}" \
        -S "pipeline_names=${PIPE_JSON}" \
        ${PDVD_PR_TLA:-} \
        -o "$CFG_JSON" wct-pr-perevt.jsonnet)
    if [ ! -s "$CFG_JSON" ]; then
        echo "ERROR: wcsonnet failed to compile wct-pr-perevt.jsonnet" >&2
        return 1
    fi
    # doc pdvd/27: geometry provenance guard.  The pctree's 3D points and face
    # ids were sampled with the Q/L job's wires file; the PR retile re-samples
    # the same blobs with THIS job's anodes.  v6 -> v7-uvwfit swapped the face
    # idents of anodes 2,3,6,7, so a v6 pctree under a v7 PR job moved every
    # retile on those anodes one face height in y (039349/53's "isolated piece"
    # 75 cm from its own track).  Refuse the mix; PDVD_ALLOW_STALE_GEOMETRY=1
    # downgrades to a warning; a pre-doc-27 sidecar (no wires= line) warns.
    local PR_WIRES TLA_WIRES
    PR_WIRES=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(sorted({n["data"]["filename"] for n in c if n.get("type")=="WireSchemaFile"})[0])' "$CFG_JSON" 2>/dev/null)
    TLA_WIRES=$(awk -F= '$1=="wires"{print $2}' "$TLAS")
    if [ -z "$TLA_WIRES" ]; then
        echo "WARNING: $TLAS has no wires= line (pre-doc-27 sidecar): cannot prove the pctree was sampled with ${PR_WIRES:-?}; regenerate imaging + clustering if the wires file changed since (doc pdvd/27)" >&2
    elif [ "$TLA_WIRES" != "$PR_WIRES" ]; then
        if [ "${PDVD_ALLOW_STALE_GEOMETRY:-0}" = 1 ]; then
            echo "WARNING: pctree wires=$TLA_WIRES but this PR job compiles wires=$PR_WIRES (allowed by PDVD_ALLOW_STALE_GEOMETRY=1)" >&2
        else
            echo "ERROR: run=$RUN evt=$EVT: pctree was sampled with wires=$TLA_WIRES but this PR job compiles wires=$PR_WIRES -- regenerate imaging + clustering (run_img_evt.sh, run_clus_evt.sh -save-pctree) before PR, or set PDVD_ALLOW_STALE_GEOMETRY=1 (doc pdvd/27)" >&2
            rm -f "$CFG_JSON"; return 3
        fi
    fi
    # doc pdvd/39 round 2: -unmerge is INERT on a pctree written without
    # -save-assoc.  ClusteringUnmergeBundle's "no usable provenance => skip"
    # stance is deliberate (it must never fall back to splitting on graph
    # connectivity), so the run finishes rc=0 with a normal-looking Bee set and
    # nothing split -- a vacuous arm that reads like a real one.  Its
    # require_provenance knob does NOT cover this: it guards only the wasmain
    # array under restore_demoted_mains (ClusteringUnmergeBundle.cxx:426).
    # So check here.  The probe reads only the tarball's metadata members (~0.1 s
    # on a 20 MB pctree).
    # PDVD PRODUCTION 2026-09-04: unmerge_assoc is in the default chain, so this
    # guard must key on the SELECTED PIPELINE, not on MODE -- otherwise the
    # default path (MODE=stm) would run the stage unguarded and go silently inert.
    local _need_assoc
    case ",$PIPE," in *,unmerge_assoc,*) _need_assoc=1 ;; *) _need_assoc=0 ;; esac
    if [ "$_need_assoc" = 1 ]; then
        # NB grep -q/-m1 would exit on the first match, SIGPIPE the tar, and
        # under this script's `set -o pipefail` the pipeline would report
        # FAILURE on a match -- the guard would then reject every tree,
        # provenance or not.  Count instead, so grep drains the stream.
        local _nprov
        _nprov=$(tar -xzOf "$PCTREE" --wildcards '*_metadata.json' 2>/dev/null \
                 | grep -c perblob || true)
        if [ "${_nprov:-0}" -gt 0 ]; then
            :
        elif [ "${PDVD_ALLOW_NO_ASSOC:-0}" = 1 ]; then
            echo "WARNING: $PCTREE has no perblob provenance; -unmerge will split NOTHING (allowed by PDVD_ALLOW_NO_ASSOC=1)" >&2
        else
            echo "ERROR: run=$RUN evt=$EVT: unmerge_assoc (in the DEFAULT chain since 2026-09-04) needs a pctree written with run_clus_evt.sh -save-assoc, but $PCTREE carries no 'perblob' point cloud -- unmerge_assoc would split nothing and the arm would be silently identical to -stm (doc pdvd/39 sec 12.2).  Re-run clustering with -save-assoc, or set PDVD_ALLOW_NO_ASSOC=1 to run it anyway." >&2
            rm -f "$CFG_JSON"; return 4
        fi
    fi
    if [ "${PDVD_PR_COMPILE_ONLY:-0}" = 1 ]; then
        echo "[compile-only] wrote $CFG_JSON"
        return 0
    fi
    local RES_TXT="$WORKDIR/pr_resource_${RUN_PADDED}_${EVT}.txt"
    local RES_CSV="$WORKDIR/pr_rss_${RUN_PADDED}_${EVT}.csv"
    local _t0=$SECONDS _smpid=""
    # PDVD_LOG_LEVEL: file-sink level (default debug; 'trace' exposes the
    # CreateSteinerGraph / ImproveCluster / NeutrinoPattern phase timers, doc
    # pdvd/28).  PDVD_LOG_LOGGERS: the -L argument (default = the sink level;
    # e.g. 'clus:trace' restricts trace to the clus logger).
    env $WC_PRELOAD GOGC=off wire-cell \
        -l stderr \
        -l "${LOG}:${PDVD_LOG_LEVEL:-debug}" \
        -L "${PDVD_LOG_LOGGERS:-${PDVD_LOG_LEVEL:-debug}}" \
        -c "$CFG_JSON" &
    local _wcpid=$!
    if [ "${PDVD_RESMON:-on}" != "off" ]; then
        echo "epoch_s,clock,vmrss_kb,vmhwm_kb" > "$RES_CSV"
        ( while kill -0 "$_wcpid" 2>/dev/null; do
            _line=$(awk '/^VmRSS:/{r=$2}/^VmHWM:/{h=$2}END{print r","h}' \
                    "/proc/$_wcpid/status" 2>/dev/null)
            [ -n "$_line" ] && echo "$(date +%s),$(date +%H:%M:%S.%2N),$_line" >> "$RES_CSV"
            sleep 2
          done ) &
        _smpid=$!
    fi
    local _rc=0
    wait "$_wcpid" || _rc=$?
    if [ -n "$_smpid" ]; then
        kill "$_smpid" 2>/dev/null || true
        wait "$_smpid" 2>/dev/null || true
    fi
    local _wall=$((SECONDS - _t0))
    local _peak_kb=0
    [ -f "$RES_CSV" ] && _peak_kb=$(awk -F, 'NR>1 && $4>m{m=$4}END{print m+0}' "$RES_CSV")
    awk -v r="$RUN_PADDED" -v e="$EVT" -v w="$_wall" -v p="$_peak_kb" -v m="$MODE" -v f="$STM_FIT" 'BEGIN{
        printf "run=%s evt=%s wall_s=%d peak_rss_gb=%.2f mode=%s stmfit=%s\n", r,e,w,p/1048576,m,f}' | tee "$RES_TXT"
    [ "${PDVD_KEEP_CFG:-0}" = 1 ] || rm -f "$CFG_JSON"
    if [ "$_rc" -ne 0 ]; then
        echo "ERROR: wire-cell PR failed (rc=$_rc) for run=$RUN evt=$EVT; see $LOG" >&2
        return "$_rc"
    fi
    local NSTM NTGM
    NSTM=$(grep -c 'TaggerCheckSTM: cluster' "$LOG" 2>/dev/null || true)
    NTGM=$(grep -c 'TaggerCheckTGM: cluster' "$LOG" 2>/dev/null || true)
    echo "PR done -> $WORKDIR  (STM verdict lines: $NSTM, TGM verdict lines: $NTGM)"
}

if [ "$EVT" = "all" ]; then
    RUN_PADDED=$(printf '%06d' "$((10#$RUN))")
    mapfile -t _events < <(ls -d "$PDVD_DIR"/work/${RUN_PADDED}_*${SEL_TAG:+_$SEL_TAG} 2>/dev/null \
             | sed -E "s#.*/${RUN_PADDED}_([0-9]+)${SEL_TAG:+_$SEL_TAG}\$#\1#" | grep -E '^[0-9]+$' | sort -n)
    if [ ${#_events[@]} -eq 0 ]; then
        echo "no work/${RUN_PADDED}_<idx>${SEL_TAG:+_$SEL_TAG} dirs" >&2; exit 1
    fi
    export PDVD_MAX_JOBS=${PDVD_MAX_JOBS:-6}
    batch_init
    echo "Found ${#_events[@]} event(s) for run=$RUN${SEL_TAG:+ tag=$SEL_TAG}: ${_events[*]}"
    echo "Parallel jobs: $BATCH_MAX"
    for _e in "${_events[@]}"; do
        _blogfile="$PDVD_DIR/work/.batch_pr_${RUN_PADDED}_${_e}${SEL_TAG:+_$SEL_TAG}.log"
        batch_wait_slot
        ( process_event "$RUN" "$_e" ) > "$_blogfile" 2>&1 &
        BATCH_PIDS[$!]=$_e
        echo "  [start] evt=$_e  log: $_blogfile"
    done
    batch_drain
    batch_summary
    exit $?
else
    ( process_event "$RUN" "$EVT" )
    exit $?
fi
