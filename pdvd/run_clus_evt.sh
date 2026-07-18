#!/bin/bash
# Run clustering for one event.
# Usage: ./run_clus_evt.sh [-a anode] [-s sel_tag] [-noq] [-calib] [-op] <run> <evt|all> [subrun]
#        ./run_clus_evt.sh               # list available runs
#
# Q/L (charge-light) matching runs by DEFAULT; -noq (or PDVD_QLMATCH=0) disables it.
# An event with no converted light (work/<RUN6>_light<EVENTNO>/opflash_pdvd-wct.tar.gz,
# looked up by the ART EVENT NUMBER parsed from the cluster tarball — the charge work
# dirs are INDEX-named) auto-falls back to the no-matching chain, so 'evt all' never
# fails on a light-less event (e.g. run 039324, no raw light staged).
#   -calib          also dump the hand-scan calib JSON (calib-evt<EVENTNO>.json)
#   -op / -noop     optical "op" bee instance (default ON when matching)
#   PDVD_LIGHT_MODEL=semi        semi-analytical visibility backend (default library)
#   PDVD_DRIFT_SPEED_BOT_MMUS / PDVD_DRIFT_SPEED_TOP_MMUS
#                                per-side drift speeds in mm/us (bottom =
#                                anodes 0-3, top = 4-7).  Unset => the
#                                default 1.48073 (= 0.148073 cm/us, the A<->C
#                                crosser W-decon measurement, run 039252
#                                evt 298651, docs/qlmatch/
#                                pdvd-drift-crosser-298651.md, 2026-07-13;
#                                superseded 1.53 FR-Argon which superseded the
#                                1.586 cathode-pinned convention; the toolkit
#                                default stays the legacy 1.568).
#                                Set to 'null' for the legacy value.
#   PDVD_TRIGGER_OFFSET_US=<us>  override the light<->charge time-base offset
#                                (BOTH crates; diagnostics only)
#   PDVD_QL_DIAG=1               offset-calibration diagnostic mode: containment off,
#                                flash_minPE=100, trigger offsets forced 0
#   PDVD_QL_DIAG=2               same loosened knobs but KEEP the measured offsets
#                                (closure validation)
# The light<->charge offsets are PER CRATE and PER EVENT: opflash metadata
# offset_bot_us (BDE, bottom volume) / offset_top_us (TDE, top volume), measured
# from the rawwf trigoff tree by run_light_evt.sh, + the per-run residual from
# data/ql_trigger_offset.txt ("<run> <offset_us>" lines, applied to both).
# Legacy archives without the per-crate keys fall back to metadata offset_us.
#
# EVT may be 'all' to run every discovered event in parallel (capped at nproc,
# override with PDVD_MAX_JOBS=N).  Events with missing inputs are skipped.
#
# Input:  work/<run>_<evt>[_sel<TAG>]/ (from imaging) or input_data event dir as fallback
# Output: work/<run>_<evt>[_sel<TAG>]/mabc-anode{N}.zip, mabc-group0123.zip, mabc-group4567.zip, mabc-all-apa.zip

set -e

PDVD_DIR=$(cd "$(dirname "$0")" && pwd)

WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

# Preload tcmalloc for the clustering wire-cell process (~20% faster on
# busy events).  Unlocked by the get_closest_blob pointer-order fix: with
# it, glibc and tcmalloc outputs are byte-identical on the A/B event set.
# Disable with WCT_TCMALLOC=off.
TCMALLOC_SO=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
WC_PRELOAD=""
if [ "${WCT_TCMALLOC:-on}" = "on" ] && [ -f "$TCMALLOC_SO" ]; then
    WC_PRELOAD="LD_PRELOAD=$TCMALLOC_SO"
fi

. "$PDVD_DIR/_runlib.sh"

ANODE=""
SEL_TAG=""
# Charge-light (Q/L) matching before the final all-TPC clustering: ON by default
# (-noq or PDVD_QLMATCH=0 forces the historical no-matching chain).  Per-event,
# matching auto-disables when the converted light is absent (see QLMATCH_EVT).
QLMATCH=${PDVD_QLMATCH:-1}
CALIB=0
OPDUMP=${PDVD_OPDUMP:-1}   # optical "op" bee instance; default ON, -noop to disable
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -a) ANODE="$2"; shift 2 ;;
        -a*) ANODE="${1#-a}"; shift ;;
        -s) SEL_TAG="$2"; shift 2 ;;
        -s*) SEL_TAG="${1#-s}"; shift ;;
        -q) QLMATCH=1; shift ;;
        -noq|--no-qlmatch) QLMATCH=0; shift ;;
        -calib|--calib) CALIB=1; QLMATCH=1; shift ;;
        -op|--op) OPDUMP=1; QLMATCH=1; shift ;;
        -noop|--no-op) OPDUMP=0; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

if [ $# -eq 0 ]; then
    list_runs; exit 0
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 [-a anode] [-s sel_tag] [-noq] [-calib] [-noop] <run> <evt|all> [subrun]" >&2
    exit 1
fi
RUN=$1
EVT=$2
SUBRUN=${3:-0}

RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//')
[ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

find_evtdir() {
    local base="$PDVD_DIR/input_data"
    for rname in "run${RUN}" "run${RUN_PADDED}" "run${RUN_STRIPPED}"; do
        local rdir="$base/$rname"
        [ -d "$rdir" ] || continue
        for ename in "evt${EVT}" "evt_${EVT}"; do
            local cand="$rdir/$ename"
            if [ -d "$cand" ] && [ -n "$(ls -A "$cand" 2>/dev/null)" ]; then
                echo "$cand"; return 0
            fi
        done
        if ls "$rdir/clusters-apa-anode"*"-ms-active.tar.gz" >/dev/null 2>&1; then
            echo "$rdir"; return 0
        fi
    done
    return 1
}

process_event() {
    local RUN=$1 EVT=$2
    local RUN_STRIPPED RUN_PADDED WORKDIR EVTDIR CLUS_INPUT ANODE_CODE TAG_SUFFIX LOG
    local ANODE0_CLUS EVENT_NO
    RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//')
    [ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
    RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

    if [ -n "$SEL_TAG" ]; then
        WORKDIR="$PDVD_DIR/work/${RUN_PADDED}_${EVT}_${SEL_TAG}"
    else
        WORKDIR="$PDVD_DIR/work/${RUN_PADDED}_${EVT}"
    fi
    mkdir -p "$WORKDIR"

    CLUS_INPUT=""
    if ls "$WORKDIR/clusters-apa-anode"*"-ms-active.tar.gz" >/dev/null 2>&1; then
        CLUS_INPUT="$WORKDIR"
    else
        EVTDIR=$(find_evtdir) || EVTDIR=""
        if [ -n "$EVTDIR" ] && ls "$EVTDIR/clusters-apa-anode"*"-ms-active.tar.gz" >/dev/null 2>&1; then
            CLUS_INPUT="$EVTDIR"
        fi
    fi

    if [ -z "$CLUS_INPUT" ]; then
        echo "[skip] run=$RUN evt=$EVT: no cluster tarballs found" >&2
        return 2
    fi
    echo "Cluster input: $CLUS_INPUT"
    echo "Work dir:      $WORKDIR"

    ANODE0_CLUS=$(ls "$CLUS_INPUT/clusters-apa-anode"*"-ms-active.tar.gz" 2>/dev/null | head -1)
    EVENT_NO=$(tar tzf "$ANODE0_CLUS" | head -1 | sed -E 's/.*cluster_([0-9]+)_.*/\1/')
    if ! echo "$EVENT_NO" | grep -qE '^[0-9]+$'; then
        echo "ERROR: could not parse event number from $ANODE0_CLUS (got: '$EVENT_NO')" >&2
        return 1
    fi
    echo "Art event number: $EVENT_NO"

    # Q/L matching needs this event's converted light.  The charge work dirs are
    # INDEX-named (work/<RUN6>_<idx>) but the light chain's dirs are keyed by the
    # ART EVENT NUMBER (work/<RUN6>_light<EVENTNO>) — bridge via EVENT_NO above.
    local QLMATCH_EVT=$QLMATCH
    # PDVD_LIGHT_SUFFIX selects a tagged light-chain variant dir
    # (work/<RUN6>_light<EVENTNO><SUFFIX>, cf. run_light_evt.sh -s), e.g.
    # "_satoff" for the saturation-veto-off study.  Default empty = canonical.
    local OPFLASH_TAR="$PDVD_DIR/work/${RUN_PADDED}_light${EVENT_NO}${PDVD_LIGHT_SUFFIX:-}/opflash_pdvd-wct.tar.gz"
    if [ "$QLMATCH_EVT" = 1 ] && [ ! -f "$OPFLASH_TAR" ]; then
        echo "[note] no $OPFLASH_TAR -> skipping Q/L matching for this event" >&2
        QLMATCH_EVT=0
    fi

    # Light<->charge time-base offsets, PER CRATE: opflash metadata
    # offset_bot_us/offset_top_us (measured per event from the rawwf trigoff
    # tree by run_light_evt.sh; the BDE/TDE charge windows open up to ~32 us
    # apart) + the per-run residual constant from data/ql_trigger_offset.txt
    # (applied to both).  Old archives without the per-crate keys fall back to
    # the legacy scalar offset_us for both sides.  PDVD_TRIGGER_OFFSET_US
    # overrides everything (both sides); the diagnostic mode (PDVD_QL_DIAG=1)
    # forces 0.  Also guard that the opflash metadata 'event' matches the
    # charge EVENT_NO.
    TRIGGER_OFFSET_BOT_US=0
    TRIGGER_OFFSET_TOP_US=0
    READOUT_NTICKS=10000
    if [ "$QLMATCH_EVT" = 1 ]; then
        local META_LINE META_BOT META_TOP OPFLASH_EVENT
        META_LINE=$(python3 - "$OPFLASH_TAR" <<'PY' || echo "0 0"
import sys, json, tarfile
bot = top = 0.0
evt = ""
with tarfile.open(sys.argv[1]) as tf:
    for m in tf.getmembers():
        if m.name.endswith("_metadata.json"):
            md = json.loads(tf.extractfile(m).read())
            if "offset_bot_us" in md and "offset_top_us" in md:
                bot = float(md["offset_bot_us"])
                top = float(md["offset_top_us"])
            elif "offset_us" in md:
                bot = top = float(md["offset_us"])
            if "event" in md:
                evt = int(md["event"])
            break
print(bot, top, evt)
PY
)
        META_BOT=$(echo "$META_LINE" | awk '{print $1}')
        META_TOP=$(echo "$META_LINE" | awk '{print $2}')
        OPFLASH_EVENT=$(echo "$META_LINE" | awk '{print $3}')
        if [ -n "$OPFLASH_EVENT" ] && [ "$OPFLASH_EVENT" != "$EVENT_NO" ]; then
            echo "ERROR: charge/light event mismatch: charge art_event=$EVENT_NO but opflash event=$OPFLASH_EVENT ($OPFLASH_TAR)." >&2
            return 1
        fi
        local RUN_OFF=0
        if [ -f "$PDVD_DIR/data/ql_trigger_offset.txt" ]; then
            RUN_OFF=$(awk -v r="$RUN_STRIPPED" '$1+0==r+0{print $2}' "$PDVD_DIR/data/ql_trigger_offset.txt" | head -1)
            RUN_OFF=${RUN_OFF:-0}
        fi
        # PDVD_QL_EXTRA_OFFSET_US: an ADDITIVE common pull applied to BOTH crates
        # (preserving the per-crate BDE/TDE metadata skew, unlike the absolute
        # PDVD_TRIGGER_OFFSET_US override).  Larger trigger_offset => smaller u
        # (du/dtrig = -v), i.e. a toward-ANODE charge-placement pull; +13.507 us
        # = 2.0 cm at v=0.148073.
        # PRODUCTION DEFAULT 13.507 us (2 cm) as of 2026-07-14 for ALL PDVD runs
        # (run 039252 evt298567 cathode-crosser study, docs/qlport/
        # pdvd-cathode-containment-flash-demotion.md §10).  TRIAL value fit to two
        # hand labels, NOT census-pinned -- set PDVD_QL_EXTRA_OFFSET_US=0 to
        # recover the pre-study (metadata-only) offsets.
        local QL_EXTRA_OFF=${PDVD_QL_EXTRA_OFFSET_US:-13.507}
        TRIGGER_OFFSET_BOT_US=$(python3 -c "print(${META_BOT:-0} + ${RUN_OFF:-0} + ${QL_EXTRA_OFF:-0})")
        TRIGGER_OFFSET_TOP_US=$(python3 -c "print(${META_TOP:-0} + ${RUN_OFF:-0} + ${QL_EXTRA_OFF:-0})")
        if [ -n "${PDVD_TRIGGER_OFFSET_US:-}" ]; then
            TRIGGER_OFFSET_BOT_US="$PDVD_TRIGGER_OFFSET_US"
            TRIGGER_OFFSET_TOP_US="$PDVD_TRIGGER_OFFSET_US"
        fi
        # PDVD_QL_DIAG=1: loosened knobs AND offsets forced 0 (the historical
        # offset-hunt mode).  PDVD_QL_DIAG=2: loosened knobs but KEEP the
        # measured offsets (closure validation against the =1 dumps).
        if [ "${PDVD_QL_DIAG:-0}" = 1 ]; then
            TRIGGER_OFFSET_BOT_US=0
            TRIGGER_OFFSET_TOP_US=0
        fi
        echo "Trigger offsets: bot=${TRIGGER_OFFSET_BOT_US} top=${TRIGGER_OFFSET_TOP_US} us (metadata ${META_BOT}/${META_TOP} + run table ${RUN_OFF} + extra ${QL_EXTRA_OFF})"

        # Real readout window (post-resample SP frame length, 10000 ticks x
        # 0.5 us = 5 ms) for the window-truncation flag.
        local _SPF
        _SPF=$(ls "$CLUS_INPUT"/protodune-sp-dnnroi-frames-anode*.tar.bz2 2>/dev/null | head -1)
        if [ -n "$_SPF" ]; then
            local _NT
            _NT=$(python3 - "$_SPF" <<'PY'
import sys, tarfile, io, numpy as np
with tarfile.open(sys.argv[1]) as tf:
    name = next(m.name for m in tf.getmembers() if m.name.startswith("frame_"))
    print(np.load(io.BytesIO(tf.extractfile(name).read())).shape[1])
PY
)
            READOUT_NTICKS=${_NT:-10000}
            echo "Readout window: ${READOUT_NTICKS} ticks (from $(basename "$_SPF"))"
        fi
    fi

    if [ -n "$ANODE" ]; then
        ANODE_CODE="[$ANODE]"
        TAG_SUFFIX="_a${ANODE}"
    else
        ANODE_CODE="[0,1,2,3,4,5,6,7]"
        TAG_SUFFIX=""
    fi

    LOG="$WORKDIR/wct_clus_${RUN_PADDED}_${EVT}${TAG_SUFFIX}.log"
    echo "Log:           $LOG"

    cd "$PDVD_DIR"
    rm -f "$LOG"
    # Pre-compile the config with wcsonnet and feed wire-cell pure JSON,
    # with GOGC=off.  Rationale: an intermittent clustering SIGABRT was
    # identified as the embedded gojsonnet Go runtime GC crashing
    # ("traceback did not unwind completely").  libgojsonnet is hard-linked
    # so its runtime threads exist regardless of config format, but with a
    # pure-JSON config wire-cell never evaluates jsonnet (no Go heap) and
    # GOGC=off disables the Go collector entirely, including the 2-minute
    # periodic forced GC that is the crash vector.  Config is identical
    # (same pattern as imaging -P).
    local CFG_JSON="$WORKDIR/.wct-clus${TAG_SUFFIX}.json"
    # Q/L diagnostic modes: containment off + a higher PE floor to bound the
    # un-pruned bundle count.  =1 also forces offsets 0 (offset hunt);
    # =2 keeps the measured offsets (closure validation).
    local QL_CONTAIN=true QL_MINPE=${PDVD_QL_FLASH_MINPE:-25}
    if [ "${PDVD_QL_DIAG:-0}" = 1 ] || [ "${PDVD_QL_DIAG:-0}" = 2 ]; then
        QL_CONTAIN=false; QL_MINPE=100
    fi
    # PDVD_QL_CATHODE_EXT1_CM: cathode-side containment tolerance (cm) past the
    # cathode.  PRODUCTION DEFAULT 2.0 cm as of 2026-07-14 for ALL PDVD runs
    # (widened from the C++ +1.2 cm; run 039252 evt298567 study, docs/qlport/
    # pdvd-cathode-containment-flash-demotion.md §10).  Set
    # PDVD_QL_CATHODE_EXT1_CM=1.2 to recover the pre-study C++ tolerance.  The
    # toolkit qlmatching.jsonnet default stays null (byte-identical) -- this
    # runner is where PDVD processing turns the wider cushion on.
    local QL_CATHEXT1_ARG=(-S "ql_cathode_ext1_cm=${PDVD_QL_CATHODE_EXT1_CM:-2.0}")
    # PDVD_QL_ANODE_MARGIN_CM: anode-side containment slack (cm) below anode_ext1.
    # The anode floor is anode_ext1 - margin, gating BOTH containment and the
    # at_x_boundary / close_to_PMT flag window.  PRODUCTION DEFAULT 2.0 cm
    # (floor -4 cm) as of 2026-07-16, widened from the C++/prototype 1.0 cm
    # (floor -3 cm): the 2 cm anode pull above pushes a full-gap cathode-crosser's
    # anode end past the old floor, prefiltering the correct (bright) flash out of
    # its candidate pool -- run 039252 evt298567 cluster top:22 missed by 0.501 cm
    # and was demoted to the 244.0 us flash instead of 274.4 us.  Set
    # PDVD_QL_ANODE_MARGIN_CM=1.0 to recover the pre-study C++ floor.  The toolkit
    # qlmatching.jsonnet default stays null (byte-identical) -- this runner is
    # where PDVD processing turns the wider floor on.
    local QL_ANODEMARGIN_ARG=(-S "ql_anode_margin_cm=${PDVD_QL_ANODE_MARGIN_CM:-2.0}")
    # PDVD_QL_USE_SAT_FLAG: per-flash DAPHNE-rail channel masking in
    # QLMatching.  PRODUCTION DEFAULT ON since 2026-07-14 (keep-and-mark
    # operating point, docs/qlmatch/pdvd-saturation-recovery.md) -- needs a
    # flag_saturation light archive, which run_light_evt.sh now produces by
    # default.  Set PDVD_QL_USE_SAT_FLAG=0 for legacy (veto-era) archives;
    # the toolkit C++/jsonnet defaults stay OFF (byte-identical).
    # PDVD_QL_ROBUST_TRIM=1: robust-endpoint trim -- snap a drift endpoint back
    # inside the detector edge when the outer material is overclustered junk rather
    # than a real track end, before the containment gate reads it.  PRODUCTION
    # DEFAULT ON since 2026-07-17 (with WALK_FLOOR): the entire doc-19 scan-tuning
    # campaign (cathxa reference + tune_c2_cr operating point, 18 evts x ~20 tags)
    # ran with it, so the tuned defaults assume it.  PDVD_QL_ROBUST_TRIM=0
    # reverts (also disables WALK_FLOOR).  The trim thresholds below are PDHD's
    # proven operating point (cfg/.../pdhd/qlmatching.jsonnet:306-332); each is
    # overridable.
    #
    # Reproduces the evt298567 clus 34 demo: 8 points carrying 0.08% of that
    # cluster's charge sit 28.4 cm off its anode end and stretch its drift extent to
    # 371.7 cm, past the 336.9 cm drift depth, so no flash T0 can contain it and
    # every candidate bundle is dropped before the fit.  Only the GAP judge trims
    # them -- they are detached, not diffuse, so the density judge (CHARGE_ABS)
    # protects them like a genuine track tip.
    # See docs/qlmatch/15_pdvd-clus34-unmatched-evt298567.md.
    #
    # PDVD_QL_ROBUST_WALK_FLOOR=1 (needs PDVD_QL_ROBUST_TRIM=1): break the anode-end
    # walk at the floor the containment gate actually uses (anode_ext1 - margin, -4 cm)
    # instead of anode_ext1 (-2 cm), so it stops counting material the gate would have
    # accepted as "outside".  Without it the rescue depends on cluster SIZE -- evt298567
    # apa-0 ident 34 (2649 pts, allowance 26.5) is rescued but apa-4 ident 1 (1375 pts,
    # allowance 15) is not, on the same 20 swallowed body points.  DEFAULT ON
    # since 2026-07-17 (doc 19; PDVD_QL_ROBUST_WALK_FLOOR=0 reverts).
    local QL_ROBUSTGAP_ARG=()
    if [ "${PDVD_QL_ROBUST_TRIM:-1}" = 1 ]; then
        QL_ROBUSTGAP_ARG=(-S "ql_robust_trim=true"
                          -S "ql_robust_walk_to_floor=$([ "${PDVD_QL_ROBUST_WALK_FLOOR:-1}" = 1 ] && echo true || echo false)"
                          -S "ql_robust_gap_cathode=$([ "${PDVD_QL_ROBUST_GAP_CATHODE:-0}" = 1 ] && echo true || echo false)"
                          -S "ql_robust_frac=${PDVD_QL_ROBUST_FRAC:-0.01}"
                          -S "ql_robust_count=${PDVD_QL_ROBUST_COUNT:-15}"
                          -S "ql_robust_charge_frac=${PDVD_QL_ROBUST_CHARGE_FRAC:-0.005}"
                          -S "ql_robust_charge_abs=${PDVD_QL_ROBUST_CHARGE_ABS:-1500}"
                          -S "ql_robust_gap_cm=${PDVD_QL_ROBUST_GAP_CM:-3.0}"
                          -S "ql_robust_gap_charge_frac=${PDVD_QL_ROBUST_GAP_FRAC:-0.01}")
    fi
    # PDVD_QL_XTPC_CATHODE_TOL_CM (+ optional _QFRAC): xtpc cathode rescue -- admit
    # a cathode-crosser half that overshoots the containment gate (or falls short of
    # the at_cathode window) by up to TOL cm FOR XTPC PAIRING ONLY; kept only if
    # cull_cross_tpc confirms the pair (scenario 1, one half contained), purged
    # pre-fit otherwise.  QFRAC = charge fraction discardable as overclustered junk
    # when measuring the overshoot (evt298567 clus 97: 2.6% junk drags the raw
    # extreme 47 cm past the gate; the real end is 3.3 cm past).  Independent of
    # PDVD_QL_ROBUST_TRIM.  PRODUCTION DEFAULT 10 / 0.05 since 2026-07-17 (part
    # of the doc-19 tune_c2_cr operating point baseline; the phase-3
    # xtpc_cathode_ks_max=0.32 gate now guards its phantom mode, rescue-survivor
    # phantom rate 69%->9%).  Export PDVD_QL_XTPC_CATHODE_TOL_CM= (empty) to
    # disable (keys omitted).  Demo derivation doc 16 §10.
    : "${PDVD_QL_XTPC_CATHODE_TOL_CM=10}"
    local QL_XTPCRESCUE_ARG=()
    if [ -n "${PDVD_QL_XTPC_CATHODE_TOL_CM:-}" ]; then
        QL_XTPCRESCUE_ARG=(-S "ql_xtpc_cathode_tol_cm=${PDVD_QL_XTPC_CATHODE_TOL_CM}"
                           -S "ql_xtpc_cathode_qfrac=${PDVD_QL_XTPC_CATHODE_QFRAC:-0.05}")
    fi
    # PDVD_CC_TIP_TOUCH_CUT (+ optional _ANGLE): post-QLMatching cathode-crossing
    # STITCH tip-touch relaxation in the stage-4 ClusteringCathodeConnect (ports the
    # PDHD tip-touch branch, cfg/.../pdhd/clus.jsonnet).  When a genuine crosser's two
    # halves reach the cathode and nearly touch (~1cm gap), CUT (cm) drops the
    # cc_pca connection-alignment term and ANGLE (deg) accepts on the local
    # charge-weighted Hough even if a curved half inflates the global PCA above
    # angle_cut.  Only meaningful with QL matching (needs the matched cluster_t0).
    # PRODUCTION DEFAULT 3.0 cm / 12 deg since 2026-07-17: the 28-event census
    # (039252 x18 + 039349 x10, QL-matched) recovered a genuine cathode crosser in
    # 039252 evt298707 (halves touching at the cathode, 0.32 cm tip gap, 4.4 deg
    # collinear) with ZERO spurious merges in 28 events, matching PDHD's values.
    # NOT byte-identical (flips evt298707).  Export PDVD_CC_TIP_TOUCH_CUT= (empty)
    # to disable (keys omitted => legacy split).  Toolkit C++/jsonnet defaults stay
    # OFF.  See clus/docs/cathode-crossing-clustering.md section 6.1.
    : "${PDVD_CC_TIP_TOUCH_CUT=3.0}"
    local CC_TIPTOUCH_ARG=()
    if [ -n "${PDVD_CC_TIP_TOUCH_CUT:-}" ]; then
        CC_TIPTOUCH_ARG=(-S "cc_tip_touch_cut_cm=${PDVD_CC_TIP_TOUCH_CUT}"
                         -S "cc_tip_touch_angle_cut=${PDVD_CC_TIP_TOUCH_ANGLE:-12.0}")
    fi
    # PDVD_CC_DIS_CUT / _DRIFT_CUT / _CATHODE_X_CUT (all cm): enlarge the stage-4
    # ClusteringCathodeConnect distance gates for PDVD's ~6 cm-thick cathode.  A
    # genuine crosser's two halves stop at their own cathode face (~+-3 cm) and
    # meet 6-15 cm apart -- past the SBND-tuned dis_cut=5 / drift_cut=8 /
    # cathode_x_cut=5, so they land in the far regime whose connection-vector test
    # the SCE offset corrupts (owner: directions consistent, closest-point vector
    # offset).  Data-driven from the 151 QL-confirmed xtpc_pin pairs across 28
    # evts (dis p95=15, dX p95=12, tip|x| p95=6.6): dis_cut=16 makes 95% of
    # confirmed crossers CLOSE-regime, where the pass accepts on track collinearity
    # ALONE (no connection-vector test).  Empty => clus.jsonnet keeps 5/8/5 =>
    # byte-identical.  Toolkit C++/jsonnet defaults stay OFF.  Census tags cc1*.
    # ---- Cathode-crossing merge operating point cc3a (ADOPTED 2026-07-17).
    # Data-driven from the QL-confirmed cross-cathode crossers (120-event feature
    # study, docs/qlmatch/): PDVD's 6cm-thick cathode leaves each crosser half at
    # its own cathode face, so the SBND-tuned gates cannot span the gap.  The four
    # merge knobs recover cross-cathode crossers 76%->89% at ~0 spurious.  Runner
    # DEFAULTS them ON (override via env); the toolkit C++/jsonnet defaults stay OFF
    # so the compiled config is byte-identical when these are unset there.
    #   dis_cut 16 / drift_cut 14 / cathode_x_cut 8   (span the 6cm cathode)
    #   crosser_conn_relax 75   (SCE-noisy tip connection; relax not drop)
    #   crosser_pca_angle 15    (admit bent crossers on the reliable PCA direction)
    # cathode_band_dis stays OFF (near-cathode retry; validated NULL -- the band
    # tips' connection is ~perpendicular, indistinguishable from coincidences).
    local CC_DIST_ARG=()
    CC_DIST_ARG+=(-S "cc_dis_cut_cm=${PDVD_CC_DIS_CUT:-16}")
    CC_DIST_ARG+=(-S "cc_drift_cut_cm=${PDVD_CC_DRIFT_CUT:-14}")
    CC_DIST_ARG+=(-S "cc_cathode_x_cut_cm=${PDVD_CC_CATHODE_X_CUT:-8}")
    CC_DIST_ARG+=(-S "cc_crosser_conn_relax=${PDVD_CC_CROSSER_CONN_RELAX:-75}")
    CC_DIST_ARG+=(-S "cc_crosser_pca_angle=${PDVD_CC_CROSSER_PCA_ANGLE:-15}")
    # near-cathode closest-approach retry (cm); empty => OFF (validated null lever)
    [ -n "${PDVD_CC_CATHODE_BAND_DIS:-}" ] && CC_DIST_ARG+=(-S "cc_cathode_band_dis=${PDVD_CC_CATHODE_BAND_DIS}")
    # Per-drift-side raw-imaging Bee split (img-side-bot / img-side-top alongside
    # img-global), for making per-side video frames.  PDVD_BEE_IMG_PER_SIDE=1 =>
    # add the set; unset/0 => arg omitted => compiled config byte-identical.
    [ "${PDVD_BEE_IMG_PER_SIDE:-0}" = 1 ] && CC_DIST_ARG+=(-S "bee_img_per_side=true")
    # ---- Cathode-XA-anchored operating point (2026-07-16, docs/qlmatch/18_*.md).
    # Motivated by the evt298567 hand-scan PD-family study (doc 17): the cathode
    # XAs are the only PD family with full-stream readout and proportional
    # response; the wall XAs are bimodal (zero in 9/15 predicted-bright matches).
    # Toolkit C++/jsonnet defaults all stay OFF (byte-identical); this runner is
    # where PDVD production turns them on.  Each is individually revertible.
    #
    # PDVD_QL_MASK_WALL_XA (default 1): mask live membrane/wall XAs 0,1,3,12,18,19
    # out of the chi2/LASSO/KS (mask_ks already on).  =0 for the pre-study fit set.
    # M1 DEPLOY GATE: the wall channels must show pred_pe=0 in the calib dump
    # (dump prediction loop skips masked channels).
    local QL_CATHOP_ARG=()
    if [ "${PDVD_QL_MASK_WALL_XA:-1}" = 1 ]; then
        QL_CATHOP_ARG+=(-S "ql_mask_wall_xa=true")
    fi
    # PDVD_QL_WALL_FLAGS (default 0 = disabled): the +-y wall proximity flag
    # (vd_surface_flags wall component).  With the wall XAs masked the wall relax
    # channels protect nothing, and the flag only exempted wall-hugging junk from
    # the overpred prefilter; the hand scan showed it helped only marginally on
    # the (subdominant) wall PMTs.  =1 restores the wall flags.  The bottom-anode
    # proximity flag is untouched either way.
    if [ "${PDVD_QL_WALL_FLAGS:-0}" = 0 ]; then
        QL_CATHOP_ARG+=(-S "ql_wall_flags=false")
    fi
    # PDVD_QL_FLASH_SEL (default 1): cathode-scoped flash admission on top of the
    # all-PD ql_flash_minpe floor: sum(cathode PE) >= MINPE and >= MINFIRED cathode
    # channels at >= FIRED_PE PE.  Operating point 5/2/1.0 keeps 289/297 (97.3%)
    # of the keep-round confirmed candle flashes and cuts ~7% of admitted flashes
    # (the 8 lost candles are dim <=160 PE wall-huggers, two with zero cathode
    # signal).  M1 DEPLOY GATE:
    #   grep 'QLMatching flash_sel: admission over' work/<tag>/wct_clus_*.log
    if [ "${PDVD_QL_FLASH_SEL:-1}" = 1 ]; then
        QL_CATHOP_ARG+=(-S "ql_flash_sel_cathode=true"
                        -S "ql_flash_sel_minpe=${PDVD_QL_FLASH_SEL_MINPE:-5}"
                        -S "ql_flash_sel_min_fired=${PDVD_QL_FLASH_SEL_MINFIRED:-2}"
                        -S "ql_flash_sel_fired_pe=${PDVD_QL_FLASH_SEL_FIRED_PE:-1.0}")
    fi
    # PDVD_QL_REJECT_OVERPRED (default 1): cathode-scoped over-prediction prefilter,
    # R_total <= 15 / R_max <= 50 (18-event keep-round tuning: keeps every evt298567
    # full-scan match, max R 1.43/4.97; culls ~16% of non-exempt candidate bundles).
    # M1 DEPLOY GATE:
    #   grep 'QLMatching reject_overpred scoped to' work/<tag>/wct_clus_*.log
    if [ "${PDVD_QL_REJECT_OVERPRED:-1}" = 1 ]; then
        QL_CATHOP_ARG+=(-S "ql_reject_overpred=true"
                        -S "ql_overpred_total_ratio=${PDVD_QL_OVERPRED_TOTAL:-15}"
                        -S "ql_overpred_maxch_ratio=${PDVD_QL_OVERPRED_MAXCH:-50}")
    fi
    # ---- Per-PD-family PE-error overrides (scan-tuning docs/qlmatch/19_*.md).
    # PDVD_QL_PEERR_{CATH,PMT}_{FLOOR,FRAC,LOWPE_FRAC,LOWPE_KNEE}: override the
    # global pe_err model per family (cathode XAs 4-11 / PMTs) in BOTH error
    # paths (LASSO floor/frac + bundle-chi2 incl. lowpe branch).  Any set env
    # activates the family knob; unset members fall back to the global values.
    # PRODUCTION DEFAULTS since 2026-07-17 (doc 19 phase 2 pull calibration,
    # tune_c2_cr operating point): cathode-XA frac 0.55 / lowpe_frac 2.6 @
    # knee 80, PMT frac 0.75.  Export a member EMPTY (PDVD_QL_PEERR_CATH_FRAC=)
    # to fall back to the global model; toolkit defaults stay OFF.
    : "${PDVD_QL_PEERR_CATH_FRAC=0.55}"
    : "${PDVD_QL_PEERR_CATH_LOWPE_FRAC=2.6}"
    : "${PDVD_QL_PEERR_CATH_LOWPE_KNEE=80}"
    : "${PDVD_QL_PEERR_PMT_FRAC=0.75}"
    local QL_PEERR_ARG=()
    local _pe_env _pe_var
    for _pe_env in CATH_FLOOR CATH_FRAC CATH_LOWPE_FRAC CATH_LOWPE_KNEE \
                   PMT_FLOOR PMT_FRAC PMT_LOWPE_FRAC PMT_LOWPE_KNEE; do
        _pe_var="PDVD_QL_PEERR_${_pe_env}"
        if [ -n "${!_pe_var:-}" ]; then
            QL_PEERR_ARG+=(-S "ql_peerr_$(echo "$_pe_env" | tr '[:upper:]' '[:lower:]')=${!_pe_var}")
        fi
    done
    # ---- xtpc / selection quality gates (scan-tuning docs/qlmatch/19_*.md).
    # PDVD_QL_PIN_MIN_STRENGTH: pinned bundle loses the strength-cutoff exemption
    #   below this LASSO solution (scan: phantom pins strength p50 0.00).
    # PDVD_QL_SC1_LIGHT_GATE=1: xtpc consistent/scenario1 flags require the
    #   bundle's own light to pass (ks/c2n ceilings PDVD_QL_SC1_KS/_C2N).
    # PDVD_QL_CATHODE_KS_MAX: cathode-rescue survivors additionally need ks <= this.
    # PDVD_QL_POSTCULL=1: post-fit cull of unflagged selections failing
    #   PDVD_QL_POSTCULL_KS/_C2N.
    # PRODUCTION DEFAULTS since 2026-07-17 (doc 19 phase 3, tune_c2_cr
    # operating point): pin floor 0.02, sc1 light gate ON (C++ ks 0.3 /
    # c2n 50), cathode ks ceiling 0.32, postcull ON (C++ 0.30/20).  Revert:
    # PDVD_QL_PIN_MIN_STRENGTH= PDVD_QL_CATHODE_KS_MAX= (empty) and
    # PDVD_QL_SC1_LIGHT_GATE=0 PDVD_QL_POSTCULL=0.  Toolkit defaults stay OFF.
    : "${PDVD_QL_PIN_MIN_STRENGTH=0.02}"
    : "${PDVD_QL_CATHODE_KS_MAX=0.32}"
    local QL_QGATE_ARG=()
    if [ -n "${PDVD_QL_PIN_MIN_STRENGTH:-}" ]; then
        QL_QGATE_ARG+=(-S "ql_xtpc_pin_min_strength=${PDVD_QL_PIN_MIN_STRENGTH}")
    fi
    if [ "${PDVD_QL_SC1_LIGHT_GATE:-1}" = 1 ]; then
        QL_QGATE_ARG+=(-S "ql_xtpc_sc1_light_gate=true")
        [ -n "${PDVD_QL_SC1_KS:-}" ] && QL_QGATE_ARG+=(-S "ql_xtpc_sc1_ks_max=${PDVD_QL_SC1_KS}")
        [ -n "${PDVD_QL_SC1_C2N:-}" ] && QL_QGATE_ARG+=(-S "ql_xtpc_sc1_c2n_max=${PDVD_QL_SC1_C2N}")
    fi
    if [ -n "${PDVD_QL_CATHODE_KS_MAX:-}" ]; then
        QL_QGATE_ARG+=(-S "ql_xtpc_cathode_ks_max=${PDVD_QL_CATHODE_KS_MAX}")
    fi
    if [ "${PDVD_QL_POSTCULL:-1}" = 1 ]; then
        QL_QGATE_ARG+=(-S "ql_postcull_unflagged=true")
        [ -n "${PDVD_QL_POSTCULL_KS:-}" ] && QL_QGATE_ARG+=(-S "ql_postcull_ks_max=${PDVD_QL_POSTCULL_KS}")
        [ -n "${PDVD_QL_POSTCULL_C2N:-}" ] && QL_QGATE_ARG+=(-S "ql_postcull_c2n_max=${PDVD_QL_POSTCULL_C2N}")
        # PDVD_QL_POSTCULL_WTRUNC=1: window-truncated overprediction cull
        # (doc 23 phase 2) -- drop a selected window-truncated bundle whose
        # total pred/meas exceeds PDVD_QL_POSTCULL_WTRUNC_RATIO (xtpc pairs
        # protected; sat-dominated flashes exempt).  ADOPTED as default
        # 2026-07-17 (tag ac3, with the pin cull: agree 761->764, phantom
        # 138->118, missed 81->78, 0 agreed pairs touched).  Revert:
        # PDVD_QL_POSTCULL_WTRUNC=0.
        if [ "${PDVD_QL_POSTCULL_WTRUNC:-1}" = 1 ]; then
            QL_QGATE_ARG+=(-S "ql_postcull_wtrunc=true"
                           -S "ql_postcull_wtrunc_ratio_hi=${PDVD_QL_POSTCULL_WTRUNC_RATIO:-2.0}"
                           -S "ql_postcull_wtrunc_sat_frac=${PDVD_QL_POSTCULL_WTRUNC_SATFRAC:-0.5}")
        fi
        # PDVD_QL_POSTCULL_PIN=1: xtpc-pin overprediction cull (doc 23 phase
        # 2) -- drop a pinned selection whose total pred/meas exceeds
        # PDVD_QL_POSTCULL_PIN_RATIO (ratio-ONLY; ks gates on pins kill
        # legitimate geometric matches; sat-dominated flashes exempt).
        # ADOPTED as default 2026-07-17 (tag ac3, see wtrunc cull above).
        # Revert: PDVD_QL_POSTCULL_PIN=0.
        if [ "${PDVD_QL_POSTCULL_PIN:-1}" = 1 ]; then
            QL_QGATE_ARG+=(-S "ql_postcull_pin=true"
                           -S "ql_postcull_pin_ratio_hi=${PDVD_QL_POSTCULL_PIN_RATIO:-2.0}")
        fi
        # PDVD_QL_POSTCULL_EARLY=1: rescue blind-spot fix (doc 23 phase 1a) --
        # run the unflagged cull BEFORE the cluster rescues too, so a
        # postcull-doomed selection cannot hide its cluster from the rescue.
        # ADOPTED as default 2026-07-17 (tag ac1: agree 751->755, missed
        # 91->87, phantom flat 138, 0 pairs moved/removed, +74 additive
        # adoptions).  Revert: PDVD_QL_POSTCULL_EARLY=0.
        [ "${PDVD_QL_POSTCULL_EARLY:-1}" = 1 ] \
            && QL_QGATE_ARG+=(-S "ql_postcull_before_rescue=true")
    fi
    # ---- Sweepable ladder ceilings + LASSO regularization (doc 19 phase 4).
    # PDVD_QL_HC_{CLEAN,GOOD,TB,MISS}_{KS,C2N}, PDVD_QL_HC_MISS_MIN_NDF,
    # PDVD_QL_LASSO_LAMBDA, PDVD_QL_DELTA_{CHARGE,LIGHT,SHAPE},
    # PDVD_QL_BKG_WEIGHT, PDVD_QL_STRENGTH_CUTOFF, PDVD_QL_LASSO_BWEIGHT.
    # PRODUCTION DEFAULTS since 2026-07-17 (doc 19 phase 4 sweep, tune_c2_cr
    # operating point): high-consistent chi2/ndf ceilings 12 (miss branch 30)
    # retightened to the phase-2 recalibrated chi2 scale, LASSO lambda 0.2
    # (sparser solutions, best phantom kill).  Export EMPTY (e.g.
    # PDVD_QL_LASSO_LAMBDA=) to recover the pre-tuning literals (35/60, 0.1).
    # Other members unset => operating literals / C++ defaults unchanged.
    : "${PDVD_QL_HC_CLEAN_C2N=12}"
    : "${PDVD_QL_HC_GOOD_C2N=12}"
    : "${PDVD_QL_HC_TB_C2N=12}"
    : "${PDVD_QL_HC_MISS_C2N=30}"
    : "${PDVD_QL_LASSO_LAMBDA=0.2}"
    local QL_SWEEP_ARG=()
    [ -n "${PDVD_QL_HC_CLEAN_KS:-}" ]  && QL_SWEEP_ARG+=(-S "ql_hc_clean_ks=${PDVD_QL_HC_CLEAN_KS}")
    [ -n "${PDVD_QL_HC_CLEAN_C2N:-}" ] && QL_SWEEP_ARG+=(-S "ql_hc_clean_c2=${PDVD_QL_HC_CLEAN_C2N}")
    [ -n "${PDVD_QL_HC_GOOD_KS:-}" ]   && QL_SWEEP_ARG+=(-S "ql_hc_good_ks=${PDVD_QL_HC_GOOD_KS}")
    [ -n "${PDVD_QL_HC_GOOD_C2N:-}" ]  && QL_SWEEP_ARG+=(-S "ql_hc_good_c2=${PDVD_QL_HC_GOOD_C2N}")
    [ -n "${PDVD_QL_HC_TB_KS:-}" ]     && QL_SWEEP_ARG+=(-S "ql_hc_tb_ks=${PDVD_QL_HC_TB_KS}")
    [ -n "${PDVD_QL_HC_TB_C2N:-}" ]    && QL_SWEEP_ARG+=(-S "ql_hc_tb_c2=${PDVD_QL_HC_TB_C2N}")
    [ -n "${PDVD_QL_HC_MISS_KS:-}" ]   && QL_SWEEP_ARG+=(-S "ql_hc_miss_ks=${PDVD_QL_HC_MISS_KS}")
    [ -n "${PDVD_QL_HC_MISS_C2N:-}" ]  && QL_SWEEP_ARG+=(-S "ql_hc_miss_c2=${PDVD_QL_HC_MISS_C2N}")
    [ -n "${PDVD_QL_HC_MISS_MIN_NDF:-}" ] && QL_SWEEP_ARG+=(-S "ql_hc_miss_min_ndf=${PDVD_QL_HC_MISS_MIN_NDF}")
    [ -n "${PDVD_QL_LASSO_LAMBDA:-}" ]  && QL_SWEEP_ARG+=(-S "ql_lasso_lambda=${PDVD_QL_LASSO_LAMBDA}")
    [ -n "${PDVD_QL_DELTA_CHARGE:-}" ]  && QL_SWEEP_ARG+=(-S "ql_delta_charge=${PDVD_QL_DELTA_CHARGE}")
    [ -n "${PDVD_QL_DELTA_LIGHT:-}" ]   && QL_SWEEP_ARG+=(-S "ql_delta_light=${PDVD_QL_DELTA_LIGHT}")
    [ -n "${PDVD_QL_DELTA_SHAPE:-}" ]   && QL_SWEEP_ARG+=(-S "ql_delta_shape=${PDVD_QL_DELTA_SHAPE}")
    [ -n "${PDVD_QL_BKG_WEIGHT:-}" ]    && QL_SWEEP_ARG+=(-S "ql_bkg_weight=${PDVD_QL_BKG_WEIGHT}")
    [ -n "${PDVD_QL_STRENGTH_CUTOFF:-}" ] && QL_SWEEP_ARG+=(-S "ql_strength_cutoff=${PDVD_QL_STRENGTH_CUTOFF}")
    [ -n "${PDVD_QL_LASSO_BWEIGHT:-}" ] && QL_SWEEP_ARG+=(-S "ql_lasso_boundary_weight=${PDVD_QL_LASSO_BWEIGHT}")
    # ---- Shared-flash-aware rescues (doc 19 phase 5).
    # PDVD_QL_EMPTY_RESCUE=1 (+PDVD_QL_RESCUE_METRIC_MAX): joint-emptiness
    #   empty-flash rescue across drift sides.
    # PDVD_QL_CLUSTER_RESCUE=1 (+PDVD_QL_CRESCUE_{KS,C2N,RLO,RHI}): per-run
    #   cluster-centric adoption (gates REQUIRED — C++ 0 gates are inert).
    # CLUSTER RESCUE = PRODUCTION DEFAULT since 2026-07-17 (doc 19 phase 5,
    # tune_c2_cr: missed 120->109 at +1 phantom); PDVD_QL_CLUSTER_RESCUE=0
    # reverts.  EMPTY RESCUE stays opt-in: on 039252 it force-fills
    # legitimately-empty flashes and steals correct matches at metric 0.5
    # AND 0.1 (doc 19 phase 5 verdict) — do not enable without a rescan.
    local QL_RESCUE_ARG=()
    if [ "${PDVD_QL_EMPTY_RESCUE:-0}" = 1 ]; then
        QL_RESCUE_ARG+=(-S "ql_empty_rescue_shared=true")
        [ -n "${PDVD_QL_RESCUE_METRIC_MAX:-}" ] && QL_RESCUE_ARG+=(-S "ql_rescue_metric_max=${PDVD_QL_RESCUE_METRIC_MAX}")
    fi
    # CLUSTER-RESCUE OPERATING POINT (nm3), PRODUCTION DEFAULT since 2026-07-17
    # (doc 20, run 039252 non-match census): additive precull + TIGHT gates.
    # Effect vs the pre-nm3 baseline (snapshot pool, loose .25/15/.3-3): 36 fewer
    # non-matched long tracks in Bee (missed 109->95 on the scanned set, -13 %),
    # phantom flat 137, +19 unlabeled new matches (rescan-gated).  Adopted by owner
    # request 2026-07-17 after the nm0-vs-nm3 decision-number comparison (doc 20 §7).
    # NOT byte-identical (flips output); the toolkit C++/jsonnet defaults stay OFF.
    # Revert to the pre-nm3 baseline point:
    #   PDVD_QL_CRESCUE_PRECULL=0 PDVD_QL_CRESCUE_KS=0.25 PDVD_QL_CRESCUE_C2N=15 \
    #   PDVD_QL_CRESCUE_RLO=0.3 PDVD_QL_CRESCUE_RHI=3.0
    # Config-only fallback (no additive C++ lib): PDVD_QL_CRESCUE_PRECULL_ADD=0
    # (= nm2a, one cluster worse, keeps the two gross wrong-flash switches).
    if [ "${PDVD_QL_CLUSTER_RESCUE:-1}" = 1 ]; then
        QL_RESCUE_ARG+=(-S "ql_cluster_rescue_shared=true"
                        -S "ql_cluster_rescue_ks_max=${PDVD_QL_CRESCUE_KS:-0.15}"
                        -S "ql_cluster_rescue_c2n_max=${PDVD_QL_CRESCUE_C2N:-3}"
                        -S "ql_cluster_rescue_ratio_lo=${PDVD_QL_CRESCUE_RLO:-0.5}"
                        -S "ql_cluster_rescue_ratio_hi=${PDVD_QL_CRESCUE_RHI:-2.0}")
        # PDVD_QL_CRESCUE_PRECULL: draw the rescue pool from the pre-cull
        # all_bundles universe so cull_inconsistent victims become reachable
        # (doc 20).  PRODUCTION DEFAULT 1 as of nm3 adoption (2026-07-17); the
        # toolkit jsonnet default stays false (key omitted when off => byte-identical).
        [ "${PDVD_QL_CRESCUE_PRECULL:-1}" = 1 ] \
            && QL_RESCUE_ARG+=(-S "ql_cluster_rescue_precull=true")
        # PDVD_QL_CRESCUE_PRECULL_ADD: additive precull -- snapshot pool primary,
        # pre-cull pool fallback-only, so precull adds without re-deciding/mis-switching
        # already-rescued clusters (doc 20).  PRODUCTION DEFAULT 1 as of nm3 adoption
        # (2026-07-17); toolkit C++ default stays false (key omitted => byte-identical).
        [ "${PDVD_QL_CRESCUE_PRECULL:-1}" = 1 ] && [ "${PDVD_QL_CRESCUE_PRECULL_ADD:-1}" = 1 ] \
            && QL_RESCUE_ARG+=(-S "ql_cluster_rescue_precull_additive=true")
        # PDVD_QL_CRESCUE_RELAX (+PDVD_QL_CRESCUE_RELAX_{KS,C2N,RLO,RHI,MINLEN_CM}):
        # relaxed SECOND-CHANCE tier (doc 21) -- clusters the tight gates above
        # leave unmatched get one more accept() with these relaxed gates,
        # restricted to LONG clusters (>= MINLEN_CM).  Additive-only (existing
        # matches can never change, verified empirically on all nm4* tags);
        # adoptions carry the low-confidence `cluster_rescue_relaxed` dump flag.
        # RELAXED TIER (nm4b) = PRODUCTION DEFAULT since 2026-07-17 (doc 21 §5,
        # owner adoption): gates .25/15/.3-3.0 @ 50 cm = the nm4b sweep winner
        # (missed 95->91 on the scanned 18-evt set; 26 adoptions = 4 provably
        # right / 7 provably wrong / 15 rescan-only, ALL flagged in the dump).
        # NOT byte-identical (adds matches); toolkit C++/jsonnet defaults stay
        # OFF.  Revert: PDVD_QL_CRESCUE_RELAX=0.  The nm4a PDHD-gate point
        # (.20/8/.4-2.5) underperformed (1 recovered / 5 wrong) -- see doc 21.
        if [ "${PDVD_QL_CRESCUE_RELAX:-1}" = 1 ]; then
            QL_RESCUE_ARG+=(-S "ql_cluster_rescue_relaxed=true"
                            -S "ql_cluster_rescue_relaxed_ks_max=${PDVD_QL_CRESCUE_RELAX_KS:-0.25}"
                            -S "ql_cluster_rescue_relaxed_c2n_max=${PDVD_QL_CRESCUE_RELAX_C2N:-15}"
                            -S "ql_cluster_rescue_relaxed_ratio_lo=${PDVD_QL_CRESCUE_RELAX_RLO:-0.3}"
                            -S "ql_cluster_rescue_relaxed_ratio_hi=${PDVD_QL_CRESCUE_RELAX_RHI:-3.0}"
                            -S "ql_cluster_rescue_relaxed_minlen_cm=${PDVD_QL_CRESCUE_RELAX_MINLEN_CM:-50}")
        fi
        # PDVD_QL_CRESCUE_SATRELAX=1: saturation-aware ratio-high extension
        # (doc 23 phase 1b).  On a flash whose railed channels carry more than
        # PDVD_QL_CRESCUE_SAT_FRAC of the measured PE the measurement is a
        # lower bound, so both rescue tiers accept pred/meas up to
        # ratio_hi * PDVD_QL_CRESCUE_SAT_MULT.  ADOPTED as default 2026-07-17
        # (tag ac2 vs ac1: agree 755->761, missed 87->81, phantom flat 138;
        # all 6 changed pairs verified at scan truth times).  Revert:
        # PDVD_QL_CRESCUE_SATRELAX=0.
        if [ "${PDVD_QL_CRESCUE_SATRELAX:-1}" = 1 ]; then
            QL_RESCUE_ARG+=(-S "ql_cluster_rescue_sat_relax=true"
                            -S "ql_cluster_rescue_sat_frac_min=${PDVD_QL_CRESCUE_SAT_FRAC:-0.5}"
                            -S "ql_cluster_rescue_sat_ratio_mult=${PDVD_QL_CRESCUE_SAT_MULT:-2.0}")
        fi
    fi
    local QL_SATFLAG_ARG=()
    if [ "${PDVD_QL_USE_SAT_FLAG:-1}" = 1 ]; then
        QL_SATFLAG_ARG=(-S "ql_use_saturation_flag=true")
    fi
    # PDVD_QL_USE_COV_FLAG: per-flash readout-coverage masking in QLMatching
    # (self-trigger channels with no snippet over the flash window carry NO
    # data).  PRODUCTION DEFAULT ON since 2026-07-14
    # (docs/qlmatch/pdvd-lightpattern-sp-investigation.md) -- graceful no-op
    # on archives without the flash_cov tensor (get_cov == 1 everywhere).
    # Toolkit C++/jsonnet defaults stay OFF (byte-identical).
    if [ "${PDVD_QL_USE_COV_FLAG:-1}" = 1 ]; then
        QL_SATFLAG_ARG+=(-S "ql_use_coverage_flag=true")
    fi
    # PDVD_QL_COV_MASK_FIT: whether an uncovered channel is also DROPPED from
    # the fit, or kept at its measured 0 and merely labelled "nodata".
    # PRODUCTION DEFAULT 0 (keep) since 2026-07-16: the drop assumed silence
    # meant "no measurement", but DAPHNE has no dead time (min inter-snippet
    # gap 0.336 us over 18 evts of run 039252) and the self-trigger threshold
    # is ~1 PE, so silence measures "< ~1 PE" -- dropping it lets a prediction
    # over-shoot a blind channel for free.  See doc 14 section 12 and
    # docs/qlmatch/scripts/analyze_coverage_deadtime.py.  Set
    # PDVD_QL_COV_MASK_FIT=1 for the 2026-07-14..16 (_spcov / _am2) behaviour.
    # Toolkit C++/jsonnet defaults stay at the drop (byte-identical).
    #
    # M1 DEPLOY GATE: a toolkit that is pulled but NOT rebuilt compiles this
    # key into the config and then ignores it (old lib has no
    # m_coverage_mask_fit) => the masking silently stays ON and the run looks
    # fine.  On the first event of any reprocess, confirm the sentinel:
    #   grep 'coverage_mask_fit=false => uncovered channels stay in the fit' \
    #        work/<tag>/wct_clus_*.log
    # No line => the new libWireCellMatch.so is not live; wcbuild first.
    if [ "${PDVD_QL_COV_MASK_FIT:-0}" = 0 ]; then
        QL_SATFLAG_ARG+=(-S "ql_coverage_mask_fit=false")
    fi
    # PDVD_QL_SAT_MASK_FIT: whether a DAPHNE-rail-flagged channel is also
    # DROPPED from the chi2/KS, or kept there at its clipped PE and merely
    # labelled "sat".  PRODUCTION DEFAULT 0 (keep) since 2026-07-16, for the
    # same reason as COV_MASK_FIT above: the clipped PE is a LOWER BOUND on the
    # true light (doc 11 section 2.1: `clip` is a tight deterministic
    # underestimate, and this chain also repairs it via PDVD_SAT_REPAIR=1), not
    # a missing measurement -- dropping it makes a WRONG prediction free on
    # exactly the bright cathode channels that anchor Q/L matching.  The LASSO
    # rows stay zeroed either way (a lower bound must not pull fitted charge
    # down).  Set PDVD_QL_SAT_MASK_FIT=1 for the 2026-07-14..16 (_spcov / _am2)
    # behaviour.  Toolkit C++/jsonnet defaults stay at the drop
    # (byte-identical).  See doc 11 section 6.
    #
    # M1 DEPLOY GATE (as above; old lib has no m_saturation_mask_fit):
    #   grep 'saturation_mask_fit=false => railed channels stay in the chi2/KS' \
    #        work/<tag>/wct_clus_*.log
    if [ "${PDVD_QL_SAT_MASK_FIT:-0}" = 0 ]; then
        QL_SATFLAG_ARG+=(-S "ql_saturation_mask_fit=false")
        # PDVD_QL_CHI2_SAT_INFLATE: extra chi2 width on a railed channel,
        # denom += (pe*inflate)^2 -- the chi2_pmt_inflate form the near-PMT
        # over-response already uses, gated on the rail flag alone.  REQUIRED
        # whenever the channel is kept: measured PE on a railed channel of a
        # SELECTED match runs ~3.7x the prediction (evt298567: 100 railed
        # terms, median meas 4179 vs pred 1212), because the repair
        # over-recovers AND the photon model under-predicts the bright cathode
        # -- at inflate 0 those terms are median 19 / p90 4008 / max 14601 and
        # would brand good matches inconsistent, i.e. WORSE than the masking
        # this replaces.  0.5 (the chi2_pmt_inflate default) puts them at
        # median 1.9 / p90 4.0.  See doc 11 section 6.
        QL_SATFLAG_ARG+=(-S "ql_chi2_sat_inflate=${PDVD_QL_CHI2_SAT_INFLATE:-0.5}")
    fi
    wcsonnet \
        -A "input=${CLUS_INPUT}" \
        -S "anode_indices=${ANODE_CODE}" \
        -A "output_dir=${WORKDIR}" \
        -S "run=${RUN_STRIPPED}" \
        -S "subrun=${SUBRUN}" \
        -S "event=${EVENT_NO}" \
        -S "stepped_center_fallback=${PDVD_STEPPED_CENTER_FALLBACK:-false}" \
        -S "do_qlmatch=$([ "$QLMATCH_EVT" = 1 ] && echo true || echo false)" \
        -A "opflash_input=${OPFLASH_TAR}" \
        -S "calib=$([ "$CALIB" = 1 ] && echo true || echo false)" \
        -S "save_opflash=$([ "$OPDUMP" = 1 ] && [ "$QLMATCH_EVT" = 1 ] && echo true || echo false)" \
        -S "trigger_offset_bot_us=${TRIGGER_OFFSET_BOT_US:-0}" \
        -S "trigger_offset_top_us=${TRIGGER_OFFSET_TOP_US:-0}" \
        -S "readout_window_ticks=${READOUT_NTICKS:-10000}" \
        -A "light_model=${PDVD_LIGHT_MODEL:-library}" \
        -S "ql_require_containment=${QL_CONTAIN}" \
        -S "ql_flash_minpe=${QL_MINPE}" \
        -S "drift_speed_bot_mmus=${PDVD_DRIFT_SPEED_BOT_MMUS:-1.48073}" \
        -S "drift_speed_top_mmus=${PDVD_DRIFT_SPEED_TOP_MMUS:-1.48073}" \
        "${QL_CATHEXT1_ARG[@]}" \
        "${QL_ANODEMARGIN_ARG[@]}" \
        "${QL_ROBUSTGAP_ARG[@]}" \
        "${QL_XTPCRESCUE_ARG[@]}" \
        "${QL_CATHOP_ARG[@]}" \
        "${QL_PEERR_ARG[@]}" \
        "${QL_QGATE_ARG[@]}" \
        "${QL_SWEEP_ARG[@]}" \
        "${QL_RESCUE_ARG[@]}" \
        "${QL_SATFLAG_ARG[@]}" \
        "${CC_TIPTOUCH_ARG[@]}" \
        "${CC_DIST_ARG[@]}" \
        -o "$CFG_JSON" wct-clustering.jsonnet
    if [ ! -s "$CFG_JSON" ]; then
        echo "ERROR: wcsonnet failed to compile wct-clustering.jsonnet" >&2
        return 1
    fi
    # Compile-only mode (compiled-config proofs): write CFG_JSON and stop before
    # running wire-cell.  Default unset => full run.
    if [ "${PDVD_CLUS_COMPILE_ONLY:-0}" = 1 ]; then
        echo "[compile-only] wrote $CFG_JSON"
        return 0
    fi
    # Resource recording (additive; does not change reco output, disable with
    # PDVD_RESMON=off): run wire-cell in the background and sample its
    # /proc/<pid>/status every 2 s.  Writes a per-event summary
    # (clus_resource_<run>_<evt>.txt: wall_s + peak_rss_gb) and the full
    # RSS-vs-wallclock trace (clus_rss_<run>_<evt>.csv), whose timestamps align
    # with the debug-log stage markers so the peak can be attributed to a stage.
    # Ported from pdhd/run_clus_evt.sh (PDHD_RESMON).
    local RES_TXT="$WORKDIR/clus_resource_${RUN_PADDED}_${EVT}${TAG_SUFFIX}.txt"
    local RES_CSV="$WORKDIR/clus_rss_${RUN_PADDED}_${EVT}${TAG_SUFFIX}.csv"
    local _t0=$SECONDS _smpid=""
    env $WC_PRELOAD GOGC=off wire-cell \
        -l stderr \
        -l "${LOG}:debug" \
        -L debug \
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
    awk -v r="$RUN_PADDED" -v e="$EVT" -v w="$_wall" -v p="$_peak_kb" \
        -v q="$QLMATCH_EVT" -v c="$CALIB" -v o="$OPDUMP" 'BEGIN{
        printf "run=%s evt=%s wall_s=%d peak_rss_gb=%.2f flags=q%s,calib%s,op%s\n",
               r,e,w,p/1048576,q,c,o}' | tee "$RES_TXT"
    [ "${PDVD_KEEP_CFG:-0}" = 1 ] || rm -f "$CFG_JSON"
    if [ "$_rc" -ne 0 ]; then
        echo "ERROR: wire-cell exited with status $_rc" >&2
        return "$_rc"
    fi

    echo "Clustering done -> $WORKDIR"
}

mkdir -p "$PDVD_DIR/work"
if [ "$EVT" = "all" ]; then
    batch_init
    mapfile -t _events < <(discover_events "$RUN" "$RUN_PADDED")
    if [ ${#_events[@]} -eq 0 ]; then
        echo "no events found for run=$RUN under input_data/ or work/" >&2; exit 1
    fi
    echo "Found ${#_events[@]} event(s) for run=$RUN: ${_events[*]}"
    echo "Parallel jobs: $BATCH_MAX"
    for _e in "${_events[@]}"; do
        _blogfile="$PDVD_DIR/work/.batch_clus_${RUN_PADDED}_${_e}.log"
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
