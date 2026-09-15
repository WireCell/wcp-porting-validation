#!/bin/bash
# doc pdvd/101 Phase 2 -- one known-truth muon event through sim -> NF -> SP -> imaging ->
# clustering (stops after clustering; the PR step is run later by the owner).
#
#   DET=pdvd|pdhd K=<event index> [TAG=d101] [STAGES="sim img clus"] [TIME_OFFSET_NS=]
#       [SIMROOT=/home/xqian/tmp/d101/sim] [PIN=/home/xqian/tmp/d101/libpin]
#       d101_sim_chain.sh
#
# Inputs : $SIMROOT/truth_<det>.json from d101_make_muons.py (event k carries the tracks TLA,
#          the anode and the simulation seed).
# Stages :
#   sim   <det>_sim/wct-sim-xtrack-sp.jsonnet (doc 47 driver, production-like defaults: noise,
#         fluctuation, NF, production SP; lifetime 1e4 ms) COMPILED with wcsonnet to
#         $SIMROOT/<det>/evt<k>/cfg/sim.json, provenance read from the compiled JSON, then
#         `wire-cell -c` -> evt<k>-anode<N>-{sp,raw}.tar.bz2.  Reused (not re-run) when a
#         previous sim of this k finished rc=0 (the scratch sim dir is TAG-independent).
#   img   copy the SP archive to <det>/work/<RUN6>_<k>_<TAG>/input/<sp-name>-anode<N>.tar.bz2
#         and run <det>/run_img_evt.sh -d off -a <N> -s <TAG> <RUN> <k>  (the selection-tag
#         path: a synthetic run has no input_data/ dir, which the plain path requires).
#   clus  <det>/run_clus_evt.sh -a <N> -s <TAG> -noq -noop -save-pctree -save-assoc <RUN> <k>
#         with the drift speed / readout window of the SIMULATION (from the compiled sim
#         config): PDVD_DRIFT_SPEED_{BOT,TOP}_MMUS and PDVD_READOUT_NTICKS (the PDVD runner
#         defaults are the data calibration 1.48073 mm/us and 10000 ticks); PDHD's clus.jsonnet
#         hard-codes 1.576 and -noq writes 6000 ticks -- both are ASSERTED equal to the sim.
#         TIME_OFFSET_NS goes in as <DET>_CLUS_TLA="-S time_offset=<ns>" (BlobSampler +
#         DetectorVolumes: x = x_W + dirx*(t_slice + time_offset)*v).  Unset = the value DERIVED
#         for this simulation (below); "none" = key not passed (runner default 0, the pilot).
#
# The time offset (measured on the pilot, TAG=d101pilot k=0,15, time_offset=0: the clustered
# points sat +35.4..+35.8 cm (PDVD) / +39.5..+39.7 cm (PDHD) too far from the anode at a
# constant, slope-free offset).  The slice starts written by imaging carry NO frame time
# (clusters-apa-*-ms-active snodes start 1284000 ns on PDVD k=0), while the sim frame's
# tickinfo time is t_frame = ductor start_time + reframer tbin*tick = -249.93 us (PDVD) /
# -249.95 us (PDHD).  Three terms, all read from the compiled sim config / wires file:
#   time_offset = t_frame - ctoffset(SP) - (R_FR - (x_response - x_W))/v
#   PDVD: -249933.67 - 4000 - (181.0 - 223.863 mm)/1.568e-3 = -226598 ns
#         (the Drifter response plane of params det.volumes sits 223.86 mm from the v7 W
#          plane, while the ductor/FR time assumes 181 mm: sim charge arrives as if 4.29 cm
#          nearer the anode; with t_frame alone PDVD stays at -3.4..-3.8 cm)
#   PDHD: -249951.78 - 1000 - (100.0 - 99.555 mm)/1.576e-3 = -251234 ns
# On the pilot pctrees this leaves +0.22/-0.11 cm (PDVD) and -0.11/+0.08 cm (PDHD).  The
# clus stage REFUSES the derived value if the compiled sim config no longer has the
# t_frame / ctoffset / drift speed / response x it was derived from.
# The PR job (wct-pr-perevt.jsonnet) takes time_offset in MICROSECONDS and run_pr_evt.sh does
# not read it from the .tlas (PDHD's .tlas even hard-codes time_offset=0): pass
# <DET>_PR_TLA="-S time_offset=<us>" -- the value is in <workdir>/d101-time-offset.txt.
# The WORK dir must not exist when the img stage starts (never overwrite a record, M13).
# Every wire-cell/wcsonnet resolves from the pin (a peer's wcbuild must not swap the binary).
# Outputs: $SIMROOT/<det>/evt<k>/{stages.tsv, provenance.txt, *.out, DONE | FAIL_<stage>}
set -u
DET=${DET:?DET=pdvd|pdhd}
K=${K:?K=event index}
TAG=${TAG:-d101}
STAGES=${STAGES:-"sim img clus"}
SIMROOT=${SIMROOT:-/home/xqian/tmp/d101/sim}
PIN=${PIN:-/home/xqian/tmp/d101/libpin}
TIME_OFFSET_NS=${TIME_OFFSET_NS:-derived}
WCT=/home/xqian/toolkit-dev
WCP=$WCT/wcp-porting-img
TRUTH=${TRUTH:-$SIMROOT/truth_$DET.json}
case $DET in
    pdvd) RUN=900101; V=""; SPNAME=protodune-sp-frames; DETDIR=$WCP/pdvd
          DERIVED_TOFF_NS=-226598; EXP="frame_t0_ns=-249933.673 sp_ctoffset_ns=4000 drift_speed_mmus=1.568 response_x_mm=-3191.6375" ;;
    pdhd) RUN=900102; V="-V elecGain=14"; SPNAME=protodunehd-sp-frames; DETDIR=$WCP/pdhd
          DERIVED_TOFF_NS=-251234; EXP="frame_t0_ns=-249951.777 sp_ctoffset_ns=1000 drift_speed_mmus=1.576 response_x_mm=3430.465" ;;
    *) echo "DET must be pdvd|pdhd" >&2; exit 2 ;;
esac
SIMDIR=$WCP/${DET}_sim
# the runners PREPEND toolkit/cfg:wire-cell-data to the inherited WIRECELL_PATH; hand them the same two
RUNNER_WCP=$WCT/toolkit/cfg:$WCT/wire-cell-data
EV=$SIMROOT/$DET/evt$K
WORKDIR=$DETDIR/work/$(printf '%06d' $RUN)_${K}_${TAG}
mkdir -p "$EV/cfg"
[ -d "$PIN" ] && [ -x "$PIN/bin/wire-cell" ] || { echo "no pin $PIN" >&2; exit 2; }
export LD_LIBRARY_PATH=$PIN
export PATH=$PIN/bin:$PATH
if ldd "$PIN/bin/wire-cell" | grep -i wirecell | grep -qv "$PIN"; then
    echo "REFUSING: wire-cell libs not resolved from $PIN" >&2; exit 2
fi
[ "$(command -v wire-cell)" = "$PIN/bin/wire-cell" ] || { echo "REFUSING: wire-cell in PATH is $(command -v wire-cell)" >&2; exit 2; }

# ---- event record from the truth file
read -r ANODE SEED < <(python3 - "$TRUTH" "$K" "$EV/tracks.json" <<'PY'
import json, sys
t = json.load(open(sys.argv[1])); k = int(sys.argv[2])
e = next(x for x in t["events"] if x["k"] == k)
json.dump(e["tracks"], open(sys.argv[3], "w"))
print(t["anode"], e["sim_seed"])
PY
)
[ -n "${ANODE:-}" ] && [ -n "${SEED:-}" ] || { echo "event $K not in $TRUTH" >&2; exit 2; }

stage_log() { echo -e "$1\t$2\t$3\t$(date -Is)" >> "$EV/stages.tsv"; }
fail() { touch "$EV/FAIL_$1"; echo "FAIL stage=$1 det=$DET k=$K (see $EV)" >&2; exit 1; }
rm -f "$EV/DONE"

# ---- sim
if [[ " $STAGES " == *" sim "* ]]; then
    SP=$EV/evt$K-anode$ANODE-sp.tar.bz2
    if [ -f "$EV/sim.rc" ] && [ "$(cat "$EV/sim.rc")" = 0 ] && [ -s "$SP" ]; then
        echo "sim: reusing $SP (sim.rc=0)"
    else
        t0=$SECONDS
        (cd "$SIMDIR" && WIRECELL_PATH=$SIMDIR:$WCT/toolkit/cfg:$WCT/wire-cell-data wcsonnet $V \
            --tla-code "tracks=$(cat "$EV/tracks.json")" --tla-code anode_index=$ANODE \
            --tla-code seed=$SEED --tla-str output_prefix="$EV/evt$K" \
            -o "$EV/cfg/sim.json" wct-sim-xtrack-sp.jsonnet) > "$EV/cfg/sim.compile.log" 2>&1
        rc=$?
        if [ $rc -ne 0 ] || [ ! -s "$EV/cfg/sim.json" ]; then stage_log sim_compile $rc $((SECONDS-t0)); fail sim_compile; fi
        rm -f "$EV/evt$K-anode$ANODE-"*.tar.bz2 "$EV/FAIL_"*
        (cd "$SIMDIR" && WIRECELL_PATH=$SIMDIR:$WCT/toolkit/cfg:$WCT/wire-cell-data \
            wire-cell -l "$EV/sim.log:debug" -L debug -c "$EV/cfg/sim.json") > "$EV/sim.out" 2>&1
        rc=$?
        echo $rc > "$EV/sim.rc"
        stage_log sim $rc $((SECONDS-t0))
        [ $rc -eq 0 ] && [ -s "$SP" ] || fail sim
    fi
    # compiled-config provenance (never from jsonnet defaults)
    python3 - "$EV/cfg/sim.json" "$SEED" > "$EV/provenance.txt" <<'PY' || fail provenance
import json, sys
cfg = json.load(open(sys.argv[1])); by = {}
for n in cfg: by.setdefault(n.get("type"), []).append(n)
dr = by["Drifter"][0]["data"]; dt = by["DepoTransform"][0]["data"]; rf = by["Reframer"][0]["data"]
seeds = by["Random"][0]["data"]["seeds"]
assert seeds[0] == int(sys.argv[2]), ("seed not in compiled config", seeds)
t0 = dt["start_time"] + rf["tbin"] * dt["tick"]
print("DL_cm2_s=%.6g" % (dr["DL"] * 1e7)); print("DT_cm2_s=%.6g" % (dr["DT"] * 1e7))
print("drift_speed_mmus=%.6g" % (dr["drift_speed"] * 1e3)); print("lifetime_ms=%.6g" % (dr["lifetime"] / 1e6))
print("fluctuate=%s" % dr.get("fluctuate")); print("tick_ns=%g" % dt["tick"])
print("nticks=%d" % rf["nticks"]); print("ductor_start_time_ns=%.6f" % dt["start_time"])
print("reframer_tbin=%d" % rf["tbin"]); print("frame_t0_ns=%.6f" % t0)
print("nsigma=%s" % dt.get("nsigma")); print("first_frame_number=%s" % dt.get("first_frame_number"))
print("seeds=%s" % seeds); print("addnoise_nodes=%d" % len(by.get("AddNoise", [])))
print("wires=%s" % sorted({n["data"]["filename"] for n in by["WireSchemaFile"]})[0])
print("xregions=%s" % json.dumps(dr["xregions"]))
print("sp_node=%s nf_node=%s" % (len(by.get("OmnibusSigProc", [])), len(by.get("OmnibusNoiseFilter", []))))
osp = by["OmnibusSigProc"][0]["data"]
print("sp_ctoffset_ns=%g" % osp.get("ctoffset", 0)); print("sp_ftoffset_ns=%g" % osp.get("ftoffset", 0))
anode = by["AnodePlane"][0]["data"]
# the drift-volume face nearest the cathode plane x=0 (PDHD: the interior face; PDVD: both faces share it)
print("response_x_mm=%.4f" % min((f["response"] for f in anode["faces"] if f), key=abs))
PY
fi
prov() { awk -F= -v k="$1" '$1==k{print $2}' "$EV/provenance.txt"; }

# ---- imaging
if [[ " $STAGES " == *" img "* ]]; then
    [ -s "$EV/provenance.txt" ] || fail img_noprov
    if [ -e "$WORKDIR" ]; then echo "REFUSING: $WORKDIR exists (never overwrite a record)" >&2; fail img_exists; fi
    mkdir -p "$WORKDIR/input"
    cp "$EV/evt$K-anode$ANODE-sp.tar.bz2" "$WORKDIR/input/$SPNAME-anode$ANODE.tar.bz2" || fail img_stage
    { echo "doc=pdvd/101 phase 2 simulated muon"; echo "det=$DET k=$K anode=$ANODE sim_seed=$SEED";
      echo "truth=$TRUTH"; echo "sim_dir=$EV"; echo "pin=$PIN"; cat "$EV/provenance.txt"; } > "$WORKDIR/d101-sim-provenance.txt"
    t0=$SECONDS
    (cd "$DETDIR" && WIRECELL_PATH=$RUNNER_WCP ./run_img_evt.sh -d off -a "$ANODE" -s "$TAG" "$RUN" "$K") > "$EV/img.out" 2>&1
    rc=$?
    stage_log img $rc $((SECONDS-t0))
    [ $rc -eq 0 ] || fail img
    ls "$WORKDIR"/clusters-apa-*"$ANODE"-ms-active.tar.gz >/dev/null 2>&1 || fail img_noarchive
fi

# ---- clustering
if [[ " $STAGES " == *" clus "* ]]; then
    [ -s "$EV/provenance.txt" ] || fail clus_noprov
    DS=$(prov drift_speed_mmus); NT=$(prov nticks)
    TOFF=$TIME_OFFSET_NS
    if [ "$TOFF" = derived ]; then
        for kv in $EXP; do
            want=${kv#*=}; got=$(prov "${kv%%=*}")
            python3 -c "import sys; sys.exit(0 if abs(float('$got') - float('$want')) < 1e-2 else 1)" 2>/dev/null \
                || { echo "REFUSING derived time_offset: ${kv%%=*}=$got in the compiled sim config, derived from $want" >&2; fail clus_toff; }
        done
        TOFF=$DERIVED_TOFF_NS
    fi
    CTLA=""
    [ "$TOFF" != none ] && CTLA="-S time_offset=$TOFF"
    { echo "clus_time_offset_ns=$TOFF   # <DET>_CLUS_TLA=\"$CTLA\" (wct-clustering.jsonnet, WCT units = ns)"
      [ "$TOFF" != none ] && echo "pr_time_offset_us=$(python3 -c "print($TOFF/1000.0)")   # pass <DET>_PR_TLA=\"-S time_offset=<us>\" to run_pr_evt.sh (wct-pr-perevt.jsonnet multiplies by wc.us)"
      echo "note: run_pr_evt.sh does not read time_offset from the .tlas; PDHD's .tlas line time_offset=0 is hard-coded by run_clus_evt.sh"
      echo "derivation: time_offset = t_frame - ctoffset - (R_FR - (x_response - x_W))/v  (d101_sim_chain.sh header)"
      echo "run=$RUN k=$K event_no_in_files=first_frame_number(100) of the sim driver, identical for every k"
    } > "$WORKDIR/d101-time-offset.txt"
    t0=$SECONDS
    if [ "$DET" = pdvd ]; then
        (cd "$DETDIR" && WIRECELL_PATH=$RUNNER_WCP PDVD_DRIFT_SPEED_BOT_MMUS=$DS PDVD_DRIFT_SPEED_TOP_MMUS=$DS \
            PDVD_READOUT_NTICKS=$NT PDVD_CLUS_TLA="$CTLA" \
            ./run_clus_evt.sh -a "$ANODE" -s "$TAG" -noq -noop -save-pctree -save-assoc "$RUN" "$K") > "$EV/clus.out" 2>&1
    else
        (cd "$DETDIR" && WIRECELL_PATH=$RUNNER_WCP PDHD_CLUS_TLA="$CTLA" \
            ./run_clus_evt.sh -a "$ANODE" -s "$TAG" -noq -noop -save-pctree -save-assoc "$RUN" "$K") > "$EV/clus.out" 2>&1
    fi
    rc=$?
    stage_log clus $rc $((SECONDS-t0))
    [ $rc -eq 0 ] || fail clus
    TL=$(ls "$WORKDIR"/pctree-evt*.tlas 2>/dev/null | head -1)
    [ -n "$TL" ] && ls "${TL%.tlas}.tar.gz" >/dev/null 2>&1 || fail clus_nopctree
    # the clustering job must have used the simulation's drift speed and readout window
    python3 - "$TL" "$DS" "$NT" "$DET" <<'PY' || fail clus_consistency
import sys
kv = dict(l.strip().split("=", 1) for l in open(sys.argv[1]) if "=" in l)
ds, nt, det = float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
if det == "pdvd":
    got = [float(kv["drift_speed_bot_mmus"]), float(kv["drift_speed_top_mmus"])]
else:
    got = [float(kv["drift_speed_mmus"])]
bad = [g for g in got if abs(g - ds) > 1e-6]
if bad or int(kv["readout_window_ticks"]) != nt or kv.get("save_assoc_id", "true") != "true":
    print("tlas inconsistent with sim: drift %s vs %s, ticks %s vs %s" % (got, ds, kv["readout_window_ticks"], nt)); sys.exit(1)
print("tlas consistent: drift %s mm/us, readout %s ticks" % (got, nt))
PY
fi
touch "$EV/DONE"
echo "DONE det=$DET k=$K tag=$TAG workdir=$WORKDIR"
