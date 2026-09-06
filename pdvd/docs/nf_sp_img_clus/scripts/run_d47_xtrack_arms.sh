#!/bin/bash
# doc pdvd/47 -- run the controlled-track sim -> NF -> SP arms for one detector.
#
#   DET=pdhd|pdvd|sbnd  [ARMS="S0 S1 S2 S3 S5 nsig5 S6a S6b top"]  [NPAR=6]  [OUT=/home/xqian/tmp/xtrack/$DET]
#       ./run_d47_xtrack_arms.sh
#
# Every arm is COMPILED first (wcsonnet -> $OUT/cfg/<arm>.json, the compiled-config proof
# the doc quotes), then run with `wire-cell -c` on the pinned libraries
# /home/xqian/tmp/xtrack_libpin (a peer's wcbuild must not swap the binary mid-study).
# Outputs: $OUT/<arm>-anode<N>-{raw,sp,splat}.tar.bz2, $OUT/<arm>.log, $OUT/rc.txt (one
# "<arm> rc=<n>" line per arm -- never judge through a pipe).
#
# Arms (TLAs of <det>_sim/wct-sim-xtrack-sp.jsonnet; the default = the data-production-like chain):
#   S0     splat=true                      DepoFluxSplat truth (no response, no noise, no SP)
#   S1     (defaults) dump_rawdecon=true   sim + noise -> NF -> SP        [S4 = its rawdecon tag]
#   S2     noise=false fluctuate=false     + dump_rawdecon
#   S3     S2 + diffusion=false            the constant term alone
#   S5     S1 + wire_filter=passthrough    the wire filter's share
#   nsig5  S2 + nsigma=5                   the sim's 3-sigma truncation
#   S0n5   S0 + nsigma=5
#   S6a    (PDHD) S1 + l1sp_mode=process   PDHD data production (L1SPFilterPD writeback)
#   S6b    (PDHD) S1 on anode 0            APA0: np04hd FR + per-wire-region fltresp filters
#   top    (PDVD) S1 on anode 4            top CRP (the other FR / electronics / noise model)
#   S1n05 / S1n2   S1 with the noise spectra amplitudes x0.5 / x2 (d47_scale_noise.py)
set -u
DET=${DET:?DET=pdhd|pdvd|sbnd}
ARMS=${ARMS:-"S0 S1 S2 S3 S5 nsig5 S0n5"}
NPAR=${NPAR:-6}
OUT=${OUT:-/home/xqian/tmp/xtrack/$DET}
PIN=${PIN:-/home/xqian/tmp/xtrack_libpin}
SIMDIR=/home/xqian/toolkit-dev/wcp-porting-img/${DET}_sim
WC=/home/xqian/toolkit-dev/local/bin/wire-cell
export WIRECELL_PATH=$SIMDIR:/home/xqian/toolkit-dev/toolkit/cfg:/home/xqian/toolkit-dev/wire-cell-data
export LD_LIBRARY_PATH=$PIN
[ -d "$PIN" ] || { echo "no pin $PIN (cp -a local/lib/. $PIN/)" >&2; exit 2; }
if ldd $WC | grep -i wirecell | grep -qv "$PIN"; then echo "REFUSING: wire-cell libs not resolved from $PIN" >&2; exit 2; fi
mkdir -p "$OUT/cfg"
# noiseless arms: SBND's mbOneChannelNoise flags every flat channel bad whatever the RMS cuts,
# so the noise filter is bypassed there (nf=false); PDHD/PDVD keep it (neutralised RMS cuts).
NONF=""
case $DET in
    pdhd) V="-V elecGain=14"; ANODE=1; TRACKS=$OUT/tracks_pdhd_a1.json ;;
    pdvd) V=""; ANODE=0; TRACKS=$OUT/tracks_pdvd_a0.json ;;
    sbnd) V=""; ANODE=0; TRACKS=$OUT/tracks_sbnd_a0.json; NONF="--tla-code nf=false" ;;
esac
arm_tlas() {
    case $1 in
        S0)    echo "--tla-code splat=true" ;;
        S1)    echo "--tla-code dump_rawdecon=true" ;;
        S2)    echo "--tla-code noise=false --tla-code fluctuate=false --tla-code dump_rawdecon=true $NONF" ;;
        S3)    echo "--tla-code noise=false --tla-code fluctuate=false --tla-code diffusion=false --tla-code dump_rawdecon=true $NONF" ;;
        S5)    echo "--tla-code dump_rawdecon=true --tla-str wire_filter=passthrough" ;;
        nsig5) echo "--tla-code noise=false --tla-code fluctuate=false --tla-code dump_rawdecon=true --tla-code nsigma=5 $NONF" ;;
        S0n5)  echo "--tla-code splat=true --tla-code nsigma=5" ;;
        S6a)   echo "--tla-code dump_rawdecon=true --tla-str l1sp_mode=process" ;;
        S6b)   echo "--tla-code dump_rawdecon=true" ;;
        top)   echo "--tla-code dump_rawdecon=true" ;;
        S1n05) echo "--tla-code dump_rawdecon=true --tla-str noise_tag=x0.5" ;;
        S1n2)  echo "--tla-code dump_rawdecon=true --tla-str noise_tag=x2" ;;
        *) echo "unknown arm $1" >&2; return 1 ;;
    esac
}
run_arm() {
    local arm=$1 anode=$ANODE tracks=$TRACKS
    [ "$arm" = S6b ] && { anode=0; tracks=$OUT/tracks_pdhd_a0.json; }
    [ "$arm" = top ] && { anode=4; tracks=$OUT/tracks_pdvd_a4.json; }
    local tl; tl=$(arm_tlas "$arm") || return 1
    (cd "$SIMDIR" && wcsonnet $V --tla-code "tracks=$(cat "$tracks")" --tla-code anode_index=$anode \
        --tla-str output_prefix="$OUT/$arm" $tl -o "$OUT/cfg/$arm.json" wct-sim-xtrack-sp.jsonnet) > "$OUT/cfg/$arm.compile.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ] || [ ! -s "$OUT/cfg/$arm.json" ]; then echo "$arm compile rc=$rc" | tee -a "$OUT/rc.txt"; return 1; fi
    rm -f "$OUT/$arm-anode$anode-"*.tar.bz2
    (cd "$SIMDIR" && $WC -l "$OUT/$arm.log:debug" -L debug -c "$OUT/cfg/$arm.json") > "$OUT/$arm.out" 2>&1
    rc=$?
    echo "$arm rc=$rc anode=$anode tracks=$(basename "$tracks") files=$(ls "$OUT/$arm-anode$anode-"*.tar.bz2 2>/dev/null | wc -l) $(date -Is)" | tee -a "$OUT/rc.txt"
}
md5sum "$PIN"/libWireCellSigProc.so "$PIN"/libWireCellGen.so >> "$OUT/libpin.md5"
echo "DET=$DET ARMS=$ARMS tracks=$TRACKS anode=$ANODE pin=$PIN $(date -Is)" >> "$OUT/rc.txt"
n=0
for arm in $ARMS; do
    run_arm "$arm" &
    n=$((n+1))
    if [ $n -ge $NPAR ]; then wait -n; n=$((n-1)); fi
done
wait
echo "DET=$DET done $(date -Is)" | tee -a "$OUT/rc.txt"
