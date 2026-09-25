#!/bin/bash
# doc pdvd/119 knob-ON arms on the two-sided production chain (run 039349).
# Per event (<idx> = evt dir, art event = light dir):
#   light: run_light_evt.sh -s _d119beam with PDVD_BEAM_LABEL=1 (production
#          light defaults; only the archive metadata gains trigger_us/tc_type)
#   clus : two FRESH tagged dirs with the q35flip imaging symlinked in,
#          d119beam (PDVD_BEAM_LABEL=1) and d119beamoff (label off), both on
#          the SAME private arm-B install (PREFIX), same light archive.
# Nothing existing is written into (M13): every output dir is new.
# Usage: PREFIX=<install> ./run_d119_beam_arms.sh <idx> [idx ...]
set -u
KDIR=$(cd "$(dirname "$0")" && pwd)
PDVD=$(dirname "$KDIR")
PREFIX=${PREFIX:?set PREFIX to the arm install prefix}
MAXJ=${MAXJ:-3}
SRC_TAG=${SRC_TAG:-q35flip}
export PATH="$PREFIX/bin:$PATH" LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"
mkdir -p "$KDIR/logs"

one() {
    local idx=$1 art tag dst
    art=$(awk -F= '$1=="event"{print $2}' "$PDVD/work/039349_${idx}_${SRC_TAG}"/pctree-evt*.tlas)
    [ -n "$art" ] || { echo "idx $idx: no art event"; return 4; }
    cd "$PDVD"
    if [ ! -s "work/039349_light${art}_d119beam/opflash_pdvd-wct.tar.gz" ]; then
        PDVD_BEAM_LABEL=1 setarch x86_64 -R ./run_light_evt.sh -s _d119beam 39349 "$art" \
            > "$KDIR/logs/d119_light_${idx}.log" 2>&1 || { echo "idx $idx light rc=$?"; return 5; }
    fi
    for tag in d119beam d119beamoff; do
        dst="work/039349_${idx}_${tag}"
        [ -e "$dst" ] && { echo "idx $idx: $dst exists -- refusing (M13)"; return 3; }
        mkdir -p "$dst"
        for f in "work/039349_${idx}_${SRC_TAG}"/clusters-apa-*.tar.gz "work/039349_${idx}_${SRC_TAG}"/img-provenance.txt; do
            ln -s "$PDVD/$f" "$dst/"
        done
        local lab=0; [ "$tag" = d119beam ] && lab=1
        PDVD_BEAM_LABEL=$lab PDVD_LIGHT_SUFFIX=_d119beam setarch x86_64 -R \
            ./run_clus_evt.sh -s "$tag" -calib 39349 "$idx" > "$KDIR/logs/d119_clus_${tag}_${idx}.log" 2>&1
        echo "idx $idx art $art $tag rc=$?"
    done
}
for idx in "$@"; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
    one "$idx" &
done
wait
echo "=== d119 beam arms done ==="
