#!/bin/bash
# doc 96 sec 9 -- run the track_recarve feasibility probe (see d96_recarve_probe.py).
# Config-level A/B: the OFF arm is the event's own production compiled config with
# only its output paths moved, so OFF must reproduce production byte-identically.
set -u
cd -P "$(dirname "$0")/.." || exit 1
export LD_LIBRARY_PATH=$HOME/tmp/doc94r3b-libsnap:${LD_LIBRARY_PATH:-}
export WIRECELL_PATH=/nfs/data/1/xqian/toolkit-dev/toolkit/cfg:/nfs/data/1/xqian/toolkit-dev/wire-cell-data:/nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd/photodet
R=/home/xqian/tmp/d96/recarve
for evt in "$@"; do
    for arm in ${ARMS:-off on}; do
        echo "=== evt$evt $arm  $(date -Is)"
        setarch x86_64 -R wire-cell -l "$R/$arm/evt$evt.log:info" -L info \
            -c "$R/$arm/evt$evt.json" > "$R/$arm/evt$evt.out" 2>&1
        echo "   rc=$?  recarve fires: $(grep -c 'Separate track_recarve' "$R/$arm/evt$evt.out" 2>/dev/null)"
        grep 'Separate track_recarve' "$R/$arm/evt$evt.out" 2>/dev/null | sed 's/^/     /'
    done
done
echo "=== done $(date -Is)"
