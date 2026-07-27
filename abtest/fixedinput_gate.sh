#!/bin/bash
# The PDHD/PDVD NF+SP stage is nondeterministic run-to-run on some anodes
# (demonstrated with the PRE-merge binary: pdhd 027305 evt0 anode0 gave two
# different content hashes on two identical runs).  That makes a naive
# pre-vs-post img/clus comparison inconclusive.
#
# This gate removes the SP variable: it feeds the POST-merge binary the EXACT
# SP frames the PRE arm used (copied back from the pre snapshot), re-runs only
# imaging + clustering, and compares against the pre arm's img/clus outputs.
# Any difference is then attributable to img/clus code, not to SP input.
#
# Usage: ./fixedinput_gate.sh
set -u
W=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
AB=$W/abtest
PRE=/home/xqian/tmp/mergegate/pre
DEST=/home/xqian/tmp/mergegate/fixedinput
MANIFEST=$AB/events.txt
mkdir -p "$DEST"

one() {
    local det=$1 run=$2 evt=$3
    local rp d out
    rp=$(printf '%06d' "$((10#$run))")
    d=$W/$det/work/${rp}_${evt}
    out=$DEST/${det}_${rp}_${evt}
    mkdir -p "$out"
    # restore the PRE arm's SP frames as the input for both arms
    cp -f "$PRE/${det}_${rp}_${evt}"/*sp-frames-anode*.tar.bz2 "$d/" || return 1
    setarch x86_64 -R python3 "$AB/timecmd.py" "$out/img_meta.txt" \
        "$W/$det/run_img_evt.sh" -d off "$run" "$evt" > "$out/img.log" 2>&1
    [ $? -eq 0 ] || { echo "[$det $rp $evt] IMG FAILED"; return 1; }
    cp -f "$d"/clusters-apa-*.tar.gz "$out/" 2>/dev/null
    setarch x86_64 -R python3 "$AB/timecmd.py" "$out/clus_meta.txt" \
        "$W/$det/run_clus_evt.sh" "$run" "$evt" > "$out/clus.log" 2>&1
    [ $? -eq 0 ] || { echo "[$det $rp $evt] CLUS FAILED"; return 1; }
    cp -f "$d"/mabc-*.zip "$out/" 2>/dev/null
    echo "[$det $rp $evt] done"
}

pids=()
while read -r det run evt; do
    case "$det" in ''|\#*) continue ;; esac
    one "$det" "$run" "$evt" & pids+=($!)
    while [ "$(jobs -rp | wc -l)" -ge 6 ]; do sleep 5; done
done < "$MANIFEST"
for p in "${pids[@]}"; do wait "$p"; done
echo "FIXEDINPUT_DONE -> $DEST"
