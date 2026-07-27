#!/bin/bash
# Master-merge validation arm: NF+SP -> imaging -> clustering for the standard
# abtest manifest (4 PDHD + 2 PDVD), snapshotting ALL THREE stages so sigproc,
# img and clus can each be gated separately.
#
# Snapshots go to /home/xqian/tmp/mergegate/<arm>/ -- deliberately NOT
# abtest/snap/ (that tree is the scientific record, M13).
#
# Usage: ./mergegate_run.sh <arm>        e.g. ./mergegate_run.sh pre
set -u
ARM=${1:?usage: mergegate_run.sh <arm>}
W=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
AB=$W/abtest
DEST=/home/xqian/tmp/mergegate/$ARM
MANIFEST=${MANIFEST:-$AB/events.txt}
MAXJOBS=${MAXJOBS:-6}
mkdir -p "$DEST"

# traditional (non-DNN) SP: exercises the sigproc code master changed and is
# deterministic; the DNN-ROI path adds a torch inference master did not touch.
one() {
    local det=$1 run=$2 evt=$3
    local rp d out sp rc
    rp=$(printf '%06d' "$((10#$run))")
    d=$W/$det/work/${rp}_${evt}
    out=$DEST/${det}_${rp}_${evt}
    mkdir -p "$out"
    [ "$det" = pdhd ] && sp=protodunehd-sp-frames || sp=protodune-sp-frames

    # --- stage 1: NF + SP ---
    setarch x86_64 -R python3 "$AB/timecmd.py" "$out/nfsp_meta.txt" \
        "$W/$det/run_nf_sp_evt.sh" "$run" "$evt" > "$out/nfsp.log" 2>&1
    rc=$?; if [ $rc -ne 0 ]; then echo "[$det $rp $evt] NFSP FAILED rc=$rc"; return $rc; fi
    cp -f "$d"/${sp}-anode*.tar.bz2 "$out/" 2>/dev/null

    # --- stage 2: imaging (-d off => the traditional SP frames just made) ---
    setarch x86_64 -R python3 "$AB/timecmd.py" "$out/img_meta.txt" \
        "$W/$det/run_img_evt.sh" -d off "$run" "$evt" > "$out/img.log" 2>&1
    rc=$?; if [ $rc -ne 0 ]; then echo "[$det $rp $evt] IMG FAILED rc=$rc"; return $rc; fi
    cp -f "$d"/clusters-apa-*.tar.gz "$out/" 2>/dev/null

    # --- stage 3: clustering ---
    setarch x86_64 -R python3 "$AB/timecmd.py" "$out/clus_meta.txt" \
        "$W/$det/run_clus_evt.sh" "$run" "$evt" > "$out/clus.log" 2>&1
    rc=$?; if [ $rc -ne 0 ]; then echo "[$det $rp $evt] CLUS FAILED rc=$rc"; return $rc; fi
    cp -f "$d"/mabc-*.zip "$out/" 2>/dev/null

    echo "[$det $rp $evt] done"
}

pids=(); n=0
while read -r det run evt; do
    case "$det" in ''|\#*) continue ;; esac
    one "$det" "$run" "$evt" & pids+=($!); n=$((n+1))
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
done < "$MANIFEST"

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "=== arm '$ARM' complete ($n events, fail=$fail) -> $DEST ==="
exit $fail
