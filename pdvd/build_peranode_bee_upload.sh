#!/bin/bash
# Build a PDVD per-anode Bee link (bee idx 0..7 = anode0..7) for ONE event,
# repackaging existing work/ products, then zip + upload and print the URL.
# Forked from pdvd/build_peranode_bee.sh (adds label/upload); does not modify it.
#
# Usage: build_peranode_bee_upload.sh <label>
#   reads work/039324_0  (run 39324, event 0)
set -e
cd /nfs/data/1/xqian/toolkit-dev/toolkit/pdvd

LABEL=${1:-v5}
RUN_STRIPPED=39324; SUBRUN=0; EVT=0
wd="work/039324_${EVT}"
BDIR="peranodebee_${LABEL}"

drift_args() {
    if [ "$1" -le 3 ]; then
        echo '--speed "-1.56*mm/us" --t0 "0*us" --x0 "-341.5*cm"'
    else
        echo '--speed "1.56*mm/us" --t0 "0*us" --x0 "341.5*cm"'
    fi
}

rm -rf "$BDIR"; mkdir -p "$BDIR/data"
for n in 0 1 2 3 4 5 6 7; do
    active="$wd/clusters-apa-anode${n}-ms-active.tar.gz"
    mabc="$wd/mabc-anode${n}.zip"
    [ -s "$active" ] || { echo "[skip] anode $n: no active tarball"; continue; }
    event_no=$(tar tzf "$active" | head -1 | sed -E 's/.*cluster_([0-9]+)_.*/\1/')
    echo "[anode $n -> bee idx $n] art_event=$event_no"
    mkdir -p "$BDIR/data/$n"
    eval wirecell-img bee-blobs -g protodunevd -s uniform -d 1 \
        --rse "$RUN_STRIPPED" "$SUBRUN" "$event_no" \
        $(drift_args "$n") \
        -o "$BDIR/data/${n}/${n}-imaging-anode${n}.json" \
        "$active"
    if [ -s "$mabc" ]; then
        unzip -p "$mabc" data/0/0-clustering-global.json \
            > "$BDIR/data/${n}/${n}-clustering-anode${n}.json"
        tmp=$(mktemp -d /home/xqian/tmp/peranode.XXXXXX)
        unzip -q -o "$mabc" -d "$tmp"
        for f in "$tmp"/data/0/0-channel-deadarea-*.json; do
            [ -e "$f" ] || continue
            suf=$(basename "$f"); suf=${suf#0-}
            cp "$f" "$BDIR/data/${n}/${n}-${suf}"
        done
        rm -rf "$tmp"
    fi
done

ZIP="upload-peranode-run039324-evt0-${LABEL}.zip"
rm -f "$ZIP"; ( cd "$BDIR" && zip -rq "../$ZIP" data )
echo "=== entries ==="; ls -1 "$BDIR"/data/*/ | sed 's/^/  /'
echo "Uploading $ZIP ..."
./upload-to-bee.sh "$ZIP"
