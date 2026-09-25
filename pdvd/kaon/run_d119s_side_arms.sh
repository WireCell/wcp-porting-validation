#!/bin/bash
# doc pdvd/119 sec 8: one clustering arm (Q/L + op dump, -calib) on the two-sided
# production chain, into FRESH dirs work/<run6>_<idx>_<TAG>/ with the imaging
# archives of work/<run6>_<idx>_<SRC_TAG>/ symlinked (read, never written; M13).
# The light archive is the existing work/<run6>_light<art><LIGHT_SUFFIX>/.
# Pinned to the install PREFIX; CLUS_TLA is passed as PDVD_CLUS_TLA.
# Usage: PREFIX=<install> TAG=<tag> [CLUS_TLA="-S k=v"] [RUN=039349]
#        [SRC_TAG=q35flip] [LIGHT_SUFFIX=_d119beam] [BEAM=1] ./run_d119s_side_arms.sh <idx> ...
set -u
KDIR=$(cd "$(dirname "$0")" && pwd)
PDVD=$(dirname "$KDIR")
PREFIX=${PREFIX:?set PREFIX}; TAG=${TAG:?set TAG}
RUN=${RUN:-039349}; SRC_TAG=${SRC_TAG:-q35flip}; LIGHT_SUFFIX=${LIGHT_SUFFIX:-_d119beam}
BEAM=${BEAM:-1}; MAXJ=${MAXJ:-3}; CLUS_TLA=${CLUS_TLA:-}
export PATH="$PREFIX/bin:$PATH" LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"
mkdir -p "$KDIR/logs"
one() {
    local idx=$1 src dst art rc
    src="work/${RUN}_${idx}_${SRC_TAG}"; dst="work/${RUN}_${idx}_${TAG}"
    cd "$PDVD"
    art=$(awk -F= '$1=="event"{print $2}' "$src"/pctree-evt*.tlas)
    [ -n "$art" ] || { echo "idx $idx: no art event"; return 4; }
    [ -s "work/${RUN}_light${art}${LIGHT_SUFFIX}/opflash_pdvd-wct.tar.gz" ] || { echo "idx $idx: no light"; return 5; }
    [ -e "$dst" ] && { echo "idx $idx: $dst exists -- refusing (M13)"; return 3; }
    mkdir -p "$dst"
    for f in "$src"/clusters-apa-*.tar.gz "$src"/img-provenance.txt; do [ -e "$f" ] && ln -s "$PDVD/$f" "$dst/"; done
    PDVD_BEAM_LABEL=$BEAM PDVD_LIGHT_SUFFIX=$LIGHT_SUFFIX PDVD_CLUS_TLA="$CLUS_TLA" setarch x86_64 -R \
        ./run_clus_evt.sh -s "$TAG" -calib "$RUN" "$idx" > "$KDIR/logs/d119s_${TAG}_${RUN}_${idx}.log" 2>&1
    rc=$?; echo "idx $idx art $art $TAG rc=$rc"; return $rc
}
nfail=0
for idx in "$@"; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n || nfail=$((nfail+1)); done
    one "$idx" &
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || nfail=$((nfail+1)); done
echo "=== d119s arms $TAG done, $nfail failed ==="
exit $nfail
