#!/bin/bash
# Seed imaging into a fresh nueCC48 work root, then run the 48-event nusel arm.
# Imaging (icluster-*.npz, sp-frames.tar.bz2) is upstream of clustering, so it is
# identical across arms by construction -- verified: evt172230 apa0-active md5
# ecdb89d7b5fe5391 in oc19on / u17on / cbron / vveto.  These are OUR OWN pipeline
# products (run_img_evt.sh), so M11 is satisfied.
set -u
R=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
SRC=$R/work-nuecc48-oc19on
ARM=${1:?arm suffix, e.g. cathA12on}
DST=$R/work-nuecc48-$ARM

echo "=== seeding $DST from $SRC ==="
mkdir -p "$DST"
n=0
for d in "$SRC"/evt*; do
    b=$(basename "$d")
    mkdir -p "$DST/$b"
    cp -n "$d"/icluster-*.npz "$d"/sp-frames.tar.bz2 "$DST/$b/" 2>/dev/null
    n=$((n+1))
done
echo "seeded $n evt dirs"

md5sum /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCellClus.so

cd "$R" || exit 1
seq 1 48 | xargs -P 6 -I{} env SBND_WORK_ROOT="$DST" \
    SBND_INPUT_DIR="$R/input_files_reco1/extracted-2025fall-48evt-fsprod" \
    ./run_nusel_evt.sh data {}
echo "nuecc48 rc=$?"
echo "done: $(ls "$DST"/.status 2>/dev/null | wc -l)/48"
echo "S4-nuecc48-$ARM DONE"
