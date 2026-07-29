#!/bin/bash
# Master-merge validation: SBND img -> clus -> Q/L arm, into a FRESH work root
# (SBND_WORK_ROOT override) so no existing work-*/ label is touched (M13).
# The 30-event PR/tagger stage is gated separately by run_perf54_nusel.sh.
#
# Usage: ./sbndgate_run.sh <arm> [idx ...]     e.g. ./sbndgate_run.sh pre 1 2
set -u
ARM=${1:?usage: sbndgate_run.sh <arm> [idx ...]}; shift
IDXS=${*:-1 2}
SB=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
ROOT=$SB/work-mcp10-m66${ARM}sb
DEST=/home/xqian/tmp/mergegate/sbnd-$ARM
mkdir -p "$ROOT" "$DEST"
cd "$SB" || exit 1

for idx in $IDXS; do
    echo "== idx=$idx =="
    for stage in img clus ql; do
        SBND_WORK_ROOT=$ROOT setarch x86_64 -R ./run_${stage}_evt.sh mc "$idx" \
            > "$DEST/idx${idx}_${stage}.log" 2>&1
        echo "  $stage rc=$?"
    done
done

# snapshot the per-event products
for d in "$ROOT"/evt* "$ROOT"/ql_evt*; do
    [ -d "$d" ] || continue
    b=$(basename "$d"); mkdir -p "$DEST/$b"
    cp -f "$d"/*.npz "$d"/*.tar.gz "$d"/*.zip "$DEST/$b/" 2>/dev/null
done
echo "=== sbnd arm '$ARM' -> $DEST (root $ROOT) ==="
find "$DEST" -name '*.npz' -o -name '*.zip' -o -name '*.tar.gz' | wc -l | sed 's/^/  artifacts: /'
