#!/bin/bash
# Seed a fresh QL-only work tag from the canonical _keep clustering archives.
#   Usage: ./scripts/stage_ql_tag.sh <run> <idx> <tag>
# Creates work/<run6>_<idx>_<tag>/ with symlinks to the _keep
# clusters-apa-*.tar.gz so run_clus_evt.sh -s <tag> reruns matching only
# (clustering inputs identical by construction).  Refuses to touch an
# existing tag dir: past tags are records (CLAUDE.md M13) — new run, new tag.
set -e
PDVD_DIR=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?run} IDX=${2:?idx} TAG=${3:?tag}
RUN_PADDED=$(printf '%06d' "$((10#$RUN))")
SRC="$PDVD_DIR/work/${RUN_PADDED}_${IDX}_keep"
DST="$PDVD_DIR/work/${RUN_PADDED}_${IDX}_${TAG}"
[ -d "$SRC" ] || { echo "ERROR: missing $SRC" >&2; exit 1; }
ls "$SRC"/clusters-apa-*.tar.gz >/dev/null 2>&1 \
    || { echo "ERROR: no clusters-apa-*.tar.gz in $SRC" >&2; exit 1; }
if [ -e "$DST" ]; then
    echo "ERROR: $DST already exists (fresh tags only, M13)" >&2; exit 1
fi
mkdir -p "$DST"
( cd "$DST" && ln -s "../${RUN_PADDED}_${IDX}_keep"/clusters-apa-*.tar.gz . )
echo "staged $DST ($(ls "$DST" | wc -l) archives)"
