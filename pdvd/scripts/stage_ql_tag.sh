#!/bin/bash
# Seed a fresh QL-only work tag from an existing clustering-archive tag.
#   Usage: ./scripts/stage_ql_tag.sh <run> <idx> <tag> [src_tag]
# Creates work/<run6>_<idx>_<tag>/ with symlinks to the src_tag dir's
# clusters-apa-*.tar.gz (and its img-provenance.txt) so run_clus_evt.sh -s <tag>
# reruns matching only (clustering inputs identical by construction).
# src_tag defaults to d27fresh (imaging on protodunevd-wires-larsoft-v7-uvwfit,
# doc pdvd/27).  The July _keep archives were tiled with the v6 wires file;
# run_clus_evt.sh refuses them under today's anodes (doc pdvd/27 sec 5.2).
# Refuses to touch an existing tag dir: past tags are records (CLAUDE.md M13)
# -- new run, new tag.
set -e
PDVD_DIR=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?run} IDX=${2:?idx} TAG=${3:?tag} SRC_TAG=${4:-d27fresh}
RUN_PADDED=$(printf '%06d' "$((10#$RUN))")
SRC="$PDVD_DIR/work/${RUN_PADDED}_${IDX}_${SRC_TAG}"
DST="$PDVD_DIR/work/${RUN_PADDED}_${IDX}_${TAG}"
[ -d "$SRC" ] || { echo "ERROR: missing $SRC" >&2; exit 1; }
ls "$SRC"/clusters-apa-*.tar.gz >/dev/null 2>&1 \
    || { echo "ERROR: no clusters-apa-*.tar.gz in $SRC" >&2; exit 1; }
if [ -e "$DST" ]; then
    echo "ERROR: $DST already exists (fresh tags only, M13)" >&2; exit 1
fi
mkdir -p "$DST"
( cd "$DST" && ln -s "../${RUN_PADDED}_${IDX}_${SRC_TAG}"/clusters-apa-*.tar.gz . )
if [ -f "$SRC/img-provenance.txt" ]; then
    ( cd "$DST" && ln -s "../${RUN_PADDED}_${IDX}_${SRC_TAG}/img-provenance.txt" . )
else
    echo "WARNING: $SRC has no img-provenance.txt; run_clus_evt.sh will not be able to prove its wires file (doc pdvd/27)" >&2
fi
echo "staged $DST ($(ls "$DST" | wc -l) archives) from ${SRC_TAG}"
