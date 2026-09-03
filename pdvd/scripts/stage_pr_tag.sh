#!/bin/bash
# Seed a fresh PR-only work tag from an existing point-tree tag (doc pdvd/28).
#   Usage: ./scripts/stage_pr_tag.sh <run> <idx> <tag> [src_tag]
# Creates work/<run6>_<idx>_<tag>/ with symlinks to the src_tag dir's
# pctree-evt<N>.tar.gz + pctree-evt<N>.tlas (+ img-provenance.txt) so
# run_pr_evt.sh -s <tag> reruns the PR stage only, on byte-identical input.
# src_tag defaults to d27fresh (the v7-uvwfit baseline, doc pdvd/27).
# Refuses to touch an existing tag dir (CLAUDE.md M13): new run, new tag.
set -e
PDVD_DIR=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?run} IDX=${2:?idx} TAG=${3:?tag} SRC_TAG=${4:-d27fresh}
RUN_PADDED=$(printf '%06d' "$((10#$RUN))")
SRC="$PDVD_DIR/work/${RUN_PADDED}_${IDX}_${SRC_TAG}"
DST="$PDVD_DIR/work/${RUN_PADDED}_${IDX}_${TAG}"
[ -d "$SRC" ] || { echo "ERROR: missing $SRC" >&2; exit 1; }
ls "$SRC"/pctree-evt*.tar.gz >/dev/null 2>&1 \
    || { echo "ERROR: no pctree-evt*.tar.gz in $SRC (run_clus_evt.sh -save-pctree first)" >&2; exit 1; }
if [ -e "$DST" ]; then
    echo "ERROR: $DST already exists (fresh tags only, M13)" >&2; exit 1
fi
mkdir -p "$DST"
( cd "$DST" && ln -s "../${RUN_PADDED}_${IDX}_${SRC_TAG}"/pctree-evt*.tar.gz "../${RUN_PADDED}_${IDX}_${SRC_TAG}"/pctree-evt*.tlas . )
[ -f "$SRC/img-provenance.txt" ] && ( cd "$DST" && ln -s "../${RUN_PADDED}_${IDX}_${SRC_TAG}/img-provenance.txt" . )
echo "staged $DST from ${SRC_TAG}"
