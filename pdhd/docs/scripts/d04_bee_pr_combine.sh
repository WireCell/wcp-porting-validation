#!/bin/bash
# doc pdhd/04 -- combine several events' PR Bee zips (mabc-pr.zip) into ONE Bee
# set, so a scan of the STM tagger's fit population is one link.
#
# Fork BY DUPLICATION of pdvd/run_bee_combined_tagged.sh (which combines the
# CLUSTERING zip mabc-all-apa.zip); that script is untouched.  LOCAL ONLY: it
# writes the zip and an index and STOPS -- uploading is outward-facing.
#
# Usage: ./docs/scripts/d04_bee_pr_combine.sh <TAG> <run> -e evt,evt,... [-o out.zip]
#   The -e ORDER is the Bee event index order and is written to <out>.index.txt.
#   e.g. ./docs/scripts/d04_bee_pr_combine.sh d04bee 029107 -e 12,1,16,20,22,0
set -e
PDHD_DIR=$(cd "$(dirname "$0")/../.." && pwd)
TAG=${1:?usage: $0 <TAG> <run> -e evt,...}; shift
RUN=${1:?usage: $0 <TAG> <run> -e evt,...}; shift
EVT_LIST=""; OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        -e) EVT_LIST="$2"; shift 2 ;;
        -o) OUT="$2"; shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
done
[ -n "$EVT_LIST" ] || { echo "-e evt,evt,... is required (it fixes the Bee index order)" >&2; exit 1; }
RUN_PADDED=$(printf '%06d' "$((10#$RUN))")
OUT=${OUT:-$PDHD_DIR/bee-pr-run${RUN_PADDED}-${TAG}.zip}
IDXF="${OUT%.zip}.index.txt"

STAGE=$(mktemp -d /home/xqian/tmp/beeprcomb-${TAG}.XXXXXX)
mkdir -p "$STAGE/data"
: > "$IDXF"
echo "# doc pdhd/04 -- PR Bee set: run $RUN_PADDED tag $TAG  built $(date -Is)" >> "$IDXF"
echo -e "# bee_idx\tevent\tlayers (nclusters)" >> "$IDXF"
_idx=0
for _e in $(echo "$EVT_LIST" | tr ',' ' '); do
    _mabc="$PDHD_DIR/work/${RUN_PADDED}_${_e}_${TAG}/mabc-pr.zip"
    [ -s "$_mabc" ] || { echo "ERROR: evt=$_e has no $_mabc" >&2; exit 2; }
    echo "  [evt $_e -> bee index $_idx]  $_mabc"
    mkdir -p "$STAGE/data/$_idx"
    _tmp=$(mktemp -d /home/xqian/tmp/beeprcomb.XXXXXX)
    unzip -q -o "$_mabc" -d "$_tmp"
    for _f in "$_tmp"/data/0/0-*.json; do
        [ -e "$_f" ] || continue
        _suffix=$(basename "$_f"); _suffix=${_suffix#0-}
        cp "$_f" "$STAGE/data/${_idx}/${_idx}-${_suffix}"
    done
    rm -rf "$_tmp"
    # per-event layer census straight from the staged JSON (never from the log)
    _cens=$(python3 - "$STAGE/data/$_idx" <<'PY'
import json,sys,glob,os
out=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],'*.json'))):
    b=os.path.basename(f)
    if 'deadarea' in b: continue
    lay=b.split('-',1)[1].replace('-global.json','')
    d=json.load(open(f))
    if not isinstance(d,dict) or 'cluster_id' not in d: continue
    out.append(f"{lay}={len(set(d['cluster_id']))}")
print(" ".join(out))
PY
)
    printf '%d\t%s\t%s\n' "$_idx" "$_e" "$_cens" >> "$IDXF"
    _idx=$((_idx + 1))
done
rm -f "$OUT"
( cd "$STAGE" && zip -rq "$OUT" data )
rm -rf "$STAGE"
echo "Wrote $OUT ($_idx event(s));  index: $IDXF"
echo "NOT uploaded.  To upload by hand:  ./upload-to-bee.sh $OUT"
