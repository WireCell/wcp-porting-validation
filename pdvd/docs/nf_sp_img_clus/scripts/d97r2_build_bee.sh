#!/bin/bash
# doc pdvd/97 round 2 -- build the PDVD Bee set from the round-2 picks (scan/d97r2/picks.tsv, rank 1 only).
#
# Adapted by duplication from d97_build_bee.sh (left untouched, so the round-1 index files stay reproducible).
# LOCAL ONLY: writes the zip and the index file and STOPS -- uploading is a separate, owner-authorised step.
#
#   bash d97r2_build_bee.sh      -> /home/xqian/tmp/d97r2/bee-d97r2-pdvd.zip
#                                   pdvd/docs/scan/d97r2/bee-d97r2-pdvd.index.txt
#
# Every member of each pick's PRODUCTION mabc-pr.zip (PDVD d103vflip, the flipped trajectory config) is copied
# verbatim to data/<i>/<i>-<layer>.json; nothing is recomputed.  Bee event <i> = the row order of picks.tsv (the
# pre-registered class order of d97r2_video_picks.py), which is also the upload order Bee numbers events by.
# The set carries every layer the production zip has (clustering, track_fit, stm_fit, stm, shower_track, vertices,
# steiner_graph, steiner_terminals, channel dead areas, mc); img-global / op are not in the PR zip (doc 97 sec 2).
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
ARM=d103vflip
PICKS=$IMG/pdvd/docs/scan/d97r2/picks.tsv
OUTD=/home/xqian/tmp/d97r2
IDX=$IMG/pdvd/docs/scan/d97r2/bee-d97r2-pdvd.index.txt
OUT=$OUTD/bee-d97r2-pdvd.zip
mkdir -p "$OUTD"
[ -s "$PICKS" ] || { echo "ERROR: $PICKS missing" >&2; exit 2; }

stage=$(mktemp -d $OUTD/stage-pdvd.XXXXXX)
mkdir -p "$stage/data"
rows=$(python3 - "$PICKS" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader((l for l in open(sys.argv[1]) if not l.startswith("#")), delimiter="\t")
        if r["rank"] == "1"]
for r in rows:
    print(r["cls"], r["key"])
PY
)
[ -n "$rows" ] || { echo "ERROR: no rank-1 picks" >&2; exit 2; }
{
    echo "# doc pdvd/97 round 2 -- Bee set pdvd, production arm $ARM, built $(date -Is)"
    echo -e "# bee_idx\tclass\tkey (run_evt/cluster)\tsource zip\tlayers (nclusters)"
} > "$IDX"
i=0
while read -r cls key; do
    evt=${key%/*}
    src=$IMG/pdvd/work/${evt}_${ARM}/mabc-pr.zip
    [ -s "$src" ] || { echo "ERROR: $src missing" >&2; exit 3; }
    tmp=$(mktemp -d $OUTD/unz.XXXXXX)
    unzip -q -o "$src" -d "$tmp" || { echo "ERROR: unzip $src" >&2; exit 3; }
    mkdir -p "$stage/data/$i"
    for f in "$tmp"/data/0/0-*.json; do
        b=$(basename "$f"); cp "$f" "$stage/data/$i/$i-${b#0-}"
    done
    rm -rf "$tmp"
    cens=$(python3 - "$stage/data/$i" <<'PY'
import glob, json, os, sys
out = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    b = os.path.basename(f)
    if "deadarea" in b:
        continue
    lay = b.split("-", 1)[1].replace("-global.json", "").replace(".json", "")
    d = json.load(open(f))
    out.append(f"{lay}={len(set(d['cluster_id']))}" if isinstance(d, dict) and "cluster_id" in d else lay)
print(" ".join(out))
PY
)
    echo -e "$i\t$cls\t$key\t${evt}_${ARM}/mabc-pr.zip\t$cens" >> "$IDX"
    echo "  [pdvd event $i] $cls $key"
    i=$((i + 1))
done <<< "$rows"
rm -f "$OUT"
( cd "$stage" && zip -rq "$OUT" data ) || { echo "ERROR: zip $OUT" >&2; exit 4; }
rm -rf "$stage"
echo "wrote $OUT ($i events); index $IDX"

# ---- verification: every member byte-identical (sha256) to its source member; every event has mc + track_fit
python3 - "$IDX" "$OUT" "$IMG" <<'PY'
import hashlib, sys, zipfile
idx, out, img = sys.argv[1:4]
bad = 0
z = zipfile.ZipFile(out)
names = {x for x in z.namelist() if not x.endswith("/")}      # `zip -r` also stores directory entries
rows = [l.rstrip("\n").split("\t") for l in open(idx) if not l.startswith("#")]
n = 0
for i, cls, key, src, _ in rows:
    s = zipfile.ZipFile(f"{img}/pdvd/work/{src}")
    for m in s.namelist():
        if not m.startswith("data/0/0-"):
            continue
        t = f"data/{i}/{i}-" + m.split("data/0/0-", 1)[1]
        if t not in names or hashlib.sha256(z.read(t)).digest() != hashlib.sha256(s.read(m)).digest():
            print("MISMATCH", t); bad += 1
        n += 1
    for need in ("mc.json", "track_fit-global.json", "stm_fit-global.json", "clustering-global.json"):
        if f"data/{i}/{i}-{need}" not in names:
            print("MISSING", i, need); bad += 1
extra = len(names) - n
print(f"verify pdvd: {len(rows)} events, {n} members sha256-identical to source, extra members {extra}")
bad += extra != 0
sys.exit(1 if bad else 0)
PY
rc=$?
echo "verify rc=$rc"
echo "NOT uploaded.  Upload by hand:  $IMG/pdvd/upload-to-bee.sh $OUT"
exit $rc
