#!/bin/bash
# doc pdvd/97 -- build ONE Bee set per detector from the video picks (scan/d97/picks.tsv).
#
# Adapted by duplication from pdhd/docs/scripts/d04_bee_pr_combine.sh (left untouched). LOCAL ONLY:
# it writes the zips and the index files and STOPS -- uploading is a separate, owner-authorised step.
#
#   bash d97_build_bee.sh            -> /home/xqian/tmp/d97/bee-d97-{pdhd,pdvd}.zip
#                                       pdvd/docs/scan/d97/bee-d97-{pdhd,pdvd}.index.txt
#
# Every member of each pick's PRODUCTION mabc-pr.zip (PDHD h28prod, PDVD p96vprod) is copied verbatim to
# data/<i>/<i>-<layer>.json; nothing is recomputed.  Bee event <i> = the row order below (class order
# dots, attached, both, bare; rank 1 then 2), which is also the upload order Bee numbers events by.
# img-global / op are deliberately NOT included (doc pdvd/97 sec 2: img-global is the raw drift frame,
# 50-60 cm off the t0-corrected fit on a cosmic; op's cluster ids need a remap).
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
PICKS=$IMG/pdvd/docs/scan/d97/picks.tsv
OUTD=/home/xqian/tmp/d97
IDXD=$IMG/pdvd/docs/scan/d97
mkdir -p "$OUTD" "$IDXD"
[ -s "$PICKS" ] || { echo "ERROR: $PICKS missing" >&2; exit 2; }

for det in pdhd pdvd; do
    case $det in pdhd) arm=h28prod ;; pdvd) arm=p96vprod ;; esac
    out=$OUTD/bee-d97-$det.zip
    idx=$IDXD/bee-d97-$det.index.txt
    stage=$(mktemp -d /home/xqian/tmp/d97/stage-$det.XXXXXX)
    mkdir -p "$stage/data"
    rows=$(python3 - "$PICKS" "$det" <<'PY'
import csv, sys
order = {"dots": 0, "attached": 1, "both": 2, "bare": 3}
rows = [r for r in csv.DictReader((l for l in open(sys.argv[1]) if not l.startswith("#")), delimiter="\t")
        if r["det"] == sys.argv[2]]
rows.sort(key=lambda r: (order[r["cls"]], int(r["rank"])))
for r in rows:
    print(r["cls"], r["rank"], r["key"])
PY
)
    [ -n "$rows" ] || { echo "ERROR: no $det picks" >&2; exit 2; }
    {
        echo "# doc pdvd/97 -- Bee set $det, production arm $arm, built $(date -Is)"
        echo -e "# bee_idx\tclass\trank\tkey (run_evt/cluster)\tsource zip\tlayers (nclusters)"
    } > "$idx"
    i=0
    while read -r cls rank key; do
        evt=${key%/*}
        src=$IMG/$det/work/${evt}_${arm}/mabc-pr.zip
        [ -s "$src" ] || { echo "ERROR: $src missing" >&2; exit 3; }
        tmp=$(mktemp -d /home/xqian/tmp/d97/unz.XXXXXX)
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
        echo -e "$i\t$cls\t$rank\t$key\t${evt}_${arm}/mabc-pr.zip\t$cens" >> "$idx"
        echo "  [$det event $i] $cls #$rank $key"
        i=$((i + 1))
    done <<< "$rows"
    rm -f "$out"
    ( cd "$stage" && zip -rq "$out" data ) || { echo "ERROR: zip $out" >&2; exit 4; }
    rm -rf "$stage"
    echo "wrote $out ($i events); index $idx"
done

# ---- verification: every member byte-identical (sha256) to its source member; every event has mc + track_fit
python3 - "$IDXD" "$OUTD" "$IMG" <<'PY'
import hashlib, os, sys, zipfile
idxd, outd, img = sys.argv[1:4]
bad = 0
for det, arm in (("pdhd", "h28prod"), ("pdvd", "p96vprod")):
    z = zipfile.ZipFile(os.path.join(outd, f"bee-d97-{det}.zip"))
    names = {x for x in z.namelist() if not x.endswith("/")}      # `zip -r` also stores directory entries
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(idxd, f"bee-d97-{det}.index.txt")) if not l.startswith("#")]
    n = 0
    for i, cls, rank, key, src, _ in rows:
        s = zipfile.ZipFile(os.path.join(img, det, "work", src))
        for m in s.namelist():
            if not m.startswith("data/0/0-"):
                continue
            t = f"data/{i}/{i}-" + m.split("data/0/0-", 1)[1]
            if t not in names or hashlib.sha256(z.read(t)).digest() != hashlib.sha256(s.read(m)).digest():
                print("MISMATCH", det, t); bad += 1
            n += 1
        for need in ("mc.json", "track_fit-global.json"):
            if f"data/{i}/{i}-{need}" not in names:
                print("MISSING", det, i, need); bad += 1
    extra = len(names) - n
    print(f"verify {det}: {len(rows)} events, {n} members sha256-identical to source, extra members {extra}")
    bad += extra != 0
sys.exit(1 if bad else 0)
PY
rc=$?
echo "verify rc=$rc"
echo "NOT uploaded.  Upload by hand:  $IMG/pdvd/upload-to-bee.sh $OUTD/bee-d97-<det>.zip"
exit $rc
