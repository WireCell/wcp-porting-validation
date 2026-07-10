#!/bin/bash
# Sweep all filelist events through run_one.sh with bounded concurrency and
# collect per-event wall/RSS/node-exec plus Bee-zip content hashes.
#
# Usage: ./sweep_5384.sh <label> [concurrency]
#   label        -> sweep/<label>/{results.tsv,hashes.txt,<idx>_<EV>/...}
#   concurrency  default 6 (box policy: never saturate this shared host)
#
# results.tsv columns: idx ev subrun wall_s maxrss_kb rc node_exec_s
# hashes.txt: hash_archive.py rollup per mabc_<idx>.zip (mtime-insensitive).
set -u
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
QLPORT=$(dirname "$SCRIPTS")
ABTEST=$(cd "$QLPORT/../abtest" && pwd)

LABEL=${1:?usage: sweep_5384.sh <label> [concurrency]}
PJOBS=${2:-6}

OUT="$SCRIPTS/sweep/$LABEL"
mkdir -p "$OUT"
TSV="$OUT/results.tsv"
printf 'idx\tev\tsubrun\twall_s\tmaxrss_kb\trc\tnode_exec_s\n' > "$TSV"

NEVT=$(grep -c . "$QLPORT/filelist")

one() {
    local idx=$1
    "$SCRIPTS/run_one.sh" "$idx" "$LABEL" > /dev/null 2>&1
    local dest meta ev sr
    dest=$(ls -d "$OUT/${idx}_"* | head -1)
    meta="$dest/meta.txt"
    ev=$(sed -nE 's/.*ev=([0-9]+).*/\1/p' "$meta")
    sr=$(sed -nE 's/.* subrun=([0-9]+).*/\1/p' "$meta")
    local rc wall kb node
    rc=$(awk -F= '/^rc=/{print $2}' "$meta")
    wall=$(awk -F= '/^wall_s=/{print $2}' "$meta")
    kb=$(awk -F= '/^maxrss_kb=/{print $2}' "$meta")
    node=$(awk -F= '/^node_exec_s=/{print $2}' "$meta")
    # single short O_APPEND write is atomic
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$idx" "$ev" "$sr" "$wall" "$kb" "$rc" "$node" >> "$TSV"
    echo "[idx=$idx ev=$ev] wall=${wall}s rss=${kb}kB node=${node}s rc=$rc"
}

running=0
for ((idx=0; idx<NEVT; idx++)); do
    one "$idx" &
    running=$((running+1))
    if [ "$running" -ge "$PJOBS" ]; then
        wait -n
        running=$((running-1))
    fi
done
wait

# stable order for diffing across labels
{ head -1 "$TSV"; tail -n +2 "$TSV" | sort -n; } > "$TSV.tmp" && mv "$TSV.tmp" "$TSV"

python3 "$ABTEST/hash_archive.py" "$OUT"/*/mabc_*.zip \
    | awk '{n=split($3,p,"/"); print $1"  "$2"  "p[n]}' | sort -t_ -k2 -n > "$OUT/hashes.txt"

n=$(($(wc -l < "$TSV") - 1))
nfail=$(awk -F'\t' 'NR>1 && $6!=0' "$TSV" | wc -l)
echo "=== sweep $LABEL: $n events, $nfail failures -> $TSV ==="
