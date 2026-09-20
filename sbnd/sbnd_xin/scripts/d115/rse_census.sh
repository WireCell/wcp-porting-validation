#!/bin/bash
# doc sbnd_xin/115 -- build the per-file (entry, run, subrun, event) census of a round-3 sample,
# and gate the campaign's central architectural assumption.
#
# WHY.  run_chain_group.sh --layout perevt writes ql_evt<ID>/ and run_pr_chain_batch.sh writes
# pr_evt<ID>/, both keyed by the art event NUMBER alone.  SBND MC event numbers are only unique
# within a (run, subrun): mc-cv has 50 distinct event numbers across its 2017 events.  A single
# out_root per sample would therefore overwrite most of the sample silently.  The campaign runs
# ONE out_root PER RECO1 FILE instead, which is sufficient only if each file is internally
# unique -- that is what this script proves, file by file, before anything expensive runs.
#
# Usage: rse_census.sh <cv|nuecc|off> [jobs]
# Writes: products/d115/<sample>/file_rse.tsv   fileidx  file  entry  run  subrun  event
#         products/d115/<sample>/files.lst      fileidx  abspath           (campaign file order)
# Exit 0 only when every file is internally unique and no file failed to read.
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
R3=$SX/xin-round3-samples
S=${1:?usage: rse_census.sh <cv|nuecc|off> [jobs]}
J=${2:-16}
case "$S" in
    cv)    D=$R3/mc-cv/reco1 ;;
    nuecc) D=$R3/mc-nuecc/reco1 ;;
    off)   D=$R3/beam-off/reco1 ;;
    *) echo "unknown sample: $S (cv|nuecc|off)" >&2; exit 2 ;;
esac
OUT=$SX/products/d115/$S
mkdir -p "$OUT"
TMP=${TMPDIR:-/home/xqian/tmp}/d115-rse-$S.$$
mkdir -p "$TMP"

# Campaign file order is `ls` order (sorted) -- fixed once here and reused by every later stage,
# so fileidx <-> f<NNN> out_root never drifts.
ls -1 "$D"/*.root | sort > "$TMP/files.raw"
NF=$(wc -l < "$TMP/files.raw")
[ "$NF" -gt 0 ] || { echo "ERROR: no reco1 files under $D" >&2; exit 1; }
awk '{printf "%03d\t%s\n", NR-1, $0}' "$TMP/files.raw" > "$OUT/files.lst"
echo "=== d115 rse census: sample=$S files=$NF jobs=$J"

one() {  # one <fileidx> <path>
    local i=$1 f=$2
    root -l -b -q "$SXX/scripts/d115/rse_list.C(\"$f\")" 2>/dev/null \
        | awk -v i="$i" -v f="$(basename "$f")" '$1=="RSE"{print i"\t"f"\t"$2"\t"$3"\t"$4"\t"$5}' \
        > "$TMPP/rows.$i"
    [ -s "$TMPP/rows.$i" ] || echo "READFAIL $i $f" > "$TMPP/fail.$i"
}
export -f one
export SXX=$SX TMPP=$TMP
awk -F'\t' '{print $1"\t"$2}' "$OUT/files.lst" \
    | xargs -P "$J" -I{} bash -c 'IFS=$(printf "\t"); set -- {}; one "$1" "$2"'

printf 'fileidx\tfile\tentry\trun\tsubrun\tevent\n' > "$OUT/file_rse.tsv"
cat "$TMP"/rows.* 2>/dev/null | sort -k1,1n -k3,3n >> "$OUT/file_rse.tsv"

nfail=$(ls "$TMP"/fail.* 2>/dev/null | wc -l)
if [ "$nfail" -gt 0 ]; then
    echo "ERROR: $nfail file(s) unreadable:" >&2; cat "$TMP"/fail.* >&2
fi

# The gate: event numbers unique WITHIN each file, and one subrun per file (informational).
awk -F'\t' 'NR>1{
        n++; k=$1"\t"$6; if (seen[k]++) dup[$1]++;
        sr[$1"\t"$4"\t"$5]=1; nf[$1]++
    }
    END{
        nd=0; for (i in dup) { printf "  DUP fileidx=%s repeated event numbers=%d\n", i, dup[i]; nd++ }
        nsr=0; for (k in sr) nsr++
        printf "  events=%d files=%d distinct(run,subrun) pairs=%d files_with_duplicates=%d\n",
               n, length(nf), nsr, nd
        exit (nd > 0)
    }' "$OUT/file_rse.tsv"
gate=$?

nev=$(( $(wc -l < "$OUT/file_rse.tsv") - 1 ))
nuniq_evt=$(awk -F'\t' 'NR>1{e[$6]=1}END{print length(e)}' "$OUT/file_rse.tsv")
nuniq_rse=$(awk -F'\t' 'NR>1{k[$4"_"$5"_"$6]=1}END{print length(k)}' "$OUT/file_rse.tsv")
echo "  sample totals: events=$nev unique(run,subrun,event)=$nuniq_rse unique(event alone)=$nuniq_evt"
echo "  -> $OUT/file_rse.tsv"
rm -rf "$TMP"
[ "$nfail" -eq 0 ] && [ "$gate" -eq 0 ] \
    && { echo "  RSE CENSUS PASS (per-file event numbers unique)"; exit 0; }
echo "  RSE CENSUS FAIL"; exit 1
