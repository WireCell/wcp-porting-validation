#!/bin/bash
# doc sbnd_xin/115: true deposited energy per generator interaction, straight from the reco1
# files, with bare ROOT.
#
# WHY.  Haiwang's truth TSV carries the event-level GENIE truth but no Edep, and doc 107's
# signal definition (sec 5.6) and its Edep-binned vertex table both need it.  d108_reco1_truth_vectors.C
# already computes exactly the quantity Bee's TensorSetLabeler reports -- the SimEnergyDeposit sum
# over the descendants of each generator MCTruth -- and doc 108 sec 3.3 validated it against Bee
# to +-0.11 MeV on 15 interactions of 10 events.  It also re-derives pdg / ccnc / mode / vertex /
# time, so the merge in build_truth.py cross-checks both truth sources against each other.
#
# ONE ROOT PROCESS PER ENTRY: the macro's own header records that the emulated read aborts with
# "free(): invalid pointer" after a few entries in a single process, and that each entry in its
# own process reads cleanly.
#
# Usage: [JOBS=n] truth_edep.sh <cv|nuecc>
# Writes: products/d115/<sample>/edep_raw.txt   (the macro's raw output, one block per entry)
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
S=${1:?usage: [JOBS=n] truth_edep.sh <cv|nuecc>}
J=${JOBS:-24}
case "$S" in cv|nuecc) ;; *) echo "unknown sample: $S (cv|nuecc -- beam-off data has no truth)" >&2; exit 2;; esac
CEN=$SX/products/d115/$S/file_rse.tsv
LST=$SX/products/d115/$S/files.lst
[ -s "$CEN" ] || { echo "ERROR: no census $CEN -- run rse_census.sh first" >&2; exit 1; }
OUT=$SX/products/d115/$S
TMP=$HOME/tmp/d115-edep-$S; mkdir -p "$TMP"
export ROOT_INCLUDE_PATH=$TK/root/src

NENT=$(( $(wc -l < "$CEN") - 1 ))
echo "=== d115 truth Edep: sample=$S entries=$NENT jobs=$J"

one() {  # one <fileidx> <entry>
    local i=$1 e=$2
    local f; f=$(awk -F'\t' -v i="$i" '$1==i{print $2; exit}' "$LSTT")
    root -l -b -q "$SXX/d108_reco1_truth_vectors.C(\"$f\", $e, $e)" 2>/dev/null > "$TMPP/o.$i.$e"
    grep -q '^r[0-9]' "$TMPP/o.$i.$e" || echo "EDEPFAIL $i $e $f" > "$TMPP/fail.$i.$e"
}
export -f one
export SXX=$SX TMPP=$TMP LSTT=$LST
awk -F'\t' 'NR>1{print $1"\t"$3}' "$CEN" \
  | xargs -P "$J" -I{} bash -c 'IFS=$(printf "\t"); set -- {}; one "$1" "$2"'

{ echo "# d115 truth Edep, sample=$S, macro d108_reco1_truth_vectors.C, $(date -Is)"
  for k in $(ls "$TMP" | grep '^o\.' | sed 's/^o\.//' | sort -t. -k1,1n -k2,2n); do cat "$TMP/o.$k"; done
} > "$OUT/edep_raw.txt"
nfail=$(ls "$TMP"/fail.* 2>/dev/null | wc -l)
nblk=$(grep -c '^r[0-9]' "$OUT/edep_raw.txt")
echo "  entry blocks=$nblk / $NENT   failures=$nfail"
[ "$nfail" -eq 0 ] && cat "$TMP"/fail.* 2>/dev/null
rm -rf "$TMP"
[ "$nblk" -eq "$NENT" ] && [ "$nfail" -eq 0 ] && { echo "  EDEP EXTRACTION COMPLETE -> $OUT/edep_raw.txt"; exit 0; }
echo "  EDEP EXTRACTION INCOMPLETE"; exit 1
