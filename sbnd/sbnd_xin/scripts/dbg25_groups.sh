#!/bin/bash
# doc 95 phase A2 -- rebuild the 25 staged single-event dirs into TWO
# collision-free multi-event sample dirs in the yuhw layout.
#
# WHY TWO.  The 25 entries are 25 genuinely distinct events (all 25 RSE
# distinct, 20 different runs) but only 20 distinct BARE event ids: ids
# 12, 14, 22, 31, 34 each occur twice, in different runs.  Every downstream
# name -- frame_dnnsp_<ID>.npy, work/evt<ID>, ql_evt<ID>, pr_evt<ID> -- is
# keyed on the bare id, so all 25 in one sample dir silently loses five
# events with no error.  Splitting so that each group has internally unique
# ids is the smallest fix that keeps the doc-93 per-event chain verbatim.
#
#   group a = the first occurrence of each id            -> 20 events
#   group b = the five second occurrences                ->  5 events
#
# Membership is DERIVED from entry_event_map.tsv (first occurrence wins), not
# hand-typed, so it cannot drift from the map.
set -u
SBND=$(cd -P "$(dirname "$0")/.." && pwd)
STAGE=$SBND/input_files_reco1/staged-dbg25
MAP=$STAGE/entry_event_map.tsv
[ -s "$MAP" ] || { echo "ERROR: missing $MAP" >&2; exit 1; }

WORK=/home/xqian/tmp/dbg25/groupbuild
rm -rf "$WORK"; mkdir -p "$WORK"

# derive membership
awk -F'\t' 'NR>1 { if (seen[$4]++) print $1"\tb"; else print $1"\ta" }' "$MAP" \
    > "$WORK/entry_group.tsv"
echo "group sizes:"; cut -f2 "$WORK/entry_group.tsv" | sort | uniq -c

for g in a b; do
    OUT=$SBND/input_files_reco1/extracted-dbg25$g
    if [ -e "$OUT/frames-dnn.tar.bz2" ]; then
        echo "SKIP $OUT -- already built (M13)"; continue
    fi
    mkdir -p "$OUT"
    F=$WORK/$g/frames; O0=$WORK/$g/op0; O1=$WORK/$g/op1
    mkdir -p "$F" "$O0" "$O1"
    n=0
    while IFS=$'\t' read -r e grp; do
        [ "$grp" = "$g" ] || continue
        tar xjf "$STAGE/e$e/frames-dnn.tar.bz2"   -C "$F"
        tar xzf "$STAGE/e$e/opflash_apa0.tar.gz"  -C "$O0"
        tar xzf "$STAGE/e$e/opflash_apa1.tar.gz"  -C "$O1"
        n=$((n+1))
    done < "$WORK/entry_group.tsv"
    # sorted member order so the archive is reproducible
    (cd "$F"  && tar cjf "$OUT/frames-dnn.tar.bz2"  $(ls | sort))
    (cd "$O0" && tar czf "$OUT/opflash_apa0.tar.gz" $(ls | sort))
    (cd "$O1" && tar czf "$OUT/opflash_apa1.tar.gz" $(ls | sort))
    nf=$(tar tjf "$OUT/frames-dnn.tar.bz2" | grep -c '^frame_dnnsp_')
    nu=$(tar tjf "$OUT/frames-dnn.tar.bz2" | grep '^frame_dnnsp_' | sort -u | wc -l)
    echo "group $g -> $OUT: entries=$n frame members=$nf unique=$nu"
    if [ "$nf" -ne "$n" ] || [ "$nu" -ne "$n" ]; then
        echo "FAIL: group $g is not collision-free ($n/$nf/$nu)" >&2; exit 1
    fi
done
echo "OK -- both groups collision-free"
