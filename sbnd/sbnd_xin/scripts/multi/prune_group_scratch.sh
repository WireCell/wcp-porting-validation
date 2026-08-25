#!/bin/bash
# doc 81: drop a stage-A root's per-GROUP scratch once the per-event products
# have been gated.  With --layout perevt every event's products already live in
# evt<ID>/ and ql_evt<ID>/, so the g<K>/ archives are a second copy of the same
# bytes plus the frames the dump can regenerate from the reco1 file.
#
# KEPT (the record layer, CLAUDE.md M13): every log, .time.meta, events.txt,
# rse.json, split.log and *.stdout.  DROPPED: frames-dnn.tar.bz2,
# icluster-*.npz, opflash_apa*.tar.gz, pctree-ql.tar.gz.
#
# usage: prune_group_scratch.sh <work_root> [<work_root> ...]      # dry run
#        CONFIRM=yes prune_group_scratch.sh <work_root> ...        # do it
set -u
CONFIRM=${CONFIRM:-no}
total=0
for root in "$@"; do
    [ -d "$root" ] || { echo "no such root: $root" >&2; continue; }
    # refuse a root that does not carry the per-event layout: without it the
    # group archives are the ONLY copy.
    nq=$(ls -d "$root"/ql_evt* 2>/dev/null | wc -l)
    ng=$(ls -d "$root"/g* 2>/dev/null | wc -l)
    if [ "$nq" -eq 0 ]; then
        echo "REFUSING $root: no ql_evt*/ -- the g*/ archives are the only copy" >&2
        continue
    fi
    sz=0
    for f in "$root"/g*/frames-dnn.tar.bz2 "$root"/g*/icluster-*.npz \
             "$root"/g*/opflash_apa*.tar.gz "$root"/g*/pctree-ql.tar.gz; do
        [ -e "$f" ] || continue
        s=$(stat -c %s "$f"); sz=$((sz+s))
        if [ "$CONFIRM" = yes ]; then rm -f "$f"; fi
    done
    total=$((total+sz))
    printf "%-28s %3d groups, %4d ql_evt -> %s %s\n" "$(basename "$root")" "$ng" "$nq" \
           "$(numfmt --to=iec $sz)" "$([ "$CONFIRM" = yes ] && echo REMOVED || echo '(dry run)')"
done
echo "total: $(numfmt --to=iec $total) $([ "$CONFIRM" = yes ] && echo removed || echo 'would be removed')"
