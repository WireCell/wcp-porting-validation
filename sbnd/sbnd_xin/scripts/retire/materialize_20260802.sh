#!/bin/bash
# Retirement round 2026-08-02, step 1: make work-oc19scan-old self-contained
# so the doc pr/22 exhibits (work-pr22gap-{a,b,c,input} -> work-oc19scan-old)
# survive the removal of work-mcp1kall-u17on1kb and work-nuecc48-u17on.
#
#   ./materialize_20260802.sh          # dry run
#   CONFIRM=yes ./materialize_20260802.sh
#
# What it does, per symlink in work-oc19scan-old that points into the removal
# set (9 ql_evt* -> work-mcp1kall-u17on1kb, ql_evt444187 + evt444187 ->
# work-nuecc48-u17on):
#   - if readlink -f resolves into a SURVIVOR dir (evt444187 -> work/evt444187):
#     repoint the link to the resolved target (no copy);
#   - else copy the target dir next to the link (<name>.mat.tmp): regular files
#     copied byte-for-byte and cmp-verified, inner symlinks repointed to their
#     readlink -f resolution (all resolve into work-mcp1000 / work, verified),
#     then atomically swap link -> real dir.
# The 9 evt* -> work-mcp1000 links are left alone (work-mcp1000 is KEEP).
# Nothing inside any other work-* dir is touched.
set -u
cd "$(dirname "$0")/../.." || exit 1        # -> sbnd_xin
BASE=$PWD
D=$BASE/work-oc19scan-old
CONFIRM=${CONFIRM:-no}

# survivors of the 2026-08-02 round (see docs/work-tags.md)
survivor() {
    case "$1" in
        work|work-mcp1000|work-mcp10|work-mcp1kall-d59k|\
        work-nuecc48-base|work-nuecc48-nuf|work-nuecc48-prsmoke|work-nuecc48-prsmoke2|\
        work-r1ql-*|work-r2patrec-*|\
        work-mcp1kall-cath01|work-nuecc48-cath01|\
        work-stmcamp-d66new|work-oc19scan-old|work-pr22gap-*) return 0;;
        *) return 1;;
    esac
}
# top-level work-* dir a path under $BASE belongs to, or ""
topdir() { local rel=${1#"$BASE"/}; echo "${rel%%/*}"; }

fail=0; nrepoint=0; ncopy=0
for link in "$D"/*; do
    [ -L "$link" ] || continue
    tgt=$(readlink "$link")
    td=$(topdir "$tgt")
    survivor "$td" && continue          # points at a KEEP hub, leave it
    phys=$(readlink -f "$link")
    [ -e "$phys" ] || { echo "FAIL missing target: $link -> $tgt"; fail=1; continue; }
    ptd=$(topdir "$phys")
    if survivor "$ptd"; then
        echo "REPOINT $(basename "$link") -> $phys"
        [ "$CONFIRM" = yes ] && { ln -sfn "$phys" "$link" || fail=1; }
        nrepoint=$((nrepoint+1))
        continue
    fi
    # copy the (removal-set) target dir, repointing inner links
    echo "COPY    $(basename "$link") <- $phys"
    ncopy=$((ncopy+1))
    [ "$CONFIRM" = yes ] || continue
    tmp="$link.mat.tmp"
    lfail=0
    rm -rf "$tmp"; mkdir -p "$tmp" || { fail=1; continue; }
    for f in "$phys"/* "$phys"/.[!.]*; do
        [ -e "$f" ] || [ -L "$f" ] || continue
        b=$(basename "$f")
        if [ -L "$f" ]; then
            ftgt=$(readlink -f "$f")
            ftd=$(topdir "$ftgt")
            if [ ! -e "$ftgt" ] || ! survivor "$ftd"; then
                echo "FAIL inner link does not resolve into a survivor: $f -> $ftgt"
                lfail=1; continue
            fi
            ln -s "$ftgt" "$tmp/$b" || lfail=1
        elif [ -f "$f" ]; then
            cp -p "$f" "$tmp/$b" && cmp -s "$f" "$tmp/$b" \
                || { echo "FAIL copy/cmp: $f"; lfail=1; }
        else
            echo "FAIL unexpected non-regular entry: $f"; lfail=1
        fi
    done
    if [ $lfail -eq 0 ]; then
        rm "$link" && mv "$tmp" "$link" || { echo "FAIL swap: $link"; lfail=1; }
    else
        echo "LEFT UNTOUCHED (copy failed, symlink kept): $link"
    fi
    [ $lfail -ne 0 ] && fail=1
done

echo
echo "repoint=$nrepoint copy=$ncopy fail=$fail CONFIRM=$CONFIRM"
[ "$CONFIRM" = yes ] || { echo "dry run only -- re-run with CONFIRM=yes"; exit 0; }
[ $fail -eq 0 ] || { echo "!! failures above -- work-oc19scan-old NOT fully materialized"; exit 1; }

echo "== verification =="
left=$(find "$D" -maxdepth 1 -type l -exec readlink {} \; | awk -F/ '{print $(NF-1)}' | sort -u)
echo "remaining top-level link targets:"; echo "$left" | sed 's/^/    /'
bad=0
for t in $left; do survivor "$t" || { echo "!! still points into removal set: $t"; bad=1; }; done
echo "broken links under work-oc19scan-old + work-pr22gap-*: $(find "$D" "$BASE"/work-pr22gap-* -xtype l | wc -l) (MUST be 0)"
exit $bad
