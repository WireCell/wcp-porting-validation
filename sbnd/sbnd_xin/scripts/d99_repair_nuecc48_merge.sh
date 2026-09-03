#!/bin/bash
# doc 99 -- repair the truncated arm-wide merge in work-nuecc48-d97fvpr2.
#
#   ./scripts/d99_repair_nuecc48_merge.sh              # dry run (default)
#   CONFIRM=yes ./scripts/d99_repair_nuecc48_merge.sh  # apply
#
# WHAT IS BROKEN.  run_pr_chain_batch.sh used to merge the events of ITS
# INVOCATION into the arm-wide nusel-table.tsv / nusel-events.tsv.  A one-event
# re-run of this arm (evt 256587, 2026-09-02 11:32) therefore rewrote the merge
# with that single event: nusel-events.tsv 49 rows -> 2, nusel-table.tsv 547 ->
# 10.  The runner is fixed (the merge is arm-scoped now); this repairs the file
# that invocation damaged.
#
# WHY THIS IS SAFE, and why it still asks.  The two merged files are a PURE
# FUNCTION of the 48 per-event pr_evt*/nusel-evt*.tsv, every one of which is
# intact and untouched -- so nothing irreproducible is at stake and no published
# number ever moved (the score tables and doc 92's figures read the per-event
# files, not the merge).  But CLAUDE.md sec 5.2 says do not write into an
# existing label without the owner saying so, so this is opt-in and it keeps the
# damaged originals rather than discarding them.
#
# INTERLOCKS, all of which must pass before anything is written:
#   1. all 48 per-event tables present and non-empty;
#   2. the re-merge produces exactly 49 event rows and 547 table rows;
#   3. every row of the CURRENT truncated files appears verbatim in the re-merge
#      -- i.e. the regeneration reproduces what survived, so we are restoring
#      rows rather than replacing them with something new;
#   4. nothing is written until all of the above hold.
set -u
cd -P "$(dirname "$0")/.." || exit 1
ARM=${ARM:-work-nuecc48-d97fvpr2}
CONFIRM=${CONFIRM:-no}
BK=$ARM/merge-truncated-20260902
TMP=$(mktemp -d /home/xqian/tmp/d99repair.XXXXXX) || exit 2
trap 'rm -rf "$TMP"' EXIT

[ -d "$ARM" ] || { echo "no arm at $ARM"; exit 2; }

# --- interlock 1: the inputs -------------------------------------------------
ls -d "$ARM"/pr_evt*/ 2>/dev/null | sed -E 's#.*/pr_evt([0-9]+)/?$#\1#' | sort -n > "$TMP/evts"
nevt=$(wc -l < "$TMP/evts")
TSVS=()
while read -r e; do
    t="$ARM/pr_evt$e/nusel-evt$e.tsv"
    [ -s "$t" ] || { echo "REFUSE: missing or empty $t"; exit 2; }
    TSVS+=("$t")
done < "$TMP/evts"
echo "interlock 1: $nevt per-event tables, all non-empty  OK"
[ "$nevt" -eq 48 ] || { echo "REFUSE: expected 48 events, found $nevt"; exit 2; }

# --- re-merge into a temp ----------------------------------------------------
python3 nusel_extract.py --merge "${TSVS[@]}" \
    --out "$TMP/nusel-table.tsv" --events-out "$TMP/nusel-events.tsv" > "$TMP/merge.log" 2>&1
rc=$?
[ "$rc" -eq 0 ] || { echo "REFUSE: nusel_extract.py --merge rc=$rc"; cat "$TMP/merge.log"; exit 2; }

# --- interlock 2: the shape --------------------------------------------------
ne=$(wc -l < "$TMP/nusel-events.tsv"); nt=$(wc -l < "$TMP/nusel-table.tsv")
echo "interlock 2: re-merge gives $ne event rows (want 49) and $nt table rows (want 547)"
[ "$ne" -eq 49 ] && [ "$nt" -eq 547 ] || { echo "REFUSE: unexpected shape"; exit 2; }

# --- interlock 3: the survivors are reproduced -------------------------------
miss=0
for f in nusel-events.tsv nusel-table.tsv; do
    cur=$ARM/$f
    [ -s "$cur" ] || continue
    while IFS= read -r line; do
        grep -qxF -- "$line" "$TMP/$f" || { echo "REFUSE: row not reproduced in $f: $line"; miss=1; }
    done < "$cur"
done
[ "$miss" -eq 0 ] || exit 2
echo "interlock 3: every row of the current truncated files is reproduced verbatim  OK"

if [ "$CONFIRM" != "yes" ]; then
    echo
    echo "DRY RUN -- nothing written."
    echo "  would back up  $ARM/nusel-{events,table}.tsv  ->  $BK/"
    echo "  would install  $ne-row nusel-events.tsv and $nt-row nusel-table.tsv"
    echo "To apply:  CONFIRM=yes ./scripts/d99_repair_nuecc48_merge.sh"
    exit 0
fi

# --- apply -------------------------------------------------------------------
mkdir -p "$BK"
for f in nusel-events.tsv nusel-table.tsv; do
    [ -e "$BK/$f" ] || cp -p "$ARM/$f" "$BK/$f"
done
cp "$TMP/nusel-events.tsv" "$ARM/nusel-events.tsv"
cp "$TMP/nusel-table.tsv"  "$ARM/nusel-table.tsv"
cat > "$BK/README.txt" <<EOF
The TRUNCATED merged tables this arm shipped with, kept verbatim.  Repaired
$(date -Is) by scripts/d99_repair_nuecc48_merge.sh (sbnd_xin/docs/99).

run_pr_chain_batch.sh merged the events of ITS INVOCATION, not of the arm, so a
one-event re-run (evt 256587, 2026-09-02 11:32) rewrote the arm-wide merge with
that single event: nusel-events.tsv 49 rows -> 2, nusel-table.tsv 547 -> 10.

All 48 per-event pr_evt*/nusel-evt*.tsv were intact and were never touched, so
no published number was ever affected -- the score tables and doc 92's figures
read the per-event files.  The repair re-merged those same 48 files with the
same tool (nusel_extract.py --merge), and checked that every row surviving in
the truncated files came back verbatim.  The runner is fixed: the merge is
arm-scoped and idempotent now, so a partial re-run can only refresh rows.
EOF
echo "repaired: $(wc -l < "$ARM/nusel-events.tsv") event rows, $(wc -l < "$ARM/nusel-table.tsv") table rows"
echo "originals kept in $BK/"
