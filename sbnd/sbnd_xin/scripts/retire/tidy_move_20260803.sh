#!/bin/bash
# Tidy round 2026-08-03: move the top-level scripts into scripts/<sub>/ per
# scripts/retire/tidy_map_20260803.tsv.
#
#   ./tidy_move_20260803.sh          # dry run
#   CONFIRM=yes ./tidy_move_20260803.sh
#
# git subtleties this handles (wcp-porting-img ignores *.sh, *.json, *.zip):
#   - a TRACKED file moves with `git mv`, which preserves rename detection;
#   - an IGNORED-but-tracked file (the *.sh that were force-added) also moves
#     with `git mv` -- it is tracked, so git does not care that it matches an
#     ignore rule; only a fresh `git add` at the destination would need -f;
#   - an UNTRACKED file moves with plain `mv`, and is force-added afterwards
#     only if its extension is ignored.
# The dry run prints which of the three paths each file takes, so the plan is
# reviewable before anything moves.
set -u
cd "$(dirname "$0")/../.." || exit 1        # -> sbnd_xin
BASE=$PWD
REPO=$(cd "$BASE/../.." && pwd)             # -> wcp-porting-img
MAP=$BASE/scripts/retire/tidy_map_20260803.tsv
CONFIRM=${CONFIRM:-no}

[ -f "$MAP" ] || { echo "no map: $MAP  (run tidy_map_20260803.py first)"; exit 1; }

nmv=0; ngit=0; nplain=0; nforce=0; nskip=0
while IFS=$'\t' read -r name newpath act; do
    [ "$act" = "MOVE" ] || continue
    [ -e "$BASE/$name" ] || { echo "SKIP (gone): $name"; nskip=$((nskip+1)); continue; }
    dir=$(dirname "$newpath")

    rel="sbnd/sbnd_xin"
    if git -C "$REPO" ls-files --error-unmatch "$rel/$name" >/dev/null 2>&1; then
        mode=git
    else
        mode=plain
        git -C "$REPO" check-ignore -q "$rel/$newpath" && mode=plain-force
    fi

    if [ "$CONFIRM" = yes ]; then
        mkdir -p "$BASE/$dir"
        case $mode in
            git)   git -C "$REPO" mv "$rel/$name" "$rel/$newpath" || exit 1 ;;
            plain) mv "$BASE/$name" "$BASE/$newpath" || exit 1 ;;
            plain-force)
                   mv "$BASE/$name" "$BASE/$newpath" || exit 1 ;;
        esac
    fi
    echo "[$mode] $name -> $newpath"
    nmv=$((nmv+1))
    case $mode in
        git) ngit=$((ngit+1));; plain) nplain=$((nplain+1));; *) nforce=$((nforce+1));;
    esac
done < <(awk -F'\t' '$3=="MOVE"{print $1"\t"$2"\t"$3}' "$MAP")

echo
echo "moves=$nmv  (git mv=$ngit  plain=$nplain  plain-into-ignored=$nforce)  skipped=$nskip  CONFIRM=$CONFIRM"
[ "$CONFIRM" = yes ] || { echo "dry run only -- re-run with CONFIRM=yes"; exit 0; }

echo
echo "== nothing left behind at top level =="
left=0
while IFS=$'\t' read -r name newpath act; do
    [ "$act" = "MOVE" ] || continue
    [ -e "$BASE/$name" ] && { echo "  !! still present: $name"; left=$((left+1)); }
    [ -e "$BASE/$newpath" ] || { echo "  !! missing at destination: $newpath"; left=$((left+1)); }
done < <(awk -F'\t' '$3=="MOVE"{print $1"\t"$2"\t"$3}' "$MAP")
[ "$left" = 0 ] && echo "  all $nmv moved files accounted for"
exit 0
