#!/bin/bash
# valfast A/B report, chunk-parallel edition of valfast_compare.sh.
#
# Usage: valfast/valfast_compare_par.sh <tagA> <tagB> [sample ...]
#   samples default: mcp1k nuecc48 r1qlmc r2mc
#   VF_CMP_JOBS (default 8) parallel vf_tree_compare.py chunks per sample.
#
# Gates, pass/fail logic, and per-event output lines are IDENTICAL to
# valfast_compare.sh -- gate 1 just runs vf_tree_compare.py over event-list
# chunks concurrently (the compare is read-only: archive hashing + uproot
# reads), then concatenates chunk outputs in list order so the report reads
# the same.  Wall clock on the 572-event mcp1k sample drops ~JOBSx.
set -u
SBND_DIR=$(cd "$(dirname "$0")/.." && pwd -P)
VF=$SBND_DIR/valfast
cd "$SBND_DIR" || exit 1

A=${1:?usage: valfast_compare_par.sh <tagA> <tagB> [sample ...]}
B=${2:?usage: valfast_compare_par.sh <tagA> <tagB> [sample ...]}
shift 2
SAMPLES=${*:-"mcp1k nuecc48 r1qlmc r2mc"}
JOBS=${VF_CMP_JOBS:-8}

TMP=$(mktemp -d /home/xqian/tmp/vfcmp-par.XXXXXX) || exit 1
trap 'rm -rf "$TMP"' EXIT

nusel_root() {
    case "$1" in
        mcp1k)   echo "work-mcp1kall-vf$2";;
        nuecc48) echo "work-nuecc48-vf$2";;
        r1qlmc)  echo "work-r1qlmc-vf$2";;
        r2mc)    echo "work-r2mc-vf$2";;
    esac
}

fail=0
for s in $SAMPLES; do
    ra=work-vf$s-$A; rb=work-vf$s-$B
    echo "=================== [$s] $ra vs $rb ==================="
    if [ ! -d "$ra" ] || [ ! -d "$rb" ]; then
        echo "[$s] MISSING ARM ($ra or $rb) -- FAIL"; fail=1; continue
    fi
    # 1. PR archives (HARD gate) + trees (INFORMATIONAL) -- chunk-parallel.
    #    Same interpretation as valfast_compare.sh: a rows=/= line with mabc==
    #    and pctree== and a clean scores-diff is the known M4-residual
    #    T_tagger vector instability, not a DIFF.
    nlines=$(wc -l < "$VF/events-$s.txt")
    nj=$JOBS; [ "$nlines" -lt "$nj" ] && nj=$nlines
    rm -rf "$TMP/$s"; mkdir -p "$TMP/$s"
    split -d -n l/"$nj" "$VF/events-$s.txt" "$TMP/$s/chunk."
    for c in "$TMP/$s"/chunk.*; do
        [ -s "$c" ] || continue
        # shellcheck disable=SC2046
        ( python3 "$VF/vf_tree_compare.py" "$ra" "$rb" $(tr '\n' ' ' < "$c") \
              > "$c.out" 2>&1; echo $? > "$c.rc" ) &
    done
    wait
    rc1=0
    for c in "$TMP/$s"/chunk.*; do
        case "$c" in *.out|*.rc) continue;; esac
        [ -s "$c" ] || continue
        cat "$c.out"
        [ "$(cat "$c.rc")" -eq 0 ] || rc1=1
    done
    if grep -q 'mabc=≠\|pctree=≠\|MISSING' "$TMP/$s"/chunk.*.out; then
        echo "[$s] ARCHIVE DIFF -- hard gate FAIL"; fail=1
    elif [ $rc1 -ne 0 ]; then
        echo "[$s] tree-feature instability only (archives identical) -- informational, see valfast/README.md"
    fi
    # 2. physics columns (identical to valfast_compare.sh)
    if [ -f "$ra/vf-scores.tsv" ] && [ -f "$rb/vf-scores.tsv" ]; then
        python3 "$VF/vf_scores_diff.py" "$ra/vf-scores.tsv" "$rb/vf-scores.tsv"
        rc2=$?; [ $rc2 -eq 0 ] || fail=1
        echo "[$s] scores-diff rc=$rc2"
    else
        echo "[$s] vf-scores.tsv missing in one arm -- FAIL"; fail=1
    fi
    # 3. nusel-side archives (-full arms only; identical to valfast_compare.sh)
    na=$(nusel_root "$s" "$A"); nb=$(nusel_root "$s" "$B")
    if [ -d "$na" ] && [ -d "$nb" ]; then
        python3 "$VF/nusel_hash_compare.py" "$na" "$nb" "$VF/events-$s.txt"
        rc3=$?; [ $rc3 -eq 0 ] || fail=1
        echo "[$s] nusel-archives rc=$rc3"
    else
        echo "[$s] nusel roots absent (PR-tail arms) -- gate 3 skipped"
    fi
done

echo
if [ $fail -eq 0 ]; then
    echo "VALFAST PASS: $A vs $B identical on [$SAMPLES] (all gates that ran)"
else
    echo "VALFAST DIFF: $A vs $B -- see per-sample output above"
fi
exit $fail
