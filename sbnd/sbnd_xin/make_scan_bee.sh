#!/usr/bin/env bash
# Doc 59: build + upload the Bee event displays for the hand-scan subset.
#
# One upload per chunk file (nusel_scan_filter.py --chunk writes them), because
# a whole 1000-event sample's worth of layers is far past what one Bee set
# should carry -- the 48-event upload of doc 21 is the precedent for the size.
#
# Each event's Bee tree = make_stmfit_bee.py's merge:
#   from ql_evt<ID>/mabc-all-apa.zip : img-global, clustering-global, op
#                                      (the Q/L flash<->cluster match), and the
#                                      channel-deadarea layers
#   from nusel_evt<ID>/mabc-pr.zip   : stm_fit-global  (PR_LAYERS -- ONLY the
#                                      fit layer, so clustering-global stays
#                                      the Q/L one)
# with stm_fit's cluster ids remapped into img-global's space so Bee's flash
# navigation shows a fit together with the bundle it belongs to.
#
# Bee identifies events by directory index, so data/<i>/ is line i+1 of the
# chunk file: the chunk files ARE the Bee-index -> event-id maps and are kept
# next to the zips, together with make_stmfit_bee.py's <zip>.stmid-map.txt
# (PR cluster id -> img cluster id, needed to get from a Bee color back to the
# TSV / tracking-stm.root ids).
#
# Usage: ./make_scan_bee.sh <outdir> <work_root> <chunk.txt> [<chunk.txt> ...]
#   Set NOUPLOAD=1 to build the zips without uploading (offline check).
# Uploading is outward-facing (CLAUDE.md escalation rule 6) -- run it only when
# the owner asked for Bee links.
set -u
cd "$(dirname "$0")"
SBND_DIR=$PWD
OUT=${1:?outdir}
ROOT=${2:?work root}
shift 2
UPLOAD=$SBND_DIR/../../upload-to-bee.sh
mkdir -p "$OUT"

fail=0
for chunk in "$@"; do
    base=$(basename "$chunk" .txt)
    zip=$OUT/$base.zip
    evts=$(tr '\n' ' ' < "$chunk")
    n=$(echo "$evts" | wc -w)

    # Pre-validate: make_stmfit_bee.py sys.exit()s on the first missing input,
    # so a single hole would waste the whole chunk's work.
    miss=0
    for e in $evts; do
        [ -s "$ROOT/ql_evt$e/mabc-all-apa.zip" ] || { echo "MISSING ql_evt$e/mabc-all-apa.zip"; miss=$((miss+1)); }
        [ -s "$ROOT/nusel_evt$e/mabc-pr.zip" ]   || { echo "MISSING nusel_evt$e/mabc-pr.zip";  miss=$((miss+1)); }
    done
    if [ "$miss" -ne 0 ]; then
        echo "[$base] SKIPPED: $miss missing input(s)"; fail=1; continue
    fi

    cp "$chunk" "$OUT/$base.index.txt"       # Bee index i  <->  line i+1
    if [ ! -s "$zip" ]; then
        echo "[$base] building $n events -> $zip"
        python3 "$SBND_DIR/make_stmfit_bee.py" -w "$ROOT" -o "$zip" $evts \
            > "$OUT/$base.build.log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "[$base] BUILD FAILED rc=$rc (see $OUT/$base.build.log)"; fail=1; continue
        fi
    else
        echo "[$base] zip exists, reusing $zip"
    fi
    echo "[$base] $(du -h "$zip" | cut -f1)  $n events"

    if [ "${NOUPLOAD:-0}" = 1 ]; then continue; fi
    url=$(cd "$OUT" && BROWSER=echo bash "$UPLOAD" "$(basename "$zip")" | tail -1)
    echo "$url" > "$OUT/$base.url"
    echo "[$base] $url"
done
echo "=== bee done fail=$fail ==="
exit $fail
