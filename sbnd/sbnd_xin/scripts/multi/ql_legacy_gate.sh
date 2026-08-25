#!/bin/bash
# doc 76 round 2: prove the round's C++ left the Q/L job byte-identical.
#
# The PR gates (pr85/pr94/nusel) cannot do this.  They never open
# mabc-all-apa.zip, and the Q/L job is the ONE production job that drives
# MultiAlgBlobClustering through a SHARED BeeSink across the per-APA and
# all-APA nodes -- the very path this round touched (the flush() early return
# that stops the redundant finalize flush from advancing the shared Bee event
# index, and BeeSink's per-event zip mode).  A compiled-config gate cannot see a
# runtime index shift either.  So: re-run Q/L per event on the current binary
# into a FRESH work root and member-hash it against the recorded products.
#
# usage: ql_legacy_gate.sh <ql_root_with_recorded_products> <fresh_root> <evt> [evt...]
set -u
SX=$(cd "$(dirname "$0")/../.." && pwd -P)
AB=$SX/../../abtest
REF=${1:?ref ql root}; NEW=${2:?fresh root}; shift 2
[ $# -gt 0 ] || { echo "give at least one event id" >&2; exit 1; }

STAGE=${SBND_STAGE:-$SX/input_files_reco1/staged-mcp2025c-1000evt}
IMGBASE=${SBND_IMGBASE:-$SX/work-img-mcp1k}

mkdir -p "$NEW"
MAP=${SBND_STAGE_MAP:-$NEW/.stage_map}
if [ ! -s "$MAP" ]; then
    ls -d "$STAGE"/e* | xargs -P 16 -I{} bash -c \
        'e=$(tar tjf "{}"/frames-dnn.tar.bz2 2>/dev/null | sed -n "s/^frame_dnnsp_\([0-9]*\)\.npy$/\1/p" | head -1); [ -n "$e" ] && echo "$e {}"' \
        > "$MAP" 2>/dev/null
    echo "staged-dir index: $(wc -l < "$MAP") events -> $MAP"
fi
rc=0
for evt in "$@"; do
    # Find this event's staged single-event sample dir.  The staged tree has no
    # index, so build one ONCE, in parallel, and cache it: scanning 1000 bz2
    # archives serially per event is minutes of nothing.
    i=$(awk -v e="$evt" '$1==e {print $2; exit}' "$MAP")
    [ -n "$i" ] || { echo "evt $evt: no staged dir under $STAGE" >&2; rc=1; continue; }
    ln -sfn "$IMGBASE/evt$evt" "$NEW/evt$evt"
    SBND_INPUT_DIR="$i" SBND_WORK_ROOT="$NEW" \
        setarch x86_64 -R "$SX/run_ql_evt.sh" data -save-pctree 1 \
        > "$NEW/.ql_evt$evt.log" 2>&1 || { echo "evt $evt: Q/L FAILED" >&2; rc=1; continue; }

    for f in mabc-all-apa.zip mabc-apa0-face0.zip mabc-apa1-face0.zip "pctree-evt$evt.tar.gz"; do
        a="$REF/ql_evt$evt/$f"; b="$NEW/ql_evt$evt/$f"
        [ -s "$a" ] && [ -s "$b" ] || { echo "evt $evt $f: MISSING"; rc=1; continue; }
        ha=$(python3 "$AB/hash_archive.py" "$a" | awk '{print $1}')
        hb=$(python3 "$AB/hash_archive.py" "$b" | awk '{print $1}')
        if [ "$ha" = "$hb" ]; then echo "evt $evt $f: SAME"
        else echo "evt $evt $f: DIFFER $ha vs $hb"; rc=1; fi
    done
done
echo "ql_legacy_gate rc=$rc"
exit $rc
