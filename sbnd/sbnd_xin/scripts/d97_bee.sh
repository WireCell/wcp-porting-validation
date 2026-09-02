#!/bin/bash
# doc 97 -- build the owner's Bee A/B set: OFF / sep_track_recarve / sep_fv_point.
#
# Three zips over ONE event list in ONE order (bee-review sec 3): the owner's
# two symptom events first, then every event whose in-beam verdict moved under
# either knob, then the events whose reconstructed physics moved most, then
# non-firing controls.  Bee indexes events by position, so the order is fixed
# once and the annotated index references it forever.
#
# The two symptom events are MC (doc-95 debug sample, reality=sim); everything
# else is the data validation sample.  make_pr_bee.py searches several -q/-p
# roots in order, so one call covers both -- the bare ids 21/30 cannot collide
# with the 5-6 digit data ids.
#
# Usage: ./scripts/d97_bee.sh <evt> [<evt> ...]
set -u
cd -P "$(dirname "$0")/.." || exit 1
OUT=bee/d97; mkdir -p "$OUT"
QL_OFF=(-q work-dbg25a-d97off -q work-ncpi0-d97off2 -q work-nuecc48-d97off2 -q work-mcp1k-d97off2 -q work-mcp2k-d97off2)
PR_OFF=(-p work-dbg25a-d97offpr -p work-ncpi0-d97off2pr -p work-nuecc48-d97off2pr -p work-mcp1k-d97off2pr -p work-mcp2k-d97off2pr)
QL_ON=(-q work-dbg25a-d97on -q work-ncpi0-d97on -q work-nuecc48-d97on -q work-mcp1k-d97on -q work-mcp2k-d97on)
PR_ON=(-p work-dbg25a-d97onpr -p work-ncpi0-d97onpr -p work-nuecc48-d97onpr -p work-mcp1k-d97onpr -p work-mcp2k-d97onpr)
QL_FV=(-q work-dbg25a-d97fv -q work-ncpi0-d97fv -q work-nuecc48-d97fv -q work-mcp1k-d97fv -q work-mcp2k-d97fv)
PR_FV=(-p work-dbg25a-d97fvpr3 -p work-ncpi0-d97fvpr2 -p work-nuecc48-d97fvpr2 -p work-mcp1k-d97fvpr2 -p work-mcp2k-d97fvpr2)
for arm in off on fv; do
    case $arm in
        off) Q=("${QL_OFF[@]}"); P=("${PR_OFF[@]}") ;;
        on)  Q=("${QL_ON[@]}");  P=("${PR_ON[@]}")  ;;
        fv)  Q=("${QL_FV[@]}");  P=("${PR_FV[@]}")  ;;
    esac
    echo "=== $arm  $(date -Is)"
    # --allow-unevaluated is REQUIRED for this set, not a convenience.  On the
    # OFF side both symptom events are TGM, so nu_skip_cosmic refuses them and
    # TaggerCheckNeutrino selects nothing; under sep_fv_point mcp2k 105074 loses
    # its candidate the same way.  "No candidate" is exactly the state the owner
    # is judging.  And Bee addresses events by POSITION: dropping an event from
    # one arm shifts every later index, so the three zips would no longer line
    # up and the annotated index would point at the wrong events.
    python3 scripts/bee/make_pr_bee.py --allow-unevaluated "${Q[@]}" "${P[@]}" \
        -o "$OUT/d97-$arm.zip" "$@" > "$OUT/d97-$arm.build.log" 2>&1
    echo "    rc=$?  $(ls -la "$OUT/d97-$arm.zip" 2>/dev/null | awk '{print $5" bytes"}')"
    tail -3 "$OUT/d97-$arm.build.log" | sed 's/^/    /'
done
