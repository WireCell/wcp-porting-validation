#!/usr/bin/env bash
# doc pdvd/92 -- the compiled-config proofs for the wide Bragg-peak read.
#
# NOTHING is flipped this round: no jsonnet file changes, and the knob reaches the two ON arms only
# through the existing -S stm_michel_extra bag.  So the proofs are the other way round from a flip:
#   A  bare production compiles WITHOUT any of the three keys -> the OFF path is production's own
#      compiled config, unchanged by this round (the C++ member default is 0 = off).
#   B  each ON arm's TLA puts all three keys in the compiled component config, at the right values.
#      Without this a "passing" ON arm could simply be running the knob OFF -- a vacuous pass.
#   C  bare production vs bare production before this round: no jsonnet changed, so the two compiled
#      configs are identical (diff -> 0 lines).
#   D  PDHD untouched: no key in its file, and the file matches git HEAD.
# Fork by duplication (CLAUDE.md M10) of d90_proofs.sh; the keys and the direction changed.
# Usage: bash d92_proofs.sh > /home/xqian/tmp/p92/proofs.txt 2>&1   (run from anywhere)
set -u
D=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
H=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd
W=/home/xqian/tmp/p92/proofs; mkdir -p $W
W13="bragg_wide_anchor_cm:8.0,bragg_wide_anchor_rise_min:1.3,bragg_wide_anchor_tail_max:0.8"
W15="bragg_wide_anchor_cm:8.0,bragg_wide_anchor_rise_min:1.5,bragg_wide_anchor_tail_max:0.8"
PJ='["switch_scope","flag_mains","unmerge_assoc","steiner","fiducialutils","tagger_check_tgm","tagger_check_stm","tagger_check_fc","protect_bundle","steiner_refresh","check_stm_michel","tracking_visitor","pr_display"]'
comp() {   # out [extra-args...]
    local out=$1; shift
    rm -f $W/$out.json
    (cd $D && wcsonnet -P $D -A input=x.tar.gz -A output_dir=$W -S run=39252 -S subrun=0 -S event=1 \
        -S pipeline_names="$PJ" "$@" -o $W/$out.json wct-pr-perevt.jsonnet) > $W/$out.log 2>&1
    local rc=$?
    echo "  compile $out rc=$rc size=$(stat -c %s $W/$out.json 2>/dev/null)"
    if [ $rc != 0 ] || [ ! -s $W/$out.json ]; then echo "ABORT: $out did not compile (see $W/$out.log)"; exit 2; fi
}
pj() { python3 -m json.tool --sort-keys $W/$1.json; }
keys() { grep -o '"bragg_wide_anchor[a-z_]*" *: *[0-9.]*' $W/$1.json | sort -u | tr '\n' ' '; }

comp off
comp on13 -S "stm_michel_extra={$W13}"
comp on15 -S "stm_michel_extra={$W15}"

echo "A  bare production: bragg_wide_* keys in the compiled config: [$(keys off)]  (expected: none)"
echo "B  p92v13 TLA -> [$(keys on13)]"
echo "   p92v15 TLA -> [$(keys on15)]"
echo "   C++ initializers: $(grep -o 'm_bragg_wide_anchor[a-z_]*{[^}]*}' /home/xqian/toolkit-dev/toolkit/clus/src/CheckSTM_Michel.cxx | tr '\n' ' ')"
echo "C  bare production vs git HEAD's version of the same file:"
git -C $D/.. show HEAD:pdvd/wct-pr-perevt.jsonnet > $W/head-pr-perevt.jsonnet 2>/dev/null
if cmp -s $W/head-pr-perevt.jsonnet $D/wct-pr-perevt.jsonnet; then
    echo "   pdvd/wct-pr-perevt.jsonnet is byte-identical to git HEAD (no jsonnet changed this round)"
else
    echo "   *** pdvd/wct-pr-perevt.jsonnet DIFFERS from git HEAD:"; diff $W/head-pr-perevt.jsonnet $D/wct-pr-perevt.jsonnet | head -20
fi
echo "D  PDHD: bragg_wide lines in its file: $(grep -c 'bragg_wide' $H/wct-pr-perevt.jsonnet)" \
     "| vs git HEAD: $(git -C $H/.. diff --quiet HEAD -- pdhd/wct-pr-perevt.jsonnet && echo unchanged || echo CHANGED)"
echo PROOFS_DONE
