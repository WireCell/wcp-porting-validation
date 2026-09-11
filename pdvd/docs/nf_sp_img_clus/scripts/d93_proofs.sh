#!/usr/bin/env bash
# doc pdvd/93 -- the compiled-config proofs for the flip of doc 92's wide Bragg-peak read in
# pdvd/wct-pr-perevt.jsonnet.  Three NEW keys, none present in the compiled config before:
#   bragg_wide_anchor_cm       0.0 -> 8.0   (the master switch; C++ default 0.0 = off)
#   bragg_wide_anchor_rise_min 1.5 -> 1.3   (a real change from the C++ default)
#   bragg_wide_anchor_tail_max 0.8 -> 0.8   (equal to the C++ default: an INERT key)
#
# All three are written even though tail_max is inert, because proof A is what the flip rests on:
# the arm p92v13 that MEASURED the +4/-1 trade ran the TLA
#   {bragg_wide_anchor_cm:8.0,bragg_wide_anchor_rise_min:1.3,bragg_wide_anchor_tail_max:0.8}
# and "the flipped file compiles to exactly what the measured arm ran" is only a 0-line claim if the
# file carries the same three keys.  Docs 70/90 left their minima unset for the inert-key reason --
# but those keys were NOT in their arm's TLA.  Ours is.  The inert lines move to proof B instead.
#
#   A  PRE + `-S stm_michel_extra={TLA}` (the arm's config) vs POST unset -> 0 lines
#   B  POST + all three forced back to their C++ initializers vs PRE -> exactly three lines, each
#      a key present at its C++ default vs absent.  The inert-key form of docs 58 / 61 / 82: the
#      component reads the same number either way.  The initializers are grepped and printed here.
#   C  PRE vs POST -> exactly the three keys
#   D  PDHD untouched: pdhd/wct-pr-perevt.jsonnet unchanged against git HEAD and carrying no
#      bragg_wide_anchor key at all (PDHD stays OFF).
# Fork by duplication (CLAUDE.md M10) of d90_proofs.sh; the keys, W and the default TLA changed.
set -u
: "${PRE_EVT:?}"
TLA=${TLA:-"bragg_wide_anchor_cm:8.0,bragg_wide_anchor_rise_min:1.3,bragg_wide_anchor_tail_max:0.8"}
OFF=${OFF:-"bragg_wide_anchor_cm:0.0,bragg_wide_anchor_rise_min:1.5,bragg_wide_anchor_tail_max:0.8"}
D=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
H=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd
SRC=/home/xqian/toolkit-dev/toolkit/clus/src/CheckSTM_Michel.cxx
W=/home/xqian/tmp/p93/proofs; mkdir -p $W
PJ='["switch_scope","flag_mains","unmerge_assoc","steiner","fiducialutils","tagger_check_tgm","tagger_check_stm","tagger_check_fc","protect_bundle","steiner_refresh","check_stm_michel","tracking_visitor","pr_display"]'
comp() {   # out src [extra-args...]
    local out=$1 src=$2; shift 2
    rm -f $W/$out.json
    (cd $D && wcsonnet -P $D -A input=x.tar.gz -A output_dir=$W -S run=39252 -S subrun=0 -S event=1 \
        -S pipeline_names="$PJ" "$@" -o $W/$out.json "$src") > $W/$out.log 2>&1
    local rc=$?
    echo "  compile $out rc=$rc size=$(stat -c %s $W/$out.json 2>/dev/null)"
    if [ $rc != 0 ] || [ ! -s $W/$out.json ]; then echo "ABORT: $out did not compile (see $W/$out.log)"; exit 2; fi
}
pj() { python3 -m json.tool --sort-keys $W/$1.json; }
comp pre      "$PRE_EVT"
comp pre_on   "$PRE_EVT" -S "stm_michel_extra={$TLA}"
comp post     wct-pr-perevt.jsonnet
comp post_off wct-pr-perevt.jsonnet -S "stm_michel_extra={$OFF}"
echo "A  PRE + the arm's TLA ON vs POST unset: $(diff <(pj pre_on) <(pj post) | grep -c '^[<>]') lines  (0 == the flipped file runs exactly what p92v13 ran)"
echo "B  POST + all three forced back vs PRE:"; diff <(pj pre) <(pj post_off) | grep '^[<>]'
echo "   C++ initializers:"; grep -o 'm_bragg_wide_anchor_[a-z_]*{[^}]*}' $SRC | sed 's/^/     /'
echo "C  PRE vs POST:"; diff <(pj pre) <(pj post) | grep '^[<>]'
echo "   keys in the compiled POST: $(grep -o '"bragg_wide_anchor_[a-z_]*" *: *[0-9.]*' $W/post.json | sort -u | tr '\n' ' ')"
git -C $H/.. diff --quiet HEAD -- pdhd/wct-pr-perevt.jsonnet; echo "D  PDHD file vs git HEAD: $([ $? = 0 ] && echo unchanged || echo CHANGED)" \
    "| bragg_wide_anchor lines in PDHD: $(grep -c 'bragg_wide_anchor' $H/wct-pr-perevt.jsonnet)"
echo PROOFS_DONE
