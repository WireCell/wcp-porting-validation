#!/bin/bash
# valfast A/B report: compare two valfast arms over the 629-event subset.
#
# Usage: valfast/valfast_compare.sh <tagA> <tagB> [sample ...]
#   samples default: mcp1k nuecc48 r1qlmc r2mc ncpi0 (ncpi0 added 2026-08-05,
#   doc 71 campaign -- no legacy history, see run_valfast.sh header)
#
# Per sample, three gates (report by label -- quote this output in the doc):
#   1. PR archives + trees + calib: vf_tree_compare_all.py, HARD
#      (hash_archive.py member hashes of mabc-pr.zip + pctree-pr-evt*.tar.gz;
#       EXACT uproot compare of ALL SEVEN trees in tracking-pr.root; and
#       calib-pr-evt<ID>.json split into tagger/kine/other keys)
#   2. physics columns: vf_scores_diff.py on the two vf-scores.tsv
#      (timing/RSS columns excluded by name; kine_reco_Enu_MeV compared with
#       1e-5 relative tolerance -- the chain's documented noise floor)
#   3. nusel-side archives (only when BOTH tags were run -full and the nusel
#      roots exist): nusel_hash_compare.py on mabc-all-apa.zip +
#      pctree-evt*.tar.gz + nusel_evt*/mabc-pr.zip
# Exit 0 iff every gate that ran is identical. A knob-ON arm is EXPECTED to
# differ -- this script reports, the doc interprets.
#
# Gate 1 was INFORMATIONAL and covered T_tagger/T_kine as sorted multisets until
# 2026-08-05.  doc pr/37 sec.2.5 re-measured the A/A' floor at toolkit 2457320d:
# 0 of 629 events differ, EXACT, on all seven trees, in BOTH a matched-layout
# pair (setarch -R twice) and a cross-layout pair (setarch -R vs ASLR on).  The
# instability that forced the multiset compare is gone, so the gate is now wide
# and hard.  VF_CMP_LEGACY=1 restores the old path exactly.
#
# Build arms with PR_EXTRA_STAGES=pr_display or the calib channel is an
# absent-side skip and pr/36 F1's match_isFC stays invisible (it is not booked
# in T_tagger).
set -u
SBND_DIR=$(cd "$(dirname "$0")/.." && pwd -P)
VF=$SBND_DIR/valfast
cd "$SBND_DIR" || exit 1

A=${1:?usage: valfast_compare.sh <tagA> <tagB> [sample ...]}
B=${2:?usage: valfast_compare.sh <tagA> <tagB> [sample ...]}
shift 2
SAMPLES=${*:-"mcp1k nuecc48 r1qlmc r2mc ncpi0"}
# Gate 1 is WIDE and HARD by default since 2026-08-05 (doc pr/37 sec.2.5
# measured the floor at 0/629 on BOTH layouts).  VF_CMP_LEGACY=1 restores the
# pre-promotion path exactly -- T_tagger/T_kine only, vectors as multisets,
# tree result INFORMATIONAL -- so every number in valfast/README.md and in the
# round docs stays reproducible.
if [ -n "${VF_CMP_LEGACY:-}" ]; then CMP=vf_tree_compare.py; WIDE=; else CMP=vf_tree_compare_all.py; WIDE=1; fi

nusel_root() {
    case "$1" in
        mcp1k)   echo "work-mcp1kall-vf$2";;
        nuecc48) echo "work-nuecc48-vf$2";;
        r1qlmc)  echo "work-r1qlmc-vf$2";;
        r2mc)    echo "work-r2mc-vf$2";;
        ncpi0)   echo "work-ncpi0-vf$2";;
    esac
}

fail=0
for s in $SAMPLES; do
    ra=work-vf$s-$A; rb=work-vf$s-$B
    echo "=================== [$s] $ra vs $rb ==================="
    if [ ! -d "$ra" ] || [ ! -d "$rb" ]; then
        echo "[$s] MISSING ARM ($ra or $rb) -- FAIL"; fail=1; continue
    fi
    # 1. PR archives + all seven trees + calib -- ALL HARD (doc pr/37 sec.2.5).
    #    Under VF_CMP_LEGACY=1 the tree half reverts to INFORMATIONAL, because
    #    the 2026-08-02 A/A' measured fill-order permutations and occasional
    #    value flips (shw_sp_lol_1_v_angle on the vfsmk arms) in the T_tagger
    #    per-candidate feature branches.  That is history, not current
    #    behaviour: the 2026-08-05 floor is 0/629 exact on both layouts.
    # shellcheck disable=SC2046
    out1=$(python3 "$VF/$CMP" "$ra" "$rb" $(tr '\n' ' ' < "$VF/events-$s.txt"))
    rc1=$?
    echo "$out1"
    if echo "$out1" | grep -q 'mabc=≠\|pctree=≠\|MISSING'; then
        echo "[$s] ARCHIVE DIFF -- hard gate FAIL"; fail=1
    elif [ $rc1 -ne 0 ]; then
        if [ -n "$WIDE" ]; then
            echo "[$s] TREE/CALIB DIFF -- hard gate FAIL (wide comparator)"; fail=1
        else
            echo "[$s] tree-feature instability only (archives identical) -- informational, see valfast/README.md"
        fi
    fi
    # 2. physics columns (vf_scores_diff.py = pr20_scores_diff.py fork with a
    #    1e-5 relative tolerance on kine_reco_Enu_MeV ONLY -- the documented
    #    run-to-run noise floor of the PR chain; everything else exact)
    if [ -f "$ra/vf-scores.tsv" ] && [ -f "$rb/vf-scores.tsv" ]; then
        python3 "$VF/vf_scores_diff.py" "$ra/vf-scores.tsv" "$rb/vf-scores.tsv"
        rc2=$?; [ $rc2 -eq 0 ] || fail=1
        echo "[$s] scores-diff rc=$rc2"
    else
        echo "[$s] vf-scores.tsv missing in one arm -- FAIL"; fail=1
    fi
    # 3. nusel-side archives (-full arms only)
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
