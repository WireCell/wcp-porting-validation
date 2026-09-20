#!/bin/bash
# doc sbnd_xin/116: the doc-115 analysis stage, run on ONE study cell's stage-B arm.
# Fork by duplication (CLAUDE.md M10) of scripts/d115/analyze.sh, which stays untouched.
#
# Differences from the doc-115 driver, and why:
#   - the truth build (truth_edep.sh + build_truth.py) is NOT re-run: products/d115/<s>/truth_base.tsv
#     is reco-free (GENIE TSV + Edep from the reco1 files) and is reused as-is.  truth.tsv is NOT
#     reusable -- it carries event_has_candidate / n_candidates / min_cand_dist_cm from the arm -- so
#     the join regenerates it per cell;
#   - every output goes under a per-cell name: products/d116/<s>-<cell>/, docs/116_sel/<s>-<cell>/,
#     docs/116_sel_edep100/<s>-<cell>/, docs/116_{vtx,scan,time}/ with --label <s>-<cell>,
#     docs/116_off/<cell>/;
#   - the beam-off gate census (file_rse.tsv) is a property of the sample, copied from products/d115/off.
# The analysis scripts themselves (d115_truth_join.py, d107_selection.py, d115_vertex.py,
# d115_score_scan.py, d115_time_assoc.py, d115_beamoff_rate.py) are the doc-115 ones, unmodified,
# so every table is directly comparable to docs/115_*.
#
# Usage: [JOBS=n] analyze_cell.sh <cell> <cv|nuecc|off>
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
CELL=${1:?usage: [JOBS=n] analyze_cell.sh <cell> <cv|nuecc|off>}
S=${2:?usage: [JOBS=n] analyze_cell.sh <cell> <cv|nuecc|off>}
J=${JOBS:-24}
P=$SX/products/d116/$S-$CELL
P115=$SX/products/d115/$S
case "$S" in
    cv)    ARM=$SX/work-r3cv-d116$CELL;  MC=1 ;;
    nuecc) ARM=$SX/work-r3nue-d116$CELL; MC=1 ;;
    off)   ARM=$SX/work-r3off-d116$CELL; MC=0 ;;
    *) echo "unknown sample: $S" >&2; exit 2 ;;
esac
[ -d "$ARM" ] || { echo "ERROR: no stage-B arm $ARM" >&2; exit 1; }
[ "$(cat "$ARM/.d116_cell" 2>/dev/null)" = "$CELL" ] || { echo "ERROR: $ARM is not a '$CELL' arm" >&2; exit 1; }
mkdir -p "$P"
rc_all=0
step() { echo; echo "=== $* "; }
L=$S-$CELL

if [ "$MC" = 1 ]; then
    [ -s "$P115/truth_base.tsv" ] || { echo "ERROR: no $P115/truth_base.tsv (doc 115 truth build)" >&2; exit 1; }
    step "truth join -> candidates.tsv / truth.tsv / summary.txt  (truth_base.tsv reused from doc 115)"
    python3 d115_truth_join.py --arm "$ARM" --truth "$P115/truth_base.tsv" --out "$P" --jobs "$J" || rc_all=1

    step "doc 107 sec 5.5 selection tables"
    python3 d107_selection.py "$P" "$SX/docs/116_sel/$L" || rc_all=1
    step "doc 107 sec 5.6 selection tables (Edep > 100 MeV, nue cut also at 4)"
    python3 d107_selection.py "$P" "$SX/docs/116_sel_edep100/$L" \
        --edep-min 100 --cuts numu:0.9,nue:7.0,nue:4.0 || rc_all=1

    step "vertex assessment"
    python3 d115_vertex.py "$P" "$SX/docs/116_vtx" --label "$L" || rc_all=1
    python3 d115_vertex.py "$P" "$SX/docs/116_vtx" --label "${L}_edep100" --edep-min 100 || rc_all=1

    step "score scan"
    python3 d115_score_scan.py "$P" "$SX/docs/116_scan" --label "$L" || rc_all=1

    step "time-association check (d115 V11)"
    python3 d115_time_assoc.py "$P" "$SX/docs/116_time" --label "$L" || rc_all=1
else
    step "candidate table (no truth: off-beam data)"
    python3 d115_truth_join.py --arm "$ARM" --out "$P" --jobs "$J" || rc_all=1
    [ -s "$P115/file_rse.tsv" ] || { echo "ERROR: no $P115/file_rse.tsv -- the doc-115 beam-gate census" >&2; exit 1; }
    cp -p "$P115/file_rse.tsv" "$P/file_rse.tsv"
    step "beam-off per-gate rates"
    python3 d115_beamoff_rate.py "$P" "$SX/docs/116_off/$CELL" || rc_all=1
    step "score scan (rate vs cut)"
    python3 d115_score_scan.py "$P" "$SX/docs/116_scan" --label "$L" \
        --gates "$(( $(wc -l < "$P/file_rse.tsv") - 1 ))" || rc_all=1
fi
echo
echo "=== analyze $L rc=$rc_all"
exit $rc_all
