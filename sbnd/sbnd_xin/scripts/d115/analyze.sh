#!/bin/bash
# doc sbnd_xin/115: the analysis stage for one sample, end to end.
#
#   truth_edep.sh      Edep per interaction, from the reco1 files (MC only)
#   build_truth.py     + the sample's GENIE TSV -> truth_base.tsv, cross-checked row by row
#   d115_truth_join.py + the stage-B tracking-pr.root -> candidates.tsv / truth.tsv / summary.txt
#   d107_selection.py  the doc-107 sec 5.5 and 5.6 tables, REUSED UNMODIFIED so the two
#                      records are directly comparable
#   d115_vertex.py     the vertex assessment and its miss taxonomy
#   d115_score_scan.py efficiency/purity (or, with no truth, rate) vs the BDT cut
#   d115_time_assoc.py the flash-time vs true-time offset, an independent check on the join
#   d115_beamoff_rate.py  the beam-off per-gate rate table (data only)
#
# Usage: [JOBS=n] analyze.sh <cv|nuecc|off>
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
S=${1:?usage: analyze.sh <cv|nuecc|off>}
J=${JOBS:-24}
P=$SX/products/d115/$S
case "$S" in
    cv)    ARM=$SX/work-r3cv-d115pr;  MC=1 ;;
    nuecc) ARM=$SX/work-r3nue-d115pr; MC=1 ;;
    off)   ARM=$SX/work-r3off-d115pr; MC=0 ;;
    *) echo "unknown sample: $S" >&2; exit 2 ;;
esac
rc_all=0
step() { echo; echo "=== $* "; }

if [ "$MC" = 1 ]; then
    if [ ! -s "$P/edep_raw.txt" ]; then
        step "truth Edep from the reco1 files"
        JOBS=$J ./scripts/d115/truth_edep.sh "$S" || rc_all=1
    else
        echo "=== truth Edep already extracted -- skipped"
    fi
    step "truth_base.tsv (TSV + Edep, cross-checked)"
    python3 scripts/d115/build_truth.py "$S" || rc_all=1

    step "truth join -> candidates.tsv / truth.tsv / summary.txt"
    python3 d115_truth_join.py --arm "$ARM" --truth "$P/truth_base.tsv" --out "$P" --jobs "$J" || rc_all=1

    step "doc 107 sec 5.5 selection tables"
    python3 d107_selection.py "$P" "$SX/docs/115_sel/$S" || rc_all=1
    step "doc 107 sec 5.6 selection tables (Edep > 100 MeV, nue cut also at 4)"
    python3 d107_selection.py "$P" "$SX/docs/115_sel_edep100/$S" \
        --edep-min 100 --cuts numu:0.9,nue:7.0,nue:4.0 || rc_all=1

    step "vertex assessment"
    python3 d115_vertex.py "$P" "$SX/docs/115_vtx" --label "$S" || rc_all=1
    python3 d115_vertex.py "$P" "$SX/docs/115_vtx" --label "${S}_edep100" --edep-min 100 || rc_all=1

    step "score scan"
    python3 d115_score_scan.py "$P" "$SX/docs/115_scan" --label "$S" || rc_all=1

    step "time-association check (d115 V11)"
    python3 d115_time_assoc.py "$P" "$SX/docs/115_time" --label "$S" || rc_all=1
else
    step "candidate table (no truth: off-beam data)"
    python3 d115_truth_join.py --arm "$ARM" --out "$P" --jobs "$J" || rc_all=1
    # The per-gate denominator comes from the census and nowhere else (see d115_beamoff_rate.py).
    [ -s "$P/file_rse.tsv" ] || { echo "ERROR: no $P/file_rse.tsv -- run rse_census.sh off" >&2; exit 1; }
    step "beam-off per-gate rates"
    python3 d115_beamoff_rate.py "$P" "$SX/docs/115_off" || rc_all=1
    step "score scan (rate vs cut)"
    python3 d115_score_scan.py "$P" "$SX/docs/115_scan" --label off \
        --gates "$(( $(wc -l < "$P/file_rse.tsv") - 1 ))" || rc_all=1
fi
echo
echo "=== analyze $S rc=$rc_all"
exit $rc_all
