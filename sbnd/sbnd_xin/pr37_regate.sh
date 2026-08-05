#!/bin/bash
# doc pr/37 sec.2 -- re-gate every surviving 48-event arm pair of pr/30..pr/36
# with pr36_cmp.py, the only comparator in the series that opens
# tracking-pr.root.  No wire-cell run: every artifact already exists.
#
# Why: each round's comparator was blind to the artifact family its successor
# then found necessary (sec.2.1).  pr32_cmp.py imports uproot ZERO times while
# its docstring line 9 advertises a T_tagger/T_kine leaf compare, and it never
# opens mabc-pr.zip -- so pr/29, pr/31 and pr/32's "48/48 byte-identical" cover
# the pctree rollup and two TSVs only.
#
# TRAPS (both cost real time when this was first run):
#   * work-pr31r2-allon48 and -f1on48 hold 1 of 48 tracking-pr.root files -- they
#     are the arms doc pr/31 sec.12 records as CRASHED on the ParticleInfo
#     validator.  Use -allonb48 / -f1onb48, the re-runs.  A hash-only comparator
#     would have compared 48 pctree files and reported a clean PASS.
#   * work-pr32r2-* were produced from work-nuecc48-prod0803 and the pr/31/34/35/36
#     arms from work-nuecc48-0804.  The BAR row below crosses that boundary and
#     comes back 48/48 on every class, which is simultaneously the bar holding
#     and proof that the two input roots are data-equivalent for these events.
#   * NEVER hand-roll np.array_equal over uproot branches to attribute a
#     movement: on jagged/object arrays it manufactures phantoms (~30 of them,
#     sec.6.4).  Use pr36_cmp.py's own _to_py/branch_equal.
#
# Nothing is written into any work-* tree (CLAUDE.md M13).
set -u
cd "$(dirname "$0")" || exit 2

run() { echo "############ $1"; echo "### $2  vs  $3"
        python3 pr36_cmp.py "$2" "$3"; echo "### rc=$?"; echo; }

# --- the byte-identical bar: knob-OFF arm vs the previous round's production ---
run "BAR pr/30 knob-off gate arm"   work-pr30-baseHEAD  work-pr30-final
run "BAR pr/32 off vs pr/31 f2off"  work-pr31-f2off48   work-pr32r2-off48
run "BAR pr/34 off vs pr/31r2 prod" work-pr31r2-prod48  work-pr34-off48
run "BAR pr/35 off vs pr/34 prod"   work-pr34-prod48    work-pr35-off48
run "BAR pr/36 off vs pr/35 prod"   work-pr35-prod48    work-pr36-off48

# --- the knob-ON effect, each round's own pair --------------------------------
run "GEN pr/31r2 off vs off-b (binary-generation control)" \
                                    work-pr31r2-off48   work-pr31r2-off48b
run "ON  pr/31r2 off-b vs allon-b"  work-pr31r2-off48b  work-pr31r2-allonb48
run "ATTRIB pr/31 F3 shower_topo_reset alone" \
                                    work-pr31r2-off48b  work-pr31r2-f3on48
run "ON  pr/32r2 off vs allon"      work-pr32r2-off48   work-pr32r2-allon48
run "ON  pr/34   off vs allon"      work-pr34-off48     work-pr34-allon48
run "ON  pr/35   off vs prod"       work-pr35-off48     work-pr35-prod48

# --- control: pr/36's own pair, already gated with this exact comparator.
#     Must reproduce doc pr/36 sec.11.4 (numu_score 6/48, nue_score 1/48,
#     shw_sp_lol_* on 268067 + 350186, neutrino_type schema-only on all 47)
#     or the instrument is not trustworthy and no row above may be believed.
run "CTRL pr/36 off vs allon"       work-pr36-off48     work-pr36-allon48

# --- sec.6.2: the repeat-run floor at 2457320d.  Regenerate the arms with
#     PR_EXTRA_STAGES=pr_display (the pr/37 round-1 arms emitted no calib dump):
#
#   setarch x86_64 -R env SBND_WCT_LOGLEVEL=debug PR_JOBS=5 PR_EXTRA_STAGES=pr_display \
#     ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-pr37b-repeatA data
#   ... identical -> work-pr37b-repeatB
#
#     Record libWireCellClus.so's mtime BEFORE, BETWEEN and AFTER: a concurrent
#     session moved HEAD under both rounds of this document.
run "FLOOR pr/37b repeatA vs repeatB" work-pr37b-repeatA work-pr37b-repeatB
python3 pr37_repeat_cmp.py work-pr37b-repeatA work-pr37b-repeatB
