#!/bin/bash
# usage: compile_consumers.sh <cfgroot> <outdir>
#
# doc 77 round 2.  Compile EVERY live consumer of the SBND/common clustering
# config against <cfgroot>.  Run it once against a pristine tree
# (git archive HEAD cfg | tar -x -C <dir>) and once against the working tree,
# then cmp_consumers.sh the two output dirs: a cfg refactor is only "no
# behavior change" if all of them are byte-identical.
set -u
CFG=${1:?}; OUT=${2:?}
mkdir -p "$OUT"
AB=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
QL=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/qlport
DATA=/nfs/data/1/xqian/toolkit-dev/wire-cell-data
W=/nfs/data/1/xqian/toolkit-dev/local/bin/wcsonnet

# (a) the broad harness: sbnd pr/img/clus/ql + pdhd + pdvd + sim checks
CFGROOT=$CFG $AB/compile_all_cfg.sh "$OUT" > "$OUT/_compile_all.log" 2>&1
echo "compile_all rc=$?"

# (b) SBND PR job at the PRODUCTION operating point (full PR pipeline + BDTs)
$SX/scripts/cfg/compile_prjob_cfg.sh "$CFG" "$OUT/prod_prjob.json" 2> "$OUT/prod_prjob.err"
echo "prod_prjob rc=$?"

# (c) SBND wcls imaging+clustering and the legacy standalone Q/L job
$SX/scripts/cfg/compile_sbnd_prod.sh "$CFG" "$OUT/prod" > "$OUT/_prod.log" 2>&1
echo "prod_sbnd rc=$?"

# (d) uBooNE MABC (the other caller of common/clus.jsonnet tagger_check_neutrino)
$QL/scripts/compile_ub_cfg.sh "$CFG" "$OUT/uboone.json" 2> "$OUT/uboone.err"
echo "uboone rc=$?"

# (f) The three detectors' TrackFitting parameter JSONs.  doc sbnd_xin/118: these are read at
# RUNTIME by TaggerCheckSTM / CheckSTM_Michel / TaggerCheckNeutrino (Persist::resolve + a plain
# ifstream), never compiled, so every artifact above is blind to them -- doc 118's own flip added
# fit_weight_pow 1.5 / assoc_cont_center 1 to the SBND file and moved ZERO of the 21.  A flip of
# this family is a production operating-point change like any other; hashing the files is what
# makes the tripwire able to see it.  They are copied, not compiled: the gate hashes bytes.
for _det_tf in sbnd/sbnd_track_fitting.json pdhd/pdhd_track_fitting.json \
               protodunevd/pdvd_track_fitting.json; do
    cp -f "$CFG/pgrapher/experiment/$_det_tf" "$OUT/$(basename "$_det_tf")" 2>/dev/null
    echo "trackfit $(basename "$_det_tf") rc=$?"
done

# (g) The LArSoft 1-step chain that actually runs the PR taggers in SBND production.  doc
# sbnd_xin/118: (c) above compiles wcls-img-clus.jsonnet and the standalone Q/L job, NEITHER of
# which calls pr(), so the chain that does was in none of the artifacts.  That is the same shape of
# hole as the runtime fit JSONs in (f): doc 118 had to verify by hand that this chain tracked its
# flip.
#
# doc sbnd_xin/120 moved this file IN-TREE and deleted the generated
# sbnd/pr-operating-point.jsonnet it used to read.  Both edits matter here:
#   * the path below is the in-tree one, NOT the work-dir re-export shim -- hashing the shim would
#     leave the tripwire firing happily while the file production actually reads dropped out of it
#     (doc 119 sec 11.8's trap, same family);
#   * WIRECELL_PATH no longer needs wcp-porting-img/sbnd, because nothing in this chain resolves
#     out of the working repo any more.  Leaving it on would hide exactly the regression where an
#     operating point creeps back into the work dir.
# pr_operating_point is still accepted as an extVar and is now ignored (the mode is retired); it is
# passed below only to prove that an fcl still setting it compiles.
export WIRECELL_PATH=$CFG:$DATA:$DATA/sbnd/photodet
$W -V reality=data -V DL=4.0 -V DT=8.8 -V lifetime=35 -V driftSpeed=1.563 \
   -V semimodel_file="" -V pr_operating_point=sync -V enable_tracking_root=true \
   -V 'input_mask_tags=[]' -V 'output_mask_tags=[]' -V 'recobwire_tags=["gauss"]' \
   -V 'summary_tags=[]' -V 'trace_tags=["gauss"]' \
   -V opflash0_input_label=opflashtpc0 -V opflash1_input_label=opflashtpc1 \
   --ext-code joint=false --ext-code pmt_nl=true \
   "$CFG/pgrapher/experiment/sbnd/wcls-img-clus-matching-xin.jsonnet" \
   > "$OUT/sbnd_larsoft_1step.json" 2> "$OUT/sbnd_larsoft_1step.err"
echo "sbnd_larsoft_1step rc=$?"

# (e) SBND PR job bare (default pipeline, default operating point)
export WIRECELL_PATH=$CFG:$DATA:$DATA/sbnd/photodet
$W -A input=in.tar.gz -A output_dir=out -S run=1 -S subrun=1 -S event=1 -A reality=data \
   "$CFG/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet" > "$OUT/bare_prjob.json" 2> "$OUT/bare_prjob.err"
echo "bare_prjob rc=$?"

# (h) The PROCESS ENVIRONMENT of the three SBND runners -- which allocator the job gets.
# doc sbnd_xin/119 sec 6.4 named this hole and sec 11 closes it.  Round 1 flipped
# libtcmalloc_minimal into the PR chain for -10.9..-16.7 % in-job CPU; that is a production
# operating-point change living in a RUNNER, and no runner is a compiled artifact, so the
# tripwire reported 25/25 while it landed.  The failure it has to catch is not an edit -- it is
# run_pr_chain_batch.sh's `[ -e "$SBND_TCMALLOC_LIB" ]` going false after a package change and
# the job dropping to glibc with (before round 5) no log line at all.
#
# MINIMAL ON PURPOSE (sec 6.4's own condition).  These runners are large and churn constantly
# with A/B scaffolding; hashing them whole would make the tripwire noisy enough to be ignored.
# So this extracts only the lines that DECIDE the allocator, with leading whitespace and comments
# stripped and no line numbers, and hashes that.  Moving the block, re-indenting it or rewriting
# its comments does not fire; changing which library is preloaded, or deleting the round-5
# warning that makes a fallback visible, does.
#
# `^PYLIB=` is in the pattern ON PURPOSE and is not a false positive waiting to happen.  It pins
# libpython3.11.so.1.0 by soname, and a python bump moving that string IS a production change of
# exactly this family: doc sbnd_xin/118 sec 8 records that dropping $PYLIB from the preload
# silently disables the SCN import, so the job runs the GEOMETRIC vertex instead of the DL one.
# That is a worse silent regression than an allocator swap.  Do not narrow the pattern to make an
# upgrade quieter.
#
# doc sbnd_xin/119 sec 11.8 added SBND_PR_ALLOC_LIB to the alternation WITH the flip, not after it.
# The preload sites read $SBND_TCMALLOC_LIB either way, so the LD_PRELOAD lines would have kept
# matching and this artifact would have gone on hashing happily -- while the line that names the
# library actually loaded silently dropped out of it.  A tripwire that still fires but no longer
# contains the fact is worse than one that breaks.
{
    for _r in run_pr_chain_batch.sh run_pr_evt.sh run_clus_evt.sh; do
        echo "## $_r"
        grep -hE '(LD_PRELOAD|SBND_PR_ALLOC_LIB|SBND_TCMALLOC_LIB|SBND_PR_TCMALLOC|TCMALLOC_SO|WCT_TCMALLOC|WC_PRELOAD|^PYLIB=)' \
             "$SX/$_r" | grep -vE '^[[:space:]]*#' | sed 's/^[[:space:]]*//'
    done
} > "$OUT/runner_alloc.txt" 2> "$OUT/runner_alloc.err"
echo "runner_alloc rc=$?"
