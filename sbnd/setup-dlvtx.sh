# DL (SCN) neutrino-vertex environment for the PR chain.  OPT-IN: source this
# AFTER setup-ap.sh, and ONLY for jobs that run 'tagger_check_neutrino'.
#
#   source .../sbnd/setup-ap.sh
#   source .../sbnd/setup-dlvtx.sh
#
# WHAT IT FIXES
#   TaggerCheckNeutrino's DL vertex goes through WCPPyUtil::SCN_Vertex, which
#   embeds CPython and imports torch + sparseconvnet + SCN.DeepVtx.  Without
#   them NeutrinoVertexFinder.cxx catches the ImportError, logs
#     "DL vertex failed: SCN_Vertex: import failed ... No module named 'torch'"
#   at WARN, and SILENTLY falls back to the geometric vertex.  Nothing else in
#   the job changes, so an unprepared environment produces a complete, plausible
#   result computed the wrong way.  Always grep the log (see VERIFY below).
#
# WHERE THE PACKAGES COME FROM
#   The uboone UPS product `scn v01_00_00`, flavor Linux64bit+3.10-2.17.  Its
#   bundled venv is python 3.9.15 -- the SAME version as our container python
#   (wire-cell-python/venv), which is why this works at all: WCPPyUtil embeds
#   libpython, so a venv built against a different python would not load.
#   It carries torch 2.6.0+cu124, numpy 2.0.2 and the sparseconvnet 0.2 egg.
#   We replicate the product's ups/scn.table by hand (two PYTHONPATH prepends)
#   rather than `setup scn`, to avoid pulling a second `python` UPS product on
#   top of the one setup-local-opt.sh already established.
#
#   SCN.DeepVtx and SCN_Vertex.py are OURS, installed by `wcb install` into
#   opt/python -- not part of the UPS product.
#
# LD_PRELOAD IS *NOT* NEEDED HERE.
#   sbnd_xin's run_pr_chain_batch.sh sets LD_PRELOAD=libpython3.11.so.1.0
#   because its toolkit build resolves libpython differently.  Verified
#   2026-08-17 on our build: the full 15-stage PR chain runs with zero
#   "DL vertex failed" and no preload.  If that ever regresses, the preload for
#   THIS python is
#     /cvmfs/larsoft.opensciencegrid.org/products/python/v3_9_15/Linux64bit+3.10-2.17/lib/libpython3.9.so.1.0
#
# COST (1 MC event, 8 cores, 2026-08-17)
#   "overall main vertex" 29 ms -> 6523 ms; full event 14.86 s -> 21.63 s.
#   The DL vertex is ~45% of the event on a cosmic-dominated event.  Budget for
#   it before any scale run.
#
# NUMPY WARNING -- why this is a separate script and not part of setup-ap.sh:
#   this PREPENDS a site-packages carrying numpy 2.0.2, shadowing the 1.24.3 in
#   wire-cell-python's venv for every python started in this shell.  wirecell,
#   h5py and ROOT still import (checked), but do NOT source this in a shell you
#   also use for wire-cell-python tooling or plotting unless you have verified
#   that tooling against numpy 2.
#
# VERIFY after a run:
#   grep -c 'DL vertex failed' <log>          # want 0
#   grep -oE 'overall main vertex took [0-9.]+ ms' <log>   # want seconds, not ~30 ms
#   # or force the geometric vertex for an A/B:  -A dl_weights=''

SBND_SCN_PROD=/cvmfs/uboone.opensciencegrid.org/products/scn/v01_00_00/Linux64bit+3.10-2.17
SBND_SCN_SP="$SBND_SCN_PROD/venv/lib/python3.9/site-packages"

if [ ! -d "$SBND_SCN_SP" ]; then
    echo "setup-dlvtx.sh: ERROR: scn UPS product not found at $SBND_SCN_SP" >&2
    echo "  (is /cvmfs/uboone.opensciencegrid.org mounted?)" >&2
else
    # Mirror ups/scn.table: site-packages then the sparseconvnet egg.
    path-prepend "$SBND_SCN_SP" PYTHONPATH
    path-prepend "$SBND_SCN_SP/sparseconvnet-0.2-py3.9-linux-x86_64.egg" PYTHONPATH
    # Ours: SCN_Vertex.py + SCN/DeepVtx.py, installed by wcb.
    path-prepend /exp/sbnd/app/users/yuhw/opt/python PYTHONPATH
    export PYTHONPATH
    # The SCN net is CPU-only here and torch would otherwise grab every core;
    # SCN_Vertex.py already calls torch.set_num_threads(1), this covers the
    # BLAS underneath it.  Harmless if already set by the harness.
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
fi
