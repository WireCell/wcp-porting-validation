# AP-branch env for the wcls imaging+clustering+matching chain.
#
# Source this INSIDE the SL7 apptainer (it builds on setup-local-opt.sh: the
# /exp/sbnd/app/users/yuhw/opt install built from wire-cell-toolkit
# `apply-pointcloud`).  It additionally prepends sbnd_xin/ to WIRECELL_PATH so
# the matching jsonnet (wcls-img-clus-matching.jsonnet) and its helpers resolve
# from there.
#
# Usage (from a login shell, in the sbnd/ dir):
#   /cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer exec \
#       -B /cvmfs,/exp,/nashome,/pnfs \
#       /cvmfs/singularity.opensciencegrid.org/fermilab/fnal-dev-sl7:latest bash
#   # then inside the container:
#   source /nashome/y/yuhw/.bashrc
#   source /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/setup-ap.sh
#   time lar --nskip 0 -n 1 -c wcls-img-clus-matching.fcl \
#       -s standalone-sample/2025f-mc.root --no-output >& wcls-img-clus-matching.log

SBND_DIR="/exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd"

source "$SBND_DIR/setup-local-opt.sh"

# Xin's matching/imaging chain uses the TOOLKIT cfg (img.jsonnet multi-3view +
# full_deghost, clus.jsonnet, qlmatching.jsonnet, cathode_fiducial.jsonnet,
# simparams), NOT sbndcode's older fork.  setup-local-opt.sh prepends sbndcode
# cfg (so it wins by default); prepend the toolkit cfg here so it wins for this
# AP chain instead.  Both use the same wires geometry
# (sbnd-wires-geometry-v0206.json.bz2), so the artROOT channel<->anode mapping is
# unchanged.  Use this setup ONLY for the AP matching/imaging job, not for sim.
path-prepend /exp/sbnd/app/users/yuhw/wire-cell-toolkit/cfg WIRECELL_PATH

# QLMatching reads the photo-detector semi-analytical model by bare name
# (semi-analytical-sbnd.json); it lives under wire-cell-data/sbnd/photodet.
path-prepend /exp/sbnd/app/users/yuhw/wire-cell-data/sbnd/photodet WIRECELL_PATH
