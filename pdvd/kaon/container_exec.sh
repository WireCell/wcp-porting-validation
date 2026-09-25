#!/bin/bash
# Run a command inside the fnal-dev-sl7 apptainer (xning's container.sh, exec form,
# with a scratch HOME so nothing is written into ~).
exec /cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer exec \
  --no-home -H /home/xqian/tmp/apptainer_home \
  -B /cvmfs,/nfs,/home/xqian,/home/jjo:/home/jjo:ro,/opt,/etc/hostname,/etc/hosts,/etc/krb5.conf --ipc --pid \
  /cvmfs/singularity.opensciencegrid.org/fermilab/fnal-dev-sl7:latest "$@"
