#!/bin/bash
# doc pr/129 -- the owner's ruling, implemented as an arm to look at.
#   "all three should be reject, overclustering, so not counting in the enr
#    eneryg, OK tobe in PR, I guess."  (2026-08-29)
# PF display and Enu accounting are separate concerns, and the codebase already
# splits them into separate knobs.  So this arm keeps pf_orphan_guard_freed ON
# (the object still appears in the PR tree) and turns kine_count_guard_freed
# OFF (it stops entering kine_reco_Enu).  No new code -- an existing knob.
#
# Population: kine_count_guard_freed counts exactly 3 objects across all 239
# events of both manifests, 710.66 MeV -- 393505 268.70 (owner-adjudicated
# reject), 171572 304.75, 94392 137.21.  393505 is included as the calibration
# reference; 171572 and 94392 are the two that still need a read.
set -euo pipefail
cd "$(dirname "$0")/.."
export SBND_KINE_COUNT_GUARD_FREED=0     # Enu: stop counting
export SBND_PF_ORPHAN_GUARD_FREED=1      # PR tree: keep showing (unchanged)
PR_JOBS=3 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
  work-mcp2k-grp0825 work-pr129gf-nokine-mcp2k data 94392 171572 393505 \
  > /home/xqian/tmp/pr129_gf_nokine.log 2>&1
echo "rc=$?"
