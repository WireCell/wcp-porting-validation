#!/bin/bash
# doc pr/129 -- DISPLAY-ONLY probe arm for the owner's read on the class-A
# objects that pr/128's shipped continuation predicate REJECTS.
#
# These objects are rejections, so a production OFF/ON pair shows nothing
# (identical).  To let the owner see what would be added we force-admit them,
# exactly the way pr/128 displayed 72786's cosmics: keep the production
# operating point but open the two terms that reject them --
#   kink 30 -> 180 deg   (admits 399118 47.3, 393505 41.0/137.3/58.8, 318769 42.0)
#   gap   5 -> 30 cm     (admits 393505 seg 15005 at 28.74 cm)
# end_tol stays at the production 10 cm (every target touches at 0.00/0.00) and
# min_len stays at 30 cm (shortest target 39.8 cm).
#
# THIS IS NOT A PRODUCTION CANDIDATE POINT.  A 180 deg kink also re-admits two
# of 72786's cosmics; that is why the arm runs on these three events only.
#
# Baseline for the A/B is the existing production arms (work-pr128r1-on141-*),
# which are the post-flip operating point -- nothing to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
export SBND_PF_ORPHAN_NEAR_CROSS_CLUSTER=1 SBND_PF_ORPHAN_NEAR_KINK=180 SBND_PF_ORPHAN_NEAR_GAP=30
export SBND_KINE_NEAR_CROSS_CLUSTER=1     SBND_KINE_NEAR_KINK=180     SBND_KINE_NEAR_GAP=30
export WCT_PFNEAR_DEBUG=1
PR_JOBS=2 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
  work-mcp2k-grp0825 work-pr129probe-mcp2k data 393505 318769 \
  > /home/xqian/tmp/pr129_probe_mcp2k.log 2>&1
echo "mcp2k rc=$?"
PR_JOBS=1 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
  work-mcp1k-grp0825 work-pr129probe-mcp1k data 399118 \
  > /home/xqian/tmp/pr129_probe_mcp1k.log 2>&1
echo "mcp1k rc=$?"
