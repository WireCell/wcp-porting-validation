#!/bin/bash
# doc pr/139 phase 1 -- the four ON arms, ONE KNOB EACH, run sequentially.
# Each knob is measured ALONE before any combination (doc pr/139 sec 1 bar,
# and doc pr/138 B4's reason: a gate answering two questions answers neither).
set -u
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# P1.2 -- the impact-parameter veto at 12 cm.  WCT_SHOWER_SPLIT_DEBUG is on for
# THIS arm only: the tape now carries b_cm and veto, so the C++ impact parameter
# can be checked against pr139_pointing.py on the same arm (the pr/138 B1
# discipline).  The probe is stderr-only and changes no decision.
( export SBND_SHOWER_SPLIT_MAX_IMPACT=12 WCT_SHOWER_SPLIT_DEBUG=1
  ./scripts/pr139_arms.sh onb12 )

# P1.1 -- refuse a peel of segments another shower also owns
( export SBND_SHOWER_SPLIT_SKIP_SHARED=1
  ./scripts/pr139_arms.sh onshared )

# P1.3 -- seed the daughter on its nearest EM-typed member
( export SBND_SHOWER_SPLIT_EM_START=1 WCT_SHOWER_SPLIT_DEBUG=1
  ./scripts/pr139_arms.sh onemst )

# P1.4 -- re-home the orphan daughter
( export SBND_SHOWER_SPLIT_REHOME=1 WCT_SHOWER_SPLIT_DEBUG=1
  ./scripts/pr139_arms.sh onrehome )

echo "ALL ON ARMS DONE"
