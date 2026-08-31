#!/bin/bash
# doc pr/136 round 2 -- launch the three arms of the pass-4 angle_v1 escape.
#
#   off1     new binary, probes, NO knob env  -> the byte-identity gate against
#            work-pr136-f086probe-* (which is itself gated 478/478 against
#            work-pr134-f086-*).  Gating against the PROBE arm rather than the
#            no-probe one is the tighter comparison and makes the two arms'
#            emprep sidecars directly diffable.
#   onV1     escape ON, no angle_v2 ceiling      (810 predicted fires, 123 evts)
#   onV1c90  escape ON, angle_v2 ceiling 90 deg  (350 predicted fires,  92 evts)
#
# Run concurrently at PR_JOBS=16 each (48 of 64 cores; check loadavg after, M5).
# Each arm covers the full 239-event manifest (98-set then 141-set).
set -u
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
J=${PR_JOBS:-16}
run() {  # run <TAG> [KEY=VAL ...]
  local tag=$1; shift
  PR_JOBS=$J ./scripts/pr136_arms.sh 98  "$tag" 1 "$@"
  PR_JOBS=$J ./scripts/pr136_arms.sh 141 "$tag" 1 "$@"
  echo "ARM $tag COMPLETE"
}
case "${PR136_ROUND:-2}" in
2)
  run off1                                              > /home/xqian/tmp/pr136_r2_off1.log    2>&1 &
  run onV1    SBND_PASS4_V1_ESCAPE=1                    > /home/xqian/tmp/pr136_r2_onV1.log    2>&1 &
  run onV1c90 SBND_PASS4_V1_ESCAPE=1 SBND_PASS4_V1_MAXV2=90 > /home/xqian/tmp/pr136_r2_onV1c90.log 2>&1 &
  ;;
3)
  # round 3: the proximity brake.  off2 re-gates the THREE-knob binary (the
  # round-2 gate was run on the two-knob one; a default-0 knob that is only read
  # inside the unreachable escape branch cannot change the OFF path, but the
  # house rule is a gate, not an argument).
  run off2                                              > /home/xqian/tmp/pr136_r3_off2.log 2>&1 &
  run onV1c90d25 SBND_PASS4_V1_ESCAPE=1 SBND_PASS4_V1_MAXV2=90 SBND_PASS4_V1_MAXDIS=25 \
                                                        > /home/xqian/tmp/pr136_r3_d25.log  2>&1 &
  ;;
esac
wait
echo "ALL THREE ARMS DONE"
