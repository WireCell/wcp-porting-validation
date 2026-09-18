#!/bin/bash
# sbnd_xin/docs/109 rev 4: the DL-vertex arms of the nu_bundle_flash_group gate.
#
# The byte-gate arms run the GEOMETRIC vertex (SBND_NO_DL=1) because the DL
# vertex is not bit-stable (CLAUDE.md M4).  Production runs the DL vertex, and
# it is the DL path (determine_overall_main_vertex_DL -> swap_main_cluster)
# that can move the main onto a companion -- exactly what a merged bundle
# whose longer activity is the muon half needs.  So the eligible events are
# run once more with the production vertex, knob off and knob on, and graded
# on CONTENT (scripts/d109r4_group_census.py), never on bytes.
#
# Usage: scripts/d109r4_dl_arm.sh <label> <libsnap> <0|1 knob> [VAR=value ...]
set -u
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
LABEL=${1:?usage: d109r4_dl_arm.sh <label> <libsnap> <0|1> [VAR=value ...]}
PIN=${2:?}
KNOB=${3:?}
shift 3
exec "$SX/scripts/d109r4_arms.sh" "$LABEL" "$PIN" eligible "SBND_NU_BUNDLE_FLASH_GROUP=$KNOB" "$@"
