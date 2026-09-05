#!/bin/bash
# doc 100 sec 12 -- regenerate work-dbg25a-ql's IMAGING half, owner-authorised
# 2026-09-05, after doc 100 sec 4 deleted it under work-dbg25a-d97prodchk.
#
#   [D100_JOBS=32] ./scripts/d100_dbg25a_img_regen.sh
#
# WHY A FORK AND NOT scripts/dbg25_run.sh.  That script is the doc-95 record and
# stays byte-untouched, but it cannot be run as-is: line 33 pins
# LD_LIBRARY_PATH to ~/tmp/doc94r3b-libsnap, which THIS ROUND'S ~/tmp sweep
# deleted (it was dropped on the ground "backs work-dbg25a-*, which retire").
# A missing dir in LD_LIBRARY_PATH is silently ignored, so running it unchanged
# would fall back to live local/lib with no warning -- the M1 shape exactly.
# This fork names a fresh pin explicitly and fails if it is absent.
#
# IMAGING ONLY.  work-dbg25a-d97prodchk's 100 dangling links all resolve THROUGH
# work-dbg25a-ql/evt<N>/:
#     d97prodchk/evt16                       -> work-dbg25a-ql/evt16
#     d97prodchk/ql_evt16/icluster-*.npz     -> d97prodchk/evt16/icluster-*.npz
# so recreating evt<N>/ repairs all 100 with NO write into the protected arm.
# The ql_evt<N>/ half of work-dbg25a-ql is NOT rebuilt: nothing points at it,
# and d97prodchk carries its own ql_evt<N>/ already.
#
# OPERATING POINT, STATED.  The original imaging ran 2026-09-02 under
# ~/tmp/doc94r3b-libsnap at ref/prod-2026-09-03.  That pin is gone, so this runs
# on today's local/lib.  It is therefore a REGENERATION, not a byte-restoration,
# until the gate below says otherwise.  The gate is the one number the archived
# record kept: campaign-close-20260904/doc95-debug25/work-dbg25a-ql.manifest.tsv
# says the npz class was 80 files / 105545029 bytes.  Matching that exactly is
# strong evidence the imaging stage did not move; a mismatch must be reported,
# not rounded away.
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
PIN=/home/xqian/tmp/d100regen-libsnap
QL=$BASE/work-dbg25a-ql
IN=$BASE/input_files_reco1/extracted-dbg25a
LOGD=/home/xqian/tmp/d100regen; mkdir -p "$LOGD"

[ -d "$PIN" ] || { echo "REFUSE: binary pin $PIN missing"; exit 2; }
[ -d "$IN" ]  || { echo "REFUSE: input $IN missing"; exit 2; }
[ -e "$QL" ]  && { echo "REFUSE: $QL exists -- fresh label only (M13)"; exit 2; }

nexp=$(tar tjf "$IN/frames-dnn.tar.bz2" | grep -c '^frame_dnnsp_')
echo "=== dbg25a imaging regen: $nexp events, jobs=${D100_JOBS:-32}  $(date -Is)"
echo "    pin  $PIN  libWireCellImg.so md5 $(md5sum $PIN/libWireCellImg.so | cut -c1-16)"

export LD_LIBRARY_PATH=$PIN:${LD_LIBRARY_PATH:-}
export PR_CFG_TREE=$HOME/tmp/dbg25-cfgsnap
export SBND_MAX_JOBS=${D100_JOBS:-32}
export SBND_INPUT_DIR=$IN
export SBND_WORK_ROOT=$QL
mkdir -p "$QL"

# mc, not data: reality=sim throughout, or switch_scope's pos_offset shifts
# every point by ~6.8 cm in y-z (doc 95 header, doc pr/38 round 3).
./run_img_evt.sh mc all > "$LOGD/img-a.log" 2>&1
rc=$?
echo "    img rc=$rc  evt dirs=$(find "$QL" -maxdepth 1 -type d -name 'evt*' | wc -l) / $nexp"

echo "=== GATE vs the archived record ==="
n=$(find "$QL" -name 'icluster-apa*.npz' | wc -l)
b=$(find "$QL" -name 'icluster-apa*.npz' -printf '%s\n' | awk '{s+=$1} END{print s+0}')
echo "    npz files : $n  (archived record: 80)"
echo "    npz bytes : $b  (archived record: 105545029)"
if [ "$n" -eq 80 ] && [ "$b" -eq 105545029 ]; then
    echo "    GATE PASS -- byte-total identical to the deleted arm; imaging did not move"
else
    echo "    GATE MISMATCH -- REPORT THIS, do not round it away."
    echo "    A different byte total means the imaging stage moved between"
    echo "    2026-09-02 and today; the arm is then a fresh product at a new"
    echo "    operating point, which is a physics statement, not a bookkeeping one."
fi
echo "    dangling links in work-dbg25a-d97prodchk: $(find "$BASE/work-dbg25a-d97prodchk" -xtype l | wc -l)  (was 100)"
echo "=== done $(date -Is) rc=$rc"
exit $rc
