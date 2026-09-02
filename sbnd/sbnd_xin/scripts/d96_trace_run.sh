#!/bin/bash
# doc 96 -- per-clustering-step Bee trace for the owner's three doc-95 scan
# findings (272-2-30 "separation not working", 105-23-5 "missing a long
# track", 105-23-21 "separation not working").
#
# Re-runs ONLY the Q/L stage of group a with -trace-bee (doc 51): one extra
# Bee "clustering" layer per clustering visitor, named tr<NN>_<Type>, in the
# per-APA zips AND in mabc-all-apa.zip.  Knob is default OFF in the config, so
# production is untouched; this writes into a FRESH work root (M13) and
# symlinks the production imaging in (M11 -- same charge, our own pipeline).
#
# Group-a indices (from `./run_ql_evt.sh mc` with SBND_INPUT_DIR set):
#   evt30 -> 12   evt21 -> 6   evt5 -> 19
#
# Binary pinned to the same snapshot the doc-95 arm used
# (~/tmp/doc94r3b-libsnap, md5-verified equal to local/lib at 2026-09-02 09:00).
# setarch -R for determinism (M4).
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export LD_LIBRARY_PATH=$HOME/tmp/doc94r3b-libsnap:${LD_LIBRARY_PATH:-}
export SBND_INPUT_DIR=$BASE/input_files_reco1/extracted-dbg25a
export SBND_WORK_ROOT=$BASE/work-dbg25a-trace95
LOGD=/home/xqian/tmp/d96; mkdir -p "$LOGD"

for spec in 12:30 6:21 19:5; do
    idx=${spec%%:*}; evt=${spec##*:}
    echo "=== evt$evt (idx $idx)  $(date -Is)"
    setarch x86_64 -R ./run_ql_evt.sh mc -save-pctree -trace-bee "$idx" \
        > "$LOGD/ql-evt$evt.log" 2>&1
    echo "   rc=$?  $(ls -la "$SBND_WORK_ROOT/ql_evt$evt/mabc-all-apa.zip" 2>/dev/null | awk '{print $5}') bytes"
done
echo "=== done $(date -Is)"
