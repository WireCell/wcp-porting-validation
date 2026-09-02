#!/bin/bash
# doc 96 -- capture the ClusteringSeparate gate quantities for the owner's
# three doc-95 findings.  WCT_SEP_DEBUG=1 is a permanent, log-only env hatch in
# clus/src/clustering_separate.cxx:30 ("kept permanently because separation
# triggers depend on detector FV tuning and re-diagnosing them needs these
# numbers"); it prints SEPDBG lines on std::cout and changes no behaviour.
#
# Fresh work root again (M13) so the doc-96 trace layers in
# work-dbg25a-trace95 are not overwritten.  Imaging symlinked (M11).
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export LD_LIBRARY_PATH=$HOME/tmp/doc94r3b-libsnap:${LD_LIBRARY_PATH:-}
export SBND_INPUT_DIR=$BASE/input_files_reco1/extracted-dbg25a
export SBND_WORK_ROOT=$BASE/work-dbg25a-sepdbg95
export WCT_SEP_DEBUG=1
LOGD=/home/xqian/tmp/d96; mkdir -p "$LOGD"
mkdir -p "$SBND_WORK_ROOT"
for e in 30 21 5; do ln -sfn "$BASE/work-dbg25a-ql/evt$e" "$SBND_WORK_ROOT/evt$e"; done

for spec in 12:30 6:21 19:5; do
    idx=${spec%%:*}; evt=${spec##*:}
    echo "=== evt$evt (idx $idx)  $(date -Is)"
    setarch x86_64 -R ./run_ql_evt.sh mc "$idx" > "$LOGD/sepdbg-evt$evt.log" 2>&1
    echo "   rc=$?  SEPDBG lines: $(grep -c SEPDBG "$LOGD/sepdbg-evt$evt.log" 2>/dev/null) in runner log, $(grep -c SEPDBG "$SBND_WORK_ROOT/ql_evt$evt/wct_ql_evt$evt.log" 2>/dev/null) in wct log"
done
echo "=== done $(date -Is)"
