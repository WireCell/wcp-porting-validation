#!/bin/bash
# Revert-proof for clus/test/doctest_pr11_guards.cxx.
#
# For each guard: revert the production file(s) to origin/master (an exact,
# guard-only revert -- every one of these files' branch diff IS the guard),
# rebuild, run the pinning test case, then restore.
#
# NOTE the filter is -tc (TEST CASE name).  A first pass of this script used
# -sf, which is doctest's SOURCE FILE filter: it matched nothing, ran zero
# cases, exited 0, and reported a bogus PASS for every guard.  run_case()
# therefore now also requires a non-zero executed-case count.
#
# SAFETY (why this is safe in the primary shared tree):
#   * `wcb install` is NEVER run here, so local/lib -- the path wire-cell
#     actually loads (M1) -- keeps the apply-pointcloud build throughout.  No
#     other session's RUNTIME can observe any of this.
#   * one guard reverted at a time, restored immediately after.
#   * the restore is an EXIT trap, so an interrupt cannot leave a revert behind.
#   * the caller runs `wcbuild` + the freshness proof afterwards.
set -u
T=/nfs/data/1/xqian/toolkit-dev/toolkit
SP="$(cd "$(dirname "$0")" && pwd)"
cd "$T" || exit 1

FILES="clus/src/DynamicPointCloud.cxx clus/src/Facade_Cluster.cxx clus/src/FiducialUtils.cxx clus/src/PRShower.cxx clus/inc/WireCellClus/PRShower.h"

restore_all() {
    echo "=== RESTORE all production files ==="
    for f in $FILES; do git checkout HEAD -- "$f" 2>/dev/null; done
    git status --porcelain -- clus/ | sed 's/^/  /'
}
trap restore_all EXIT

run_case() {   # $1 = doctest -tc filter
    ./wcb build --notests -p > "$SP/rp_build.log" 2>&1
    if [ $? -ne 0 ]; then echo "BUILD_FAILED"; return; fi
    (cd build && ./clus/wcdoctest-clus -tc="$1" > "$SP/rp_test.log" 2>&1)
    local trc=$?
    # "[doctest] test cases: N | P passed | F failed | S skipped"
    local n
    n=$(grep -oE "test cases: *[0-9]+" "$SP/rp_test.log" | grep -oE "[0-9]+" | head -1)
    n=${n:-0}
    if [ "$n" -eq 0 ]; then echo "NO_CASES_RAN(filter bug)"; return; fi
    if [ $trc -eq 0 ]; then echo "PASS ($n case)"; else echo "FAIL ($n case)"; fi
}

proof() {      # $1 = label   $2 = files to revert   $3 = -tc filter
    echo "############################################################"
    echo "### GUARD: $1"
    echo "###   revert: $2"
    echo "###   -tc:    $3"
    for f in $2; do
        git checkout origin/master -- "$f" || { echo "  revert of $f FAILED"; return; }
    done
    echo -n "  WITHOUT guard: "
    res=$(run_case "$3")
    echo "$res"
    case "$res" in
        BUILD_FAILED) grep -E "error:|undefined reference" "$SP/rp_build.log" | head -3 | sed 's/^/    /' ;;
        FAIL*)        grep -E "ERROR:|THREW|is NOT correct|values:" "$SP/rp_test.log" | head -6 | sed 's/^/    /' ;;
    esac
    for f in $2; do git checkout HEAD -- "$f"; done
    echo -n "  WITH guard:    "
    run_case "$3"
}

proof "1a. DPC self-append (cloud identity)" \
      "clus/src/DynamicPointCloud.cxx" \
      "pr11 dpc self-append via cloud identity is a no-op"

proof "1b. DPC self-append (batch identity)" \
      "clus/src/DynamicPointCloud.cxx" \
      "pr11 dpc self-append via batch identity is a no-op"

proof "3. Cluster all-excluded extremes fallback" \
      "clus/src/Facade_Cluster.cxx" \
      "pr11 extremes fall back when every point is excluded"

# KNOWN TO REPORT BUILD_FAILED -- kept to document why.  Reverting
# PRShower.h too is required for PRShower.cxx to compile, but its 2-line
# branch change (get_stem_dQ_dx / update_particle_type signatures) is
# consumed by NeutrinoTaggerNuE.cxx and NeutrinoTaggerSinglePhoton.cxx, so
# the whole-file revert breaks 3 unrelated TUs.  The WORKING proof for this
# guard is the surgical revert in revert_proof4.sh -- run that one.
proof "4. Shower add_segment clone + idempotence (expect BUILD_FAILED; see revert_proof4.sh)" \
      "clus/src/PRShower.cxx clus/inc/WireCellClus/PRShower.h" \
      "pr11 shower add_segment is idempotent and does not alias"

proof "5. inside_dead_region sentinel apa/face" \
      "clus/src/FiducialUtils.cxx" \
      "pr11 inside_dead_region rejects sentinel apa/face before dereferencing"

echo "############################################################"
echo "### GUARD 2 (Dataset size vs size_major): NO REVERT APPLIES."
echo "### That test pins a WireCell::PointCloud::Dataset API invariant"
echo "### (size() counts arrays, size_major() counts points), not a call"
echo "### site.  Reverting MyFCN.cxx/TrackFitting.cxx cannot change it."
echo "############################################################"
