#!/bin/bash
# Guard 4 revert-proof, SURGICAL.
#
# `git checkout origin/master -- clus/src/PRShower.cxx clus/inc/.../PRShower.h`
# does not build: PRShower.h's 2-line branch change (get_stem_dQ_dx /
# update_particle_type signatures) is consumed by NeutrinoTaggerNuE.cxx and
# NeutrinoTaggerSinglePhoton.cxx, so a whole-file revert breaks 3 unrelated
# translation units.  Instead restore ONLY the pre-fix aliasing behavior in
# Shower::add_segment's "fit" branch -- verbatim the master lines:
#     this->dpcloud(cloud_name_fit, seg_dpc_fit);
#     } else {
set -u
T=/nfs/data/1/xqian/toolkit-dev/toolkit
SP="$(cd "$(dirname "$0")" && pwd)"
F=clus/src/PRShower.cxx
TC="pr11 shower add_segment is idempotent and does not alias"
cd "$T" || exit 1

restore() { echo "=== RESTORE $F ==="; git checkout HEAD -- "$F"; git status --porcelain -- clus/src/; }
trap restore EXIT

run_case() {
    ./wcb build --notests -p > "$SP/rp4_build.log" 2>&1
    if [ $? -ne 0 ]; then echo "BUILD_FAILED"; return; fi
    (cd build && ./clus/wcdoctest-clus -tc="$TC" > "$SP/rp4_test.log" 2>&1)
    local trc=$?
    local n
    n=$(grep -oE "test cases: *[0-9]+" "$SP/rp4_test.log" | grep -oE "[0-9]+" | head -1)
    n=${n:-0}
    if [ "$n" -eq 0 ]; then echo "NO_CASES_RAN"; return; fi
    [ $trc -eq 0 ] && echo "PASS ($n case)" || echo "FAIL ($n case)"
}

python3 - <<'PY'
import re
p='/nfs/data/1/xqian/toolkit-dev/toolkit/clus/src/PRShower.cxx'
s=open(p).read()
# Only inside add_segment (the LAST occurrence of the fit-seeding pattern).
old = ("                    this->dpcloud(cloud_name_fit, clone_dpc(*seg_dpc_fit));\n"
       "                } else if (!was_member && shower_dpc_fit != seg_dpc_fit) {\n")
new = ("                    this->dpcloud(cloud_name_fit, std::const_pointer_cast<Facade::DynamicPointCloud>(seg_dpc_fit));\n"
       "                } else {\n")
n = s.count(old)
assert n >= 1, "pattern not found"
# replace the LAST one (add_segment); set_start_segment is the earlier copy
i = s.rfind(old)
s = s[:i] + new + s[i+len(old):]
open(p,'w').write(s)
print("patched occurrence %d of %d (add_segment)" % (n, n))
PY

echo "############################################################"
echo "### GUARD: 4. Shower add_segment clone_dpc + membership gate"
echo "###   surgical revert of the fit-seeding branch in add_segment"
echo -n "  WITHOUT guard: "
res=$(run_case); echo "$res"
case "$res" in
    BUILD_FAILED) grep -E "error:" "$SP/rp4_build.log" | head -3 | sed 's/^/    /' ;;
    FAIL*)        grep -E "ERROR:|THREW|is NOT correct|values:" "$SP/rp4_test.log" | head -6 | sed 's/^/    /' ;;
esac
git checkout HEAD -- "$F"
echo -n "  WITH guard:    "
run_case
