#!/bin/bash
# doc pdvd/25 sec 13.8: shared-component gate for the clus/ changes of the
# execution round (DynamicPointCloud resolve_wpid_params, TrackFitting
# dqdx_path_point_role, TaggerCheckSTM readout_edge_guard [default OFF]).
# OLD = the same tree with those five tracked files stashed (the peer session's
# unrelated WIP and the new PDVD-only components stay in both arms);
# NEW = the working tree.  Each arm runs under its own library snapshot so a
# concurrent rebuild by anyone cannot swap the binary mid-arm (memory: pin the
# binary).  Arms:
#   SBND : run_pr_chain_batch.sh over work-nuecc48-grp0825 and work-ncpi0-grp0825
#          (production defaults, PR_EXTRA_STAGES=pr_display) -> work-<root>-doc25{old,new}
#   uBooNE: qlport/scripts/sweep_5384.sh doc25{old,new} (35 events) + ab_check.sh
# Compare: SBND per event mabc-pr.zip member hashes, calib-pr JSON (minus the
# vertex_scoreboard.dual_chain timer), nusel-evt TSV; uBooNE ab_check.sh.
# Usage: ./shared_gate.sh [build|arms|compare|all]   (default all); logs in /home/xqian/tmp/doc25gate/
set -u
TK=/home/xqian/toolkit-dev/toolkit
SX=/home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
QL=/home/xqian/toolkit-dev/wcp-porting-img/qlport
G=/home/xqian/tmp/doc25gate; mkdir -p "$G"
FILES="clus/src/DynamicPointCloud.cxx clus/inc/WireCellClus/DynamicPointCloud.h clus/src/TrackFitting.cxx clus/inc/WireCellClus/TrackFitting.h clus/src/TaggerCheckSTM.cxx"
STEP=${1:-all}

snap() {   # <dir>
    mkdir -p "$1"; rm -f "$1"/*.so
    for d in "$TK"/build/*/; do cp -p "$d"/libWireCell*.so "$1"/ 2>/dev/null; done
    for f in /home/xqian/toolkit-dev/local/lib/libWireCell*.so; do b=$(basename "$f"); [ -e "$1/$b" ] || cp -p "$f" "$1"/; done
    ls "$1" | wc -l
}
do_build() {
    cd "$TK" || exit 1
    # new-arm snapshot = the tree as it is now (must already be built + installed)
    echo "new snapshot: $(snap $G/libnew) libs; clus $(md5sum $TK/build/clus/libWireCellClus.so | cut -c1-12)"
    git stash push -q -m doc25gate -- $FILES || { echo "stash failed"; exit 1; }
    ./wcb build --notests -p -k > $G/build_old.log 2>&1; ./wcb install --notests -p -k >> $G/build_old.log 2>&1
    echo "old build: $(grep -c ' error' $G/build_old.log) error lines; clus $(md5sum $TK/build/clus/libWireCellClus.so | cut -c1-12)"
    echo "old snapshot: $(snap $G/libold) libs"
    git stash pop -q || { echo "STASH POP FAILED -- restore by hand: git stash list"; exit 1; }
    ./wcb build --notests -p -k > $G/build_new.log 2>&1; ./wcb install --notests -p -k >> $G/build_new.log 2>&1
    wcbuild >> $G/build_new.log 2>&1; echo "restore build rc=$?; clus $(md5sum $TK/build/clus/libWireCellClus.so | cut -c1-12) vs snapshot $(md5sum $G/libnew/libWireCellClus.so | cut -c1-12)"
    git status --short -- $FILES
}
sbnd_arm() {   # <arm> <libdir>
    local arm=$1 lib=$2
    cd "$SX" || exit 1
    for s in nuecc48 ncpi0; do
        local out="work-$s-doc25$arm"
        [ -e "$out" ] && { echo "SKIP $out exists"; continue; }
        printf "dl_weights=''\n" > $G/tla_nodl.txt   # M4: the DL/SCN vertex is not bit-stable; gates run DL-off
        LD_LIBRARY_PATH=$lib:${LD_LIBRARY_PATH:-} PR_JOBS=${PR_JOBS:-6} PR_EXTRA_STAGES=pr_display PR_EXTRA_TLA=$G/tla_nodl.txt \
            ./run_pr_chain_batch.sh "work-$s-grp0825" "$out" data > "$G/sbnd_${s}_$arm.log" 2>&1
        echo "sbnd $s $arm rc=$? pr_evt=$(find "$out" -maxdepth 1 -type d -name 'pr_evt*' | wc -l) calib=$(ls "$out"/pr_evt*/calib-pr-evt*.json 2>/dev/null | wc -l)"
    done
}
ub_arm() {   # <arm> <libdir>
    cd "$QL/scripts" || exit 1
    [ -e "sweep/doc25$1" ] && { echo "SKIP sweep/doc25$1 exists"; return; }
    LD_LIBRARY_PATH=$2:${LD_LIBRARY_PATH:-} ./sweep_5384.sh "doc25$1" 4 > "$G/ub_$1.log" 2>&1
    echo "ub $1 rc=$? events=$(ls -d sweep/doc25$1/*_* 2>/dev/null | wc -l)"
}
do_arms() {
    sbnd_arm old $G/libold; ub_arm old $G/libold
    sbnd_arm new $G/libnew; ub_arm new $G/libnew
}
do_compare() {
    cd "$SX" || exit 1
    python3 - "$G" <<'PY'
import sys, os, glob, json, hashlib, subprocess
G = sys.argv[1]; H = "/home/xqian/toolkit-dev/wcp-porting-img/abtest/hash_archive.py"
def h_zip(p): return subprocess.check_output(["python3", H, p]).split()[0].decode()
def h_json(p):
    d = json.load(open(p)); d.get("vertex_scoreboard", {}).pop("dual_chain", None)
    if isinstance(d.get("candidates"), list):
        for c in d["candidates"]: c.get("vertex_scoreboard", {}).pop("dual_chain", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
tot = {"same": 0, "diff": 0, "missing": 0}; lines = []
for s in ("nuecc48", "ncpi0"):
    A = f"work-{s}-doc25old"; B = f"work-{s}-doc25new"
    evs = sorted(os.path.basename(p) for p in glob.glob(f"{A}/pr_evt*"))
    for ev in evs:
        for kind, pat, fn in (("bee", "mabc-pr.zip", h_zip), ("calib", "calib-pr-evt*.json", h_json), ("nusel", "nusel-evt*.tsv", lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest())):
            a = glob.glob(f"{A}/{ev}/{pat}"); b = glob.glob(f"{B}/{ev}/{pat}")
            if not a or not b: tot["missing"] += 1; lines.append(f"MISSING {s} {ev} {kind}"); continue
            ha, hb = fn(a[0]), fn(b[0]); k = "same" if ha == hb else "diff"; tot[k] += 1
            lines.append(f"{k.upper()} {s} {ev} {kind} {ha[:16]} {hb[:16]}")
with open(f"{G}/sbnd_compare.txt", "w") as fh: fh.write("\n".join(lines) + "\n")
print("SBND gate:", tot, "->", f"{G}/sbnd_compare.txt")
PY
    cd "$QL/scripts" && ./ab_check.sh doc25new doc25old > "$G/ub_ab_check.txt" 2>&1; echo "uBooNE ab_check rc=$? -> $G/ub_ab_check.txt"; tail -3 "$G/ub_ab_check.txt"
}
case "$STEP" in
    build) do_build ;; arms) do_arms ;; compare) do_compare ;;
    all) do_build; do_arms; do_compare ;;
esac
echo "shared_gate step '$STEP' done $(date -Is)"
