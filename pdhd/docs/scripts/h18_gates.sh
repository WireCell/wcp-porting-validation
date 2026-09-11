#!/usr/bin/env bash
# doc pdhd/18 -- the additivity gate for the two hand-scan arms (h18_arms.sh).
#
# Fork by duplication (CLAUDE.md M10) of pdvd/docs/nf_sp_img_clus/scripts/d79_gates.sh
# (doc pdvd/79 ran the same test when PDVD's cap went 8 -> 64).  Differences: PDHD
# paths, two (arm, baseline) pairs, no scan record to grade the new candidates
# against (this round MAKES that record), and no prep/census step.
#
#   h18s vs p82hoff   (both carry the scan SURVEY bag; h18s adds max_candidates 64)
#   h18b vs p82bhoff  (both bare production;       h18b adds max_candidates 64)
#
# PASS = on every SHARED candidate, every T_stm_michel branch and every point row is
# bit-identical (d51g_branch_census --pts), no candidate is DROPPED, and the Bee zip
# members / calib json / tracking-pr.root trees differ only on events where the cap
# fired (new candidates add fits).  The NEW candidates are named with their verdict.
# Usage: bash h18_gates.sh | tee /home/xqian/tmp/h18/gates/gates.log
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/h18/gates; mkdir -p $W
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
cd $IMG

echo "=== 0. completeness, loader deaths, the cap's log line"
for a in p82hoff p82bhoff h18s h18b; do
    n=$(ls -d pdhd/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in pdhd/work/*_$a; do [ -s $e/tracking-pr.root ] && [ -s $e/mabc-pr.zip ] && echo 1; done | wc -l)
    lt=$(grep -l "file too short" pdhd/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    cap=$(grep -h -o "CheckSTM_Michel: [0-9]* candidates, keeping the first [0-9]*" pdhd/work/*_$a/wct_pr_*.log 2>/dev/null | sort | uniq -c | tr '\n' ';')
    echo "  $a dirs $n complete $ok loader-deaths $lt | cap lines: ${cap:-none}"
done
for a in h18s h18b; do echo "  $a: $(grep -h 'md5\|complete' /home/xqian/tmp/h18/arm_$a.log | tr '\n' ' ')"; done

zipcmp() {   # arm base
ARM=$1 BASE=$2 python3 - <<'EOF'
import glob, hashlib, os, zipfile, collections, re
arm, base = os.environ["ARM"], os.environ["BASE"]
same = diff = miss = 0; layers = collections.Counter(); ev_diff = []
def members(fn):
    out = {}
    with zipfile.ZipFile(fn) as z:
        for n in z.namelist():
            if n.endswith("/"): continue
            out[n] = hashlib.sha256(z.read(n)).hexdigest()
    return out
def layer(n): return re.sub(r"^\d+-", "", os.path.basename(n)).replace(".json", "")
for eb in sorted(glob.glob("pdhd/work/*_" + base)):
    pre = eb[: -len(base) - 1]; ea = pre + "_" + arm
    if not (os.path.exists(eb + "/mabc-pr.zip") and os.path.exists(ea + "/mabc-pr.zip")):
        miss += 1; continue
    A, B = members(eb + "/mabc-pr.zip"), members(ea + "/mabc-pr.zip")
    d = [n for n in set(A) | set(B) if A.get(n) != B.get(n)]
    if d:
        diff += 1; ev_diff.append(os.path.basename(pre))
        for n in d: layers[layer(n)] += 1
    else: same += 1
cs = cd = 0; cd_ev = []
for eb in sorted(glob.glob("pdhd/work/*_" + base)):
    pre = eb[: -len(base) - 1]; ea = pre + "_" + arm
    ja = glob.glob(eb + "/calib-pr-evt*.json"); jb = glob.glob(ea + "/calib-pr-evt*.json")
    if ja and jb:
        if open(ja[0], "rb").read() == open(jb[0], "rb").read(): cs += 1
        else: cd += 1; cd_ev.append(os.path.basename(pre))
print("  %s vs %s: events with an identical zip %d, differing %d, missing %d | calib json same %d diff %d"
      % (arm, base, same, diff, miss, cs, cd))
if diff:
    print("    zip members that differ, by layer (events): %s" % dict(sorted(layers.items(), key=lambda kv: -kv[1])))
    print("    zip events: %s" % " ".join(ev_diff))
if cd: print("    calib events: %s" % " ".join(cd_ev))
EOF
}
treecmp() {   # arm base
ARM=$1 BASE=$2 python3 - <<'EOF'
import glob, os, uproot, collections, awkward as ak
arm, BASE = os.environ["ARM"], os.environ["BASE"]
same = collections.Counter(); diff = collections.Counter(); diff_ev = collections.defaultdict(list); nev = 0
for eb in sorted(glob.glob("pdhd/work/*_" + BASE)):
    pre = eb[: -len(BASE) - 1]; ea = pre + "_" + arm
    fa, fb = eb + "/tracking-pr.root", ea + "/tracking-pr.root"
    if not (os.path.exists(fa) and os.path.exists(fb)): continue
    nev += 1
    A, B = uproot.open(fa), uproot.open(fb)
    for t in sorted(set(k.split(";")[0] for k in A.keys()) | set(k.split(";")[0] for k in B.keys())):
        try:
            ta, tb = A[t].arrays(library="ak"), B[t].arrays(library="ak")
            def L(x):
                try: return ak.to_list(ak.nan_to_none(x))
                except Exception: return ak.to_list(x)
            ok = set(ta.fields) == set(tb.fields) and all(L(ta[n]) == L(tb[n]) for n in ta.fields)
        except Exception:
            ok = False
        (same if ok else diff)[t] += 1
        if not ok: diff_ev[t].append(os.path.basename(pre))
print("  %s vs %s over %d events: trees identical on every event: %s" % (arm, BASE, nev, sorted(t for t in same if t not in diff)))
for t in sorted(diff): print("    %s differs on %d events: %s" % (t, diff[t], " ".join(diff_ev[t])))
EOF
}
newcands() {   # arm base
ARM=$1 BASE=$2 python3 - <<'EOF'
import glob, os, uproot
arm, base = os.environ["ARM"], os.environ["BASE"]
def cands(a):
    out = {}
    for d in sorted(glob.glob("pdhd/work/*_" + a)):
        ev = os.path.basename(d)[: -len(a) - 1]
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn): continue
        f = uproot.open(fn)
        if "T_stm_michel" not in [k.split(";")[0] for k in f.keys()]: continue
        t = f["T_stm_michel"].arrays(["cluster_id", "is_stm", "reject_bits", "michel_found",
                                      "michel_ke_best", "michel_len", "muon_len"], library="np")
        for i in range(len(t["cluster_id"])):
            out["%s/%d" % (ev, t["cluster_id"][i])] = {k: (float(v[i]) if v.dtype.kind == "f" else int(v[i]))
                                                       for k, v in t.items()}
    return out
A, B = cands(base), cands(arm)
new = sorted(set(B) - set(A)); gone = sorted(set(A) - set(B))
print("  %s: baseline candidates %d, arm %d, NEW %d, DROPPED %d" % (arm, len(A), len(B), len(new), len(gone)))
for k in new:
    v = B[k]
    print("    NEW %-16s is_stm %d bits %5d michel_found %d ke %6.1f len %5.1f muon_len %6.1f"
          % (k, v["is_stm"], v["reject_bits"], v["michel_found"], v["michel_ke_best"], v["michel_len"], v["muon_len"]))
for k in gone: print("    DROPPED %s" % k)
EOF
}
G="matched|BIT-IDENTICAL|NO shared|moved|FLIPS|IDENTICAL point|role labels|geometry changed|NEW|DROPPED"

for pair in "h18s p82hoff" "h18b p82bhoff"; do
    set -- $pair; arm=$1 base=$2
    echo "=== $arm vs $base"
    zipcmp $arm $base
    treecmp $arm $base
    python3 $X/d51g_branch_census.py --before "pdhd/work/*_$base" --after "pdhd/work/*_$arm" \
        --before-arm $base --after-arm $arm --pts --out $W/g_$arm.txt > $W/g_$arm.log 2>&1
    echo "  branch census rc=$?  ($W/g_$arm.txt)"; grep -E "$G" $W/g_$arm.txt
    newcands $arm $base
done
echo GATES_DONE
