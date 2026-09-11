#!/usr/bin/env bash
# doc pdvd/81 -- the charge-based Michel energy: OFF gates on both detectors,
# and what the ON arms write.  Fork of d80_gates.sh.  The *3 arms are wave 1c
# (the final pin libpin_p81g); the leg arms are wave 1c's too (libpin_p75).
#   OFF: p81voff3 vs p81vleg2 (PDVD), p81hoff3 vs p81hleg2 (PDHD), both pairs run side by side on
#        the same production config -> every Bee zip member, calib json, every tracking-pr.root
#        tree, every T_stm_michel branch and point row byte-identical; no T_stm_michel_2d
#   ON : p81vq2d3 vs p81voff3 / p81hq2d3 vs p81hoff3 -> Bee zip and calib identical, every
#        production T_stm_michel branch identical, the new scalars appear, T_stm_michel_2d
#        appears, every point row identical; census on the record identical
# Usage: bash d81_gates.sh | tee /home/xqian/tmp/p81/gates.log
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p81
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json
cd $IMG

echo "=== 0. completeness, pins, markers"
for spec in "pdvd p81vleg2" "pdvd p81voff3" "pdvd p81vq2d3" "pdhd p81hleg2" "pdhd p81hoff3" "pdhd p81hq2d3"; do
    set -- $spec; det=$1 a=$2
    n=$(ls -d $det/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in $det/work/*_$a; do [ -s $e/tracking-stm.root ] && [ -s $e/mabc-pr.zip ] && echo 1; done | wc -l)
    lt=$(grep -l "file too short" $det/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    echo "  $det $a dirs $n complete $ok loader-deaths $lt | $(grep -h 'md5\|tla=' $W/$a.out 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
done
echo "  markers: wave1c $(grep -c '^DONE' $W/arms_wave1c.log) DONE / ALL_DONE $(grep -c '^ALL_DONE' $W/arms_wave1c.log)"

zipcmp() {   # det arm base
DET=$1 ARM=$2 BASE=$3 python3 - <<'PYEOF'
import glob, hashlib, os, zipfile, collections, re
det, arm, base = os.environ["DET"], os.environ["ARM"], os.environ["BASE"]
same = diff = miss = 0; layers = collections.Counter(); ev_diff = []
def members(fn):
    out = {}
    with zipfile.ZipFile(fn) as z:
        for n in z.namelist():
            if n.endswith("/"): continue
            out[n] = hashlib.sha256(z.read(n)).hexdigest()
    return out
def layer(n): return re.sub(r"^\d+-", "", os.path.basename(n)).replace(".json", "")
for eb in sorted(glob.glob(det + "/work/*_" + base)):
    pre = eb[: -len(base) - 1]; ea = pre + "_" + arm
    if not (os.path.exists(eb + "/mabc-pr.zip") and os.path.exists(ea + "/mabc-pr.zip")):
        miss += 1; continue
    A, B = members(eb + "/mabc-pr.zip"), members(ea + "/mabc-pr.zip")
    d = [n for n in set(A) | set(B) if A.get(n) != B.get(n)]
    if d:
        diff += 1; ev_diff.append(os.path.basename(pre))
        for n in d: layers[layer(n)] += 1
    else: same += 1
cs = cd = 0
for eb in sorted(glob.glob(det + "/work/*_" + base)):
    pre = eb[: -len(base) - 1]; ea = pre + "_" + arm
    ja = glob.glob(eb + "/calib-pr-evt*.json"); jb = glob.glob(ea + "/calib-pr-evt*.json")
    if ja and jb:
        if open(ja[0], "rb").read() == open(jb[0], "rb").read(): cs += 1
        else: cd += 1
print("  %s %s vs %s: events with an identical zip %d, differing %d, missing %d | calib json same %d diff %d" % (det, arm, base, same, diff, miss, cs, cd))
if diff:
    print("    members that differ, by layer (events): %s" % dict(sorted(layers.items(), key=lambda kv: -kv[1])))
    print("    events: %s" % " ".join(ev_diff[:12]))
PYEOF
}
treecmp() {   # det arm base
DET=$1 ARM=$2 BASE=$3 python3 - <<'PYEOF'
import glob, os, uproot, collections, awkward as ak
det, arm, BASE = os.environ["DET"], os.environ["ARM"], os.environ["BASE"]
same = collections.Counter(); diff = collections.Counter(); diff_ev = collections.defaultdict(list); nev = 0
only_a = collections.Counter(); only_b = collections.Counter()
for eb in sorted(glob.glob(det + "/work/*_" + BASE)):
    pre = eb[: -len(BASE) - 1]; ea = pre + "_" + arm
    fa, fb = eb + "/tracking-pr.root", ea + "/tracking-pr.root"
    if not (os.path.exists(fa) and os.path.exists(fb)): continue
    nev += 1
    A, B = uproot.open(fa), uproot.open(fb)
    ka = set(k.split(";")[0] for k in A.keys()); kb = set(k.split(";")[0] for k in B.keys())
    for t in ka - kb: only_a[t] += 1
    for t in kb - ka: only_b[t] += 1
    for t in sorted(ka & kb):
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
print("  %s %s vs %s over %d events: trees identical on every event: %s" % (det, arm, BASE, nev, sorted(t for t in same if t not in diff)))
for t in sorted(diff): print("    %s differs on %d events: %s" % (t, diff[t], " ".join(diff_ev[t][:10])))
if only_a: print("    trees only in %s: %s" % (BASE, dict(only_a)))
if only_b: print("    trees only in %s: %s" % (arm, dict(only_b)))
PYEOF
}
# NOTE: each census takes the arm name TWICE (the work-dir glob and --*-arm); a stale
# glob silently censuses the wrong arm or exits 2.  Keep the pairs in step.
G="matched|BIT-IDENTICAL|NO shared|moved|FLIPS|IDENTICAL point|role labels|geometry changed|NEW|DROPPED"

echo "=== 1. OFF gate, PDVD: p81voff3 vs p81vleg2"
zipcmp pdvd p81voff3 p81vleg2; treecmp pdvd p81voff3 p81vleg2
python3 $X/d51g_branch_census.py --before "pdvd/work/*_p81vleg2" --after "pdvd/work/*_p81voff3" --before-arm p81vleg2 --after-arm p81voff3 --pts --out $W/g_p81voff.txt > /dev/null 2>&1
echo "  branch census rc=$?"; grep -E "$G" $W/g_p81voff.txt
echo "=== 2. OFF gate, PDHD: p81hoff3 vs p81hleg2"
zipcmp pdhd p81hoff3 p81hleg2; treecmp pdhd p81hoff3 p81hleg2
python3 $X/d51g_branch_census.py --before "pdhd/work/*_p81hleg2" --after "pdhd/work/*_p81hoff3" --before-arm p81hleg2 --after-arm p81hoff3 --pts --out $W/g_p81hoff.txt > /dev/null 2>&1
echo "  branch census rc=$?"; grep -E "$G" $W/g_p81hoff.txt
echo "=== 2b. the fresh PDVD baseline against production (p81vleg2 vs p79vprod: the config did not move under us)"
zipcmp pdvd p81vleg2 p79vprod; treecmp pdvd p81vleg2 p79vprod

echo "=== 3. ON arms vs their OFF twins: Bee zip, trees, the production branches"
zipcmp pdvd p81vq2d3 p81voff3; treecmp pdvd p81vq2d3 p81voff3
python3 $X/d51g_branch_census.py --before "pdvd/work/*_p81voff3" --after "pdvd/work/*_p81vq2d3" --before-arm p81voff3 --after-arm p81vq2d3 --pts --out $W/g_p81vq2d.txt > /dev/null 2>&1
echo "  branch census rc=$?"; grep -E "$G|branches only in" $W/g_p81vq2d.txt
zipcmp pdhd p81hq2d3 p81hoff3; treecmp pdhd p81hq2d3 p81hoff3
python3 $X/d51g_branch_census.py --before "pdhd/work/*_p81hoff3" --after "pdhd/work/*_p81hq2d3" --before-arm p81hoff3 --after-arm p81hq2d3 --pts --out $W/g_p81hq2d.txt > /dev/null 2>&1
echo "  branch census rc=$?"; grep -E "$G|branches only in" $W/g_p81hq2d.txt

echo "=== 4. ON arm: prep + census on the record must equal the OFF arm's"
cd $IMG/pdhd/stm_michel_scan
for a in p81voff3 p81vq2d3; do
    ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
        --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$a.log 2>&1
    echo "  prep $a rc=$? payloads $(ls $W/prep_$a/smprep-*.json 2>/dev/null | wc -l)"
    STM_SCAN_RECORD=$REC python3 census_score.py --prep $W/prep_$a --baseline /home/xqian/tmp/p79/prep_p79vprod --arm $a > $W/score_$a.txt 2>&1
    echo "  census $a rc=$?"; sed -n '/=== 14.1 is_stm/,/^$/p;/=== 14.2 michel_found/,/^$/p' $W/score_$a.txt | grep census
done
echo "  baseline p79vprod:"; sed -n '/=== 14.1 is_stm/,/^$/p;/=== 14.2 michel_found/,/^$/p' /home/xqian/tmp/p79/score_p79vprod.txt | grep census
python3 census_score.py --check > $W/check.txt 2>&1; echo "  --check rc=$?: $(tail -1 $W/check.txt)"
echo GATES_DONE
