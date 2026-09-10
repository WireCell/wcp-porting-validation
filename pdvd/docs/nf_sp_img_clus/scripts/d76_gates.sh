#!/usr/bin/env bash
# doc pdvd/76 (P5) -- what the two smoke arms show, against the production
# baseline p75vprod (doc 75's confirmation arm, the same P75 pin).
#   p76vsame : must be identical to p75vprod on every output (the plumbing proof)
#   p76vdx4  : the STM chain moves; which Bee-zip members / branches moved, by name;
#              census on the record as an OBSERVATION (doc 70 sec 2.3: not flippable)
# Output: tee /home/xqian/tmp/p76/gates.log
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p76
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json
BASE=p75vprod
cd $IMG

echo "=== 0. completeness, pins, markers"
for a in $BASE p76vsame p76vdx4; do
    n=$(ls -d pdvd/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in pdvd/work/*_$a; do [ -s $e/tracking-stm.root ] && [ -s $e/mabc-pr.zip ] && echo 1; done | wc -l)
    lt=$(grep -l "file too short" pdvd/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    echo "  $a dirs $n complete $ok loader-deaths $lt | $(grep -h 'md5\|tla=' $W/$a.out /home/xqian/tmp/p75/$a.out 2>/dev/null | tr '\n' ' ' | cut -c1-250)"
done
echo "  markers: $(grep -c '^DONE' $W/arms.log) DONE, ALL_DONE $(grep -c '^ALL_DONE' $W/arms.log)"

echo "=== 1. Bee zip, member by member (sha256 of each member's content), and calib json"
for arm in p76vsame p76vdx4; do
ARM=$arm BASE=$BASE python3 - <<'EOF'
import glob, hashlib, os, zipfile, collections
arm, base = os.environ["ARM"], os.environ["BASE"]
same = diff = miss = 0; layers = collections.Counter(); layers_same = collections.Counter(); ev_diff = []
def members(fn):
    out = {}
    with zipfile.ZipFile(fn) as z:
        for n in z.namelist():
            if n.endswith("/"): continue
            out[n] = hashlib.sha256(z.read(n)).hexdigest()
    return out
def layer(n):
    b = os.path.basename(n)
    import re
    return re.sub(r"^\d+-", "", b).replace(".json", "")
for eb in sorted(glob.glob("pdvd/work/*_" + base)):
    pre = eb[: -len(base) - 1]; ea = pre + "_" + arm
    if not (os.path.exists(eb + "/mabc-pr.zip") and os.path.exists(ea + "/mabc-pr.zip")):
        miss += 1; continue
    A, B = members(eb + "/mabc-pr.zip"), members(ea + "/mabc-pr.zip")
    d = [n for n in set(A) | set(B) if A.get(n) != B.get(n)]
    if d:
        diff += 1; ev_diff.append(os.path.basename(pre))
        for n in d: layers[layer(n)] += 1
    else:
        same += 1
    for n in set(A) & set(B):
        if A[n] == B[n]: layers_same[layer(n)] += 1
cs = cd = 0
for eb in sorted(glob.glob("pdvd/work/*_" + base)):
    pre = eb[: -len(base) - 1]; ea = pre + "_" + arm
    ja = glob.glob(eb + "/calib-pr-evt*.json"); jb = glob.glob(ea + "/calib-pr-evt*.json")
    if ja and jb:
        if open(ja[0], "rb").read() == open(jb[0], "rb").read(): cs += 1
        else: cd += 1
print("  %s vs %s: events with an identical zip %d, differing %d, missing %d | calib json same %d diff %d" % (arm, base, same, diff, miss, cs, cd))
if diff:
    print("    members that differ, by layer (events): %s" % dict(sorted(layers.items(), key=lambda kv: -kv[1])))
    print("    layers never differing on any event: %s" % sorted(set(layers_same) - set(layers)))
    print("    events: %s%s" % (" ".join(ev_diff[:12]), " ..." if len(ev_diff) > 12 else ""))
EOF
done

echo "=== 2. every T_stm_michel branch and point row (d51g_branch_census --pts)"
G="matched|BIT-IDENTICAL|NO shared|moved|FLIPS|IDENTICAL point|role labels|geometry changed|NEW|DROPPED"
for arm in p76vsame p76vdx4; do
    python3 $X/d51g_branch_census.py --before "pdvd/work/*_$BASE" --after "pdvd/work/*_$arm" \
        --before-arm $BASE --after-arm $arm --pts --out $W/g_$arm.txt > /dev/null 2>&1
    echo "  $BASE -> $arm rc=$?"; grep -E "$G" $W/g_$arm.txt
done

echo "=== 3. tracking-pr.root: every tree, is it identical? (the neutrino-path and PR trees are the non-STM outputs)"
python3 - <<'EOF'
import glob, os, hashlib, uproot, numpy as np, collections
BASE="p75vprod"
for arm in ("p76vsame", "p76vdx4"):
    same = collections.Counter(); diff = collections.Counter(); nev = 0
    for eb in sorted(glob.glob("pdvd/work/*_" + BASE)):
        pre = eb[: -len(BASE) - 1]; ea = pre + "_" + arm
        fa, fb = eb + "/tracking-pr.root", ea + "/tracking-pr.root"
        if not (os.path.exists(fa) and os.path.exists(fb)): continue
        nev += 1
        A, B = uproot.open(fa), uproot.open(fb)
        for t in sorted(set(k.split(";")[0] for k in A.keys()) | set(k.split(";")[0] for k in B.keys())):
            # jagged (STL vector) branches must be compared as lists: numpy's array_equal on an
            # object array of vectors compares identities and reports a false DIFF (memory trap)
            try:
                import awkward as ak
                ta, tb = A[t].arrays(library="ak"), B[t].arrays(library="ak")
                def L(x):   # NaN == NaN must hold (reduced_chi2 carries NaN): map NaN to None first
                    try: return ak.to_list(ak.nan_to_none(x))
                    except Exception: return ak.to_list(x)
                ok = set(ta.fields) == set(tb.fields) and all(L(ta[n]) == L(tb[n]) for n in ta.fields)
            except Exception as e:
                ok = False
            (same if ok else diff)[t] += 1
    print("  %s vs %s over %d events: trees identical on every event: %s" % (arm, BASE, nev, sorted(t for t in same if t not in diff)))
    print("    trees differing (events): %s" % dict(sorted(diff.items())))
EOF

echo "=== 4. prep + census on the record (p76vdx4 as an observation; p76vsame must equal the baseline census)"
cd $IMG/pdhd/stm_michel_scan
for a in p76vsame p76vdx4; do
    ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
        --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$a.log 2>&1
    echo "  prep $a rc=$? payloads $(ls $W/prep_$a/smprep-*.json 2>/dev/null | wc -l)"
    STM_SCAN_RECORD=$REC python3 census_score.py --prep $W/prep_$a --baseline /home/xqian/tmp/p75/prep_$BASE --arm $a > $W/score_$a.txt 2>&1
    echo "  census $a rc=$?"; sed -n '/=== 14.1 is_stm/,/^$/p;/=== 14.2 michel_found/,/^$/p' $W/score_$a.txt | grep census
done
echo "  baseline $BASE:"; sed -n '/=== 14.1 is_stm/,/^$/p;/=== 14.2 michel_found/,/^$/p' /home/xqian/tmp/p75/score_$BASE.txt | grep census
python3 census_score.py --check > $W/check.txt 2>&1; echo "  --check rc=$?: $(tail -1 $W/check.txt)"
echo GATES_DONE
