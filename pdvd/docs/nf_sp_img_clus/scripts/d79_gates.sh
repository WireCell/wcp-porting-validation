#!/usr/bin/env bash
# doc pdvd/79 -- the max_candidates cap: what the arm shows against the production
# baseline p75vprod (doc 75's confirmation arm, the same P75 pin).
#   ARM (default p79vcap): the 578 baseline candidates must be bit-identical on every
#   T_stm_michel branch and point row; the NEW candidates are named with their verdict;
#   the two owner-confirmed STM_MICHEL items the cap dropped (039253_8/65, 039349_81/62)
#   are graded; every other output is identical except on the events where the cap fired.
# Usage: ARM=p79vcap bash d79_gates.sh | tee /home/xqian/tmp/p79/gates_p79vcap.log
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p79
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json
BASE=p75vprod
ARM=${ARM:-p79vcap}
cd $IMG

echo "=== 0. completeness, pins, markers, the cap's log line"
for a in $BASE $ARM; do
    n=$(ls -d pdvd/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in pdvd/work/*_$a; do [ -s $e/tracking-stm.root ] && [ -s $e/mabc-pr.zip ] && echo 1; done | wc -l)
    lt=$(grep -l "file too short" pdvd/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    cap=$(grep -h -o "CheckSTM_Michel: [0-9]* candidates, keeping the first [0-9]*" pdvd/work/*_$a/wct_pr_*.log 2>/dev/null | sort | uniq -c | tr '\n' ';')
    echo "  $a dirs $n complete $ok loader-deaths $lt | cap lines: ${cap:-none} | $(grep -h 'md5\|tla=' $W/$a.out /home/xqian/tmp/p75/$a.out 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
done
echo "  markers: $(grep -c '^DONE' $W/arms_wave1.log) DONE, ALL_DONE $(grep -c '^ALL_DONE' $W/arms_wave1.log)"

echo "=== 1. Bee zip, member by member (sha256 of each member's content), and calib json"
ARM=$ARM BASE=$BASE python3 - <<'EOF'
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
for eb in sorted(glob.glob("pdvd/work/*_" + base)):
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
for eb in sorted(glob.glob("pdvd/work/*_" + base)):
    pre = eb[: -len(base) - 1]; ea = pre + "_" + arm
    ja = glob.glob(eb + "/calib-pr-evt*.json"); jb = glob.glob(ea + "/calib-pr-evt*.json")
    if ja and jb:
        if open(ja[0], "rb").read() == open(jb[0], "rb").read(): cs += 1
        else: cd += 1
print("  %s vs %s: events with an identical zip %d, differing %d, missing %d | calib json same %d diff %d" % (arm, base, same, diff, miss, cs, cd))
if diff:
    print("    members that differ, by layer (events): %s" % dict(sorted(layers.items(), key=lambda kv: -kv[1])))
    print("    events: %s" % " ".join(ev_diff))
EOF

echo "=== 2. every T_stm_michel branch and point row of the SHARED candidates (d51g_branch_census --pts); NEW / DROPPED named"
G="matched|BIT-IDENTICAL|NO shared|moved|FLIPS|IDENTICAL point|role labels|geometry changed|NEW|DROPPED"
python3 $X/d51g_branch_census.py --before "pdvd/work/*_$BASE" --after "pdvd/work/*_$ARM" \
    --before-arm $BASE --after-arm $ARM --pts --out $W/g_$ARM.txt > /dev/null 2>&1
echo "  $BASE -> $ARM rc=$?"; grep -E "$G" $W/g_$ARM.txt

echo "=== 3. tracking-pr.root: every tree, identical on which events? (NaN-aware, list compare)"
ARM=$ARM BASE=$BASE python3 - <<'EOF'
import glob, os, uproot, collections, awkward as ak
arm, BASE = os.environ["ARM"], os.environ["BASE"]
same = collections.Counter(); diff = collections.Counter(); diff_ev = collections.defaultdict(list); nev = 0
for eb in sorted(glob.glob("pdvd/work/*_" + BASE)):
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

echo "=== 4. the new candidates by name, with their verdict and the record's word; the two named items"
ARM=$ARM BASE=$BASE REC=$REC python3 - <<'EOF'
import glob, os, json, uproot, numpy as np
arm, base, rec = os.environ["ARM"], os.environ["BASE"], os.environ["REC"]
R = {v["key"]: v for v in json.load(open(rec))}
def cands(a):
    out = {}
    for d in sorted(glob.glob("pdvd/work/*_" + a)):
        ev = os.path.basename(d)[: -len(a) - 1]
        fn = d + "/tracking-pr.root"          # T_stm_michel lives in the PR file (d76_gates.sh sec 3)
        if not os.path.exists(fn): continue
        f = uproot.open(fn)
        if "T_stm_michel" not in [k.split(";")[0] for k in f.keys()]: continue
        t = f["T_stm_michel"].arrays(["cluster_id", "is_stm", "reject_bits", "michel_found", "michel_ke_best", "michel_len", "muon_len"], library="np")
        for i in range(len(t["cluster_id"])):
            out["%s/%d" % (ev, t["cluster_id"][i])] = {k: (float(v[i]) if v.dtype.kind == "f" else int(v[i])) for k, v in t.items()}
    return out
A, B = cands(base), cands(arm)
new = sorted(set(B) - set(A)); gone = sorted(set(A) - set(B))
print("  baseline candidates %d, arm %d, new %d, dropped %d" % (len(A), len(B), len(new), len(gone)))
for k in new:
    v = B[k]; r = R.get(k)
    print("    NEW %-14s is_stm %d bits %4d michel_found %d ke %6.1f len %5.1f muon_len %6.1f | record: %s" % (
        k, v["is_stm"], v["reject_bits"], v["michel_found"], v["michel_ke_best"], v["michel_len"], v["muon_len"],
        ("%s / %s / %s" % (r["verdict"], r["michel_kind"], r["confidence"])) if r else "unjudged (never a candidate)"))
for k in gone: print("    DROPPED %s" % k)
for k in ("039253_8/65", "039349_81/62"):
    print("  named %s: baseline %s -> arm %s" % (k, "present" if k in A else "absent", ("is_stm %d bits %d michel_found %d" % (B[k]["is_stm"], B[k]["reject_bits"], B[k]["michel_found"])) if k in B else "ABSENT"))
EOF

echo "=== 5. prep + census on the record (baseline census for reference)"
cd $IMG/pdhd/stm_michel_scan
./prep_stm_michel_scan.py --det pdvd --arm $ARM --outdir $W/prep_$ARM --sheetdir $W/sheet_$ARM \
    --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$ARM.log 2>&1
echo "  prep $ARM rc=$? payloads $(ls $W/prep_$ARM/smprep-*.json 2>/dev/null | wc -l)"
STM_SCAN_RECORD=$REC python3 census_score.py --prep $W/prep_$ARM --baseline /home/xqian/tmp/p75/prep_$BASE --arm $ARM > $W/score_$ARM.txt 2>&1
echo "  census $ARM rc=$?"; sed -n '/=== 14.1 is_stm/,/^$/p;/=== 14.2 michel_found/,/^$/p' $W/score_$ARM.txt | grep census
echo "  baseline $BASE:"; sed -n '/=== 14.1 is_stm/,/^$/p;/=== 14.2 michel_found/,/^$/p' /home/xqian/tmp/p75/score_$BASE.txt | grep census
echo "  movers by name:"; grep -E "^\s+(GAIN|LOSS|NEW|FLIP|\+|-) |gained|lost" $W/score_$ARM.txt | head -20
python3 census_score.py --check > $W/check.txt 2>&1; echo "  --check rc=$?: $(tail -1 $W/check.txt)"
echo GATES_DONE
