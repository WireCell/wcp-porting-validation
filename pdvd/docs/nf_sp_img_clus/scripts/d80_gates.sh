#!/usr/bin/env bash
# doc pdvd/80 -- the segment census (segment_census, role 8): OFF gates on both
# detectors and what the ON arm writes.
#   OFF: $VOFF vs p79vprod (PDVD, production post doc-79 flip), $HOFF vs p75hoff (PDHD)
#        -> every Bee zip member, calib json, every tracking-pr.root tree, every T_stm_michel
#           branch and point row byte-identical
#   ON : $VCEN vs p79vprod -> every production T_stm_michel branch identical, two new scalars;
#        every role 1-7 point row identical, role-8 rows appended, three new columns; the doc 78
#        orphan michel tags now carry a row, with their rej code; census on the record identical
# Usage: bash d80_gates.sh | tee /home/xqian/tmp/p80/gates.log
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p80
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json
# arm labels: wave 1 (Clus-only pin, VOID -- see d80_arms.sh) p80voff / p80vcen / p80hoff;
# wave 3 (full pin) p80boff/p80bcen/p80bhoff, the ones the doc reports
VOFF=${VOFF:-p80boff}; VCEN=${VCEN:-p80bcen}; HOFF=${HOFF:-p80bhoff}; WAVELOG=${WAVELOG:-$W/arms_wave3.log}
cd $IMG

echo "=== 0. completeness, pins, markers"
for spec in "pdvd p79vprod /home/xqian/tmp/p79" "pdvd $VOFF $W" "pdvd $VCEN $W" "pdhd p75hoff /home/xqian/tmp/p75" "pdhd $HOFF $W"; do
    set -- $spec; det=$1 a=$2 wd=$3
    n=$(ls -d $det/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in $det/work/*_$a; do [ -s $e/tracking-stm.root ] && [ -s $e/mabc-pr.zip ] && echo 1; done | wc -l)
    lt=$(grep -l "file too short" $det/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    echo "  $det $a dirs $n complete $ok loader-deaths $lt | $(grep -h 'md5\|tla=' $wd/$a.out 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
done
echo "  markers: $(grep -c '^DONE' $WAVELOG) DONE, ALL_DONE $(grep -c '^ALL_DONE' $WAVELOG)"

zipcmp() {   # det arm base
DET=$1 ARM=$2 BASE=$3 python3 - <<'EOF'
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
EOF
}
treecmp() {   # det arm base
DET=$1 ARM=$2 BASE=$3 python3 - <<'EOF'
import glob, os, uproot, collections, awkward as ak
det, arm, BASE = os.environ["DET"], os.environ["ARM"], os.environ["BASE"]
same = collections.Counter(); diff = collections.Counter(); diff_ev = collections.defaultdict(list); nev = 0
for eb in sorted(glob.glob(det + "/work/*_" + BASE)):
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
print("  %s %s vs %s over %d events: trees identical on every event: %s" % (det, arm, BASE, nev, sorted(t for t in same if t not in diff)))
for t in sorted(diff): print("    %s differs on %d events: %s" % (t, diff[t], " ".join(diff_ev[t][:10])))
EOF
}
G="matched|BIT-IDENTICAL|NO shared|moved|FLIPS|IDENTICAL point|role labels|geometry changed|NEW|DROPPED"

echo "=== 1. OFF gate, PDVD: $VOFF vs p79vprod"
zipcmp pdvd $VOFF p79vprod; treecmp pdvd $VOFF p79vprod
python3 $X/d51g_branch_census.py --before "pdvd/work/*_p79vprod" --after "pdvd/work/*_$VOFF" --before-arm p79vprod --after-arm $VOFF --pts --out $W/g_$VOFF.txt > /dev/null 2>&1
echo "  branch census rc=$?"; grep -E "$G" $W/g_$VOFF.txt
echo "=== 2. OFF gate, PDHD: $HOFF vs p75hoff"
zipcmp pdhd $HOFF p75hoff; treecmp pdhd $HOFF p75hoff
python3 $X/d51g_branch_census.py --before "pdhd/work/*_p75hoff" --after "pdhd/work/*_$HOFF" --before-arm p75hoff --after-arm $HOFF --pts --out $W/g_$HOFF.txt > /dev/null 2>&1
echo "  branch census rc=$?"; grep -E "$G" $W/g_$HOFF.txt

echo "=== 3. ON arm $VCEN vs p79vprod: Bee zip, trees, the production branches"
zipcmp pdvd $VCEN p79vprod; treecmp pdvd $VCEN p79vprod
python3 $X/d51g_branch_census.py --before "pdvd/work/*_p79vprod" --after "pdvd/work/*_$VCEN" --before-arm p79vprod --after-arm $VCEN --out $W/g_$VCEN.txt > /dev/null 2>&1
echo "  branch census (scalars) rc=$?"; grep -E "$G|branches only in" $W/g_$VCEN.txt

echo "=== 4. ON arm: the point rows -- roles 1-7 identical, role 8 appended, the new columns"
VCEN=$VCEN python3 - <<'EOF'
import glob, os, uproot, numpy as np, collections
B, A = "p79vprod", os.environ["VCEN"]
n_id = n_diff = n_cand = 0; role8 = collections.Counter(); rej = collections.Counter(); cols_new = None; nrows8 = 0
for eb in sorted(glob.glob("pdvd/work/*_" + B)):
    pre = eb[: -len(B) - 1]; ea = pre + "_" + A
    fa, fb = eb + "/tracking-pr.root", ea + "/tracking-pr.root"
    if not (os.path.exists(fa) and os.path.exists(fb)): continue
    Fa, Fb = uproot.open(fa), uproot.open(fb)
    ka = [k.split(";")[0] for k in Fa.keys()]; kb = [k.split(";")[0] for k in Fb.keys()]
    if "T_stm_michel_pts" not in ka or "T_stm_michel_pts" not in kb: continue
    ta, tb = Fa["T_stm_michel_pts"].arrays(library="np"), Fb["T_stm_michel_pts"].arrays(library="np")
    if cols_new is None: cols_new = sorted(set(tb) - set(ta))
    for cid in sorted(set(ta["cluster_id"].tolist())):
        n_cand += 1
        sa = ta["cluster_id"] == cid; sb = (tb["cluster_id"] == cid) & (tb["role"] != 8)
        ok = all(np.array_equal(ta[c][sa], tb[c][sb], equal_nan=(ta[c].dtype.kind == "f")) for c in ta)
        n_id += ok; n_diff += (not ok)
        s8 = (tb["cluster_id"] == cid) & (tb["role"] == 8)
        nrows8 += int(s8.sum())
        if s8.sum():
            role8[cid and 1] += 1
            for r in tb["rej"][s8]: rej[int(r)] += 1
print("  candidates %d: role 1-7 rows identical (every column, same order) on %d, differing on %d" % (n_cand, n_id, n_diff))
print("  new columns in T_stm_michel_pts: %s; role-8 rows %d on %d candidates; rej codes over role-8 rows: %s" % (cols_new, nrows8, role8[1], dict(sorted(rej.items()))))
EOF

echo "=== 5. ON arm: prep, the doc 78 orphan michel tags now have a row -- and their rej"
cd $IMG/pdhd/stm_michel_scan
./prep_stm_michel_scan.py --det pdvd --arm $VCEN --outdir $W/prep_$VCEN --sheetdir $W/sheet_$VCEN \
    --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$VCEN.log 2>&1
echo "  prep $VCEN rc=$? payloads $(ls $W/prep_$VCEN/smprep-*.json 2>/dev/null | wc -l)"
STM_SCAN_RECORD=$REC python3 census_score.py --prep $W/prep_$VCEN --baseline /home/xqian/tmp/p79/prep_p79vprod --arm $VCEN > $W/score_$VCEN.txt 2>&1
echo "  census $VCEN rc=$?"; sed -n '/=== 14.1 is_stm/,/^$/p;/=== 14.2 michel_found/,/^$/p' $W/score_$VCEN.txt | grep census
echo "  baseline p79vprod:"; sed -n '/=== 14.1 is_stm/,/^$/p;/=== 14.2 michel_found/,/^$/p' /home/xqian/tmp/p79/score_p79vprod.txt | grep census
cd $IMG
W=$W REC=$REC VCEN=$VCEN python3 - <<'EOF'
import json, glob, os, collections
W, REC, VCEN = os.environ["W"], os.environ["REC"], os.environ["VCEN"]
R = {v["key"]: v for v in json.load(open(REC))}
P = {}
for fn in glob.glob(W + "/prep_" + VCEN + "/smprep-*.json"):
    d = json.load(open(fn)); P["%s/%d" % (d["event"], d["cluster_id"])] = d
STOP = {"STM_MICHEL", "STM_ONLY", "FRAG_STM_MICHEL", "FRAG_STM_ONLY"}
REJ = {0: "claimed", 2: "too far from the stop", 3: "longer than the piece cap", 4: "body exclusion", 9: "T3b off", 10: "no stop vertex", 13: "attached to the chain, not taken", 14: "passes every gate, unclaimed", 15: "no endpoint vertices"}
tot = collections.Counter(); rows = []
for k in sorted(R):
    v = R[k]
    if v["verdict"] not in STOP or k not in P: continue
    p = P[k]
    for t, g in (v.get("tags") or {}).items():
        if g != "michel" or str(t).startswith("C") or int(t) // 1000 != p["cluster_id"]: continue
        r = p["pf"]["chain_role"].get(str(t))
        if r is None: tot["main cluster, still no row"] += 1; rows.append((k, t, "NO ROW", None)); continue
        if r != 8: tot["row role %d" % r] += 1; continue
        rj = p["pf"]["seg_rej"].get(str(t), {})
        tot["role 8, rej %s" % rj.get("rej")] += 1; rows.append((k, t, "role 8", rj))
print("  record michel tags on MAIN-cluster segments (judged stoppers), by chain reading on the census arm: %s" % dict(sorted(tot.items())))
print("  the role-8 / no-row ones, by name (mf = michel_found on this arm):")
for k, t, w, rj in rows:
    vd = P[k]["verdict"]
    print("    %-14s seg %-7s mf %d is_stm %d | %s %s" % (k, t, vd["michel_found"], vd["is_stm"], w,
          ("rej %s (%s) d_stop %.1f d_body %.1f" % (rj.get("rej"), REJ.get(rj.get("rej"), "?"), rj.get("d_stop", -1), rj.get("d_body", -1))) if rj else ""))
n8 = sum(1 for p in P.values() for r in p["pf"]["chain_role"].values() if r == 8)
print("  role-8 segments over all %d payloads: %d; candidates with n_census_admissible > 0: %s" % (len(P), n8,
      sorted(k for k, p in P.items() if p["verdict"].get("n_census_admissible", 0) > 0)))
EOF
python3 $IMG/pdhd/stm_michel_scan/census_score.py --check > $W/check.txt 2>&1; echo "  --check rc=$?: $(tail -1 $W/check.txt)"
echo GATES_DONE
