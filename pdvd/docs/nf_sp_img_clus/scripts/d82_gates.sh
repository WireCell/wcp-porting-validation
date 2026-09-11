#!/usr/bin/env bash
# doc pdvd/82 -- doc 78 action item 2, the peak-then-drop stop.
#   Every arm is BARE PRODUCTION plus this round's keys -- no d53 survey bag.  Wave 1
#   (p82v*) carried the survey and is void as a gate: the baselines on disk are bare, so
#   the comparison read the survey's own effect as this round's (see d82_arms.sh).
#   OFF: p82boff vs p80boff (PDVD), p82bhoff vs p80bhoff (PDHD) -- every Bee zip member,
#        calib json, every tracking-pr.root tree, every T_stm_michel branch and point row.
#        NOTE p80b* ran on the pre-doc-81 binary, so this gate also re-proves doc 81's OFF path.
#   ON : p82vcs / p82vtp / p82vtp0 / p82vk10 vs p79vprod (production after the doc 79 cap flip)
#        -- every stop that moved, by item name, against the scanner's pin_rr; the is_stm and
#        michel_found census; the through-going control.
# Usage: bash d82_gates.sh | tee /home/xqian/tmp/p82/gates.log
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p82
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json
PROD=${PROD:-p79vprod}          # the production baseline arm (post doc-79 flip)
VOFF=${VOFF:-p82boff}; HOFF=${HOFF:-p82bhoff}
ARMS=${ARMS:-"p82bcs p82btp p82btp0 p82bk10"}
cd $IMG

echo "=== 0. completeness, pins, markers"
for spec in "pdvd $PROD /home/xqian/tmp/p79" "pdvd $VOFF $W" "pdhd p80bhoff /home/xqian/tmp/p80" "pdhd $HOFF $W"; do
    set -- $spec; det=$1 a=$2 wd=$3
    n=$(ls -d $det/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in $det/work/*_$a; do [ -s $e/tracking-pr.root ] && echo 1; done | wc -l)
    lt=$(grep -l "file too short" $det/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    echo "  $det $a dirs $n with tracking-pr.root $ok loader-deaths $lt"
done
for a in $ARMS; do
    n=$(ls -d pdvd/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in pdvd/work/*_$a; do [ -s $e/tracking-pr.root ] && echo 1; done | wc -l)
    echo "  pdvd $a dirs $n with tracking-pr.root $ok | $(grep -h 'md5' $W/arm_$a.log 2>/dev/null | tr '\n' ' ')"
done
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

echo "=== 1. OFF gate, PDVD: $VOFF vs p80boff"
zipcmp pdvd $VOFF p80boff; treecmp pdvd $VOFF p80boff
python3 $X/d51g_branch_census.py --before "pdvd/work/*_p80boff" --after "pdvd/work/*_$VOFF" --before-arm p80boff --after-arm $VOFF --pts --out $W/g_$VOFF.txt > /dev/null 2>&1
echo "  branch census rc=$?"; grep -E "$G" $W/g_$VOFF.txt

echo "=== 2. OFF gate, PDHD: $HOFF vs p80bhoff"
zipcmp pdhd $HOFF p80bhoff; treecmp pdhd $HOFF p80bhoff
python3 $X/d51g_branch_census.py --before "pdhd/work/*_p80bhoff" --after "pdhd/work/*_$HOFF" --before-arm p80bhoff --after-arm $HOFF --pts --out $W/g_$HOFF.txt > /dev/null 2>&1
echo "  branch census rc=$?"; grep -E "$G" $W/g_$HOFF.txt

echo "=== 3. every stop that moved, by item name, against the scanner's pin (T_stm_michel, read from the arms)"
PROD=$PROD ARMS="$ARMS" REC=$REC python3 - <<'EOF'
import glob, os, uproot, numpy as np, json, collections
PROD = os.environ["PROD"]; ARMS = os.environ["ARMS"].split(); REC = os.environ["REC"]
R = {v["key"]: v for v in json.load(open(REC))}
def read(arm):
    out = {}
    for d in sorted(glob.glob("pdvd/work/*_" + arm)):
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn): continue
        ev = os.path.basename(d)[: -len(arm) - 1]
        try: t = uproot.open(fn)["T_stm_michel"].arrays(library="np")
        except Exception: continue
        for i in range(len(t["cluster_id"])):
            out["%s/%d" % (ev, t["cluster_id"][i])] = {k: t[k][i] for k in t}
    return out
P = read(PROD)
print("  production arm %s: %d candidates" % (PROD, len(P)))
for arm in ARMS:
    A = read(arm)
    movers = []
    for k, a in sorted(A.items()):
        p = P.get(k)
        if p is None: continue
        da = float(a["retreat_len"]) + float(a["split_len"])
        dp = float(p["retreat_len"]) + float(p["split_len"])
        if abs(da - dp) < 1e-6 and int(a["n_retreat"]) == int(p["n_retreat"]) and int(a["n_split"]) == int(p["n_split"]):
            continue
        movers.append((k, dp, da, a, p))
    bit2 = sum(1 for k, a in A.items() if int(a.get("stop_move_p3_bits", 0)) & 4)
    print("  --- %s vs %s: %d candidates, %d with a DIFFERENT stop move, %d carrying the doc-82 bit (bit2)" % (
        arm, PROD, len(A), len(movers), bit2))
    for k, dp, da, a, p in movers:
        r = R.get(k, {})
        pin = r.get("pin_rr")
        note = ""
        if pin is not None: note = "pin_rr %5.1f  |d-pin| %5.1f" % (pin, abs(da - pin))
        print("    %-14s %-16s moved %5.2f -> %5.2f cm  bits %d  %s | is_stm %d->%d michel_found %d->%d" % (
            k, r.get("verdict", "unjudged"), dp, da, int(a.get("stop_move_p3_bits", 0)),
            note, int(p["is_stm"]), int(a["is_stm"]), int(p["michel_found"]), int(a["michel_found"])))
    # verdict flips, whatever caused them
    fl_s = [(k, int(P[k]["is_stm"]), int(a["is_stm"])) for k, a in sorted(A.items()) if k in P and int(P[k]["is_stm"]) != int(a["is_stm"])]
    fl_m = [(k, int(P[k]["michel_found"]), int(a["michel_found"])) for k, a in sorted(A.items()) if k in P and int(P[k]["michel_found"]) != int(a["michel_found"])]
    def tag(k): return R.get(k, {}).get("verdict", "unjudged")
    print("    is_stm flips %d: %s" % (len(fl_s), " ".join("%s(%s %d->%d)" % (k, tag(k), b, c) for k, b, c in fl_s)))
    print("    michel_found flips %d: %s" % (len(fl_m), " ".join("%s(%s %d->%d)" % (k, tag(k), b, c) for k, b, c in fl_m)))
EOF

echo "=== 4. prep + census score for every arm, against the record"
cd $IMG/pdhd/stm_michel_scan
for a in $PROD $ARMS; do
    [ -d $W/prep_$a ] || ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
        --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$a.log 2>&1
    echo "  --- $a: $(ls $W/prep_$a/smprep-*.json 2>/dev/null | wc -l) payloads"
    python3 census_score.py --prep $W/prep_$a --arm $a --json $W/score_$a.json 2>&1 | grep -E "is_stm|michel|check|F1|purity" | head -12
done
