#!/usr/bin/env bash
# doc pdvd/90 -- the gates for doc 89's 0-FP bundle: p90vb (topology_michel_ke_min 3 + plateau_mip_hi
# 2.0), p90vk (the energy floor alone), p90vp (the plateau edge alone), each against production p88vprod.
#   Fork by duplication (CLAUDE.md M10) of d88_gates.sh: the same zip / tree / branch comparisons and
#   movers by name against production.  No OFF gate: no C++ this round, the arms run doc 88's pin
#   (libpin_p88, the one p88vprod ran on) and the same file, and differ from p88vprod only in the TLA.
#   Section 1 checks every arm's movers against the offline twin written before the arms ran
#   (/home/xqian/tmp/p90/twin.json; pred.txt), and every differing zip against the movers' events.
#   The record is the merged smx1a+smx3+smx4+smx5+smx6 one (doc 88 sec 9.2).
# Usage: bash d90_gates.sh > /home/xqian/tmp/p90/gates.log 2>&1
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/p90
X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
REC=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_verdicts.json
export STM_SCAN_RECORD=$REC
PROD=${PROD:-p88vprod}
ARMS=${ARMS:-"p90vb p90vk p90vp"}
CFGS=${CFGS:-"p90vb:$W/cfgcheck/b.json p90vk:$W/cfgcheck/k.json p90vp:$W/cfgcheck/p.json p90vprod:$W/proofs/post.json"}
cd $IMG
echo "record: $REC"

echo "=== 0. completeness, pin, loader deaths"
for a in $PROD $ARMS; do
    n=$(ls -d pdvd/work/*_$a 2>/dev/null | wc -l)
    ok=$(for e in pdvd/work/*_$a; do [ -s $e/tracking-pr.root ] && echo 1; done | wc -l)
    lt=$(grep -l "file too short" pdvd/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    echo "  pdvd $a dirs $n with tracking-pr.root $ok loader-deaths $lt"
done
md5sum /home/xqian/tmp/p88/libpin_p88/*.so* > $W/libpin_md5_after.txt
cmp -s $W/libpin_md5_before.txt $W/libpin_md5_after.txt && echo "  pin libpin_p88: $(wc -l < $W/libpin_md5_after.txt) libraries, md5 identical before and after" \
    || echo "  PIN CHANGED between launch and gates"

echo "=== 1. every candidate on which a branch or a point row moved, by name, against the twin"
PROD=$PROD ARMS="$ARMS" REC=$REC TWIN=$W/twin.json python3 - <<'EOF'
import collections, glob, hashlib, json, os, zipfile
import numpy as np, uproot
PROD = os.environ["PROD"]; ARMS = os.environ["ARMS"].split()
R = {v["key"]: v for v in json.load(open(os.environ["REC"]))}
TW = json.load(open(os.environ["TWIN"]))
def read(arm):
    out = {}; pts = {}
    for d in sorted(glob.glob("pdvd/work/*_" + arm)):
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn): continue
        ev = os.path.basename(d)[: -len(arm) - 1]
        f = uproot.open(fn)
        if "T_stm_michel" not in f:        # an event with no candidate writes no T_stm_michel (d88's read skipped it too)
            continue
        t = f["T_stm_michel"].arrays(library="np")
        q = f["T_stm_michel_pts"].arrays(["cluster_id", "role", "seg_id", "x", "y", "z"], library="np")
        for i in range(len(t["cluster_id"])):
            k = "%s/%d" % (ev, t["cluster_id"][i])
            out[k] = {kk: t[kk][i] for kk in t}
            sel = q["cluster_id"] == t["cluster_id"][i]
            pts[k] = [(int(r), int(s), round(float(x), 4), round(float(y), 4), round(float(z), 4))
                      for r, s, x, y, z in zip(q["role"][sel], q["seg_id"][sel], q["x"][sel], q["y"][sel], q["z"][sel])]
    return out, pts
def same(a, b):
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
        if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)) and np.isnan(a) and np.isnan(b): return True
        return bool(a == b)
    except Exception:
        return False
def members(fn):
    with zipfile.ZipFile(fn) as z:
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith("/")}
tag = lambda k: R.get(k, {}).get("verdict", "unjudged")
P, PP = read(PROD)
print("  production arm %s: %d candidates" % (PROD, len(P)))
for arm in ARMS:
    A, AP = read(arm)
    tw = TW.get(arm, {})
    print("  --- %s vs %s: %d candidates" % (arm, PROD, len(A)))
    print("    branches only in %s: %s; only in %s: %s" % (arm, sorted({b for a in A.values() for b in a} - {b for p in P.values() for b in p}),
          PROD, sorted({b for p in P.values() for b in p} - {b for a in A.values() for b in a})))
    for k in sorted(set(A) - set(P)): print("    NEW candidate %s" % k)
    for k in sorted(set(P) - set(A)): print("    DROPPED candidate %s" % k)
    moved = {}; bset = collections.Counter()
    for k in sorted(set(A) & set(P)):
        bs = sorted(b for b in P[k] if not same(A[k].get(b), P[k].get(b)))
        rows = AP[k] != PP[k]
        if bs or rows:
            moved[k] = (bs, rows); bset.update(bs)
    g = lambda r, b: int(r.get(b, 0))
    for k, (bs, rows) in moved.items():
        a, p = A[k], P[k]
        print("    %-14s %-13s is_stm %d->%d mf %d->%d bits %d->%d topo_cleared %d->%d | rows moved %s | branches %s" % (
            k, tag(k), g(p, "is_stm"), g(a, "is_stm"), g(p, "michel_found"), g(a, "michel_found"), g(p, "reject_bits"), g(a, "reject_bits"),
            g(p, "topology_cleared_bits"), g(a, "topology_cleared_bits"), rows, " ".join(bs)))
    fl_s = sorted(k for k in moved if g(A[k], "is_stm") != g(P[k], "is_stm"))
    fl_m = sorted(k for k in moved if g(A[k], "michel_found") != g(P[k], "michel_found"))
    print("    candidates moved %d; branches moved (candidates): %s" % (len(moved), dict(sorted(bset.items()))))
    print("    is_stm flips %d: %s" % (len(fl_s), " ".join("%s(%s %d->%d)" % (k, tag(k), g(P[k], "is_stm"), g(A[k], "is_stm")) for k in fl_s)))
    print("    michel_found flips %d: %s" % (len(fl_m), " ".join(fl_m) or "-"))
    ts, tb = set(tw.get("is_stm_movers", [])), set(tw.get("branch_movers", []))
    poss = set(tw.get("possible_geo", [])) | set(tw.get("possible_fb", []))
    print("    TWIN is_stm movers: predicted %d, on the arm %d; missing %s; unpredicted %s" % (
        len(ts), len(fl_s), sorted(ts - set(fl_s)) or "none", sorted(set(fl_s) - ts) or "none"))
    unexp = sorted(k for k in moved if k not in ts | tb | poss)
    print("    TWIN branch-only movers predicted %s: moved %s" % (sorted(tb), sorted(k for k in tb if k in moved)))
    print("    moved, in the twin's possible list: %s" % (sorted(k for k in moved if k in poss) or "none"))
    print("    MOVED AND NOT PREDICTED: %d %s" % (len(unexp), " ".join(unexp)))
    ev_mv = {k.split("/")[0] for k in fl_s}
    zd = []
    for eb in sorted(glob.glob("pdvd/work/*_" + PROD)):
        ev = os.path.basename(eb)[: -len(PROD) - 1]; ea = "pdvd/work/%s_%s" % (ev, arm)
        if os.path.exists(eb + "/mabc-pr.zip") and os.path.exists(ea + "/mabc-pr.zip") and members(eb + "/mabc-pr.zip") != members(ea + "/mabc-pr.zip"):
            zd.append(ev)
    print("    zips differing on %d events: %s; of them WITHOUT an is_stm mover: %s" % (len(zd), " ".join(zd), [e for e in zd if e not in ev_mv] or "none"))
EOF
[ "${ONLY1:-0}" = 1 ] && { echo "ONLY1: stopping after section 1"; exit 0; }

echo "=== 2. branch census (d51g) per arm"
G="matched|BIT-IDENTICAL|NO shared|moved|FLIPS|IDENTICAL point|role labels|geometry changed|NEW|DROPPED"
for a in $ARMS; do
    python3 $X/d51g_branch_census.py --before "pdvd/work/*_$PROD" --after "pdvd/work/*_$a" --before-arm $PROD --after-arm $a --pts --out $W/g_$a.txt > /dev/null 2>&1
    echo "  branch census $a rc=$?"; grep -E "$G" $W/g_$a.txt | head -12
done

echo "=== 3. trees and calib per arm against production"
treecmp() {   # det arm base
DET=$1 ARM=$2 BASE=$3 python3 - <<'EOF'
import glob, os, uproot, collections, awkward as ak
det, arm, BASE = os.environ["DET"], os.environ["ARM"], os.environ["BASE"]
same = collections.Counter(); diff = collections.Counter(); diff_ev = collections.defaultdict(list); nev = 0
cs = cd = 0; cd_ev = []
for eb in sorted(glob.glob(det + "/work/*_" + BASE)):
    pre = eb[: -len(BASE) - 1]; ea = pre + "_" + arm
    fa, fb = eb + "/tracking-pr.root", ea + "/tracking-pr.root"
    ja = glob.glob(eb + "/calib-pr-evt*.json"); jb = glob.glob(ea + "/calib-pr-evt*.json")
    if ja and jb:
        if open(ja[0], "rb").read() == open(jb[0], "rb").read(): cs += 1
        else: cd += 1; cd_ev.append(os.path.basename(pre))
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
print("  %s %s vs %s over %d events: trees identical on every event: %s | calib json same %d diff %d %s" % (
    det, arm, BASE, nev, sorted(t for t in same if t not in diff), cs, cd, " ".join(cd_ev)))
for t in sorted(diff): print("    %s differs on %d events: %s" % (t, diff[t], " ".join(diff_ev[t][:12])))
EOF
}
for a in $ARMS; do treecmp pdvd $a $PROD; done

echo "=== 4. prep + census for every arm, against the record; the missed-stopper partition on each"
cd $IMG/pdhd/stm_michel_scan
for a in $ARMS; do
    [ -d $W/prep_$a ] || ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
        --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv > $W/prep_$a.log 2>&1
    echo "  --- $a: $(ls $W/prep_$a/smprep-*.json 2>/dev/null | wc -l) payloads"
    STM_SCAN_RECORD=$REC python3 census_score.py --prep $W/prep_$a --arm $a --json $W/score_merged_$a.json 2>&1 | grep -E "^census|as shipped"
    cfg=$(for c in $CFGS; do [ "${c%%:*}" = "$a" ] && echo "${c#*:}"; done)
    STM_SCAN_RECORD=$REC python3 $X/d89_miss_census.py --prep $W/prep_$a --arm $a --cfg $cfg --json $W/census_$a.json > $W/census_$a.txt 2>&1
    echo "  d89_miss_census $a rc=$? (cfg $cfg) -> $W/census_$a.txt"
    sed -n '/=== 1\./,/^$/p' $W/census_$a.txt; grep -E "^  -- \(" $W/census_$a.txt
done
STM_SCAN_RECORD= python3 census_score.py --check 2>&1 | tail -3
echo GATES_DONE
