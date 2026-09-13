#!/usr/bin/env bash
# doc pdhd/28 -- gates for the wire-lookup fix (michel_q2d_region_wire_lookup), both detectors.
#
#   MODE=identical   ARM vs BASE must agree on EVERYTHING: every T_stm_michel branch (both ways), every
#                    T_stm_michel_pts row, every tree in tracking-pr.root / tracking-stm.root (T_stm_michel_2d
#                    included), every mabc-pr.zip member, every calib json, the census.  G1, G2, F.
#   MODE=wirelookup  ARM (knob on) vs BASE (knob off), same pin.  preregistered.txt K1 + K2:
#                    moved T_stm_michel branches only in the registered region/control set, exactly one new
#                    branch michel_q2d_n_rewired, 0 verdict changes, every pts row / other tree / zip / calib /
#                    census identical; in T_stm_michel_2d every W row and every row of a channel with ONE
#                    (face, wire) identical in every column, rows added or removed only as role-0 rows.
#
# Fork by duplication (CLAUDE.md M10) of ../d26/d26_gates.sh; d26 untouched.  Differences: DET is a parameter,
# the mode set, the registered branch set, and the T_stm_michel_2d row comparison with the wire geometry.
# Usage: DET=pdhd MODE=identical ARM=h28off BASE=h26q2dprod bash d28_gates.sh > gate_h28off.txt 2>&1; echo rc=$?
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
export STM_SCAN_RECORD=${STM_SCAN_RECORD:-$IMG/pdhd/docs/scan/pdhd_stm_michel_smx27_verdicts.json}
DET=${DET:?pdhd or pdvd}; MODE=${MODE:?identical or wirelookup}; ARM=${ARM:?}; BASE=${BASE:?}
cd $IMG
echo "det $DET mode $MODE: $ARM vs $BASE | PDHD record $STM_SCAN_RECORD"
echo "=== 0. completeness, loader deaths, pins"
for a in $BASE $ARM; do
    n=$(compgen -G "$DET/work/*_$a" | wc -l)
    ok=$(for e in $DET/work/*_$a; do [ -s $e/tracking-pr.root ] && [ -s $e/tracking-stm.root ] && echo 1; done 2>/dev/null | wc -l)
    lt=$(grep -l "file too short" $DET/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    echo "  $DET $a dirs $n with tracking-pr.root+tracking-stm.root $ok loader-deaths $lt"
    grep -h "libWireCellClus md5" /home/xqian/tmp/h28/arm_$a.log 2>/dev/null | sed 's/^/  /'
done

DET=$DET MODE=$MODE ARM=$ARM BASE=$BASE python3 - <<'EOF'
import bz2, collections, glob, hashlib, json, os, sys, zipfile
import numpy as np, uproot, awkward as ak
E = os.environ
DET, MODE, ARM, BASE = E["DET"], E["MODE"], E["ARM"], E["BASE"]
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/docs/scan/h25")
import d25_bragg_michel as BM

ALLOWED = set(["michel_q2d_region", "michel_ke_q2d_region", "michel_q2d_region_dropped_plane", "michel_q2d_n_role0",
               "michel_q2d_ctl", "michel_ke_q2d_ctl", "michel_q2d_ctl_dropped_plane"]
              + ["michel_q2d_%s_%s" % (g, p) for g in ("region", "region_mu", "region_n", "region_nd", "ctl", "ctl_n")
                 for p in "uvw"])
NEW_EXPECTED = ["michel_q2d_n_rewired"] if MODE == "wirelookup" else []
GEOM = {"pdhd": "protodunehd-wires-larsoft-v1.json.bz2", "pdvd": "protodunevd-wires-larsoft-v7-uvwfit.json.bz2"}[DET]

def evdirs(arm):
    return {os.path.basename(d)[: -len(arm) - 1]: d for d in sorted(glob.glob(DET + "/work/*_" + arm))}

def read(arm):
    out, pts = {}, {}
    for ev, d in evdirs(arm).items():
        f = uproot.open(d + "/tracking-pr.root")
        if "T_stm_michel" not in f: continue
        t = f["T_stm_michel"].arrays(library="np"); q = f["T_stm_michel_pts"].arrays(library="np")
        for i in range(len(t["cluster_id"])):
            c = int(t["cluster_id"][i]); k = "%s/%d" % (ev, c)
            out[k] = {kk: t[kk][i] for kk in t}
            sel = q["cluster_id"] == c
            pts[k] = {kk: q[kk][sel] for kk in q}
    return out, pts

def same(a, b):
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            a, b = np.asarray(a), np.asarray(b)
            return a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
        if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)) and np.isnan(a) and np.isnan(b): return True
        return bool(a == b)
    except Exception:
        return False

def members(fn):
    with zipfile.ZipFile(fn) as z:
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith("/")}

def L(x):
    try: return ak.to_list(ak.nan_to_none(x))
    except Exception: return ak.to_list(x)

fail = []
A, AP = read(ARM); B, BP = read(BASE)
print("\n=== 1. T_stm_michel: %s %d candidates vs %s %d" % (ARM, len(A), BASE, len(B)))
if not A or not B: print("    *** ONE SIDE IS EMPTY -- a dead net ***"); sys.exit(1)
ba = {b for v in A.values() for b in v}; bb = {b for v in B.values() for b in v}
only_a, only_b = sorted(ba - bb), sorted(bb - ba)
print("    branches %s %d, %s %d | only in %s: %s | only in %s: %s" % (ARM, len(ba), BASE, len(bb), ARM, only_a or "none", BASE, only_b or "none"))
if sorted(only_a) != sorted(NEW_EXPECTED) or only_b: fail.append("branch set")
if set(A) != set(B):
    fail.append("candidate set")
    for k in sorted(set(A) ^ set(B))[:10]: print("    candidate only in one arm: %s" % k)
moved = collections.Counter(); bad_moved = collections.Counter(); rowmoved = []; nmoved = 0
for k in sorted(set(A) & set(B)):
    bs = [b for b in B[k] if not same(A[k].get(b), B[k].get(b))]
    rows = [c for c in BP[k] if not same(AP[k].get(c), BP[k][c])] + sorted(set(AP[k]) ^ set(BP[k]))
    if bs: nmoved += 1
    for b in bs:
        moved[b] += 1
        if MODE == "identical" or b not in ALLOWED: bad_moved[b] += 1
    if rows: rowmoved.append(k)
g = lambda r, b: int(r.get(b, 0))
common = sorted(set(A) & set(B))
fl = sum(g(A[k], "is_stm") != g(B[k], "is_stm") for k in common)
mf = sum(g(A[k], "michel_found") != g(B[k], "michel_found") for k in common)
rb = sum(g(A[k], "reject_bits") != g(B[k], "reject_bits") for k in common)
print("    candidates with any moved branch %d / %d; moved branches (candidates): %s" % (nmoved, len(common), dict(moved.most_common()) or "none"))
print("    branches moved OUTSIDE the %s rule: %s" % ("no-change" if MODE == "identical" else "registered region/control", dict(bad_moved) or "none"))
print("    is_stm flips %d, michel_found flips %d, reject_bits changes %d, candidates with a moved T_stm_michel_pts row %d" % (fl, mf, rb, len(rowmoved)))
if bad_moved: fail.append("branch outside rule")
if fl or mf or rb: fail.append("verdict moved")
if rowmoved: fail.append("pts rows moved")
if MODE == "wirelookup":
    nr = [int(a["michel_q2d_n_rewired"]) for a in A.values()]
    print("    michel_q2d_n_rewired: > 0 on %d / %d candidates, p50 %.0f p90 %.0f" % (sum(x > 0 for x in nr), len(nr), np.median(nr), np.percentile(nr, 90)))

print("\n=== 2. every tree, zip members, calib json")
EA, EB = evdirs(ARM), evdirs(BASE)
if set(EA) != set(EB): fail.append("event set"); print("    event sets differ")
tsame, tdiff, tdiff_ev = collections.Counter(), collections.Counter(), collections.defaultdict(list)
zd, cd_ev = [], []
for ev in sorted(set(EA) & set(EB)):
    for fn in ("tracking-pr.root", "tracking-stm.root"):
        fa, fb = uproot.open(EA[ev] + "/" + fn), uproot.open(EB[ev] + "/" + fn)
        na = set(k.split(";")[0] for k in fa.keys()); nb = set(k.split(";")[0] for k in fb.keys())
        for t in sorted(na ^ nb): tdiff[(fn, t + " (only one arm)")] += 1; tdiff_ev[(fn, t + " (only one arm)")].append(ev)
        for t in sorted(na & nb):
            if MODE == "wirelookup" and fn == "tracking-pr.root" and t in ("T_stm_michel", "T_stm_michel_2d"):
                continue                                   # section 1 / section 3
            try:
                ta, tb = fa[t].arrays(library="ak"), fb[t].arrays(library="ak")
                ok = set(ta.fields) == set(tb.fields) and all(L(ta[n]) == L(tb[n]) for n in tb.fields)
            except Exception:
                ok = False
            (tsame if ok else tdiff)[(fn, t)] += 1
            if not ok: tdiff_ev[(fn, t)].append(ev)
    za, zb = EA[ev] + "/mabc-pr.zip", EB[ev] + "/mabc-pr.zip"
    if os.path.exists(za) != os.path.exists(zb) or (os.path.exists(za) and members(za) != members(zb)): zd.append(ev)
    ja, jb = sorted(glob.glob(EA[ev] + "/calib-pr-evt*.json")), sorted(glob.glob(EB[ev] + "/calib-pr-evt*.json"))
    if not ([os.path.basename(x) for x in ja] == [os.path.basename(x) for x in jb] and all(open(x, "rb").read() == open(y, "rb").read() for x, y in zip(ja, jb))):
        cd_ev.append(ev)
print("    %d events.  trees identical on every event: %s" % (len(set(EA) & set(EB)), sorted("%s:%s" % t for t in tsame if t not in tdiff)))
for t in sorted(tdiff): print("    *** %s:%s differs on %d events: %s" % (t[0], t[1], tdiff[t], " ".join(tdiff_ev[t][:8])))
print("    mabc-pr.zip members differ on %d events %s | calib json differ on %d events %s" % (len(zd), " ".join(zd[:8]), len(cd_ev), " ".join(cd_ev[:8])))
if tdiff: fail.append("tree differs")
if zd: fail.append("zip")
if cd_ev: fail.append("calib")

if MODE == "wirelookup":
    print("\n=== 3. T_stm_michel_2d rows, the negative control (K2)")
    g_ = json.load(bz2.open("/nfs/data/1/xqian/toolkit-dev/wire-cell-data/" + GEOM))["Store"]
    nfw = collections.Counter()
    for an in g_["anodes"]:
        an = an["Anode"]
        for fi in an["faces"]:
            for pi in g_["faces"][fi]["Face"]["planes"]:
                for wi in g_["planes"][pi]["Plane"]["wires"]:
                    nfw[(an["ident"], g_["wires"][wi]["Wire"]["channel"])] += 1
    KEY = ["cluster_id", "apa", "plane", "channel", "time"]
    FIXED = ["charge", "charge_err", "pred_mu", "pred_all", "flag", "shared", "xshared", "sel"]
    MOVABLE = ["face", "wire", "time_slice", "d_stop_cm", "d_ctl_cm", "own_blob", "role"]
    c = collections.Counter(); unknown_ch = 0; no_table = 0
    for ev in sorted(set(EA) & set(EB)):
        fa_, fb_ = uproot.open(EA[ev] + "/tracking-pr.root"), uproot.open(EB[ev] + "/tracking-pr.root")
        ha, hb = "T_stm_michel_2d" in fa_, "T_stm_michel_2d" in fb_
        if ha != hb:
            c["T_stm_michel_2d in one arm only"] += 1; continue
        if not ha:
            no_table += 1; continue                   # an event with no candidate writes no table in either arm
        ta = fa_["T_stm_michel_2d"].arrays(library="np")
        tb = fb_["T_stm_michel_2d"].arrays(library="np")
        if set(ta) != set(tb): c["column set differs"] += 1; continue
        ka = {tuple(int(ta[k][i]) for k in KEY): i for i in range(len(ta["cluster_id"]))}
        kb = {tuple(int(tb[k][i]) for k in KEY): i for i in range(len(tb["cluster_id"]))}
        if len(ka) != len(ta["cluster_id"]) or len(kb) != len(tb["cluster_id"]): c["non-unique row key"] += 1
        for key in set(ka) | set(kb):
            m = nfw.get((key[1], key[3]), 0)
            if m == 0: unknown_ch += 1
            single = key[2] == 2 or m <= 1
            cls = "W" if key[2] == 2 else ("U/V single" if m <= 1 else "U/V multi")
            if key not in ka or key not in kb:
                i = ka.get(key, kb.get(key)); t = ta if key in ka else tb
                role = int(t["role"][i])
                c[(cls, "row only in " + (ARM if key in ka else BASE), "role %d" % role)] += 1
                continue
            ia, ib = ka[key], kb[key]
            fixed_ok = all(same(ta[f][ia], tb[f][ib]) for f in FIXED)
            mov_ok = all(same(ta[f][ia], tb[f][ib]) for f in MOVABLE)
            c[(cls, "identical" if fixed_ok and mov_ok else ("placement moved" if fixed_ok else "CHARGE COLUMN MOVED"))] += 1
    for k in sorted(c, key=str): print("    %-60s %d" % (" | ".join(k) if isinstance(k, tuple) else k, c[k]))
    print("    rows whose (apa, channel) is not in %s: %d | events with no T_stm_michel_2d in either arm: %d" % (GEOM, unknown_ch, no_table))
    bad = [k for k in c if isinstance(k, tuple) and (
        (k[0] in ("W", "U/V single") and k[1] != "identical") or k[1] == "CHARGE COLUMN MOVED" or
        (k[1].startswith("row only") and k[2] != "role 0"))]
    bad += [k for k in c if not isinstance(k, tuple)]
    if bad: fail.append("2-D rows outside K2: %s" % bad)

print("\n=== 4. census on the record")
pops = ("strict", "majority", "all") if DET == "pdhd" else ("all",)
for pop in pops:
    ca = BM.census(BM.items(DET, ARM, pop)[0]); cb = BM.census(BM.items(DET, BASE, pop)[0])
    print("    %-8s %s is_stm %s michel %s | %s is_stm %s michel %s -> %s" % (pop, ARM, ca[0], ca[1], BASE, cb[0], cb[1], "same" if ca == cb else "*** MOVED ***"))
    if ca != cb: fail.append("census " + pop)

print("\nVERDICT: %s" % ("PASS -- " + ("BIT-IDENTICAL on every branch, row, tree, zip, calib and census" if MODE == "identical" else
                         "only the registered region/control branches and multi-wire U/V placement moved; W and single-wire rows, "
                         "verdicts, pts, trees, zips, calib and census identical") if not fail else "*** FAIL: %s ***" % "; ".join(map(str, fail))))
sys.exit(0 if not fail else 1)
EOF
