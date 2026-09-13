#!/usr/bin/env bash
# doc pdhd/26 part A -- the gates for flipping the region-based Michel energy into PDHD production.
#
# Two questions, in this order (pre-registered in preregistered.txt before h26q2d launched).
#   MODE=additive  h26q2d (production file + tla_q2d.txt) vs h26conf (production).  Every PRE-EXISTING
#                  T_stm_michel branch and T_stm_michel_pts row bit-identical, exactly the 49 registered new
#                  branches and the one registered new tree (T_stm_michel_2d), 0 is_stm / michel_found /
#                  reject_bits changes, every other tree + zip member + calib json identical, census unmoved.
#                  FAIL => the production file is not edited.
#   MODE=confirm   h26q2dprod (the flipped file, no TLA) vs h26q2d.  Everything identical, including the
#                  49 new branches and every T_stm_michel_2d row: production runs the arm that was gated.
#
# Why the branch-set and tree-set lines are not optional (doc pdvd/93's defect): a per-candidate loop over
# the BASELINE's branches cannot see a branch present only in the new arm.  Here the new arm adds 49 of them
# by design, so the set is checked against the registered list, both ways.
#
# Fork by duplication (CLAUDE.md M10) of pdvd/docs/nf_sp_img_clus/scripts/d95_gates.sh; d95 untouched.
# Usage: MODE=additive bash d26_gates.sh > gate_h26q2d.txt 2>&1; echo rc=$?
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=/home/xqian/tmp/h27
REC=${STM_SCAN_RECORD:-$IMG/pdhd/docs/scan/pdhd_stm_michel_smx27_verdicts.json}
# d25_bragg_michel reads STM_SCAN_RECORD AT IMPORT and defaults to smx23.  The first run of this gate
# (gate_h26q2d_run1.txt) set REC but not the env, so its section 4 census was taken on smx23 -- both arms
# on the same record, so "same" still held, but not on the record the prediction named.  Export it.
export STM_SCAN_RECORD=$REC
MODE=${MODE:?set MODE=additive or MODE=confirm}
if [ "$MODE" = additive ]; then ARM=${ARM:-h26q2d}; BASE=${BASE:-h26conf}
else ARM=${ARM:-h26q2dprod}; BASE=${BASE:-h26q2d}; fi
PINDIR=${PINDIR:-/home/xqian/tmp/p96/libpin_p96}
cd $IMG
echo "mode $MODE: $ARM vs $BASE | record $REC"
[ -s "$REC" ] || { echo "ABORT: record missing"; exit 2; }

echo "=== 0. completeness, pin, loader deaths"
for a in $BASE $ARM; do
    n=$(compgen -G "pdhd/work/*_$a" | wc -l)
    ok=$(for e in pdhd/work/*_$a; do [ -s $e/tracking-pr.root ] && [ -s $e/tracking-stm.root ] && echo 1; done 2>/dev/null | wc -l)
    lt=$(grep -l "file too short" pdhd/work/*_$a/wct_pr_*.log 2>/dev/null | wc -l)
    echo "  pdhd $a dirs $n with tracking-pr.root+tracking-stm.root $ok loader-deaths $lt"
done
md5sum $PINDIR/*.so* > $W/libpin_md5_after.txt 2>/dev/null
cmp -s $W/libpin_md5_before.txt $W/libpin_md5_after.txt \
    && echo "  pin $PINDIR: $(wc -l < $W/libpin_md5_after.txt) libraries, md5 identical to the manifest taken before h26q2d launched" \
    || echo "  *** PIN CHANGED since the manifest -- every comparison below is VOID ***"
for a in $ARM; do grep -h "^arm=" /home/xqian/tmp/h25/DONE_$a 2>/dev/null | sed 's/^/  /'; done

MODE=$MODE ARM=$ARM BASE=$BASE REC=$REC python3 - <<'EOF'
import collections, glob, hashlib, json, os, sys, zipfile
import numpy as np, uproot, awkward as ak
E = os.environ
MODE, ARM, BASE = E["MODE"], E["ARM"], E["BASE"]
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/docs/scan/h25")
import d25_bragg_michel as BM

# PRE-REGISTERED (preregistered.txt, prediction 1c): the 49 branches p96vprod carries beyond h26conf's 149.
NEW_EXPECTED = sorted(
    ["michel_ke_q2d", "michel_ke_q2d_ctl", "michel_ke_q2d_gamma", "michel_ke_q2d_region", "michel_ke_q2d_total",
     "michel_q2d", "michel_q2d_ctl", "michel_q2d_ctl_dropped_plane", "michel_q2d_ctl_valid",
     "michel_q2d_dropped_plane", "michel_q2d_gamma", "michel_q2d_n_role0", "michel_q2d_reason",
     "michel_q2d_region", "michel_q2d_region_dropped_plane", "michel_q2d_valid"]
    + ["michel_q2d_%s_%s" % (g, p) for g in ("ctl_n", "ctl", "mu", "n", "nx", "raw", "region_mu", "region_n",
                                             "region_nd", "region") for p in "uvw"]
    + ["michel_q2d_%s" % p for p in "uvw"])
assert len(NEW_EXPECTED) == 49, len(NEW_EXPECTED)
NEW_TREES = {"tracking-pr.root": ["T_stm_michel_2d"], "tracking-stm.root": []} if MODE == "additive" else \
            {"tracking-pr.root": [], "tracking-stm.root": []}

def evdirs(arm):
    return {os.path.basename(d)[: -len(arm) - 1]: d for d in sorted(glob.glob("pdhd/work/*_" + arm))}

def read(arm):
    out, pts = {}, {}
    for ev, d in evdirs(arm).items():
        f = uproot.open(d + "/tracking-pr.root")
        if "T_stm_michel" not in f: continue
        t = f["T_stm_michel"].arrays(library="np")
        q = f["T_stm_michel_pts"].arrays(library="np")
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

fail = []
A, AP = read(ARM); B, BP = read(BASE)
print("\n=== 1. T_stm_michel: %s %d candidates vs %s %d" % (ARM, len(A), BASE, len(B)))
if not A or not B:
    print("    *** ONE SIDE IS EMPTY -- a dead net, not a pass ***"); sys.exit(1)
ba = {b for v in A.values() for b in v}; bb = {b for v in B.values() for b in v}
only_a, only_b = sorted(ba - bb), sorted(bb - ba)
print("    branches %s %d, %s %d | only in %s: %d | only in %s: %s" % (ARM, len(ba), BASE, len(bb), ARM, len(only_a), BASE, only_b or "none"))
want_new = NEW_EXPECTED if MODE == "additive" else []
unexp, miss = sorted(set(only_a) - set(want_new)), sorted(set(want_new) - set(only_a))
print("    new branches registered %d, got %d; unexpected: %s; missing: %s" % (len(want_new), len(only_a), unexp or "none", miss or "none"))
if unexp or miss or only_b: fail.append("branch set")
if set(A) != set(B):
    fail.append("candidate set")
    for k in sorted(set(A) ^ set(B)): print("    candidate only in one arm: %s" % k)
moved, bset, rowmoved = {}, collections.Counter(), []
for k in sorted(set(A) & set(B)):
    bs = sorted(b for b in B[k] if not same(A[k].get(b), B[k].get(b)))
    rows = sorted(c for c in BP[k] if not same(AP[k].get(c), BP[k][c])) + sorted(set(AP[k]) ^ set(BP[k]))
    if bs or rows:
        moved[k] = (bs, rows); bset.update(bs)
        if rows: rowmoved.append(k)
ident = len(set(A) & set(B)) - len(moved)
print("    candidates bit-identical on every %s branch and every T_stm_michel_pts column: %d / %d" % (
    "PRE-EXISTING" if MODE == "additive" else "", ident, len(set(A) | set(B))))
for k, (bs, rows) in list(moved.items())[:20]:
    print("    MOVED %-14s branches %s | pts columns %s" % (k, " ".join(bs[:10]), " ".join(rows[:6])))
g = lambda r, b: int(r.get(b, 0))
fl = [k for k in moved if g(A[k], "is_stm") != g(B[k], "is_stm")]
mf = [k for k in moved if g(A[k], "michel_found") != g(B[k], "michel_found")]
rb = [k for k in moved if g(A[k], "reject_bits") != g(B[k], "reject_bits")]
print("    is_stm flips %d, michel_found flips %d, reject_bits changes %d, point tables moved %d" % (len(fl), len(mf), len(rb), len(rowmoved)))
if moved: fail.append("pre-existing branch or point row moved")

print("\n=== 2. every tree in tracking-pr.root and tracking-stm.root, zip members, calib json")
EA, EB = evdirs(ARM), evdirs(BASE)
if set(EA) != set(EB): fail.append("event set"); print("    event sets differ")
tsame, tdiff, tdiff_ev = collections.Counter(), collections.Counter(), collections.defaultdict(list)
extra_trees = collections.Counter(); zd = []; cs = cd = 0; cd_ev = []
def L(x):
    try: return ak.to_list(ak.nan_to_none(x))
    except Exception: return ak.to_list(x)
for ev in sorted(set(EA) & set(EB)):
    for fn in ("tracking-pr.root", "tracking-stm.root"):
        fa, fb = uproot.open(EA[ev] + "/" + fn), uproot.open(EB[ev] + "/" + fn)
        na = set(k.split(";")[0] for k in fa.keys()); nb = set(k.split(";")[0] for k in fb.keys())
        for t in sorted(na - nb):
            extra_trees[(fn, t)] += 1
        for t in sorted(nb - na):
            tdiff[(fn, t + " (LOST)")] += 1; tdiff_ev[(fn, t + " (LOST)")].append(ev)
        for t in sorted(na & nb):
            try:
                ta, tb = fa[t].arrays(library="ak"), fb[t].arrays(library="ak")
                fields = tb.fields
                if fn == "tracking-pr.root" and t == "T_stm_michel":
                    ok = all(L(ta[n]) == L(tb[n]) for n in fields)            # the base's branches; set checked in 1
                else:
                    ok = set(ta.fields) == set(fields) and all(L(ta[n]) == L(tb[n]) for n in fields)
            except Exception as e:
                ok = False
            (tsame if ok else tdiff)[(fn, t)] += 1
            if not ok: tdiff_ev[(fn, t)].append(ev)
    za, zb = EA[ev] + "/mabc-pr.zip", EB[ev] + "/mabc-pr.zip"
    if os.path.exists(za) != os.path.exists(zb) or (os.path.exists(za) and members(za) != members(zb)):
        zd.append(ev)
    ja, jb = sorted(glob.glob(EA[ev] + "/calib-pr-evt*.json")), sorted(glob.glob(EB[ev] + "/calib-pr-evt*.json"))
    if [os.path.basename(x) for x in ja] == [os.path.basename(x) for x in jb] and all(open(x, "rb").read() == open(y, "rb").read() for x, y in zip(ja, jb)):
        cs += 1
    else:
        cd += 1; cd_ev.append(ev)
print("    %d events.  trees identical on every event: %s" % (len(set(EA) & set(EB)), sorted("%s:%s" % t for t in tsame if t not in tdiff)))
for t in sorted(tdiff): print("    *** %s:%s differs on %d events: %s" % (t[0], t[1], tdiff[t], " ".join(tdiff_ev[t][:10])))
want_trees = {(fn, t) for fn, ts in NEW_TREES.items() for t in ts}
got_trees = set(extra_trees)
print("    trees only in %s: %s (registered: %s)" % (ARM, {("%s:%s" % t): n for t, n in sorted(extra_trees.items())} or "none",
                                                     sorted("%s:%s" % t for t in want_trees) or "none"))
if tdiff: fail.append("tree differs")
if got_trees != want_trees: fail.append("tree set")
print("    mabc-pr.zip members differ on %d events: %s" % (len(zd), " ".join(zd) or "none"))
print("    calib-pr-evt*.json byte-identical %d, differ %d %s" % (cs, cd, " ".join(cd_ev)))
if zd: fail.append("zip")
if cd: fail.append("calib")

if MODE == "confirm":
    # T_stm_michel_2d is in section 2's tree loop (present in both arms), so its rows were compared there.
    pass
else:
    print("\n=== 3. what the new branches say (%s)" % ARM)
    v = list(A.values())
    nval = sum(1 for a in v if int(a["michel_q2d_valid"]) == 1)
    print("    michel_q2d_valid == 1 on %d / %d candidates; reason codes %s" % (
        nval, len(v), dict(sorted(collections.Counter(int(a["michel_q2d_reason"]) for a in v).items()))))
    print("    michel_q2d_ctl_valid == 1 on %d / %d" % (sum(1 for a in v if int(a["michel_q2d_ctl_valid"]) == 1), len(v)))
    fnd = [a for a in v if int(a["michel_found"]) == 1 and int(a["is_stm"]) == 1]
    print("    accepted + michel_found: %d; michel_ke_q2d_region median %.2f MeV, michel_ke_best median %.2f MeV" % (
        len(fnd), np.median([float(a["michel_ke_q2d_region"]) for a in fnd]), np.median([float(a["michel_ke_best"]) for a in fnd])))

print("\n=== 4. census on the record (d25_bragg_michel.items / census, APA0 strict / majority / all)")
for pop in ("strict", "majority", "all"):
    ca = BM.census(BM.items("pdhd", ARM, pop)[0]); cb = BM.census(BM.items("pdhd", BASE, pop)[0])
    print("    %-8s %s is_stm %s michel %s | %s is_stm %s michel %s -> %s" % (
        pop, ARM, ca[0], ca[1], BASE, cb[0], cb[1], "same" if ca == cb else "*** MOVED ***"))
    if ca != cb: fail.append("census " + pop)

print("\nVERDICT: %s" % (("PASS -- " + ("PURELY ADDITIVE: every pre-existing branch, point row, tree, zip, calib and census unmoved; "
                                        "exactly the 49 registered branches and T_stm_michel_2d added"
                                        if MODE == "additive" else
                                        "BIT-IDENTICAL: production runs exactly the gated arm (all branches, rows, trees incl. T_stm_michel_2d, zips, calib)"))
                     if not fail else "*** FAIL: %s ***" % "; ".join(fail)))
sys.exit(0 if not fail else 1)
EOF
