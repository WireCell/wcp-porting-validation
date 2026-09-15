#!/usr/bin/env bash
# doc pdvd/101 sec 6 -- the knob-OFF byte-identity gates for this round's C++ (TrackFitting
# fit_weight_pow / assoc_cont_center, BlobSampler Stepped half_pitch, Facade point2wind_cont /
# Grouping::convert_3Dpoint_wire_cont, ClusteringFlagMatchedMains flag_unmatched).
#
# Every pair is the SAME config on two binaries: <new> ran libpin_k (this round's build, knobs at
# their defaults), <old> ran a pin that predates the round's edits.  The compiled configs are
# proven identical separately (sec 6: wcsonnet of wct-pr-perevt / wct-clustering against HEAD cfg).
#   1  SIM PR    d101koff (libpin_k)  vs d101b  (libpin_new = flag_unmatched build, before the three
#                lattice knobs)  -- 18+18 simulated muons, flag_mains_unmatched ON (the only path
#                that runs a fit on a light-less event)
#   2  PDVD PR   d101vnew (libpin_k)  vs d101vold (libpin, pre-round) -- production -nu on d16vnu
#   3  PDHD PR   d101hnew             vs d101hold                     -- production -nu on d16hnu
#   4  SBND PR   work-<s>-d101snew    vs work-<s>-d101sold            -- pr146 manifest
#   5  CLUS      <det>/work/<ev>_d101cnew vs _d101cold               -- BlobSampler in PointTreeBuilding
# Compared per event: every TTree of every *.root (awkward, NaN-safe), every member of every *.zip
# (sha256), every *.tar.gz via abtest/hash_archive.py --members, every *.json byte for byte.
# Symlinked inputs are skipped (they are the same file on both sides by construction).
# A pair set with ZERO compared events is reported as a dead net, never as a pass.
#
# Usage: bash d101_gates.sh > /home/xqian/tmp/d101/gates.log 2>&1; echo rc=$?
set -u
IMG=/home/xqian/toolkit-dev/wcp-porting-img
SB=$IMG/sbnd/sbnd_xin
echo "=== 0. pins"
for p in libpin libpin_new libpin_k; do
    echo "  $p clus md5 $(md5sum /home/xqian/tmp/d101/$p/libWireCellClus.so | cut -c1-12)"
done
md5sum /home/xqian/tmp/d101/libpin_k/libWireCell*.so > /home/xqian/tmp/d101/libpin_k_md5_after.txt
cmp -s /home/xqian/tmp/d101/libpin_k_md5_before.txt /home/xqian/tmp/d101/libpin_k_md5_after.txt \
    && echo "  libpin_k unchanged since creation ($(wc -l < /home/xqian/tmp/d101/libpin_k_md5_after.txt) libs)" \
    || echo "  *** libpin_k CHANGED -- every gate below is VOID ***"

IMG=$IMG SB=$SB python3 - <<'EOF'
import glob, hashlib, os, subprocess, zipfile
import awkward as ak, uproot
IMG, SB = os.environ["IMG"], os.environ["SB"]
HA = IMG + "/abtest/hash_archive.py"

def real_outputs(d):
    out = {}
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if os.path.islink(p) or not os.path.isfile(p):
            continue
        if f.endswith((".root", ".zip", ".tar.gz", ".json", ".tsv")) and not f.startswith("."):
            out[f] = p
    return out

def trees(fn):
    f = uproot.open(fn)
    res = {}
    for k in sorted(set(x.split(";")[0] for x in f.keys())):
        o = f[k]
        if not hasattr(o, "arrays"):
            continue
        a = o.arrays(library="ak")
        try:
            res[k] = ak.to_list(ak.nan_to_none(a))
        except Exception:
            res[k] = ak.to_list(a)
    return res

def zipm(fn):
    with zipfile.ZipFile(fn) as z:
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith("/")}

def tarm(fn):
    r = subprocess.run(["python3", HA, "--members", fn], capture_output=True, text=True)
    return r.stdout.replace(fn, "<f>")

def same_file(a, b):
    if a.endswith(".root"):
        ta, tb = trees(a), trees(b)
        if set(ta) != set(tb):
            return False, "tree set %s vs %s" % (sorted(ta), sorted(tb))
        bad = [t for t in ta if ta[t] != tb[t]]
        return (not bad), ("trees differ: %s" % bad if bad else "")
    if a.endswith(".zip"):
        za, zb = zipm(a), zipm(b)
        bad = sorted(n for n in set(za) | set(zb) if za.get(n) != zb.get(n))
        return (not bad), ("members differ: %s" % bad[:5] if bad else "")
    if a.endswith(".tar.gz"):
        return tarm(a) == tarm(b), "hash_archive members differ"
    return open(a, "rb").read() == open(b, "rb").read(), "bytes differ"

def gate(label, pairs):
    n_ev = n_files = 0; fails = []
    for new, old in pairs:
        if not (os.path.isdir(new) and os.path.isdir(old)):
            fails.append((os.path.basename(new), "missing dir")); continue
        A, B = real_outputs(new), real_outputs(old)
        if not B:
            fails.append((os.path.basename(new), "baseline has no outputs")); continue
        n_ev += 1
        for f in sorted(set(A) | set(B)):
            if f not in A or f not in B:
                fails.append((os.path.basename(new), "file only on one side: " + f)); continue
            ok, why = same_file(A[f], B[f]); n_files += 1
            if not ok:
                fails.append((os.path.basename(new), f + ": " + why))
    print("\n=== %s: %d event dirs, %d files compared" % (label, n_ev, n_files))
    for e, w in fails[:20]:
        print("    DIFF %s  %s" % (e, w))
    if n_ev == 0:
        print("    VERDICT: *** DEAD NET -- nothing compared, NOT a pass ***")
    elif fails:
        print("    VERDICT: *** NOT IDENTICAL on %d item(s) ***" % len(fails))
    else:
        print("    VERDICT: BYTE-IDENTICAL")

def arm_pairs(det, new, old):
    out = []
    for d in sorted(glob.glob("%s/%s/work/*_%s" % (IMG, det, old))):
        out.append((d[: -len(old)] + new, d))
    return out

# GATES selects sections (default all), so a finished section can be read while a later arm is
# still running; the authoritative run quoted in the doc uses the default.
sel = set(x.strip() for x in os.environ.get("GATES", "1,2,3,4,5").split(",") if x.strip())
if "1" in sel:
    gate("1. SIM PR, PDVD (d101koff vs d101b)", arm_pairs("pdvd", "d101koff", "d101b"))
    gate("1. SIM PR, PDHD (d101koff vs d101b)", arm_pairs("pdhd", "d101koff", "d101b"))
if "2" in sel:
    gate("2. PDVD PR data (d101vnew vs d101vold)", arm_pairs("pdvd", "d101vnew", "d101vold"))
if "3" in sel:
    gate("3. PDHD PR data (d101hnew vs d101hold)", arm_pairs("pdhd", "d101hnew", "d101hold"))
sb = []
for old in sorted(glob.glob(SB + "/work-*-d101sold")):
    new = old[: -len("d101sold")] + "d101snew"
    for e in sorted(glob.glob(old + "/pr_evt*")):
        sb.append((new + "/" + os.path.basename(e), e))
if "4" in sel:
    gate("4. SBND PR (d101snew vs d101sold)", sb)
# 6: the comparator's causal negative control -- a pair that DOES differ (the fit knobs on vs off,
# same pin, same input) must come out NOT IDENTICAL, or every PASS above is a blind comparator.
if "6" in sel:
    gate("6. CONTROL, must NOT be identical: PDVD sim d101kf vs d101koff", arm_pairs("pdvd", "d101kf", "d101koff"))
    gate("6. CONTROL, must NOT be identical: PDHD sim d101ks vs d101koff", arm_pairs("pdhd", "d101ks", "d101koff"))
if "5" not in sel:
    raise SystemExit(0)
# Staging-only tags left as they are (M13), none of which ran wire-cell: pdhd+pdvd d101cold/d101cnew
# (a log-name typo killed both runs), pdvd d101cold2/d101cnew2 (the replay lacked img-provenance.txt,
# which run_clus_evt.sh's wires guard reads under set -e).  The gate uses the tags that ran.
gate("5. CLUS PDHD 027409_0 (d101cnew2 vs d101cold2)", arm_pairs("pdhd", "d101cnew2", "d101cold2"))
gate("5. CLUS PDVD sim 900101_0 (d101cnew3 vs d101cold3)", arm_pairs("pdvd", "d101cnew3", "d101cold3"))
EOF
echo GATES_DONE
