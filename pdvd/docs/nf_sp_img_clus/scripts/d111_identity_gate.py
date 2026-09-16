#!/usr/bin/env python3
"""doc pdvd/111 -- identity gate between two PR arms on the same pctrees: is arm B's output the same as arm A's?

Used for (i) the log-only traces (WCT_STM_PATH_DEBUG / WCT_TRAJ_ASSOC_DEBUG) and (ii) every knob-OFF gate.  Per event
present in both arms:
  * mabc-pr.zip: member content hashes (abtest/hash_archive.py rule: sha256(name + payload), sorted by name) -- never
    raw bytes (M2, zip timestamps);
  * tracking-stm.root and tracking-pr.root: every TTree, every branch, value for value (numpy array_equal, NaN == NaN;
    jagged branches compared element-wise).
Also reports events present in only one arm.  Exit status 0 only when every common event matches everywhere and the
event sets agree (or --allow-subset is given and B's events are a subset of A's).

Usage:
  d111_identity_gate.py --det pdhd --a d108hflip --b d111htr [--events 029107_16,...] [--allow-subset]
      [--ignore-members stm_tagged,stm,stm_fit,steiner_graph,steiner_terminals]
"""
import argparse, glob, hashlib, os, sys, zipfile
import numpy as np
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"


def arm_events(det, arm):
    return {os.path.basename(d)[:-len(arm) - 1]: d for d in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}"))}


def zip_hash(path, ignore=()):
    roll = hashlib.sha256(); n = 0; per = {}
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            layer = name.split("/")[-1].split("-", 1)[-1].rsplit(".json", 1)[0]
            layer = layer[:-len("-global")] if layer.endswith("-global") else layer
            if name.endswith("/") or layer in ignore:
                continue
            h = hashlib.sha256(name.encode() + zf.read(name)).hexdigest()
            per[name] = h; roll.update(h.encode()); n += 1
    return roll.hexdigest(), n, per


def same_array(x, y):
    if isinstance(x, np.ndarray) and x.dtype == object:
        if len(x) != len(y):
            return False
        return all(same_array(np.asarray(a), np.asarray(b)) for a, b in zip(x, y))
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return False
    if x.dtype.kind == "f":
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def cmp_root(pa, pb):
    """return (ok, message) comparing all trees/branches of two ROOT files."""
    fa, fb = uproot.open(pa), uproot.open(pb)
    ka = sorted({k.split(";")[0] for k, c in fa.classnames().items() if c == "TTree"})
    kb = sorted({k.split(";")[0] for k, c in fb.classnames().items() if c == "TTree"})
    if ka != kb:
        return False, f"tree sets differ: {ka} vs {kb}"
    nbr = 0
    for t in ka:
        ta, tb = fa[t], fb[t]
        if sorted(ta.keys()) != sorted(tb.keys()):
            return False, f"{t}: branch sets differ"
        if ta.num_entries != tb.num_entries:
            return False, f"{t}: entries {ta.num_entries} vs {tb.num_entries}"
        if ta.num_entries == 0:
            continue
        A = ta.arrays(library="np"); B = tb.arrays(library="np")
        for br in sorted(A):
            nbr += 1
            if not same_array(A[br], B[br]):
                return False, f"{t}.{br} differs"
    return True, f"{len(ka)} trees, {nbr} branches"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--events", default="")
    ap.add_argument("--allow-subset", action="store_true")
    ap.add_argument("--ignore-members", default="",
                    help="comma-separated Bee layer names excluded from the zip hash (e.g. the doc-109 re-scoped PDHD "
                         "layers stm_tagged,stm,stm_fit,steiner_graph,steiner_terminals when A predates 9d8fd892)")
    a = ap.parse_args()
    A, B = arm_events(a.det, a.a), arm_events(a.det, a.b)
    if a.events:
        want = set(a.events.split(","))
        A = {k: v for k, v in A.items() if k in want}; B = {k: v for k, v in B.items() if k in want}
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    print(f"# doc pdvd/111 identity gate ({a.det}): A={a.a} ({len(A)} evt)  B={a.b} ({len(B)} evt)"
          + (f"  zip members ignored: {a.ignore_members}" if a.ignore_members else ""))
    fail = 0
    if only_b or (only_a and not a.allow_subset):
        print(f"  EVENT SETS DIFFER: only in A {only_a[:10]}{'...' if len(only_a) > 10 else ''}; only in B {only_b}")
        fail += 1
    for ev in sorted(set(A) & set(B)):
        msgs = []; ok = True
        za, zb = f"{A[ev]}/mabc-pr.zip", f"{B[ev]}/mabc-pr.zip"
        if os.path.exists(za) and os.path.exists(zb):
            ign = tuple(x for x in a.ignore_members.split(",") if x)
            ha, na, pa = zip_hash(za, ign); hb, nb, pb = zip_hash(zb, ign)
            if ha != hb:
                ok = False
                bad = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
                msgs.append(f"zip DIFF ({na} vs {nb} members; first differing {bad[:3]})")
            else:
                msgs.append(f"zip {ha[:12]} ({na})")
        else:
            ok = False; msgs.append("zip MISSING")
        for kind in ("stm", "pr"):
            ra, rb = f"{A[ev]}/tracking-{kind}.root", f"{B[ev]}/tracking-{kind}.root"
            if not (os.path.exists(ra) and os.path.exists(rb)):
                ok = False; msgs.append(f"{kind} MISSING"); continue
            same, m = cmp_root(ra, rb)
            ok &= same
            msgs.append(f"{kind} {'ok' if same else 'DIFF'} ({m})")
        print(f"  {ev}: {'IDENTICAL' if ok else 'DIFFERS'} | " + " | ".join(msgs), flush=True)
        fail += (not ok)
    n = len(set(A) & set(B))
    print(f"RESULT {'PASS' if fail == 0 else 'FAIL'}: {n - (fail if fail <= n else n)} / {n} common events identical; failures {fail}")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
