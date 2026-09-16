#!/usr/bin/env python3
"""doc pdvd/109 -- the gate for the PDHD STM Bee-layer rescope: prove it is DISPLAY-ONLY.

    python3 d109_display_gate.py --a d108hflip --b d109hstm

Arm A is the pre-change arm (doc pdvd/108 production config), arm B the same events re-run
with the four `require_flag:'STM'` gates and `stm_tagged` removed.  Same pctrees (B symlinks
A's), same pinned libraries -- ONLY the compiled jsonnet differs -- so any difference outside
the Bee point layers would be a defect.

Four checks:

  G1  RECONSTRUCTION UNCHANGED.  Every branch of T_stm_michel / T_stm_michel_pts
      (tracking-pr.root) and T_stm_pass / T_stm_eval (tracking-stm.root) must be
      element-wise identical.  These are the tagger and Michel verdict tables -- the
      physics the display is a view of.

  G2  CALIB DUMP UNCHANGED.  calib-pr-evt*.json compared as parsed JSON.

  G3  BEE LAYERS: only the intended ones move.  Every layer other than the four STM
      layers must be content-identical (arrays compared member-wise, never raw zip bytes:
      archives embed mtimes, M2).

  G4  THE RESCOPE ITSELF.  B's `stm_fit` must equal A's `stm_fit` restricted to the
      STM-tagged clusters, and B's `stm` must equal A's `stm_tagged` on every array --
      which is what makes dropping the `stm_tagged` layer a rename rather than a loss.
"""
import argparse, glob, json, os, sys, zipfile
import numpy as np

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
DATA = "data/0/"


def arm_dirs(det, arm):
    return {os.path.basename(d)[: -len(arm) - 1]: d for d in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}"))}


def bee_layers(d):
    z = os.path.join(d, "mabc-pr.zip")
    out = {}
    if not os.path.exists(z):
        return out
    with zipfile.ZipFile(z) as zz:
        for n in zz.namelist():
            if n.startswith(DATA) and n.endswith(".json"):
                out[n[len(DATA):-len(".json")]] = json.loads(zz.read(n))
    return out


def same_json(a, b):
    """Deep equality for Bee layer payloads.  Most are dicts of numeric arrays; the
    particle-flow layer is a list of dicts, so fall back to plain == for anything
    that is not a dict."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return (a == b), ("" if a == b else "payload differs (non-dict layer)")
    if set(a) != set(b):
        return False, "keys %s vs %s" % (sorted(a), sorted(b))
    for k in sorted(a):
        va, vb = a[k], b[k]
        if isinstance(va, list) and isinstance(vb, list):
            if len(va) != len(vb):
                return False, "%s len %d vs %d" % (k, len(va), len(vb))
            if va and isinstance(va[0], (int, float)):
                if not np.array_equal(np.asarray(va), np.asarray(vb)):
                    return False, "%s values differ" % k
            elif va != vb:
                return False, "%s differ" % k
        elif va != vb:
            return False, "%s %r vs %r" % (k, va, vb)
    return True, ""


def trees(path, names):
    import uproot
    out = {}
    if not os.path.exists(path):
        return out
    f = uproot.open(path)
    for t in names:
        if t in f or t + ";1" in f:
            out[t] = f[t].arrays(library="np")
    return out


def cmp_trees(ta, tb, tag, fails):
    for t in sorted(set(ta) | set(tb)):
        if t not in ta or t not in tb:
            fails.append("%s: tree %s present in only one arm" % (tag, t)); continue
        A, B = ta[t], tb[t]
        if set(A) != set(B):
            fails.append("%s/%s: branch sets differ" % (tag, t)); continue
        for br in sorted(A):
            x, y = A[br], B[br]
            if len(x) != len(y):
                fails.append("%s/%s/%s: %d vs %d rows" % (tag, t, br, len(x), len(y))); continue
            try:
                ok = all(np.array_equal(np.asarray(p), np.asarray(q)) for p, q in zip(x, y)) \
                     if x.dtype == object else np.array_equal(x, y)
            except Exception as e:
                ok = False
            if not ok:
                fails.append("%s/%s/%s: values differ" % (tag, t, br))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="d108hflip"); ap.add_argument("--b", default="d109hstm")
    ap.add_argument("--det", default="pdhd")
    a = ap.parse_args()
    A, B = arm_dirs(a.det, a.a), arm_dirs(a.det, a.b)
    evs = sorted(set(A) & set(B))
    print("# doc pdvd/109 display-only gate (%s): A=%s (pre) vs B=%s (rescoped)" % (a.det, a.a, a.b))
    print("# %d shared event(s): %s" % (len(evs), ", ".join(evs)))
    print("# A-only: %s   B-only: %s" % (sorted(set(A) - set(B)) or "none", sorted(set(B) - set(A)) or "none"))
    if not evs:
        print("FAIL: no shared events"); return 2

    STM_LAYERS = {"0-stm_fit-global", "0-stm-global", "0-steiner_graph-global",
                  "0-steiner_terminals-global", "0-stm_tagged-global"}
    fails, notes = [], []
    for ev in evs:
        da, db = A[ev], B[ev]
        # --- G1
        cmp_trees(trees(f"{da}/tracking-pr.root", ["T_stm_michel", "T_stm_michel_pts"]),
                  trees(f"{db}/tracking-pr.root", ["T_stm_michel", "T_stm_michel_pts"]), f"{ev} G1", fails)
        cmp_trees(trees(f"{da}/tracking-stm.root", ["T_stm_pass", "T_stm_eval"]),
                  trees(f"{db}/tracking-stm.root", ["T_stm_pass", "T_stm_eval"]), f"{ev} G1", fails)
        # --- G2
        ca, cb = glob.glob(f"{da}/calib-pr-evt*.json"), glob.glob(f"{db}/calib-pr-evt*.json")
        if len(ca) == len(cb) == 1:
            ok, why = same_json(json.load(open(ca[0])), json.load(open(cb[0])))
            if not ok: fails.append("%s G2 calib: %s" % (ev, why))
        elif ca or cb:
            fails.append("%s G2 calib: %d vs %d file(s)" % (ev, len(ca), len(cb)))
        # --- G3 / G4
        la, lb = bee_layers(da), bee_layers(db)
        for k in sorted((set(la) | set(lb)) - STM_LAYERS):
            if k not in la or k not in lb:
                fails.append("%s G3: layer %s in only one arm" % (ev, k)); continue
            ok, why = same_json(la[k], lb[k])
            if not ok: fails.append("%s G3 layer %s: %s" % (ev, k, why))
        # G4a: stm_fit gated
        fa, fb = la.get("0-stm_fit-global"), lb.get("0-stm_fit-global")
        tagged = set(np.asarray(lb.get("0-stm-global", {}).get("cluster_id", []), int).tolist())
        if fa and fb:
            cid = np.asarray(fa["cluster_id"], int); keep = np.isin(cid, list(tagged))
            pred = {k: (list(np.asarray(v)[keep]) if isinstance(v, list) and len(v) == len(cid) else v)
                    for k, v in fa.items()}
            ok, why = same_json(pred, fb)
            if not ok: fails.append("%s G4a stm_fit != A gated to tagged: %s" % (ev, why))
            notes.append("%s  stm_fit %5d -> %5d pts, %3d -> %3d clusters"
                         % (ev, len(cid), len(fb["cluster_id"]),
                            len(set(cid.tolist())), len(set(np.asarray(fb["cluster_id"], int).tolist()))))
        # G4b: B's stm == A's stm_tagged
        sa, sb = la.get("0-stm_tagged-global"), lb.get("0-stm-global")
        if sa and sb:
            xa = {k: v for k, v in sa.items() if k != "type"}
            xb = {k: v for k, v in sb.items() if k != "type"}
            ok, why = same_json(xa, xb)
            if not ok: fails.append("%s G4b stm != A stm_tagged: %s" % (ev, why))
        elif sa or sb:
            fails.append("%s G4b: stm_tagged/stm missing on one side" % ev)
        # layer inventory
        if ev == evs[0]:
            print("#\n# Bee layer inventory\n#   A: %s\n#   B: %s" % (sorted(la), sorted(lb)))

    print("#\n# stm_fit layer, per event:")
    for n in notes: print("#   " + n)
    print()
    if fails:
        print("GATE FAIL (%d):" % len(fails))
        for f in fails[:40]: print("   " + f)
        return 1
    print("GATE PASS: G1 verdict trees identical, G2 calib dump identical, G3 every non-STM Bee "
          "layer identical,\n           G4 stm_fit == pre-arm gated to the tagged set and stm == pre-arm "
          "stm_tagged (type string aside).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
