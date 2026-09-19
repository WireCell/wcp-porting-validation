#!/usr/bin/env python3
"""doc pdvd/116 -- compiled-config proof: diff two compiled wire-cell PR jobs (d102_compile_pr.sh output) node by
node.  Nodes are keyed by (type, name); a differing key is reported as added / removed / changed with both values.
Read-only.  Exit 0 when the differing (node, key) set equals --expect (a comma list of type/name/key, or empty for
"byte-identical"), else 1.

Usage: d116_cfgdiff.py A.json B.json [--expect CreateSteinerGraph/pr/base_weight_blank_alpha,...] [--show]
"""
import argparse, json, sys


def nodes(path):
    j = json.load(open(path))
    out = {}
    for n in j:
        if not isinstance(n, dict):
            continue
        k = (n.get("type", ""), n.get("name", ""))
        out[k] = n.get("data", {}) if isinstance(n.get("data"), dict) else {"__data__": n.get("data")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--expect", default="")
    ap.add_argument("--show", action="store_true", help="print every differing value")
    a = ap.parse_args()
    A, B = nodes(a.a), nodes(a.b)
    diffs = []
    for k in sorted(set(A) | set(B)):
        if k not in A:
            diffs.append((k, "<node>", "added", None, None)); continue
        if k not in B:
            diffs.append((k, "<node>", "removed", None, None)); continue
        da, db = A[k], B[k]
        for key in sorted(set(da) | set(db)):
            if key not in da:
                diffs.append((k, key, "added", None, db[key]))
            elif key not in db:
                diffs.append((k, key, "removed", da[key], None))
            elif da[key] != db[key]:
                diffs.append((k, key, "changed", da[key], db[key]))
    got = sorted({f"{k[0]}/{k[1]}/{key}" for k, key, *_ in diffs})
    exp = sorted(x for x in a.expect.split(",") if x)
    print(f"{a.a} vs {a.b}: {len(diffs)} differing (node, key) pairs")
    for k, key, how, va, vb in diffs:
        if a.show or len(diffs) <= 40:
            sa = json.dumps(va)[:80] if va is not None else "-"
            sb = json.dumps(vb)[:80] if vb is not None else "-"
            print(f"  {k[0]}/{k[1]} {key} {how}: {sa} -> {sb}")
    ok = got == exp
    print(("RESULT PASS: " if ok else "RESULT FAIL: ") + ("byte-level node/key set identical" if not exp and ok else
          f"differing keys {got} vs expected {exp}"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
