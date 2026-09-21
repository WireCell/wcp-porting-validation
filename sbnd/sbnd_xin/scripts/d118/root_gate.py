#!/usr/bin/env python3
"""doc sbnd_xin/118 gate G1b: compare tracking-pr.root by TREE CONTENT, not container bytes.

A ROOT file is not byte-reproducible (timestamps, UUIDs, and here two writers -- the tracking
visitor opens RECREATE and the tagger output opens UPDATE), so hashing the file answers a question
nobody asked.  This walks every tree and every branch and compares the values, the way
pdhd/stm/perf/d30_hash_gate.py does for the PDHD/PDVD gates.

Arrays are compared as awkward -> python lists: T_proj_data is doubly jagged and numpy's
array_equal raises on it rather than returning False, which on a first pass silently reported ~150
healthy T_tagger branches as differing.  A comparator that throws is a comparator that lies.

Usage: python3 scripts/d118/root_gate.py <a.root> <b.root> [label]
       python3 scripts/d118/root_gate.py --arms <armA> <armB> [label]   (all common events)
"""
import glob, os, sys
import uproot
import awkward as ak

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _nan_equal(x, y):
    """Structural equality treating NaN as equal to NaN."""
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(_nan_equal(p, q) for p, q in zip(x, y))
    if isinstance(x, float) and isinstance(y, float):
        return x == y or (x != x and y != y)
    return x == y


def brdiff(fa, fb):
    """{tree: [differing branch ...]} over the two open files."""
    out = {}
    ta = sorted({k.split(";")[0] for k in fa.keys()})
    tb = sorted({k.split(";")[0] for k in fb.keys()})
    if ta != tb:
        return {"<objects>": [f"only A {sorted(set(ta)-set(tb))} only B {sorted(set(tb)-set(ta))}"]}
    for tn in ta:
        A, B = fa[tn], fb[tn]
        if not hasattr(A, "keys"):
            continue
        ka, kb = sorted(A.keys()), sorted(B.keys())
        if ka != kb:
            out[tn] = ["<branch set differs>"]; continue
        bad = []
        for br in ka:
            try:
                x = ak.to_list(A[br].array(library="ak"))
                y = ak.to_list(B[br].array(library="ak"))
            except Exception as e:
                bad.append(f"{br}(read: {e})"); continue
            # NaN != NaN, so a plain comparison reports a branch that is bit-identical in both
            # arms as differing.  T_rec_charge.reduced_chi2 carries NaN on rows with no fit and
            # was flagged on every arm pair, INCLUDING the same-config control, until this.
            if x != y and not _nan_equal(x, y):
                bad.append(br)
        if bad:
            out[tn] = bad
    return out


def one(a, b, label):
    with uproot.open(a) as fa, uproot.open(b) as fb:
        d = brdiff(fa, fb)
    if not d:
        print(f"  {label}: every tree identical")
    else:
        for tn, bad in sorted(d.items()):
            show = bad[:8] + ([f"... +{len(bad)-8}"] if len(bad) > 8 else [])
            print(f"  {label}: {tn} differs on {len(bad)} branch(es): {show}")
    return d


def evmap(arm):
    out = {}
    # The sub-root key must be "" for a flat arm (beam-off has no f*/): taking basename(dirname)
    # there yields the ARM name, which differs between the two arms and silently matches 0 events.
    for d in glob.glob(os.path.join(arm, "f*", "pr_evt*")):
        p = os.path.join(d, "tracking-pr.root")
        if os.path.isfile(p):
            out[(os.path.basename(os.path.dirname(d)), os.path.basename(d))] = p
    for d in glob.glob(os.path.join(arm, "pr_evt*")):
        p = os.path.join(d, "tracking-pr.root")
        if os.path.isfile(p):
            out[("", os.path.basename(d))] = p
    return out


def main(argv):
    if argv[:1] == ["--arms"]:
        A, B = evmap(os.path.join(SX, argv[1])), evmap(os.path.join(SX, argv[2]))
        label = argv[3] if len(argv) > 3 else f"{argv[1]} vs {argv[2]}"
        common = sorted(set(A) & set(B))
        print(f"## {label}: {len(common)} common events")
        tally, nbad = {}, 0
        for k in common:
            d = brdiff(*(uproot.open(A[k]), uproot.open(B[k])))
            for tn, bad in d.items():
                tally.setdefault(tn, set()).update(bad)
                nbad += 1
        if not tally:
            print(f"   every tree of every event identical")
        for tn, bad in sorted(tally.items()):
            b = sorted(bad)
            print(f"   {tn}: {len(b)} branch(es) ever differ: {b[:10]}{' ...' if len(b) > 10 else ''}")
        return 0 if not tally else 1
    return 0 if not one(argv[0], argv[1], argv[2] if len(argv) > 2 else "pair") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
