#!/usr/bin/env python3
"""doc 87: the acceptance test for save_in_scope.

The in-scope set is what the PR chain's taggers evaluate: switch_scope stamps
Cluster::set_scope_filter (clustering_switch_scope.cxx:125) and every tagger's
require_in_scope consults get_scope_filter().  Until doc 87 nothing SERIALIZED
it -- mabc-pr.zip recorded it only implicitly, by containing exactly the
in-scope clusters (MultiAlgBlobClustering.cxx:2906-2923 gates the Bee
clustering layer on literally that call), which is why nusel_extract.py's
parse_prbee() was the only reader that had it.

save_in_scope adds T_cluster to tracking-pr.root.  This gate asserts the claim
that matters and cannot pass by accident:

    set(T_cluster.cluster_id where in_scope==1)  ==  parse_prbee(mabc-pr.zip)

exactly, on every event.  A set that merely RESEMBLES the Bee one is a failure:
the whole point is that downstream can stop reading the Bee zip.

Usage:
    scripts/pr87_inscope_gate.py <arm> [<arm> ...]
Exit 0 = every event matches; 1 = any mismatch (the first few are named).
"""
import glob, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nusel_extract as ne  # noqa: E402  (parse_prbee, unmodified)


def check(arm):
    ok = bad = skip = 0
    fails = []
    for d in sorted(glob.glob(os.path.join(arm, "pr_evt*"))):
        evt = os.path.basename(d)[6:]
        bee = os.path.join(d, "mabc-pr.zip")
        rt = os.path.join(d, "tracking-pr.root")
        if not (os.path.exists(bee) and os.path.exists(rt)):
            skip += 1
            continue
        import uproot
        f = uproot.open(rt)
        if "T_cluster" not in {k.split(";")[0] for k in f.keys()}:
            bad += 1
            fails.append((evt, "no T_cluster -- was save_in_scope on?"))
            continue
        t = f["T_cluster"]
        cid = t["cluster_id"].array(library="np")
        ins = t["in_scope"].array(library="np")
        got = {int(c) for c, i in zip(cid, ins) if i == 1}
        want = ne.parse_prbee(bee)
        if got == want:
            ok += 1
        else:
            bad += 1
            fails.append((evt, "T_cluster %d vs bee %d; extra=%s missing=%s"
                          % (len(got), len(want),
                             sorted(got - want)[:6], sorted(want - got)[:6])))
    return ok, bad, skip, fails


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tot_ok = tot_bad = tot_skip = 0
    for arm in sys.argv[1:]:
        ok, bad, skip, fails = check(arm)
        tot_ok += ok; tot_bad += bad; tot_skip += skip
        print("%-32s match %4d  MISMATCH %4d  skipped %3d" % (arm, ok, bad, skip))
        for evt, why in fails[:8]:
            print("    evt %s: %s" % (evt, why))
    print("\n# events matching the Bee in-scope set exactly: %d  mismatching: %d"
          "  skipped(no bee/root): %d" % (tot_ok, tot_bad, tot_skip))
    print("PASS" if tot_bad == 0 else "FAIL")
    return 1 if tot_bad else 0


if __name__ == "__main__":
    sys.exit(main())
