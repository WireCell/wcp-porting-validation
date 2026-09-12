#!/usr/bin/env python3
"""doc pdhd/25 sec 5 -- grade an arm's pre-registered twin ITEM BY ITEM, and name every candidate whose
non-verdict branches moved.

    python3 d25_twin_check.py --arm h25k --base h25base --ks -0.10 --ke 5 --len 1.5
    python3 d25_twin_check.py --arm h25r --base h25base

1. The twin: d25_misses.reverdict on the BASE arm's payload with the given thresholds (omitted ones stay
   production's), per population (strict / majority / all, d25_bragg_michel's), against the arm's own
   is_stm on the same items.  Printed: predicted-and-happened, predicted-not-happened, happened-not-predicted.
   With no thresholds given the twin is "no verdict change" -- use it for h25r, whose key moves the KS
   INPUTS, which the offline model cannot see; the arm's movers are then listed for the named prediction.
2. Every candidate (scored or not) whose value moved on any branch OTHER than the verdict bits
   (is_stm, reject_bits, topology_cleared_bits), with the branches named and whether its is_stm flipped.
3. Per candidate, each pre-P1 reject bit (reject_bits | topology_cleared_bits) before -> after, counted by
   bit name, with the no_bragg movers named -- h25r's pre-registered falsifier 5.
"""
import argparse, collections, glob, os, sys
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d25_bragg_michel as Q
import d25_misses as M

VERDICT = {"is_stm", "reject_bits", "topology_cleared_bits"}


def all_branches(arm):
    out = {}
    for f in sorted(glob.glob(f"{Q.IMG}/pdhd/work/*_{arm}/tracking-pr.root")):
        evt = os.path.basename(os.path.dirname(f)).replace("_" + arm, "")
        u = uproot.open(f)
        if "T_stm_michel" not in [k.split(";")[0] for k in u.keys()]:
            continue
        t = u["T_stm_michel"]
        s = t.arrays(library="np")
        for i in range(len(s["cluster_id"])):
            out[f"{evt}/{int(s['cluster_id'][i])}"] = {k: s[k][i] for k in s}
    return out


def same(a, b):
    try:
        if isinstance(a, float) or isinstance(b, float) or np.issubdtype(np.asarray(a).dtype, np.floating):
            return (np.isnan(a) and np.isnan(b)) or a == b
    except TypeError:
        pass
    return np.array_equal(np.asarray(a), np.asarray(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base", default="h25base")
    ap.add_argument("--ks", type=float, default=None)
    ap.add_argument("--ke", type=float, default=None)
    ap.add_argument("--len", type=float, default=None)
    ap.add_argument("--named", default=None,
                    help="comma list: the pre-registered gains BY NAME (for a key that moves the KS inputs, which "
                         "the offline model cannot see); replaces the model's gains on every population")
    a = ap.parse_args()
    named = set(a.named.split(",")) if a.named else None
    P0 = M.prod_params("pdhd")
    P = dict(P0)
    if a.ks is not None: P["ks_margin"] = a.ks
    if a.ke is not None: P["ke_min"] = a.ke
    if a.len is not None: P["len_min"] = a.len
    print(f"twin thresholds {P}\n")

    print("==== 1. the twin, item by item (scored population, truth = the record in use) ====")
    for pop in ("strict", "majority", "all"):
        base, _ = Q.items("pdhd", a.base, pop)
        arm = {x[0]: x for x in Q.items("pdhd", a.arm, pop)[0]}
        pred, happ, lost_p, lost_h = set(), set(), set(), set()
        for k, v, kind, src, d in base:
            was = int(d["is_stm"]) == 1
            now_pred = (was or k in named) if named is not None else M.reverdict(d, P)[1] == 0
            if k not in arm:
                print(f"  {pop}: {k} has no candidate in {a.arm}")
                continue
            now = int(arm[k][4]["is_stm"]) == 1
            if now_pred and not was: pred.add(k)
            if now and not was: happ.add(k)
            if was and not now_pred: lost_p.add(k)
            if was and not now: lost_h.add(k)
        hs = {x[0]: x[1] for x in base}
        tag = lambda ks: " ".join(f"{k}({'stop' if hs[k] in Q.STOP else hs[k]})" for k in sorted(ks)) or "-"
        print(f"  --- {pop}: predicted gains {len(pred)}, arm gains {len(happ)}; predicted losses {len(lost_p)}, arm losses {len(lost_h)}")
        print(f"      predicted and happened      {len(pred & happ)}")
        print(f"      predicted, did NOT happen   {tag(pred - happ)}")
        print(f"      happened, NOT predicted     {tag(happ - pred)}")
        print(f"      lost: predicted {tag(lost_p)} | arm {tag(lost_h)}")
        print(f"      TWIN {'HELD' if pred == happ and lost_p == lost_h else 'FAILED'} on {pop}")

    print("\n==== 2. candidates whose NON-verdict branches moved (every candidate, scored or not) ====")
    B0, B1 = all_branches(a.base), all_branches(a.arm)
    print(f"  candidates {len(B0)} -> {len(B1)}; only in base {sorted(set(B0)-set(B1))}; only in arm {sorted(set(B1)-set(B0))}")
    moved = collections.Counter()
    for k in sorted(set(B0) & set(B1)):
        br = sorted(b for b in B0[k] if b in B1[k] and b not in VERDICT and not same(B0[k][b], B1[k][b]))
        for b in br: moved[b] += 1
        if br and not (set(br) <= {"ks_mu", "ks_flat", "ratio_mu", "ratio_flat", "n_cmp_live", "dead_frac_cmp"}
                       or all(b.startswith(("comp_fwd", "comp_bwd", "ks_", "ratio_", "n_cmp_live", "dead_frac_cmp")) for b in br)):
            flip = f"is_stm {int(B0[k]['is_stm'])}->{int(B1[k]['is_stm'])}"
            print(f"  {k:15s} {flip}  moved: {' '.join(br)}")
    print("  branch move counts: " + ", ".join(f"{b} {n}" for b, n in moved.most_common()))

    print("\n==== 3. pre-P1 reject bits, per candidate, before -> after ====")
    set_c, clr_c, nb = collections.Counter(), collections.Counter(), []
    for k in sorted(set(B0) & set(B1)):
        p0 = int(B0[k]["reject_bits"]) | int(B0[k]["topology_cleared_bits"])
        p1 = int(B1[k]["reject_bits"]) | int(B1[k]["topology_cleared_bits"])
        for i, n in enumerate(M.BITS):
            b0, b1 = p0 >> i & 1, p1 >> i & 1
            if b1 and not b0: set_c[n] += 1
            if b0 and not b1: clr_c[n] += 1
            if n == "no_bragg" and b0 != b1: nb.append(f"{k}({b0}->{b1})")
    print(f"  bits newly SET:     {dict(set_c) or '-'}")
    print(f"  bits newly CLEARED: {dict(clr_c) or '-'}")
    print(f"  no_bragg movers ({len(nb)}): {' '.join(nb) or '-'}")


if __name__ == "__main__":
    main()
