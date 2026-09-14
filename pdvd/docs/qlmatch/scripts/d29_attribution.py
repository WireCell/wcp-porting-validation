#!/usr/bin/env python3
"""doc qlmatch/29 -- where the Q/L hand-scan agreement went, step by step, with the scorer's own functions.  (read-only)

    python3 d29_attribution.py --chain p99wflip p98voffq p100flip p101q [--july tm0k] > ../d29/attribution.txt

1. Reproduction gate: each tag's agree / phantom / missed, recomputed in memory exactly as ql_agree_score.py main() does
   (truth mapped through --ref, default keep), must equal work/ql_scores/<tag>/scores.json "total".  Any mismatch aborts.
2. Full truth and COMMON truth (only the entries whose cluster maps in every tag of the chain) per tag.
3. Per consecutive step, by ORIGINAL truth key (event, truth uid, time): objective positives newly missed / recovered,
   negatives newly phantom / resolved, split by drift volume and y/z quadrant of the truth cluster in --ref.
   (The totals are the scorer's; the transition lists use this key and the scorer's covered / phantom logic.)
4. --july TAG: the July tag's scores.json (scored without a map, truth uid space) against the first chain tag's, keyed by
   (event, volume, flash time to 0.1 us) -- the uid spaces differ, so this list diff is approximate.
"""
import argparse
import collections
import json
import os

import d29_common as C


def recorded(tag):
    with open(os.path.join(C.PDVD, "work/ql_scores", tag, "scores.json")) as fh:
        return json.load(fh)


def totals(arms, T, keep_entries=None):
    tot = collections.Counter()
    for idx, A in arms.items():
        evt = C.S.evt_of_idx(idx)
        ents = T[evt] if keep_entries is None else [e for e in T[evt] if C.key(evt, e) in keep_entries]
        r = A.score(ents)
        for k in ("agree", "phantom", "missed", "unknown", "covered", "pos_short", "pos_cluster_missing"):
            tot[k] += r[k]
    return tot


def fmt(t):
    return f"{t['agree']}/{t['phantom']}/{t['missed']} (unknown {t['unknown']}, positives {t['covered'] + t['missed']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", nargs="+", required=True)
    ap.add_argument("--ref", default="keep")
    ap.add_argument("--july")
    a = ap.parse_args()
    T = C.truth()
    print(f"# doc qlmatch/29 -- Q/L hand-scan attribution (d29_attribution.py); truth uid map through '{a.ref}', no time map/shift")
    arms = {tag: {idx: C.Arm(tag, idx, ref=a.ref) for idx in C.IDX} for tag in a.chain}
    ref = {idx: C.Arm(a.ref, idx, ref=a.ref) for idx in C.IDX}

    print("\n== 1. reproduction of the recorded scores")
    full = {}
    for tag in a.chain:
        full[tag] = totals(arms[tag], T)
        rec = recorded(tag)["total"]
        ok = all(full[tag][k] == rec[k] for k in ("agree", "phantom", "missed"))
        print(f"  {tag:10s} recomputed {fmt(full[tag])}  recorded {rec['agree']}/{rec['phantom']}/{rec['missed']}  "
              f"{'REPRODUCED' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit("reproduction failed: stop")

    common = set()
    for idx in C.IDX:
        evt = C.S.evt_of_idx(idx)
        for e in T[evt]:
            if all(e["uid"] in arms[tag][idx].map for tag in a.chain):
                common.add(C.key(evt, e))
    ntruth = sum(len(T[C.S.evt_of_idx(i)]) for i in C.IDX)
    print(f"\n== 2. full truth ({ntruth} entries) and common truth ({len(common)} entries mapping in every tag)")
    for tag in a.chain:
        ct = totals(arms[tag], T, common)
        print(f"  {tag:10s} full {fmt(full[tag])}   common {fmt(ct)}")

    print("\n== 3. per step, by original truth key (objective tiers)")
    for A, B in zip(a.chain, a.chain[1:]):
        trans = collections.defaultdict(collections.Counter)
        lists = collections.defaultdict(list)
        for idx in C.IDX:
            evt = C.S.evt_of_idx(idx)
            for e in T[evt]:
                if e["conf"] not in C.S.OBJECTIVE_TIERS:
                    continue
                rc = ref[idx].clusters.get(e["uid"])
                reg = C.region(rc) if rc is not None and rc.get("y") else "truth cluster not in ref"
                if e["positive"]:
                    sa, _ = arms[A][idx].positive_status(e)
                    sb, _ = arms[B][idx].positive_status(e)
                    kind = ("newly missed" if (sa, sb) == ("covered", "missed") else
                            "recovered" if (sa, sb) == ("missed", "covered") else
                            None if sa == sb else f"positive {sa}->{sb}")
                else:
                    sa, _ = arms[A][idx].negative_status(e)
                    sb, _ = arms[B][idx].negative_status(e)
                    kind = ("new phantom" if (sa, sb) == ("rejected", "phantom") else
                            "phantom resolved" if (sa, sb) == ("phantom", "rejected") else
                            None if sa == sb else f"negative {sa}->{sb}")
                if kind:
                    trans[kind][reg] += 1
                    lists[kind].append((evt, e["uid"], round(e["time"], 1), reg))
        print(f"  -- {A} -> {B}")
        for kind in sorted(trans):
            c = trans[kind]
            vol = collections.Counter()
            for r, n in c.items():
                vol[r.split(" ")[0]] += n
            print(f"     {kind:22s} {sum(c.values()):3d}  by volume {dict(vol)}  by region {dict(sorted(c.items()))}")
        for kind in ("newly missed", "recovered"):
            print(f"     {kind} list: {sorted(lists[kind])}")

    if a.july:
        print(f"\n== 4. {a.july} (July, no map) -> {a.chain[0]}: list diff keyed by (event, volume, time 0.1 us); approximate")
        J, P = recorded(a.july), recorded(a.chain[0])
        print(f"  recorded totals: {a.july} {J['total']['agree']}/{J['total']['phantom']}/{J['total']['missed']}  "
              f"{a.chain[0]} {P['total']['agree']}/{P['total']['phantom']}/{P['total']['missed']}")
        for lst in ("missed_list", "phantom_list"):
            def keys(rec):
                return {(ev, e["uid"] // C.S.GID_STRIDE, round(e["time"], 1)) for ev, v in rec["detail"].items() for e in v[lst]}
            kj, kp = keys(J), keys(P)
            new, gone = kp - kj, kj - kp
            print(f"  {lst}: new in {a.chain[0]} {len(new)} (top {sum(1 for k in new if k[1] == 4)}, bottom {sum(1 for k in new if k[1] == 0)}); "
                  f"gone {len(gone)} (top {sum(1 for k in gone if k[1] == 4)}, bottom {sum(1 for k in gone if k[1] == 0)}); common {len(kj & kp)}")


if __name__ == "__main__":
    main()
