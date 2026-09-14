#!/usr/bin/env python3
"""doc qlmatch/29 -- apply the pre-registered adoption rule (d29/prereg.md sec 3) to scored arms.  (read-only)

    python3 d29_lever_eval.py --base p100flip --control q29base --arms q29l1 ... q29v7nw p101q \
        --cfg-dir /home/xqian/tmp/p29/cfg > ../d29/lever_eval.txt

Per arm, with the scorer's own functions (truth mapped through keep, no time map / shift), against --base:
  full truth, even-idx half, odd-idx half, and common truth (entries mapping in base, control and every arm):
  agree / phantom / missed and the change.  Rule: at least one metric improves and none regresses by more than 2 pairs,
  on full truth AND on each half AND on common truth.  Score (agree - phantom - missed change) ranks passing arms.
--control: must reproduce --base exactly (calib dumps byte-identical on 18/18 and identical scores) or everything aborts.
--cfg-dir: compile-only configs <arm>.json; each arm's differing leaves against <control>.json are printed.
"""
import argparse
import collections
import filecmp
import json
import os

import d29_common as C

MET = ("agree", "phantom", "missed")


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, o


def score(arms, T, idxs, keys=None):
    tot = collections.Counter()
    for idx in idxs:
        evt = C.S.evt_of_idx(idx)
        ents = T[evt] if keys is None else [e for e in T[evt] if C.key(evt, e) in keys]
        r = arms[idx].score(ents)
        for k in MET:
            tot[k] += r[k]
    return tot


def verdict(base, arm, allow=2):
    d = {k: arm[k] - base[k] for k in MET}
    better = d["agree"] > 0 or d["phantom"] < 0 or d["missed"] < 0
    worse = d["agree"] < -allow or d["phantom"] > allow or d["missed"] > allow
    return d, better and not worse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--cfg-dir")
    a = ap.parse_args()
    T = C.truth()
    tags = [a.base, a.control] + a.arms
    arms = {t: {i: C.Arm(t, i) for i in C.IDX} for t in tags}
    print(f"# doc qlmatch/29 -- lever evaluation (d29_lever_eval.py) against {a.base}; rule: d29/prereg.md sec 3")

    same = sum(filecmp.cmp(C.calib_path(a.base, i), C.calib_path(a.control, i), shallow=False) for i in C.IDX)
    print(f"control {a.control} calib dumps byte-identical to {a.base}: {same}/18")
    if same != 18:
        raise SystemExit("control does not reproduce the base: stop")

    common = set()
    for i in C.IDX:
        evt = C.S.evt_of_idx(i)
        for e in T[evt]:
            if all(e["uid"] in arms[t][i].map for t in tags):
                common.add(C.key(evt, e))
    halves = {"full": list(C.IDX), "even": [i for i in C.IDX if i % 2 == 0], "odd": [i for i in C.IDX if i % 2 == 1]}
    base = {h: score(arms[a.base], T, ix) for h, ix in halves.items()}
    base["common"] = score(arms[a.base], T, C.IDX, common)
    print(f"base {a.base}: " + "  ".join(f"{h} {base[h]['agree']}/{base[h]['phantom']}/{base[h]['missed']}" for h in base)
          + f"   (common truth {len(common)} entries)\n")

    cfg_base = dict(leaves(json.load(open(os.path.join(a.cfg_dir, f"{a.control}.json"))))) if a.cfg_dir else None
    ranked = []
    for t in a.arms:
        res = {h: score(arms[t], T, ix) for h, ix in halves.items()}
        res["common"] = score(arms[t], T, C.IDX, common)
        parts, ok_all = [], True
        for h in ("full", "even", "odd", "common"):
            d, ok = verdict(base[h], res[h])
            ok_all &= ok
            parts.append(f"{h} {res[h]['agree']}/{res[h]['phantom']}/{res[h]['missed']} "
                         f"({d['agree']:+d}/{d['phantom']:+d}/{d['missed']:+d}) {'ok' if ok else 'x'}")
        dfull = verdict(base["full"], res["full"])[0]
        merit = dfull["agree"] - dfull["phantom"] - dfull["missed"]
        print(f"{t:9s} {'PASS' if ok_all else 'fail'} merit {merit:+d} | " + " | ".join(parts))
        if cfg_base is not None:
            p = os.path.join(a.cfg_dir, f"{t}.json")
            if os.path.exists(p):
                L = dict(leaves(json.load(open(p))))
                diff = [(k, cfg_base.get(k), L.get(k)) for k in sorted(set(L) | set(cfg_base)) if L.get(k) != cfg_base.get(k)]
                print(f"          config leaves differing from {a.control}: {len(diff)} {diff[:8]}")
            else:
                print("          (no compile-only config recorded)")
        if ok_all:
            ranked.append((merit, t))
    print("\npassing arms by merit:", sorted(ranked, reverse=True) or "none")


if __name__ == "__main__":
    main()
