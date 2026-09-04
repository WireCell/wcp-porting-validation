#!/usr/bin/env python3
"""Per-cluster tagger-verdict census across PR arms that differ only in the
tagger fiducial margins (doc pdvd/35).

  usage: fv_margin_census.py <base-tag> <tag> [<tag> ...]
         fv_margin_census.py d34base d34x d34yz d34all d34alli3

pr_arm_census_diff.py answers "did the totals move".  This answers "which
clusters moved, and which axis moved them" -- the x margin (30 -> 2.5) and the
y/z margins (3/5/3 -> 17.5/18/18) push the containment population in OPPOSITE
directions, so a net total can be near zero while both are large.  Every number
here is per distinct (run, event, cluster).

Verdicts come from the PR log's own lines, so they mean exactly what the tagger
wrote:
    TaggerCheckTGM: cluster N -> TGM=(true|false)
    TaggerCheckSTM: cluster N -> STM=k TGM=m
    check_stm_conditions: cluster N no STM fit: <reason>
"""
import glob, os, re, sys
from collections import Counter, OrderedDict

PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

RE_TGM = re.compile(r"TaggerCheckTGM: cluster (\d+) . TGM=(\w+)")
RE_STM = re.compile(r"TaggerCheckSTM: cluster (\d+) . STM=(\d+)")
RE_NOF = re.compile(r"check_stm_conditions: cluster (\d+) no STM fit: (.+?)\s*$")


def arm(tag):
    """{(run, idx, cluster): {'tgm':bool,'stm':int,'nofit':str}}"""
    out = {}
    dirs = sorted(glob.glob(os.path.join(PDVD, "work", f"*_{tag}")))
    nev = 0
    for d in dirs:
        logs = glob.glob(os.path.join(d, "wct_pr_*.log"))
        if not logs:
            continue
        nev += 1
        run, idx = os.path.basename(d).split("_")[:2]
        with open(logs[0], errors="ignore") as fh:
            for line in fh:
                m = RE_TGM.search(line)
                if m:
                    out.setdefault((run, idx, int(m.group(1))), {})["tgm"] = m.group(2) == "true"
                    continue
                m = RE_STM.search(line)
                if m:
                    out.setdefault((run, idx, int(m.group(1))), {})["stm"] = int(m.group(2))
                    continue
                m = RE_NOF.search(line)
                if m:
                    # A cluster gets SEVERAL "no STM fit:" lines -- check_stm_conditions
                    # emits one per exit test, so "fully contained (Mid Point A)" and
                    # "evaluated but no pass recorded" are both written for the same
                    # cluster (doc 33 sec 8 correction 3).  Keeping only the last one
                    # silently drops the containment population, so collect the set.
                    out.setdefault((run, idx, int(m.group(1))), {}) \
                       .setdefault("nofit", set()).add(m.group(2))
    return out, nev, len(dirs)


def totals(a):
    return OrderedDict(
        clusters=len(a),
        tgm=sum(1 for v in a.values() if v.get("tgm")),
        stm1=sum(1 for v in a.values() if v.get("stm") == 1),
        stm_eval=sum(1 for v in a.values() if "stm" in v),
        nofit=sum(1 for v in a.values() if "nofit" in v),
        contained=sum(1 for v in a.values()
                      if any("fully contained" in r for r in v.get("nofit", ()))),
    )


def main():
    tags = sys.argv[1:]
    if len(tags) < 2:
        raise SystemExit(__doc__)
    arms = OrderedDict()
    for t in tags:
        a, nev, nd = arm(t)
        arms[t] = a
        print(f"{t:<10} {nev:3d}/{nd:3d} event logs, {len(a):6,} distinct clusters")
    base_tag = tags[0]
    base = arms[base_tag]

    print("\n== totals ==")
    keys = list(totals(base).keys())
    print(f"{'tag':<10}" + "".join(f"{k:>11}" for k in keys))
    for t, a in arms.items():
        tt = totals(a)
        print(f"{t:<10}" + "".join(f"{tt[k]:>11,}" for k in keys))
    print(f"{'delta':<10}" + "".join(" " * 11 for _ in keys))
    b = totals(base)
    for t, a in arms.items():
        if t == base_tag:
            continue
        tt = totals(a)
        print(f"{t:<10}" + "".join(f"{tt[k]-b[k]:>+11,}" for k in keys))

    print(f"\n== per-cluster flips vs {base_tag} (clusters present in both) ==")
    for t, a in arms.items():
        if t == base_tag:
            continue
        common = set(base) & set(a)
        tgm_on = sum(1 for k in common if a[k].get("tgm") and not base[k].get("tgm"))
        tgm_off = sum(1 for k in common if base[k].get("tgm") and not a[k].get("tgm"))
        stm_on = sum(1 for k in common if a[k].get("stm") == 1 and base[k].get("stm") != 1)
        stm_off = sum(1 for k in common if base[k].get("stm") == 1 and a[k].get("stm") != 1)
        def cont(v):
            return any("fully contained" in r for r in v.get("nofit", ()))
        con_on = sum(1 for k in common if cont(a[k]) and not cont(base[k]))
        con_off = sum(1 for k in common if cont(base[k]) and not cont(a[k]))
        print(f"  {t:<10} common {len(common):6,}  "
              f"TGM +{tgm_on:4d}/-{tgm_off:4d}   STM +{stm_on:4d}/-{stm_off:4d}   "
              f"contained +{con_on:4d}/-{con_off:4d}")
        if len(common) != len(base) or len(common) != len(a):
            print(f"      (base-only {len(set(base)-set(a)):,}, {t}-only {len(set(a)-set(base)):,}"
                  "  -- cluster ids should be identical; a non-zero count means the"
                  " PR stage saw different input, not a margin effect)")

    print(f"\n== no-fit reasons (clusters carrying each; a cluster can carry several) ==")
    import re as _re
    def norm(r):
        return _re.sub(r"[-+]?\d+(\.\d+)?", "N", r)
    reasons = sorted({norm(r) for a in arms.values() for v in a.values()
                      for r in v.get("nofit", ())})
    print(f"{'reason':<52}" + "".join(f"{t:>10}" for t in tags))
    for r in reasons:
        row = [sum(1 for v in a.values() if any(norm(x) == r for x in v.get("nofit", ())))
               for a in arms.values()]
        print(f"{r[:52]:<52}" + "".join(f"{c:>10,}" for c in row))

    print(f"\n== biggest movers vs {base_tag} (events by |TGM delta|), for hand-scan ==")
    for t, a in arms.items():
        if t == base_tag:
            continue
        per = Counter()
        for k in set(base) & set(a):
            if bool(a[k].get("tgm")) != bool(base[k].get("tgm")):
                per[(k[0], k[1])] += 1
        # most_common ties break on insertion order, which comes from a set
        # iteration -- sort explicitly or the same arms print a different list.
        ranked = sorted(per.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
        top = ", ".join(f"{r}/{i}:{n}" for (r, i), n in ranked)
        print(f"  {t:<10} {top}")


if __name__ == "__main__":
    main()
