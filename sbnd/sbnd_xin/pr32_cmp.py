#!/usr/bin/env python3
"""doc pr/32 §11 -- compare two PR-chain arms over the nueCC48 manifest.

Usage: pr32_cmp.py <base_arm_dir> <test_arm_dir>

Reports, in order of how much it matters:
  1. pctree-pr member-content hashes (hash_archive.py) -- the byte-identical gate
  2. nusel-table.tsv / nusel-events.tsv -- the selection outcome
  3. tracking-pr.root T_tagger + T_kine leaves -- the physics scores
  4. PR32AUDIT counters from the per-event logs
"""
import sys, os, subprocess, glob, re, hashlib

AB = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest"
sys.path.insert(0, AB)

# Which counter is the population behind each per-event mean, so the aggregate
# is a weighted mean rather than a sum of means.
MEAN_WEIGHT = {
    "f1_mean_cm": "f1_fit_valid",
    "pca_mean_cm": "pca_moved",
}


def evts(d):
    return sorted(int(re.search(r"pr_evt(\d+)", p).group(1))
                  for p in glob.glob(os.path.join(d, "pr_evt*")))


def arch_hash(path):
    """Member-content hash via abtest/hash_archive.py (never md5 the tarball, M2).

    hash_archive.py prints `<sha256>  <nmembers>  <path>` -- the PATH IS IN THE
    OUTPUT, so hashing its stdout compares directory names and every arm
    "differs".  Take the first two fields only.
    """
    r = subprocess.run(["python3", os.path.join(AB, "hash_archive.py"), path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return "ERROR:" + r.stderr.strip()[:80]
    out = r.stdout.strip().split()
    if len(out) < 2:
        return "ERROR:unparsed:" + r.stdout.strip()[:60]
    return out[0] + ":" + out[1]


def audit(logpath, tag):
    if not os.path.exists(logpath):
        return {}
    last = None
    for line in open(logpath, errors="replace"):
        if tag in line:
            last = line
    if not last:
        return {}
    out = {}
    for m in re.finditer(r"(\w+)=([-\d.]+|true|false)", last):
        out[m.group(1)] = m.group(2)
    return out


def main():
    base, test = sys.argv[1], sys.argv[2]
    ev = evts(base)
    assert ev == evts(test), "arms cover different events"
    print(f"# base={base}\n# test={test}\n# {len(ev)} events")

    same = diff = missing = 0
    difflist = []
    jobs, skipped = [], []
    for e in ev:
        b = os.path.join(base, f"pr_evt{e}", f"pctree-pr-evt{e}.tar.gz")
        t = os.path.join(test, f"pr_evt{e}", f"pctree-pr-evt{e}.tar.gz")
        if not (os.path.exists(b) and os.path.exists(t)):
            skipped.append(e)
            continue
        jobs.append((e, b, t))
    missing = len(skipped)
    difflist += [(e, "MISSING") for e in skipped]
    # 96 archives per comparison at ~1 s each: hash them in parallel.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=int(os.environ.get("PR32_JOBS", "24"))) as ex:
        paths = [p for _, b, t in jobs for p in (b, t)]
        hashes = list(ex.map(arch_hash, paths))
    for i, (e, _, _) in enumerate(jobs):
        hb, ht = hashes[2 * i], hashes[2 * i + 1]
        if hb == ht:
            same += 1
        else:
            diff += 1
            difflist.append((e, "differs"))
    print(f"pctree-pr member-content hashes: {same}/{len(ev)} identical, "
          f"{diff} differ, {missing} missing")
    if difflist:
        print("  " + " ".join(f"{e}:{w}" for e, w in difflist))

    for name in ("nusel-table.tsv", "nusel-events.tsv"):
        b, t = os.path.join(base, name), os.path.join(test, name)
        if not (os.path.exists(b) and os.path.exists(t)):
            print(f"{name}: MISSING on one side")
            continue
        bl, tl = open(b).read().split("\n"), open(t).read().split("\n")
        nd = sum(1 for x, y in zip(bl, tl) if x != y)
        print(f"{name}: {'identical' if nd == 0 and len(bl) == len(tl) else str(nd) + ' line(s) differ'}")
        if nd:
            for x, y in zip(bl, tl):
                if x != y:
                    print("   -", x[:150])
                    print("   +", y[:150])

    # PR32AUDIT / PR30AUDIT counters, summed over events
    for tag in ("PR32AUDIT", "PR30AUDIT"):
        acc = {}
        for arm, label in ((base, "base"), (test, "test")):
            tot = {}
            # Counters SUM across events; *_mean_* must be re-weighted by its
            # population and *_max_* must be maxed.  Summing a mean gives
            # 48x the answer -- which is how "mean fit-to-wcpt gap = 28.9 cm"
            # was reported for what is really 0.61 cm.
            wsum = {}
            for e in ev:
                a = audit(os.path.join(arm, f"pr_evt{e}", f"wct_pr_evt{e}.log"), tag)
                for k, v in a.items():
                    if v in ("true", "false"):
                        tot.setdefault(k, set()).add(v)
                        continue
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if "_max_" in k or k.endswith("_max"):
                        tot[k] = max(tot.get(k, 0.0), fv)
                    elif "_mean_" in k or k.endswith("_mean"):
                        w = float(a.get(MEAN_WEIGHT.get(k, ""), 0) or 0)
                        wsum[k] = wsum.get(k, 0.0) + fv * w
                        tot.setdefault(k + "__w", 0.0)
                        tot[k + "__w"] += w
                    else:
                        tot[k] = tot.get(k, 0) + fv
            for k, s in wsum.items():
                w = tot.pop(k + "__w", 0.0)
                tot[k] = s / w if w else 0.0
            acc[label] = tot
        if not acc["base"]:
            continue
        print(f"{tag} (summed over {len(ev)} events)")
        for k in acc["base"]:
            bv, tv = acc["base"].get(k), acc["test"].get(k)
            mark = "" if bv == tv else "   <-- differs"
            fmt = lambda v: (",".join(sorted(v)) if isinstance(v, set)
                             else (f"{v:.0f}" if float(v).is_integer() else f"{v:.3f}"))
            print(f"   {k:24s} base={fmt(bv):>12s}  test={fmt(tv):>12s}{mark}")


if __name__ == "__main__":
    main()
