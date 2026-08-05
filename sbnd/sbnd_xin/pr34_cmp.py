#!/usr/bin/env python3
"""doc pr/34 §11 -- compare two PR-chain arms on the PARTICLE-FLOW artifact.

Usage: pr34_cmp.py <base_arm_dir> <test_arm_dir>

The pr/34 knobs are display-only: they move mabc-pr.zip::data/0/0-mc.json and
nothing else.  pr32_cmp.py's pctree gate is VACUOUS for them (the PF tree is
not in pctree-pr), so this driver reports, in order of how much it matters:

  1. mabc-pr.zip::data/0/0-mc.json member-content hashes -- THE pr/34 gate
  2. the other mabc-pr.zip members -- must never move (any diff = escalate)
  3. pctree-pr member-content hashes -- must never move (the display-only proof:
     an ON arm that moves 1 while 2+3 stay identical measures the claim)
  4. nusel-table.tsv / nusel-events.tsv -- the selection outcome, ditto

Member hashing matches abtest/hash_archive.py: sha256(name + payload) per
member, compared by name -- never cmp on the zip/tarball (M2, timestamps).
"""
import sys, os, glob, re, hashlib, zipfile, tarfile

MC = "data/0/0-mc.json"


def evts(d):
    return sorted(int(re.search(r"pr_evt(\d+)", p).group(1))
                  for p in glob.glob(os.path.join(d, "pr_evt*")))


def zip_members(path):
    """{member_name: sha256(name+payload)} for a zip archive."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            if n.endswith("/"):
                continue
            out[n] = hashlib.sha256(n.encode() + z.read(n)).hexdigest()
    return out


def tar_rollup(path):
    """Single rollup hash over member-content hashes (hash_archive.py shape)."""
    h = hashlib.sha256()
    with tarfile.open(path) as t:
        for m in sorted(t.getmembers(), key=lambda m: m.name):
            if not m.isfile():
                continue
            payload = t.extractfile(m).read()
            h.update(hashlib.sha256(m.name.encode() + payload).hexdigest().encode())
    return h.hexdigest()


def main():
    base, test = sys.argv[1], sys.argv[2]
    ev = evts(base)
    assert ev == evts(test), "arms cover different events"
    print(f"# base={base}\n# test={test}\n# {len(ev)} events")

    mc_same = mc_diff = other_diff = missing = 0
    mc_difflist, other_difflist = [], []
    for e in ev:
        b = os.path.join(base, f"pr_evt{e}", "mabc-pr.zip")
        t = os.path.join(test, f"pr_evt{e}", "mabc-pr.zip")
        if not (os.path.exists(b) and os.path.exists(t)):
            missing += 1
            mc_difflist.append((e, "MISSING"))
            continue
        hb, ht = zip_members(b), zip_members(t)
        if hb.get(MC) == ht.get(MC):
            mc_same += 1
        else:
            mc_diff += 1
            mc_difflist.append((e, "differs"))
        others = sorted((set(hb) | set(ht)) - {MC})
        moved = [n for n in others if hb.get(n) != ht.get(n)]
        if moved:
            other_diff += 1
            other_difflist.append((e, ",".join(os.path.basename(n) for n in moved)))
    print(f"mabc mc.json member hashes: {mc_same}/{len(ev)} identical, "
          f"{mc_diff} differ, {missing} missing")
    if mc_difflist:
        print("  " + " ".join(f"{e}:{w}" for e, w in mc_difflist))
    print(f"mabc other members: {len(ev) - other_diff - missing}/{len(ev)} identical, "
          f"{other_diff} events differ" + ("  <-- ESCALATE, not display-only" if other_diff else ""))
    if other_difflist:
        print("  " + " ".join(f"{e}:{w}" for e, w in other_difflist))

    # pctree-pr: must be untouched by every pr/34 knob (display-only proof).
    # Processes, not threads: gzip+sha256 in pure Python holds the GIL, so a
    # thread pool runs the 96 archives serially (~1.4 s each).
    from concurrent.futures import ProcessPoolExecutor as ThreadPoolExecutor
    jobs = []
    pc_missing = []
    for e in ev:
        b = os.path.join(base, f"pr_evt{e}", f"pctree-pr-evt{e}.tar.gz")
        t = os.path.join(test, f"pr_evt{e}", f"pctree-pr-evt{e}.tar.gz")
        if not (os.path.exists(b) and os.path.exists(t)):
            pc_missing.append(e)
            continue
        jobs.append((e, b, t))
    with ThreadPoolExecutor(max_workers=int(os.environ.get("PR34_JOBS", "24"))) as ex:
        hashes = list(ex.map(tar_rollup, [p for _, b, t in jobs for p in (b, t)]))
    pc_diff = [(e, "differs") for i, (e, _, _) in enumerate(jobs)
               if hashes[2 * i] != hashes[2 * i + 1]]
    print(f"pctree-pr member-content hashes: {len(jobs) - len(pc_diff)}/{len(ev)} identical, "
          f"{len(pc_diff)} differ, {len(pc_missing)} missing"
          + ("  <-- ESCALATE, not display-only" if pc_diff else ""))
    if pc_diff:
        print("  " + " ".join(f"{e}:{w}" for e, w in pc_diff))

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


if __name__ == "__main__":
    main()
