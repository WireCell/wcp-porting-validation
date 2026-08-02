#!/usr/bin/env python3
"""valfast helper: member-content hash compare of the nusel-side archives of
two -full valfast nusel roots, per event.

usage: nusel_hash_compare.py <rootA> <rootB> <events.txt>

Compares, per event: ql_evt<ID>/{mabc-all-apa.zip, pctree-evt<ID>.tar.gz} and
nusel_evt<ID>/mabc-pr.zip (the d60_ab_report.py archive scope). Hashing is
member-name+payload (abtest/hash_archive.py `members`), never raw archive
bytes (M2). tracking-stm.root is deliberately NOT hashed (ROOT embeds
timestamps); score-level physics is gated by pr20_scores_diff.py instead.
Exit 0 iff every archive pair is content-identical.
"""
import sys, os, importlib.util

AB = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest"
spec = importlib.util.spec_from_file_location("hash_archive", os.path.join(AB, "hash_archive.py"))
ha = importlib.util.module_from_spec(spec); spec.loader.exec_module(ha)

def roll(path):
    """Rollup sha256 over sorted (member-name, payload) pairs, as hash_archive does."""
    if not os.path.exists(path): return "MISSING"
    import hashlib
    h = hashlib.sha256()
    try:
        for name, payload in ha.members(path):
            h.update(name.encode()); h.update(payload)
    except Exception as e:
        return f"ERROR:{e}"
    return h.hexdigest()

def main():
    rootA, rootB, evfile = sys.argv[1], sys.argv[2], sys.argv[3]
    events = [l.strip() for l in open(evfile) if l.strip()]
    n_same = n_diff = 0
    for ev in events:
        rels = [f"ql_evt{ev}/mabc-all-apa.zip",
                f"ql_evt{ev}/pctree-evt{ev}.tar.gz",
                f"nusel_evt{ev}/mabc-pr.zip"]
        for rel in rels:
            a, b = roll(os.path.join(rootA, rel)), roll(os.path.join(rootB, rel))
            if a == b: n_same += 1
            else:
                n_diff += 1
                print(f"DIFF {rel}")
    print(f"NUSEL-ARCHIVES {os.path.basename(rootA)} vs {os.path.basename(rootB)}: "
          f"identical {n_same}/{n_same+n_diff}")
    return 1 if n_diff else 0

if __name__ == "__main__":
    sys.exit(main())
