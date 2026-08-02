#!/usr/bin/env python3
"""doc pr/20 Part II census: beam-verdict + nu-npts diff between two nusel arms.

Generalization of oc19_census_mcp1k.py (pr/19 sec 8.2) with the two arms given
on the command line, so the same script serves the mcp1k 1000-event sweep and
the nueCC48 sweep, and the A1/A2 (clustering) and B0 (PR) rounds alike.

  ./pr20_census.py --base work-mcp1kall-cathA12off --on work-mcp1kall-cathA12on
  ./pr20_census.py --base work-nuecc48-cathA12off  --on work-nuecc48-cathA12on

Whitespace-split tsv parsing (the nusel tables are space-aligned, not real
tabs -- see project_evt444187_isolated_absorb GOTCHAS).  main_id / flash_gid are
excluded from the row-identity comparison because they are per-run bookkeeping
ids, not physics.
"""
import argparse
import collections
import glob
import os

BASE = os.path.dirname(os.path.abspath(__file__))
KEY = ("nu-candidate", "tgm", "stm", "lm", "fc")


def load(arm, evt):
    # nusel arms (run_full1k_nusel.sh) write nusel_evt<ID>/; PR-chain arms
    # (run_pr_chain_batch.sh) write pr_evt<ID>/ with the same tsv inside.
    for sub in (f"nusel_evt{evt}", f"pr_evt{evt}"):
        path = os.path.join(arm, sub, f"nusel-evt{evt}.tsv")
        if os.path.exists(path):
            break
    else:
        return None
    rows = []
    with open(path) as f:
        hdr = f.readline().split()
        for line in f:
            t = line.split()
            if len(t) == len(hdr):
                rows.append(dict(zip(hdr, t)))
    return rows


def keyof(lab):
    lab = lab.lower()
    for k in KEY:
        if k in lab:
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='baseline arm (knob OFF)')
    ap.add_argument('--on', required=True, help='arm under test (knob ON)')
    ap.add_argument('--edges', help='optional file of event ids that gained a '
                                    'connector edge; changes are split by membership')
    args = ap.parse_args()

    A = args.base if os.path.isabs(args.base) else os.path.join(BASE, args.base)
    B = args.on if os.path.isabs(args.on) else os.path.join(BASE, args.on)

    edges = set()
    if args.edges and os.path.exists(args.edges):
        with open(args.edges) as f:
            edges = {l.split()[0] for l in f if l.strip() and not l.startswith('#')}

    def enum(root):
        out = set()
        for pat, cut in (("nusel_evt*", 9), ("pr_evt*", 6)):
            out |= {os.path.basename(d)[cut:] for d in glob.glob(os.path.join(root, pat))}
        return out

    evA, evB = enum(A), enum(B)
    print(f"base = {A}")
    print(f"on   = {B}")
    print(f"events A={len(evA)} B={len(evB)} common={len(evA & evB)} "
          f"onlyA={len(evA - evB)} onlyB={len(evB - evA)}")

    n_ident = n_diff = n_missing = 0
    verdict_changes = []
    extra_only = []
    drifts = []
    flow = collections.Counter()

    for evt in sorted(evA & evB, key=int):
        ra, rb = load(A, evt), load(B, evt)
        if ra is None or rb is None:
            n_missing += 1
            continue

        def norm(rs):
            return sorted(tuple(v for k, v in r.items()
                                if k not in ("main_id", "flash_gid")) for r in rs)

        if norm(ra) == norm(rb):
            n_ident += 1
            continue
        n_diff += 1
        ca = collections.Counter(keyof(r["label"]) for r in ra if keyof(r["label"]))
        cb = collections.Counter(keyof(r["label"]) for r in rb if keyof(r["label"]))
        if ca != cb:
            verdict_changes.append((evt, dict(ca), dict(cb)))
            for k in set(ca) | set(cb):
                d = cb.get(k, 0) - ca.get(k, 0)
                if d:
                    flow[(k, "+" if d > 0 else "-")] += abs(d)
        else:
            extra_only.append(evt)

        na = {r["label"]: r for r in ra if "nu" in r["label"]}
        nb = {r["label"]: r for r in rb if "nu" in r["label"]}
        for lab in set(na) & set(nb):
            d = int(nb[lab]["npts_bundle"]) - int(na[lab]["npts_bundle"])
            if d:
                drifts.append((int(evt), lab, int(na[lab]["npts_bundle"]),
                               int(nb[lab]["npts_bundle"]), d))

    print(f"\nidentical tables: {n_ident}   differing: {n_diff}   missing tsv: {n_missing}")
    print(f"\nVERDICT-CLASS multiset changes: {len(verdict_changes)}")
    for evt, ca, cb in verdict_changes:
        tag = ""
        if edges:
            tag = "  [new-edge]" if evt in edges else "  [NO new edge -- explain]"
        print(f"   evt {evt}: {ca} -> {cb}{tag}")
    print("\nlabel-class flow:", dict(flow))
    print(f"\ndiffering-with-verdicts-unchanged (npts / extra untagged rows only): "
          f"{len(extra_only)}")
    if edges and extra_only:
        off = [e for e in extra_only if e not in edges]
        print(f"   of which WITHOUT a new edge: {len(off)}  {off[:20]}")
    print(f"\nnu-npts drifts: {len(drifts)}  range: "
          f"{min((d[4] for d in drifts), default=0)}..{max((d[4] for d in drifts), default=0)}")
    for v in sorted(drifts, key=lambda x: abs(x[4]), reverse=True)[:20]:
        print("  evt%-7d %-14s %6d -> %6d  (%+d)" % v)


if __name__ == '__main__':
    main()
