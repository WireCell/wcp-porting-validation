#!/usr/bin/env python3
"""doc pdhd/22 -- read the STM/Michel knob bag out of a COMPILED wire-cell config.

    python3 d22_cfg_keys.py <compiled.json> [<compiled_after.json>]

One argument  : print the bag.
Two arguments : print the BEFORE -> AFTER diff as "added / removed / changed",
                which is the compiled-config proof (M6) in the form the docs quote.

Why a script and not a grep: the compiled JSON is written as `"key" : value`
(space BEFORE the colon), so the natural `grep '"key":'` matches nothing and
reports every knob absent -- a vacuous PASS.  That exact failure bit this round
twice (a `strings -x` test that "proved" ks_margin missing from a binary that
uses it, and a patch-id loop that never ran).  Parse the structure instead of
pattern-matching the text.

The bag is located by content, not by path: the one config block that carries
the CheckSTM_Michel knobs.
"""
import json, sys

MARKERS = ("ks_margin", "bragg_peak_anchor", "michel_gamma_collect",
           "plateau_mip_lo", "max_candidates")


def bag(path):
    """-> the single config dict holding the STM/Michel knobs."""
    d = json.load(open(path))
    hits = []

    def walk(o):
        if isinstance(o, dict):
            if sum(m in o for m in MARKERS) >= 2:
                hits.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(d)
    if len(hits) != 1:
        sys.exit(f"{path}: expected exactly 1 STM/Michel config block, found {len(hits)}")
    return hits[0]


a = bag(sys.argv[1])
if len(sys.argv) == 2:
    print(f"{len(a)} keys in {sys.argv[1]}")
    for k in sorted(a):
        print(f"  {k:34s} {a[k]!r}")
    sys.exit(0)

b = bag(sys.argv[2])
added = sorted(set(b) - set(a))
removed = sorted(set(a) - set(b))
changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
print(f"before {len(a)} keys -> after {len(b)} keys")
print(f"ADDED   ({len(added)}):")
for k in added:
    print(f"  + {k:34s} {b[k]!r}")
print(f"REMOVED ({len(removed)}):")
for k in removed:
    print(f"  - {k:34s} {a[k]!r}")
print(f"CHANGED ({len(changed)}):")
for k in changed:
    print(f"  ~ {k:34s} {a[k]!r} -> {b[k]!r}")
print()
print("PROOF LINE: %d added, %d removed, %d changed" % (len(added), len(removed), len(changed)))
