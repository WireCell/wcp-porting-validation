#!/usr/bin/env python3
"""pr/84 round 2 -- PF-tree movers census between two PR arms.

For every event present in BOTH arms, parses mabc-pr.zip::data/0/0-mc.json
and compares the multiset of (node text, parent text, rounded length).
Reports per event: pseudo-parent (gamma/neutron) nodes that disappeared,
appeared, or changed length -- the enumerated mover list the flip decision
requires (every flip must be accounted for).

Usage:
  pr84r2_pf_census.py <arm_a> <arm_b>          # a = reference (off), b = on
"""
import glob
import json
import os
import sys
import zipfile


def load_tree(zpath):
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith("0-mc.json")]
        if not names:
            return None
        return json.loads(z.read(names[0]))


def norm(node):
    data = node.get("data") or {}
    s, e = data.get("start"), data.get("end")
    ln = -1.0
    if isinstance(s, list) and isinstance(e, list) and len(s) == 3 == len(e):
        ln = sum((a - b) ** 2 for a, b in zip(s, e)) ** 0.5
    elif isinstance(s, dict) and isinstance(e, dict):
        ln = sum((s[k] - e[k]) ** 2 for k in ("x", "y", "z")) ** 0.5
    return round(ln, 2)


def flatten(tree):
    """(text, parent_text, length) rows for every node."""
    rows = []

    def walk(node, parent_text):
        t = node.get("text", "?")
        rows.append((t, parent_text, norm(node)))
        for c in node.get("children", []):
            walk(c, t)

    for n in (tree if isinstance(tree, list) else [tree]):
        walk(n, "<root>")
    return rows


def is_pseudo(text):
    return text.startswith("gamma") or text.startswith("neutron")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    arm_a, arm_b = sys.argv[1], sys.argv[2]
    n_events = n_moved = 0
    for d in sorted(glob.glob(os.path.join(arm_a, "pr_evt*"))):
        evt = os.path.basename(d).replace("pr_evt", "")
        za = os.path.join(d, "mabc-pr.zip")
        zb = os.path.join(arm_b, f"pr_evt{evt}", "mabc-pr.zip")
        if not (os.path.exists(za) and os.path.exists(zb)):
            continue
        n_events += 1
        ta, tb = load_tree(za), load_tree(zb)
        if ta is None or tb is None:
            print(f"evt={evt} MISSING mc.json a={ta is not None} "
                  f"b={tb is not None}")
            continue
        ra, rb = flatten(ta), flatten(tb)
        if ra == rb:
            continue
        n_moved += 1
        from collections import Counter
        ca, cb = Counter(ra), Counter(rb)
        gone = list((ca - cb).elements())
        new = list((cb - ca).elements())
        print(f"== evt={evt} nodes a={len(ra)} b={len(rb)}")
        for t, p, ln in sorted(gone):
            tag = "PSEUDO-GONE" if is_pseudo(t) else "gone"
            print(f"  {tag:12} {t!r} under {p!r} len={ln}")
        for t, p, ln in sorted(new):
            tag = "PSEUDO-NEW" if is_pseudo(t) else "new"
            print(f"  {tag:12} {t!r} under {p!r} len={ln}")
    print(f"# pf census: events-compared={n_events} events-moved={n_moved}")


if __name__ == "__main__":
    main()
