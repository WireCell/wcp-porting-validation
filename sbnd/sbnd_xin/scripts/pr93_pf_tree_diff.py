#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/93 round 4 -- PF-tree (Bee 0-mc.json) A/B diff.

Round 3's pr93_shower_ab_diff.py compares shower LABELS from the calib
dumps; round 4's fixes (shower_detach_track_stem, pf_track_owns_loose_vertex,
pf_orphan_confident_track) change the PF tree's STRUCTURE -- parentage and
node membership -- so this script diffs the rendered jsTree itself.

For each pr_evt<ID> present in both arms, load mabc-pr.zip::data/0/0-mc.json,
flatten to {id: (name, parent_id)}, and report per event:
  ROOT   n_before -> n_after      (root-node count; the F4 invariant says it
                                   must never INCREASE under
                                   pf_track_owns_loose_vertex alone)
  ADD    id name                  (node only in the after arm)
  DEL    id name                  (node only in the before arm)
  REPAR  id name parent_b -> parent_a
  RENAME id name_b -> name_a      (same id, text changed = label/KE moved)

Pseudo-carrier nodes (small serial ids: gamma/neutron/pi0) can be
re-numbered between arms with no physical meaning -- a carrier ADD+DEL pair
with the same name and parent is renumber noise, read them together.

Usage: pr93_pf_tree_diff.py BEFORE_DIR AFTER_DIR [--events E1,E2,...]
"""
import argparse
import json
import os
import sys
import zipfile


def load_tree(workdir, evt):
    zp = os.path.join(workdir, f"pr_evt{evt}", "mabc-pr.zip")
    if not os.path.exists(zp):
        return None
    with zipfile.ZipFile(zp) as z:
        with z.open("data/0/0-mc.json") as f:
            return json.load(f)


def flatten(tree):
    """Return ({id: (name, parent_id)}, [root ids]) from the jsTree array."""
    nodes = {}
    roots = []

    def walk(node, parent):
        nid = node["id"]
        nodes[nid] = (node.get("text", ""), parent)
        for ch in node.get("children", []) or []:
            walk(ch, nid)

    for root in tree:
        roots.append(root["id"])
        walk(root, None)
    return nodes, roots


def diff_event(evt, before, after, out):
    nb, rb = flatten(before)
    na, ra = flatten(after)
    lines = []
    if len(rb) != len(ra):
        lines.append(f"  ROOT   {len(rb)} -> {len(ra)}"
                     + ("   ** ROOT COUNT INCREASED **" if len(ra) > len(rb) else ""))
    only_b = set(nb) - set(na)
    only_a = set(na) - set(nb)
    for nid in sorted(only_a):
        lines.append(f"  ADD    {nid} {na[nid][0]!r} parent={na[nid][1]}")
    for nid in sorted(only_b):
        lines.append(f"  DEL    {nid} {nb[nid][0]!r} parent={nb[nid][1]}")
    for nid in sorted(set(nb) & set(na)):
        name_b, par_b = nb[nid]
        name_a, par_a = na[nid]
        if par_b != par_a:
            lines.append(f"  REPAR  {nid} {name_a!r} {par_b} -> {par_a}")
        if name_b != name_a:
            lines.append(f"  RENAME {nid} {name_b!r} -> {name_a!r}")
    if lines:
        print(f"evt {evt}:", file=out)
        for ln in lines:
            print(ln, file=out)
    return bool(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--events", default=None,
                    help="comma-separated event ids (default: all common pr_evt dirs)")
    args = ap.parse_args()

    if args.events:
        evts = args.events.split(",")
    else:
        eb = {d[6:] for d in os.listdir(args.before) if d.startswith("pr_evt")}
        ea = {d[6:] for d in os.listdir(args.after) if d.startswith("pr_evt")}
        evts = sorted(eb & ea, key=lambda s: int(s))

    n_moved = 0
    for evt in evts:
        tb = load_tree(args.before, evt)
        ta = load_tree(args.after, evt)
        if tb is None or ta is None:
            print(f"evt {evt}: MISSING mc.json "
                  f"(before={'ok' if tb is not None else 'absent'}, "
                  f"after={'ok' if ta is not None else 'absent'})")
            continue
        if diff_event(evt, tb, ta, sys.stdout):
            n_moved += 1
    print(f"# {n_moved}/{len(evts)} event(s) with PF-tree differences")


if __name__ == "__main__":
    main()
