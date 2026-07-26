#!/usr/bin/env python3
"""Repair the absolute symlinks that tag dirs use to share event data.

Why this exists (docs/work-tags.md): a `work-<manifest>-<tag>/` dir does not
hold a private copy of every event.  Stages it did not re-run are ABSOLUTE
symlinks into an earlier tag's dir, e.g.

    work-mcp10-dq48v3/ql_evt284349
        -> /nfs/.../sbnd_xin/work-mcp10-fvxy/ql_evt284349
        -> /nfs/.../sbnd_xin/work-mcp10-mainreal/ql_evt284349   (chained)

so the tags form a dependency graph, and MOVING A TAG BREAKS EVERY DEPENDENT.
Relocating the doc-29..49 tags into archive/ on 2026-07-25 broke 1536 links and
silently zeroed `stm_fv_census.py` (it reported 0 "contained" clusters instead
of 147 -- a missing pctree is a `continue`, not an error).  Run this after any
move; it rewrites the tag component of every broken link to wherever that tag
now lives, at the top level or under archive/<campaign>/.

    python3 relink_tags.py            # dry run, reports what it would fix
    python3 relink_tags.py --apply    # rewrite

Then verify:  find . -xtype l | wc -l   # must be 0

Chained links resolve in one pass because each link is rewritten by NAME, not
by resolving the target -- an inner link still broken at the time does not stop
the outer one from being fixed.
"""
import os
import sys
import glob

BASE = os.path.dirname(os.path.abspath(__file__))


def tag_locations():
    """{tag dir name: path relative to sbnd_xin/} for top level and archive."""
    loc = {}
    for d in os.listdir(BASE):
        if d.startswith('work') and os.path.isdir(os.path.join(BASE, d)):
            loc[d] = d
    for a in glob.glob(os.path.join(BASE, 'archive', '*', '*')):
        if os.path.isdir(a):
            loc[os.path.basename(a)] = os.path.relpath(a, BASE)
    return loc


def main(apply_it):
    loc = tag_locations()
    fixed = unresolved = 0
    unk = {}
    for root, dirs, files in os.walk(BASE, followlinks=False):
        for name in dirs + files:
            p = os.path.join(root, name)
            if not os.path.islink(p) or os.path.exists(p):
                continue                      # only broken links
            tgt = os.readlink(p)
            if '/sbnd_xin/' not in tgt:
                unresolved += 1
                continue
            prefix, rest = tgt.split('/sbnd_xin/', 1)
            parts = rest.split('/', 1)
            tag, tail = parts[0], (parts[1] if len(parts) > 1 else '')
            if tag == 'archive':              # already points into the archive
                unresolved += 1
                continue
            new = loc.get(tag)
            if not new:
                unk[tag] = unk.get(tag, 0) + 1
                unresolved += 1
                continue
            newtgt = f'{prefix}/sbnd_xin/{new}' + (f'/{tail}' if tail else '')
            if apply_it:
                os.remove(p)
                os.symlink(newtgt, p)
            fixed += 1
    print(('APPLIED ' if apply_it else 'DRY-RUN ') +
          f'repaired={fixed} unresolved={unresolved}')
    for k, v in sorted(unk.items()):
        print(f'   target tag no longer exists: {k} ({v} link(s))')
    return 0 if not unk else 1


if __name__ == '__main__':
    sys.exit(main('--apply' in sys.argv[1:]))
