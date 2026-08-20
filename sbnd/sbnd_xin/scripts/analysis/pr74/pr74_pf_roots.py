#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/74 round 3 -- DANGLING PARTICLE-FLOW ROOT census.

The gate that doc pr/74 round 2 was missing.

`on_compare.py` compares archive member hashes, and round 2's orphan sweep
looked for painted objects with NO node in `0-mc.json`.  Neither sees the
failure mode that the owner actually reported: a node that SURVIVES in the
tree but LOSES ITS PARENT.  In jsTree terms it moves to the top level; in the
Bee particle-flow view it stops being part of the neutrino interaction, which
reads to a scanner as "identified as EM shower but missing from the particle
flow" (18255-469665, doc pr/74 round 3 Q1).

How a node is orphaned: `fill_bee_pf_tree` BFSs from the main vertex through
TRACK-ONLY segments and records `vtx_incoming_seg[V]` = the segment that
reached V (MultiAlgBlobClustering.cxx:1204-1254).  Anything anchored at V --
a shower, a pseudo-gamma, a nested shower -- hangs off that segment.  Absorb
that segment into a shower and it stops being a track, V drops out of the
map, and every object anchored there becomes a root.

The discriminant is therefore NOT "is it a root" -- every primary is a root,
legitimately, and they all start AT the neutrino vertex.  It is "is it a root
whose start point is nowhere near the neutrino vertex".

The neutrino vertex is read from `T_tagger` nu_{x,y,z} row 0 of
`tracking-pr.root` -- the same source doc pr/51's nuvtx_census.py uses, and
deliberately NOT the "most common root start point", which degenerates on a
tree with one root.

--cut-cm is NOT a tuned number and the answer does not depend on it.  Measured
on the pr/74 arms: every anchored root sits <= 1 cm from the neutrino vertex
(they ARE the vertex), and every dangling root found so far is >= 13.6 cm out.
Anything in roughly [2, 13] cm gives the identical verdict; 3.0 is the middle
of a wide plateau, not a threshold anyone fitted.

Usage:
  pr74_pf_roots.py <ARM>                     # census one arm
  pr74_pf_roots.py <ARM_BASE> <ARM_NEW>      # diff two arms (gate form)
  ... [--cut-cm 3.0] [--quiet]

Exit status in the two-arm form: 0 iff no event GAINED a dangling root.
"""
import glob
import json
import math
import os
import sys
import zipfile

SB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))



# doc pr/94 Phase 5: T_tagger/T_kine hold ONE ROW PER IN-BEAM-WINDOW BUNDLE when
# the nu_per_bundle knob is on, so a hard [0] silently reports whichever bundle
# was enumerated first.  primary_index() reproduces the legacy meaning of "the
# candidate" (longest selected main activity) and falls back to 0 for pre-pr/94
# and knob-off files.
import os as _pr94_os, sys as _pr94_sys
_pr94_sys.path.insert(0, _pr94_os.path.join(
    _pr94_os.path.dirname(_pr94_os.path.abspath(__file__)), "../.."))
from pr94_rows import primary_index  # noqa: E402

def nu_vertex(arm, evt):
    """T_tagger nu_{x,y,z} row 0, in cm.  None if unreadable."""
    p = os.path.join(SB, arm, 'pr_evt%d' % evt, 'tracking-pr.root')
    if not os.path.exists(p):
        return None
    try:
        import uproot
        t = uproot.open(p)['T_tagger']
        i = primary_index(t)
        return (float(t['nu_x'].array()[i]),
                float(t['nu_y'].array()[i]),
                float(t['nu_z'].array()[i]))
    except Exception:
        return None


def pf_roots(arm, evt):
    """[(id, text, start_xyz)] for every TOP-LEVEL node of 0-mc.json."""
    zp = os.path.join(SB, arm, 'pr_evt%d' % evt, 'mabc-pr.zip')
    if not os.path.exists(zp):
        return None
    with zipfile.ZipFile(zp) as z:
        hits = [n for n in z.namelist() if n.endswith('-mc.json')]
        if not hits:
            return None
        mc = json.load(z.open(sorted(hits)[0]))
    return [(n['id'], n['text'].strip(), tuple(n['data']['start'])) for n in mc]


def dangling(arm, evt, cut):
    """Roots further than `cut` cm from the neutrino vertex.

    None when the event cannot be measured (missing archive or missing
    T_tagger row) -- reported separately, never silently counted as clean.
    """
    roots = pf_roots(arm, evt)
    if roots is None:
        return None
    nv = nu_vertex(arm, evt)
    if nv is None:
        return None
    out = []
    for nid, text, start in roots:
        d = math.dist(start, nv)
        if d > cut:
            out.append((nid, text, round(d, 1)))
    return sorted(out, key=lambda r: str(r[0]))


def events_of(arm):
    return sorted(int(os.path.basename(d)[6:])
                  for d in glob.glob(os.path.join(SB, arm, 'pr_evt*')))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    cut = 3.0
    if '--cut-cm' in sys.argv:
        cut = float(sys.argv[sys.argv.index('--cut-cm') + 1])
    quiet = '--quiet' in sys.argv
    if not args:
        raise SystemExit(__doc__)

    if len(args) == 1:
        arm = args[0]
        tot = bad = skipped = 0
        for evt in events_of(arm):
            d = dangling(arm, evt, cut)
            if d is None:
                skipped += 1
                continue
            tot += 1
            if d:
                bad += 1
                print('evt %d: %d dangling root(s): %s' % (evt, len(d), d))
        print('\n%s: %d/%d events with a dangling PF root (cut %.1f cm)%s'
              % (arm, bad, tot, cut,
                 '' if not skipped else ', %d unmeasurable' % skipped))
        return 0

    base, new = args[0], args[1]
    tot = changed = worse = skipped = 0
    for evt in events_of(base):
        da, db = dangling(base, evt, cut), dangling(new, evt, cut)
        if da is None or db is None:
            skipped += 1
            continue
        tot += 1
        if da == db:
            continue
        changed += 1
        gained = [r for r in db if r not in da]
        healed = [r for r in da if r not in db]
        if gained:
            worse += 1
        if not quiet:
            print('--- evt %d: dangling %d -> %d' % (evt, len(da), len(db)))
            if gained:
                print('    GAINED : %s' % (gained,))
            if healed:
                print('    healed : %s' % (healed,))
    print('\n%s -> %s : %d events compared%s'
          % (base, new, tot, '' if not skipped else ', %d unmeasurable' % skipped))
    print('  changed dangling-root list : %d' % changed)
    print('  GAINED a dangling root     : %d   (gate: must be 0)' % worse)
    return 1 if worse else 0


if __name__ == '__main__':
    sys.exit(main())
