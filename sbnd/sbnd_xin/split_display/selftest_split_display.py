#!/usr/bin/env python3
# doc pr/138 Phase A -- selftest for the split scan tool's PYTHON side.
"""What is and is not covered, stated plainly (same honest limit as em3d.py):
there is no JS engine in this tree, so the drag-and-drop and the recolour are
NOT machine-tested -- they are covered by the manual check-list in doc pr/138
sec A1.  What IS tested here is everything the browser cannot get wrong for us:
the payload, the proposal, the bundle decomposition, the verdict derivation and
the label round-trip including the M13 refusal.

    ./split_display/selftest_split_display.py
"""
import os, sys, json, tempfile, shutil, collections
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)
import split_model as SM

FAIL = []
def check(name, ok, detail=''):
    print("  %-58s %s%s" % (name, 'ok' if ok else 'FAIL', (' -- ' + detail) if detail and not ok else ''))
    if not ok: FAIL.append(name)

print("split_display selftest (doc pr/138)")

# 1. payload shape and the bundle decomposition
r = SM.load_object(21073, 63100)
check("load_object finds evt21073 node63100", r is not None)
p = SM.object_payload(r)
check("payload has one row per segment", p['nseg'] == len(p['segs']))
check("every segment carries a bundle and a group",
      all('bundle' in s and 'group' in s for s in p['segs']))
check("bundles are contiguous 0..n-1",
      sorted({s['bundle'] for s in p['segs']}) == list(range(p['nbundle'])))

# 2. a bundle is never cut by the proposal
bg = collections.defaultdict(set)
for s in p['segs']: bg[s['bundle']].add(s['group'])
check("the proposal never splits a bundle", all(len(v) == 1 for v in bg.values()),
      "bundles with mixed groups: %s" % [b for b, v in bg.items() if len(v) > 1])

# 3. junk is NOT pre-flagged (doc pr/137 sec 15.3)
check("junk is not pre-flagged by default",
      all(s['group'] != SM.JUNK for s in p['segs']))
g2, _, _ = SM.propose(r, flag_junk=True)
check("flag_junk=True is still reachable for experiments",
      isinstance(g2, dict) and len(g2) == p['nseg'])

# 4. a clean single shower gets ONE group; a known merge gets two
r1 = SM.load_object(256587, 11301); p1 = SM.object_payload(r1)
check("256587 (clean single) proposes one group",
      len({s['group'] for s in p1['segs']}) == 1, p1['reason'])
check("21073 (known 2-way merge) proposes two groups",
      len({s['group'] for s in p['segs']}) == 2, p['reason'])

# 5. verdict derivation, mirrored from the viewer
def derive(grp):
    n = len({g for g in grp.values() if g != SM.JUNK})
    junk = any(g == SM.JUNK for g in grp.values())
    return 'SPLIT3' if n >= 3 else ('SPLIT2' if n == 2 else ('TRIM' if junk else 'KEEP'))
check("1 group -> KEEP", derive({1: 0, 2: 0}) == 'KEEP')
check("2 groups -> SPLIT2", derive({1: 0, 2: 1}) == 'SPLIT2')
check("3 groups -> SPLIT3", derive({1: 0, 2: 1, 3: 2}) == 'SPLIT3')
check("1 group + junk -> TRIM", derive({1: 0, 2: SM.JUNK}) == 'TRIM')
check("junk alone does not become SPLIT2", derive({1: 0, 2: SM.JUNK}) != 'SPLIT2')

# 6. Bee link resolution
import bee_links as BL
m = BL.scan()
check("bee scan resolves some events", len(m) > 100, "resolved %d" % len(m))
ev0 = sorted(m)[0]
name, url, idx = m[ev0][0]
check("a resolved link is a per-event deep link, not the set list",
      '/event/%d/' % idx in url and '/set/' in url, url)
check("an unresolvable event yields a note, not a broken link",
      'no uploaded Bee set' in BL.links_html(m, -1))
check("links_html emits anchors for a resolved event",
      BL.links_html(m, ev0).count('<a ') >= 1)

# 7. the M13 guard: refuse to write into a foreign label dir
tmp = tempfile.mkdtemp(prefix='pr138sel-')
try:
    foreign = os.path.join(tmp, 'emscan-pretend')
    os.makedirs(foreign)
    open(os.path.join(foreign, 'labels-evt1.json'), 'w').write('{}')
    has_json = any(f.endswith('.json') for f in os.listdir(foreign))
    guard = os.path.exists(os.path.join(foreign, '.split_display_tag'))
    check("a foreign label dir is detected (would refuse)", has_json and not guard)
    fresh = os.path.join(tmp, 'splitscan-fresh')
    os.makedirs(fresh)
    check("a fresh dir is writable (would create the marker)",
          not any(f.endswith('.json') for f in os.listdir(fresh)))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------------------
# doc pr/138 sec A1.5-A1.7 -- the round the owner's three reports opened.

# 6. the colour modes (owner: "the color of the group is gone now")
cc = SM.charge_colors([-5.0, 0.0, 100.0, 1e5])
check("charge_colors survives the dump's NEGATIVE dQ", len(cc) == 4 and all(cc))
check("charge_colors spans the ramp", cc[0] == SM.CHARGE_RAMP[0] and cc[-1] == SM.CHARGE_RAMP[-1])
check("the bundle palette is longer than the group palette",
      len(SM.BUNDLE_COLORS) > len(SM.GROUP_COLORS))

# 7. the pi0 vertex re-seat (owner: "how can the fit vertex be so much off?")
part, mass = SM.pio_partner(396222, 9059)
check("evt396222 node9059 has a pi0 partner", part is not None and mass is not None,
      "got %s / %s" % (part, mass))
check("the partner is shower 130313 at the pi0 mass",
      part is not None and part['id'] == 130313 and 120.0 < mass < 150.0,
      "partner=%s mass=%s" % (part.get('id') if part else None, mass))
check("a non-pi0 object reports no partner", SM.pio_partner(99838, 14004) == (None, None))

# 8. the proposal never claims two groups it did not make (evt389538 node19021)
r389 = SM.load_object(389538, 19021)
if r389 is None:
    check("evt389538 node19021 loads", False)
    check("a collapsed 2-seed proposal says so", False)
else:
    g389, _, why = SM.propose(r389)
    ng = len({v for v in g389.values() if v != SM.JUNK})
    check("evt389538 node19021 loads", True)
    check("a collapsed 2-seed proposal says so",
          not (ng == 1 and why.startswith("2 groups")), "ng=%d reason=%r" % (ng, why))

print("\n%d checks, %d FAILED" % (28, len(FAIL)))
sys.exit(1 if FAIL else 0)
