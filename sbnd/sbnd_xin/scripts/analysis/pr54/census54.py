#!/usr/bin/env python3
"""doc pr/54 round 4: the production footprint census for `other_seg_keep_isolated`
(SBND PRODUCTION ON since toolkit 3a202461). Answers what round 3's single-event
demo could not: how many events across the 48/50 manifests actually keep an
isolated residual segment, whether any nusel variable moves, and whether the
142421 "strict superset" outcome generalizes.

Combines four things scripts/analysis/pr49/on_compare.py and
scripts/analysis/pr54/oc54_after.py already do separately, plus two new ones:

  1. sentinel census        -- per-event keep/drop counts from the ON arm's
                                own PR30AUDIT oseg_iso_keep/oseg_iso_drop
                                counters (authoritative for counts; log lines
                                tear mid-word in this tree) + `pr54
                                keep-isolated:` sentinel coordinates.
  2. bidirectional gate      -- (a) every archive mover has oseg_iso_keep>0 in
                                the ON log; (b) every event with
                                oseg_iso_keep==0 is archive-byte-identical.
                                Either violated => report the first divergent
                                archive and STOP (this script exits nonzero,
                                does not paper over it).
  3. nusel diff              -- both granularities, reusing the on_compare.py
                                loader.
  4. superset-at-scale       -- per-cluster track_fit-global comparison
                                (oc54_after.py's logic) run over EVERY mover,
                                not just 142421: which clusters are
                                byte-unchanged, which are purely additive (old
                                points survive at 0.000 cm), which have
                                existing fits displaced (magnitude reported).
  5. floor-margin            -- n_points/length distribution of ON-arm keeps,
                                observation only (never used to retune the
                                floors -- CLAUDE.md S5.1/S5.7).

Usage: census54.py OFF_LABEL ON_LABEL
  e.g. census54.py work-pr54-off48a work-pr54-on48a
       census54.py work-pr54-off50a work-pr54-on50a
No defaults on purpose (doc pr/49 memory: a no-arg default silently compares
the wrong pair and PASSes vacuously).
"""
import sys, os, re, glob, json, zipfile
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest')
import hash_archive as ha

if len(sys.argv) != 3:
    sys.exit('usage: census54.py OFF_LABEL ON_LABEL (no defaults, see docstring)')

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
OFF = os.path.join(SB, sys.argv[1])
ON = os.path.join(SB, sys.argv[2])

MIN_POINTS, MIN_LENGTH = 25, 3.0

AUDIT_RE = re.compile(r'oseg_iso_drop=(\d+) oseg_iso_keep=(\d+)')
KEEP_RE = re.compile(
    r'pr54 keep-isolated: cluster (-?\d+) n_points=(\d+) length=([\d.]+) cm '
    r'v1=\(([-\d.]+),([-\d.]+),([-\d.]+)\) v2=\(([-\d.]+),([-\d.]+),([-\d.]+)\) cm')
DROP_RE = re.compile(
    r'pr54 isolated-residual drop: cluster (-?\d+) n_points=(\d+) length=([\d.]+) cm')

off_evts = sorted(int(os.path.basename(p).replace('pr_evt', ''))
                   for p in glob.glob(os.path.join(OFF, 'pr_evt*')))
on_evts = sorted(int(os.path.basename(p).replace('pr_evt', ''))
                  for p in glob.glob(os.path.join(ON, 'pr_evt*')))
common = sorted(set(off_evts) & set(on_evts))
print('OFF:', OFF, '(%d events)' % len(off_evts))
print('ON :', ON, '(%d events)' % len(on_evts))
print('common:', len(common))
if len(off_evts) != len(on_evts) or len(common) != len(off_evts):
    print('WARNING: event sets are not identical -- on_compare.py-style '
          'intersection would be silently vacuous; investigate before trusting counts below')

# ---- 1. sentinel census from the ON arm's own logs -------------------------
keep_counts, drop_counts, keeps_by_evt, kept_margin = {}, {}, {}, []
for evt in common:
    log = os.path.join(ON, 'pr_evt%d' % evt, 'wct_pr_evt%d.log' % evt)
    if not os.path.exists(log):
        print('  MISSING ON log for evt', evt); continue
    text = open(log, errors='ignore').read()
    audits = AUDIT_RE.findall(text)
    kd = sum(int(m[0]) for m in audits)
    kk = sum(int(m[1]) for m in audits)
    # PR30AUDIT is cumulative-per-call in this stage; take the max seen (last
    # call is the running total) rather than summing across calls.
    kd = max((int(m[0]) for m in audits), default=0)
    kk = max((int(m[1]) for m in audits), default=0)
    keep_counts[evt] = kk
    drop_counts[evt] = kd
    keeps = KEEP_RE.findall(text)
    keeps_by_evt[evt] = keeps
    for cl, npts, length, *_ in keeps:
        kept_margin.append((evt, int(cl), int(npts), float(length)))

n_keep_events = sum(1 for e in common if keep_counts.get(e, 0) > 0)
print('\n=== 1. sentinel census (ON arm, PR30AUDIT authoritative for counts) ===')
print('events with >=1 kept segment: %d/%d' % (n_keep_events, len(common)))
print('total kept segments: %d   total dropped candidates: %d'
      % (sum(keep_counts.values()), sum(drop_counts.values())))
for evt in common:
    if keep_counts.get(evt, 0) > 0:
        print('  evt %-7d keep=%d drop=%d  sentinels: %s'
              % (evt, keep_counts[evt], drop_counts[evt],
                 ['cl%s n=%s len=%scm' % (k[0], k[1], k[2]) for k in keeps_by_evt[evt]]))

# ---- 2. archive movers + bidirectional gate --------------------------------
movers = []
for evt in common:
    bp = os.path.join(OFF, 'pr_evt%d' % evt, 'mabc-pr.zip')
    np_ = os.path.join(ON, 'pr_evt%d' % evt, 'mabc-pr.zip')
    if not (os.path.exists(bp) and os.path.exists(np_)):
        print('  MISSING artifact evt', evt); continue
    a = dict(ha.members(bp))
    b = dict(ha.members(np_))
    dm = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    if dm:
        movers.append((evt, dm))

print('\n=== 2. archive-level movers + bidirectional gate ===')
print('%d/%d events differ' % (len(movers), len(common)))
for evt, dm in movers:
    print('  %d  %s' % (evt, dm))

gate_ok = True
mover_evts = set(e for e, _ in movers)
# (a) every mover must have oseg_iso_keep>0
for evt in sorted(mover_evts):
    if keep_counts.get(evt, 0) == 0:
        print('GATE VIOLATION (a): evt %d is an archive mover but oseg_iso_keep=0 '
              '-- diverges for a reason OTHER than this knob. STOP.' % evt)
        gate_ok = False
# (b) every zero-keep event must be byte-identical
for evt in common:
    if keep_counts.get(evt, 0) == 0 and evt in mover_evts:
        pass  # already reported above as (a) from the other direction
    if keep_counts.get(evt, 0) == 0 and evt not in mover_evts:
        continue  # expected: no keep, no move
    if keep_counts.get(evt, 0) == 0 and evt in mover_evts:
        gate_ok = False
print('bidirectional gate: %s' % ('PASS' if gate_ok else 'FAIL -- see GATE VIOLATION lines above'))

# ---- 3. nusel diff, both granularities -------------------------------------
def load_nusel(root, fname):
    rows = {}
    path = os.path.join(root, fname)
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        header = f.readline().split()
        for line in f:
            parts = line.split()
            d = dict(zip(header, parts))
            rows[int(d.get('event', d.get('evt', -1)))] = d
    return rows

print('\n=== 3. nusel diff ===')
for fname in ('nusel-events.tsv', 'nusel-table.tsv'):
    nb = load_nusel(OFF, fname)
    nn = load_nusel(ON, fname)
    ndiff = 0
    for evt in common:
        if evt in nb and evt in nn and nb[evt] != nn[evt]:
            ndiff += 1
            keys = [k for k in nb[evt] if nb[evt].get(k) != nn[evt].get(k)]
            print('  %s evt %d differs: %s' % (
                fname, evt, {k: (nb[evt].get(k), nn[evt].get(k)) for k in keys}))
    print('%s: %d/%d events differ (rows OFF=%d ON=%d)'
          % (fname, ndiff, len(common), len(nb), len(nn)))

# ---- 4. superset-at-scale: per-cluster track_fit-global, every mover -------
def fit_points(zpath):
    with zipfile.ZipFile(zpath) as z:
        ft = json.loads(z.read('data/0/0-track_fit-global.json'))
    P = np.c_[ft['x'], ft['y'], ft['z']]
    rc = np.array(ft['real_cluster_id'])
    return P, rc

print('\n=== 4. superset-at-scale (per-cluster track_fit-global, all movers) ===')
total_unchanged, total_additive, total_moved = 0, 0, 0
for evt, dm in movers:
    if not any('track_fit-global' in m for m in dm):
        continue
    bp = os.path.join(OFF, 'pr_evt%d' % evt, 'mabc-pr.zip')
    np_ = os.path.join(ON, 'pr_evt%d' % evt, 'mabc-pr.zip')
    Fb, rb = fit_points(bp)
    Fo, ro = fit_points(np_)
    cb, co = rb // 1000, ro // 1000
    clusters = sorted(set(cb.tolist()) | set(co.tolist()))
    unchanged, additive, moved = [], [], []
    for c in clusters:
        A, B = Fb[cb == c], Fo[co == c]
        if A.shape == B.shape and len(A) and np.allclose(np.sort(A, axis=0), np.sort(B, axis=0), atol=1e-9):
            unchanged.append(c); continue
        if not len(A):
            additive.append(c); continue  # brand-new cluster id, nothing to preserve
        dAB = cKDTree(B).query(A)[0] if len(B) else np.full(len(A), np.inf)
        if dAB.max() < 1e-6:
            additive.append((c, len(A), len(B)))
        else:
            moved.append((c, len(A), len(B), float(dAB.max()), float(np.median(dAB))))
    total_unchanged += len(unchanged)
    total_additive += len(additive)
    total_moved += len(moved)
    print('  evt %-7d clusters unchanged=%d additive=%d moved=%d'
          % (evt, len(unchanged), len(additive), len(moved)))
    for c, na, nb_ in [(x, x[1], x[2]) if isinstance(x, tuple) else (x, None, None) for x in additive]:
        pass
    for m in moved:
        print('      MOVED cluster %s: n %d->%d, existing-point max displacement %.3f cm, median %.3f cm'
              % m)
print('totals across all movers: unchanged=%d additive=%d moved=%d'
      % (total_unchanged, total_additive, total_moved))

# ---- 5. floor-margin distribution (observation only) -----------------------
print('\n=== 5. floor-margin distribution (kept candidates, observation only) ===')
if kept_margin:
    pts = np.array([m[2] for m in kept_margin])
    lens = np.array([m[3] for m in kept_margin])
    print('n kept = %d' % len(kept_margin))
    print('n_points: min=%d p25=%.0f median=%.0f p75=%.0f max=%d  (floor=%d)'
          % (pts.min(), np.percentile(pts, 25), np.median(pts), np.percentile(pts, 75), pts.max(), MIN_POINTS))
    print('length  : min=%.2f p25=%.2f median=%.2f p75=%.2f max=%.2f cm  (floor=%.1f cm)'
          % (lens.min(), np.percentile(lens, 25), np.median(lens), np.percentile(lens, 75), lens.max(), MIN_LENGTH))
    near_floor = np.sum((pts >= MIN_POINTS) & (pts < MIN_POINTS + 10))
    near_floor_len = np.sum((lens >= MIN_LENGTH) & (lens < MIN_LENGTH + 2))
    print('within 10 pts of the point floor: %d/%d   within 2cm of the length floor: %d/%d'
          % (near_floor, len(kept_margin), near_floor_len, len(kept_margin)))
else:
    print('no kept candidates in this manifest')

print('\n=== SUMMARY ===')
print('%s vs %s: %d/%d keep-events, %d/%d archive movers, gate=%s'
      % (os.path.basename(OFF), os.path.basename(ON), n_keep_events, len(common),
         len(movers), len(common), 'PASS' if gate_ok else 'FAIL'))
sys.exit(0 if gate_ok else 1)
