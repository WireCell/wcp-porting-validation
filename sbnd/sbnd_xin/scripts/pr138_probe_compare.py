#!/usr/bin/env python3
# doc pr/138 Phase B stage B1 -- does the C++ probe reproduce the offline trigger?
"""Join the WCT_SHOWER_SPLIT_DEBUG tape onto the Phase A offline features.

sec B1's success criterion, stated before the work: the probe's own fire list
must reproduce the offline one on the scanned objects; a probe that does not is
wired to the wrong population.  This script is that check, and it is deliberately
a CONTROLLED one -- where the two disagree it re-runs the offline kernel with the
vertex THE C++ ACTUALLY USED, because the pi0 finders re-seat the main vertex at
an accepted two-photon decay point (:7886, doc sec A1.4) and they run AFTER the
splitter, so on a pi0 event the calib dump's main_vertex is not the point the
probe measured from.  Without that control, "the fire lists differ" degenerates
into an excuse generator.

Repro:
    python3 scripts/pr138_probe_compare.py --arm-glob 'work-pr138r1-dbg-*'
"""
import os, sys, re, glob, csv, json, argparse, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'split_display'))
import numpy as np
import pr137_lib as L
import split_model as SM

ap = argparse.ArgumentParser()
ap.add_argument('--arm-glob', default='work-pr138r1-dbg-*')
ap.add_argument('--set', default='docs/pr/pr138-scan-analysis.tsv')
ap.add_argument('--tsv', default='docs/pr/pr138-probe-compare.tsv')
args = ap.parse_args()

V_ACCEPT, F_ACCEPT, SEP, MAXS = 0.95, 0.03, 1.6, 4

CAND = re.compile(r'SHOWER_SPLIT cand shower=(-?\d+) pdg=(-?\d+) nseg=(\d+) npts=(\d+) '
                  r'Q=(\S+) n_seed=(\d+) valley_best=(\S+) angle_best=(\S+) nacc=(\d+) '
                  r'nparts=(\d+) fired=(\d) decim=(\d) vtx=(\S+) vgap_cm=(\S+) '
                  r'vchi2=(\S+) vdQ=(\S+) vfit=(\d)')
PART = re.compile(r'SHOWER_SPLIT part shower=(-?\d+) part=(\d+) nseg=(\d+) q=(\S+) segs=(\S+)')


def read_tape():
    tape, parts = {}, collections.defaultdict(dict)
    for log in sorted(glob.glob(os.path.join(args.arm_glob, 'pr_evt*', 'stdout.log'))):
        m = re.search(r'pr_evt(\d+)', log)
        if not m:
            continue
        ev = int(m.group(1))
        for ln in open(log, errors='replace'):
            if 'SHOWER_SPLIT' not in ln:
                continue
            c = CAND.search(ln)
            if c:
                g = c.groups()
                tape[(ev, int(g[0]))] = dict(
                    pdg=int(g[1]), nseg=int(g[2]), npts=int(g[3]), Q=float(g[4]),
                    n_seed=int(g[5]), valley_best=float(g[6]), angle_best=float(g[7]),
                    nacc=int(g[8]), nparts=int(g[9]), fired=int(g[10]), decim=int(g[11]),
                    vtx=np.array([float(x) for x in g[12].split(',')]),
                    vgap=float(g[13]), vchi2=float(g[14]), vdQ=float(g[15]), vfit=int(g[16]))
                continue
            p = PART.search(ln)
            if p:
                parts[(ev, int(p.group(1)))][int(p.group(2))] = \
                    [int(x) for x in p.group(5).split(',')]
    return tape, parts


def offline(row, v=None):
    """the offline seed/accept, optionally forced onto a supplied vertex."""
    pts, q, _ = L.pack(row['P'], row['segs'])
    if pts is None or len(pts) < 8:
        return None
    vv = row['v'] if v is None else np.asarray(v, float)
    M = L.angular_maxima(pts, q, vv, L.profile_sigma_fn(), sep_scale=SEP, max_seeds=MAXS)
    k = len(M['dirs'])
    vb = 1.0
    for i in range(k):
        for j in range(i + 1, k):
            if min(M['frac'][i], M['frac'][j]) < F_ACCEPT:
                continue
            vb = min(vb, float(M['valley'][i, j]))
    acc = [0]
    for s in range(1, k):
        if len(acc) >= MAXS or M['frac'][s] < F_ACCEPT:
            continue
        if min(M['valley'][s, a] for a in acc) <= V_ACCEPT:
            acc.append(s)
    if len(acc) < 2:
        acc = []
    return dict(n_seed=k, valley_best=vb, nacc=len(acc))


tape, parts = read_tape()
print("doc pr/138 B1 -- probe vs offline")
print("tape: %d candidate lines over %d event(s)"
      % (len(tape), len({e for e, _ in tape})))

FEAT = {}
for r in csv.DictReader((l for l in open(args.set) if not l.startswith('#')), delimiter='\t'):
    FEAT[(int(r['event']), int(r['node']))] = r
print("scanned objects: %d" % len(FEAT))

rows = []
miss = []
for k in sorted(FEAT):
    t = tape.get(k)
    if t is None:
        miss.append(k)
        continue
    row = SM.load_object(k[0], k[1])
    o_dump = offline(row) if row is not None else None
    o_cxx = offline(row, t['vtx']) if row is not None else None
    dv = float(np.linalg.norm(np.asarray(row['v'], float) - t['vtx'])) if row is not None else -1.0
    rows.append(dict(event=k[0], node=k[1], dv_cm=dv, tape=t, dump=o_dump, cxxv=o_cxx,
                     nseg_off=int(FEAT[k]['nseg']), npts_off=int(FEAT[k]['npts'])))

print("\n=== 1. is the probe wired to the same POPULATION? ===")
print("  scanned objects present on the tape : %d" % len(rows))
print("  scanned objects ABSENT from the tape: %d" % len(miss))
for k in miss[:10]:
    print("      evt%-8d node%-8d owner=%s Q=%s nseg=%s"
          % (k[0], k[1], FEAT[k]['owner_verdict'], FEAT[k]['Q'], FEAT[k]['nseg']))
same_nseg = sum(1 for r in rows if r['tape']['nseg'] == r['nseg_off'])
same_npts = sum(1 for r in rows if r['tape']['npts'] == r['npts_off'])
print("  member count identical              : %d / %d" % (same_nseg, len(rows)))
print("  point   count identical             : %d / %d" % (same_npts, len(rows)))

print("\n=== 2. the reference VERTEX (the pi0 re-seat control) ===")
d = sorted(r['dv_cm'] for r in rows if r['dv_cm'] >= 0)
if d:
    moved = [r for r in rows if r['dv_cm'] > 0.1]
    print("  |v_probe - v_dump| median %.4f cm, max %.2f cm; moved >0.1 cm on %d of %d"
          % (d[len(d) // 2], d[-1], len(moved), len(d)))
    for r in sorted(moved, key=lambda r: -r['dv_cm'])[:8]:
        print("      evt%-8d node%-8d moved %.2f cm" % (r['event'], r['node'], r['dv_cm']))

print("\n=== 3. the FEATURES, against the offline kernel on the SAME vertex ===")
for lab, key in (("offline @ dump vertex", 'dump'), ("offline @ probe vertex", 'cxxv')):
    ok_seed = ok_fire = n = 0
    dvb = []
    for r in rows:
        o = r[key]
        if o is None:
            continue
        n += 1
        ok_seed += (o['n_seed'] == r['tape']['n_seed'])
        ok_fire += ((o['nacc'] >= 2) == (r['tape']['nacc'] >= 2))
        dvb.append(abs(o['valley_best'] - r['tape']['valley_best']))
    dvb.sort()
    print("  %-24s n=%3d  n_seed match %3d (%.3f)  accept match %3d (%.3f)  "
          "|dvalley| med %.2e max %.2e"
          % (lab, n, ok_seed, ok_seed / max(n, 1), ok_fire, ok_fire / max(n, 1),
             dvb[len(dvb) // 2] if dvb else -1, dvb[-1] if dvb else -1))

print("\n=== 4. the objects where the C++ and the offline-at-the-same-vertex differ ===")
bad = [r for r in rows if r['cxxv'] is not None
       and ((r['cxxv']['nacc'] >= 2) != (r['tape']['nacc'] >= 2)
            or r['cxxv']['n_seed'] != r['tape']['n_seed'])]
print("  %d object(s)" % len(bad))
for r in bad[:15]:
    print("      evt%-8d node%-8d  cxx n_seed=%d valley=%.4f nacc=%d | off n_seed=%d valley=%.4f nacc=%d"
          % (r['event'], r['node'], r['tape']['n_seed'], r['tape']['valley_best'], r['tape']['nacc'],
             r['cxxv']['n_seed'], r['cxxv']['valley_best'], r['cxxv']['nacc']))

print("\n=== 5. the FIRE LIST, as sec B1 asks for it ===")
cf = [r for r in rows if r['tape']['fired']]
print("  C++ fires on %d of %d scanned objects" % (len(cf), len(rows)))
own = collections.Counter(FEAT[(r['event'], r['node'])]['owner_verdict'] for r in cf)
print("  owner verdicts of those fires:", dict(own))
SPL = ('SPLIT2', 'SPLIT3', 'SPLIT4+')
tp = sum(1 for r in cf if FEAT[(r['event'], r['node'])]['owner_verdict'] in SPL)
pos = sum(1 for r in rows if FEAT[(r['event'], r['node'])]['owner_verdict'] in SPL)
print("  efficiency %.3f   purity %.3f   (tp %d of %d positives, %d fires)"
      % (tp / max(pos, 1), tp / max(len(cf), 1), tp, pos, len(cf)))

print("\n=== 6. the whole-population fire rate (what an ON arm would do) ===")
allc = list(tape.values())
print("  candidates above the floors: %d" % len(allc))
print("  fired                      : %d (%.1f%%)"
      % (sum(1 for t in allc if t['fired']),
         100.0 * sum(1 for t in allc if t['fired']) / max(len(allc), 1)))
print("  decimated (>4000 points)   : %d" % sum(1 for t in allc if t['decim']))

with open(args.tsv, 'w') as f:
    w = csv.writer(f, delimiter='\t')
    f.write("# doc pr/138 B1 -- C++ probe vs offline kernel, per scanned object\n")
    w.writerow(['event', 'node', 'dv_cm', 'cxx_nseg', 'off_nseg', 'cxx_npts', 'off_npts',
                'cxx_n_seed', 'cxx_valley', 'cxx_nacc', 'cxx_nparts', 'cxx_fired',
                'off_n_seed', 'off_valley', 'off_nacc',
                'offv_n_seed', 'offv_valley', 'offv_nacc',
                'vgap_cm', 'owner'])
    for r in rows:
        t, o, c = r['tape'], r['dump'] or {}, r['cxxv'] or {}
        w.writerow([r['event'], r['node'], '%.4f' % r['dv_cm'], t['nseg'], r['nseg_off'],
                    t['npts'], r['npts_off'], t['n_seed'], '%.4f' % t['valley_best'],
                    t['nacc'], t['nparts'], t['fired'],
                    o.get('n_seed', ''), ('%.4f' % o['valley_best']) if o else '',
                    o.get('nacc', ''),
                    c.get('n_seed', ''), ('%.4f' % c['valley_best']) if c else '',
                    c.get('nacc', ''), '%.2f' % t['vgap'],
                    FEAT[(r['event'], r['node'])]['owner_verdict']])
print("\nwrote %s" % args.tsv)
