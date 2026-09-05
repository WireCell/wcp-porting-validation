#!/usr/bin/env python3
"""doc pdvd/40 round 3 -- grade a knob-ON arm against the census arm over a manifest.

Per event pair (base, arm):
  * asserts the Bee 'clustering' layer has the same point count and cluster-id
    multiset in both arms -- the retile removes SHADOW blobs only and must never
    touch the live grouping; if this fails the by-id verdict comparison below is
    invalid and is not printed (feedback_match_objects_across_layers_before_comparing)
  * cosmic-tagger verdict SETS by cluster id (TGM / STM / FC), symmetric difference
    (feedback_count_vs_set_census)
  * Steiner totals from the calib dump; wall and peak RSS from pr_resource_*.txt

Usage: d40r3_grade.py <base_tag> <arm_tag> [events.txt]
"""
import sys, os, glob, json, re, zipfile
import numpy as np
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
W = os.path.join(PDVD, 'work')
base, arm = sys.argv[1], sys.argv[2]
ev_file = sys.argv[3] if len(sys.argv) > 3 else os.path.join(PDVD, 'stm', 'events.txt')
events = ['%06d_%s' % (int(l.split()[0]), l.split()[1]) for l in open(ev_file) if l.strip() and not l.startswith('#')]
PAT_STM = re.compile(r"visit: TaggerCheckSTM: cluster (\d+) \S+ STM=([01]) TGM=([01])")
PAT = re.compile(r"visit: TaggerCheck(TGM|STM|FC): cluster (\d+) \S+ \1=(true|false|1|0)")

def verdicts(d):
    logs = glob.glob(os.path.join(d, 'wct_pr_*.log'))
    s = {'TGM': set(), 'STM': set(), 'FC': set()}
    for line in open(logs[0], errors='replace'):
        m = PAT_STM.search(line)
        if m:
            if m[2] == '1': s['STM'].add(int(m[1]))
            continue
        m = PAT.search(line)
        if m and m[3] in ('true', '1'): s[m[1]].add(int(m[2]))
    return s

def live(d):
    z = zipfile.ZipFile(os.path.join(d, 'mabc-pr.zip'))
    k = [n for n in z.namelist() if n.endswith('clustering-global.json')][0]
    j = json.loads(z.read(k))
    return len(j['x']), tuple(sorted(np.unique(j['cluster_id'], return_counts=True)[1].tolist()))

def steiner(d):
    f = glob.glob(os.path.join(d, 'calib-pr-evt*.json'))
    if not f: return None
    return sum(len(e['x']) for e in json.load(open(f[0])).get('steiner', []))

def res(d):
    f = glob.glob(os.path.join(d, 'pr_resource_*.txt'))
    if not f: return (0, 0)
    t = open(f[0]).read()
    return (float(re.search(r'wall_s=(\d+)', t)[1]), float(re.search(r'peak_rss_gb=([\d.]+)', t)[1]))

tot = {k: [0, 0, 0, 0] for k in ('TGM', 'STM', 'FC')}   # base n, arm n, only-base, only-arm
bad_live = []
st_b = st_a = 0; wall = [0, 0]; rss = [0, 0]; n = 0
per_event = []
for e in events:
    db, da = os.path.join(W, e + '_' + base), os.path.join(W, e + '_' + arm)
    if not (os.path.exists(os.path.join(db, 'mabc-pr.zip')) and os.path.exists(os.path.join(da, 'mabc-pr.zip'))):
        print(e, 'MISSING'); continue
    n += 1
    if live(db) != live(da):
        bad_live.append(e); continue
    vb, va = verdicts(db), verdicts(da)
    moved = []
    for k in tot:
        tot[k][0] += len(vb[k]); tot[k][1] += len(va[k])
        tot[k][2] += len(vb[k] - va[k]); tot[k][3] += len(va[k] - vb[k])
        if vb[k] != va[k]: moved.append('%s -%s +%s' % (k, sorted(vb[k] - va[k]), sorted(va[k] - vb[k])))
    if moved: per_event.append((e, '; '.join(moved)))
    sb, sa = steiner(db), steiner(da)
    if sb is not None and sa is not None: st_b += sb; st_a += sa
    wb, rb = res(db); wa, ra = res(da); wall[0] += wb; wall[1] += wa; rss[0] = max(rss[0], rb); rss[1] = max(rss[1], ra)
print('== %s vs %s: %d events' % (base, arm, n))
print('   live clustering layer identical (points + cluster-id multiset): %d/%d%s' % (n - len(bad_live), n, '' if not bad_live else '  MISMATCH ' + ' '.join(bad_live)))
if bad_live:
    print('   by-id verdict comparison NOT VALID for the mismatching events; they are excluded below')
for k in tot:
    print('   %-3s tagged ids: base %4d  arm %4d   only-base %3d  only-arm %3d' % (k, *tot[k]))
print('   events with any verdict-set change: %d' % len(per_event))
for e, m in per_event[:40]: print('     ', e, m)
print('   steiner points (calib): base %d  arm %d  (%.1f%%)' % (st_b, st_a, 100. * (st_a - st_b) / max(1, st_b)))
print('   wall sum: base %.0f s  arm %.0f s ;  peak RSS max: base %.2f GB  arm %.2f GB' % (wall[0], wall[1], rss[0], rss[1]))
