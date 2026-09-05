#!/usr/bin/env python3
"""doc pdvd/40 -- 21-event A/B on the ctpc anisotropic metric, measured on the
fabricated-Steiner-point fraction and on the cosmic-tagger verdict SETS.

Both arms read the SAME pctree per event (the doc pdvd/39 round-2 provenance
tree, tag d39r2prov) and run the SAME pipeline (-stm).  The only variable is
ctpc_aniso_metric, ON in the PDVD production PR config since the owner's
2026-09-04 flip (doc pdvd/36 sec 11, toolkit 38245d18).

"Fabricated" = a steiner_graph point with no live 3D point of ANY cluster
within N cm.  Sentinel-T0 live points (|x| > 1e4 cm, see
d40_steiner_void_census.py) are dropped from the live cloud first; they are
1480 km away and would not be anyone's nearest neighbour, but the bboxes and
the point counts must not quote them.

Verdict sets are compared by cluster id, which is legitimate HERE and only
here: neither arm splits or merges clusters, so ids denote the same objects
(unlike the doc pdvd/39 round-2 un-merge A/B, which needed a geometric
matcher).  The check below asserts that -- identical clustering point counts
per arm -- and refuses to report if it fails.

Usage:
    cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
    docs/nf_sp_img_clus/scripts/d40_aniso_arm_summary.py <events.txt> <base_tag> <arm_tag>
        events.txt : one "<run6>_<evt>" per line
"""
import zipfile, json, os, re, sys, glob
import numpy as np
from scipy.spatial import cKDTree

SENTINEL_CM = 1e4
PAT = re.compile(r"visit: TaggerCheck(TGM|STM|FC): cluster (\d+) \S+ \1=(true|false|1|0)")
PAT_STM = re.compile(r"visit: TaggerCheckSTM: cluster (\d+) \S+ STM=([01]) TGM=([01])")


def verdicts(d):
    logs = glob.glob(os.path.join(d, "wct_pr_*.log"))
    s = {"TGM": set(), "STM": set(), "FC": set()}
    if not logs:
        return None
    for line in open(logs[0], errors="replace"):
        m = PAT_STM.search(line)
        if m:
            if m.group(2) == "1":
                s["STM"].add(int(m.group(1)))
            continue
        m = PAT.search(line)
        if m and m.group(3) in ("true", "1"):
            s[m.group(1)].add(int(m.group(2)))
    return s


def zipstat(d):
    zp = os.path.join(d, 'mabc-pr.zip')
    if not os.path.exists(zp):
        return None
    z = zipfile.ZipFile(zp)
    def L(n):
        k = 'data/0/0-%s-global.json' % n
        if k not in z.namelist():
            return None
        j = json.loads(z.read(k))
        return (np.array(j['x'], float), np.array(j['y'], float),
                np.array(j['z'], float), np.array(j.get('cluster_id', [])))
    c = L('clustering'); s = L('steiner_graph')
    if c is None or s is None:
        return None
    m = np.abs(c[0]) < SENTINEL_CM
    C = np.stack([c[0][m], c[1][m], c[2][m]], 1)
    S = np.stack([s[0], s[1], s[2]], 1)
    d_any, _ = cKDTree(C).query(S)
    wall = None
    for rf in glob.glob(os.path.join(d, 'pr_resource_*.txt')):
        for tok in open(rf).read().split():
            if tok.startswith('wall_s='):
                wall = int(tok.split('=')[1])
    return {'nlive': len(C), 'nsent': int((~m).sum()), 'nste': len(S),
            'far': {t: int((d_any > t).sum()) for t in (3, 10, 30)},
            'maxd': float(d_any.max()) if len(S) else 0.0,
            'nclus_ste': len(set(s[3].tolist())), 'wall': wall}


def main(argv):
    if len(argv) != 4:
        raise SystemExit(__doc__)
    events = [l.split()[0] for l in open(argv[1]) if l.strip() and not l.startswith('#')]
    base_tag, arm_tag = argv[2], argv[3]
    tot = {k: np.zeros(2, int) for k in ('nste', 'far3', 'far10', 'far30')}
    walls = [[], []]
    vsum = {k: np.zeros(2, int) for k in ('TGM', 'STM', 'FC')}
    moved = {k: 0 for k in ('TGM', 'STM', 'FC')}
    print('%-11s | %-33s | %-33s' % ('', 'BASE ' + base_tag, 'ARM ' + arm_tag))
    print('%-11s | %7s %6s %6s %5s | %7s %6s %6s %5s | verdict sym-diff'
          % ('event', 'nSteine', '>3cm', '>10cm', '>30', 'nSteine', '>3cm', '>10cm', '>30'))
    ok = True
    for ev in events:
        run, evt = ev.split('_')
        db = 'work/%s_%s_%s' % (run, evt, base_tag)
        da = 'work/%s_%s_%s' % (run, evt, arm_tag)
        sb, sa = zipstat(db), zipstat(da)
        if sb is None or sa is None:
            print('%-11s | MISSING (%s / %s)' % (ev, sb is not None, sa is not None)); ok = False; continue
        if sb['nlive'] != sa['nlive']:
            print('%-11s | LIVE COUNT DIFFERS %d vs %d -- ids are not comparable, refusing'
                  % (ev, sb['nlive'], sa['nlive'])); ok = False; continue
        vb, va = verdicts(db), verdicts(da)
        sym = {}
        for k in ('TGM', 'STM', 'FC'):
            if vb is None or va is None:
                sym[k] = -1; continue
            sym[k] = len(vb[k] ^ va[k])
            vsum[k] += (len(vb[k]), len(va[k]))
            moved[k] += sym[k]
        for key, t in (('far3', 3), ('far10', 10), ('far30', 30)):
            tot[key] += (sb['far'][t], sa['far'][t])
        tot['nste'] += (sb['nste'], sa['nste'])
        for i, s in enumerate((sb, sa)):
            if s['wall'] is not None:
                walls[i].append(s['wall'])
        print('%-11s | %7d %6d %6d %5d | %7d %6d %6d %5d | TGM %d STM %d FC %d'
              % (ev, sb['nste'], sb['far'][3], sb['far'][10], sb['far'][30],
                 sa['nste'], sa['far'][3], sa['far'][10], sa['far'][30],
                 sym['TGM'], sym['STM'], sym['FC']))
    print('-' * 110)
    b, a = tot['nste']
    print('TOTAL       | %7d %6d %6d %5d | %7d %6d %6d %5d'
          % (b, tot['far3'][0], tot['far10'][0], tot['far30'][0],
             a, tot['far3'][1], tot['far10'][1], tot['far30'][1]))
    for key, t in (('far3', 3), ('far10', 10), ('far30', 30)):
        pb = 100. * tot[key][0] / b if b else 0
        pa = 100. * tot[key][1] / a if a else 0
        print('   fabricated > %2d cm : base %6d (%.2f%%)  ->  arm %6d (%.2f%%)' % (t, tot[key][0], pb, tot[key][1], pa))
    for k in ('TGM', 'STM', 'FC'):
        print('   %-3s tagged: base %4d -> arm %4d   (cluster ids moving either way: %d)'
              % (k, vsum[k][0], vsum[k][1], moved[k]))
    if walls[0] and walls[1]:
        print('   mean wall: base %.1f s -> arm %.1f s' % (np.mean(walls[0]), np.mean(walls[1])))
    print('STATUS: %s' % ('all events read' if ok else 'INCOMPLETE -- see MISSING/REFUSED rows above'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
