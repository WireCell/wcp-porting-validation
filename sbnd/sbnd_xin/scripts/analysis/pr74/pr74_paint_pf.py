#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/74 round 1 -- read the track/shower verdict and the
particle-flow tree straight out of an already-archived pr_evt<ID>/mabc-pr.zip.

No re-run is needed for either: the Bee zip carries both answers.

  0-shower_track-global.json   q == 15000 => painted SHOWER, q == 0 => painted
                               TRACK (MultiAlgBlobClustering.cxx:876-881).
                               cluster_id     = the segment's own cluster id
                               real_cluster_id= owning shower's start-segment
                                                encoded id (cid*1000 + seg id)
  0-mc.json                    IS the particle-flow tree (jstree node list;
                               `id` is the segment id, `text` the pdg + KE,
                               `data.start`/`data.end` the node endpoints).

IMPORTANT (the caveat that governs every reading of the paint layer): the
paint takes shower MEMBERSHIP first (seg_to_shower) and falls back to the
per-segment kShowerTrajectory/kShowerTopology/pdg==11 flags only for segments
belonging to no shower.  So "painted track" does NOT mean kShowerTopology is
clear, and "painted shower" does NOT mean a shower test fired.  Use this
script for what the OWNER SEES; use WCT_PID_WRITE_DEBUG / WCT_SHOWER_TOPO_DEBUG
traces for mechanism.

Usage:
  pr74_paint_pf.py <arm> <evt> [<evt> ...]            # paint census + PF tree
  pr74_paint_pf.py <arm> <evt> --at X,Y,Z [--at ...]  # + nearest-point lookup
  pr74_paint_pf.py <arm> <evt> --min-pts 20           # paint census floor

Example (reproduces every number in doc pr/74 section "Evidence"):
  python3 scripts/analysis/pr74/pr74_paint_pf.py work-pr51r7-on48 90055 469665
  python3 scripts/analysis/pr74/pr74_paint_pf.py work-pr51r7-on19 142421 \
      --at 96.1,-74.5,232.3 --at 109.2,-77.5,219.5
  python3 scripts/analysis/pr74/pr74_paint_pf.py work-pr51r7-on50 53361
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np

SB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

LAYERS = ('shower_track-global', 'clustering-global', 'track_fit-global',
          'vertices-global', 'mc')


def load_layers(arm, evt):
    """Return {layer_name: parsed json} from this event's mabc-pr.zip."""
    zp = os.path.join(SB, arm, 'pr_evt%d' % evt, 'mabc-pr.zip')
    if not os.path.exists(zp):
        raise SystemExit('no such archive: %s' % zp)
    out = {}
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        for lay in LAYERS:
            hits = [n for n in names if n.endswith('-%s.json' % lay)]
            if not hits:
                continue
            with z.open(sorted(hits)[0]) as fh:
                out[lay] = json.load(fh)
    return out


def xyz(layer):
    return np.stack([layer['x'], layer['y'], layer['z']], axis=1)


def pca_extent(pts):
    """Length of the point set along its leading principal axis, in cm."""
    if len(pts) < 2:
        return 0.0
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    proj = (pts - c) @ vt[0]
    return float(proj.max() - proj.min())


def paint_census(st, min_pts):
    """Per-shower/segment track-vs-shower verdict from the shower_track layer."""
    p = xyz(st)
    q = np.asarray(st['q'])
    cid = np.asarray(st['cluster_id'])
    rcid = np.asarray(st['real_cluster_id'])
    rows = []
    for r in sorted(set(rcid.tolist())):
        m = rcid == r
        n = int(m.sum())
        if n < min_pts:
            continue
        pts = p[m]
        rows.append(dict(rcid=int(r), cid=int(cid[m][0]), npts=n,
                         verdict='SHOWER' if q[m][0] > 0 else 'track',
                         lpca=pca_extent(pts),
                         lo=pts.min(axis=0), hi=pts.max(axis=0)))
    rows.sort(key=lambda d: -d['npts'])
    return rows


def walk_pf(node, depth, out):
    d = node.get('data', {}) or {}
    s, e = d.get('start'), d.get('end')
    straight = 0.0
    if s and e:
        straight = float(np.linalg.norm(np.asarray(s) - np.asarray(e)))
    out.append((depth, node.get('id'), node.get('text', ''), straight, s, e))
    for c in node.get('children', []) or []:
        walk_pf(c, depth + 1, out)


def nearest(layer, tgt):
    p = xyz(layer)
    d = np.linalg.norm(p - np.asarray(tgt), axis=1)
    i = int(d.argmin())
    return float(d[i]), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm')
    ap.add_argument('events', nargs='+', type=int)
    ap.add_argument('--at', action='append', default=[],
                    help='X,Y,Z point to locate in every layer (repeatable)')
    ap.add_argument('--min-pts', type=int, default=20,
                    help='paint-census floor on points per shower/segment')
    args = ap.parse_args()

    targets = []
    for s in args.at:
        targets.append(tuple(float(v) for v in s.split(',')))

    for evt in args.events:
        lay = load_layers(args.arm, evt)
        print('=' * 78)
        print('arm %s   event %d' % (args.arm, evt))

        st = lay.get('shower_track-global')
        if st is not None:
            print('\n-- paint census (shower_track layer; q=15000 SHOWER, q=0 track) --')
            print('%-10s %-7s %-8s %-7s %-9s  bbox' %
                  ('rcid', 'cid', 'npts', 'verdict', 'L_pca/cm'))
            for r in paint_census(st, args.min_pts):
                print('%-10d %-7d %-8d %-7s %9.1f  (%.0f,%.0f,%.0f)-(%.0f,%.0f,%.0f)' %
                      (r['rcid'], r['cid'], r['npts'], r['verdict'], r['lpca'],
                       r['lo'][0], r['lo'][1], r['lo'][2],
                       r['hi'][0], r['hi'][1], r['hi'][2]))

        mc = lay.get('mc')
        if mc is not None:
            print('\n-- particle-flow tree (0-mc.json) --')
            rows = []
            for root in mc:
                walk_pf(root, 0, rows)
            for depth, nid, text, straight, s, e in rows:
                loc = ''
                if s and e:
                    loc = 'start=(%.0f,%.0f,%.0f) end=(%.0f,%.0f,%.0f)' % (
                        s[0], s[1], s[2], e[0], e[1], e[2])
                print('   ' + '  ' * depth + '- id %-8s %-24s |start-end|=%6.1fcm  %s'
                      % (nid, text, straight, loc))
            pf_ids = set(str(r[1]) for r in rows)
            if st is not None:
                painted = set(str(r['rcid']) for r in paint_census(st, args.min_pts))
                missing = sorted(painted - pf_ids, key=int)
                if missing:
                    print('\n   !! painted objects with NO node of that id in the PF tree: %s'
                          % ', '.join(missing))

        for tgt in targets:
            print('\n-- nearest point to (%.1f, %.1f, %.1f) --' % tgt)
            for name in ('shower_track-global', 'clustering-global', 'track_fit-global'):
                if name not in lay:
                    continue
                d, i = nearest(lay[name], tgt)
                extra = ''
                if name == 'shower_track-global':
                    extra = '  verdict=%s' % ('SHOWER' if lay[name]['q'][i] > 0 else 'track')
                print('   %-22s dist=%6.2f cm  cid=%-6s rcid=%-8s%s' %
                      (name, d, lay[name]['cluster_id'][i],
                       lay[name]['real_cluster_id'][i], extra))
        print()


if __name__ == '__main__':
    sys.exit(main())
