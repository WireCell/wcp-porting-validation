#!/usr/bin/env python3
'''doc pr/112 sec 11.7 -- the five strategies under the owner's per-arm-target metric.

Owner (2026-08-23): "My hand scan selected a position; for each graph case, instead
of using this position exact, find the vertex that is closest to this position and
use THAT as the true target -- it will be different for each situation (nofitX, prod,
uniW0, snapD2, snapD3).  Since I hand-scanned on the nofitX case and clicked on the
graph, the click is only a rough position, so this is a better evaluation."

Per ARM, independently:
  target      = the arm's own hv_cloud vertex row nearest the click
  IDENTITY    = the row nearest the arm's shipped main_vertex IS that target
  DIST(X cm)  = |shipped main_vertex - target row position| <= X cm
Also reported: d(click -> target row), i.e. how far the click is from the nearest
candidate in that arm's graph (candidate-set quality, not a score).

Usage: ./pr112_five_eval.py --sample mcp2k [--arms nofitx off uniW0 snapD2 snapD3]
'''
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(os.path.dirname(HERE), 'dl_vtx_training')
sys.path.insert(0, TRAIN)
from scn_vtx import io as vio      # noqa: E402

TAGS = {'nuecc48': ['vtxscan-harv3-nuecc48'], 'ncpi0': ['vtxscan-harv3-ncpi0'],
        'mcp1k': ['vtxscan-harv3-mcp1k'],
        'mcp2k': ['vtxscan-mcp2k', 'vtxscan-mcp2k-auto', 'vtxscan-mcp2k-ragree']}
CUTS = (1.0, 1.5, 3.0)


def longest_segment_cm(j):
    '''Longest single fitted segment in the event, cm (build_dataset.py:63).'''
    return max([float(sg.get('length') or 0.0) for sg in (j.get('segments') or [])] or [0.0])


def event_longest(root, arm, evt):
    p = os.path.join(root, arm, 'pr_evt%d' % evt, 'calib-pr-evt%d.json' % evt)
    if not os.path.exists(p):
        return None
    return longest_segment_cm(vio.load_calib(p))


def seg_len_map(j):
    '''vertex id -> length (cm) of the LONGEST segment attached to it.'''
    out = {}
    for sg in (j.get('segments') or []):
        for k in ('start_vertex_id', 'end_vertex_id'):
            vid = sg.get(k)
            if vid is None:
                continue
            out[vid] = max(out.get(vid, 0.0), float(sg.get('length') or 0.0))
    return out


def read(root, arm, evt, min_seg=0.0):
    p = os.path.join(root, arm, 'pr_evt%d' % evt, 'calib-pr-evt%d.json' % evt)
    if not os.path.exists(p):
        return None
    j = vio.load_calib(p)
    sb = j.get('vertex_scoreboard') or {}
    c = sb.get('hv_cloud')
    mv = j.get('main_vertex') or {}
    if not c or not int(c.get('n_vertex_rows', 0)) or mv.get('x') is None:
        return None
    n = int(c['n_vertex_rows'])
    xyz = np.array([c['x'][:n], c['y'][:n], c['z'][:n]], float).T
    fx = np.array([mv['x'], mv['y'], mv['z']], float)
    if min_seg > 0.0:
        L = seg_len_map(j)
        ids = list(c['vertex_ids'][:n])
        keep = [i for i, vid in enumerate(ids) if L.get(vid, 0.0) >= min_seg]
        if not keep:
            return None
        xyz = xyz[keep]
    return (xyz, fx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', required=True)
    ap.add_argument('--arms', nargs='*', default=['nofitx', 'off', 'uniW0', 'snapD2', 'snapD3'])
    ap.add_argument('--pattern', default='work-pr112i-%s-%s')
    ap.add_argument('--min-seg', type=float, default=0.0,
                    help='drop candidate vertices whose LONGEST attached segment is shorter than '
                         'this many cm (owner 2026-08-23: "remove the very short track"; the same '
                         'quantity fit_vertex_min_seg_length gates at 1.0 cm, doc pr/9 sec 12)')
    ap.add_argument('--drop-unscannable', action='store_true',
                    help='doc pr/88 sec 8 / vtx_rules/scannability.py: drop events whose LONGEST '
                         'fitted segment is under --unscannable-cm (5.0) -- "only dots", the owner '
                         'cannot hand-scan a vertex there.  Judged on --scan-arm (the arm scanned).')
    ap.add_argument('--unscannable-cm', type=float, default=5.0)
    ap.add_argument('--scan-arm', default='nofitx')
    ap.add_argument('--tsv', default=None)
    a = ap.parse_args()
    root = vio.default_sbnd_root()

    per = {arm: [] for arm in a.arms}
    rows = []
    n_drop = [0]
    for lab in vio.iter_labels(root, TAGS[a.sample]):
        e = int(lab['eventNo'])
        got = {arm: read(root, a.pattern % (arm, a.sample), e, a.min_seg) for arm in a.arms}
        if any(v is None for v in got.values()):
            continue                      # score only events every arm produced
        if a.drop_unscannable:
            L0 = event_longest(root, a.pattern % (a.scan_arm, a.sample), e)
            if L0 is None or L0 < a.unscannable_cm:
                n_drop[0] += 1
                continue
        tr = np.asarray(lab['truth_xyz'], float)
        row = {'evt': e}
        for arm in a.arms:
            xyz, fx = got[arm]
            ti = int(np.argmin(np.linalg.norm(xyz - tr, axis=1)))
            pi = int(np.argmin(np.linalg.norm(xyz - fx, axis=1)))
            d_to_target = float(np.linalg.norm(fx - xyz[ti]))
            per[arm].append((int(ti == pi), d_to_target, float(np.linalg.norm(xyz[ti] - tr))))
            row[arm + '_id'] = int(ti == pi)
            row[arm + '_d'] = round(d_to_target, 3)
        rows.append(row)

    n = len(rows)
    print('=== sample %s   n=%d   (all %d arms)   min_seg=%.1f cm   unscannable dropped: %d%s'
          % (a.sample, n, len(a.arms), a.min_seg, n_drop[0],
             (' (longest segment < %.1f cm on %s)' % (a.unscannable_cm, a.scan_arm))
             if a.drop_unscannable else ' (filter OFF)'))
    hdr = '%-9s %8s   ' % ('arm', 'IDENT') + '  '.join('DIST<=%.1f' % c for c in CUTS) + '   med_d   click->target med'
    print(hdr)
    for arm in a.arms:
        v = per[arm]
        ident = sum(x[0] for x in v)
        d = np.array([x[1] for x in v]); ct = np.array([x[2] for x in v])
        cells = '  '.join('%4d (%4.1f%%)' % (int((d <= c).sum()), 100.0 * (d <= c).mean()) for c in CUTS)
        print('%-9s %4d (%4.1f%%)   %s   %5.2f   %5.2f' %
              (arm, ident, 100.0 * ident / max(n, 1), cells, np.median(d), np.median(ct)))
    if a.tsv:
        cols = list(rows[0].keys())
        with open(a.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in sorted(rows, key=lambda r: r['evt']):
                fh.write('\t'.join(str(r[c]) for c in cols) + '\n')
        print('wrote %s' % a.tsv)


if __name__ == '__main__':
    sys.exit(main())
