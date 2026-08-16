#!/usr/bin/env python3
'''
doc pr/89 Arm D -- measure the post-DL geometry adjustment stage.

Two different things are called "rerank" (doc pr/89 sec 4): (a) the in-DL
composite rerank, and (b) the post-DL adjustment that runs on
final_main_vertex REGARDLESS of route (TaggerCheckNeutrino.cxx:1415-1454:
snap_main_vertex_to_kink -> improve_vertex -> main_vertex_graph_audit).
This probes (b): per labelled event, PRE = the winning scoreboard row's
position (recorded at rerank time, i.e. before the adjustment -- the DL
winner on the accept route, the traditional winner otherwise), POST = the
stashed final answer (main_vertex, else final_* when filled).

Successor to drift_probe.py (pr/79 sec 8), which is left untouched as that
round's record: it had no argparse and its ma10-era arms are archived.

Known bias, stated not hidden: labels were hand-scanned on panels rendered
from the POST-adjustment dump geometry, so d_post inherits whatever pull the
display's fit points exert on a human click.  The pr/79 sec 8 and pr/89
sec 4-D numbers carry the same bias; deltas between arms do not.

Usage:
  python3 arm_d_probe.py --tags vtxscan-mcp2k vtxscan-mcp2k-auto ... \
      --arm-roots vtxscan-mcp2k=work-mcp2k-harv3 ... \
      --ipw-file runs/ipw-mcp2k-<date>.tsv --tsv runs/armd-<date>.tsv
'''
import argparse
import csv
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scn_vtx import io as vio                                    # noqa: E402


def pre_post(calib):
    """(pre_xyz, post_xyz, route, kind) or None when either side is absent."""
    sb = calib.get('vertex_scoreboard') or {}
    route = sb.get('route', '')
    rows = sb.get('rows') or []
    pre = kind = None
    if route == 'dl-rerank-accept':
        r = next((r for r in rows if r.get('dl_winner')), None)
        kind = 'dl-winner'
    else:
        r = next((r for r in rows if r.get('trad_winner')), None)
        kind = 'trad-winner'
    if r is not None:
        pre = np.array([r['x'], r['y'], r['z']], float)
    mv = calib.get('main_vertex') or {}
    if mv.get('x') is not None:
        post = np.array([mv['x'], mv['y'], mv['z']], float)
    elif sb.get('filled'):
        post = np.array([sb['final_x'], sb['final_y'], sb['final_z']], float)
    else:
        post = None
    if pre is None or post is None:
        return None
    return pre, post, route, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--arm-roots', nargs='+', required=True)
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--ipw-file', default=None,
                    help='TSV (evt, weight) from ipw_weights.py; unlisted '
                         'events weight 1.0')
    ap.add_argument('--exclude-events', default=None,
                    help='file of eventNo to exclude (the sealed held-out '
                         'list, when this measurement feeds a knob choice)')
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()
    excl = set()
    if args.exclude_events:
        with open(args.exclude_events) as fh:
            excl = {int(l.split()[0]) for l in fh
                    if l.strip() and not l.startswith('#')}

    ipw = {}
    if args.ipw_file:
        with open(args.ipw_file) as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                ipw[int(r['evt'])] = float(r['weight'])

    roots = vio.parse_arm_roots(args.arm_roots, args.sbnd_root)
    recs = []
    n_skip = 0
    for label in vio.iter_labels(args.sbnd_root, args.tags):
        if label['eventNo'] in excl:
            continue
        path = vio.calib_path_in_roots(roots, label, args.sbnd_root)
        if not path or not os.path.exists(path):
            n_skip += 1
            continue
        pp = pre_post(vio.load_calib(path))
        if pp is None:
            n_skip += 1
            continue
        pre, post, route, kind = pp
        truth = np.asarray(label['truth_xyz'], float)
        d_pre = float(np.linalg.norm(pre - truth))
        d_post = float(np.linalg.norm(post - truth))
        recs.append(dict(evt=label['eventNo'], tag=label['scan_tag'],
                         route=route, kind=kind,
                         moved=float(np.linalg.norm(post - pre)),
                         d_pre=d_pre, d_post=d_post,
                         w=ipw.get(label['eventNo'], 1.0)))
    print('probe over %d labelled events (%d skipped: no dump / no pre / '
          'no post)' % (len(recs), n_skip))

    def block(name, rs):
        if not rs:
            print('\n== %s: 0 events ==' % name)
            return
        mv = [r['moved'] for r in rs]
        t = args.tol
        fixed = [r for r in rs if r['d_pre'] > t and r['d_post'] <= t]
        broken = [r for r in rs if r['d_pre'] <= t and r['d_post'] > t]
        closer = [r for r in rs if r['moved'] > 1e-9 and r['d_post'] < r['d_pre']
                  and r not in fixed and r not in broken]
        further = [r for r in rs if r['moved'] > 1e-9 and r['d_post'] >= r['d_pre']
                   and r not in fixed and r not in broken]
        still = [r for r in rs if r['moved'] <= 1e-9]
        w = lambda xs: sum(r['w'] for r in xs)          # noqa: E731
        print('\n== %s: %d events (ipw-weighted %.1f) ==' % (name, len(rs), w(rs)))
        print('  displacement pre->post cm: p50 %.3f  p90 %.3f  p99 %.3f  '
              'max %.2f | moved %d (>1cm %d)'
              % (np.percentile(mv, 50), np.percentile(mv, 90),
                 np.percentile(mv, 99), max(mv),
                 sum(1 for m in mv if m > 1e-9), sum(1 for m in mv if m > 1.0)))
        print('  FIXED %d (w %.1f)   BROKEN %d (w %.1f)   net %+d (w %+.1f)'
              % (len(fixed), w(fixed), len(broken), w(broken),
                 len(fixed) - len(broken), w(fixed) - w(broken)))
        print('  closer-same-side %d   further-same-side %d   no-move %d'
              % (len(closer), len(further), len(still)))
        if broken:
            print('  broken evts: %s'
                  % ' '.join('evt%d(%.1f->%.1f)' % (r['evt'], r['d_pre'],
                                                    r['d_post'])
                             for r in sorted(broken, key=lambda r: -r['d_post'])[:8]))

    block('ALL', recs)
    for route in sorted({r['route'] for r in recs}):
        block('route ' + route, [r for r in recs if r['route'] == route])

    if args.tsv:
        cols = ['evt', 'tag', 'route', 'kind', 'moved', 'd_pre', 'd_post', 'w']
        with open(args.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in sorted(recs, key=lambda r: r['evt']):
                fh.write('\t'.join(str(r[c]) for c in cols) + '\n')
        print('\nwrote %s' % args.tsv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
