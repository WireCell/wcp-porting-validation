#!/usr/bin/env python3
'''
doc pr/77 -- build a frozen training snapshot from hand-scan labels + calib
dumps.

Per labelled event: rebuild the DL input cloud (scn_vtx/io.py), take the
label's rank-1 pick as the truth vertex, write data/<name>/evt<ID>.npz and one
row in data/<name>/manifest.tsv.  The manifest freezes label mtimes/counts so
a training set stays reproducible while a scan is still live (the mcp1k tag
grows daily).  Labels are read-only (M13); a snapshot dir REFUSES to be
overwritten -- new snapshot => new name.

Usage:
  python3 build_dataset.py --name practice66 \
      --tags vtxscan-prod0813 vtxscan-prod0813-ncpi0
'''
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio

MANIFEST_COLS = ['evt', 'tag', 'arm', 'runNo', 'subRunNo', 'n_cloud',
                 'n_vtx_points', 'n_seg_points', 'n_invalid_fit',
                 'truth_x', 'truth_y', 'truth_z', 'pick_kind',
                 'not_a_candidate', 'dis_to_main', 'corrective',
                 'sample', 'numu_score', 'prod_x', 'prod_y', 'prod_z',
                 'lockbox', 'label_saved_utc', 'label_mtime', 'npz']


def numu_top_events(sbnd_root, tags, k):
    """Top-k labeled mcp1k events by numu_score (doc pr/77 round 2: the
    owner's 50-numu data anchor; deterministic, ties broken by eventNo)."""
    scores = vio.load_scores_tsv(sbnd_root, 'mcp1k')
    cand = []
    for label in vio.iter_labels(sbnd_root, [t for t in tags if 'mcp1k' in t]):
        row = scores.get(label['eventNo'])
        ns = row['numu_score'] if row else None
        if ns is not None:
            cand.append((ns, label['eventNo']))
    cand.sort(key=lambda t: (-t[0], t[1]))
    return {evt for _, evt in cand[:k]}, dict((e, s) for s, e in cand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='snapshot name under data/')
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--event-list', default=None,
                    help='file of eventNo (one per line) or comma list; keep only these')
    ap.add_argument('--numu-top', type=int, default=0,
                    help='keep only the top-K mcp1k-tag events by numu_score '
                         '(non-mcp1k tags unaffected)')
    ap.add_argument('--numu-flag', type=int, default=0,
                    help='round 3: flag the top-K mcp1k-tag events by '
                         'numu_score as sample=numu<K> WITHOUT filtering '
                         '(the whole tag is kept)')
    ap.add_argument('--lockbox', type=float, default=0.0,
                    help='fraction of events flagged lockbox=1 (stratified by '
                         'sample x corrective; excluded from ALL training and '
                         'model selection, reported once at the end)')
    ap.add_argument('--lockbox-seed', type=int, default=20260814)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, 'data', args.name)
    if os.path.exists(os.path.join(out_dir, 'manifest.tsv')):
        print('REFUSING to overwrite existing snapshot %s -- pick a new --name'
              % out_dir)
        return 1
    os.makedirs(out_dir, exist_ok=True)

    keep_events = None
    if args.event_list:
        if os.path.exists(args.event_list):
            with open(args.event_list) as fh:
                keep_events = {int(l.split()[0]) for l in fh if l.strip()}
        else:
            keep_events = {int(t) for t in args.event_list.split(',')}

    numu_set, numu_scores = (set(), {})
    numu_name = 'numu50'
    if any('mcp1k' in t for t in args.tags):
        numu_set, numu_scores = numu_top_events(
            args.sbnd_root, args.tags,
            args.numu_top or args.numu_flag or 10**9)
        if args.numu_flag:
            numu_name = 'numu%d' % args.numu_flag
        elif not args.numu_top:
            numu_set = set()

    rows = []
    for label in vio.iter_labels(args.sbnd_root, args.tags):
        evt = label['eventNo']
        if keep_events is not None and evt not in keep_events:
            continue
        if args.numu_top and 'mcp1k' in (label['scan_tag'] or '') \
                and evt not in numu_set:
            continue
        calib_path = vio.calib_path_for_label(args.sbnd_root, label)
        calib = vio.load_calib(calib_path)
        xyz, q, info = vio.rebuild_cloud(calib)
        truth = label['truth_xyz']
        dis = label['dis_to_main']
        mv = label.get('main_vertex') or {}
        prod = np.array([mv.get('x', np.nan), mv.get('y', np.nan),
                         mv.get('z', np.nan)], dtype=np.float32)
        sample = vio.sample_of_label(label, numu_set, numu_name)
        nscore = numu_scores.get(evt)
        npz = 'evt%d.npz' % evt
        np.savez_compressed(
            os.path.join(out_dir, npz),
            xyz=xyz, q=q, truth_xyz=truth, prod_xyz=prod,
            eventNo=evt, runNo=label['runNo'] or -1, subRunNo=label['subRunNo'] or -1,
            arm=str(label['arm']), scan_tag=str(label['scan_tag']),
            pick_kind=str(label['pick_kind']),
            not_a_candidate=label['not_a_candidate'],
            dis_to_main=-1.0 if dis is None else float(dis),
            calib_path=str(calib_path))
        rows.append(dict(
            evt=evt, tag=label['scan_tag'], arm=label['arm'],
            runNo=label['runNo'], subRunNo=label['subRunNo'],
            n_cloud=len(q), n_vtx_points=info['n_vtx_points'],
            n_seg_points=info['n_seg_points'], n_invalid_fit=info['n_invalid_fit'],
            truth_x='%.6f' % truth[0], truth_y='%.6f' % truth[1],
            truth_z='%.6f' % truth[2], pick_kind=label['pick_kind'],
            not_a_candidate=int(label['not_a_candidate']),
            dis_to_main='%.4f' % (-1.0 if dis is None else dis),
            corrective=int(dis is not None and dis > 1e-9),
            sample=sample,
            numu_score='' if nscore is None else '%.4f' % nscore,
            prod_x='%.6f' % prod[0], prod_y='%.6f' % prod[1],
            prod_z='%.6f' % prod[2],
            lockbox=0,
            label_saved_utc=label['saved_utc'],
            label_mtime='%.0f' % os.path.getmtime(label['label_path']),
            npz=npz))
        print('evt%-8d %-28s %-7s cloud=%-6d corrective=%d' %
              (evt, label['scan_tag'], sample, len(q), rows[-1]['corrective']))

    if args.lockbox > 0:
        rng = np.random.default_rng(args.lockbox_seed)
        groups = {}
        for r in rows:
            groups.setdefault((r['sample'], r['corrective']), []).append(r)
        n_lock = 0
        for grp in groups.values():
            k = int(round(args.lockbox * len(grp)))
            for i in rng.permutation(len(grp))[:k]:
                grp[i]['lockbox'] = 1
                n_lock += 1
        print('lockbox: %d/%d events flagged (stratified sample x corrective, '
              'seed %d)' % (n_lock, len(rows), args.lockbox_seed))

    with open(os.path.join(out_dir, 'manifest.tsv'), 'w') as fh:
        fh.write('\t'.join(MANIFEST_COLS) + '\n')
        for r in rows:
            fh.write('\t'.join(str(r[c]) for c in MANIFEST_COLS) + '\n')
    ncorr = sum(r['corrective'] for r in rows)
    print('\nwrote %d events (%d corrective, %d confirming) -> %s'
          % (len(rows), ncorr, len(rows) - ncorr, out_dir))
    return 0


if __name__ == '__main__':
    sys.exit(main())
