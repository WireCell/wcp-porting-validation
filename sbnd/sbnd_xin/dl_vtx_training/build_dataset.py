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
                 'label_saved_utc', 'label_mtime', 'npz']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='snapshot name under data/')
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, 'data', args.name)
    if os.path.exists(os.path.join(out_dir, 'manifest.tsv')):
        print('REFUSING to overwrite existing snapshot %s -- pick a new --name'
              % out_dir)
        return 1
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for label in vio.iter_labels(args.sbnd_root, args.tags):
        evt = label['eventNo']
        calib_path = vio.calib_path_for_label(args.sbnd_root, label)
        calib = vio.load_calib(calib_path)
        xyz, q, info = vio.rebuild_cloud(calib)
        truth = label['truth_xyz']
        dis = label['dis_to_main']
        npz = 'evt%d.npz' % evt
        np.savez_compressed(
            os.path.join(out_dir, npz),
            xyz=xyz, q=q, truth_xyz=truth,
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
            label_saved_utc=label['saved_utc'],
            label_mtime='%.0f' % os.path.getmtime(label['label_path']),
            npz=npz))
        print('evt%-8d %-28s cloud=%-6d corrective=%d' %
              (evt, label['scan_tag'], len(q), rows[-1]['corrective']))

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
