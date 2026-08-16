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

doc pr/79 §11 -- harvest mode: --harvest-roots reads the EXACT live SCN input
cloud from a dl_vtx_harvest arm's hv_cloud payload instead of rebuilding from
vertices[]/segments[] (the rebuilt cloud is post-refit; the live net saw the
pre-refit graph -- the ft2u transfer trap, doc pr/79 §3).  hv_cloud.q is
ALREADY the scaled charge dQ*scale+offset; q_scale/q_offset ride along as
provenance and are never re-applied.  --inherit-manifest copies the
lockbox/sample/numu_score columns per (tag, evt) from an existing snapshot so
the spent full473 lockbox is never redrawn.
'''
import argparse
import csv
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


# doc pr/88: the owner's "only dots" cut, after scanning instalment 1 --
# "there are many events in there, there are only dots.  In this case, they
# are impossible to hand scan the true vertices, so I just skipped them.  For
# our later fine tuning etc, we should filter out this kind of events."
#
# An event with no fitted object long enough to carry a direction has no
# readable vertex, so whatever position its label holds is noise that a net
# will happily fit.  The cut and its validation against the owner's own
# skip/label behaviour live in vtx_rules/scannability.py; this is a
# DELIBERATE DUPLICATE of the two lines that matter rather than an import,
# because vtx_rules is decoupled from this directory on purpose (see the
# vtx_rules/vtx_io.py docstring) and a cross-import would let an edit there
# silently move a training denominator here.
#
# UNSCANNABLE_LONGEST_CM MUST TRACK scannability.DEFAULT_LONGEST_CM (5.0).
# If you change one, change both -- there is no assertion that can catch it.
UNSCANNABLE_LONGEST_CM = 5.0


def longest_segment_cm(calib):
    """Longest single fitted segment in the event, cm."""
    return max([float(s.get('length') or 0.0)
                for s in (calib.get('segments') or [])] or [0.0])


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


def harvest_cloud(calib, evt):
    """calib of a dl_vtx_harvest arm -> (xyz, q, info), same contract as
    vio.rebuild_cloud but sourced from the recorded live cloud.  Structure
    asserts mirror verify_harvest.py.  Order is load-bearing -- never sort."""
    sb = calib.get('vertex_scoreboard') or {}
    if not sb.get('harvest'):
        raise ValueError('evt%d: calib has no harvest payload '
                         '(not a dl_vtx_harvest arm?)' % evt)
    c = sb['hv_cloud']
    x = np.array(c['x'], np.float32)
    y = np.array(c['y'], np.float32)
    z = np.array(c['z'], np.float32)
    q = np.array(c['q'], np.float32)
    if not (len(x) == len(y) == len(z) == len(q)):
        raise ValueError('evt%d: hv_cloud length mismatch' % evt)
    nvr = int(c['n_vertex_rows'])
    if not (nvr == len(c['vertex_ids']) <= len(x)):
        raise ValueError('evt%d: hv_cloud vertex block inconsistent' % evt)
    xyz = np.stack([x, y, z], axis=1)
    info = dict(n_vtx_points=nvr, n_seg_points=len(q) - nvr,
                n_invalid_fit=-1,  # not applicable: cloud recorded, not rebuilt
                dQdx_scale=float(c['q_scale']), dQdx_offset=float(c['q_offset']))
    return xyz, q, info


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
    ap.add_argument('--harvest-roots', nargs='+', default=None,
                    help='doc pr/79 §11: tag=path pairs (or one bare path, '
                         'relative to --sbnd-root) of dl_vtx_harvest arms; '
                         'cloud comes from hv_cloud instead of rebuild_cloud')
    ap.add_argument('--drop-unscannable', action='store_true',
                    help='drop "only dots" events -- longest fitted segment '
                         '< %.1f cm, i.e. nothing a vertex can be read from '
                         '(doc pr/88).  DEFAULT OFF so every pr/77-82 '
                         'snapshot rebuilds byte-for-byte; the count present '
                         'is always reported either way.  Round-3 training '
                         'pools should pass it.' % UNSCANNABLE_LONGEST_CM)
    ap.add_argument('--inherit-manifest', default=None,
                    help='existing snapshot manifest.tsv: copy lockbox / '
                         'sample / numu_score per (tag, evt) instead of '
                         'recomputing; the event sets must match exactly')
    args = ap.parse_args()

    if args.inherit_manifest and args.lockbox > 0:
        print('--inherit-manifest and --lockbox are mutually exclusive '
              '(inheriting exists precisely to avoid a fresh draw)')
        return 1
    harvest_roots = None
    if args.harvest_roots:
        harvest_roots = vio.parse_arm_roots(args.harvest_roots, args.sbnd_root)
    inherit = None
    if args.inherit_manifest:
        with open(args.inherit_manifest) as fh:
            inherit = {(r['tag'], int(r['evt'])): r
                       for r in csv.DictReader(fh, delimiter='\t')}

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
    if inherit is None and any('mcp1k' in t for t in args.tags):
        numu_set, numu_scores = numu_top_events(
            args.sbnd_root, args.tags,
            args.numu_top or args.numu_flag or 10**9)
        if args.numu_flag:
            numu_name = 'numu%d' % args.numu_flag
        elif not args.numu_top:
            numu_set = set()

    rows = []
    n_dots = []
    for label in vio.iter_labels(args.sbnd_root, args.tags):
        evt = label['eventNo']
        if keep_events is not None and evt not in keep_events:
            continue
        if args.numu_top and 'mcp1k' in (label['scan_tag'] or '') \
                and evt not in numu_set:
            continue
        if harvest_roots is not None:
            calib_path = vio.calib_path_in_roots(harvest_roots, label)
            calib = vio.load_calib(calib_path)
            xyz, q, info = harvest_cloud(calib, evt)
        else:
            calib_path = vio.calib_path_for_label(args.sbnd_root, label)
            calib = vio.load_calib(calib_path)
            xyz, q, info = vio.rebuild_cloud(calib)
        # Counted always, dropped only on request -- an unreported drop and a
        # silent inclusion are both ways to get a denominator wrong.
        if longest_segment_cm(calib) < UNSCANNABLE_LONGEST_CM:
            n_dots.append(evt)
            if args.drop_unscannable:
                continue
        truth = label['truth_xyz']
        dis = label['dis_to_main']
        mv = label.get('main_vertex') or {}
        prod = np.array([mv.get('x', np.nan), mv.get('y', np.nan),
                         mv.get('z', np.nan)], dtype=np.float32)
        if inherit is not None:
            old = inherit.get((label['scan_tag'], evt))
            if old is None:
                print('MISSING in --inherit-manifest: tag=%s evt=%d'
                      % (label['scan_tag'], evt))
                return 1
            sample = old['sample']
        else:
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
            # dQdx_offset: dataset.py's charge_jitter needs it; older
            # snapshots relied on its -1000.0 fallback matching by luck.
            dQdx_offset=float(info['dQdx_offset']),
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
            numu_score=(inherit[(label['scan_tag'], evt)].get('numu_score', '')
                        if inherit is not None
                        else ('' if nscore is None else '%.4f' % nscore)),
            prod_x='%.6f' % prod[0], prod_y='%.6f' % prod[1],
            prod_z='%.6f' % prod[2],
            lockbox=0 if inherit is None
            else int(inherit[(label['scan_tag'], evt)]['lockbox'] or 0),
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

    if inherit is not None:
        got = {(r['tag'], r['evt']) for r in rows}
        missing = sorted(set(inherit) - got)
        if missing:
            print('EVENT-SET MISMATCH vs --inherit-manifest, %d events of the '
                  'inherited snapshot are absent here, e.g. %s'
                  % (len(missing), missing[:5]))
            return 1

    with open(os.path.join(out_dir, 'manifest.tsv'), 'w') as fh:
        fh.write('\t'.join(MANIFEST_COLS) + '\n')
        for r in rows:
            fh.write('\t'.join(str(r[c]) for c in MANIFEST_COLS) + '\n')
    ncorr = sum(r['corrective'] for r in rows)
    print('\nwrote %d events (%d corrective, %d confirming) -> %s'
          % (len(rows), ncorr, len(rows) - ncorr, out_dir))
    if n_dots:
        print('"only dots" events (longest fitted segment < %.1f cm): %d %s'
              % (UNSCANNABLE_LONGEST_CM, len(n_dots),
                 'DROPPED' if args.drop_unscannable
                 else 'KEPT -- pass --drop-unscannable to remove them'))
        print('  ' + ' '.join('evt%d' % e for e in sorted(n_dots)[:20]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
