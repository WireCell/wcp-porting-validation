#!/usr/bin/env python3
"""Extract the per-bundle neutrino-selection table for one event.

Inputs (driven by run_nusel_evt.sh):
  - the post-QL point-cloud tree tarball (run_ql_evt.sh -save-pctree):
    bundle structure = per-cluster scalars written by QLMatching
    (ident, cluster_t0, matched_flash_gid, flag_main/associated_cluster),
    per-cluster 3d points (npoints/length), and the root opflash PC;
  - the PR-job log (run_nusel_evt.sh -> wct_nusel_evt<ID>.log): the
    TaggerCheckTGM / TaggerCheckSTM per-main verdict lines.  The verdicts
    are taken from the log, NOT from a re-saved tree: set_flag writes the
    scalar PC only on tagged clusters, and non-uniform per-cluster arrays
    do not survive TensorDM serialization (sbnd-pattern-recognition.md 2.2).

One row per matched bundle (= per main cluster), plus one row per
beam-window flash that matched no bundle (label no-bundle).  Labels:
  TGM / STM        tagged by the cosmic taggers
  nu-candidate     in-beam-window bundle, not tagged
  not-tagged       out-of-window bundle, not tagged
  no-bundle        beam-window flash without a matched bundle

Usage:
  nusel_extract.py --pctree QL.tar.gz --prlog PR.log --out row.tsv \
      [--beam-window 0.2,2.2] [--run R --subrun S] [--no-header]
  nusel_extract.py --merge row1.tsv row2.tsv ... --out table.tsv \
      [--events-out events.tsv]
"""
import argparse
import io
import json
import re
import sys
import tarfile

import numpy as np

US = 1000.0     # WCT time unit is ns; cluster_t0 / opflash time are ns
MM = 10.0       # WCT length unit is mm

COLUMNS = ['run', 'subrun', 'event', 'main_id', 'flash_gid', 'flash_apa',
           'flash_time_us', 'flash_pe', 'in_beam', 'n_bundle', 'n_assoc',
           'npts_main', 'npts_bundle', 'len_main_cm', 'tgm', 'stm', 'label']


def load_pctree(fname):
    """Return {datapath: (metadata, array-or-None)} from a TensorDM tarball."""
    metas, arrays = {}, {}
    with tarfile.open(fname) as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if m.name.endswith('_metadata.json'):
                metas[m.name[:-len('_metadata.json')]] = json.load(f)
            elif m.name.endswith('_array.npy'):
                arrays[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(f.read()))
    by_path = {}
    for base, md in metas.items():
        if 'datapath' in md:
            by_path[md['datapath']] = (md, arrays.get(base))
    return by_path


def parse_pctree(fname):
    """Bundle structure from the post-QL tree.

    Returns (event, clusters, flashes):
      clusters: list of dicts (ident, t0_us, gid, main, assoc, npts, len_cm)
      flashes:  {gid: (time_us, pe_sum, apa)}
    """
    by_path = load_pctree(fname)
    live = [p for p in by_path if re.fullmatch(r'pointtrees/\d+/live', p)]
    if len(live) != 1:
        sys.exit(f'ERROR: {fname}: expected one live tree, got {live}')
    live = live[0]
    event = int(live.split('/')[1])

    md = by_path[live][0]
    items = by_path[md['pointclouds']][0]['items']
    lpc = by_path[md['lpcmaps']][0]['arrays']

    def ds_arrays(pcname):
        return by_path[items[pcname]][0]['arrays']

    def arr(pcname, aname):
        return by_path[ds_arrays(pcname)[aname]][1]

    cs = ds_arrays('cluster_scalar')
    ident = arr('cluster_scalar', 'ident').astype(int)
    t0 = arr('cluster_scalar', 'cluster_t0')
    gid = arr('cluster_scalar', 'matched_flash_gid').astype(int)
    main = arr('cluster_scalar', 'flag_main_cluster').astype(int)
    assoc = (arr('cluster_scalar', 'flag_associated_cluster').astype(int)
             if 'flag_associated_cluster' in cs else np.zeros_like(ident))

    # Partition the concatenated 3d PC per cluster: the lpcmaps arrays give,
    # per tree node in serialization order, how many rows that node contributed
    # to each named PC.  A cluster node contributes its cluster_scalar row; the
    # 3d rows of the following nodes (its blobs) belong to that cluster.
    map_cs = by_path[lpc['cluster_scalar']][1].astype(int)
    map_3d = by_path[lpc['3d']][1].astype(int)
    d3 = ds_arrays('3d')
    # Taggers run in the corrected scope: prefer the t0/pos-corrected arrays.
    xn = 'x_t0cor' if 'x_t0cor' in d3 else 'x'
    yn = 'y_cor' if 'y_cor' in d3 else 'y'
    zn = 'z_cor' if 'z_cor' in d3 else 'z'
    x, y, z = arr('3d', xn), arr('3d', yn), arr('3d', zn)

    starts = {}   # cluster row index -> (start, npts) into the 3d PC
    ci, pos = -1, 0
    for n in range(len(map_cs)):
        if map_cs[n]:
            ci += 1
            starts[ci] = [pos, 0]
        if map_3d[n]:
            if ci >= 0:
                starts[ci][1] += int(map_3d[n])
            pos += int(map_3d[n])

    clusters = []
    for i in range(len(ident)):
        s, n = starts.get(i, (0, 0))
        if n:
            cx, cy, cz = x[s:s+n], y[s:s+n], z[s:s+n]
            dx = (cx.max()-cx.min())**2 + (cy.max()-cy.min())**2 + (cz.max()-cz.min())**2
            length_cm = float(np.sqrt(dx)) / MM
        else:
            length_cm = 0.0
        clusters.append(dict(ident=int(ident[i]), t0_us=float(t0[i]) / US,
                             gid=int(gid[i]), main=int(main[i]),
                             assoc=int(assoc[i]), npts=int(n),
                             len_cm=length_cm))

    fl_gid = arr('opflash', 'gid').astype(int)
    fl_t = arr('opflash', 'time')
    fl_pe = arr('opflash', 'pe')
    fl_apa = arr('opflash', 'apa').astype(int)
    flashes = {}
    for g, t, p, a in zip(fl_gid, fl_t, fl_pe, fl_apa):
        if g not in flashes:
            flashes[g] = [float(t) / US, 0.0, int(a)]
        flashes[g][1] += float(p)
    return event, clusters, flashes


RE_TGM = re.compile(r'TaggerCheckTGM: cluster (\d+) \S+ TGM=(\w+)')
RE_STM = re.compile(r'TaggerCheckSTM: cluster (\d+) \S+ STM=(\w+) TGM=(\w+)')
RE_SKIP = re.compile(r'TaggerCheckSTM: cluster (\d+) already TGM; skipping')


def parse_prlog(fname):
    """Per-main verdicts {ident: {'tgm': 0/1, 'stm': 0/1}} from the PR log."""
    verdicts = {}
    as_int = {'true': 1, 'false': 0, '1': 1, '0': 0}
    with open(fname, errors='replace') as f:
        for line in f:
            m = RE_TGM.search(line)
            if m:
                verdicts.setdefault(int(m.group(1)), {})['tgm'] = as_int[m.group(2)]
                continue
            m = RE_STM.search(line)
            if m:
                v = verdicts.setdefault(int(m.group(1)), {})
                v['stm'] = as_int[m.group(2)]
                v['tgm'] = as_int[m.group(3)]
                continue
            m = RE_SKIP.search(line)
            if m:
                v = verdicts.setdefault(int(m.group(1)), {})
                v.setdefault('tgm', 1)
                v['stm'] = 0
    return verdicts


def label_of(tgm, stm, in_beam):
    if tgm == 1:
        return 'TGM'
    if stm == 1:
        return 'STM'
    return 'nu-candidate' if in_beam else 'not-tagged'


def one_event(args):
    lo, hi = (float(v) for v in args.beam_window.split(','))
    event, clusters, flashes = parse_pctree(args.pctree)
    verdicts = parse_prlog(args.prlog)

    mains = [c for c in clusters if c['main']]
    for c in mains:
        if c['ident'] not in verdicts:
            print(f'WARNING: main cluster {c["ident"]} has no tagger verdict '
                  f'in {args.prlog}', file=sys.stderr)

    rows = []
    for c in sorted(mains, key=lambda c: c['t0_us']):
        peers = [o for o in clusters if o['gid'] == c['gid'] and o is not c]
        v = verdicts.get(c['ident'], {})
        tgm, stm = v.get('tgm', -1), v.get('stm', -1)
        in_beam = int(lo <= c['t0_us'] < hi)
        fl = flashes.get(c['gid'])
        rows.append([args.run, args.subrun, event, c['ident'], c['gid'],
                     fl[2] if fl else -1,
                     f'{c["t0_us"]:.3f}', f'{fl[1]:.1f}' if fl else '-1',
                     in_beam, 1 + len(peers),
                     sum(o['assoc'] for o in peers),
                     c['npts'], c['npts'] + sum(o['npts'] for o in peers),
                     f'{c["len_cm"]:.1f}', tgm, stm,
                     label_of(tgm, stm, in_beam)])

    # Beam-window flashes that matched no bundle.
    matched_gids = {c['gid'] for c in mains}
    for g, (t, pe, apa) in sorted(flashes.items(), key=lambda kv: kv[1][0]):
        if lo <= t < hi and g not in matched_gids:
            rows.append([args.run, args.subrun, event, -1, g, apa,
                         f'{t:.3f}', f'{pe:.1f}', 1, 0, 0, 0, 0, '0.0',
                         -1, -1, 'no-bundle'])

    with open(args.out, 'w') if args.out else sys.stdout as f:
        if not args.no_header:
            print('\t'.join(COLUMNS), file=f)
        for r in rows:
            print('\t'.join(str(v) for v in r), file=f)


def merge(args):
    rows, header = [], None
    for fname in args.merge:
        with open(fname) as f:
            lines = [ln.rstrip('\n') for ln in f if ln.strip()]
        if not lines:
            continue
        if lines[0].startswith(COLUMNS[0] + '\t'):
            header = lines[0]
            lines = lines[1:]
        rows += [ln.split('\t') for ln in lines]

    idx = {c: i for i, c in enumerate(COLUMNS)}
    rows.sort(key=lambda r: (int(r[idx['run']]), int(r[idx['subrun']]),
                             int(r[idx['event']]), float(r[idx['flash_time_us']])))
    with open(args.out, 'w') if args.out else sys.stdout as f:
        print(header or '\t'.join(COLUMNS), file=f)
        for r in rows:
            print('\t'.join(r), file=f)

    if args.events_out:
        # Per-event summary from the in-window rows.  Precedence:
        # nu-candidate > cosmic-tagged (all in-window bundles TGM/STM)
        # > no-bundle (in-window flash but nothing matched) > no-beam-flash.
        events = {}
        for r in rows:
            key = (r[idx['run']], r[idx['subrun']], r[idx['event']])
            ev = events.setdefault(key, {'labels': [], 'nrows': 0})
            ev['nrows'] += 1
            if r[idx['in_beam']] == '1':
                ev['labels'].append(r[idx['label']])
        with open(args.events_out, 'w') as f:
            print('run\tsubrun\tevent\tn_bundles\tn_inbeam\tevent_label', file=f)
            for key in sorted(events, key=lambda k: tuple(int(v) for v in k)):
                labels = events[key]['labels']
                bundles = [l for l in labels if l != 'no-bundle']
                if 'nu-candidate' in labels:
                    elabel = 'nu-candidate'
                elif bundles:
                    elabel = 'cosmic-tagged'
                elif labels:
                    elabel = 'no-bundle'
                else:
                    elabel = 'no-beam-flash'
                nb = sum(1 for r in rows
                         if (r[idx['run']], r[idx['subrun']], r[idx['event']]) == key
                         and r[idx['main_id']] != '-1')
                print('\t'.join([*key, str(nb), str(len(labels)), elabel]), file=f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pctree', help='post-QL pctree tarball (bundle structure)')
    ap.add_argument('--prlog', help='PR-job log (tagger verdicts)')
    ap.add_argument('--beam-window', default='0.2,2.2',
                    help='low,high in us on cluster_t0 / flash time (default 0.2,2.2)')
    ap.add_argument('--run', type=int, default=0)
    ap.add_argument('--subrun', type=int, default=0)
    ap.add_argument('--out', help='output TSV (default stdout)')
    ap.add_argument('--no-header', action='store_true')
    ap.add_argument('--merge', nargs='+',
                    help='merge per-event TSVs into one table instead')
    ap.add_argument('--events-out', help='with --merge: per-event summary TSV')
    args = ap.parse_args()

    if args.merge:
        merge(args)
    elif args.pctree and args.prlog:
        one_event(args)
    else:
        ap.error('need --pctree + --prlog, or --merge')


if __name__ == '__main__':
    main()
