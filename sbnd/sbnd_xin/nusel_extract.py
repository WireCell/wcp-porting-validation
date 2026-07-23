#!/usr/bin/env python3
"""Extract the per-bundle neutrino-selection table for one event.

Inputs (driven by run_nusel_evt.sh):
  --pctree  post-QL point-cloud tree (run_ql_evt.sh -save-pctree): bundle
            structure from the per-cluster scalars QLMatching writes (ident,
            cluster_t0, matched_flash_gid, flag_main/associated_cluster), the
            per-cluster 3d points, and the root opflash PC.
  --prbee   the PR job's mabc-pr.zip: its clustering layer contains exactly the
            clusters that PASS switch_scope's active-volume filter, so its
            cluster_id set is the in-scope set.  scope_filter is runtime state
            and is NOT persisted in the pctree, so this zip is the only way to
            recover it (and it is more robust than the log, whose lines can be
            torn by interleaved writes -- seen on evt286329).
  --prlog   the PR-job log: TaggerCheckTGM / TaggerCheckSTM verdicts.

A row is emitted per QUALIFYING BUNDLE, defined as a cluster that is
  (a) flagged flag_main_cluster  -- QLMatching's matched main, and
  (b) in scope                   -- passes the active-volume filter,
i.e. exactly the population the taggers evaluate (clus_pr require_in_scope).
Main-flagged clusters that fail (b) are non-physical shards switch_scope split
off; they are counted and reported, not emitted.

Flashes are deduplicated across APAs first: SBND reconstructs light per TPC, so
one physical flash yields one flash object per APA (APA1 gids offset +1000000).
Flashes within FLASH_GROUP_NS of each other on opposite sides are one physical
flash and are counted once -- otherwise a bright beam flash double-counts.

Labels: TGM | STM | nu-candidate (in-window, untagged) | not-tagged
        (out-of-window, untagged) | no-bundle (in-window physical flash with no
        qualifying bundle).

Usage:
  nusel_extract.py --pctree QL.tar.gz --prbee mabc-pr.zip --prlog PR.log \
      --out row.tsv [--beam-window 0.2,2.2] [--run R --subrun S] [--no-header]
  nusel_extract.py --merge row1.tsv ... --out table.tsv [--events-out ev.tsv]
"""
import argparse
import io
import json
import re
import sys
import tarfile
import zipfile

import numpy as np

US = 1000.0            # WCT time unit is ns
MM = 10.0              # WCT length unit is mm
FLASH_GROUP_NS = 80.0  # cross-APA coincidence window (= MABC flash_group_window)

COLUMNS = ['run', 'subrun', 'event', 'main_id', 'flash_gid', 'flash_apa', 'flash_grp',
           'flash_time_us', 'flash_pe', 'flash_pe_grp', 'in_beam', 'n_bundle',
           'npts_main', 'npts_bundle', 'len_main_cm', 'n_frag', 'tgm', 'stm', 'label']


def load_pctree(fname):
    metas, arrays = {}, {}
    with tarfile.open(fname) as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if m.name.endswith('_metadata.json'):
                metas[m.name[:-len('_metadata.json')]] = json.load(f)
            elif m.name.endswith('_array.npy'):
                arrays[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(f.read()))
    return {md['datapath']: (md, arrays.get(b))
            for b, md in metas.items() if 'datapath' in md}


def parse_pctree(fname):
    """(event, clusters, flashes) from the post-QL tree."""
    bp = load_pctree(fname)
    live = [p for p in bp if re.fullmatch(r'pointtrees/\d+/live', p)]
    if len(live) != 1:
        sys.exit(f'ERROR: {fname}: expected one live tree, got {live}')
    live = live[0]
    event = int(live.split('/')[1])
    md = bp[live][0]
    items = bp[md['pointclouds']][0]['items']
    lpc = bp[md['lpcmaps']][0]['arrays']

    def ds(pcname):
        return bp[items[pcname]][0]['arrays']

    def arr(pcname, aname):
        return bp[ds(pcname)[aname]][1]

    cs = ds('cluster_scalar')
    ident = arr('cluster_scalar', 'ident').astype(int)
    t0 = arr('cluster_scalar', 'cluster_t0')
    gid = arr('cluster_scalar', 'matched_flash_gid').astype(int)
    main = arr('cluster_scalar', 'flag_main_cluster').astype(int)
    assoc = (arr('cluster_scalar', 'flag_associated_cluster').astype(int)
             if 'flag_associated_cluster' in cs else np.zeros_like(ident))

    # Per-cluster point slices: the lpcmaps arrays give, per tree node in
    # serialization order, how many rows that node contributed to each named PC.
    # A cluster node contributes its cluster_scalar row; the 3d rows of the
    # following nodes (its blobs) belong to that cluster.
    map_cs = bp[lpc['cluster_scalar']][1].astype(int)
    map_3d = bp[lpc['3d']][1].astype(int)
    d3 = ds('3d')
    xn = 'x_t0cor' if 'x_t0cor' in d3 else 'x'
    yn = 'y_cor' if 'y_cor' in d3 else 'y'
    zn = 'z_cor' if 'z_cor' in d3 else 'z'
    x, y, z = arr('3d', xn), arr('3d', yn), arr('3d', zn)

    starts, ci, pos = {}, -1, 0
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
        # End-to-end (farthest-pair) length, NOT the bounding-box diagonal: a
        # flash-merged cluster can carry a far-away fragment, which inflates the
        # bbox diagonal into a fictitious "cathode crosser" (evt284349 cluster 11
        # read 447 cm across a 3-point speck; the real track is 202 cm).
        length_cm = 0.0
        if n > 1:
            P = np.c_[x[s:s+n], y[s:s+n], z[s:s+n]]
            a = int(np.argmax(((P - P[0]) ** 2).sum(1)))
            b = int(np.argmax(((P - P[a]) ** 2).sum(1)))
            length_cm = float(np.linalg.norm(P[a] - P[b])) / MM
        clusters.append(dict(ident=int(ident[i]), t0_us=float(t0[i]) / US,
                             gid=int(gid[i]), main=int(main[i]),
                             assoc=int(assoc[i]), npts=int(n), len_cm=length_cm))

    fl_gid = arr('opflash', 'gid').astype(int)
    fl_t, fl_pe = arr('opflash', 'time'), arr('opflash', 'pe')
    fl_apa = arr('opflash', 'apa').astype(int)
    flashes = {}
    for g, t, p, a in zip(fl_gid, fl_t, fl_pe, fl_apa):
        g = int(g)
        if g not in flashes:
            flashes[g] = {'t_us': float(t) / US, 'pe': 0.0, 'apa': int(a)}
        flashes[g]['pe'] += float(p)
    return event, clusters, flashes


def group_flashes(flashes):
    """Assign each flash gid a physical-flash group id.

    Single-linkage in time with FLASH_GROUP_NS, restricted to one flash per APA
    per group: the same scintillation flash is reconstructed once per TPC, but
    two same-side flashes that close in time are genuinely distinct.
    """
    order = sorted(flashes, key=lambda g: (flashes[g]['t_us'], g))
    grp, next_grp = {}, 0
    cur, cur_sides = [], set()
    for g in order:
        t = flashes[g]['t_us']
        side = flashes[g]['apa']
        if cur and (t - flashes[cur[-1]]['t_us']) * US <= FLASH_GROUP_NS \
                and side not in cur_sides:
            cur.append(g)
            cur_sides.add(side)
        else:
            next_grp += 1
            cur, cur_sides = [g], {side}
        grp[g] = next_grp
    return grp


RE_TGM = re.compile(r'TaggerCheckTGM: cluster (\d+) \S+ TGM=(\w+)')
RE_STM = re.compile(r'TaggerCheckSTM: cluster (\d+) \S+ STM=(\w+) TGM=(\w+)')
RE_SKIP = re.compile(r'TaggerCheckSTM: cluster (\d+) already TGM; skipping')


def parse_prlog(fname):
    verdicts, as_int = {}, {'true': 1, 'false': 0, '1': 1, '0': 0}
    with open(fname, errors='replace') as f:
        for line in f:
            m = RE_TGM.search(line)
            if m:
                verdicts.setdefault(int(m.group(1)), {})['tgm'] = as_int[m.group(2)]
                continue
            m = RE_STM.search(line)
            if m:
                v = verdicts.setdefault(int(m.group(1)), {})
                v['stm'], v['tgm'] = as_int[m.group(2)], as_int[m.group(3)]
                continue
            m = RE_SKIP.search(line)
            if m:
                v = verdicts.setdefault(int(m.group(1)), {})
                v.setdefault('tgm', 1)
                v['stm'] = 0
    return verdicts


def parse_qlbee(fname):
    """{cluster_id: (len_cm_of_largest_merge_component, n_components)}.

    examine_bundles' flash merge can graft a far-away fragment onto a cluster
    (evt284349 cluster 11 = a 202 cm TPC0 track + a 3-point speck 300 cm away in
    the other TPC).  Any whole-cluster extent -- bounding box OR farthest pair --
    then reports the gap, not the track (447 / 426 cm here).  The Bee clustering
    layer carries per-point real_cluster_id = the pre-merge cluster each blob
    came from, so measure within the dominant component and report how many
    components were merged.  (real_cluster_id is written only on merged clusters,
    so it is NOT uniform per cluster and does not survive into the pctree --
    the zip is the only place it exists.)

    NB: Bee coordinates are already in cm, unlike the pctree arrays which are in
    WCT internal units (mm).  The returned length is in cm.
    """
    with zipfile.ZipFile(fname) as z:
        names = [n for n in z.namelist() if n.endswith('-clustering-global.json')]
        if not names:
            return {}
        d = json.loads(z.read(names[0]))
    if 'real_cluster_id' not in d:
        return {}
    x, y, z_ = (np.array(d[k], float) for k in ('x', 'y', 'z'))
    cid = np.array(d['cluster_id'], int)
    rid = np.array(d['real_cluster_id'], int)
    out = {}
    for c in set(cid.tolist()):
        m = cid == c
        comps = sorted(set(rid[m].tolist()))
        best_n, best_len = -1, 0.0
        for r in comps:
            s_ = m & (rid == r)
            n = int(s_.sum())
            if n <= best_n:
                continue
            P = np.c_[x[s_], y[s_], z_[s_]]
            if len(P) > 1:
                a = int(np.argmax(((P - P[0]) ** 2).sum(1)))
                b = int(np.argmax(((P - P[a]) ** 2).sum(1)))
                best_len = float(np.linalg.norm(P[a] - P[b]))
            else:
                best_len = 0.0
            best_n = n
        out[c] = (best_len, len(comps))
    return out


def parse_prbee(fname):
    """Cluster ids present in the PR Bee clustering layer = the in-scope set."""
    with zipfile.ZipFile(fname) as z:
        names = [n for n in z.namelist() if n.endswith('-clustering-global.json')]
        if not names:
            sys.exit(f'ERROR: {fname}: no clustering layer')
        d = json.loads(z.read(names[0]))
    return set(int(c) for c in d['cluster_id'])


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
    in_scope = parse_prbee(args.prbee)
    qlgeom = parse_qlbee(args.qlbee) if args.qlbee else {}
    fgrp = group_flashes(flashes)
    pe_grp = {}
    for g, f in flashes.items():
        pe_grp[fgrp[g]] = pe_grp.get(fgrp[g], 0.0) + f['pe']

    # Qualifying bundles: main-flagged AND in scope.
    mains = [c for c in clusters if c['main']]
    bundles = [c for c in mains if c['ident'] in in_scope]
    dropped = [c for c in mains if c['ident'] not in in_scope]
    for c in bundles:
        if c['ident'] not in verdicts:
            print(f'WARNING: evt{event} in-scope main {c["ident"]} has no tagger '
                  f'verdict in {args.prlog}', file=sys.stderr)
    for c in dropped:
        if c['ident'] in verdicts:
            print(f'WARNING: evt{event} out-of-scope main {c["ident"]} WAS evaluated '
                  f'by the taggers -- is require_in_scope on?', file=sys.stderr)

    rows = []
    for c in sorted(bundles, key=lambda c: c['t0_us']):
        peers = [o for o in clusters if o['gid'] == c['gid'] and o is not c]
        v = verdicts.get(c['ident'], {})
        tgm, stm = v.get('tgm', -1), v.get('stm', -1)
        in_beam = int(lo <= c['t0_us'] < hi)
        fl = flashes.get(c['gid'])
        g = fgrp.get(c['gid'], -1)
        glen, nfrag = qlgeom.get(c['ident'], (c['len_cm'], 1))   # both in cm
        rows.append([args.run, args.subrun, event, c['ident'], c['gid'],
                     fl['apa'] if fl else -1, g,
                     f'{c["t0_us"]:.3f}', f'{fl["pe"]:.1f}' if fl else '-1',
                     f'{pe_grp.get(g, 0.0):.1f}', in_beam, 1 + len(peers),
                     c['npts'], c['npts'] + sum(o['npts'] for o in peers),
                     f'{glen:.1f}', nfrag, tgm, stm,
                     label_of(tgm, stm, in_beam)])

    # In-window PHYSICAL flashes (deduped) with no qualifying bundle.
    bundle_grps = {fgrp.get(c['gid'], -1) for c in bundles
                   if lo <= c['t0_us'] < hi}
    seen = set()
    for g in sorted(flashes, key=lambda g: flashes[g]['t_us']):
        f = flashes[g]
        gr = fgrp[g]
        if not (lo <= f['t_us'] < hi) or gr in bundle_grps or gr in seen:
            continue
        seen.add(gr)
        rows.append([args.run, args.subrun, event, -1, g, f['apa'], gr,
                     f'{f["t_us"]:.3f}', f'{f["pe"]:.1f}', f'{pe_grp[gr]:.1f}',
                     1, 0, 0, 0, '0.0', 0, -1, -1, 'no-bundle'])

    with open(args.out, 'w') if args.out else sys.stdout as fh:
        if not args.no_header:
            print('\t'.join(COLUMNS), file=fh)
        for r in rows:
            print('\t'.join(str(v) for v in r), file=fh)
    if dropped:
        print(f'[evt {event}] {len(dropped)} main-flagged cluster(s) dropped as '
              f'out-of-scope (idents {sorted(c["ident"] for c in dropped)})',
              file=sys.stderr)


def merge(args):
    rows, header = [], None
    for fname in args.merge:
        with open(fname) as f:
            lines = [ln.rstrip('\n') for ln in f if ln.strip()]
        if not lines:
            continue
        if lines[0].startswith(COLUMNS[0] + '\t'):
            header, lines = lines[0], lines[1:]
        rows += [ln.split('\t') for ln in lines]

    i = {c: n for n, c in enumerate(COLUMNS)}
    rows.sort(key=lambda r: (int(r[i['run']]), int(r[i['subrun']]),
                             int(r[i['event']]), float(r[i['flash_time_us']])))
    with open(args.out, 'w') if args.out else sys.stdout as fh:
        print(header or '\t'.join(COLUMNS), file=fh)
        for r in rows:
            print('\t'.join(r), file=fh)

    if not args.events_out:
        return
    ev = {}
    for r in rows:
        key = (r[i['run']], r[i['subrun']], r[i['event']])
        e = ev.setdefault(key, {'bundles': 0, 'inbeam_bundles': [],
                                'inbeam_grps': set()})
        is_bundle = r[i['main_id']] != '-1'
        if is_bundle:
            e['bundles'] += 1
        if r[i['in_beam']] == '1':
            e['inbeam_grps'].add(r[i['flash_grp']])
            if is_bundle:
                e['inbeam_bundles'].append(r[i['label']])
    with open(args.events_out, 'w') as fh:
        print('run\tsubrun\tevent\tn_bundles\tn_inbeam_flash\tn_inbeam_bundle\t'
              'event_label', file=fh)
        for key in sorted(ev, key=lambda k: tuple(int(v) for v in k)):
            e = ev[key]
            labels = e['inbeam_bundles']
            if any(l not in ('TGM', 'STM') for l in labels):
                elabel = 'nu-candidate'      # an in-window bundle survived both taggers
            elif labels:
                elabel = 'cosmic-tagged'     # every in-window bundle is TGM/STM
            elif e['inbeam_grps']:
                elabel = 'no-bundle'
            else:
                elabel = 'no-beam-flash'
            print('\t'.join([*key, str(e['bundles']), str(len(e['inbeam_grps'])),
                             str(len(labels)), elabel]), file=fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pctree', help='post-QL pctree tarball (bundle structure)')
    ap.add_argument('--prbee', help="PR job's mabc-pr.zip (in-scope cluster set)")
    ap.add_argument('--qlbee', help="QL job's mabc-all-apa.zip (per-merge-component "
                                    "geometry via real_cluster_id)")
    ap.add_argument('--prlog', help='PR-job log (tagger verdicts)')
    ap.add_argument('--beam-window', default='0.2,2.2',
                    help='low,high in us on cluster_t0 (default 0.2,2.2)')
    ap.add_argument('--run', type=int, default=0)
    ap.add_argument('--subrun', type=int, default=0)
    ap.add_argument('--out', help='output TSV (default stdout)')
    ap.add_argument('--no-header', action='store_true')
    ap.add_argument('--merge', nargs='+', help='merge per-event TSVs instead')
    ap.add_argument('--events-out', help='with --merge: per-event summary TSV')
    args = ap.parse_args()

    if args.merge:
        merge(args)
    elif args.pctree and args.prbee and args.prlog:
        one_event(args)
    else:
        ap.error('need --pctree + --prbee + --prlog, or --merge')


if __name__ == '__main__':
    main()
