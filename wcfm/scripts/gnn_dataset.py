#!/usr/bin/env python3
"""gnn_dataset.py -- the sub-blob graph dataset for the GNN ghost ablation (wcfm/docs/05 sec 2).

One graph per (event, anode), built from our own pipeline products:

  work/<base><sub>/clusters-tru0-anode<N>-ms-active.tar.gz   every sub-blob after BlobClustering with its
                                                            BlobDepoFill true charge (ghost <=> val == 0)
  work/<base><sub>/clusters-apa-anode<N>-ms-active.tar.gz    the legacy chain's survivors (charge solving +
                                                            deghosting); solver "kept" <=> present with val > 0
  work/<base><fm>/fm-features-anode<N>.tar.gz                the FMFeatureExtract sidecar (doc 04 F3)
  work/<base>/sim-frames-anode<N>.tar.bz2                    the SP gauss frame, packed as the FM sees it
                                                            (pixel = 0.25 x sum of the 4 ticks, doc 04 sec 1)

Nodes and edges (the charge solver's own factor structure, doc 05 sec 2.1):
  blob node  = one sub-blob: 15 charge summaries (exactly crossview_probe.build_matrix), the 3 x 128
               mean-pooled FM descriptor, label, solver decision, face/slice/ident;
  wire node  = one (plane, channel, slice) pixel covered by >= 1 blob strip: packed charge, plane,
               and the 128-d FM row at that pixel (zeros when the FM saw no active pixel there);
  bw edge    = blob covers wire (one per strip wire, so sum_b q_b over a wire's blobs is the solver's
               measurement constraint);
  bb edge    = BlobClustering's blob-blob edge (adjacent slices, all-plane overlap), undirected.

    python3 scripts/gnn_dataset.py --sub _sub4 --fm _fm --out /home/xqian/tmp/wcfm-gnn/graphs --jobs 8 1-100
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from wirecell.util import ario

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crossview_probe import COLS, LAYER, load_charge, WCFM_DIR   # noqa: E402
from fm_parity import read_sidecar                              # noqa: E402

RUN = 1
KEYCOLS = [COLS['faceid'], COLS['sliceid'], COLS['u0'], COLS['u1'], COLS['v0'], COLS['v1'], COLS['w0'], COLS['w1']]


def parse_events(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def load_cluster(path):
    f = ario.load(path)
    keys = [k for k in f.keys() if k.endswith('_bnodes')]
    if len(keys) == 0:      # an anode the tracks only graze: no blobs, no cluster written
        return None, None, None
    if len(keys) != 1:
        raise RuntimeError(f'{path}: expected one cluster, found {keys}')
    pre = keys[0][:-len('bnodes')]
    return f[pre + 'bnodes'], f[pre + 'wnodes'], f[pre + 'bbedges']


def build_graph(args):
    evt, anode, sub, fm, outdir, legacy_sub = args
    base = f'{RUN:06d}_{evt}'
    out = os.path.join(outdir, f'graph-{base}-anode{anode}.npz')
    if os.path.exists(out):
        return f'{base} anode {anode}: exists'
    ev = json.load(open(os.path.join(WCFM_DIR, 'events', base + '.json')))
    bt, wt, bbe = load_cluster(os.path.join(WCFM_DIR, 'work', base + sub, f'clusters-tru0-anode{anode}-ms-active.tar.gz'))
    if bt is None:
        return f'{base} anode {anode}: no blobs, skipped'
    ba, _, _ = load_cluster(os.path.join(WCFM_DIR, 'work', base + legacy_sub, f'clusters-apa-anode{anode}-ms-active.tar.gz'))
    if ba is None:
        ba = np.zeros((0, bt.shape[1]))
    per, md = read_sidecar(os.path.join(WCFM_DIR, 'work', base + fm, f'fm-features-anode{anode}.tar.gz'))
    fdim = int(md['feature_dim'])
    qmap = load_charge(base, anode)
    wmap = {(int(r[5]), int(r[2])): int(r[4]) for r in wt}
    fidx = {p: {(int(a), int(b)): i for i, (a, b) in enumerate(per[p]['coords'])} for p in per}

    # solver decision by geometric key (idents are shared across the tiers, the key is the safer join)
    solver = {tuple(int(v) for v in r[KEYCOLS]): float(r[COLS['val']]) for r in ba}
    # doc 08 truth-only tier: the apa archive comes from another (uncut) tier, so a sub-blob's legacy
    # verdict is that of the uncut blob CONTAINING it (same face and slice, all three wire ranges).
    # Grouped by (face, slice): ranges as an int array + solver value per uncut blob.
    contain = {}
    if legacy_sub != sub:
        for r in ba:
            contain.setdefault((int(r[COLS['faceid']]), int(r[COLS['sliceid']])), []).append(
                [int(v) for v in r[KEYCOLS[2:]]] + [float(r[COLS['val']])])
        contain = {k: np.array(v) for k, v in contain.items()}

    nb = len(bt)
    b_charge = np.zeros((nb, 15), np.float32)
    b_fm = np.zeros((nb, 3 * fdim), np.float32)
    b_y = np.zeros(nb, np.int8); b_qtrue = np.zeros(nb, np.float32)
    b_kept = np.zeros(nb, np.int8); b_present = np.zeros(nb, np.int8)
    b_face = np.zeros(nb, np.int8); b_slice = np.zeros(nb, np.int32); b_ident = np.zeros(nb, np.int64)
    b_nw = np.zeros((nb, 3), np.int16)
    wire_index = {}          # (plane, channel, slice) -> wire node index
    w_key = []               # in index order
    bw_src, bw_dst = [], []
    for i, r in enumerate(bt):
        fid = int(r[COLS['faceid']]); a = fid >> 4; face = (fid >> 3) & 1
        s = int(r[COLS['sliceid']])
        b_face[i] = face; b_slice[i] = s; b_ident[i] = int(r[COLS['ident']])
        b_qtrue[i] = r[COLS['val']]; b_y[i] = int(r[COLS['val']] == 0)
        key = tuple(int(v) for v in r[KEYCOLS])
        if legacy_sub == sub:
            if key in solver:
                b_present[i] = 1; b_kept[i] = int(solver[key] > 0)
        else:
            cc = contain.get((fid, s))
            if cc is not None:
                k = np.array(key[2:])
                inside = (cc[:, 0] <= k[0]) & (k[1] <= cc[:, 1]) & (cc[:, 2] <= k[2]) & (k[3] <= cc[:, 3]) & (cc[:, 4] <= k[4]) & (k[5] <= cc[:, 5])
                if inside.any():
                    b_present[i] = 1; b_kept[i] = int((cc[inside, 6] > 0).any())
        sums = []
        for p, (lo, hi) in enumerate(((8, 9), (10, 11), (12, 13))):
            pid = (a << 4) | (face << 3) | LAYER[p]
            chans = sorted({wmap[(pid, w)] for w in range(int(r[lo]), int(r[hi]))})
            rows = [fidx[p][(c, s)] for c in chans if (c, s) in fidx[p]]
            qs = [float(qmap[c - a * 2560, s]) for c in chans]
            b_charge[i, 4 * p:4 * p + 4] = (len(chans), len(rows), np.log1p(sum(qs)), np.log1p(np.mean(qs)) if qs else 0.0)
            sums.append(np.log1p(sum(qs)))
            if rows:
                b_fm[i, p * fdim:(p + 1) * fdim] = per[p]['feat'][rows].mean(axis=0)
            b_nw[i, p] = len(chans)
            for c in chans:
                k = (p, c, s)
                j = wire_index.get(k)
                if j is None:
                    j = len(w_key); wire_index[k] = j; w_key.append(k)
                bw_src.append(i); bw_dst.append(j)
        b_charge[i, 12:15] = (sums[0] - sums[1], sums[0] - sums[2], sums[1] - sums[2])
    nw = len(w_key)
    w_plane = np.array([k[0] for k in w_key], np.int8)
    w_channel = np.array([k[1] for k in w_key], np.int32)
    w_slice = np.array([k[2] for k in w_key], np.int32)
    w_q = np.array([qmap[k[1] - anode * 2560, k[2]] for k in w_key], np.float32)
    w_fm = np.zeros((nw, fdim), np.float32); w_haspix = np.zeros(nw, np.int8)
    for j, (p, c, s) in enumerate(w_key):
        row = fidx[p].get((c, s))
        if row is not None:
            w_fm[j] = per[p]['feat'][row]; w_haspix[j] = 1
    # bb edges: (desc, tail, head) node indices into bnodes; keep both directions once
    bb = bbe[:, 1:3].astype(np.int64) if len(bbe) else np.zeros((0, 2), np.int64)
    assert bb.size == 0 or (bb.max() < nb and (b_slice[bb[:, 0]] != b_slice[bb[:, 1]]).all()), 'bbedges are not cross-slice node indices'
    np.savez_compressed(out, event=evt, anode=anode, kind=ev['kind'], ntracks=len(ev['tracks']),
                        b_charge=b_charge, b_fm=b_fm.astype(np.float16), b_y=b_y, b_qtrue=b_qtrue,
                        b_kept=b_kept, b_present=b_present, b_face=b_face, b_slice=b_slice, b_ident=b_ident, b_nw=b_nw,
                        w_plane=w_plane, w_channel=w_channel, w_slice=w_slice, w_q=w_q, w_fm=w_fm.astype(np.float16),
                        w_haspix=w_haspix, bw_src=np.array(bw_src, np.int32), bw_dst=np.array(bw_dst, np.int32),
                        bb=bb.astype(np.int32))
    ng = int(b_y.sum()); nk = int(b_kept.sum())
    wrong = int(((b_kept == 1) & (b_y == 1)).sum() + ((b_kept == 0) & (b_y == 0)).sum())
    return (f'{base} anode {anode} {ev["kind"]}: blobs {nb} ghosts {ng} wires {nw} bw {len(bw_src)} bb {len(bb)} '
            f'solver kept {nk} wrong {wrong} nopix {int((w_haspix == 0).sum())}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('events', help='e.g. 1-100 or 1,2,5')
    ap.add_argument('--sub', default='_sub4', help='work suffix of the sub-blob tier')
    ap.add_argument('--fm', default='_fm', help='work suffix of the FM sidecars')
    ap.add_argument('--legacy-sub', default=None,
                    help="work suffix holding the apa (legacy chain) archive; default = --sub.  For a truth-only "
                         "tier (run_img_evt.sh -U, doc 08) give the uncut tier ('' = work/<base>/): a sub-blob is then "
                         "'kept' when an uncut apa blob of the same face/slice containing it survived the solver")
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for evt in parse_events(a.events):
        base = f'{RUN:06d}_{evt}'
        anodes = sorted(int(os.path.basename(f).split('anode')[1].split('-')[0])
                        for f in glob.glob(os.path.join(WCFM_DIR, 'work', base + a.sub, 'clusters-tru0-anode*-ms-active.tar.gz')))
        if not anodes:
            print(f'[dataset] {base}: no {a.sub} tier, skipped', flush=True)
        for an in anodes:
            # a finished imaging job wrote time_img_a<N>.txt with rc=0 (a tar still being written is truncated)
            tf = os.path.join(WCFM_DIR, 'work', base + a.sub, f'time_img_a{an}.txt')
            if not (os.path.exists(tf) and 'rc=0' in open(tf).read()):
                print(f'[dataset] {base} anode {an}: imaging not finished ({tf}), skipped', flush=True)
                continue
            jobs.append((evt, an, a.sub, a.fm, a.out, a.sub if a.legacy_sub is None else a.legacy_sub))
    with Pool(a.jobs) as pool:
        for line in pool.imap_unordered(build_graph, jobs):
            print('[dataset]', line, flush=True)
    print(f'[dataset] {len(jobs)} graphs -> {a.out}')


if __name__ == '__main__':
    main()
