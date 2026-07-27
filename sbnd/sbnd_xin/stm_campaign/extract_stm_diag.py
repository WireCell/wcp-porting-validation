#!/usr/bin/env python3
"""Extract per-bundle STM tagger diagnostics from stored pctree-pr tarballs.

For every bundle in scan-d59k/stm-baseline.tsv, open the d59k
pctree-pr-evt<ID>.tar.gz and pull, for the main cluster (ident == main_id):

  - cluster_scalar flags (flag_STM, flag_TGM),
  - the stm_pass records (pass/status/kink_num/exit_L/left_L/dqdx),
  - the stm_eval records (KS values, per-call verdicts),
  - trajectory-derived quantities from stm_fit (per pass):
      n points, total L, MAX consecutive-point 3D gap (multi-object smell),
      end-region dQ/dx: peak in last 5 cm rr / MIP, median of track body / MIP,
      median of last 5 cm rr,
  - the "no STM fit:" pre-fit exit reason from the log when no pass exists.

Output: one TSV row per (bundle, pass) plus a bundle-level summary TSV.

Usage: python3 extract_stm_diag.py [--root WORKROOT] [--baseline TSV] --out PREFIX
Defaults target the d59k production and the doc-62 baseline.
Read-only on the work root (M13).
"""
import argparse
import io
import json
import os
import re
import tarfile

import numpy as np

MIP = 56000.0  # e/cm, SBND production -mip 56000

AP = argparse.ArgumentParser()
AP.add_argument('--root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               '..', 'work-mcp1kall-d59k'))
AP.add_argument('--baseline', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   '..', 'scan-d59k', 'stm-baseline.tsv'))
AP.add_argument('--out', required=True, help='output prefix: writes PREFIX-pass.tsv, PREFIX-eval.tsv, PREFIX-bundle.tsv')
args = AP.parse_args()


def load_tar(fname):
    metas, arrays = {}, {}
    with tarfile.open(fname) as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if m.name.endswith('_metadata.json'):
                metas[m.name[:-len('_metadata.json')]] = json.load(f)
            elif m.name.endswith('_array.npy'):
                arrays[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(f.read()))
    return {md['datapath']: (md, arrays.get(base)) for base, md in metas.items()
            if 'datapath' in md}


def pc_blocks(byPath, evt, name):
    """Return {cluster_ident: {array_name: rows}} for a per-node named PC.

    lpcmaps/arrays/<name>[i] = number of rows node i contributes; the named PC
    concatenates the blocks in node order.  cluster idents come from the
    cluster_scalar PC, which sits one row per cluster node in the same order.
    """
    pre = f'pointtrees/{evt}/live/'
    lm = byPath.get(pre + f'lpcmaps/arrays/{name}')
    lm_cs = byPath.get(pre + 'lpcmaps/arrays/cluster_scalar')
    idents = byPath.get(pre + 'pointclouds/namedpcs/cluster_scalar/arrays/ident')
    if lm is None or lm_cs is None or idents is None:
        return {}
    counts, cs_counts, idents = lm[1], lm_cs[1], idents[1]
    arrays = {p.split('/')[-1]: arr for p, (md, arr) in byPath.items()
              if p.startswith(pre + f'pointclouds/namedpcs/{name}/arrays/')}
    if not arrays:
        return {}
    # node index -> cluster ident (nodes with a cluster_scalar row, in order)
    cs_nodes = np.nonzero(cs_counts)[0]
    node2ident = {int(n): int(idents[k]) for k, n in enumerate(cs_nodes)}
    out = {}
    off = 0
    for node in np.nonzero(counts)[0]:
        n = int(counts[node])
        ident = node2ident.get(int(node))
        blk = {k: v[off:off + n] for k, v in arrays.items()}
        if ident is not None:
            out.setdefault(ident, {k: [] for k in arrays})
            for k in arrays:
                out[ident][k].append(blk[k])
        off += n
    return {ident: {k: np.concatenate(v) for k, v in blks.items()}
            for ident, blks in out.items()}


def scalars(byPath, evt):
    pre = f'pointtrees/{evt}/live/pointclouds/namedpcs/cluster_scalar/arrays/'
    return {p[len(pre):]: arr for p, (md, arr) in byPath.items() if p.startswith(pre)}


def fit_diag(fit, sel, kink):
    """Trajectory diagnostics for one pass (boolean mask sel).

    The candidate STOPPING POINT is the kink (kink >= npts means the path
    end): eval_stm_core compares dQ/dx on [0, L[kink]] against the muon
    reference, so the Bragg region is the last cm BEFORE the kink, not the
    path end (a Michel/leftover residual continues past it).
    """
    x, y, z = fit['x'][sel], fit['y'][sel], fit['z'][sel]
    dQ, dx, L = fit['dQ'][sel], fit['dx'][sel], fit['L'][sel]
    n = len(x)
    if n == 0:
        return {}
    pts = np.stack([x, y, z], axis=1)
    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1) / 10.0  # internal->cm
    dqdx = dQ / (dx / 10.0 + 1e-9)   # e per cm
    Lcm = L / 10.0
    L_tot = float(Lcm[-1])
    k = min(int(kink), n - 1) if kink >= 0 else n - 1
    L_stop = float(Lcm[k])
    # distance-to-stop coordinate (only points on the muon side of the kink)
    d2s = L_stop - Lcm
    endw = (d2s >= 0) & (d2s < 5.0)
    bodyw = d2s > 10.0
    return dict(
        npts=n,
        L_cm=round(L_tot, 1),
        L_stop_cm=round(L_stop, 1),
        max_gap_cm=round(float(gaps.max()) if len(gaps) else 0.0, 2),
        end5_peak_mip=round(float(dqdx[endw].max() / MIP) if endw.any() else -1, 2),
        end5_med_mip=round(float(np.median(dqdx[endw]) / MIP) if endw.any() else -1, 2),
        body_med_mip=round(float(np.median(dqdx[bodyw]) / MIP) if bodyw.any() else -1, 2),
        neg_frac=round(float((dqdx < 0).mean()), 2),
    )


baseline = []
with open(args.baseline) as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.rstrip('\n').split('\t')
        if parts[0] == 'class':
            hdr = parts
            continue
        baseline.append(dict(zip(hdr, parts)))

pass_rows, eval_rows, bundle_rows = [], [], []
for b in baseline:
    evt, mid = b['event'], int(b['main_id'])
    tarball = os.path.join(args.root, f'nusel_evt{evt}', f'pctree-pr-evt{evt}.tar.gz')
    log = os.path.join(args.root, f'nusel_evt{evt}', f'wct_nusel_evt{evt}.log')
    if not os.path.exists(tarball):
        print(f'MISSING {tarball}')
        continue
    byPath = load_tar(tarball)
    cs = scalars(byPath, evt)
    if 'ident' not in cs or 'flag_STM' not in cs:
        print(f'evt{evt}: cluster_scalar incomplete (keys: {sorted(cs)})', flush=True)
        flag_stm = -1
    else:
        row = np.nonzero(cs['ident'] == mid)[0]
        flag_stm = int(cs['flag_STM'][row[0]]) if len(row) else -1

    passes = pc_blocks(byPath, evt, 'stm_pass').get(mid, {})
    evals = pc_blocks(byPath, evt, 'stm_eval').get(mid, {})
    fits = pc_blocks(byPath, evt, 'stm_fit').get(mid, {})

    prefit_exit = ''
    if not passes:
        if os.path.exists(log):
            with open(log, errors='replace') as f:
                for line in f:
                    m = re.search(rf'cluster {mid} no STM fit: (.*)$', line)
                    if m:
                        prefit_exit = m.group(1).strip()
    npass = len(passes.get('pass', []))
    for i in range(npass):
        pr = {k: passes[k][i] for k in passes}
        sel = fits['pass'] == pr['pass'] if fits else np.zeros(0, bool)
        fd = fit_diag(fits, sel, int(pr['kink_num'])) if fits else {}
        pass_rows.append(dict(
            event=evt, main_id=mid, cls=b['class'],
            pas=int(pr['pass']), status=int(pr['status']), kink=int(pr['kink_num']),
            exit_L_cm=round(float(pr['exit_L']) / 10.0, 1),
            left_L_cm=round(float(pr['left_L']) / 10.0, 1),
            exit_dqdx_mip=round(float(pr['exit_dqdx']) / MIP, 2),
            left_dqdx_mip=round(float(pr['left_dqdx']) / MIP, 2),
            **fd))
    for i in range(len(evals.get('pass', []))):
        er = {k: evals[k][i] for k in evals}
        eval_rows.append(dict(
            event=evt, main_id=mid, cls=b['class'],
            pas=int(er['pass']), strong=int(er['strong']), verdict=int(er['verdict']),
            peak_cm=round(float(er['peak_range']) / 10.0, 1),
            off_cm=round(float(er['offset_length']) / 10.0, 1),
            com_cm=round(float(er['com_range']) / 10.0, 1),
            ks1=round(float(er['ks1']), 4), ks2=round(float(er['ks2']), 4),
            ratio1=round(float(er['ratio1']), 3), ratio2=round(float(er['ratio2']), 3),
            res_len_cm=round(float(er['res_length']) / 10.0, 1),
            ave_res_mip=round(float(er['ave_res_dqdx']) / MIP, 2)))
    bundle_rows.append(dict(
        event=evt, main_id=mid, cls=b['class'], tagger=b['tagger'],
        owner=b['owner_verdict'], flag_stm=flag_stm, n_pass=npass,
        statuses=','.join(str(int(s)) for s in passes.get('status', [])),
        prefit_exit=prefit_exit))

for name, rows in (('pass', pass_rows), ('eval', eval_rows), ('bundle', bundle_rows)):
    if not rows:
        continue
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(f'{args.out}-{name}.tsv', 'w') as f:
        f.write('\t'.join(keys) + '\n')
        for r in rows:
            f.write('\t'.join(str(r.get(k, '')) for k in keys) + '\n')
    print(f'wrote {args.out}-{name}.tsv  ({len(rows)} rows)')
