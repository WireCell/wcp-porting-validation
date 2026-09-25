#!/usr/bin/env python3
"""crossview_probe.py -- the Phase-0 cross-view probe (wcfm/docs/01 sec 4.5 / sec 7 F5, docs/04 sec 4) on
the fine-level labelled tier: the BlobCutting sub-blobs of work/000001_*_sub/clusters-tru0-anode<N>-ms-active
(doc 03 sec 6; ghost <=> BlobDepoFill true charge == 0) joined to the FM features of the same SP frames.

Per sub-blob (face, slice, U/V/W strips) the wire nodes of the cluster file give every strip wire's
channel ((planeid, wire index) -> channel; all 1.94M strip wires of the manifest are covered), the FM
feature rows are looked up at (channel, slice) in the sidecar (C++ FMFeatureExtract tar.gz or the F2
oracle npz -- same layout), and per view the 128-d rows are mean- and max-pooled.  From those:

  * zero-shot tri-view disagreement: pairwise cosine distances of the three mean-pooled vectors
    (mean and variance, WC-PR-ML-Ideas sec 9.1) -- no label used;
  * charge-only baseline: per view n_channels, n_pixels, sum q, mean q (+ pairwise log charge ratios);
  * FM probe: charge-only features + the 3 x 128 mean-pooled descriptors (+ max-pooled with --max-pool).

Classifier = standardised logistic regression (and a small MLP with --mlp), leave-one-EVENT-out over
the manifest; metric = average precision of the ghost class on the pooled held-out predictions (and
per event).  PRE-REGISTERED margin (written before the first run): the FM probe beats the charge-only
baseline by >= +0.05 pooled held-out AP.

    python3 scripts/crossview_probe.py --features oracle:/home/xqian/tmp/wcfm-f3 --out /home/xqian/tmp/wcfm-f3/probe work/000001_*_sub
    python3 scripts/crossview_probe.py --features sidecar:_fmf4 ...      # the C++ sidecar in work/<evt>_fmf4/
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from wirecell.util import ario

WCFM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYER = {0: 1, 1: 2, 2: 4}
PLANE_NAME = {0: 'U', 1: 'V', 2: 'W'}
PLANE_CH = {0: (0, 800), 1: (800, 1600), 2: (1600, 2560)}
COLS = dict(desc=0, ident=1, val=2, unc=3, faceid=4, sliceid=5, start=6, span=7, u0=8, u1=9, v0=10, v1=11, w0=12, w1=13)
MARGIN_AP = 0.05


def load_features(spec, base, anode):
    """-> {plane: (dict (channel, slice) -> row, feat (N,128))}"""
    kind, arg = spec.split(':', 1)
    out = {}
    if kind == 'oracle':
        z = np.load(os.path.join(arg, f'oracle-{base}-anode{anode}.npz'))
        for p, name in PLANE_NAME.items():
            c = z[f'coords_{name}']; f = z[f'feat_{name}']
            out[p] = ({(int(a), int(b)): i for i, (a, b) in enumerate(c)}, f)
    elif kind == 'sidecar':
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from fm_parity import read_sidecar
        per, _md = read_sidecar(os.path.join(WCFM_DIR, 'work', base + arg, f'fm-features-anode{anode}.tar.gz'))
        for p, d in per.items():
            out[p] = ({(int(a), int(b)): i for i, (a, b) in enumerate(d['coords'])}, d['feat'])
    else:
        sys.exit(f'unknown features spec {spec}')
    return out


def load_charge(base, anode, scale=0.25):
    """(channel - anode*2560, slice) -> pixel charge, from the SP frames (the oracle packing)."""
    fn = os.path.join(WCFM_DIR, 'work', base, f'sim-frames-anode{anode}.tar.bz2')
    f = ario.load(fn)
    k = [k for k in f.keys() if k.startswith('frame_gauss')][0]
    arr = np.asarray(f[k], dtype=np.float32)
    ch = np.asarray(f[k.replace('frame', 'channels')])
    q = arr.reshape(arr.shape[0], -1, 4).astype(np.float64).sum(axis=2) * scale
    full = np.zeros((2560, q.shape[1]), np.float32)
    full[ch - anode * 2560] = q
    return full


def blobs_of(workdir, anode, feats, qmap, max_pool):
    fn = os.path.join(workdir, f'clusters-tru0-anode{anode}-ms-active.tar.gz')
    f = ario.load(fn)
    rows = []
    for k in f.keys():
        if not k.endswith('_bnodes'):
            continue
        pre = k[:-len('bnodes')]
        b = f[k]; w = f[pre + 'wnodes']
        wmap = {(int(r[5]), int(r[2])): int(r[4]) for r in w}
        for r in b:
            fid = int(r[COLS['faceid']]); a = fid >> 4; face = (fid >> 3) & 1
            s = int(r[COLS['sliceid']])
            rec = dict(anode=a, face=face, slice=s, ident=int(r[COLS['ident']]), q_true=float(r[COLS['val']]),
                       ghost=int(r[COLS['val']] == 0))
            pooled_mean, pooled_max = [], []
            for p, (lo, hi) in enumerate(((8, 9), (10, 11), (12, 13))):
                pid = (a << 4) | (face << 3) | LAYER[p]
                chans = sorted({wmap[(pid, i)] for i in range(int(r[lo]), int(r[hi]))})
                idx, feat = feats[p]
                rows_p = [idx[(c, s)] for c in chans if (c, s) in idx]
                qs = [float(qmap[c - a * 2560, s]) for c in chans]
                rec[f'nch_{p}'] = len(chans)
                rec[f'npix_{p}'] = len(rows_p)
                rec[f'sumq_{p}'] = float(sum(qs))
                rec[f'meanq_{p}'] = float(np.mean(qs)) if qs else 0.0
                if rows_p:
                    F = feat[rows_p]
                    pooled_mean.append(F.mean(axis=0)); pooled_max.append(F.max(axis=0))
                else:
                    pooled_mean.append(np.zeros(feat.shape[1], np.float32)); pooled_max.append(np.zeros(feat.shape[1], np.float32))
            rec['mean'] = np.stack(pooled_mean); rec['max'] = np.stack(pooled_max)
            rows.append(rec)
    return rows


def cosdist(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def build_matrix(recs, max_pool):
    charge, fm, y, ev, dis_mean, dis_var = [], [], [], [], [], []
    for r in recs:
        c = []
        for p in range(3):
            c += [r[f'nch_{p}'], r[f'npix_{p}'], np.log1p(r[f'sumq_{p}']), np.log1p(r[f'meanq_{p}'])]
        for (i, j) in ((0, 1), (0, 2), (1, 2)):
            c.append(np.log1p(r[f'sumq_{i}']) - np.log1p(r[f'sumq_{j}']))
        charge.append(c)
        v = [r['mean'].reshape(-1)]
        if max_pool:
            v.append(r['max'].reshape(-1))
        fm.append(np.concatenate(v))
        y.append(r['ghost']); ev.append(r['event'])
        d = [cosdist(r['mean'][i], r['mean'][j]) for (i, j) in ((0, 1), (0, 2), (1, 2))]
        dis_mean.append(np.mean(d)); dis_var.append(np.var(d))
    return (np.array(charge, np.float64), np.array(fm, np.float64), np.array(y), np.array(ev),
            np.array(dis_mean), np.array(dis_var))


def loeo(X, y, ev, make_model):
    """leave-one-event-out; returns pooled held-out scores."""
    from sklearn.preprocessing import StandardScaler
    scores = np.zeros(len(y))
    for e in sorted(set(ev)):
        tr = ev != e; te = ev == e
        if y[tr].min() == y[tr].max():
            scores[te] = 0.5; continue
        sc = StandardScaler().fit(X[tr])
        m = make_model().fit(sc.transform(X[tr]), y[tr])
        scores[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return scores


def ap_of(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score
    if y.min() == y.max():
        return float('nan'), float('nan')
    return float(average_precision_score(y, s)), float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('workdirs', nargs='+', help='work/000001_<evt>_sub dirs (tru0 tier)')
    ap.add_argument('--features', required=True, help='oracle:<dir> | sidecar:<suffix>')
    ap.add_argument('--max-pool', action='store_true')
    ap.add_argument('--mlp', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    os.makedirs(args.out, exist_ok=True)

    recs = []
    for wd in args.workdirs:
        sub = os.path.basename(wd.rstrip('/'))
        base = sub.replace('_sub', '')
        evt = int(base.split('_')[1])
        anodes = sorted(int(os.path.basename(f).split('anode')[1].split('-')[0])
                        for f in glob.glob(os.path.join(wd, 'clusters-tru0-anode*-ms-active.tar.gz')))
        for a in anodes:
            feats = load_features(args.features, base, a)
            qmap = load_charge(base, a)
            rs = blobs_of(wd, a, feats, qmap, args.max_pool)
            for r in rs:
                r['event'] = evt
            recs += rs
            nzero = sum(1 for r in rs if min(r['npix_0'], r['npix_1'], r['npix_2']) == 0)
            print(f'[probe] {sub} anode {a}: {len(rs)} sub-blobs, ghosts {sum(r["ghost"] for r in rs)}, '
                  f'views with no FM pixel in {nzero} blobs', flush=True)

    charge, fm, y, ev, dmean, dvar = build_matrix(recs, args.max_pool)
    print(f'[probe] total {len(y)} sub-blobs, ghost fraction {y.mean():.3f}, events {sorted(set(ev))}', flush=True)
    res = dict(n=int(len(y)), ghost_fraction=float(y.mean()), margin_ap=MARGIN_AP, features=args.features,
               max_pool=args.max_pool, per_event={}, pooled={})
    logit = lambda: LogisticRegression(max_iter=2000, C=1.0)
    models = {'charge_only': (charge, logit), 'fm_plus_charge': (np.concatenate([charge, fm], axis=1), logit),
              'fm_only': (fm, logit)}
    if args.mlp:
        mlp = lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=0)
        models['fm_plus_charge_mlp'] = (np.concatenate([charge, fm], axis=1), mlp)
        models['charge_only_mlp'] = (charge, mlp)
    scores = {'zero_shot_disagreement_mean': dmean, 'zero_shot_disagreement_var': dvar}
    for name, (X, mk) in models.items():
        scores[name] = loeo(X, y, ev, mk)
        print(f'[probe] {name}: done', flush=True)
    for name, s in scores.items():
        apv, auc = ap_of(y, s)
        res['pooled'][name] = dict(ap=apv, auc=auc)
        print(f'[probe] pooled {name}: AP={apv:.4f} AUC={auc:.4f}', flush=True)
    for e in sorted(set(ev)):
        m = ev == e
        res['per_event'][int(e)] = {'n': int(m.sum()), 'ghost_fraction': float(y[m].mean())}
        for name, s in scores.items():
            apv, auc = ap_of(y[m], s[m])
            res['per_event'][int(e)][name] = dict(ap=apv, auc=auc)
    d_ap = res['pooled']['fm_plus_charge']['ap'] - res['pooled']['charge_only']['ap']
    res['delta_ap_fm_plus_charge_vs_charge'] = d_ap
    res['pass_margin'] = bool(d_ap >= MARGIN_AP)
    print(f'[probe] delta AP (fm+charge - charge) = {d_ap:+.4f}  margin {MARGIN_AP:+.2f} -> {"GO" if res["pass_margin"] else "NO-GO"}', flush=True)
    with open(os.path.join(args.out, 'probe_result.json'), 'w') as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(os.path.join(args.out, 'probe_scores.npz'), y=y, event=ev, **scores)
    with open(os.path.join(args.out, 'probe_summary.md'), 'w') as f:
        f.write(f'| score | pooled AP | pooled AUC |' + ''.join(f' ev{e} AP |' for e in sorted(set(ev))) + '\n')
        f.write('|---|---|---|' + '---|' * len(set(ev)) + '\n')
        for name in scores:
            f.write(f"| {name} | {res['pooled'][name]['ap']:.3f} | {res['pooled'][name]['auc']:.3f} |"
                    + ''.join(f" {res['per_event'][e][name]['ap']:.3f} |" for e in sorted(set(ev))) + '\n')
        f.write(f"| prevalence (ghost fraction) | {y.mean():.3f} | |" + ''.join(f" {res['per_event'][e]['ghost_fraction']:.3f} |" for e in sorted(set(ev))) + '\n')
    print(f'[probe] wrote {args.out}/probe_result.json, probe_summary.md')


if __name__ == '__main__':
    main()
