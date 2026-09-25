#!/usr/bin/env python3
"""gnn_train.py -- the GNN ghost ablation (wcfm/docs/05 sec 3) on the blob-wire graphs of gnn_dataset.py.

One message-passing network (plain torch, no torch_geometric), trained twice with identical
architecture, folds, epochs and seeds:

  charge arm : blob node = the 15 charge summaries;            wire node = log1p(q), plane one-hot
  fm arm     : blob node = 15 charge + the 3 x 128 pooled FM;  wire node = charge arm + the 128-d FM row
               of that pixel (the FMFeatureExtract sidecar, doc 04)

Layers (L rounds): wire <- {mean, max} over the blobs covering it (+ log degree); blob <- mean over its
wires per plane (U, V, W kept separate) and mean over its BlobClustering neighbours (adjacent slices);
residual MLP updates, LayerNorm, hidden h.  Readout: one logit per blob, P(ghost).  The charge arm is
therefore a learned charge solver on the solver's own factor graph; the fm arm adds only the FM inputs.

Protocol: 5 folds over EVENTS (stratified by kind = iso/cosmic x n_tracks, fixed assignment for all arms
and seeds); for outer fold k the next training fold is the inner validation fold used to pick the epoch
(best validation AP on the hard population); test scores of the 5 folds are pooled.  3 seeds per arm.

Metrics (ghost = positive class, as in crossview_probe.py): pooled held-out AP and AUC on
  * all cells,
  * the HARD population = cells the legacy chain decides wrongly (kept ghosts + dropped real cells),
  * iso and cosmic events separately,
and the legacy chain's own precision/recall as the reference operating point.

PRE-REGISTERED (written before the first run): the fm arm beats the charge arm when, over the 3 seeds,
its mean pooled held-out AP on the HARD population is higher by >= +0.02 AND the gap exceeds twice the
larger of the two seed standard deviations.  Anything else is NO-GO for F4 on these features.

    python3 scripts/gnn_train.py --graphs /home/xqian/tmp/wcfm-gnn/graphs --out /home/xqian/tmp/wcfm-gnn/ablation \
        --arms charge,fm --seeds 0,1,2 --device cuda:0
"""
import argparse
import glob
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MARGIN_AP = 0.02
FOLD_SEED = 0


def labels_of(qtrue, spec):
    """ghost label (1) / real (0) / -1 = ignored, from the BlobDepoFill true charge.
    tru0        : real <=> q_true > 0 (doc 05 sec 3; 94 % of those cells hold < 1000 e of diffusion tail)
    qmin:<Q>    : real <=> q_true >= Q, ghost <=> q_true < Q
    qmin:<Q>:ig : real <=> q_true >= Q, ghost <=> q_true == 0, cells with 0 < q_true < Q ignored in loss and metrics"""
    if spec == 'tru0':
        return (qtrue <= 0).astype(np.int8)
    parts = spec.split(':')
    if parts[0] != 'qmin':
        raise SystemExit(f'unknown label spec {spec}')
    Q = float(parts[1]); y = (qtrue < Q).astype(np.int8)
    if len(parts) > 2 and parts[2] == 'ig':
        y[(qtrue > 0) & (qtrue < Q)] = -1
    return y


def load_graphs(gdir, need_fm, label='tru0'):
    """raw arrays stay on the CPU (FM blocks in float16); each step standardises one graph on the GPU."""
    graphs = []
    for fn in sorted(glob.glob(os.path.join(gdir, 'graph-*.npz'))):
        z = np.load(fn)
        if len(z['b_y']) < 2 or len(z['bw_src']) == 0:
            continue
        lab = labels_of(z['b_qtrue'].astype(np.float64), label)
        if label == 'tru0':
            assert (lab == z['b_y']).all()
        onehot = np.eye(3, dtype=np.float32)[z['w_plane']]
        bb = z['bb'].astype(np.int64)
        bb2 = np.concatenate([bb, bb[:, ::-1]]) if len(bb) else np.zeros((0, 2), np.int64)
        g = dict(name=os.path.basename(fn)[6:-4], event=int(z['event']), anode=int(z['anode']),
                 kind=str(z['kind']), ntracks=int(z['ntracks']),
                 b_charge=z['b_charge'].astype(np.float32), b_fm=z['b_fm'] if need_fm else None,
                 y=np.clip(lab, 0, 1).astype(np.float32), mask=lab >= 0, qtrue=z['b_qtrue'].astype(np.float64),
                 kept=z['b_kept'].astype(np.int8), present=z['b_present'].astype(np.int8),
                 w_plane=torch.from_numpy(z['w_plane'].astype(np.int64)),
                 w_base=np.concatenate([np.log1p(z['w_q'].astype(np.float32))[:, None], onehot], axis=1),
                 w_fm=z['w_fm'] if need_fm else None,
                 bw_src=torch.from_numpy(z['bw_src'].astype(np.int64)), bw_dst=torch.from_numpy(z['bw_dst'].astype(np.int64)),
                 bb_src=torch.from_numpy(np.ascontiguousarray(bb2[:, 0])), bb_dst=torch.from_numpy(np.ascontiguousarray(bb2[:, 1])),
                 y_t=torch.from_numpy(np.clip(lab, 0, 1).astype(np.float32)), mask_t=torch.from_numpy(lab >= 0))
        graphs.append(g)
    return graphs


def make_folds(graphs, nfold, fold_seed=FOLD_SEED):
    """event -> fold, stratified by kind+ntracks, fixed seed."""
    by_class = defaultdict(list)
    for g in graphs:
        by_class[(g['kind'], g['ntracks'])].append(g['event'])
    rng = np.random.RandomState(fold_seed)
    fold = {}
    k = 0
    for cls in sorted(by_class):
        evs = sorted(set(by_class[cls]))
        rng.shuffle(evs)
        for e in evs:
            fold[e] = k % nfold; k += 1
    return fold


class MLP(nn.Module):
    def __init__(self, din, h):
        super().__init__()
        self.l1 = nn.Linear(din, h); self.l2 = nn.Linear(h, h); self.ln = nn.LayerNorm(h)

    def forward(self, x):
        return self.ln(self.l2(F.relu(self.l1(x))))


class FactorGNN(nn.Module):
    def __init__(self, db, dw, h=64, L=3):
        super().__init__()
        self.bin = nn.Linear(db, h); self.win = nn.Linear(dw, h)
        self.wupd = nn.ModuleList(MLP(3 * h + 1, h) for _ in range(L))
        self.bupd = nn.ModuleList(MLP(5 * h, h) for _ in range(L))
        self.out = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.L = L

    def forward(self, xb, xw, bw_src, bw_dst, bb_src, bb_dst, w_plane):
        N, M, h = xb.shape[0], xw.shape[0], self.bin.out_features
        hb = F.relu(self.bin(xb)); hw = F.relu(self.win(xw))
        one = torch.ones(bw_src.shape[0], device=xb.device)
        deg_w = torch.zeros(M, device=xb.device).index_add_(0, bw_dst, one).clamp(min=1)
        logdeg = torch.log1p(deg_w).unsqueeze(1)
        plane_of_edge = w_plane[bw_dst]
        deg_bp = [torch.zeros(N, device=xb.device).index_add_(0, bw_src[plane_of_edge == p], one[plane_of_edge == p]).clamp(min=1)
                  for p in range(3)]
        deg_bb = torch.zeros(N, device=xb.device).index_add_(0, bb_src, torch.ones(bb_src.shape[0], device=xb.device)).clamp(min=1)
        idx_w = bw_dst.unsqueeze(1).expand(-1, h)
        for l in range(self.L):
            src = hb[bw_src]
            msum = torch.zeros(M, h, device=xb.device).index_add_(0, bw_dst, src)
            mmax = torch.zeros(M, h, device=xb.device).scatter_reduce_(0, idx_w, src, 'amax', include_self=False)
            hw = hw + self.wupd[l](torch.cat([hw, msum / deg_w.unsqueeze(1), mmax, logdeg], dim=1))
            msgs = [hb]
            wsrc = hw[bw_dst]
            for p in range(3):
                m = plane_of_edge == p
                acc = torch.zeros(N, h, device=xb.device).index_add_(0, bw_src[m], wsrc[m])
                msgs.append(acc / deg_bp[p].unsqueeze(1))
            accbb = torch.zeros(N, h, device=xb.device).index_add_(0, bb_src, hb[bb_dst])
            msgs.append(accbb / deg_bb.unsqueeze(1))
            hb = hb + self.bupd[l](torch.cat(msgs, dim=1))
        return self.out(hb).squeeze(1)


def inputs(g, arm):
    """raw (blob, wire) float32 input matrices of one graph."""
    xb = g['b_charge'] if arm == 'charge' else np.concatenate([g['b_charge'], g['b_fm'].astype(np.float32)], axis=1)
    xw = g['w_base'] if arm == 'charge' else np.concatenate([g['w_base'], g['w_fm'].astype(np.float32)], axis=1)
    return xb, xw


class Standardiser:
    """per-feature mean/std over the training graphs (streaming sums, nothing concatenated)."""
    def __init__(self, graphs, arm, dev):
        sb = ss = sw = sww = None; nb = nw = 0
        for g in graphs:
            xb, xw = inputs(g, arm)
            xb = xb.astype(np.float64); xw = xw.astype(np.float64)
            if sb is None:
                sb, ss, sw, sww = xb.sum(0), (xb * xb).sum(0), xw.sum(0), (xw * xw).sum(0)
            else:
                sb += xb.sum(0); ss += (xb * xb).sum(0); sw += xw.sum(0); sww += (xw * xw).sum(0)
            nb += len(xb); nw += len(xw)
        mb = sb / nb; mw = sw / nw
        vb = np.maximum(ss / nb - mb * mb, 0); vw = np.maximum(sww / nw - mw * mw, 0)
        self.mb = torch.tensor(mb, dtype=torch.float32, device=dev); self.sb = torch.tensor(np.sqrt(vb) + 1e-6, dtype=torch.float32, device=dev)
        self.mw = torch.tensor(mw, dtype=torch.float32, device=dev); self.sw = torch.tensor(np.sqrt(vw) + 1e-6, dtype=torch.float32, device=dev)
        self.arm = arm; self.dev = dev

    def to_torch(self, g):
        """one graph -> standardised GPU tensors (built per step; the raw arrays stay on the CPU)."""
        dev = self.dev
        xb = torch.from_numpy(g['b_charge']).to(dev, non_blocking=True)
        xw = torch.from_numpy(g['w_base']).to(dev, non_blocking=True)
        if self.arm == 'fm':
            xb = torch.cat([xb, torch.from_numpy(g['b_fm']).to(dev, non_blocking=True).float()], dim=1)
            xw = torch.cat([xw, torch.from_numpy(g['w_fm']).to(dev, non_blocking=True).float()], dim=1)
        return dict(xb=(xb - self.mb) / self.sb, xw=(xw - self.mw) / self.sw,
                    bw_src=g['bw_src'].to(dev, non_blocking=True), bw_dst=g['bw_dst'].to(dev, non_blocking=True),
                    bb_src=g['bb_src'].to(dev, non_blocking=True), bb_dst=g['bb_dst'].to(dev, non_blocking=True),
                    w_plane=g['w_plane'].to(dev, non_blocking=True), y=g['y_t'].to(dev, non_blocking=True),
                    mask=g['mask_t'].to(dev, non_blocking=True))


def ap_auc(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(y) == 0 or y.min() == y.max():
        return float('nan'), float('nan')
    return float(average_precision_score(y, s)), float(roc_auc_score(y, s))


def hard_mask(g):
    return (((g['kept'] == 1) & (g['y'] == 1)) | ((g['kept'] == 0) & (g['y'] == 0))) & g['mask']


@torch.no_grad()
def predict(model, tg):
    model.eval()
    return model(tg['xb'], tg['xw'], tg['bw_src'], tg['bw_dst'], tg['bb_src'], tg['bb_dst'], tg['w_plane']).float().cpu().numpy()


def pooled_ap(graphs, std, model, hard=True):
    ys, ss = [], []
    for g in graphs:
        s = predict(model, std.to_torch(g)); m = hard_mask(g) if hard else g['mask']
        ys.append(g['y'][m]); ss.append(s[m])
    return ap_auc(np.concatenate(ys), np.concatenate(ss))[0]


def train_fold(graphs, fold, k, arm, seed, a, dev, log):
    nf = a.folds
    test = [g for g in graphs if fold[g['event']] == k]
    val = [g for g in graphs if fold[g['event']] == (k + 1) % nf]
    train = [g for g in graphs if fold[g['event']] not in (k, (k + 1) % nf)]
    torch.manual_seed(seed); np.random.seed(seed)
    std = Standardiser(train, arm, dev)
    model = FactorGNN(len(std.mb), len(std.mw), a.hidden, a.layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    rng = np.random.RandomState(seed)
    best, best_state, best_ep = -1.0, None, -1
    for ep in range(a.epochs):
        model.train(); order = rng.permutation(len(train)); tot = 0.0
        for i in order:
            tg = std.to_torch(train[i])
            logit = model(tg['xb'], tg['xw'], tg['bw_src'], tg['bw_dst'], tg['bb_src'], tg['bb_dst'], tg['w_plane'])
            loss = F.binary_cross_entropy_with_logits(logit[tg['mask']], tg['y'][tg['mask']])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
            del tg, logit, loss
        vap = pooled_ap(val, std, model, hard=True)
        if vap > best:
            best, best_ep = vap, ep; best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
        log(f'  arm={arm} seed={seed} fold={k} ep={ep} loss={tot / len(train):.4f} val_hardAP={vap:.4f} '
            f'gpu_peak_GB={torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else 0:.1f}')
    model.load_state_dict(best_state)
    out = {}
    for g in test:
        out[g['name']] = predict(model, std.to_torch(g))
    return out, best_ep, best


def summarise(graphs, scores):
    """scores: name -> per-blob score. -> metrics dict."""
    res = {}
    def pool(sel, hard):
        ys, ss = [], []
        for g in graphs:
            if not sel(g): continue
            s = scores[g['name']]; m = hard_mask(g) if hard else g['mask']
            ys.append(g['y'][m]); ss.append(s[m])
        if not ys:
            return dict(n=0, ghost_fraction=float('nan'), ap=float('nan'), auc=float('nan'))
        y = np.concatenate(ys); s = np.concatenate(ss); apv, auc = ap_auc(y, s)
        return dict(n=int(len(y)), ghost_fraction=float(y.mean()), ap=apv, auc=auc)
    for label, sel in (('all', lambda g: True), ('iso', lambda g: g['kind'] == 'iso'), ('cosmic', lambda g: g['kind'] == 'cosmic')):
        res[label] = pool(sel, False); res[label + '_hard'] = pool(sel, True)
    per_ev = defaultdict(lambda: ([], []))
    for g in graphs:
        per_ev[g['event']][0].append(g['y'][g['mask']]); per_ev[g['event']][1].append(scores[g['name']][g['mask']])
    aps = [ap_auc(np.concatenate(a), np.concatenate(b))[0] for a, b in per_ev.values()]
    res['per_event_ap_median'] = float(np.nanmedian(aps)); res['per_event_ap_p10'] = float(np.nanpercentile(aps, 10))
    return res


def solver_reference(graphs):
    y = np.concatenate([g['y'][g['mask']] for g in graphs]); kept = np.concatenate([g['kept'][g['mask']] for g in graphs])
    real = y == 0
    tp = int((kept == 1)[real].sum()); fp = int((kept == 1)[~real].sum()); fn = int((kept == 0)[real].sum())
    return dict(n=int(len(y)), real=int(real.sum()), kept=int(kept.sum()), tp=tp, fp=fp, fn=fn,
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), hard=int(fp + fn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--arms', default='charge,fm'); ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--folds', type=int, default=5); ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--hidden', type=int, default=64); ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--merge', nargs='+', metavar='DIR', help='combine the arms of these run dirs into --out and stop')
    ap.add_argument('--label', default='tru0', help='tru0 | qmin:<Q> | qmin:<Q>:ig (see labels_of)')
    ap.add_argument('--fold-seed', type=int, default=FOLD_SEED)
    a = ap.parse_args()
    if a.merge:
        return merge(a.merge, a.out)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'train.log'), 'a')
    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()
    dev = torch.device(a.device)
    graphs = load_graphs(a.graphs, need_fm='fm' in a.arms.split(','), label=a.label)
    fold = make_folds(graphs, a.folds, a.fold_seed)
    ev = sorted(set(g['event'] for g in graphs))
    log(f'[gnn] {len(graphs)} graphs, {len(ev)} events, blobs {sum(len(g["y"]) for g in graphs)}, label {a.label}: '
        f'labelled {sum(int(g["mask"].sum()) for g in graphs)}, ghost fraction {np.concatenate([g["y"][g["mask"]] for g in graphs]).mean():.3f}, '
        f'fold seed {a.fold_seed}, folds {[sum(1 for e in ev if fold[e] == k) for k in range(a.folds)]}')
    ref = solver_reference(graphs)
    log(f'[gnn] legacy chain reference: {ref}')
    result = dict(args=vars(a), margin_ap=MARGIN_AP, solver=ref, fold={int(e): int(f) for e, f in fold.items()},
                  n_graphs=len(graphs), n_events=len(ev), arms={})
    for arm in a.arms.split(','):
        result['arms'][arm] = {}
        for seed in (int(s) for s in a.seeds.split(',')):
            t0 = time.time(); scores = {}; eps = []
            for k in range(a.folds):
                sc, bep, bval = train_fold(graphs, fold, k, arm, seed, a, dev, log)
                scores.update(sc); eps.append((bep, bval))
            met = summarise(graphs, scores)
            met['best_epochs'] = eps; met['wall_s'] = time.time() - t0
            result['arms'][arm][seed] = met
            np.savez_compressed(os.path.join(a.out, f'scores_{arm}_s{seed}.npz'), **scores)
            log(f'[gnn] arm={arm} seed={seed}: all AP {met["all"]["ap"]:.4f} AUC {met["all"]["auc"]:.4f} | hard AP {met["all_hard"]["ap"]:.4f} '
                f'AUC {met["all_hard"]["auc"]:.4f} | iso hard AP {met["iso_hard"]["ap"]:.4f} cosmic hard AP {met["cosmic_hard"]["ap"]:.4f} ({met["wall_s"]:.0f} s)')
            with open(os.path.join(a.out, 'ablation_result.json'), 'w') as f:
                json.dump(result, f, indent=1)
    finish(result, a.out, log)


def finish(result, outdir, log):
    ref = result['solver']
    def stat(arm, key):
        v = np.array([m[key]['ap'] for m in result['arms'][arm].values()]); return float(v.mean()), float(v.std())
    if 'charge' in result['arms'] and 'fm' in result['arms']:
        mc, sc = stat('charge', 'all_hard'); mf, sf = stat('fm', 'all_hard')
        gap = mf - mc
        result['verdict'] = dict(charge_hard_ap=mc, charge_sd=sc, fm_hard_ap=mf, fm_sd=sf, gap=gap,
                                 go=bool(gap >= MARGIN_AP and gap > 2 * max(sc, sf)))
        log(f'[gnn] hard-population AP: charge {mc:.4f}+-{sc:.4f}  fm {mf:.4f}+-{sf:.4f}  gap {gap:+.4f}  margin {MARGIN_AP:+.2f} -> '
            f'{"GO" if result["verdict"]["go"] else "NO-GO"}')
    with open(os.path.join(outdir, 'ablation_result.json'), 'w') as f:
        json.dump(result, f, indent=1)
    rows = [('all', 'all'), ('all_hard', 'all, hard'), ('iso', 'iso'), ('iso_hard', 'iso, hard'), ('cosmic', 'cosmic'), ('cosmic_hard', 'cosmic, hard')]
    with open(os.path.join(outdir, 'ablation_summary.md'), 'w') as f:
        arms = list(result['arms'])
        f.write('| population | n | ghost frac | ' + ' | '.join(f'{arm} AP (mean ± sd over seeds) | {arm} AUC' for arm in arms) + ' |\n')
        f.write('|---|---|---|' + '---|---|' * len(arms) + '\n')
        for key, label in rows:
            m0 = next(iter(result['arms'][arms[0]].values()))[key]
            cells = []
            for arm in arms:
                aps = np.array([m[key]['ap'] for m in result['arms'][arm].values()]); aucs = np.array([m[key]['auc'] for m in result['arms'][arm].values()])
                cells.append(f'{aps.mean():.4f} ± {aps.std():.4f} | {aucs.mean():.4f}')
            f.write(f'| {label} | {m0["n"]} | {m0["ghost_fraction"]:.3f} | ' + ' | '.join(cells) + ' |\n')
        f.write(f'\nlegacy chain on all cells: precision {ref["precision"]:.3f} recall {ref["recall"]:.3f} '
                f'(kept {ref["kept"]} of {ref["n"]}, real {ref["real"]}, wrong {ref["hard"]})\n')
    log(f'[gnn] wrote {outdir}/ablation_result.json, ablation_summary.md')


def merge(dirs, outdir):
    """--merge: combine the arms of several runs (same graphs, folds, seeds) into one verdict."""
    os.makedirs(outdir, exist_ok=True)
    result = None
    for d in dirs:
        r = json.load(open(os.path.join(d, 'ablation_result.json')))
        if result is None:
            result = r
        else:
            assert r['fold'] == result['fold'] and r['n_graphs'] == result['n_graphs'], 'runs differ in graphs or folds'
            result['arms'].update(r['arms'])
    finish(result, outdir, lambda s: print(s, flush=True))



if __name__ == '__main__':
    main()
