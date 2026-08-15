#!/usr/bin/env python3
'''
doc pr/77 -- fine-tune DeepVtx on hand-scan vertex labels.

Resumes the uBooNE t48k conventions (uboone-dl-vtx train1.py): truth is a
Gaussian falloff around the true vertex, loss is MSE on
prediction[:,1]-prediction[:,0] (the same sigmoid-difference score used at
inference), Adam with exponential LR decay, per-event batches, a checkpoint
per epoch.  Fine-tuning deltas: lr0 defaults 1e-6 (10x below the original
1e-5), and a freeze schedule protects the backbone on small samples.

Usage (practice run):
  python3 train.py --data data/practice66 --name ft0 --kfold 6 --epochs 30

Checkpoints land in runs/<name>/fold<k>/CP<E>.pth as squeezed-3D
state_dicts, loadable both by scn_vtx.model.load_weights and by the in-tree
SCN_Vertex._load_model (deployment = copy to wire-cell-data under a NEW name
+ dl_weights TLA; never overwrite the uBooNE file).
'''
import argparse
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from scn_vtx.model import make_model, load_weights
from scn_vtx.voxelize import top_k_voxels
from dataset import VtxSamples, load_manifest, kfold_split


def trainable_names(model, mode):
    """Which parameters train.  Default 'head': linear + final BatchNormReLU
    (sparseModel.3) + the LAST top-level block of the UNet (the full-
    resolution decoder tail) -- the small-sample setting from doc pr/52
    S2.2.1.  'linear' = linear probe.  'none' = everything trains."""
    names = [n for n, _ in model.named_parameters()]
    if mode == 'none':
        return set(names)
    if mode == 'linear':
        return {n for n in names if n.startswith('linear')}
    if mode == 'head':
        unet_top = sorted({int(n.split('.')[2]) for n in names
                           if n.startswith('sparseModel.2.')})
        last = unet_top[-1] if unet_top else None
        keep = {n for n in names if n.startswith('linear')
                or n.startswith('sparseModel.3')
                or (last is not None and n.startswith('sparseModel.2.%d' % last))}
        return keep
    raise ValueError('unknown freeze mode %r' % mode)


def set_freeze(model, mode):
    keep = trainable_names(model, mode)
    n_train = 0
    for n, p in model.named_parameters():
        p.requires_grad = n in keep
        n_train += p.numel() if n in keep else 0
    total = sum(p.numel() for p in model.parameters())
    return n_train, total


def bn_stats_snapshot(model):
    """round 3 (doc pr/78): tiny clouds can leave a single active site at the
    deepest UNet level; the unbiased batch variance is then 0/0 and the BN
    running stats go NaN -- train-mode forwards stay finite (batch stats) but
    every eval-mode forward NaNs.  SCN's BN autograd asserts train mode in
    backward, so eval-mode BN training is not an option; instead we snapshot
    the pretrained running stats and restore them after every training epoch
    (--bn-freeze): batch-stat training exactly as before, frozen stats for
    every eval and every saved checkpoint."""
    snap = {}
    for name, m in model.named_modules():
        for attr in ('running_mean', 'running_var'):
            t = getattr(m, attr, None)
            if t is not None:
                snap[(name, attr)] = t.detach().clone()
    return snap


def bn_stats_restore(model, snap):
    with torch.no_grad():
        for name, m in model.named_modules():
            for attr in ('running_mean', 'running_var'):
                t = getattr(m, attr, None)
                if t is not None:
                    t.copy_(snap[(name, attr)])


def d_argmax_cm(pred, coords, offset, truth):
    top = top_k_voxels(pred, coords, offset, k=1)
    return float(np.linalg.norm(top[0, :3] - truth))


def run_epoch(model, samples, device, criterion, optimizer=None,
              consistency=0.0, frozen=None):
    train = optimizer is not None
    model.train(train)
    losses, dists = [], []
    order = np.random.permutation(len(samples)) if train else range(len(samples))
    for k in order:
        coords_np, ft_np, target_np, meta = samples[k]
        coords = torch.LongTensor(coords_np)
        ft = torch.FloatTensor(ft_np).to(device)
        target = torch.FloatTensor(target_np).to(device)
        if train:
            optimizer.zero_grad()
            prediction = model([coords, ft])
            score = prediction[:, 1] - prediction[:, 0]
            # doc pr/81 B: --dense-weight scales the legacy dense-MSE term
            # (default 1.0 = byte-identical to the pr/77 recipe)
            loss = criterion(score, target) * getattr(model, 'dense_weight', 1.0)
            w = float(meta['row'].get('weight', 1.0))  # S8c pseudo-labels < 1
            if w != 1.0:
                loss = loss * w
            cs = getattr(model, 'cand_softmax', 0.0)
            ci = meta.get('cand_idx')
            if cs > 0 and ci is not None:
                # doc pr/81 B: per-event softmax CE over the CANDIDATE voxel
                # scores.  Invariant to a uniform additive score shift -- the
                # loss cannot be paid by en-bloc confidence inflation (the
                # pr/79 sec 11 failure mode).
                vi = torch.from_numpy(ci)
                ti = meta['cand_truth_i']
                valid = vi >= 0
                if ti >= 0 and bool(valid[ti]) and int(valid.sum()) >= 2:
                    idx = vi[valid]
                    pos = int(valid[:ti + 1].sum()) - 1
                    loss = loss + cs * torch.nn.functional.cross_entropy(
                        score[idx].unsqueeze(0), torch.tensor([pos]))
            sa = getattr(model, 'scale_anchor', 0.0)
            if sa > 0 and frozen is not None:
                # doc pr/81 B: pin the absolute score scale to the frozen
                # CP24 calibration on the SAME augmented voxelization, over
                # the frozen net's top-20 voxels (where the acceptance gate
                # lives), while the ranking term reorders.
                with torch.no_grad():
                    pf = frozen([coords, ft])
                    sf = pf[:, 1] - pf[:, 0]
                m = min(20, len(sf))
                top = torch.topk(sf, m).indices
                loss = loss + sa * torch.mean((score[top] - sf[top]) ** 2)
            cm = getattr(model, 'cand_margin', 0.0)
            if cm > 0 and ci is not None:
                # S9b: deployed-objective margin -- the truth candidate's
                # voxel score must beat every other candidate's by 0.1
                vi = torch.from_numpy(ci)
                ti = meta['cand_truth_i']
                valid = vi >= 0
                if bool(valid[ti]) and int(valid.sum()) >= 2:
                    st = score[vi[ti]]
                    others = vi[valid.clone()]
                    others = others[others != vi[ti]]
                    if len(others):
                        viol = torch.relu(0.1 - (st - score[others]))
                        loss = loss + cm * viol.mean()
            if consistency > 0:  # S8c: label-free view-agreement term
                (c1, f1, p1), (c2, f2, p2) = samples.consistency_views(k)
                pr1 = model([torch.LongTensor(c1), torch.FloatTensor(f1).to(device)])
                pr2 = model([torch.LongTensor(c2), torch.FloatTensor(f2).to(device)])
                s1 = pr1[:, 1] - pr1[:, 0]
                s2 = pr2[:, 1] - pr2[:, 0]
                loss = loss + consistency * criterion(
                    s1[torch.from_numpy(p1)], s2[torch.from_numpy(p2)])
            loss.backward()
            clip = getattr(model, 'grad_clip', 0.0)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], clip)
            optimizer.step()
        else:
            with torch.no_grad():
                prediction = model([coords, ft])
                score = prediction[:, 1] - prediction[:, 0]
                loss = criterion(score, target)
        losses.append(loss.item())
        dists.append(d_argmax_cm(score.detach().cpu().numpy(), coords_np,
                                 meta['offset'], meta['truth']))
    return float(np.mean(losses)), float(np.median(dists)), dists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='snapshot dir (data/<name>)')
    ap.add_argument('--name', required=True, help='run name under runs/')
    ap.add_argument('--weights', default=vio.default_weights())
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr0', type=float, default=1e-6)
    ap.add_argument('--lrd', type=float, default=0.05,
                    help='lr = lr0*exp(-lrd*epoch), the original t48k decay')
    ap.add_argument('--sigma', type=float, default=1.0, help='truth Gaussian sigma (cm)')
    ap.add_argument('--freeze', choices=['head', 'linear', 'none'], default='head')
    ap.add_argument('--kfold', type=int, default=6, help='0 = train on everything')
    ap.add_argument('--fold', type=int, default=-1, help='run only this fold')
    ap.add_argument('--no-flips', action='store_true')
    ap.add_argument('--no-jitter', action='store_true')
    ap.add_argument('--no-q-jitter', action='store_true')
    ap.add_argument('--dropout', action='store_true')
    ap.add_argument('--hard-negative', type=float, default=0.0,
                    help='S8b: negative-Gaussian weight at the production '
                         'pick on corrective events (0 = off)')
    ap.add_argument('--consistency', type=float, default=0.0,
                    help='S8c: label-free view-agreement loss weight (0 = off)')
    ap.add_argument('--keep-lockbox', action='store_true',
                    help='include lockbox events (default: excluded from '
                         'training AND folds)')
    ap.add_argument('--pseudo-data', default=None,
                    help='S8c: pseudo-label snapshot dir; its events join '
                         'every fold TRAIN set (never validation)')
    ap.add_argument('--pseudo-weight', type=float, default=0.25)
    ap.add_argument('--min-cloud', type=int, default=0,
                    help='round 3: drop events with fewer cloud points than '
                         'this from TRAINING folds (they stay in val).  A '
                         '3-voxel cloud reaches the deepest UNet level as a '
                         'single site and NaNs the forward (evt 171892).')
    ap.add_argument('--clip', type=float, default=0.0,
                    help='round 3: clip_grad_norm_ max norm (0 = off)')
    ap.add_argument('--cands', default=None,
                    help='S9b: candidate sidecar dir from build_cands.py; '
                         'enables the deployed-objective margin loss')
    ap.add_argument('--cand-margin', type=float, default=0.0,
                    help='S9b: weight of the candidate margin term (0 = off)')
    ap.add_argument('--cand-softmax', type=float, default=0.0,
                    help='doc pr/81 B: weight of the per-event softmax-CE '
                         'over candidate voxel scores (0 = off; needs '
                         '--cands)')
    ap.add_argument('--scale-anchor', type=float, default=0.0,
                    help='doc pr/81 B: weight of the frozen-CP24 score-scale '
                         'anchor on the top-20 frozen voxels (0 = off; '
                         'roughly doubles epoch time)')
    ap.add_argument('--dense-weight', type=float, default=1.0,
                    help='doc pr/81 B: weight of the legacy dense-MSE term '
                         '(1.0 = pr/77 recipe)')
    ap.add_argument('--bn-freeze', action='store_true',
                    help='round 3: keep BatchNorm layers in eval mode during '
                         'training (frozen running stats; fixes the '
                         'tiny-cloud NaN running_var corruption)')
    ap.add_argument('--upweight', default=None,
                    help="round 3: '<sample>:<conf|corr|all>:<w>' e.g. "
                         "'numu100:conf:3.0' -- multiply the loss weight of "
                         "matching manifest rows (sample column)")
    ap.add_argument('--seed', type=int, default=20260814)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(here, 'runs', args.name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as fh:
        json.dump(vars(args), fh, indent=1, sort_keys=True)

    rows = load_manifest(args.data, drop_lockbox=not args.keep_lockbox)
    if args.upweight:
        samp, which, w = args.upweight.split(':')
        w = float(w)
        n_up = 0
        for r in rows:
            if r.get('sample', '') != samp:
                continue
            if which == 'conf' and r['corrective']:
                continue
            if which == 'corr' and not r['corrective']:
                continue
            r['weight'] = float(r.get('weight', 1.0)) * w
            n_up += 1
        print('upweight: %d rows (sample=%s %s) x%.1f' % (n_up, samp, which, w))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.kfold > 0:
        splits = list(kfold_split(rows, args.kfold, seed=args.seed))
    else:
        splits = [(0, rows, [])]

    pseudo_rows = []
    if args.pseudo_data:
        pseudo_rows = load_manifest(args.pseudo_data)
        for r in pseudo_rows:
            r['npz'] = os.path.join(os.path.abspath(args.pseudo_data), r['npz'])
            r['weight'] = args.pseudo_weight
        print('pseudo-labels: +%d train-only events at weight %.2f'
              % (len(pseudo_rows), args.pseudo_weight))

    criterion = nn.MSELoss()
    sample_kw = dict(sigma=args.sigma, use_flips=not args.no_flips,
                     jitter=not args.no_jitter, q_jitter=not args.no_q_jitter,
                     dropout=args.dropout, hard_negative=args.hard_negative)

    oof = {}  # evt -> best out-of-fold d_argmax at selected epoch
    for fold, train_rows, val_rows in splits:
        if args.fold >= 0 and fold != args.fold:
            continue
        fold_dir = os.path.join(run_dir, 'fold%d' % fold)
        os.makedirs(fold_dir, exist_ok=True)

        model = make_model(device=args.device)
        load_weights(model, args.weights)
        bn_snap = bn_stats_snapshot(model) if args.bn_freeze else None
        model.grad_clip = args.clip
        model.cand_margin = args.cand_margin
        model.cand_softmax = args.cand_softmax
        model.scale_anchor = args.scale_anchor
        model.dense_weight = args.dense_weight
        # frozen CP24 for the scale anchor: kept OUT of the model's module
        # tree (a submodule would leak its params into the checkpoint)
        frozen = None
        if args.scale_anchor > 0:
            frozen = make_model(device=args.device)
            load_weights(frozen, vio.default_weights())
            frozen.train(False)
            for p in frozen.parameters():
                p.requires_grad = False
        n_train, n_total = set_freeze(model, args.freeze)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=args.lr0)

        if args.min_cloud > 0:
            kept = [r for r in train_rows if int(r['n_cloud']) >= args.min_cloud]
            if len(kept) != len(train_rows):
                print('[fold %d] min-cloud %d: dropped %d train events'
                      % (fold, args.min_cloud, len(train_rows) - len(kept)))
            train_rows = kept
        tr = VtxSamples(args.data, train_rows + pseudo_rows, train=True,
                        seed=args.seed + fold, cands_dir=args.cands,
                        **sample_kw)
        va = VtxSamples(args.data, val_rows, train=False,
                        seed=args.seed + fold, **sample_kw)
        print('[fold %d] train %d events (%d views), val %d events; '
              'trainable %d/%d params' % (fold, len(train_rows), len(tr),
                                          len(val_rows), n_train, n_total))

        log_path = os.path.join(fold_dir, 'log.tsv')
        best = (math.inf, -1)  # (val median d_argmax, epoch)
        with open(log_path, 'w') as log:
            log.write('epoch\tlr\ttrain_loss\ttrain_d50\tval_loss\tval_d50\n')
            for epoch in range(args.epochs):
                lr = args.lr0 * math.exp(-args.lrd * epoch)
                for g in optimizer.param_groups:
                    g['lr'] = lr
                tl, td, _ = run_epoch(model, tr, args.device, criterion, optimizer,
                                      consistency=args.consistency,
                                      frozen=frozen)
                if bn_snap is not None:
                    bn_stats_restore(model, bn_snap)
                if len(va):
                    vl, vd, _ = run_epoch(model, va, args.device, criterion)
                else:
                    vl, vd = float('nan'), float('nan')
                log.write('%d\t%.3g\t%.6f\t%.3f\t%.6f\t%.3f\n'
                          % (epoch, lr, tl, td, vl, vd))
                log.flush()
                torch.save(model.state_dict(),
                           os.path.join(fold_dir, 'CP%d.pth' % epoch))
                if len(va) and vd < best[0]:
                    best = (vd, epoch)
                print('[fold %d] epoch %2d lr=%.2e train loss=%.5f d50=%.2f | '
                      'val loss=%.5f d50=%.2f' % (fold, epoch, lr, tl, td, vl, vd))

        if len(va):
            with open(os.path.join(fold_dir, 'best.json'), 'w') as fh:
                json.dump({'best_epoch': best[1], 'val_d50': best[0]}, fh)
            # out-of-fold per-event distances at the selected epoch
            model = make_model(device=args.device)
            load_weights(model, os.path.join(fold_dir, 'CP%d.pth' % best[1]))
            _, _, dists = run_epoch(model, va, args.device, criterion)
            for r, d in zip([va.event_of(k)['evt'] for k in range(len(va))], dists):
                oof[r] = d
            print('[fold %d] best epoch %d, val median d=%.2f cm'
                  % (fold, best[1], best[0]))

    if oof:
        # fold-parallel launches (--fold k) write per-fold files so the
        # processes don't clobber each other; merge_oof.py combines them.
        oof_name = ('oof_d_argmax.tsv' if args.fold < 0
                    else 'oof_fold%d.tsv' % args.fold)
        with open(os.path.join(run_dir, oof_name), 'w') as fh:
            fh.write('evt\td_argmax_cm\n')
            for evt in sorted(oof):
                fh.write('%d\t%.3f\n' % (evt, oof[evt]))
        d = np.array(list(oof.values()))
        print('\n== out-of-fold over %d events: d_argmax p50=%.2f p90=%.2f max=%.2f cm'
              % (len(d), np.percentile(d, 50), np.percentile(d, 90), d.max()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
