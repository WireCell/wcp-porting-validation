#!/usr/bin/env python3
'''doc pr/106 sec 7 -- score SCN checkpoints on the TARGET metric's weight-independent
quantities, on the EXACT live net input (hv_cloud, dl_vtx_harvest arms).

For each event and checkpoint: run pyutil SCN_Vertex (production code path,
byte-exact vs live: verify_harvest.py) on the harvested cloud with top_k=K,
snap every voxel to the nearest pre-DL candidate vertex (brute-force NN over
the cloud's vertex rows == NeutrinoVertexFinder.cxx:5000-5037), and report:
  admit5 / admit10 : target among the snapped candidates of the top-5 / top-10
  top1             : the rank-0 voxel snaps to the target (DL-alone pick)
  best_dl          : the event's best voxel score (the "net abstained" axis)
  target_dl/rank   : score and rank of the best voxel snapping to the target
Because the cloud is the live input, rank<=5 results for the production
checkpoint must equal the recorded scoreboard (asserted: --check).

Usage: ./ckpt_target_admission.py --events-tsv docs/pr/106_events-tune.tsv [...]
           --ckpt CP24=<path> ft2u=<path> --jobs 12 --tsv out.tsv
'''
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
PYUTIL = '/nfs/data/1/xqian/toolkit-dev/toolkit/pyutil/python'
K = 10
_W = None


def _init(weights):
    global _W
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    sys.path.insert(0, PYUTIL)
    _W = weights


def run_one(job):
    evt, sample, target, arm = job
    import SCN_Vertex as sv
    p = os.path.join(ROOT, arm.format(sample=sample), 'pr_evt%d' % evt, 'calib-pr-evt%d.json' % evt)
    sb = json.load(open(p))['vertex_scoreboard']
    c = sb['hv_cloud']
    nv = c['n_vertex_rows']
    if nv == 0:
        return None
    x, y, z, q = (np.array(c[k], np.float32) for k in 'xyzq')
    ids = c['vertex_ids'][:nv]
    vxyz = np.stack([x[:nv], y[:nv], z[:nv]], axis=1)
    out = {}
    for name, w in _W.items():
        raw = sv.SCN_Vertex(w, x.tobytes(), y.tobytes(), z.tobytes(), q.tobytes(), dtype='float32', top_k=K)
        vox = np.frombuffer(raw, np.float32).reshape(-1, 4)
        snapped = []
        for r in range(len(vox)):
            d = np.linalg.norm(vxyz - vox[r, :3], axis=1)
            i = int(np.argmin(d))
            snapped.append((ids[i], float(vox[r, 3]), float(d[i])))
        tgt = [(r, s, d) for r, (vid, s, d) in enumerate(snapped) if vid == target]
        out[name] = dict(
            best_dl=float(vox[0, 3]) if len(vox) else -9,
            top1=int(bool(snapped) and snapped[0][0] == target),
            admit5=int(any(r < 5 for r, _, _ in tgt)), admit10=int(bool(tgt)),
            target_rank=tgt[0][0] if tgt else -1, target_dl=tgt[0][1] if tgt else -9,
            top5_rec=[(v['x'], v['y'], v['z'], v['dl_score']) for v in sb['voxels']][:5],
            top5_got=[tuple(map(float, vox[r])) for r in range(min(5, len(vox)))])
    return evt, sample, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--events-tsv', nargs='+', required=True, help='vtx_target_eval --events-tsv output(s)')
    ap.add_argument('--arm', default='work-vtx106-harv-base-{sample}')
    ap.add_argument('--ckpt', nargs='+', required=True, metavar='name=path')
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--dmax', type=float, default=3.0)
    ap.add_argument('--check', default='CP24', help='checkpoint whose top-5 must equal the recorded voxels')
    ap.add_argument('--tsv', required=True)
    args = ap.parse_args()
    weights = dict(a.split('=', 1) for a in args.ckpt)
    jobs = []
    for f in args.events_tsv:
        for r in csv.DictReader(open(f), delimiter='\t'):
            if float(r['d_target']) <= args.dmax:
                jobs.append((int(r['evt']), r['sample'], int(r['target']), args.arm))
    print('events %d, checkpoints %s' % (len(jobs), list(weights)))
    with Pool(args.jobs, initializer=_init, initargs=(weights,)) as pool:
        res = [r for r in pool.imap_unordered(run_one, jobs, chunksize=4) if r is not None]
    res.sort()
    nbad = 0
    with open(args.tsv, 'w') as fh:
        names = list(weights)
        fh.write('evt\tsample\t' + '\t'.join('%s_%s' % (n, k) for n in names for k in ('best_dl', 'top1', 'admit5', 'admit10', 'target_rank', 'target_dl')) + '\n')
        for evt, sample, out in res:
            if args.check in out:
                rec = np.array(out[args.check]['top5_rec'], np.float32)
                got = np.array(out[args.check]['top5_got'], np.float32)[:len(rec)]
                if not np.allclose(rec, got, rtol=0, atol=1e-5):
                    nbad += 1
            fh.write('%d\t%s\t' % (evt, sample) + '\t'.join(
                str(out[n][k]) for n in names for k in ('best_dl', 'top1', 'admit5', 'admit10', 'target_rank', 'target_dl')) + '\n')
    print('recorded-voxel reproduction (%s): %d/%d mismatches' % (args.check, nbad, len(res)))
    import collections
    for n in weights:
        tot = collections.Counter()
        for evt, sample, out in res:
            o = out[n]
            for k in ('top1', 'admit5', 'admit10'):
                tot[(sample, k)] += o[k]
                tot[('ALL', k)] += o[k]
            tot[(sample, 'n')] += 1
            tot[('ALL', 'n')] += 1
        print('\n%s:' % n)
        for s in ('nuecc48', 'ncpi0', 'mcp1k', 'mcp2k', 'ALL'):
            print('  %-8s n=%4d  top1 %4d  admit5 %4d  admit10 %4d' % (s, tot[(s, 'n')], tot[(s, 'top1')], tot[(s, 'admit5')], tot[(s, 'admit10')]))


if __name__ == '__main__':
    main()
