#!/usr/bin/env python3
"""doc sbnd_xin/113 sec 5 -- imaging cost and blob count of a fix-variant arm vs the cf0 baseline, group by group
(work-<s>-d113cf<arm>/g<K>/.img.time.meta + icluster npz 'b' node counts), plus the quiet-masking log line.

Usage: d113_arm_cost.py --sample mcp1k --arms qp2v4 qpg64 [--base cf0]
"""
import argparse, glob, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_common as C


def meta(p):
    t = open(p).read()
    return float(re.search(r'wall_s=(\d+)', t).group(1)), float(re.search(r'maxrss_kb=(\d+)', t).group(1))


def nblobs(g):
    n = 0
    for apa in range(C.NAPA):
        f = f'{g}/icluster-apa{apa}-active.npz'
        if not os.path.exists(f):
            return None
        z = np.load(f); n += sum(len(z[k]) for k in z.keys() if k.endswith('_bnodes'))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', default='mcp1k')
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--base', default='cf0')
    a = ap.parse_args()
    for arm in a.arms:
        rows = []
        for g in sorted(glob.glob(f'{C.SX}/work-{a.sample}-d113cf{arm}/g*')):
            gid = os.path.basename(g); b = f'{C.SX}/work-{a.sample}-d113cf{a.base}/{gid}'
            if not os.path.exists(f'{g}/.img.time.meta') or not os.path.exists(f'{b}/.img.time.meta'):
                continue
            nb0, nb1 = nblobs(b), nblobs(g)
            if nb0 is None or nb1 is None:
                continue
            (w0, r0), (w1, r1) = meta(f'{b}/.img.time.meta'), meta(f'{g}/.img.time.meta')
            rows.append((nb0, nb1, w0, w1, r0, r1))
        if not rows:
            print(f'{arm}: no finished groups'); continue
        R = np.array(rows, float)
        cells = []
        for lg in glob.glob(f'{C.SX}/work-{a.sample}-d113cf{arm}/g*/wct_img.log'):
            for m in re.finditer(r'quiet-plane masking: window (-?\d+)(?: gap (-?\d+))? slices, (\d+) masked-plane channels, (\d+) \(channel, slice\) cells', open(lg).read()):
                cells.append(int(m.group(4)))
        print(f'{arm} vs {a.base} on {a.sample}: {len(R)} groups; blobs/group {np.median(R[:,0]):.0f} -> {np.median(R[:,1]):.0f} (ratio median {np.median(R[:,1]/R[:,0]):.2f}, max {np.max(R[:,1]/R[:,0]):.2f}); '
              f'imaging wall {np.median(R[:,2]):.0f} s -> {np.median(R[:,3]):.0f} s (ratio median {np.median(R[:,3]/R[:,2]):.2f}); maxrss {np.median(R[:,4])/1e6:.2f} -> {np.median(R[:,5])/1e6:.2f} GB (ratio {np.median(R[:,5]/R[:,4]):.2f}); '
              f'quiet-masked cells per slicer call median {np.median(cells) if cells else float("nan"):.0f}')


if __name__ == '__main__':
    main()
