#!/usr/bin/env python3
"""doc pr/149 amendment 1: Steiner-terminal nearest-neighbour spacing of the main cluster.

Reads calib-pr-evt*.json steiner[] (flag_terminal) of the main cluster
(main_vertex.cluster_id) for every event of the given arms and prints the median
3-D nearest-neighbour distance between terminals, pooled over events.

Usage: terminal_spacing.py ARMDIR [ARMDIR ...]
"""
import glob
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

for arm in sys.argv[1:]:
    nn = []
    nterm = []
    for p in sorted(glob.glob(f'{arm}/pr_evt*/calib-pr-evt*.json')):
        d = json.load(open(p))
        cid = (d.get('main_vertex') or {}).get('cluster_id')
        for t in d.get('steiner', []):
            if t.get('cluster_id') != cid:
                continue
            f = np.asarray(t['flag_terminal'], bool)
            P = np.stack([np.asarray(t['x']), np.asarray(t['y']), np.asarray(t['z'])], axis=1)[f]
            nterm.append(len(P))
            if len(P) >= 2:
                dd, _ = cKDTree(P).query(P, k=2)
                nn.extend(dd[:, 1].tolist())
    nn = np.asarray(nn)
    print(f'{arm}: clusters {len(nterm)} terminals median {np.median(nterm):.0f}  '
          f'terminal NN spacing cm p25 {np.percentile(nn, 25):.3f} median {np.median(nn):.3f} p75 {np.percentile(nn, 75):.3f}')
