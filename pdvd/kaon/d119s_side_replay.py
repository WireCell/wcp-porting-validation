#!/usr/bin/env python3
"""doc pdvd/119 sec 8: replay the Bee PDVD side panel (bee3 experiment.js
ProtoDUNEVD.detectorFrameCorrection + sst.js drawDetectorFrame) on a Bee zip, per
matched cluster, with the OLD rule (geometric side test, raw-x tie-break) and the
NEW rule (op_cluster_anodes when present).  The exact reference is the
clustering-global layer, written in the toolkit's x_t0cor: after the fix the
img-global side-panel x of every matched cluster must equal it point for point
(checked when the two layers are point-aligned), and the clustering layer itself
is drawn unshifted.
Usage: d119s_side_replay.py <bee.zip> [nevents]"""
import json, sys, zipfile
import numpy as np
V, L, TOL = 0.148073, 341.55, 5            # bee3 PDVD driftVelocity, box |x| edge, tolerance
z = zipfile.ZipFile(sys.argv[1])
nev = int(sys.argv[2]) if len(sys.argv) > 2 else 10
tot = {'clusters': 0, 'old_bottom': 0, 'new_bottom': 0, 'old_bad': 0, 'new_bad': 0, 'no_anode': 0}
worst_new = 0.0
for ev in range(nev):
    op = json.loads(z.read(f'data/{ev}/{ev}-op.json'))
    img = json.loads(z.read(f'data/{ev}/{ev}-img-global.json'))
    clu = json.loads(z.read(f'data/{ev}/{ev}-clustering-global.json'))
    x = np.array(img['x']); cid = np.array(img['cluster_id'])
    aligned = (len(img['x']) == len(clu['x']) and img['y'] == clu['y'] and img['z'] == clu['z'])
    xc = np.array(clu['x'])
    an = op.get('op_cluster_anodes')
    first = {}
    for i, cs in enumerate(op['op_cluster_ids']):
        for j, c in enumerate(cs or []):
            if int(c) not in first:
                first[int(c)] = (i, an[i][j] if an else None)
    evbad = []
    for c, (fi, a) in first.items():
        sel = cid == c
        if not sel.any():
            continue
        tb = op['op_t'][fi]; tt = op['op_t1'][fi] if 'op_t1' in op else tb
        p = x[sel]
        fb = ((p - V*tb < -L - TOL) | (p - V*tb > TOL)).mean()
        ft = ((p + V*tt > L + TOL) | (p + V*tt < -TOL)).mean()
        old_top = (p.mean() > 0) if abs(fb - ft) < 0.05 else (ft < fb)
        new_top = old_top if (a is None or a < 0) else (a >= 4)
        tot['no_anode'] += (a is None or a < 0)
        xo = p + V*tt if old_top else p - V*tb
        xn = p + V*tt if new_top else p - V*tb
        tot['clusters'] += 1; tot['old_bottom'] += not old_top; tot['new_bottom'] += not new_top
        if aligned:
            do, dn = np.abs(xo - xc[sel]).max(), np.abs(xn - xc[sel]).max()
            tot['old_bad'] += do > 0.01; tot['new_bad'] += dn > 0.01
            worst_new = max(worst_new, dn)
            if dn > 0.01: evbad.append((c, a, round(float(dn), 2)))
    print(f"idx {ev} evt {op['eventNo']}: aligned={aligned} matched={len(first)} anodes={'yes' if an else 'no'}"
          + (f" NEW-RULE MISMATCH {evbad[:6]}" if evbad else ''))
print(tot, 'max |x_side(img) - x_t0cor(clustering)| new rule = %.4f cm' % worst_new)
