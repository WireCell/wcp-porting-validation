#!/usr/bin/env python3
"""doc 94 round 2 lead -- the ENTRY-END dQ/dx rise (owner, 2026-09-02).

Every predicate in TaggerCheckSTM looks at the STOP end: eval_stm_core searches
for the Bragg peak in a window ending at the kink, detect_proton tests the last
20-35 cm, and all five doc-63 guards plus doc 94's vertex_hadron_guard read the
stop region or the prongs.  The owner's observation is that the discriminating
feature for 827-27-4 sits at the OTHER end -- a dQ/dx RISE at the boundary
point where the fit starts.

Physics: for a single muon the boundary end is where it is most energetic, so
dQ/dx there must be at its LOWEST, ~1 MIP.  Charge well above MIP at the
boundary that decays to MIP over the next 10-20 cm is two particles sharing
that stretch, one of which leaves the detector.

Emits the entry/body ratio per event and a comparison plot.
Usage: doc94_entry_rise.py <arm> <out.png> <evt>[:<label>] ...
Read-only.
"""
import io, json, os, sys, tarfile
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MIP = 56000.0

def load(evt, root):
    metas, arrs = {}, {}
    with tarfile.open(f'{root}/pr_evt{evt}/pctree-pr-evt{evt}.tar.gz') as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if m.name.endswith('_metadata.json'):
                metas[m.name[:-len('_metadata.json')]] = json.load(f)
            elif m.name.endswith('_array.npy'):
                arrs[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(f.read()))
    byp = {md['datapath']: arrs.get(b) for b, md in metas.items() if 'datapath' in md}
    pre = f'pointtrees/{evt}/live/pointclouds/namedpcs/stm_fit/arrays/'
    if pre + 'x' not in byp:
        return None
    return byp[pre+'L']/10., byp[pre+'dQ']/(byp[pre+'dx']/10.+1e-9)/MIP

root, out = sys.argv[1], sys.argv[2]
fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
print(f"{'event':<14}{'entry 0-3cm':>12}{'body':>8}{'ratio':>8}   label")
for spec in sys.argv[3:]:
    evt, lab = (spec.split(':') + [spec])[:2]
    r = load(evt, root)
    if r is None:
        print(f'{lab:<14} no stm_fit'); continue
    L, dq = r
    body = np.median(dq[(L > 25) & (L < L[-1] - 25)]) if L[-1] > 60 else np.median(dq)
    ent = np.median(dq[L < 3])
    print(f'{lab:<14}{ent:12.2f}{body:8.2f}{ent/body:8.2f}   {spec}')
    m = L < 40
    ax[0].plot(L[m], dq[m], '-', lw=1.3, label=f'{lab}  (entry/body {ent/body:.2f})')
    rr = L[-1] - L
    m2 = rr < 40
    ax[1].plot(rr[m2], dq[m2], '-', lw=1.3, label=lab)
for a, t, xl in ((ax[0], 'ENTRY end -- distance from the boundary point', 'L from boundary [cm]'),
                 (ax[1], 'STOP end -- residual range (what the tagger looks at)', 'residual range [cm]')):
    a.axhline(1.0, color='0.6', ls='--', lw=1)
    a.set_xlabel(xl); a.set_ylabel('dQ/dx [MIP]'); a.set_title(t); a.grid(alpha=.3); a.legend(fontsize=8)
ax[1].invert_xaxis()
plt.tight_layout(); plt.savefig(out, dpi=105); print('wrote', out)
