#!/usr/bin/env python3
"""PDHD per-event PD activity from the TOOLKIT (WCT-native) OpHits -- the
toolkit analogue of pd_activity.py (which reads the LArSoft PerOpHitTree and
therefore CANNOT see the -x full-stream PDs 120-159).

Source: the per-event all-PD opflash files
  work/027980_allpd<evt>/opflash_pdhd-allpd-wct.tar.gz
whose "ophits" tensor [nhit,9] carries every OpHit (col 0 = channel, col 5 = PE)
across all 160 PDs.  We bin by the four 40-PD APA blocks and plot, per event, the
total PE and the number of fired PDs per block -- now including the -x full
block that the LArSoft OpHit tree lacks.

Plots -> pdhd/pics/ : pd_activity_wct_27980_total_pe.png, _nfired.png
"""
import glob, io, re, tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = '/nfs/data/1/xqian/toolkit-dev/toolkit/pdhd/work'
PICS = '/nfs/data/1/xqian/toolkit-dev/toolkit/pdhd/pics'

# four 40-PD APA blocks: (label, lo, hi, colour)
BLOCKS = [
    ('+x up (0-39)',     0,   40,  '#1f77b4'),
    ('+x lo (40-79)',    40,  80,  '#17becf'),
    ('-x self (80-119)', 80,  120, '#d62728'),
    ('-x full (120-159)', 120, 160, '#9467bd'),
]


def ophits(path):
    tf = tarfile.open(path, 'r:gz')
    names = tf.getnames()
    tname = {}
    arr = {}
    for n in names:
        if n.endswith('_array.npy'):
            arr[int(n.split('_')[-2])] = np.load(io.BytesIO(tf.extractfile(n).read()))
        elif n.endswith('_metadata.json') and 'tensorset' not in n:
            import json
            tname[int(n.split('_')[-2])] = json.load(tf.extractfile(n)).get('name', '')
    for i, nm in tname.items():
        if nm == 'ophits':
            return arr[i]
    return arr.get(2)


def load(run):
    files = sorted(glob.glob(f'{WORK}/{run:06d}_allpd*/opflash_pdhd-allpd-wct.tar.gz'),
                   key=lambda p: int(re.search(r'_allpd(\d+)/', p).group(1)))
    evs = [int(re.search(r'_allpd(\d+)/', p).group(1)) for p in files]
    out = {'events': np.array(evs)}
    for lbl, lo, hi, _ in BLOCKS:
        out[lbl] = {'pe': np.zeros(len(files)), 'nfired': np.zeros(len(files), int)}
    for i, p in enumerate(files):
        h = ophits(p)
        ch = h[:, 0].astype(int); pe = h[:, 5]
        for lbl, lo, hi, _ in BLOCKS:
            m = (ch >= lo) & (ch < hi)
            out[lbl]['pe'][i] = pe[m].sum()
            out[lbl]['nfired'][i] = len(np.unique(ch[m]))
    return out


d = load(27980)
x = np.arange(len(d['events']))
xt = [str(e) for e in d['events']]

# total PE per event by block
fig, ax = plt.subplots(figsize=(13, 5))
for lbl, lo, hi, col in BLOCKS:
    ax.plot(x, d[lbl]['pe'], 'o-', color=col, lw=1.5, ms=4, label=lbl)
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(xt, rotation=90, fontsize=7)
ax.set_xlabel('event'); ax.set_ylabel('total PE in block (log)')
ax.set_title('PDHD run 27980 per-event total PE by APA block (TOOLKIT WCT OpHits)\n'
             'now includes the -x full-stream block (120-159) absent from the LArSoft OpHit tree')
ax.legend(); fig.tight_layout()
fig.savefig(f'{PICS}/pd_activity_wct_27980_total_pe.png', dpi=110)
print('wrote pd_activity_wct_27980_total_pe.png')

# number of fired PDs per event by block
fig, ax = plt.subplots(figsize=(13, 5))
for lbl, lo, hi, col in BLOCKS:
    ax.plot(x, d[lbl]['nfired'], 'o-', color=col, lw=1.5, ms=4, label=f'{lbl} (max 40)')
ax.set_xticks(x); ax.set_xticklabels(xt, rotation=90, fontsize=7)
ax.set_xlabel('event'); ax.set_ylabel('# fired PDs in block')
ax.axhline(40, color='gray', ls=':', lw=1)
ax.set_title('PDHD run 27980 per-event fired-PD count by APA block (TOOLKIT WCT OpHits)\n'
             '-x full (120-159) is uniformly active -- the -x wall is not anomalous')
ax.legend(); fig.tight_layout()
fig.savefig(f'{PICS}/pd_activity_wct_27980_nfired.png', dpi=110)
print('wrote pd_activity_wct_27980_nfired.png')

# brief table
print(f"\n{'evt':>5} " + ' '.join(f'{lbl.split()[0]+lbl.split()[1]:>10}' for lbl, *_ in BLOCKS))
for i, e in enumerate(d['events']):
    print(f'{e:>5} ' + ' '.join(f"{d[lbl]['pe'][i]:7.0f}/{d[lbl]['nfired'][i]:02d}" for lbl, *_ in BLOCKS))
print('DONE.')
