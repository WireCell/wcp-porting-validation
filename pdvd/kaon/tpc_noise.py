#!/usr/bin/env python3
"""doc pdvd/118 sec 4.2: raw-ADC noise and dead channels on the TOP anodes
(4-7) from the orig frames (pre-NF), 39305 kaon inputs vs 039349 evt_0-9.
Per channel: robust RMS = 1.4826 * MAD(adc - median) over the full 6400-tick
frame; 'dead' = robust RMS < 1 ADC.  Plane from the protodunevd anode_channels
layout (per CRP: U 0-951, V 952-1903, W 1904-3071).  Also the pedestal median."""
import glob, io, os, tarfile
import numpy as np
KDIR = os.path.dirname(os.path.abspath(__file__)); PDVD = os.path.dirname(KDIR)
def frames(evtdir):
    for a in range(4, 8):
        with tarfile.open(f'{evtdir}/protodune-orig-frames-anode{a}.tar.bz2') as t:
            m = {x.name.split('_')[0]: np.load(io.BytesIO(t.extractfile(x).read())) for x in t.getmembers()}
        yield a, m['frame'], m['channels']
SETS = {'039305 kaon': sorted(glob.glob(f'{PDVD}/input_kaon/run039305/evt_*')),
        '039349 evt_0-9': [f'{PDVD}/input_data/run039349/evt_{i}' for i in range(10)]}
lines = []
for name, dirs in SETS.items():
    per = {p: [] for p in 'UVW'}; ped = {p: [] for p in 'UVW'}; dead = {p: [] for p in 'UVW'}
    for d in dirs:
        for a, fr, ch in frames(d):
            med = np.median(fr, axis=1)
            rms = 1.4826 * np.median(np.abs(fr - med[:, None]), axis=1)
            loc = ch % 3072
            pl = np.where(loc < 952, 'U', np.where(loc < 1904, 'V', 'W'))
            for p in 'UVW':
                s = pl == p
                per[p].append(np.median(rms[s])); ped[p].append(np.median(med[s])); dead[p].append(int((rms[s] < 1).sum()))
    lines.append(f'{name} ({len(dirs)} events x 4 top anodes):  ' + '  '.join(
        f'{p}: rms {np.mean(per[p]):.2f}+-{np.std(per[p]):.2f} ADC, ped {np.mean(ped[p]):.0f}, dead/anode {np.mean(dead[p]):.1f}'
        for p in 'UVW'))
print('\n'.join(lines))
open(f'{KDIR}/out/tpc_noise.txt', 'w').write('\n'.join(lines) + '\n')
