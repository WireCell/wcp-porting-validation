#!/usr/bin/env python3
"""doc pdvd/118 pre-flight on the converted kaon inputs.
Per event: orig-frame nticks/tick/nchan per top anode; light record layout;
trigoff offsets; the beam trigger on the TPC tick axis and on the chain
flash axis (PDVDOpWaveformSource t0 = min(ts - 64 samples for snippets))."""
import uproot, numpy as np, tarfile, io, glob, os, csv, sys
PDVD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = f'{PDVD}/input_kaon'
man = {r['event']: r for r in csv.DictReader(open(f'{PDVD}/input_data_kaon/MANIFEST.csv'))}
rows = []
for d in sorted(glob.glob(f'{IN}/run039305/evt_*'), key=lambda p: int(p.split('_')[-1])):
    e = int(d.split('_')[-1])
    fr = {}
    for a in range(4, 8):
        with tarfile.open(f'{d}/protodune-orig-frames-anode{a}.tar.bz2') as t:
            for m in t.getmembers():
                x = np.load(io.BytesIO(t.extractfile(m).read()))
                if m.name.startswith('frame'): fr[a] = x.shape
                if m.name.startswith('tickinfo'): ti = x
    f = uproot.open(f'{IN}/light/np02vd_raw_run039305_evt{e}_rawwf.root')
    w = f['rawdump/raw_waveform'].arrays(['opchannel', 'nsamp', 'timestamp'], library='np')
    ns, ts, ch = w['nsamp'], w['timestamp'], w['opchannel']
    fsm = ns > 1024
    t0 = np.min(np.where(ns <= 1024, ts - 64 / 62.5, ts))
    tr = {k: v[0] for k, v in f['trigoff/trigger_offset'].arrays(library='np').items()}
    fs0 = ts[fsm].min()
    r = dict(event=e,
             nticks='/'.join(str(fr[a][1]) for a in range(4, 8)),
             nchan='/'.join(str(fr[a][0]) for a in range(4, 8)),
             tick_ns=ti[1], frame_t=ti[0],
             fs_nsamp=int(ns[fsm].max()), n_fs=int(fsm.sum()), n_snip=int((~fsm).sum()),
             snip_max=int(ns[~fsm].max()),
             tc_type=int(tr['tc_type']),
             trig_minus_fs_us=tr['tc_us'] - fs0,
             tpcframe_minus_fs_us=tr['charge_tde_us'] - fs0,
             trig_in_tpc_us=tr['tc_us'] - tr['charge_tde_us'],
             fs_minus_chain_t0_us=fs0 - t0,
             beam_on_flash_axis_us=tr['tc_us'] - t0,
             offset_top_us=t0 - tr['charge_tde_us'],
             charge_bde_us=tr['charge_bde_us'],
             cert_crp=man[str(e)]['cert_crp'], cert_tick=man[str(e)]['cert_tick'])
    r['trig_tpc_tick'] = r['trig_in_tpc_us'] / 0.5
    rows.append(r)
keys = list(rows[0])
w = csv.writer(sys.stdout, delimiter='\t')
w.writerow(keys)
for r in rows:
    w.writerow([f'{r[k]:.3f}' if isinstance(r[k], (float, np.floating)) else r[k] for k in keys])
