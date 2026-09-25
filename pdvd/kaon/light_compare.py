#!/usr/bin/env python3
"""doc pdvd/118 sec 4: light conditions, run 39305 (kaon candidates) vs the
production light chain on 039349 / 039252 (same runner, same defaults; refs
re-made under work/<run6>_light<evt>_d118ref by run_light_evt.sh -s _d118ref).

Per event, from the opflash archive + the rawwf (for the time base):
  chain t0 (PDVDOpWaveformSource), full-stream window, trigger (tc_us) on the
  flash axis; flash rate (>= PECUT PE) inside the full-stream window; cosmic
  flash PE quantiles (beam +-BEAMWIN excluded); per-family PE share of bright
  cosmic flashes; fraction of flashes with a saturated channel; the flash
  nearest the trigger (dt, PE, family shares).
Writes kaon/out/light_events.tsv and prints a per-run summary; --fig writes
docs/pics/d118_light_*.png."""
import argparse, glob, io, json, os, tarfile
import numpy as np, uproot

KDIR = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.dirname(KDIR)
PECUT = 100.0
BEAMWIN = 5.0     # us
FAM = {1: 'cathode', 2: 'membrane', 3: 'pmt'}

def refs():
    out = []
    for run, pat in ((39349, 'np02vd_raw_run039349_0035_*_rawwf.root'),
                     (39252, 'np02vd_raw_run039252_1176_*_rawwf.root')):
        p = glob.glob(f'{PDVD}/input_data_light/{pat}')[0]
        for e in uproot.open(p)['trigoff/trigger_offset'].arrays(['event'], library='np')['event']:
            out.append((run, int(e), p, f'{PDVD}/work/{run:06d}_light{e}_d118ref/opflash_pdvd-wct.tar.gz'))
    return out

def kaons():
    out = []
    for p in sorted(glob.glob(f'{PDVD}/input_kaon/light/np02vd_raw_run039305_evt*_rawwf.root')):
        e = int(p.split('_evt')[1].split('_')[0])
        out.append((39305, e, p, f'{PDVD}/work/039305_light{e}/opflash_pdvd-wct.tar.gz'))
    return out

_wf_cache = {}
def timebase(p, e):
    if p not in _wf_cache:
        f = uproot.open(p)
        _wf_cache[p] = (f['rawdump/raw_waveform'].arrays(['event', 'opchannel', 'opdet', 'nsamp', 'timestamp'], library='np'),
                        f['trigoff/trigger_offset'].arrays(library='np'))
    w, tr = _wf_cache[p]
    m = w['event'] == e
    ns, ts = w['nsamp'][m], w['timestamp'][m]
    t0 = np.min(np.where(ns <= 1024, ts - 64 / 62.5, ts))
    fs = ns > 1024
    k = np.where(tr['event'] == e)[0][0]
    fam = {}
    for od, ch in zip(w['opdet'][m], w['opchannel'][m]):
        fam[int(od)] = FAM[int(ch) // 1000]
    return dict(t0=t0, fs_lo=ts[fs].min() - t0, fs_hi=ts[fs].min() - t0 + ns[fs].max() * 0.016,
                trig=tr['tc_us'][k] - t0, tc_type=int(tr['tc_type'][k])), fam

def flashes(path):
    t = tarfile.open(path)
    arr = {}
    for m in t.getmembers():
        if m.name.endswith('_array.npy'):
            idx = int(m.name.split('_')[-2])
            arr[idx] = np.load(io.BytesIO(t.extractfile(m).read()))
    return arr[0], arr[3]          # opflash (time ns, 40 PE), flash_sat (40)

def one(run, e, rawp, flp):
    tb, fam = timebase(rawp, e)
    fl, sat = flashes(flp)
    tus = fl[:, 0] / 1000.0
    pe = fl[:, 1:]
    tot = pe.sum(1)
    inwin = (tus >= tb['fs_lo']) & (tus < tb['fs_hi'])
    beam = np.abs(tus - tb['trig']) < BEAMWIN
    cos = inwin & ~beam
    bright = cos & (tot >= 1000)
    fams = np.array([fam.get(i, '?') for i in range(40)])
    share = {f: (pe[bright][:, fams == f].sum() / max(pe[bright].sum(), 1)) for f in FAM.values()}
    i = int(np.abs(tus - tb['trig']).argmin())
    bshare = {f: pe[i, fams == f].sum() / max(tot[i], 1) for f in FAM.values()}
    win_ms = (tb['fs_hi'] - tb['fs_lo']) / 1000
    return dict(run=run, event=e, tc_type=tb['tc_type'], nflash=len(tot),
                rate_per_ms=((tot >= PECUT) & cos).sum() / win_ms,
                rate_all_per_ms=cos.sum() / win_ms,
                pe_p50=np.median(tot[cos & (tot >= PECUT)]), pe_p90=np.percentile(tot[cos & (tot >= PECUT)], 90),
                sat_frac=(sat[cos & (tot >= PECUT)].sum(1) > 0).mean(),
                share_cath=share['cathode'], share_mem=share['membrane'], share_pmt=share['pmt'],
                trig_us=tb['trig'], beam_dt_us=tus[i] - tb['trig'], beam_pe=tot[i],
                beam_share_cath=bshare['cathode'], beam_share_mem=bshare['membrane'], beam_share_pmt=bshare['pmt'],
                n_in_beamwin=int(beam.sum()), n_in_beamwin_ge100=int((beam & (tot >= PECUT)).sum()),
                offset_top_us=None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig', action='store_true')
    a = ap.parse_args()
    rows = [one(*x) for x in kaons() + refs()]
    os.makedirs(f'{KDIR}/out', exist_ok=True)
    keys = [k for k in rows[0] if k != 'offset_top_us']
    with open(f'{KDIR}/out/light_events.tsv', 'w') as f:
        f.write('\t'.join(keys) + '\n')
        for r in rows:
            f.write('\t'.join(f'{r[k]:.4g}' if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
    print(f'{"run":>6} {"n":>3} {"rate>=100/ms":>13} {"rate_all/ms":>11} {"PE p50":>7} {"PE p90":>8} {"sat%":>5} '
          f'{"cath":>5} {"mem":>5} {"pmt":>5} | {"beam<5us":>8} {"|dt|med":>7} {"beamPE med":>10}')
    for run in (39305, 39349, 39252):
        R = [r for r in rows if r['run'] == run]
        g = lambda k: np.array([r[k] for r in R], float)
        close = np.abs(g('beam_dt_us')) < BEAMWIN
        print(f'{run:>6} {len(R):>3} {g("rate_per_ms").mean():>6.1f}+-{g("rate_per_ms").std():<5.1f} '
              f'{g("rate_all_per_ms").mean():>11.1f} {np.median(g("pe_p50")):>7.0f} {np.median(g("pe_p90")):>8.0f} '
              f'{100*g("sat_frac").mean():>5.1f} {g("share_cath").mean():>5.2f} {g("share_mem").mean():>5.2f} '
              f'{g("share_pmt").mean():>5.2f} | {close.sum():>3}/{len(R):<4} {np.median(np.abs(g("beam_dt_us")[close])) if close.any() else np.nan:>7.2f} '
              f'{np.median(g("beam_pe")[close]) if close.any() else np.nan:>10.0f}')
    if a.fig:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for run, c in ((39305, 'C3'), (39349, 'C0'), (39252, 'C2')):
            allpe = []
            for x in [x for x in kaons() + refs() if x[0] == run]:
                tb, _ = timebase(x[2], x[1]); fl, _ = flashes(x[3])
                t = fl[:, 0] / 1000; tot = fl[:, 1:].sum(1)
                m = (t >= tb['fs_lo']) & (t < tb['fs_hi']) & (np.abs(t - tb['trig']) >= BEAMWIN)
                allpe.append(tot[m])
            allpe = np.concatenate(allpe)
            nev = len([x for x in kaons() + refs() if x[0] == run])
            ax[0].hist(allpe, bins=np.logspace(0, 5.5, 60), histtype='step', color=c,
                       weights=np.full(len(allpe), 1 / nev), label=f'{run:06d} ({nev} evts)')
            R = [r for r in rows if r['run'] == run]
            ax[1].scatter([r['beam_dt_us'] for r in R], [r['beam_pe'] for r in R], color=c, label=f'{run:06d}', s=18)
            ax[2].scatter([r['share_cath'] for r in R], [r['share_pmt'] for r in R], color=c, label=f'{run:06d}', s=18)
        ax[0].set(xscale='log', yscale='log', xlabel='cosmic flash total PE (beam +-5 us excluded)', ylabel='flashes / event')
        ax[1].set(yscale='log', xlabel='nearest flash time - trigger (us)', ylabel='PE', xlim=(-60, 60),
                  title='flash nearest the trigger')
        ax[1].axvspan(-BEAMWIN, BEAMWIN, color='0.9')
        ax[2].set(xlabel='cathode-XA PE share (bright cosmic flashes)', ylabel='PMT PE share')
        for x in ax: x.legend(fontsize=8)
        fig.tight_layout()
        os.makedirs(f'{PDVD}/docs/pics', exist_ok=True)
        fig.savefig(f'{PDVD}/docs/pics/d118_light_compare.png', dpi=110)
        print('wrote docs/pics/d118_light_compare.png')

if __name__ == '__main__':
    main()
