#!/usr/bin/env python3
"""PDHD light run comparison from the TOOLKIT (WCT-native) flashes -- the
toolkit analogue of make_flash_stats.py (which reads the LArSoft flash_opdet
trees).  Same five views (flashes/event, PDs/flash, total-PE/flash, per-PD PE,
overview), but every flash here is reconstructed by our own chain.

All flashes here are the NO-CUT population (OpFlashFinder min_fired_pds/
min_total_pe disabled) so the full per-flash multiplicity/PE spectrum is
visible -- the production config restores the 5-PD / 20-PE quality cut.

Per-run source (one opflash tar.gz per event; matrix tensor [nflash, 1+160],
col 0 = time ns, cols 1..160 = PE per opdet):
  27980  work/027980_allpd*_nocut/opflash_pdhd-allpd-wct.tar.gz  WCT-native ALL-PD,
                                                            full -x wall (snippet 0-119
                                                            + full stream 120-159 merged)
  27305  work/027305_<n>/opflash_pdhd-wct.tar.gz            WCT-native +x (−x dark);
                                                            its dirs predate the cut
                                                            (already no-cut)
  29107  work/029107_allpd*_nocut/opflash_pdhd-allpd-wct.tar.gz  WCT-native ALL-PD,
                                                            full -x wall.  The file was
                                                            re-extracted with a full
                                                            rawdump (160 ch), so 29107 is
                                                            now reconstructed end-to-end
                                                            like 27980 (was previously
                                                            sparse ch 0-39 / no rawdump).

Side mapping: opdet 0-79 = +x, 80-159 = -x.  Plots -> pdhd/pics/ (gitignored).
"""
import glob, io, re, tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = '/nfs/data/1/xqian/toolkit-dev/toolkit/pdhd/work'
PICS = '/nfs/data/1/xqian/toolkit-dev/toolkit/pdhd/pics'

# run -> (glob of per-event opflash files, short note)
SRC = {
    27980: (f'{WORK}/027980_allpd*_nocut/opflash_pdhd-allpd-wct.tar.gz', 'all-PD (full -x wall)'),
    27305: (f'{WORK}/027305_*/opflash_pdhd-wct.tar.gz',                  '+x only (-x dark)'),
    29107: (f'{WORK}/029107_allpd*_nocut/opflash_pdhd-allpd-wct.tar.gz', 'all-PD (full -x wall)'),
}
COL = {27305: '#1f77b4', 27980: '#ff7f0e', 29107: '#2ca02c'}
PX, MX, MIXED = '+x (0-79)', '-x (80-159)', 'mixed'


def opmat(path):
    tf = tarfile.open(path, 'r:gz')
    for n in tf.getnames():
        if n.endswith('_0_array.npy'):
            return np.load(io.BytesIO(tf.extractfile(n).read()))
    return None


def numeric_event_dirs(pattern, run):
    """Keep one file per clean numeric event dir (drop _chk and suffixed dirs)."""
    rp = '%06d' % run
    out = []
    for f in sorted(glob.glob(pattern)):
        d = f.split('/')[-2]                       # e.g. 027980_allpd32_nocut or 027305_14
        tail = d[len(rp) + 1:]                      # after "027980_"
        if re.fullmatch(r'(allpd)?\d+(_nocut)?', tail):  # numeric (optionally allpd-/nocut-tagged)
            out.append(f)
    return out


def load(run):
    pattern, note = SRC[run]
    files = numeric_event_dirs(pattern, run)
    flashes, pe_px, pe_mx, nev = [], [], [], 0
    for f in files:
        m = opmat(f)
        if m is None or m.shape[0] == 0:
            continue
        nev += 1
        pe = m[:, 1:161]
        for r in range(pe.shape[0]):
            row = pe[r]
            a = row[0:80].sum(); b = row[80:160].sum()
            side = MIXED if (a > 0 and b > 0) else (PX if a > 0 else MX)
            flashes.append(dict(n_pd=int((row > 0).sum()), total_pe=float(row.sum()),
                                side=side, event=nev))
            pe_px.extend(row[0:80][row[0:80] > 0].tolist())
            pe_mx.extend(row[80:160][row[80:160] > 0].tolist())
    return dict(run=run, nev=nev, note=note, flashes=flashes,
                pe_px=np.array(pe_px), pe_mx=np.array(pe_mx))


def arr(d, key, side=None):
    return np.array([f[key] for f in d['flashes'] if side is None or f['side'] == side])


def fper_event(d, side=None):
    from collections import Counter
    c = Counter()
    for f in d['flashes']:
        if side is None or f['side'] == side:
            c[f['event']] += 1
    evs = set(f['event'] for f in d['flashes'])
    return np.array([c.get(e, 0) for e in sorted(evs)]) if evs else np.array([0])


def pct(a, q):
    return np.percentile(a, q) if len(a) else float('nan')


print('Loading runs (WCT-native opflash)...')
D = {r: load(r) for r in SRC}
runs = list(SRC)

print('\n================= SUMMARY (toolkit WCT) =================')
print(f"{'run':>6} {'nev':>4} {'flash':>6} {'f/evt':>6} | {'+x':>6} {'-x':>6} {'mix':>5} | "
      f"{'PE_med':>7} {'PE_p90':>7} {'PE_max':>9} | {'nPD_med':>7} {'nPD_p90':>7} {'nPD_max':>7} | source")
for r in runs:
    d = D[r]
    npx = sum(1 for f in d['flashes'] if f['side'] == PX)
    nmx = sum(1 for f in d['flashes'] if f['side'] == MX)
    nmix = sum(1 for f in d['flashes'] if f['side'] == MIXED)
    tpe = arr(d, 'total_pe'); npd = arr(d, 'n_pd')
    print(f"{r:>6} {d['nev']:>4} {len(d['flashes']):>6} {fper_event(d).mean():>6.1f} | "
          f"{npx:>6} {nmx:>6} {nmix:>5} | "
          f"{np.median(tpe):>7.1f} {pct(tpe,90):>7.0f} {tpe.max():>9.0f} | "
          f"{np.median(npd):>7.0f} {pct(npd,90):>7.0f} {int(npd.max()):>7} | {d['note']}")

# ---- PLOT 1: flashes per event, side split ----
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(runs)); w = 0.6
pxm = [fper_event(D[r], PX).mean() for r in runs]
mxm = [fper_event(D[r], MX).mean() for r in runs]
mixm = [fper_event(D[r], MIXED).mean() for r in runs]
ax.bar(x, pxm, w, label='+x (0-79)', color='#4878d0')
ax.bar(x, mxm, w, bottom=pxm, label='-x (80-159)', color='#ee854a')
ax.bar(x, mixm, w, bottom=np.array(pxm) + np.array(mxm), label='mixed', color='#999999')
for i, r in enumerate(runs):
    tot = pxm[i] + mxm[i] + mixm[i]
    ax.text(i, tot + max(pxm) * 0.02, f'{tot:.0f}/evt\n{D[r]["nev"]} evt', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f'run {r}\n{D[r]["note"]}' for r in runs], fontsize=8)
ax.set_ylabel('mean flashes per event')
ax.set_title('PDHD light (TOOLKIT WCT, NO-CUT): flashes per event by wall\n'
             '27980 & 29107 = all-PD full -x wall; 27305 +x only (-x dark)')
ax.legend(); fig.tight_layout()
fig.savefig(f'{PICS}/light_runcmp_wct_flashes_per_event.png', dpi=110)
print('\nwrote light_runcmp_wct_flashes_per_event.png')

# ---- PLOT 2: PDs per flash ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, r in zip(axes, runs):
    d = D[r]
    npx = arr(d, 'n_pd', PX); nmx = arr(d, 'n_pd', MX); nmix = arr(d, 'n_pd', MIXED)
    bins = np.arange(0.5, 82, 1)
    ax.hist(npx, bins=bins, histtype='step', lw=2, color='#4878d0', label=f'+x ({len(npx)})')
    if len(nmx):
        ax.hist(nmx, bins=bins, histtype='step', lw=2, color='#ee854a', label=f'-x ({len(nmx)})')
    if len(nmix):
        ax.hist(nmix, bins=bins, histtype='step', lw=2, color='#999999', label=f'mixed ({len(nmix)})')
    ax.set_yscale('log'); ax.set_xlabel('PDs per flash')
    ax.set_title(f'run {r} — {d["note"]}', fontsize=9); ax.legend(fontsize=8)
axes[0].set_ylabel('flashes (log)')
fig.suptitle('PDHD light (TOOLKIT WCT): photon detectors per flash, by wall')
fig.tight_layout()
fig.savefig(f'{PICS}/light_runcmp_wct_pds_per_flash.png', dpi=110)
print('wrote light_runcmp_wct_pds_per_flash.png')

# ---- PLOT 3: total PE per flash ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
pe_bins = np.logspace(-1, 6, 50)
for ax, r in zip(axes, runs):
    d = D[r]
    for side, c in [(PX, '#4878d0'), (MX, '#ee854a'), (MIXED, '#999999')]:
        a = arr(d, 'total_pe', side)
        if len(a):
            ax.hist(a, bins=pe_bins, histtype='step', lw=2, color=c,
                    label=f'{side.split()[0]} (med {np.median(a):.0f})')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('total PE per flash')
    ax.set_title(f'run {r}', fontsize=9); ax.legend(fontsize=8)
axes[0].set_ylabel('flashes (log)')
fig.suptitle('PDHD light (TOOLKIT WCT): total PE per flash, by wall')
fig.tight_layout()
fig.savefig(f'{PICS}/light_runcmp_wct_total_pe.png', dpi=110)
print('wrote light_runcmp_wct_total_pe.png')

# ---- PLOT 4: per-PD PE ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
ppe_bins = np.logspace(-1, 5, 50)
for ax, r in zip(axes, runs):
    d = D[r]
    ax.hist(d['pe_px'], bins=ppe_bins, histtype='step', lw=2, color='#4878d0',
            label=f'+x (med {np.median(d["pe_px"]):.0f})' if len(d['pe_px']) else '+x (none)')
    if len(d['pe_mx']):
        ax.hist(d['pe_mx'], bins=ppe_bins, histtype='step', lw=2, color='#ee854a',
                label=f'-x (med {np.median(d["pe_mx"]):.0f})')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('PE per opdet (in a flash)')
    ax.set_title(f'run {r}', fontsize=9); ax.legend(fontsize=8)
axes[0].set_ylabel('opdet hits (log)')
fig.suptitle('PDHD light (TOOLKIT WCT): per-PD PE within flashes, by wall')
fig.tight_layout()
fig.savefig(f'{PICS}/light_runcmp_wct_pe_per_pd.png', dpi=110)
print('wrote light_runcmp_wct_pe_per_pd.png')

print('\nDONE.')
