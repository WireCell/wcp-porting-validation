#!/usr/bin/env python3
"""PDHD per-event photon-detector (PD) activity: +x vs -x walls, by APA.

Question (run 27980): cosmics dominate, so per-event PD activity should be
roughly event-independent -- yet our toolkit processing showed the -x side lit
in only 11/31 events. Split the +x PDs into their two APAs and compare each
against the single self-streamed -x APA, to decide whether the -x data is weird.

Answer (see pdhd/docs/pdhd-pd-activity-per-event.md): at the OpHit level the -x
self-stream is the *uniform* side (lit every event); the toolkit's sparse -x is a
`decoana` decoding-coverage artifact. The real per-event anomaly is +x dropout.

Data source: `opflashana/PerOpHitTree` -- OpHit-level per-opdet PE, independent of
flash-finding (the right granularity for "total PE / # fired PDs"). The +x-dropout
cause is read from the raw streams `rawdump/raw_waveform` and `trigoff/trigger_offset`.

Geometry (verified via flashopdet/opdet_geo) -- four 40-PD APA blocks:
  +x upper  ch 0-39    x=+356mm  z=267.5-427.1mm  self-trigger snippet
  +x lower  ch 40-79   x=+356mm  z= 35.5-195.0mm  self-trigger snippet
  -x self   ch 80-119  x=-356mm  z=267.5-427.1mm  self-trigger snippet  <- the one -x APA
  -x full   ch 120-159 x=-356mm  z= 35.5-195.0mm  full-stream (continuous; not in PerOpHitTree)
The z-matched pair across the cathode is +x upper (0-39) <-> -x self (80-119).

Usage:  python pd_activity.py            # run 27980 (primary) + 29107 (cross-check)
Plots -> pdhd/pics/ (gitignored; regenerate with this script).
"""
import uproot, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PICS = '/home/xqian/work/scratch_wcgpu1/toolkit-dev/wcp-porting-img/pdhd/pics'

RUNS = {
    27980: '/nfs/data/1/xning/wirecell-working/data/hd/run027980/np04hd_raw_run027980_0000_dataflow0_datawriter_0_20240711T160710_final.root',
    29107: '/nfs/data/1/xning/wirecell-working/data/hd/run029107/np04hd_raw_run029107_0004_dataflow7_datawriter_0_20240906T163315_final.root',
}

# four APA blocks: (label, lo, hi, colour)
BLOCKS = [
    ('+x up (0-39)',   0,  40, '#1f77b4'),
    ('+x lo (40-79)',  40, 80, '#17becf'),
    ('-x self (80-119)', 80, 120, '#d62728'),
]


def load_ophit(fn):
    """Per-event, per-block total PE and number of fired PDs from PerOpHitTree."""
    f = uproot.open(fn)
    t = f['opflashana/PerOpHitTree'].arrays(['EventID', 'OpChannel', 'PE'], library='np')
    ev, ch, pe = t['EventID'], t['OpChannel'], t['PE']
    # geometry cross-check of the side mapping (0-79 +x, 80-159 -x)
    geo = f['flashopdet/opdet_geo'].arrays(library='np')
    gx = {int(geo['opdet'][i]): geo['x'][i] for i in range(len(geo['opdet']))}
    assert all(gx[c] > 0 for c in range(80) if c in gx), 'side/x-sign mismatch (+x)'
    assert all(gx[c] < 0 for c in range(80, 160) if c in gx), 'side/x-sign mismatch (-x)'

    evs = np.unique(ev)
    out = {'events': evs}
    # active channels per block (channels that ever fire in this run)
    out['active'] = {lbl: len(np.unique(ch[(ch >= lo) & (ch < hi)]))
                     for lbl, lo, hi, _ in BLOCKS}
    for lbl, lo, hi, _ in BLOCKS:
        tot = np.zeros(len(evs)); nfir = np.zeros(len(evs), int)
        for i, e in enumerate(evs):
            m = (ev == e) & (ch >= lo) & (ch < hi)
            tot[i] = pe[m].sum()
            nfir[i] = len(np.unique(ch[m]))
        out[lbl] = {'pe': tot, 'nfired': nfir}
    return out


def load_raw(fn):
    """Per-event recorded-channel count per block + trigger metadata.

    Used only to diagnose the +x dropout: distinguishes 'not read out' (no raw
    snippet rows) from 'read out but quiet' (rows present, OpHit reco found none).
    """
    f = uproot.open(fn)
    rw = f['rawdump/raw_waveform'].arrays(['event', 'opch', 'nsamples'], library='np')
    re, rc = rw['event'], rw['opch']
    tg = f['trigoff/trigger_offset'].arrays(['event', 'tc_type', 'offset_us'], library='np')
    tmap = {int(tg['event'][i]): (int(tg['tc_type'][i]), float(tg['offset_us'][i]))
            for i in range(len(tg['event']))}
    evs = np.unique(re)
    rows = {}
    for e in evs:
        m = re == e
        o = rc[m]
        rows[int(e)] = dict(
            n_up=int(np.unique(o[(o >= 0) & (o < 40)]).size),
            n_lo=int(np.unique(o[(o >= 40) & (o < 80)]).size),
            n_xself=int(np.unique(o[(o >= 80) & (o < 120)]).size),
            n_xfull=int(np.unique(o[(o >= 120) & (o < 160)]).size),
            tc=tmap.get(int(e), (None, None))[0],
            off=tmap.get(int(e), (None, None))[1])
    return rows


# ---------------------------------------------------------------------------
print('Loading runs (PerOpHitTree)...')
D = {r: load_ophit(fn) for r, fn in RUNS.items()}

# ===== printed per-event table + dropout diagnosis (run 27980) =====
for run in RUNS:
    d = D[run]
    print(f'\n================= run {run} : per-event PD activity =================')
    print('active channels (ever fired): ' +
          ', '.join(f'{lbl}={d["active"][lbl]}/40' for lbl, *_ in BLOCKS))
    print(f'{"evt":>5} | ' + ' | '.join(f'{lbl} PE/nfir' for lbl, *_ in BLOCKS))
    for i, e in enumerate(d['events']):
        cells = [f'{d[lbl]["pe"][i]:8.0f}/{d[lbl]["nfired"][i]:2d}' for lbl, *_ in BLOCKS]
        print(f'{e:5d} | ' + ' | '.join(cells))

print('\n================= run 27980 : +x-dropout diagnosis (raw streams) =================')
raw = load_raw(RUNS[27980])
# Abnormal events = +x upper APA (0-39) readout dropout: <20 of 40 channels recorded.
DROPOUT = {27980: [e for e in sorted(raw) if raw[e]['n_up'] < 20]}
print(f'{"evt":>5} | {"+xUp":>4} {"+xLo":>4} {"-xSelf":>6} {"-xFull":>6} (recorded channels) | tc_type offset_us')
for e in sorted(raw):
    r = raw[e]
    flag = '  <-- +x DROPOUT' if e in DROPOUT[27980] else ''
    print(f'{e:5d} | {r["n_up"]:4d} {r["n_lo"]:4d} {r["n_xself"]:6d} {r["n_xfull"]:6d}'
          f' | {r["tc"]}   {r["off"]}{flag}')
print(f'  abnormal (+x dropout): {DROPOUT[27980]}')


def bright_outliers(d, k=4.0):
    """Events whose all-block total PE exceeds k x the run median (bright outliers)."""
    tot = sum(d[lbl]['pe'] for lbl, *_ in BLOCKS)
    med = np.median(tot)
    return [int(d['events'][i]) for i in range(len(tot)) if tot[i] > k * med]


def annotate_evts(ax, evlist, xpos, ylevel, color='#d62728'):
    """Drop a vertical guide + event-number label at each abnormal event."""
    for e in evlist:
        ix = int(np.where(xpos == e)[0][0]) if e in xpos else None
        if ix is None:
            continue
        ax.axvline(ix, color=color, ls='--', lw=1, alpha=0.5, zorder=0)
        ax.text(ix, ylevel, str(e), color=color, fontsize=8, fontweight='bold',
                ha='center', va='top', rotation=0,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color, lw=0.8))


# bright outliers per run (29107 has a very bright event; 27980 has none)
BRIGHT = {r: bright_outliers(D[r]) for r in RUNS}
print(f'\nbright-outlier events (all-block PE > 4x median): '
      + ', '.join(f'{r}:{BRIGHT[r]}' for r in RUNS))

# ============ PLOT a: per-event total PE, 3 series (run 27980) ============
d = D[27980]
ev = d['events']; x = np.arange(len(ev))
fig, ax = plt.subplots(figsize=(13, 5))
for lbl, lo, hi, col in BLOCKS:
    ax.plot(x, d[lbl]['pe'], 'o-', color=col, lw=1.5, ms=5, label=lbl)
annotate_evts(ax, DROPOUT[27980], ev, ax.get_ylim()[1] * 0.98)
ax.set_xticks(x); ax.set_xticklabels(ev, rotation=90, fontsize=7)
ax.set_xlabel('event'); ax.set_ylabel('total OpHit PE')
ax.set_title('PDHD run 27980 - per-event total PE by APA block (OpHit level)\n'
             f'-x self (80-119) lit every event; ABNORMAL +x dropout: evt {DROPOUT[27980]}')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{PICS}/pd_activity_27980_total_pe.png', dpi=110)
print('\nwrote pd_activity_27980_total_pe.png')

# ============ PLOT b: per-event # fired PDs, 3 series (run 27980) ============
fig, ax = plt.subplots(figsize=(13, 5))
for lbl, lo, hi, col in BLOCKS:
    ax.plot(x, d[lbl]['nfired'], 'o-', color=col, lw=1.5, ms=5,
            label=f'{lbl}  (active {d["active"][lbl]}/40)')
annotate_evts(ax, DROPOUT[27980], ev, ax.get_ylim()[1] * 0.55)
ax.set_xticks(x); ax.set_xticklabels(ev, rotation=90, fontsize=7)
ax.set_xlabel('event'); ax.set_ylabel('# fired PDs (distinct opdets w/ >=1 OpHit)')
ax.set_title('PDHD run 27980 - per-event fired-PD count by APA block (OpHit level)\n'
             f'-x self holds 34/34 active every event; ABNORMAL +x dropout: evt {DROPOUT[27980]}')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{PICS}/pd_activity_27980_nfired.png', dpi=110)
print('wrote pd_activity_27980_nfired.png')

# ============ PLOT c: z-matched test  -x self (80-119) vs opposite +x up (0-39) ============
up = d['+x up (0-39)']; xs = d['-x self (80-119)']
abn = np.isin(ev, DROPOUT[27980])   # abnormal = +x-dropout events
norm = ~abn
fig, axc = plt.subplots(1, 2, figsize=(13, 5.2))
# left: scatter PE, y=x line; abnormal events labelled by event number
axc[0].scatter(up['pe'][norm], xs['pe'][norm], c='#2ca02c', s=40, zorder=3, label='normal evt')
axc[0].scatter(up['pe'][abn], xs['pe'][abn], c='#d62728', s=60, marker='s', zorder=3,
               label='ABNORMAL (+x dropout)')
for i in np.where(abn)[0]:
    axc[0].annotate(str(ev[i]), (up['pe'][i], xs['pe'][i]), color='#d62728',
                    fontsize=8, fontweight='bold', xytext=(6, 4),
                    textcoords='offset points')
lim = max(up['pe'].max(), xs['pe'].max()) * 1.05
axc[0].plot([0, lim], [0, lim], 'k--', alpha=0.5, label='y = x')
axc[0].set_xlim(-lim * 0.02, lim); axc[0].set_ylim(0, lim)
axc[0].set_xlabel('+x upper (0-39) total PE'); axc[0].set_ylabel('-x self (80-119) total PE')
axc[0].set_title('z-matched APAs across the cathode\n(directly opposite: same z, opposite wall)')
axc[0].legend(); axc[0].grid(alpha=0.3)
# right: per-event ratio -x/+x (PE); abnormal events flagged where +x>0, dropped where +x=0
finite = up['pe'] > 0
ratio = np.full(len(ev), np.nan); ratio[finite] = xs['pe'][finite] / up['pe'][finite]
xr = np.arange(len(ev))
axc[1].plot(xr[norm], ratio[norm], 'o-', color='#2ca02c', lw=1.3, ms=5, label='normal')
axc[1].scatter(xr[abn & finite], ratio[abn & finite], c='#d62728', s=60, marker='s',
               zorder=4, label='abnormal (+x dropout)')
for i in np.where(abn & finite)[0]:
    axc[1].annotate(str(ev[i]), (xr[i], ratio[i]), color='#d62728', fontsize=8,
                    fontweight='bold', xytext=(5, 3), textcoords='offset points')
axc[1].axhline(1.0, color='k', ls='--', alpha=0.6)
med = np.median(ratio[norm])
axc[1].axhline(med, color='#1f77b4', ls=':', label=f'normal-evt median = {med:.2f}')
axc[1].set_xticks(xr); axc[1].set_xticklabels(ev, rotation=90, fontsize=7)
axc[1].set_xlabel('event'); axc[1].set_ylabel('PE ratio  -x self / +x upper')
axc[1].set_title('-x tracks its opposite +x APA (ratio ~ O(1), stable)\n'
                 '=> -x is NOT anomalous; spikes are the +x-dropout events')
axc[1].legend(); axc[1].grid(alpha=0.3)
fig.suptitle('PDHD run 27980 - "is -x weird?" z-matched comparison')
fig.tight_layout(); fig.savefig(f'{PICS}/pd_activity_27980_zmatched.png', dpi=110)
print('wrote pd_activity_27980_zmatched.png')

# ============ PLOT d: 29107 cross-check (total PE + # fired) ============
d2 = D[29107]; ev2 = d2['events']; x2 = np.arange(len(ev2))
fig, axd = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for lbl, lo, hi, col in BLOCKS:
    axd[0].plot(x2, d2[lbl]['pe'], 'o-', color=col, lw=1.3, ms=4, label=lbl)
    axd[1].plot(x2, d2[lbl]['nfired'], 'o-', color=col, lw=1.3, ms=4,
                label=f'{lbl} (active {d2["active"][lbl]}/40)')
for e in BRIGHT[29107]:
    ix = int(np.where(ev2 == e)[0][0])
    axd[0].annotate(f'evt {e}\nbright outlier', (ix, sum(d2[lbl]['pe'][ix] for lbl, *_ in BLOCKS) * 0.6),
                    color='#d62728', fontsize=8, fontweight='bold', ha='center',
                    arrowprops=dict(arrowstyle='->', color='#d62728'),
                    xytext=(ix + 3, axd[0].get_ylim()[1] * 0.7))
    axd[0].axvline(ix, color='#d62728', ls='--', lw=1, alpha=0.5)
    axd[1].axvline(ix, color='#d62728', ls='--', lw=1, alpha=0.5)
axd[0].set_ylabel('total OpHit PE'); axd[0].legend(); axd[0].grid(alpha=0.3)
axd[0].set_title('PDHD run 29107 cross-check - per-event PD activity by APA block '
                 f'(no +x dropout; bright outlier evt {BRIGHT[29107]})')
axd[1].set_ylabel('# fired PDs'); axd[1].legend(); axd[1].grid(alpha=0.3)
axd[1].set_xticks(x2); axd[1].set_xticklabels(ev2, rotation=90, fontsize=7)
axd[1].set_xlabel('event')
fig.tight_layout(); fig.savefig(f'{PICS}/pd_activity_29107_crosscheck.png', dpi=110)
print('wrote pd_activity_29107_crosscheck.png')

# ============ PLOT e: per-PD PE distribution by block (both runs) ============
fig, axe = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
bins = np.logspace(-1, 5, 50)
for ax_, run in zip(axe, RUNS):
    f = uproot.open(RUNS[run])
    t = f['opflashana/PerOpHitTree'].arrays(['OpChannel', 'PE'], library='np')
    ch, pe = t['OpChannel'], t['PE']
    for lbl, lo, hi, col in BLOCKS:
        m = (ch >= lo) & (ch < hi)
        if m.sum():
            ax_.hist(pe[m], bins=bins, histtype='step', lw=2, color=col,
                     label=f'{lbl} (med {np.median(pe[m]):.0f})')
    ax_.set_xscale('log'); ax_.set_yscale('log')
    ax_.set_xlabel('PE per OpHit'); ax_.set_title(f'run {run}'); ax_.legend(fontsize=8)
axe[0].set_ylabel('OpHits (log)')
fig.suptitle('PDHD per-PD (per-OpHit) PE by APA block - the PDs that fire look normal on both walls')
fig.tight_layout(); fig.savefig(f'{PICS}/pd_activity_per_pd_pe.png', dpi=110)
print('wrote pd_activity_per_pd_pe.png')

print('\nDONE.')
