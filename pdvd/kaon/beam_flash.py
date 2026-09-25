#!/usr/bin/env python3
"""doc pdvd/118 sec 5: which flash is the beam flash, and which cluster does
Q/L matching give it?  Per kaon event, from our own products only:
  - beam time on the chain flash axis = tc_us - chain t0 (light_compare.py
    time base, per event from the trigoff tree);
  - calib-evt<ID>.json (clustering + Q/L, run_kaon_clus_scratch.sh): flash
    'time' = chain time + trigger_offsets_us[0] (top crate, +13.507 us pull);
  - Q/L matched pairs = auto_selected bundles;
  - jjo's certification (MANIFEST.csv cert_crp / cert_tick, TPC tick of the
    deposit, 0.5 us) -> expected raw x = anode_x - cert_tick*0.5us*v.
A cluster "hits the cert" if >= MINPTS of its points lie within +-DX cm of the
cert raw x (the deposit's drift coordinate before t0 correction).
Writes kaon/out/beam_flash.tsv."""
import csv, json, os, sys
import numpy as np
KDIR = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.dirname(KDIR)
DX, MINPTS, WIN = 5.0, 20, 5.0
man = {int(r['event']): r for r in csv.DictReader(open(f'{PDVD}/input_data_kaon/MANIFEST.csv'))}
lt = {int(r['event']): r for r in csv.DictReader(open(f'{KDIR}/out/light_events.tsv'), delimiter='\t') if r['run'] == '39305'}
rows = []
for e in sorted(man):
    d = json.load(open(f'{PDVD}/work/039305_{e}/calib-evt{e}.json'))
    v = d['drift_speed']                       # cm/us
    g = d['geometry']['4']; ax = g['anode_x']
    off = d['trigger_offsets_us'][0]
    t_beam = float(lt[e]['trig_us']) + off     # expected beam flash time in dump frame (us)
    fl = {f['id']: f for f in d['flashes']}
    near = sorted(d['flashes'], key=lambda f: abs(f['time'] - t_beam))
    bf = near[0] if abs(near[0]['time'] - t_beam) < WIN else None
    cert_tick = float(man[e]['cert_tick']); xc = ax - cert_tick * 0.5 * v
    cl = {c['uid']: c for c in d['clusters']}
    sel = [b for b in d['bundles'] if b['auto_selected']]
    matched_of = {b['main_cluster']: b['flash_id'] for b in sel}
    # clusters at the certified deposit (drift coordinate), any match
    hits = []
    for uid, c in cl.items():
        x = np.asarray(c['x']); n = int((np.abs(x - xc) < DX).sum())
        if n >= MINPTS: hits.append((uid, n, c['apa'], c['npoints']))
    hits.sort(key=lambda h: -h[1])
    bsel = [b for b in sel if bf is not None and b['flash_id'] == bf['id']]
    def cdesc(uid):
        c = cl[uid]; x = np.asarray(c['x']); q = np.asarray(c['q'])
        return x, q
    r = dict(event=e, cert_crp=man[e]['cert_crp'], cert_tick=int(cert_tick), cert_xraw=round(xc, 1),
             beam_flash_id=bf['id'] if bf else -1,
             beam_flash_dt=round(bf['time'] - t_beam, 2) if bf else None,
             beam_flash_pe=round(bf['total_PE']) if bf else None,
             n_matched_to_beam=len(bsel), matched_clusters='', matched_apa='', matched_xraw='', matched_depth_cm='',
             matched_npts='', matched_hits_cert='', matched_ks='', matched_pred_over_meas='',
             cert_clusters='', cert_cluster_matched_flash='', cert_cluster_flash_dt='')
    for b in bsel:
        x, q = cdesc(b['main_cluster'])
        xt = x + bf['time'] * v
        n_at = int((np.abs(x - xc) < DX).sum())
        r['matched_clusters'] += f"{b['main_cluster']} "
        r['matched_apa'] += f"{cl[b['main_cluster']]['apa']} "
        r['matched_xraw'] += f"[{x.min():.0f},{x.max():.0f}] "
        r['matched_depth_cm'] += f"[{ax - xt.max():.0f},{ax - xt.min():.0f}] "
        r['matched_npts'] += f"{len(x)} "
        r['matched_hits_cert'] += f"{n_at} "
        r['matched_ks'] += f"{b['ks_dis']:.2f} "
        r['matched_pred_over_meas'] += f"{b['total_pred_light'] / max(b['total_PE'], 1):.2f} "
    for uid, n, apa, npt in hits[:3]:
        fid = matched_of.get(uid)
        r['cert_clusters'] += f"{uid}(apa{apa},{n}/{npt}) "
        r['cert_cluster_matched_flash'] += f"{fid if fid is not None else '-'} "
        r['cert_cluster_flash_dt'] += (f"{fl[fid]['time'] - t_beam:+.0f} " if fid is not None and fid in fl else "- ")
    rows.append(r)
os.makedirs(f'{KDIR}/out', exist_ok=True)
w = csv.DictWriter(open(f'{KDIR}/out/beam_flash.tsv', 'w'), fieldnames=list(rows[0]), delimiter='\t')
w.writeheader(); [w.writerow(r) for r in rows]
for r in rows:
    print(' | '.join(f'{k}={v}' for k, v in r.items() if v not in ('', None)))
