#!/usr/bin/env python3
# doc sbnd_xin/pr/150: fork by duplication (CLAUDE.md M10) of scripts/analysis/pr149/pr149_metrics.py (untouched): metrics and
# outputs under docs/pr/150_figs; the pr/149 strata file 149_stage2_selection.tsv and the d102mpr metrics are read
# from 149_figs on purpose (the epoch-reference strata, never re-drawn).
"""doc pr/149: per-event metrics of a PR arm, and the pre-registered A/B tables.

Implements 149_figs/149_pred.txt secs 2, 4 and 5 (the definitions there govern;
this file is their executable form).

  extract  --arm TAG --samples S [S ...]
      reads work-<S>-<TAG>/pr_evt*/ (TAG 'd102mpr' = the epoch reference) and writes
      149_figs/metrics/<TAG>-<S>.tsv, one row per event:
        calib-pr-evt*.json  main vertex, main-cluster segments (zig-zag, iso lengths,
                            stubs), Steiner points, muon dQ/dx
        tracking-pr.root    T_proj_data coverage of the main cluster, T_kine pi0 / MCS
        nusel-evt*.tsv      per-bundle tgm/stm/fc (PR output) keyed main_id:flash_gid
        pr_scores_table.py  event_label, nu_evaluated, scores, Enu, wall/core/RSS
  compare  --a TAG --b TAG --samples S [S ...] [--ref d102mpr] [--movers out.tsv]
      prints the Q1 table per stratum (ISO / ISO-strong / control / owner cases) and
      the Q2 table, and optionally writes every Q2 mover for adjudication.

zig-zag definitions follow doc pr/73's zigzag_census.py (ratio = path/chord;
rms_dr = rms of the residual about the chord along drift x; fold = fraction of
interior turns > 30 deg), computed on the calib dump's fitted points instead of
the Bee track_fit layer (same Segment::fits()).
"""
import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np

SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
FIGS = os.path.join(SX, 'docs/pr/150_figs')
MET = os.path.join(FIGS, 'metrics')
sys.path.insert(0, SX)

OWNER_CASES = {('nuecc48', 42280), ('nuecc48', 271851), ('nuecc48', 350186), ('nuecc48', 137238),
               ('ncpi0', 21073), ('mcp1k', 57903), ('mcp1k', 56463), ('mcp1k', 284794),
               ('mcp1k', 59899), ('mcp1k', 58717)}

FIELDS = ['sample', 'event', 'run', 'rc', 'has_calib', 'main_cid', 'vx', 'vy', 'vz',
          'n_steiner_main', 'n_steiner_term', 'n_seg', 'n_stub', 'zz_len', 'zz_ratio', 'zz_rms_dr', 'zz_fold',
          'zzi_len', 'zzi_ratio', 'zzi_rms_dr',
          'iso_len75', 'iso_len80', 'iso_len_any75', 'iso_len_any85', 'mu_dqdx_med', 'mu_npts',
          'uncov', 'uncov_q', 'pio_flag', 'pio_mass', 'mcs_energy', 'mcs_amb', 'mcs_range',
          'event_label', 'nu_evaluated', 'numu_score', 'nue_score', 'Enu', 'wall_s', 'core_s',
          'maxrss_kb', 'bundles']


# ----------------------------------------------------------------------------- extract
def seg_geom(seg):
    P = np.array([[p['x'], p['y'], p['z']] for p in seg.get('points', [])], dtype=float)
    n = len(P)
    if n < 2:
        return dict(n=n, length=0.0, chord=0.0, theta=float('nan'), ratio=float('nan'),
                    rms_dr=float('nan'), fold=float('nan'))
    steps = np.diff(P, axis=0)
    sl = np.linalg.norm(steps, axis=1)
    length = float(sl.sum())
    c = P[-1] - P[0]
    chord = float(np.linalg.norm(c))
    if chord <= 0:
        return dict(n=n, length=length, chord=0.0, theta=float('nan'), ratio=float('nan'),
                    rms_dr=float('nan'), fold=float('nan'))
    u = c / chord
    theta = float(np.degrees(np.arccos(min(1.0, abs(u[0])))))
    R = (P - P[0]) - np.outer((P - P[0]) @ u, u)
    rms_dr = float(np.sqrt(np.mean(R[:, 0] ** 2)))
    good = sl > 1e-9
    st = steps[good] / sl[good, None]
    if len(st) >= 2:
        cosang = np.clip(np.sum(st[1:] * st[:-1], axis=1), -1, 1)
        fold = float(np.mean(np.degrees(np.arccos(cosang)) > 30.0))
    else:
        fold = float('nan')
    return dict(n=n, length=length, chord=chord, theta=theta, ratio=length / chord,
                rms_dr=rms_dr, fold=fold)


def wmean(vals, w):
    vals = np.asarray(vals, float)
    w = np.asarray(w, float)
    m = np.isfinite(vals)
    if not m.any() or w[m].sum() <= 0:
        return float('nan')
    return float(np.sum(vals[m] * w[m]) / w[m].sum())


def from_calib(path):
    d = json.load(open(path))
    mv = d.get('main_vertex') or {}
    cid = mv.get('cluster_id')
    out = dict(has_calib=1, main_cid=cid if cid is not None else -1,
               vx=mv.get('x', float('nan')), vy=mv.get('y', float('nan')), vz=mv.get('z', float('nan')),
               run=(d.get('meta') or {}).get('runNo', ''))
    deg = {v['id']: v.get('degree', 0) for v in d.get('vertices', [])}
    segs = [s for s in d.get('segments', []) if cid is not None and s.get('cluster_id') == cid]
    G = [(s, seg_geom(s)) for s in segs]
    out['n_seg'] = len(segs)
    out['n_stub'] = sum(1 for s, g in G if g['length'] < 3.0 and
                        (deg.get(s.get('start_vertex_id'), 0) == 1 or deg.get(s.get('end_vertex_id'), 0) == 1))
    zz = [(g['length'], g['ratio'], g['rms_dr'], g['fold']) for s, g in G if g['length'] >= 10 and g['n'] >= 10]
    out['zz_len'] = sum(z[0] for z in zz)
    out['zz_ratio'] = wmean([z[1] for z in zz], [z[0] for z in zz])
    out['zz_rms_dr'] = wmean([z[2] for z in zz], [z[0] for z in zz])
    out['zz_fold'] = wmean([z[3] for z in zz], [z[0] for z in zz])
    zi = [(g['length'], g['ratio'], g['rms_dr']) for s, g in G
          if g['length'] >= 10 and g['n'] >= 10 and g['theta'] >= 75]
    out['zzi_len'] = sum(z[0] for z in zi)
    out['zzi_ratio'] = wmean([z[1] for z in zi], [z[0] for z in zi])
    out['zzi_rms_dr'] = wmean([z[2] for z in zi], [z[0] for z in zi])

    def iso(T, showers):
        return sum(g['length'] for s, g in G if g['length'] >= 10 and g['theta'] >= T
                   and (showers or not s.get('flag_shower')))
    out['iso_len75'] = iso(75, False)
    out['iso_len80'] = iso(80, False)
    out['iso_len_any75'] = iso(75, True)
    out['iso_len_any85'] = iso(85, True)
    out['n_steiner_main'] = sum(len(t.get('x', [])) for t in d.get('steiner', []) if t.get('cluster_id') == cid)
    out['n_steiner_term'] = sum(sum(t.get('flag_terminal', [])) for t in d.get('steiner', []) if t.get('cluster_id') == cid)
    dq = []
    for s, g in G:
        if abs(s.get('particle_id', 0)) == 13 and g['length'] >= 20:
            pts = s.get('points', [])[5:-5]
            dq += [p['dQ'] / p['dx'] for p in pts if p.get('dx', 0) > 0]
    out['mu_dqdx_med'] = float(np.median(dq)) if dq else float('nan')
    out['mu_npts'] = len(dq)
    return out


def from_root(path, cid):
    import uproot
    import awkward as ak
    out = dict(uncov=float('nan'), uncov_q=float('nan'), pio_flag='', pio_mass=float('nan'),
               mcs_energy=float('nan'), mcs_amb=float('nan'), mcs_range=float('nan'))
    try:
        f = uproot.open(path)
    except Exception:  # noqa: BLE001
        return out
    if 'T_proj_data' in f and cid is not None and cid >= 0:
        t = f['T_proj_data'].arrays(library='ak')
        if len(t) > 0:
            ids = list(t['cluster_id'][0])
            if cid in ids:
                i = ids.index(cid)
                q = ak.to_numpy(t['charge'][0][i]).astype(float)
                p = ak.to_numpy(t['charge_pred'][0][i]).astype(float)
                if q.sum() > 0:
                    out['uncov'] = float(q[p < 0.1 * q].sum() / q.sum())
                    out['uncov_q'] = float(q.sum())
    if 'T_kine' in f:
        k = f['T_kine'].arrays(library='np')
        if len(k['kine_reco_Enu']) > 0:
            out['pio_flag'] = int(k['kine_pio_flag'][0])
            out['pio_mass'] = float(k['kine_pio_mass'][0])
            for b, key in (('kine_mcs_energy', 'mcs_energy'), ('kine_mcs_ambiguity', 'mcs_amb'),
                           ('kine_mcs_range_energy', 'mcs_range')):
                if b in k:
                    out[key] = float(k[b][0])
    return out


def from_nusel(path):
    rows = {}
    if not os.path.exists(path):
        return ''
    with open(path) as fh:
        lines = [l.split() for l in fh if l.strip()]
    if not lines:
        return ''
    hdr = lines[0]
    ix = {h: i for i, h in enumerate(hdr)}
    for r in lines[1:]:
        if len(r) != len(hdr):
            continue
        rows[f"{r[ix['main_id']]}:{r[ix['flash_gid']]}"] = f"{r[ix['tgm']]},{r[ix['stm']]},{r[ix['fc']]}"
    return json.dumps(rows, sort_keys=True)


def extract(arm, sample):
    root = os.path.join(SX, f'work-{sample}-{arm}')
    evdirs = sorted(glob.glob(f'{root}/pr_evt*/'), key=lambda p: int(p.rstrip('/').split('pr_evt')[1]))
    tmp = f'/home/xqian/tmp/pr150/scores-{arm}-{sample}.tsv'
    subprocess.run([sys.executable, os.path.join(SX, 'pr_scores_table.py'), '--root', root, '--sample', sample,
                    '--out', tmp], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    scores = {}
    with open(tmp) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            scores[int(r['event'])] = r
    os.makedirs(MET, exist_ok=True)
    out = os.path.join(MET, f'{arm}-{sample}.tsv')
    with open(out, 'w') as fo:
        fo.write('\t'.join(FIELDS) + '\n')
        for d in evdirs:
            e = int(d.rstrip('/').split('pr_evt')[1])
            rec = {k: '' for k in FIELDS}
            rec.update(sample=sample, event=e, has_calib=0)
            rcf = os.path.join(d, 'rc.txt')
            rec['rc'] = open(rcf).read().strip().replace('rc=', '') if os.path.exists(rcf) else ''
            cp = os.path.join(d, f'calib-pr-evt{e}.json')
            cid = None
            if os.path.exists(cp):
                rec.update(from_calib(cp))
                cid = rec['main_cid']
            rec.update(from_root(os.path.join(d, 'tracking-pr.root'), cid))
            rec['bundles'] = from_nusel(os.path.join(d, f'nusel-evt{e}.tsv'))
            s = scores.get(e, {})
            rec['event_label'] = s.get('event_label', '')
            rec['nu_evaluated'] = s.get('nu_evaluated', '')
            rec['numu_score'] = s.get('numu_score', '')
            rec['nue_score'] = s.get('nue_score', '')
            rec['Enu'] = s.get('kine_reco_Enu_MeV', '')
            rec['wall_s'] = s.get('wall_s', '')
            rec['core_s'] = s.get('core_s', '')
            rec['maxrss_kb'] = s.get('maxrss_kb', '')
            if not rec['run']:
                rec['run'] = s.get('run', '')
            fo.write('\t'.join(fmt(rec[k]) for k in FIELDS) + '\n')
    print(f'extract {arm} {sample}: {len(evdirs)} events -> {out}')


def fmt(v):
    if isinstance(v, float):
        return 'nan' if not math.isfinite(v) else f'{v:.6g}'
    return str(v)


# ----------------------------------------------------------------------------- compare
def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def load(arm, samples):
    out = {}
    for s in samples:
        p = os.path.join(MET, f'{arm}-{s}.tsv')
        with open(p) as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                out[(s, int(r['event']))] = r
    return out


def labels():
    from vtx_rules import vtx_io
    lab = {}
    for doc in vtx_io.load_labels(tags=vtx_io.TAGS_VTX105):
        if doc['truth'] is not None:
            lab[doc['eventNo']] = (doc['truth'], doc.get('runNo'))
    return lab


def vdist(r, click):
    if r is None or r.get('has_calib') != '1':
        return float('inf')
    v = (fnum(r['vx']), fnum(r['vy']), fnum(r['vz']))
    if not all(math.isfinite(c) for c in v):
        return float('inf')
    return math.dist(v, click)


def med(x):
    x = [v for v in x if math.isfinite(v)]
    return (float(np.median(x)), len(x)) if x else (float('nan'), 0)


def q1_table(A, B, keys, lab, name):
    both = [k for k in keys if A.get(k, {}).get('has_calib') == '1' and B.get(k, {}).get('has_calib') == '1']

    def d(col):
        return med([fnum(B[k].get(col)) - fnum(A[k].get(col)) for k in both])
    lines = [f'  [{name}] events={len(keys)} with calib in both={len(both)}']
    for col in ('n_steiner_main', 'n_steiner_term', 'zz_rms_dr', 'zz_ratio', 'zz_fold', 'zzi_rms_dr', 'zzi_ratio', 'uncov',
                'mu_dqdx_med'):
        m, n = d(col)
        ma, _ = med([fnum(A[k].get(col)) for k in both])
        mb, _ = med([fnum(B[k].get(col)) for k in both])
        lines.append(f'    {col:15s} median A {ma:10.4g}  B {mb:10.4g}  median(B-A) {m:+.4g}  (n={n})')
    for col in ('n_seg', 'n_stub'):
        sa = sum(int(fnum(A[k][col])) for k in both if math.isfinite(fnum(A[k][col])))
        sb = sum(int(fnum(B[k][col])) for k in both if math.isfinite(fnum(B[k][col])))
        lines.append(f'    {col:15s} sum A {sa}  B {sb}')
    lk = [k for k in keys if k[1] in lab]
    da = {k: vdist(A.get(k), lab[k[1]][0]) for k in lk}
    db = {k: vdist(B.get(k), lab[k[1]][0]) for k in lk}
    tow = sum(1 for k in lk if db[k] < da[k] - 1.0)
    away = sum(1 for k in lk if db[k] > da[k] + 1.0)
    lines.append(f'    vertex labelled={len(lk)}  A n<=1cm {sum(v <= 1 for v in da.values())} n<=3cm '
                 f'{sum(v <= 3 for v in da.values())} | B n<=1cm {sum(v <= 1 for v in db.values())} n<=3cm '
                 f'{sum(v <= 3 for v in db.values())} | toward {tow} away {away}')
    return lines, dict(both=both, tow=tow, away=away, da=da, db=db)


def wp(r):
    s = dict(nue7=False, nue43=False, numu09=False)
    if r is None or r.get('nu_evaluated') != '1':
        return s
    ne, nm = fnum(r['nue_score']), fnum(r['numu_score'])
    s['nue7'] = ne > 7.0
    s['nue43'] = ne > 4.3
    s['numu09'] = nm > 0.9
    return s


def compare(a):
    A = load(a.a, a.samples)
    B = load(a.b, a.samples)
    R = load(a.ref, a.samples)
    lab = labels()
    keys = sorted(set(A) | set(B))
    print(f'# pr149 compare A={a.a} B={a.b} ref={a.ref} samples={a.samples} events A={len(A)} B={len(B)}')
    miss = sorted(set(A) ^ set(B))
    if miss:
        print(f'# WARNING events in one arm only: {miss[:10]} ({len(miss)})')
    iso = [k for k in keys if k in R and R[k]['has_calib'] == '1' and fnum(R[k]['iso_len_any75']) >= 10]
    isos = [k for k in keys if k in R and R[k]['has_calib'] == '1' and fnum(R[k]['iso_len_any85']) >= 10]
    ctl = [k for k in keys if k in R and R[k]['has_calib'] == '1' and fnum(R[k]['iso_len_any75']) == 0]
    own = [k for k in keys if k in OWNER_CASES]
    if a.manifest_strata:
        st = {}
        with open(a.manifest_strata) as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                st.setdefault(r['stratum'], []).append((r['sample'], int(r['event'])))
        present = set(A) | set(B)          # a strata file may list samples not loaded here
        iso = sorted((set(st.get('iso', [])) | set(st.get('vtx_iso', []))) & present)
        isos = sorted(set(st.get('iso', [])) & present)
        ctl = sorted(set(st.get('control', [])) & present)
    print('## Q1')
    info = {}
    for name, ks in (('ISO', iso), ('ISO-strong', isos), ('control', ctl), ('owner-cases', own)):
        lines, info[name] = q1_table(A, B, ks, lab, name)
        print('\n'.join(lines))
    print('  owner cases, per event (A -> B):')
    for k in own:
        ra, rb = A.get(k, {}), B.get(k, {})
        cells = []
        for col in ('zz_rms_dr', 'zzi_rms_dr', 'uncov', 'n_seg', 'n_stub', 'n_steiner_main'):
            cells.append(f"{col} {fmt(fnum(ra.get(col)))}->{fmt(fnum(rb.get(col)))}")
        vd = ''
        if k[1] in lab:
            vd = f" vtx {vdist(ra, lab[k[1]][0]):.2f}->{vdist(rb, lab[k[1]][0]):.2f}cm"
        print(f'    {k[0]} {k[1]}: ' + '  '.join(cells) + vd)

    print('## Q2')
    mig = {}
    nuev = 0
    wpA = dict(nue7=0, nue43=0, numu09=0)
    wpB = dict(nue7=0, nue43=0, numu09=0)
    denu = []
    flips = dict(tgm=0, stm=0, fc=0)
    pio = dict(A_flag=0, B_flag=0, A_win=0, B_win=0)
    movers = []
    wall = []
    rss = []
    rcbad = []
    for k in keys:
        ra, rb = A.get(k), B.get(k)
        if ra is None or rb is None:
            continue
        why = []
        if ra['rc'] != '0' or rb['rc'] != '0':
            rcbad.append((k, ra['rc'], rb['rc']))
        if ra['event_label'] != rb['event_label']:
            mig[(ra['event_label'], rb['event_label'])] = mig.get((ra['event_label'], rb['event_label']), 0) + 1
            why.append(f"label {ra['event_label']}->{rb['event_label']}")
        if ra['nu_evaluated'] != rb['nu_evaluated']:
            nuev += 1
            why.append(f"nu_eval {ra['nu_evaluated']}->{rb['nu_evaluated']}")
        wa, wb = wp(ra), wp(rb)
        for w in wpA:
            wpA[w] += wa[w]
            wpB[w] += wb[w]
            if wa[w] != wb[w]:
                why.append(f'{w} {int(wa[w])}->{int(wb[w])}')
        if ra['nu_evaluated'] == '1' and rb['nu_evaluated'] == '1':
            denu.append(abs(fnum(rb['Enu']) - fnum(ra['Enu'])))
        ba = json.loads(ra['bundles']) if ra['bundles'] else {}
        bb = json.loads(rb['bundles']) if rb['bundles'] else {}
        for bk in sorted(set(ba) & set(bb)):
            ta, tb = ba[bk].split(','), bb[bk].split(',')
            for i, nm in enumerate(('tgm', 'stm', 'fc')):
                if ta[i] != tb[i]:
                    flips[nm] += 1
                    why.append(f'{nm} bundle {bk} {ta[i]}->{tb[i]}')
        if set(ba) != set(bb):
            why.append(f'bundle keys differ ({len(set(ba) ^ set(bb))})')
        for tag, r in (('A', ra), ('B', rb)):
            if r['pio_flag'] == '1':
                pio[f'{tag}_flag'] += 1
                if 100 < fnum(r['pio_mass']) < 170:
                    pio[f'{tag}_win'] += 1
        if k[1] in lab:
            dA, dB = vdist(ra, lab[k[1]][0]), vdist(rb, lab[k[1]][0])
            if dB > dA + 1.0:
                why.append(f'vertex ADVERSE {dA:.2f}->{dB:.2f}cm')
            elif dB < dA - 1.0:
                why.append(f'vertex toward {dA:.2f}->{dB:.2f}cm')
        if fnum(ra['wall_s']) > 0 and fnum(rb['wall_s']) > 0:
            wall.append(fnum(rb['wall_s']) / fnum(ra['wall_s']))
        if fnum(ra['maxrss_kb']) > 0 and fnum(rb['maxrss_kb']) > 0:
            rss.append(fnum(rb['maxrss_kb']) / fnum(ra['maxrss_kb']))
        if why:
            stratum = 'ISO' if k in iso else ('control' if k in ctl else 'other')
            movers.append((k, stratum, why))
    print(f'  event_label migrations: {sum(mig.values())}  ' + ', '.join(f'{x}->{y}:{n}' for (x, y), n in sorted(mig.items())))
    print(f'  nu_evaluated flips: {nuev}')
    print(f'  working points A/B: ' + ', '.join(f'{w} {wpA[w]}/{wpB[w]}' for w in wpA))
    if denu:
        print(f'  |dEnu| MeV over both-evaluated (n={len(denu)}): median {np.median(denu):.2f}  p90 {np.percentile(denu, 90):.2f}  '
              f'nonzero {sum(1 for v in denu if v > 1e-3)}')
    print(f'  tagger flips (bundle-matched): TGM {flips["tgm"]}  STM {flips["stm"]}  FC {flips["fc"]}'
          + ('   *** TGM FLIPS NONZERO: STOP-AND-REPORT ***' if flips['tgm'] else ''))
    print(f'  pi0 kine_pio_flag==1 A/B {pio["A_flag"]}/{pio["B_flag"]}  mass in (100,170) A/B {pio["A_win"]}/{pio["B_win"]}')
    if wall:
        print(f'  resources B/A: wall median {np.median(wall):.3f} p90 {np.percentile(wall, 90):.3f}  '
              f'RSS median {np.median(rss):.3f} p90 {np.percentile(rss, 90):.3f}')
    print(f'  nonzero rc: {rcbad if rcbad else 0}')
    print(f'  movers: {len(movers)}  (ISO {sum(1 for m in movers if m[1] == "ISO")}, control '
          f'{sum(1 for m in movers if m[1] == "control")}, other {sum(1 for m in movers if m[1] == "other")})')
    if a.movers:
        with open(a.movers, 'w') as fo:
            fo.write('sample\tevent\tstratum\twhat\n')
            for k, s, why in movers:
                fo.write(f'{k[0]}\t{k[1]}\t{s}\t{"; ".join(why)}\n')
    if a.print_movers:
        for k, s, why in movers:
            print(f'    {k[0]} {k[1]} [{s}] ' + '; '.join(why))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--arm', required=True)
    e.add_argument('--samples', nargs='+', required=True)
    c = sub.add_parser('compare')
    c.add_argument('--a', required=True)
    c.add_argument('--b', required=True)
    c.add_argument('--samples', nargs='+', required=True)
    c.add_argument('--ref', default='d102mpr')
    c.add_argument('--movers')
    c.add_argument('--print-movers', action='store_true')
    c.add_argument('--manifest-strata', help='149_stage2_selection.tsv: use its strata instead of the ref rule')
    a = ap.parse_args()
    if a.cmd == 'extract':
        for s in a.samples:
            extract(a.arm, s)
    else:
        compare(a)


if __name__ == '__main__':
    main()
