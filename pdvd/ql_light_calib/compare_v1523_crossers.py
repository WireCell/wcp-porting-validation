#!/usr/bin/env python3
"""Compare cathode-crosser flash matches between the nominal (_vcal, v=1.586)
and the try-out (_v1523, v=1.523) reprocesses of run 039252.

A cathode crosser = one flash whose auto_selected bundles include BOTH a
bottom (apa 0) and a top (apa 4) cluster.  For each such flash we ask the
physics question the try-out tests: does the lower velocity move the crosser
onto a BRIGHTER flash (X-ARAPUCAs sit on the cathode, so a real crosser must
be bright)?

Usage: python3 compare_v1523_crossers.py            # all 18 events
       python3 compare_v1523_crossers.py 298567     # one event
"""
import json, glob, sys, os

PDVD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(tag, evt):
    p = f"{PDVD}/work/{evt[0]}_{tag}/calib-evt{evt[1]}.json"
    return json.load(open(p))

def crossers(d):
    """flash_gid -> {'pe':total_PE, 'bot':[uids], 'top':[uids]} for auto_selected
    flashes matched to at least one bottom(apa0) AND one top(apa4) cluster."""
    fl = {f['gid']: f for f in d['flashes']}
    m = {}
    for b in d['bundles']:
        if not b.get('auto_selected'):
            continue
        g = b['flash_gid']
        e = m.setdefault(g, {'bot': [], 'top': []})
        (e['bot'] if b['apa'] == 0 else e['top']).append(b['main_cluster'])
    out = {}
    for g, e in m.items():
        if e['bot'] and e['top']:
            f = fl.get(g, {})
            out[g] = {'t': f.get('time'), 'pe': f.get('total_PE'),
                      'bot': sorted(e['bot']), 'top': sorted(e['top'])}
    return out

def cluster_flash(d):
    """(uid,apa) -> (flash_gid, flash_time, flash_PE) for auto_selected bundles."""
    fl = {f['gid']: f for f in d['flashes']}
    out = {}
    for b in d['bundles']:
        if b.get('auto_selected'):
            f = fl.get(b['flash_gid'], {})
            out[(b['main_cluster'], b['apa'])] = (b['flash_gid'], f.get('time'), f.get('total_PE'))
    return out

def main():
    # index<->event map from the _vcal dumps present
    idx_evt = {}
    for p in sorted(glob.glob(f"{PDVD}/work/039252_*_vcal/calib-evt*.json")):
        d = os.path.dirname(p).split('/')[-1]          # 039252_<idx>_vcal
        run_idx = '_'.join(d.split('_')[:2])           # 039252_<idx>
        evtno = os.path.basename(p)[len('calib-evt'):-len('.json')]
        idx_evt[run_idx] = evtno
    want = sys.argv[1] if len(sys.argv) > 1 else None

    n_cross = n_brighter = n_dimmer = n_same = 0
    for run_idx, evtno in sorted(idx_evt.items(), key=lambda kv: int(kv[0].split('_')[1])):
        if want and evtno != want:
            continue
        vc = f"{PDVD}/work/{run_idx}_vcal/calib-evt{evtno}.json"
        v5 = f"{PDVD}/work/{run_idx}_v1523/calib-evt{evtno}.json"
        if not (os.path.exists(vc) and os.path.exists(v5)):
            continue
        dn, dv = json.load(open(vc)), json.load(open(v5))
        cn, cv = crossers(dn), crossers(dv)
        cfn, cfv = cluster_flash(dn), cluster_flash(dv)
        print(f"\n===== evt {evtno} ({run_idx}) =====")
        print(f"  nominal 1.586: {len(cn)} crosser-flashes | v1523 1.523: {len(cv)} crosser-flashes")
        # Track each nominal crosser's BOTTOM cluster into v1523 (bottom uid is stable-ish)
        for g, e in sorted(cn.items()):
            botu = e['bot'][0]
            new = cfv.get((botu, 0))
            oldpe = e['pe'] or 0
            if new is None:
                print(f"  gid{g} t={e['t']:.1f} PE={oldpe:.0f} bot={e['bot']} top={e['top']}"
                      f"  -> bot uid{botu} now UNMATCHED")
                continue
            ng, nt, npe = new
            npe = npe or 0
            tag = ""
            if ng != g:
                n_cross += 1
                if npe > 1.3*oldpe: tag, _=("BRIGHTER",n_brighter); n_brighter+=1
                elif npe < 0.77*oldpe: tag="dimmer"; n_dimmer+=1
                else: tag="~same"; n_same+=1
            dt = (nt - e['t']) if (nt is not None and e['t'] is not None) else None
            print(f"  gid{g} t={e['t']:.1f} PE={oldpe:.0f} bot={e['bot']} top={e['top']}"
                  f"  -> bot uid{botu} gid{ng} t={nt:.1f} PE={npe:.0f}"
                  f"  dPE={npe-oldpe:+.0f} dt={dt:+.1f}us {tag}")
    print(f"\n===== SUMMARY (bottom-cluster tracked, matched flash CHANGED) =====")
    print(f"  changed matches: {n_cross}   brighter(>1.3x): {n_brighter}   "
          f"dimmer(<0.77x): {n_dimmer}   ~same: {n_same}")

if __name__ == "__main__":
    main()
