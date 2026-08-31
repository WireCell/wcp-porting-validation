#!/usr/bin/env python3
"""doc pr/111 sec 13 -- two checks the sec 12 numbers do not settle on their own.

(A) VALIDATION.  sec 12.1 measures the prototype's argmax OFFLINE, on a cloud
    rebuilt from the end-of-job T_rec_charge; sec 12.2 has the prototype's LIVE
    logged "NeutrinoID_DL: DNN:" point for the same runs.  Joining them per event
    is the one place ground truth for the reconstruction exists.  Aggregate
    agreement is not enough -- the same EVENTS must move.

(B) PROPAGATION.  The prototype gates the DL pick at dl_vtx_cut = 2.0 cm
    (wire-cell-prod-nue-port.cxx:40): a pick further than that from the nearest
    candidate is rejected and the traditional vertex stands.  So an unstable DNN
    point need not reach the output.  T_tagger.nu_{x,y,z} is the vertex that
    actually ships; this measures how much of the instability survives the gate,
    in BOTH codes.

Repro:  python3 scripts/pr111_ub_propagate.py
"""
import os, sys, glob, re, math
import numpy as np
import uproot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr111_scn_lib import infer

SW = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/qlport/scripts/sweep'
WCP = '/home/xqian/tmp/pr111_wcp'


def live_dnn(arm, ev):
    g = glob.glob(f'{WCP}/{arm}/{ev}/5384_{ev}.log')
    if not g:
        return None, None
    t = open(g[0], errors='ignore').read()
    m = re.findall(r'NeutrinoID_DL: DNN:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', t)
    dec = 'change' if 'Change to DL vertex!' in t else ('stay' if 'Stay with the traditional' in t else 'none')
    return (np.array(list(map(float, m[-1]))) if m else None), dec


def nu_vtx(path):
    try:
        a = uproot.open(path)['T_tagger'].arrays(['nu_x', 'nu_y', 'nu_z'], library='np')
        return np.array([a['nu_x'][0], a['nu_y'][0], a['nu_z'][0]])
    except Exception:
        return None


def offline_peak(path):
    a = uproot.open(path)['T_rec_charge'].arrays(['x', 'y', 'z', 'q'], library='np')
    c = (a['x'].astype(np.float32), a['y'].astype(np.float32),
         a['z'].astype(np.float32), a['q'].astype(np.float32))
    if len(c[0]) < 20:
        return None
    return np.array(infer(c, top_k=2)[0][:3])


def main():
    wcp = {arm: {re.search(r'_(\d+)\.root$', p).group(1): p
                 for p in glob.glob(f'{WCP}/{arm}/*/nue_5384_*.root')} for arm in ('on', 'off')}
    wct = {arm: {re.search(r'_(\d+)\.root$', p).group(1): p
                 for p in glob.glob(f'{SW}/pr111_wct_{arm}_dl/*/track_com_5384_*.root')}
           for arm in ('on', 'off')}
    evts = sorted(set(wcp['on']) & set(wcp['off']) & set(wct['on']) & set(wct['off']), key=int)

    print("(A) prototype: LIVE logged DNN move vs OFFLINE reconstructed argmax move, per event")
    print(f"{'evt':>6s} {'live dDNN':>10s} {'offline dpeak':>13s} {'agree(<2cm)':>11s}  decision ON->OFF")
    agree = both_big = both_small = n = 0
    for e in evts:
        lo, do = live_dnn('on', e); lf, df = live_dnn('off', e)
        po, pf = offline_peak(wcp['on'][e]), offline_peak(wcp['off'][e])
        if lo is None or lf is None or po is None or pf is None:
            continue
        dl = float(np.linalg.norm(lo - lf)); dp = float(np.linalg.norm(po - pf))
        n += 1
        ok = abs(dl - dp) < 2.0
        agree += ok
        both_big += (dl > 2 and dp > 2); both_small += (dl <= 2 and dp <= 2)
        if dl > 2 or dp > 2 or not ok:
            print(f"{e:>6s} {dl:10.2f} {dp:13.2f} {'yes' if ok else 'NO':>11s}  {do}->{df}")
    print(f"\n  n={n}: |live - offline| < 2 cm on {agree}/{n};  "
          f"both > 2 cm on {both_big}, both <= 2 cm on {both_small} "
          f"(same-class {both_big+both_small}/{n})")

    print("\n(B) does it reach the SHIPPED vertex?  T_tagger.nu_{x,y,z}, exclusion ON vs OFF")
    for code, arms in (('prototype', wcp), ('toolkit', wct)):
        ds = []
        for e in evts:
            a, b = nu_vtx(arms['on'][e]), nu_vtx(arms['off'][e])
            if a is None or b is None:
                continue
            ds.append(float(np.linalg.norm(a - b)))
        d = np.array(ds)
        print(f"  {code:>9s}: n={len(d)}  final nu vertex move median={np.median(d):.2f} cm, "
              f">2 cm {int((d>2).sum())}/{len(d)} ({100*(d>2).mean():.0f} %), "
              f">20 cm {int((d>20).sum())}/{len(d)}")


if __name__ == '__main__':
    main()
