#!/usr/bin/env python3
"""doc pr/111 sec 12 -- the owner's uBooNE 2x2, like-for-like and label-free.

  "I wonder if the MicroBooNE toolkit code (turn on and off this feature) vs.
   prototype running may help."           -- owner, 2026-08-22

Four arms, 35 uBooNE events, all with the DL vertex available:
  toolkit   pr111_wct_{on,off}_dl        (QL_FIT_EXCLUSION=true/false)
  prototype /home/xqian/tmp/pr111_wcp/{on,off}   (WCP_FIT_EXCLUSION=0 for off)

The comparison is done OFFLINE on the frozen CP24 weights with the SAME top-1
rule for both codes, which removes the live asymmetry (the toolkit runs
dl_vtx_rerank=true / top-5 composite / min_accept=4, the prototype runs top-1
argmax + dl_vtx_cut, and uboone-mabc.jsonnet exposes no rerank TLA).

Question: does fit_exclusion move the SCN argmax MORE in the toolkit than in the
prototype?  That is pr/110 (B) "the toolkit's exclusion is not WCP's exclusion"
against (A) "the net dislikes exclusion as such".

*** SCOPE CAVEAT, same as sec 8 ***  Both clouds come from each code's own
T_rec_charge, i.e. the END-OF-JOB fitted trajectory, not the DL-time cloud
(validated on SBND 46363: 887 tree points vs 731 harvested, 23 % exact match).
It is the same object on both sides, which is what a WCT-vs-WCP comparison needs,
but it does not reproduce either code's live DL decision.

Repro:  python3 scripts/pr111_ub_2x2.py
"""
import os, sys, glob, re
import numpy as np
import uproot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr111_scn_lib import infer

SW = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/qlport/scripts/sweep'
WCP = '/home/xqian/tmp/pr111_wcp'


def cloud(path):
    a = uproot.open(path)['T_rec_charge'].arrays(['x', 'y', 'z', 'q'], library='np')
    return (a['x'].astype(np.float32), a['y'].astype(np.float32),
            a['z'].astype(np.float32), a['q'].astype(np.float32))


def peak(c):
    v = infer(c, top_k=2)[0]
    return np.array(v[:3]), float(v[3])


def arms():
    out = {}
    for arm, pat in (('WCT_ON', f'{SW}/pr111_wct_on_dl/*/track_com_5384_*.root'),
                     ('WCT_OFF', f'{SW}/pr111_wct_off_dl/*/track_com_5384_*.root'),
                     ('WCP_ON', f'{WCP}/on/*/nue_5384_*.root'),
                     ('WCP_OFF', f'{WCP}/off/*/nue_5384_*.root')):
        d = {}
        for p in glob.glob(pat):
            m = re.search(r'_(\d+)\.root$', p)
            if m:
                d[m.group(1)] = p
        out[arm] = d
    return out


def main():
    A = arms()
    for k, v in A.items():
        print(f'# {k}: {len(v)} events', file=sys.stderr)
    evts = sorted(set(A['WCT_ON']) & set(A['WCT_OFF']) & set(A['WCP_ON']) & set(A['WCP_OFF']), key=int)
    print(f"# {len(evts)} events with all four arms")
    print(f"{'evt':>6s} | {'WCT n_on':>8s} {'n_off':>6s} {'cloud dr':>8s} {'S_on':>6s} {'S_off':>6s} {'argmax move':>11s} | "
          f"{'WCP n_on':>8s} {'n_off':>6s} {'cloud dr':>8s} {'S_on':>6s} {'S_off':>6s} {'argmax move':>11s}")
    mv = {'WCT': [], 'WCP': []}
    dr = {'WCT': [], 'WCP': []}
    for e in evts:
        row = [f'{e:>6s}']
        for code in ('WCT', 'WCP'):
            try:
                con, cof = cloud(A[f'{code}_ON'][e]), cloud(A[f'{code}_OFF'][e])
            except Exception:
                row.append(f" | {'ERR':>60s}"); continue
            if len(con[0]) < 20 or len(cof[0]) < 20:
                row.append(f" | {'too few pts':>60s}"); continue
            P = np.stack(con[:3], 1).astype(np.float64); Q = np.stack(cof[:3], 1).astype(np.float64)
            d = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).min(1)
            pon, son = peak(con); pof, sof = peak(cof)
            m = float(np.linalg.norm(pon - pof))
            mv[code].append(m); dr[code].append(float(np.median(d)))
            row.append(f" | {len(con[0]):8d} {len(cof[0]):6d} {np.median(d):8.3f} {son:6.3f} {sof:6.3f} {m:11.2f}")
        print(''.join(row))
    print()
    for code in ('WCT', 'WCP'):
        m = np.array(mv[code]); d = np.array(dr[code])
        if not len(m):
            continue
        print(f"{code}: n={len(m)}  cloud ON->OFF displacement median={np.median(d):.3f} cm  |  "
              f"SCN argmax move: median={np.median(m):.2f} cm, >2 cm in {int((m>2).sum())}/{len(m)} "
              f"({100*(m>2).mean():.0f} %), >20 cm in {int((m>20).sum())}/{len(m)}")


if __name__ == '__main__':
    main()
