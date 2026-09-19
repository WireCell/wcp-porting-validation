#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 sec 4 -- the STM verdict margin (doc pr/149 sec 10 item 1; doc pdvd/103 / 107 method).

TaggerCheckSTM's STM evaluation (clus/src/TaggerCheckSTM.cxx:2971-3003) accepts a candidate's dQ/dx window
only if, with ks1/ratio1 the KS distance and charge ratio to the stopping-muon template and ks2/ratio2 to a
flat MIP:
    (g)  ratio2 <= guard_ratio2_max                      (accept_guards; SBND production true, cap 2.0)
    (a)  ks1 - ks2 < 0                                    => margin_a = ks2 - ks1
    (b)  NOT ( r < 1.4  AND  s > -0.02 ),  r = sqrt((ks2/0.06)^2 + ((ratio2-1)/0.06)^2),
                                          s = ks1 - ks2 + (|ratio1-1| - |ratio2-1|)/1.5*0.3
                                                          => margin_b = max((r - 1.4)*0.06, -0.02 - s)
    then the residual-length / Michel-residual rules (not a shape boundary).
margin_shape = min(margin_a, margin_b): positive = the shape test passes by that much (ks units), negative =
fails by that much.  Every arm here carries tracking-stm.root (T_stm_eval per cluster per pass, verdict = the
pass's final answer), so the margin is known for EVERY evaluated candidate, not only the flipped ones.

Per (sample, event, cluster): the deciding eval record = the attempt with verdict 1 if any, else the attempt with
the largest margin_shape.  The CANDIDACY itself is T_stm_pass.status (doc pdvd/107 dictionary: 0 accepted, 1 TGM,
2 left-charge, 3 eval failed = flag_pass false, 4 mid-point trk, 5 proton, 6 default, 7 pre-fit exit, 8 no fit):
a cluster is an STM candidate iff any pass has status 0.  A verdict-1 eval attempt under a nonzero status means
a GUARD rejected the cluster after the shape test passed (evt 389538 cl 11: attempt 2 verdict 1, status 4).
Output: the margin distribution on all candidates of A and B, the paired |d ks1| / |d margin| on clusters
evaluated in both arms (the refit noise), the verdict flips with their margins in both arms, and the count of
flips whose margin in the LOSING arm lies within the noise p90 (a boundary flip) vs beyond it.
Usage: pr150_stm_margin.py --a pr150s0 --b pr150csp3bw --samples S... [--tsv out.tsv]
"""
import argparse, collections, glob, math, os, sys
import numpy as np
import uproot

SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
GUARD_RATIO2_MAX = 2.0
SNAME = {0: 'accepted', 1: 'TGM', 2: 'left-charge', 3: 'eval-failed', 4: 'mid-point-trk', 5: 'proton', 6: 'default',
         7: 'pre-fit-exit', 8: 'no-fit', -1: 'no-row'}


def margins(ks1, ks2, r1, r2):
    ma = ks2 - ks1
    r = math.sqrt((ks2 / 0.06) ** 2 + ((r2 - 1) / 0.06) ** 2)
    s = ks1 - ks2 + (abs(r1 - 1) - abs(r2 - 1)) / 1.5 * 0.3
    mb = max((r - 1.4) * 0.06, -0.02 - s)
    mg = GUARD_RATIO2_MAX - r2
    return ma, mb, mg, min(ma, mb)


def load(arm, samples):
    out = {}
    for s in samples:
        for d in sorted(glob.glob(f'{SX}/work-{s}-{arm}/pr_evt*')):
            ev = int(os.path.basename(d)[6:]); p = f'{d}/tracking-stm.root'
            if not os.path.exists(p):
                continue
            try:
                f = uproot.open(p)
                if 'T_stm_eval' not in f:
                    continue
                e = f['T_stm_eval'].arrays(library='np')
                sp = f['T_stm_pass'].arrays(library='np') if 'T_stm_pass' in f else {'cluster_id': [], 'status': [], 'pass': []}
            except Exception:  # noqa: BLE001
                continue
            recs = {}
            stat = {}
            for i in range(len(sp['cluster_id'])):
                k = (s, ev, int(sp['cluster_id'][i])); st = int(sp['status'][i])
                stat[k] = 0 if (st == 0 or stat.get(k) == 0) else (st if k not in stat else min(stat[k], st))
            for i in range(len(e['cluster_id'])):
                cl = int(e['cluster_id'][i]); rec = {k: float(e[k][i]) for k in ('ks1', 'ks2', 'ratio1', 'ratio2', 'res_length', 'ave_res_dqdx')}
                rec['verdict'] = int(e['verdict'][i]); rec['pass'] = int(e['pass'][i]); rec['strong'] = int(e['strong'][i])
                rec['m_a'], rec['m_b'], rec['m_g'], rec['m_shape'] = margins(rec['ks1'], rec['ks2'], rec['ratio1'], rec['ratio2'])
                cur = recs.get((s, ev, cl))
                if cur is None or (rec['verdict'] and not cur['verdict']) or (rec['verdict'] == cur['verdict'] and rec['m_shape'] > cur['m_shape']):
                    recs[(s, ev, cl)] = rec
            for k in list(stat):                        # a pass row with no eval attempt (pre-fit exit, no fit)
                if k not in recs:
                    recs[k] = dict(ks1=float('nan'), ks2=float('nan'), ratio1=float('nan'), ratio2=float('nan'), res_length=float('nan'),
                                   ave_res_dqdx=float('nan'), verdict=0, pass_=-1, strong=0, m_a=float('nan'), m_b=float('nan'), m_g=float('nan'), m_shape=float('nan'))
            for k, r in recs.items():
                r['status'] = stat.get(k, -1); r['cand'] = int(r['status'] == 0)
            out.update(recs)
    return out


def q(x, p):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if len(x) else float('nan')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--samples', nargs='+', required=True); ap.add_argument('--tsv', default=None)
    a = ap.parse_args()
    A, B = load(a.a, a.samples), load(a.b, a.samples)
    print(f'# pr150 STM margin A={a.a} B={a.b} samples={a.samples}: evaluated candidates A {len(A)} B {len(B)} common {len(set(A) & set(B))}')
    for lab, R in (('A', A), ('B', B)):
        st = collections.Counter(SNAME.get(r['status'], str(r['status'])) for r in R.values())
        print(f'{lab}: T_stm_pass status of the deciding pass: ' + ', '.join(f'{k} {v}' for k, v in st.most_common()))
        m = np.array([r['m_shape'] for r in R.values() if np.isfinite(r['m_shape'])]); v = np.array([r['verdict'] for r in R.values() if np.isfinite(r['m_shape'])])
        print(f'{lab}: verdict 1 {int(v.sum())} / 0 {int((v == 0).sum())}; margin_shape p10/p50/p90 {q(m, 10):+.4f} {q(m, 50):+.4f} {q(m, 90):+.4f}; '
              f'|margin_shape| < 0.02: {int((np.abs(m) < 0.02).sum())} ({100 * (np.abs(m) < 0.02).mean():.1f} %); '
              f'verdict 1 with margin_shape < 0.02: {int(((v == 1) & (m < 0.02)).sum())}; verdict 0 with margin_shape > 0 (failed later rules): {int(((v == 0) & (m > 0)).sum())}')
    common = sorted(set(A) & set(B))
    dks = np.array([abs(B[k]['ks1'] - A[k]['ks1']) for k in common]); dm = np.array([abs(B[k]['m_shape'] - A[k]['m_shape']) for k in common])
    same = [k for k in common if A[k]['verdict'] == B[k]['verdict']]
    dm_same = np.array([abs(B[k]['m_shape'] - A[k]['m_shape']) for k in same])
    noise = q(dm_same, 90)
    print(f'paired on common clusters ({len(common)}): |d ks1| p50/p90 {q(dks, 50):.4f}/{q(dks, 90):.4f}; |d margin_shape| p50/p90 {q(dm, 50):.4f}/{q(dm, 90):.4f}; '
          f'on the {len(same)} verdict-stable clusters |d margin_shape| p90 = {noise:.4f} (the refit noise)')
    flips = [k for k in common if A[k]['verdict'] != B[k]['verdict']]
    onlyA = [k for k in A if k not in B]; onlyB = [k for k in B if k not in A]
    print(f'verdict flips on common clusters: {len(flips)} (1->0 {sum(1 for k in flips if A[k]["verdict"])}, 0->1 {sum(1 for k in flips if B[k]["verdict"])}); '
          f'evaluated only in A {len(onlyA)} (verdict 1: {sum(A[k]["verdict"] for k in onlyA)}), only in B {len(onlyB)} (verdict 1: {sum(B[k]["verdict"] for k in onlyB)})')
    cf = [k for k in common if A[k]['cand'] != B[k]['cand']]
    trans = collections.Counter(f"{SNAME.get(A[k]['status'])} -> {SNAME.get(B[k]['status'])}" for k in cf)
    print(f'CANDIDACY flips (status 0 <-> other) on common clusters: {len(cf)} (lost {sum(A[k]["cand"] for k in cf)}, gained {sum(B[k]["cand"] for k in cf)}); '
          f'candidates only in A {sum(A[k]["cand"] for k in onlyA)}, only in B {sum(B[k]["cand"] for k in onlyB)}')
    for t, n in trans.most_common():
        print(f'   {n:4d}  {t}')
    ev3 = [k for k in cf if 3 in (A[k]['status'], B[k]['status'])]
    mlose = [(A[k] if A[k]['status'] == 3 else B[k])['m_shape'] for k in ev3]
    mlose = [x for x in mlose if np.isfinite(x)]
    print(f'   of the eval-failed (status 3) candidacy flips {len(ev3)}: losing-arm margin_shape within the noise ({noise:.4f}) {sum(1 for x in mlose if -noise <= x < 0)}, '
          f'beyond {sum(1 for x in mlose if x < -noise)}, positive (a residual/Michel rule) {sum(1 for x in mlose if x >= 0)}')
    within = beyond = later = 0
    rows = []
    for k in flips:
        lose = A[k] if A[k]['verdict'] else B[k]; win = B[k] if A[k]['verdict'] else A[k]
        kind = ('no-eval' if not np.isfinite(lose['m_shape']) else 'shape-boundary' if abs(lose['m_shape']) <= noise and lose['m_shape'] < 0
                else ('later-rule' if lose['m_shape'] > 0 else 'shape-beyond-noise'))
        within += kind == 'shape-boundary'; beyond += kind == 'shape-beyond-noise'; later += kind == 'later-rule'
        rows.append((k, A[k]['verdict'], B[k]['verdict'], A[k]['m_shape'], B[k]['m_shape'], A[k]['ks1'], B[k]['ks1'], A[k]['ks2'], B[k]['ks2'], kind))
    print(f'flip mechanism (losing arm): shape boundary within the noise {within}; shape fail beyond the noise {beyond}; shape passes, a later rule (residual / Michel) decides {later}')
    print('sample\tevent\tcluster\tverdict_A\tverdict_B\tmargin_A\tmargin_B\tks1_A\tks1_B\tks2_A\tks2_B\tkind')
    for r in sorted(rows):
        print('\t'.join([r[0][0], str(r[0][1]), str(r[0][2]), str(r[1]), str(r[2])] + [f'{x:+.4f}' for x in r[3:9]] + [r[9]]))
    if a.tsv:
        with open(a.tsv, 'w') as fh:
            fh.write('sample\tevent\tcluster\tstatus_A\tstatus_B\tverdict_A\tverdict_B\tmargin_A\tmargin_B\tks1_A\tks1_B\tks2_A\tks2_B\tratio2_A\tratio2_B\n')
            for k in sorted(set(A) | set(B)):
                ra, rb = A.get(k), B.get(k)
                g = lambda r, f: (f'{r[f]:.5f}' if r is not None and isinstance(r[f], float) else (str(r[f]) if r is not None else ''))
                fh.write('\t'.join([k[0], str(k[1]), str(k[2]), g(ra, 'status'), g(rb, 'status'), g(ra, 'verdict'), g(rb, 'verdict'), g(ra, 'm_shape'), g(rb, 'm_shape'),
                                     g(ra, 'ks1'), g(rb, 'ks1'), g(ra, 'ks2'), g(rb, 'ks2'), g(ra, 'ratio2'), g(rb, 'ratio2')]) + '\n')


if __name__ == '__main__':
    main()
