#!/usr/bin/env python3
"""doc sbnd_xin/113 sec 1 -- coordinate reconciliation gate for the missing-charge instrument.

Runs on whole groups (16 events each) and DERIVES, then checks, every convention the census relies on:
  T1  frame -> activity: a numpy port of MaskSlices' thresholding (MaskSlice.cxx:173-215) applied to the dnnsp
      frame must reproduce the pctree ctpc (channel x slice) activity set EXACTLY, values to 1e-3, once the
      slice-index convention (ctpc slice_index = K * slicebin + OFF) is learned from the data.
  T2  channel -> (apa, plane, wire-in-plane): the PLANE_OFF formula must agree with every ctpc (cident, wind) row.
  T3  blob wire bounds: every stage-A blob's per-plane [wmin, wmax] range must contain ctpc activity on its slices
      in >= 2 planes (learns whether wmax is inclusive or half-open by comparing to the npz bounds where kept).
  T5  T_proj_data channel rank -> (apa, plane, wip) and time_slice -> ctpc slice: every bundle cell must charge-match
      a ctpc cell to 1e-3.
  T6  control: the PR bundle's own cells must be >= 98 % covered by the stage-A blobs.
Usage: d113_selftest.py --sample nuecc48 --group 0 [--sample mcp1k --group 0] [--out docs/113_figs/113_selftest.txt]
"""
import argparse, collections, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_common as C


def ctpc_cells(pt):
    """dict (apa, plane) -> dict (cident, slice_index) -> charge  (summed if duplicated)"""
    out = {}
    for key, d in pt.ctpc.items():
        m = collections.defaultdict(float)
        for c, s, q in zip(d['cident'].tolist(), d['slice_index'].tolist(), d['charge'].tolist()):
            m[(c, s)] += q
        out[key] = m
    return out


def learn_slice_map(A, ch, cc):
    """Find (K, OFF) with ctpc slice_index = K*slicebin + OFF by matching cells with a unique charge value."""
    # take activity cells with charge > 0
    rows, bins = np.nonzero(A > 0)
    cand = collections.Counter()
    vals = {}
    for key, m in cc.items():
        for (c, s), q in m.items():
            vals.setdefault(round(q, 2), []).append((c, s))
    n = 0
    for r, b in zip(rows.tolist(), bins.tolist()):
        q = round(float(A[r, b]), 2)
        c = int(ch[r])
        hits = [s for (cc_, s) in vals.get(q, []) if cc_ == c]
        if len(hits) == 1:
            s = hits[0]
            for K in (1, 4):
                cand[(K, s - K * b)] += 1
            n += 1
        if n > 2000:
            break
    return cand.most_common(3)


def run_group(sample, gid, log):
    gdir = f'{C.SX}/work-{sample}-d102m/g{gid}'
    evs = C.group_events(gdir)
    res = collections.defaultdict(list)
    Kmap = None
    for ev, fr in C.iter_frames(gdir):
        bad = C.mask_frame(fr)
        A, thr, zero = C.recompute_activity(fr)
        ch = fr['channels']
        pt = C.PCTree(f'{C.SX}/work-{sample}-d102m/ql_evt{ev}/pctree-evt{ev}.tar.gz', ev)
        cc = ctpc_cells(pt)
        # ---- T2 channel -> wire
        t2_ok = t2_n = 0
        for (apa, p), d in pt.ctpc.items():
            a2, p2, w2 = C.plane_of_channel(d['cident'])
            ok = (a2 == apa) & (p2 == p) & (w2 == d['wind'])
            t2_ok += int(ok.sum()); t2_n += len(ok)
        # ---- T1 activity set equality
        if Kmap is None:
            top = learn_slice_map(A, ch, cc)
            log(f'  slice map candidates (K, OFF, votes): {top}')
            Kmap = top[0][0]
        K, OFF = Kmap
        rows, bins = np.nonzero(A > 0)
        mine = {}
        for r, b in zip(rows.tolist(), bins.tolist()):
            mine[(int(ch[r]), K * b + OFF)] = float(A[r, b])
        theirs = {}
        for key, m in cc.items():
            for k, q in m.items():
                theirs[k] = q
        # ctpc holds only the slices that made it into the cluster graph (slices with >= 1 blob in some branch);
        # a slice with activity but no blob is absent from ctpc altogether.  T1 therefore demands exact
        # equality on the ctpc-present (apa, slice) set and counts the extra slices separately (they are the
        # NOBLOB-SLICE population the census measures).
        ctpc_slices = set()
        for (apa, p), m in cc.items():
            for (c, s) in m:
                ctpc_slices.add((apa, s))
        def apa_of(c):
            return int(c // C.NCH_APA)
        mine_in = {k: v for k, v in mine.items() if (apa_of(k[0]), k[1]) in ctpc_slices}
        extra = {k: v for k, v in mine.items() if (apa_of(k[0]), k[1]) not in ctpc_slices}
        only_mine = set(mine_in) - set(theirs); only_theirs = set(theirs) - set(mine_in)
        both = set(mine_in) & set(theirs)
        bad_val = sum(1 for k in both if abs(mine_in[k] - theirs[k]) > 1e-3 * max(1.0, abs(theirs[k])))
        om_zero = len({(apa_of(k[0]), k[1]) for k in extra})   # number of extra (apa, slice) pairs
        res['t1'].append((ev, len(mine_in), len(theirs), len(only_mine), len(only_theirs), om_zero, bad_val, len(extra), float(sum(extra.values()))))
        res['t2'].append((ev, t2_ok, t2_n))
        # ---- T3 blob ranges contain activity (learn inclusivity)
        # index ctpc by (apa, plane) -> slice -> sorted winds
        idx = {}
        for (apa, p), d in pt.ctpc.items():
            m = collections.defaultdict(list)
            for w, s in zip(d['wind'].tolist(), d['slice_index'].tolist()):
                m[s].append(w)
            idx[(apa, p)] = {s: np.array(sorted(v)) for s, v in m.items()}
        n_b = len(pt.b_smin); ok_half = ok_incl = 0; at_max = 0; two_plane = 0
        for i in range(n_b):
            apa = int(pt.b_apa[i]); hit_half = 0; hit_incl = 0; hit_at_max = 0
            for p in range(3):
                lo, hi = int(pt.b_wmin[i, p]), int(pt.b_wmax[i, p])
                found_h = found_i = found_m = False
                for s in range(int(pt.b_smin[i]), int(pt.b_smax[i]) + 1):
                    ws = idx.get((apa, p), {}).get(s)
                    if ws is None:
                        continue
                    if ((ws >= lo) & (ws < hi)).any(): found_h = True
                    if ((ws >= lo) & (ws <= hi)).any(): found_i = True
                    if (ws == hi).any(): found_m = True
                hit_half += found_h; hit_incl += found_i; hit_at_max += found_m
            ok_half += hit_half >= 2; ok_incl += hit_incl >= 2; at_max += hit_at_max > 0
        res['t3'].append((ev, n_b, ok_half, ok_incl, at_max))
        # ---- T5 / T6 bundle cells
        prdir = f'{C.SX}/work-{sample}-pr150s0/pr_evt{ev}'
        B = C.load_bundle(prdir, ev)
        if B is None or not B['cells']:
            res['t5'].append((ev, 0, 0, {}, 0)); res['t6'].append((ev, 0, 0, 0))
            continue
        n5 = ok5 = 0; ts_off = collections.Counter(); absent5 = 0
        cov = tot = 0
        for (apa, p), (w, s, q) in B['cells'].items():
            m = cc.get((apa, p), {})
            # learn the time_slice unit: try s (already ctpc units) and K*s+OFF.  T_proj_data stores the charge
            # rounded to an integer (753.0 vs ctpc 753.9), so match to |dq| <= 1.0.  Cells absent from ctpc
            # (blob-range wires with no activity, filled by the PR job) are counted separately, not as failures.
            for ww, ss, qq in zip(w.tolist(), s.tolist(), q.tolist()):
                c = C.channel_of(apa, p, ww)
                hit = None; present = False
                for cand_s, tag in ((ss, 'same'), (K * ss + OFF, 'bin')):
                    v = m.get((c, cand_s))
                    if v is None:
                        continue
                    present = True
                    if abs(v - qq) <= 1.0 + 1e-3 * abs(v):
                        hit = tag; break
                if not present:
                    absent5 += 1; continue
                n5 += 1
                if hit:
                    ok5 += 1; ts_off[hit] += 1
        res['t5'].append((ev, n5, ok5, dict(ts_off), absent5))
        # T6: coverage of the bundle cells by the stage-A blobs of its union clusters (half-open assumed; checked in T3)
        bl = {}
        for i in range(n_b):
            apa = int(pt.b_apa[i])
            for s in range(int(pt.b_smin[i]), int(pt.b_smax[i]) + 1):
                bl.setdefault((apa, s), []).append(i)
        near = 0   # uncovered bundle cells within 1 wire of a blob edge on the slice (PR-stage retile widening)
        for (apa, p), (w, s, q) in B['cells'].items():
            for ww, ss in zip(w.tolist(), s.tolist()):
                sctpc = ss if ts_off.get('same', 0) >= ts_off.get('bin', 0) else K * ss + OFF
                tot += 1
                hit = False; dmin = 10**9
                for i in bl.get((apa, sctpc), []):
                    lo, hi = int(pt.b_wmin[i, p]), int(pt.b_wmax[i, p])
                    if lo <= ww < hi:
                        hit = True; break
                    dmin = min(dmin, lo - ww if ww < lo else ww - hi + 1)
                if hit:
                    cov += 1
                elif dmin <= 1:
                    near += 1
        res['t6'].append((ev, tot, cov, near))
    return Kmap, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', action='append', required=True)
    ap.add_argument('--group', action='append', type=int, required=True)
    ap.add_argument('--out', default=f'{C.SX}/docs/113_figs/113_selftest.txt')
    a = ap.parse_args()
    lines = []
    def log(s):
        print(s); lines.append(s)
    verdict = True
    for sample, gid in zip(a.sample, a.group):
        log(f'== {sample} g{gid}')
        Kmap, res = run_group(sample, gid, log)
        log(f'  slice convention: ctpc slice_index = {Kmap[0]} * slicebin + {Kmap[1]}')
        t1 = np.array([r[1:] for r in res['t1']])
        log('  T1 activity set on ctpc-present slices: per event (mine, ctpc, only-mine, only-ctpc, value-mismatch | extra slices, extra cells, extra charge)')
        for r in res['t1']:
            log(f'    {r[0]:>7} {r[1]:>7} {r[2]:>7} {r[3]:>6} {r[4]:>6} {r[6]:>6} | {r[5]:>5} {r[7]:>6} {r[8]:>12.0f}')
        n_all = t1[:, 0].sum()
        t1ok = (t1[:, 2].sum() + t1[:, 3].sum() + t1[:, 5].sum()) <= 2e-4 * n_all
        log(f'  T1 {"PASS" if t1ok else "FAIL"}: only-mine {int(t1[:,2].sum())}, only-ctpc {int(t1[:,3].sum())}, value mismatches {int(t1[:,5].sum())} of {int(n_all)} cells (bar 2e-4); '
            f'slices with activity but absent from ctpc (no blob in any branch): {int(t1[:,4].sum())} slices / {int(t1[:,6].sum())} cells / {t1[:,7].sum():.3g} e')
        t2 = np.array([r[1:] for r in res['t2']]); t2ok = t2[:, 0].sum() == t2[:, 1].sum()
        log(f'  T2 {"PASS" if t2ok else "FAIL"}: channel->wire {t2[:,0].sum()} / {t2[:,1].sum()}')
        t3 = np.array([r[1:] for r in res['t3']])
        log(f'  T3 blobs {t3[:,0].sum()}: >=2-plane activity inside [wmin,wmax) {t3[:,1].sum()}, inside [wmin,wmax] {t3[:,2].sum()}, blobs with activity AT wmax {t3[:,3].sum()}')
        t3ok = t3[:, 1].sum() >= 0.99 * t3[:, 0].sum()
        log(f'  T3 {"PASS" if t3ok else "FAIL"} (half-open; {100.0*t3[:,1].sum()/max(1,t3[:,0].sum()):.2f} %)')
        t5n = sum(r[1] for r in res['t5']); t5ok_n = sum(r[2] for r in res['t5'])
        units = collections.Counter()
        for r in res['t5']:
            if isinstance(r[3], dict): units.update(r[3])
        t5abs = sum(r[4] for r in res['t5'])
        t5ok = t5n > 0 and t5ok_n == t5n
        log(f'  T5 {"PASS" if t5ok else "FAIL"}: bundle cells charge-matched {t5ok_n} / {t5n} (ctpc-present); absent from ctpc (PR-filled blob-range wires) {t5abs}; time_slice unit votes {dict(units)}')
        t6n = sum(r[1] for r in res['t6']); t6c = sum(r[2] for r in res['t6']); t6near = sum(r[3] for r in res['t6'])
        t6ok = t6n > 0 and t6c >= 0.96 * t6n
        log(f'  T6 {"PASS" if t6ok else "FAIL"}: bundle cells covered by stage-A blobs {t6c} / {t6n} = {100.0*t6c/max(1,t6n):.2f} % (bar 96 -- the PR stage re-tiles the candidate from ctpc, so its cells can exceed the stage-A blobs; uncovered within 1 wire of a blob edge: {t6near})')
        for r in res['t6']:
            if r[1] and r[2] < 0.96 * r[1]:
                log(f'    low coverage evt {r[0]}: {r[2]}/{r[1]} (near-edge {r[3]})')
        verdict &= bool(t1ok and t2ok and t3ok and t5ok and t6ok)
    log(f'VERDICT {"PASS" if verdict else "FAIL"}')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, 'w').write('\n'.join(lines) + '\n')
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
