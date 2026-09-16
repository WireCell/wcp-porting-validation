#!/usr/bin/env python3
"""doc pr/149 round 2 -- POST HOC, descriptive: split the ISO-segment trajectory deviation into a
local JITTER and a large-scale BOW, per arm (doc pr/73 sec 4.7 made the same split on three events).

zzi_rms_dr (the pre-registered metric) is the rms drift-x residual about ONE chord per segment, so on
a 100-500 cm muon it mixes a point-to-point sawtooth with any smooth excursion of the whole segment.
Here, per main-cluster segment with length >= 10 cm, >= 10 fit points, chord angle to drift >= 75 deg
(the zzi selection), on the calib dump's fitted points (0.6 cm spacing):
    jit_x    rms of the second difference of drift x, / sqrt(6)   (white-noise amplitude, cm)
    jit_t    the same for the in-plane transverse coordinate (perpendicular to the chord and to x)
    bow_x    rms drift-x residual about the chord after a 9-point running mean (cm)
length-weighted per event.  Paired arm vs base over events with a value in both; sign test as in
r2_zzi_sign.py (|d| > 0.005 cm).

Usage: r2_local_jitter.py --stage {1,2} --pairs B:A ...    (arm = work-<s>-<arm> directory suffix)
"""
import argparse
import csv
import glob
import json
import math
import os

import numpy as np

SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'


def seg_terms(P):
    n = len(P)
    c = P[-1] - P[0]
    chord = float(np.linalg.norm(c))
    if n < 10 or chord <= 0:
        return None
    u = c / chord
    if math.degrees(math.acos(min(1.0, abs(u[0])))) < 75:
        return None
    length = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    if length < 10:
        return None
    ex = np.array([1.0, 0, 0])
    e_dr = ex - np.dot(ex, u) * u
    e_dr /= np.linalg.norm(e_dr)
    e_t = np.cross(u, e_dr)
    r = P - P[0]
    perp = r - np.outer(r @ u, u)
    x = perp @ e_dr
    t = perp @ e_t
    d2x = x[:-2] - 2 * x[1:-1] + x[2:]
    d2t = t[:-2] - 2 * t[1:-1] + t[2:]
    k = 9
    if n >= k:
        sm = np.convolve(x, np.ones(k) / k, mode='valid')
        bow = float(np.sqrt(np.mean(sm ** 2)))
    else:
        bow = float('nan')
    return dict(length=length, jit_x=float(np.sqrt(np.mean(d2x ** 2)) / math.sqrt(6)),
                jit_t=float(np.sqrt(np.mean(d2t ** 2)) / math.sqrt(6)), bow_x=bow)


def event_terms(calib):
    d = json.load(open(calib))
    mv = d.get('main_vertex') or {}
    cid = mv.get('cluster_id')
    if cid is None:
        return None
    rows = []
    for s in d.get('segments', []):
        if s.get('cluster_id') != cid:
            continue
        P = np.array([[p['x'], p['y'], p['z']] for p in s.get('points', [])], float)
        if len(P) < 10:
            continue
        g = seg_terms(P)
        if g:
            rows.append(g)
    if not rows:
        return None
    w = np.array([r['length'] for r in rows])
    out = {}
    for key in ('jit_x', 'jit_t', 'bow_x'):
        v = np.array([r[key] for r in rows])
        m = np.isfinite(v)
        out[key] = float(np.sum(v[m] * w[m]) / w[m].sum()) if m.any() else float('nan')
    return out


def arm_terms(arm, samples, keys):
    out = {}
    for s, e in keys:
        f = os.path.join(SX, f'work-{s}-{arm}', f'pr_evt{e}', f'calib-pr-evt{e}.json')
        if os.path.exists(f):
            t = event_terms(f)
            if t:
                out[(s, e)] = t
    return out


def sign_p(k, n):
    if n == 0:
        return float('nan')
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, choices=[1, 2], required=True)
    ap.add_argument('--pairs', nargs='+', required=True)
    a = ap.parse_args()
    if a.stage == 2:
        samples = ['mcp1k', 'mcp2k']
        keys = [(r['sample'], int(r['event'])) for r in
                csv.DictReader(open(os.path.join(SX, 'docs/pr/149_figs/149_stage2_selection.tsv')), delimiter='\t')
                if r['stratum'] == 'iso']
        label = 'Stage-2 iso stratum'
    else:
        samples = ['nuecc48', 'ncpi0']
        keys = []
        for s in samples:
            for r in csv.DictReader(open(os.path.join(SX, f'docs/pr/149_figs/metrics/d102mpr-{s}.tsv')), delimiter='\t'):
                if r['has_calib'] == '1' and r['iso_len_any75'] not in ('', 'nan') and float(r['iso_len_any75']) >= 10:
                    keys.append((s, int(r['event'])))
        label = 'Stage-1 ISO stratum'
    print(f'# doc pr/149 round 2 POST HOC local jitter vs bow, {label} ({len(keys)} events); cm; d = B - A')
    cache = {}
    for pair in a.pairs:
        b, base = pair.split(':')
        for arm in (b, base):
            if arm not in cache:
                cache[arm] = arm_terms(arm, samples, keys)
        A, B = cache[base], cache[b]
        both = [k for k in keys if k in A and k in B]
        line = f'{b} vs {base}: paired {len(both)}'
        for key in ('jit_x', 'jit_t', 'bow_x'):
            d = np.array([B[k][key] - A[k][key] for k in both])
            d = d[np.isfinite(d)]
            imp, wor = int((d < -0.005).sum()), int((d > 0.005).sum())
            base_med = float(np.nanmedian([A[k][key] for k in both])) if both else float('nan')
            line += (f' | {key} base {base_med:.4f} median d {float(np.median(d)) if len(d) else float("nan"):+.4f}'
                     f' imp {imp} wor {wor} p={sign_p(imp, imp + wor):.3g}')
        print(line)


if __name__ == '__main__':
    main()
