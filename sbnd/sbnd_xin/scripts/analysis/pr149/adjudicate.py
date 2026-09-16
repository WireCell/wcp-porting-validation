#!/usr/bin/env python3
"""doc pr/149: one mechanism line per Q2 mover (149_pred.txt sec 5).

For each event of a movers TSV (pr149_metrics.py compare --movers) prints, arm A -> arm B:
  main cluster (TaggerCheckNeutrino "selected main cluster N (t0, L)" log line),
  vertex route / DL best score / final vertex id (calib vertex_scoreboard),
  distance of the main vertex to the vtx105 click, segments, showers, nue/numu, Enu,
and a first-pass mechanism class:
  main-cluster   the selected main cluster (id or length) differs
  vertex-choice  same main cluster, main vertex moved > 1 cm
  pr-structure   same main cluster and vertex (<= 1 cm), segment or shower count differs
  energy-scale   none of the above; only scores / Enu moved
The class is a starting point for the doc's adjudication, not the adjudication itself.

Usage: adjudicate.py --a ARM_A --b ARM_B --movers 149_s1_movers_cs.tsv [--tsv out.tsv]
"""
import argparse
import csv
import json
import math
import os
import re
import sys

SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
sys.path.insert(0, SX)
from vtx_rules import vtx_io  # noqa: E402

RX = re.compile(r'selected main cluster (\d+) \(t0 ([-\d.]+) us, L ([\d.]+) cm')


def info(arm, s, e):
    d = os.path.join(SX, f'work-{s}-{arm}', f'pr_evt{e}')
    out = dict(main='-', L=float('nan'), route='-', dl=float('nan'), fin='-', v=None, nseg=0, nsh=0,
               nue=float('nan'), numu=float('nan'), enu=float('nan'))
    log = os.path.join(d, f'wct_pr_evt{e}.log')
    if os.path.exists(log):
        with open(log, errors='replace') as fh:
            for line in fh:
                m = RX.search(line)
                if m:
                    out['main'], out['L'] = int(m.group(1)), float(m.group(3))
                    break
    cp = os.path.join(d, f'calib-pr-evt{e}.json')
    if os.path.exists(cp):
        c = json.load(open(cp))
        vs = c.get('vertex_scoreboard') or {}
        out['route'] = vs.get('route', '-')
        out['dl'] = vs.get('dl_best_score', float('nan'))
        out['fin'] = vs.get('final_vertex_id', '-')
        out['v'] = vtx_io.xyz(c.get('main_vertex'))
        out['nseg'] = len(c.get('segments', []))
        out['nsh'] = len(c.get('showers', []))
        t = c.get('tagger') or {}
        out['nue'] = t.get('nue_score', float('nan'))
        out['numu'] = t.get('numu_score', float('nan'))
        out['enu'] = (c.get('kine') or {}).get('kine_reco_Enu', float('nan'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--movers', required=True)
    ap.add_argument('--tsv')
    a = ap.parse_args()
    lab = {d['eventNo']: d['truth'] for d in vtx_io.load_labels(tags=vtx_io.TAGS_VTX105) if d['truth']}
    rows = list(csv.DictReader(open(a.movers), delimiter='\t'))
    out = []
    counts = {}
    for r in rows:
        s, e = r['sample'], int(r['event'])
        A, B = info(a.a, s, e), info(a.b, s, e)
        moved = vtx_io.dist(A['v'], B['v']) if A['v'] and B['v'] else float('nan')
        if A['main'] != B['main'] or (math.isfinite(A['L']) and abs(A['L'] - B['L']) > 0.05):
            cls = 'main-cluster'
        elif math.isfinite(moved) and moved > 1.0:
            cls = 'vertex-choice'
        elif A['nseg'] != B['nseg'] or A['nsh'] != B['nsh']:
            cls = 'pr-structure'
        else:
            cls = 'energy-scale'
        counts[cls] = counts.get(cls, 0) + 1
        click = lab.get(e)
        dA = vtx_io.dist(click, A['v']) if click and A['v'] else float('nan')
        dB = vtx_io.dist(click, B['v']) if click and B['v'] else float('nan')
        line = (f"{s} {e} [{r['stratum']}] {cls}: main {A['main']}/{A['L']:.1f}cm -> {B['main']}/{B['L']:.1f}cm; "
                f"route {A['route']}({A['dl']:.1f}) -> {B['route']}({B['dl']:.1f}); moved {moved:.1f}cm; "
                f"click {dA:.1f}->{dB:.1f}cm; seg {A['nseg']}->{B['nseg']} sh {A['nsh']}->{B['nsh']}; "
                f"nue {A['nue']:.2f}->{B['nue']:.2f} numu {A['numu']:.2f}->{B['numu']:.2f}; "
                f"Enu {A['enu']:.0f}->{B['enu']:.0f} | {r['what']}")
        print(line)
        out.append((s, e, r['stratum'], cls, moved, dA, dB, r['what']))
    print('# classes: ' + ', '.join(f'{k} {v}' for k, v in sorted(counts.items())))
    if a.tsv:
        with open(a.tsv, 'w') as fo:
            fo.write('sample\tevent\tstratum\tclass\tvertex_moved_cm\tclick_A_cm\tclick_B_cm\twhat\n')
            for t in out:
                fo.write('\t'.join(str(x) if not isinstance(x, float) else f'{x:.2f}' for x in t) + '\n')


if __name__ == '__main__':
    main()
