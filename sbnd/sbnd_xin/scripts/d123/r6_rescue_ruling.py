#!/usr/bin/env python3
"""doc sbnd_xin/123 sec 18 -- the MC ruling on the cathode-bundle rescue's firings.

For every event where the rescue fired in a hit-flash MC arm (the 'rescue round' / 'unmatched
rescue round' lines of f*/g0/wct_ql.log, mapped to the event id through the all-APA 'loading
tensor set ident=' line that precedes them), print the truth (products/d115/<s>/truth_base.tsv)
and the candidate of every product dir given: score, reco Enu, vertex, distance to the matched
true vertex.  The rescue is judged per event by what its removal (the nr arm) does to the
truth-matched candidate.

usage: r6_rescue_ruling.py <hits_arm_root> <truth_base.tsv> <label=products_dir> [...]
"""
import csv, glob, json, os, re, sys


def firings(root):
    out = []
    for log in sorted(glob.glob(os.path.join(root, 'f*', 'g0', 'wct_ql.log'))):
        sub = os.path.basename(os.path.dirname(os.path.dirname(log)))
        rse = json.load(open(os.path.join(os.path.dirname(log), 'rse.json')))
        cur = None
        for l in open(log, errors='replace'):
            m = re.search(r'clus_all_apa> loading tensor set ident=(\d+)', l)
            if m:
                cur = m.group(1); continue
            if ('rescue round' in l) and 'CathodeBundleRescue' in l:
                r, s = rse[cur]
                out.append(((int(r), int(s), int(cur)), sub, l.split('<CathodeBundleRescue:all> ')[1].strip()))
    return out


def truth(path):
    t = {}
    for r in csv.DictReader(open(path), delimiter='\t'):
        t.setdefault((int(r['run']), int(r['subrun']), int(r['event'])), []).append(r)
    return t


def cands(path):
    c = {}
    for r in csv.DictReader(open(os.path.join(path, 'candidates.tsv')), delimiter='\t'):
        c.setdefault((int(r['run']), int(r['subrun']), int(r['event'])), []).append(r)
    return c


def main():
    root, tpath = sys.argv[1], sys.argv[2]
    prods = [a.split('=', 1) for a in sys.argv[3:]]
    T = truth(tpath)
    C = {lab: cands(p) for lab, p in prods}
    for key, sub, line in firings(root):
        print('== %s/%d/%d/%d  %s' % (sub, *key, line))
        for r in T.get(key, []):
            print('   truth: %s %s %s E %s MeV T %s us v (%s, %s, %s)' % (r['flav'], r['mode'], r['ccnc'], r['Etot'], r['T'], r['vx'], r['vy'], r['vz']))
        for lab, _ in prods:
            rows = C[lab].get(key, [])
            if not rows:
                print('   %-5s: no candidate' % lab); continue
            for r in rows:
                print('   %-5s: cl %s numu %s Enu %s vtx (%s, %s, %s) len %s cm flash %s us %s PE tpc %s | truth %s %s dist %s cm' % (
                    lab, r['cluster_id'], r['numu_score'], r['reco_Enu'], r['nu_x'], r['nu_y'], r['nu_z'], r['sel_length_cm'],
                    r['flash_time_us'], r['flash_pe'], r['flash_tpc'], r['t_flav'], r['t_ccnc'], r['t_dist_cm']))


if __name__ == '__main__':
    main()
