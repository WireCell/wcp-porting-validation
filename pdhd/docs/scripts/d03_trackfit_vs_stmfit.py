#!/usr/bin/env python3
"""doc pdhd/03 (fork BY DUPLICATION of pdvd d45_trackfit_vs_stmfit.py; doc pdvd/45) -- grade the PR chain's multi-segment fit (Bee layer track_fit) against
the STM single-track fit (stm_fit) on the SAME clusters, and attribute the loss to the
fitter's zero-quantity point drop.

Per work dir (work/<run6>_<evt>_<tag>/), reads mabc-pr.zip layers
  data/0/0-track_fit-global.json, 0-stm_fit-global.json, 0-clustering-global.json
and the PR log wct_pr_<run6>_<evt>.log.  Per cluster that has an stm_fit trajectory:
  stm      stm_fit points            tf       track_fit points (incl. -1 vertex markers)
  nseg     distinct real_cluster_id (cluster*1000+graph_index) among track_fit points
  win      segments with >2 track_fit points (the "winner" census; a segment reduced to
           its two endpoints counts 0)
  cov      fraction of stm_fit points with a track_fit point within COV_CM (1 cm)
  d50      median distance stm_fit -> nearest track_fit point (cm)
  ghost    fraction of track_fit points farther than GHOST_CM (2 cm) from any raw
           clustering point (pr94r3_gap_metric.py's question)
  gap50    median nearest-neighbour spacing among the cluster's track_fit points
  drop     worst "pre-dQ/dx form_map_graph dropped N of M" pass attributed to this
           cluster in the log (N/M), attribution = the last NeutrinoPattern
           'cluster N' line before the drop line
Per event: total dropped / total over all passes.
Usage: d45_trackfit_vs_stmfit.py [--tsv out.tsv] [--all-clusters] <workdir> [<workdir> ...]
"""
import sys, os, re, json, zipfile, glob, argparse
import numpy as np
from scipy.spatial import cKDTree
from collections import Counter, defaultdict

COV_CM, GHOST_CM = 1.0, 2.0

def layer(z, name):
    try:
        d = json.loads(z.read(f'data/0/0-{name}-global.json'))
    except KeyError:
        return None
    return {k: np.asarray(d[k]) for k in ('x', 'y', 'z', 'q', 'cluster_id', 'real_cluster_id') if k in d}

def drops_by_cluster(logpath):
    """(cluster -> list of (dropped, total)), event totals."""
    per, cur, tot_d, tot_n = defaultdict(list), None, 0, 0
    if not os.path.exists(logpath):
        return per, tot_d, tot_n
    rx_cl = re.compile(r'\bcluster[= ](\d+)\b')
    rx_dr = re.compile(r'dropped (\d+) of (\d+) trajectory')
    for line in open(logpath, errors='replace'):
        if 'NeutrinoPattern' in line:
            m = rx_cl.search(line)
            if m: cur = int(m.group(1))
        m = rx_dr.search(line)
        if m:
            d, n = int(m.group(1)), int(m.group(2))
            per[cur].append((d, n)); tot_d += d; tot_n += n
    return per, tot_d, tot_n

def grade(workdir, all_clusters=False):
    zp = os.path.join(workdir, 'mabc-pr.zip')
    z = zipfile.ZipFile(zp)
    tf, sf, cl = layer(z, 'track_fit'), layer(z, 'stm_fit'), layer(z, 'clustering')
    logs = glob.glob(os.path.join(workdir, 'wct_pr_*.log'))
    per, tot_d, tot_n = drops_by_cluster(logs[0]) if logs else ({}, 0, 0)
    rows = []
    if tf is None or cl is None:
        return rows, (tot_d, tot_n)
    C = np.c_[cl['x'], cl['y'], cl['z']]; ctree = cKDTree(C)
    clusters = sorted(set(sf['cluster_id'].tolist())) if (sf is not None and not all_clusters) else sorted(set(tf['cluster_id'].tolist()))
    for c in clusters:
        n = tf['cluster_id'] == c
        Q = np.c_[tf['x'][n], tf['y'][n], tf['z'][n]]
        segs = Counter(tf['real_cluster_id'][n].tolist())
        win = sum(1 for k, v in segs.items() if k != -1 and v > 2)
        row = dict(cluster=int(c), raw=int((cl['cluster_id'] == c).sum()), stm=0, tf=int(n.sum()),
                   nseg=len([k for k in segs if k != -1]), win=win, cov=np.nan, d50=np.nan,
                   ghost=np.nan, gap50=np.nan, drop='')
        if len(Q):
            dg, _ = ctree.query(Q); row['ghost'] = float((dg > GHOST_CM).mean())
            if len(Q) > 1:
                D = cKDTree(Q).query(Q, k=2)[0][:, 1]; row['gap50'] = float(np.median(D))
        if sf is not None:
            m = sf['cluster_id'] == c
            row['stm'] = int(m.sum())
            if m.sum() and len(Q):
                P = np.c_[sf['x'][m], sf['y'][m], sf['z'][m]]
                dc, _ = cKDTree(Q).query(P)
                row['cov'] = float((dc < COV_CM).mean()); row['d50'] = float(np.median(dc))
            elif m.sum():
                row['cov'] = 0.0
        if c in per and per[c]:
            d, nn = max(per[c], key=lambda t: (t[0] / max(t[1], 1)))
            row['drop'] = f'{d}/{nn}'
        rows.append(row)
    return rows, (tot_d, tot_n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsv'); ap.add_argument('--all-clusters', action='store_true')
    ap.add_argument('workdirs', nargs='+')
    a = ap.parse_args()
    out = open(a.tsv, 'w') if a.tsv else None
    hdr = 'workdir cluster raw stm tf nseg win cov d50 ghost gap50 drop'.split()
    if out: out.write('\t'.join(hdr) + '\n')
    for w in a.workdirs:
        w = w.rstrip('/')
        if not os.path.exists(os.path.join(w, 'mabc-pr.zip')):
            print(f'{w}: no mabc-pr.zip'); continue
        rows, (td, tn) = grade(w, a.all_clusters)
        print(f'== {os.path.basename(w)}  drop total {td}/{tn} = {td / tn if tn else float("nan"):.3f}')
        print(' cluster   raw  stm   tf nseg win   cov   d50 ghost gap50   drop')
        for r in rows:
            print(f" {r['cluster']:7d} {r['raw']:5d} {r['stm']:4d} {r['tf']:4d} {r['nseg']:4d} {r['win']:3d} "
                  f"{r['cov']:5.2f} {r['d50']:5.2f} {r['ghost']:5.2f} {r['gap50']:5.2f}   {r['drop']}")
            if out:
                out.write('\t'.join(str(x) for x in [os.path.basename(w)] + [r[k] for k in hdr[1:]]) + '\n')
    if out: out.close()

if __name__ == '__main__':
    main()
