#!/usr/bin/env python3
"""Census: per event, sizable NON-MATCHED clusters (gid<0) near the cathode
while an in-beam bundle exists — the candidate population for extending the
pr/14 cathode bundle rescue to unmatched clusters.

usage: unmatched_census.py <arm> <out.tsv> [jobs]
"""
import sys, os, re, glob, io, json, tarfile
import numpy as np
from multiprocessing import Pool

MM, US = 10.0, 1000.0
BEAM = (0.2, 2.2)      # us
MIN_LEN_CM = 30.0
MIN_NPTS = 200
CATH_X_CM = 5.0        # pr/14 cathode_x_cut

def one(path):
    evt = re.search(r'evt(\d+)', os.path.basename(path)).group(1)
    try:
        metas, arrays = {}, {}
        with tarfile.open(path) as tf:
            for m in tf.getmembers():
                f = tf.extractfile(m)
                if m.name.endswith('_metadata.json'):
                    metas[m.name[:-len('_metadata.json')]] = json.load(f)
                elif m.name.endswith('_array.npy'):
                    arrays[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(f.read()))
        bp = {md['datapath']: (md, arrays.get(b)) for b, md in metas.items() if 'datapath' in md}
        live = [p for p in bp if re.fullmatch(r'pointtrees/\d+/live', p)][0]
        md = bp[live][0]
        items = bp[md['pointclouds']][0]['items']
        lpc = bp[md['lpcmaps']][0]['arrays']
        ds = lambda pc: bp[items[pc]][0]['arrays']
        arr = lambda pc, a: bp[ds(pc)[a]][1]
        ident = arr('cluster_scalar', 'ident').astype(int)
        t0 = arr('cluster_scalar', 'cluster_t0') / US
        gid = arr('cluster_scalar', 'matched_flash_gid').astype(int)
        x, y, z = arr('3d', 'x'), arr('3d', 'y'), arr('3d', 'z')
        map_cs = bp[lpc['cluster_scalar']][1].astype(int)
        map_3d = bp[lpc['3d']][1].astype(int)
        starts, ci, pos = {}, -1, 0
        for n in range(len(map_cs)):
            if map_cs[n]:
                ci += 1
                starts[ci] = [pos, 0]
            if map_3d[n]:
                if ci >= 0:
                    starts[ci][1] += int(map_3d[n])
                pos += int(map_3d[n])
        has_beam = bool(np.any((gid >= 0) & (t0 >= BEAM[0]) & (t0 < BEAM[1])))
        rows = []
        for i in range(len(ident)):
            if gid[i] >= 0:
                continue
            s, n = starts.get(i, (0, 0))
            if n < MIN_NPTS:
                continue
            P = np.c_[x[s:s+n], y[s:s+n], z[s:s+n]] / MM
            a = int(np.argmax(((P - P[0]) ** 2).sum(1)))
            b = int(np.argmax(((P - P[a]) ** 2).sum(1)))
            ln = float(np.linalg.norm(P[a] - P[b]))
            if ln < MIN_LEN_CM:
                continue
            j = int(np.argmin(np.abs(P[:, 0])))
            rows.append((evt, int(ident[i]), n, round(ln, 1),
                         round(float(abs(P[j, 0])), 2),
                         round(float(P[j, 0]), 2), round(float(P[j, 1]), 1),
                         round(float(P[j, 2]), 1), int(has_beam)))
        return rows
    except Exception as e:
        return [(evt, -1, 0, 0.0, -1.0, 0.0, 0.0, 0.0, -1), ('ERR', str(e))][:1]

def main():
    arm, out = sys.argv[1], sys.argv[2]
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    paths = sorted(glob.glob(os.path.join(arm, 'ql_evt*', 'pctree-evt*.tar.gz')))
    print(f'{len(paths)} pctrees, {jobs} workers', flush=True)
    with Pool(jobs) as p:
        res = p.map(one, paths, chunksize=8)
    with open(out, 'w') as f:
        f.write('event\tident\tnpts\tlen_cm\tcath_dist_cm\ttip_x\ttip_y\ttip_z\thas_beam_bundle\n')
        for rows in res:
            for r in rows:
                f.write('\t'.join(str(v) for v in r) + '\n')
    print('wrote', out, flush=True)

if __name__ == '__main__':
    main()
