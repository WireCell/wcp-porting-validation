#!/usr/bin/env python3
"""Per-stage merge attribution from SBND_TRACE_BEE=1 per-APA trace layers
(doc pr/19, SBND run 18253 evt 444187).

Usage:
  unzip -q work-<tag>/ql_evt<ID>/mabc-apa0-face0.zip -d <dir>
  ./oc_stage_trace.py <dir> x,y,z=LABEL [x,y,z=LABEL ...]

Each positional after the dir is a reference point 'x,y,z=label' (cm, raw
img-global frame).  For every 0-tr*.json stage layer this prints the
cluster_id(s) within the smallest of 1/2/4/8 cm of each reference and flags
the first stage where two references share an id ("MERGED").
"""
import glob, json, os, sys
import numpy as np

def main():
    tdir = sys.argv[1]
    refs = {}
    for a in sys.argv[2:]:
        xyz, label = a.split('=')
        refs[label] = tuple(float(v) for v in xyz.split(','))
    files = sorted(glob.glob(os.path.join(tdir, 'data/0/0-tr*.json')))
    if not files:
        sys.exit(f"no 0-tr*.json under {tdir} (run with SBND_TRACE_BEE=1)")
    labels = list(refs)
    for f in files:
        d = json.load(open(f))
        P = np.c_[d['x'], d['y'], d['z']].astype(float)
        cid = np.asarray(d['cluster_id'], int)
        out, ids = [], {}
        for k, r in refs.items():
            dist = np.linalg.norm(P - np.array(r), axis=1)
            for R in (1.0, 2.0, 4.0, 8.0):
                m = dist < R
                if m.sum():
                    s = sorted(set(cid[m].tolist()))
                    ids[k] = set(s)
                    out.append(f"{k}={s if len(s) > 1 else s[0]}")
                    break
            else:
                out.append(f"{k}=?({dist.min():.0f}cm)")
                ids[k] = set()
        merged = [f"{labels[i]}+{labels[j]}"
                  for i in range(len(labels)) for j in range(i + 1, len(labels))
                  if ids[labels[i]] & ids[labels[j]]]
        tag = os.path.basename(f)[4:-5].replace('-global', '')
        print(f"{tag:34s} nclus={len(set(cid.tolist())):3d}  " + '  '.join(out)
              + ('   MERGED: ' + ','.join(merged) if merged else ''))

if __name__ == '__main__':
    main()
