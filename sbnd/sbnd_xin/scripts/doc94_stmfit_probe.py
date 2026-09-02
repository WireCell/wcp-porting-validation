#!/usr/bin/env python3
"""doc 94 -- read the save_stm_fit PCs out of an instrumented arm.

Prints, for each requested (event, main cluster): the fitted trajectory's two
endpoints and length, the dQ/dx-vs-residual-range profile near the stop, and
the stm_pass / stm_eval records.  Purpose: locate the STOP end so a prong's
attachment point (logged by check_other_tracks as `front=`) can be classified
as stop-side (Michel / delta ray) or start-side (interaction vertex).

Read-only.  Reuses the pctree access pattern of stm_campaign/extract_stm_diag.py.
"""
import io, json, os, sys, tarfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stm_campaign'))
MIP = 56000.0

def load_tar(fname):
    metas, arrays = {}, {}
    with tarfile.open(fname) as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if m.name.endswith('_metadata.json'):
                metas[m.name[:-len('_metadata.json')]] = json.load(f)
            elif m.name.endswith('_array.npy'):
                arrays[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(f.read()))
    return {md['datapath']: (md, arrays.get(base)) for base, md in metas.items() if 'datapath' in md}

def pc_blocks(byPath, evt, name):
    pre = f'pointtrees/{evt}/live/'
    lm = byPath.get(pre + f'lpcmaps/arrays/{name}')
    lm_cs = byPath.get(pre + 'lpcmaps/arrays/cluster_scalar')
    idents = byPath.get(pre + 'pointclouds/namedpcs/cluster_scalar/arrays/ident')
    if lm is None or lm_cs is None or idents is None:
        return {}
    counts, cs_counts, idents = lm[1], lm_cs[1], idents[1]
    arrays = {p.split('/')[-1]: arr for p, (md, arr) in byPath.items()
              if p.startswith(pre + f'pointclouds/namedpcs/{name}/arrays/')}
    if not arrays:
        return {}
    cs_nodes = np.nonzero(cs_counts)[0]
    node2ident = {int(n): int(idents[k]) for k, n in enumerate(cs_nodes)}
    out, off = {}, 0
    for node in np.nonzero(counts)[0]:
        n = int(counts[node]); ident = node2ident.get(int(node))
        blk = {k: v[off:off + n] for k, v in arrays.items()}
        if ident is not None:
            out.setdefault(ident, {k: [] for k in arrays})
            for k in arrays:
                out[ident][k].append(blk[k])
        off += n
    return {i: {k: np.concatenate(v) for k, v in b.items()} for i, b in out.items()}

root = sys.argv[1]
for spec in sys.argv[2:]:
    evt, main = (spec.split(':') + [None])[:2]
    tar = f'{root}/pr_evt{evt}/pctree-pr-evt{evt}.tar.gz'
    if not os.path.exists(tar):
        print(f'evt{evt}: no pctree'); continue
    bp = load_tar(tar)
    fits = pc_blocks(bp, evt, 'stm_fit')
    passes = pc_blocks(bp, evt, 'stm_pass')
    evals = pc_blocks(bp, evt, 'stm_eval')
    for ident, fit in sorted(fits.items()):
        if main is not None and int(ident) != int(main):
            continue
        print(f'\n===== evt{evt} cluster {ident} =====')
        for p in sorted(set(int(v) for v in fit['pass'])):
            sel = fit['pass'] == p
            x, y, z = fit['x'][sel]/10., fit['y'][sel]/10., fit['z'][sel]/10.
            L = fit['L'][sel]/10.
            dqdx = fit['dQ'][sel] / (fit['dx'][sel]/10. + 1e-9) / MIP
            pr = passes.get(ident, {})
            kink = None
            if pr:
                for i in range(len(pr['pass'])):
                    if int(pr['pass'][i]) == p:
                        kink = int(pr['kink_num'][i]); status = int(pr['status'][i])
            n = len(x)
            k = min(kink, n-1) if kink is not None and kink >= 0 else n-1
            print(f'  pass={p} status={status} npts={n} kink={kink} L={L[-1]:.1f}cm')
            print(f'    START (index 0)   = ({x[0]:7.1f},{y[0]:7.1f},{z[0]:7.1f}) cm')
            print(f'    STOP  (kink idx {k}) = ({x[k]:7.1f},{y[k]:7.1f},{z[k]:7.1f}) cm  L={L[k]:.1f}')
            if k < n-1:
                print(f'    PATH END (idx {n-1}) = ({x[-1]:7.1f},{y[-1]:7.1f},{z[-1]:7.1f}) cm  L={L[-1]:.1f}')
            rr = L[k] - L
            prof = [(float(rr[i]), float(dqdx[i])) for i in range(n) if -1 <= rr[i] <= 30]
            prof.sort()
            print('    dQ/dx vs residual range (MIP units, rr cm from stop):')
            line = '      ' + '  '.join(f'{r:.0f}:{q:.2f}' for r, q in prof[:26])
            print(line)
            body = dqdx[(rr > 10)]
            end5 = dqdx[(rr >= 0) & (rr < 5)]
            if len(body) and len(end5):
                print(f'    body median (rr>10cm) = {np.median(body):.2f} MIP   '
                      f'end-5cm median = {np.median(end5):.2f} MIP   end-5cm peak = {end5.max():.2f} MIP')
        ev = evals.get(ident, {})
        if ev:
            print('    stm_eval calls (pass strong verdict peak_range offset com_range ks1 ks2 ratio1 ratio2 res_len ave_res):')
            for i in range(len(ev['pass'])):
                print('      ' + ' '.join(f'{ev[k][i]:.4g}' for k in
                      ['pass','strong','verdict','peak_range','offset_length','com_range',
                       'ks1','ks2','ratio1','ratio2','res_length','ave_res_dqdx']))
