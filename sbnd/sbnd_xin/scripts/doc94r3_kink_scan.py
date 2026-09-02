#!/usr/bin/env python3
"""doc 94 round 3 -- the owner's KINK, measured offline on the labelled set.

Owner, 2026-09-02, after hand-scanning the round-2 Bee pair:

  "95500 looks like a STM, not a neutrino, the near entry point looks like
   fluctuation of dQ/dx with delta ray.  Note, another key to separate the
   other events from this event is along the track, there is a kink (large
   angle change).  This event does not have a large angle change, which makes
   it more STM like."

This script reads the save_stm_fit trajectories out of the round-2 scan arms,
re-implements entry_rise_reject()'s body / threshold / anchored-run exactly,
and then measures the largest direction change between two chords meeting at a
point on the muon segment.  It answers two questions:

  1. which chord half-length separates the owner's neutrinos from his STMs by
     the widest margin (-> the fixed 5 cm window in the C++);
  2. what every one of the 12 labelled bundles scores (-> the 22 deg bar).

Read-only.  The C++ scans fit points rather than a 0.5 cm grid, so its numbers
differ from these in the last tenth of a degree; the arm logs are the record.
"""
import io, json, sys, tarfile
import numpy as np

MIP = 56000.0
FRAC, WIN, BODY_LO, BODY_HI_EXCL = 1.3, 5.0, 20.0, 25.0

def load_tar(f):
    metas, arr = {}, {}
    with tarfile.open(f) as tf:
        for m in tf.getmembers():
            x = tf.extractfile(m)
            if m.name.endswith('_metadata.json'):
                metas[m.name[:-len('_metadata.json')]] = json.load(x)
            elif m.name.endswith('_array.npy'):
                arr[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(x.read()))
    return {md['datapath']: (md, arr.get(b)) for b, md in metas.items() if 'datapath' in md}

def blocks(bp, evt, name):
    pre = f'pointtrees/{evt}/live/'
    lm = bp.get(pre + f'lpcmaps/arrays/{name}')
    lmc = bp.get(pre + 'lpcmaps/arrays/cluster_scalar')
    idn = bp.get(pre + 'pointclouds/namedpcs/cluster_scalar/arrays/ident')
    if lm is None or lmc is None or idn is None:
        return {}
    counts, csc, idn = lm[1], lmc[1], idn[1]
    arrs = {p.split('/')[-1]: a for p, (md, a) in bp.items()
            if p.startswith(pre + f'pointclouds/namedpcs/{name}/arrays/')}
    if not arrs:
        return {}
    n2i = {int(n): int(idn[k]) for k, n in enumerate(np.nonzero(csc)[0])}
    out, off = {}, 0
    for node in np.nonzero(counts)[0]:
        n = int(counts[node]); i = n2i.get(int(node))
        blk = {k: v[off:off + n] for k, v in arrs.items()}
        if i is not None:
            out.setdefault(i, {k: [] for k in arrs})
            for k in arrs:
                out[i][k].append(blk[k])
        off += n
    return {i: {k: np.concatenate(v) for k, v in b.items()} for i, b in out.items()}

def feat(root, evt, cid, W, endx):
    """(shoulder, largest kink, L_stop) -- mirrors entry_rise_reject()."""
    bp = load_tar(f'{root}/pr_evt{evt}/pctree-pr-evt{evt}.tar.gz')
    fit = blocks(bp, evt, 'stm_fit').get(int(cid))
    pss = blocks(bp, evt, 'stm_pass').get(int(cid))
    p = sorted(set(int(v) for v in fit['pass']))[0]
    sel = fit['pass'] == p
    L = fit['L'][sel] / 10.
    dq = fit['dQ'][sel] / (fit['dx'][sel] / 10. + 1e-9) / MIP
    P = np.stack([fit['x'][sel] / 10., fit['y'][sel] / 10., fit['z'][sel] / 10.], 1)
    kink = -1
    for i in range(len(pss['pass'])):
        if int(pss['pass'][i]) == p:
            kink = int(pss['kink_num'][i])
    n = len(L); k = kink if 0 <= kink < n else n - 1
    Ls = L[k]
    m = (L[:k + 1] >= BODY_LO) & (L[:k + 1] <= Ls - BODY_HI_EXCL)
    body = float(np.median(dq[:k + 1][m])) if m.sum() else 1.0
    th = FRAC * max(body, 1.0)
    sh = Ls - L[0]
    for i in range(0, k + 1):
        w = dq[i:k + 1][L[i:k + 1] <= L[i] + WIN]
        if len(w) < 3 or np.median(w) < th:
            sh = L[i] - L[0]
            break
    def chord(lo, hi):
        idx = np.nonzero((L[:k + 1] >= lo) & (L[:k + 1] <= hi))[0]
        if len(idx) < 2:
            return None
        d = P[idx[-1]] - P[idx[0]]
        nn = np.linalg.norm(d)
        return d / nn if nn > 0 else None
    best, t = 0.0, W
    while t <= Ls - endx - W:
        d1, d2 = chord(t - W, t), chord(t, t + W)
        if d1 is not None and d2 is not None:
            best = max(best, float(np.degrees(np.arccos(np.clip(d1 @ d2, -1, 1)))))
        t += 0.5
    return sh, best, Ls

# root, event, main cluster, label, owner verdict
SET = [
    ('work-stmfb8-r2scanoff', '4',      '18', '827-27-4',  'NEUTRINO (target)'),
    ('work-mcp2k-r2scanoff',  '164466', '7',  '164466:7',  'NEUTRINO (owner)'),
    ('work-mcp1k-r2scanoff',  '350099', '15', '350099:15', 'NEUTRINO (owner)'),
    ('work-stmfb8-stmfit',    '28',     '5',  '304-6-28',  'neutrino (round 1)'),
    ('work-stmfb8-stmfit',    '31',     '12', '146-60-31', 'neutrino (round 1)'),
    ('work-stmfb8-stmfit',    '22',     '6',  '966-2-22',  'neutrino (round 1)'),
    ('work-mcp2k-r2scanoff',  '95500',  '15', '95500:15',  'STM (owner)'),
    ('work-mcp1k-r2scanoff',  '290316', '10', '290316:10', 'STM (owner)'),
    ('work-mcp1k-r2scanoff',  '282033', '13', '282033:13', 'STM (owner)'),
    ('work-mcp2k-r2scanoff',  '56257',  '13', '56257:13',  'STM (owner)'),
    ('work-stmfb8-r2scanoff', '12',     '7',  '707-18-12', 'STM (owner)'),
    ('work-stmfb8-stmfit',    '17',     '20', '36-77-17',  'control, not STM'),
]
MIN_CM, MAX_CM, BAR, MIN_LEN = 5.0, 60.0, 22.0, 70.0

print('=== window scan: margin between the FIRING neutrinos and the FIRING STM ===')
print(f'{"chord half-length":>18}{"endx":>6}{"N min":>8}{"STM max":>9}{"margin":>8}')
for endx in (0., 15.):
    for W in (5., 8., 10., 12., 15.):
        rows = [(lab, own, *feat(r, e, c, W, endx)[:2]) for r, e, c, lab, own in SET]
        fir = [x for x in rows if x[2] >= MIN_CM]
        nn = [x[3] for x in fir if 'NEUTR' in x[1].upper() or 'neutrino' in x[1]]
        ss = [x[3] for x in fir if x[1].startswith('STM')]
        if nn and ss:
            print(f'{W:18.0f}{endx:6.0f}{min(nn):8.1f}{max(ss):9.1f}{min(nn)-max(ss):8.1f}')

W, ENDX = 5.0, 15.0
print(f'\n=== the labelled set at chord {W:.0f} cm, stop exclusion {ENDX:.0f} cm ===')
print(f'{"event":<12}{"owner":<20}{"L_stop":>8}{"shoulder":>10}{"kink":>8}   round-3 verdict')
for r, e, c, lab, own in SET:
    sh, best, Ls = feat(r, e, c, W, ENDX)
    # guard_entry_min_len_cm: a muon shorter than this has no body window, so
    # the guard declines outright -- 146-60-31 (56.9 cm) is decided here, and
    # was recovered in round 1 by vertex_hadron_guard instead.
    if Ls < MIN_LEN:
        print(f'{lab:<12}{own:<20}{Ls:8.1f}{sh:10.1f}{best:8.1f}   '
              f'{"declined":<9}  <-- muon < {MIN_LEN:.0f} cm, no body window')
        continue
    fires = MIN_CM <= sh <= MAX_CM and best >= BAR
    r2 = MIN_CM <= sh <= 30.0
    note = '' if fires == r2 else ('  <-- FIXED by the kink' if r2 else '  <-- RECOVERED by max_cm 60')
    print(f'{lab:<12}{own:<20}{Ls:8.1f}{sh:10.1f}{best:8.1f}   '
          f'{"RELEASE" if fires else "keep STM":<9}{note}')
