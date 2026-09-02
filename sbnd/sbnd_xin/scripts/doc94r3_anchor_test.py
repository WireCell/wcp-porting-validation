#!/usr/bin/env python3
"""doc 94 sec 14 -- can entry_rise_guard reach 707-18-12, and does the anchor earn its place?

Owner, 2026-09-02 (after the first six verdicts): "this event 707-18-12 is
actually a neutrino, not STM.  I was wrong reading through the truth."  That
bundle was round 2's principal negative control and the stated justification
for anchoring the elevated run at L = 0, so both had to be re-measured.

Prints three things:
  1. what guard_entry_frac would have to be for 707-18-12 to have any run at
     all (answer: <= 1.00, i.e. a bar at or below MIP -- the feature is absent,
     not mis-thresholded);
  2. the anchored run under a relaxed anchor (start within A cm of the
     boundary) for A = 0..8 -- it moves nothing;
  3. the anchored run vs the longest elevated run ANYWHERE, which is what
     replaces the void justification: un-anchored, every labelled bundle lands
     in 11-15 cm and the feature separates nothing.

Read-only.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_here, 'doc94r3_kink_scan.py')).read()
exec(_src.split('# root, event, main cluster')[0])   # loaders + FRAC/WIN/BODY_* + feat()

def profile(root, evt, cid):
    """(L, dQ/dx in MIP, body, L_stop) over the muon segment [0, kink]."""
    bp = load_tar(f'{root}/pr_evt{evt}/pctree-pr-evt{evt}.tar.gz')
    fit = blocks(bp, evt, 'stm_fit').get(int(cid))
    pss = blocks(bp, evt, 'stm_pass').get(int(cid))
    p = sorted(set(int(v) for v in fit['pass']))[0]
    sel = fit['pass'] == p
    L = fit['L'][sel] / 10.
    dq = fit['dQ'][sel] / (fit['dx'][sel] / 10. + 1e-9) / MIP
    kink = -1
    for i in range(len(pss['pass'])):
        if int(pss['pass'][i]) == p:
            kink = int(pss['kink_num'][i])
    n = len(L); k = kink if 0 <= kink < n else n - 1
    Ls = L[k]
    m = (L[:k + 1] >= BODY_LO) & (L[:k + 1] <= Ls - BODY_HI_EXCL)
    body = float(np.median(dq[:k + 1][m])) if m.sum() else 1.0
    return L[:k + 1], dq[:k + 1], body, Ls

def run_from(L, dq, th, i0):
    for i in range(i0, len(L)):
        w = dq[i:][L[i:] <= L[i] + WIN]
        if len(w) < 3 or np.median(w) < th:
            return L[i] - L[i0]
    return L[-1] - L[i0]

SET = [
    ('work-stmfb8-r2scanoff', '12',     '7',  '707-18-12', 'NEUTRINO *re-adjudicated*'),
    ('work-stmfb8-r2scanoff', '4',      '18', '827-27-4',  'neutrino'),
    ('work-mcp2k-r2scanoff',  '164466', '7',  '164466:7',  'neutrino'),
    ('work-mcp1k-r2scanoff',  '350099', '15', '350099:15', 'neutrino'),
    ('work-mcp2k-r2scanoff',  '95500',  '15', '95500:15',  'STM'),
    ('work-mcp1k-r2scanoff',  '290316', '10', '290316:10', 'STM'),
    ('work-mcp1k-r2scanoff',  '282033', '13', '282033:13', 'STM'),
    ('work-mcp2k-r2scanoff',  '56257',  '13', '56257:13',  'STM'),
]

print('(1) 707-18-12 -- what would guard_entry_frac have to be?')
L, dq, body, Ls = profile('work-stmfb8-r2scanoff', '12', '7')
w0 = float(np.median(dq[L <= L[0] + WIN]))
print(f'    body = {body:.2f} MIP, entry(0-3cm) = {float(np.median(dq[L < 3])):.2f} MIP, '
      f'first {WIN:.0f} cm running median = {w0:.2f} MIP')
print(f'    threshold = frac x max(body, 1 MIP) = frac x 1.00, so a run needs frac <= {w0:.2f}')
print('    -> a bar at or below MIP, which fires on every track.  The feature is ABSENT.')

print('\n(2) relaxed anchor: longest run whose START is within A cm of the boundary')
print(f'{"event":<12}{"owner":<28}' + ''.join(f'A={a:<4.0f}' for a in (0, 1, 2, 3, 4, 5, 8)))
for root, evt, cid, lab, own in SET[:1] + SET[1:2] + SET[4:]:
    L, dq, body, Ls = profile(root, evt, cid)
    th = FRAC * max(body, 1.0)
    row = []
    for a in (0, 1, 2, 3, 4, 5, 8):
        best = 0.
        for i0 in range(len(L)):
            if L[i0] - L[0] > a:
                break
            best = max(best, run_from(L, dq, th, i0))
        row.append(best)
    print(f'{lab:<12}{own:<28}' + ''.join(f'{v:6.1f}' for v in row))

print('\n(3) anchored vs the longest elevated run ANYWHERE (drop the anchor)')
print(f'{"event":<12}{"owner":<28}{"anchored":>10}{"anywhere":>10}')
for root, evt, cid, lab, own in SET:
    L, dq, body, Ls = profile(root, evt, cid)
    th = FRAC * max(body, 1.0)
    anywhere = 0.
    for i0 in range(len(L)):
        anywhere = max(anywhere, run_from(L, dq, th, i0))
    print(f'{lab:<12}{own:<28}{run_from(L, dq, th, 0):10.1f}{anywhere:10.1f}')
