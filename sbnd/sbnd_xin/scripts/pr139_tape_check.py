#!/usr/bin/env python3
"""doc pr/139 P1.2 -- is the C++ impact parameter the one the offline table priced?

The bound b<=12 was chosen on OFFLINE numbers (pr139_pointing.py, SVD of the
sqrt(w)-scaled centred points).  The C++ computes the same quantity from a
weighted covariance eigenvector on the shower's member fits.  Those agree in
algebra; whether they agree in PRACTICE depends on which points and which vertex
each one sees -- and doc pr/138 B1 already found the splitter's reference vertex
differs from the scan's on 5 of 172 objects, by up to 60 cm, because the pi0
finders re-seat main_vertex AFTER this pass.

So this checks the two numbers against each other instead of assuming.  Reads
the SHOWER_SPLIT tape from an arm run with WCT_SHOWER_SPLIT_DEBUG=1.

  ./scripts/pr139_tape_check.py work-pr139r1-onb12
"""
import sys, os, glob, re, csv
sys.path[:0] = ['scripts', 'split_display']
import numpy as np
import pr137_lib as L
import split_model as SM

arm = sys.argv[1] if len(sys.argv) > 1 else 'work-pr139r1-onb12'
CAND = re.compile(r'SHOWER_SPLIT cand shower=(\d+) .*?fired=(\d) .*?b_cm=(-?[\d.]+) veto=(\d)')
tape = []
for lg in sorted(glob.glob(f'{arm}-*/pr_evt*/stdout.log')):
    ev = int(re.search(r'pr_evt(\d+)/', lg).group(1))
    for line in open(lg, errors='replace'):
        m = CAND.search(line)
        if m:
            tape.append(dict(event=ev, node=int(m.group(1)), fired=int(m.group(2)),
                             b_cxx=float(m.group(3)), veto=int(m.group(4))))
# The 2026-08-31 12:0x build printed b in INTERNAL units (mm), not cm -- the
# veto comparison itself was always in matching units (the config value is
# multiplied by units::cm at the wiring site), so only the TAPE was wrong and no
# arm's decisions were affected.  Detect and correct it rather than silently
# accepting a factor 10: the fixed build prints cm and this rescale is a no-op.
if tape and np.median([t['b_cxx'] for t in tape]) > 200:
    print("  NOTE: tape is in internal units (mm) -- rescaling to cm (print-only bug,")
    print("        veto decisions unaffected; see doc pr/139 P1.2)")
    for t in tape:
        if t['b_cxx'] > 0:
            t['b_cxx'] /= 10.0
print("tape rows %d  from %s-*" % (len(tape), arm))
if not tape:
    sys.exit("no SHOWER_SPLIT cand lines -- was the arm run with WCT_SHOWER_SPLIT_DEBUG=1?")
print("  fired %d   vetoed %d   (a veto only ever fires on a fired candidate)"
      % (sum(t['fired'] for t in tape), sum(t['veto'] for t in tape)))

rows = []
for t in tape:
    row = SM.load_object(t['event'], t['node'])
    if row is None:
        continue
    pts, q, _ = L.pack(row['P'], row['segs'])
    if pts is None or len(pts) < 8:
        continue
    v = np.asarray(row['v'], float)
    w = L.qwt(q)
    c = (pts * w[:, None]).sum(0) / w.sum()
    X = (pts - c) * np.sqrt(w)[:, None]
    ax = np.linalg.svd(X, full_matrices=False)[2][0]
    ax = ax / np.linalg.norm(ax)
    d = c - v
    rows.append(dict(t, b_off=float(np.linalg.norm(d - np.dot(d, ax) * ax))))

if rows:
    dd = np.array([r['b_cxx'] - r['b_off'] for r in rows])
    print("\njoined %d of %d tape rows to an offline object" % (len(rows), len(tape)))
    print("  b_cxx - b_off : median %.3f cm  |max| %.3f cm  within 0.5 cm on %d of %d"
          % (np.median(dd), np.max(np.abs(dd)), int((np.abs(dd) <= 0.5).sum()), len(dd)))
    print("\n  the ten largest disagreements (doc pr/138 B1: the splitter's vertex is")
    print("  NOT the dump's main_vertex on a pi0 event, so a large gap is EXPECTED there)")
    print("  %-9s %-9s %8s %8s %8s" % ('event', 'node', 'b_cxx', 'b_off', 'diff'))
    for r in sorted(rows, key=lambda r: -abs(r['b_cxx'] - r['b_off']))[:10]:
        print("  %-9d %-9d %8.2f %8.2f %8.2f"
              % (r['event'], r['node'], r['b_cxx'], r['b_off'], r['b_cxx'] - r['b_off']))
with open('docs/pr/pr139-tape-b.tsv', 'w') as f:
    w_ = csv.writer(f, delimiter='\t')
    w_.writerow(['event', 'node', 'fired', 'veto', 'b_cxx_cm', 'b_offline_cm'])
    for r in rows:
        w_.writerow([r['event'], r['node'], r['fired'], r['veto'],
                     '%.3f' % r['b_cxx'], '%.3f' % r['b_off']])
print("\nwrote docs/pr/pr139-tape-b.tsv")
