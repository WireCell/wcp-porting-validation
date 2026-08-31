#!/usr/bin/env python3
# doc pr/138 -- the owner's steer: how does the splitter do when the event is CLEAN?
"""Stratify the trigger and the kernel by event/vertex condition, not by outcome.

THE OWNER'S INSTRUCTION (2026-08-31), and it reframes the whole residual:

    "There are cases where it is difficult to get it right.  incorrect neutrino
     vertex, very busy events etc.  What we want is when the situation is
     reasonably clean, we get decent results against the hand scan results."

So the question is no longer "what is the splitter's purity" but "what is it
where the reconstruction is not already broken upstream" -- and the hard classes
become a SCOPE boundary to be named, not a residual to be tuned against.

THE DEFINITION IS A PRIORI.  Every axis below is measured from the arm, never
from the label or the outcome, and the thresholds are physical rather than
scanned:

  vertex on charge   vgap_cm <= 5    the nu vertex should sit ON the charge it
                                     seeds; the trigger measures EVERY feature
                                     as a ray from it, so a vertex sitting in
                                     empty space manufactures a fake bimodality
  vertex not moved   |dv| < 1 cm     the pi0 chain re-seats main_vertex AFTER
                                     the splitter (sec B1: 60.16 cm on evt76346,
                                     14.50 cm on evt396222).  When it does, the
                                     splitter and the scan measured from
                                     different points
  event not busy     n_cand <= 3     candidates above the pass's own floors in
                                     that event -- the owner's "very busy
                                     events", counted rather than eyeballed

VALIDATION THAT DOES NOT FIT.  The owner independently flagged specific objects
in his scan comments -- "incorrect neutrino vertex", "a very busy event ... I am
not sure if this event is really useful for our purpose".  Those comments were
never consulted to build the definition, so how many of them land in HARD is a
free test of it.

Repro:
    python3 scripts/pr138_clean_strata.py --tape 'work-pr138r1-dbg-*' --png
"""
import os, sys, re, json, glob, csv, argparse, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--tape', default='work-pr138r1-dbg-*')
ap.add_argument('--compare', default='docs/pr/pr138-probe-compare.tsv')
ap.add_argument('--owner-tag', default='splitscan-0901-owner')
ap.add_argument('--tsv', default='docs/pr/pr138-clean-strata.tsv')
ap.add_argument('--png', action='store_true', help='write docs/pr/pr138-clean-strata.png')
args = ap.parse_args()

SPLITS = ('SPLIT2', 'SPLIT3', 'SPLIT4+')
VGAP_MAX, DV_MAX, NCAND_MAX = 5.0, 1.0, 3
TRACKS = {(99838, 14004), (389538, 19021), (292524, 9018), (176502, 109141),
          (286681, 72040), (122660, 54071), (415278, 23047), (278420, 18002)}

CAND = re.compile(r'SHOWER_SPLIT cand shower=(-?\d+) pdg=(-?\d+) .* fired=(\d)')

ncand = collections.Counter()
for log in sorted(glob.glob(os.path.join(args.tape, 'pr_evt*', 'stdout.log'))):
    m = re.search(r'pr_evt(\d+)', log)
    if not m:
        continue
    ev = int(m.group(1))
    for ln in open(log, errors='replace'):
        if CAND.search(ln):
            ncand[ev] += 1

OWN = {}
for f in sorted(glob.glob('em_labels/%s/labels-evt*.json' % args.owner_tag)):
    j = json.load(open(f))
    try:
        ev = int(str(j.get('event', '')).replace('evt', ''))
    except Exception:
        continue
    for nd, r in (j.get('split_labels') or {}).items():
        OWN[(ev, int(nd))] = r

rows = []
for r in csv.DictReader((l for l in open(args.compare) if not l.startswith('#')),
                        delimiter='\t'):
    k = (int(r['event']), int(r['node']))
    if k in TRACKS:
        continue
    lab = OWN.get(k, {})
    vgap, dv = float(r['vgap_cm']), float(r['dv_cm'])
    nc = ncand[k[0]]
    ok_v, ok_m, ok_b = vgap <= VGAP_MAX, dv < DV_MAX, nc <= NCAND_MAX
    rows.append(dict(event=k[0], node=k[1], vgap=vgap, dv=dv, ncand=nc,
                     valley=float(r['cxx_valley']), n_seed=int(r['cxx_n_seed']),
                     fired=int(r['cxx_fired']), verdict=r['owner'],
                     pos=r['owner'] in SPLITS,
                     conf=lab.get('confidence', ''), comment=(lab.get('comment') or ''),
                     ok_v=ok_v, ok_m=ok_m, ok_b=ok_b, clean=(ok_v and ok_m and ok_b)))

print("doc pr/138 -- the splitter stratified by how CLEAN the situation is")
print("a-priori definition: vgap<=%.0f cm AND |dv|<%.0f cm AND n_cand<=%d, no label consulted"
      % (VGAP_MAX, DV_MAX, NCAND_MAX))
print("scored objects (164 EM, track-typed excluded): %d" % len(rows))


def score(rs, name):
    f = [r for r in rs if r['fired']]
    p = [r for r in rs if r['pos']]
    tp = [r for r in f if r['pos']]
    e = len(tp) / max(len(p), 1)
    u = len(tp) / max(len(f), 1)
    print("  %-34s n=%3d  positives %2d  fires %2d  eff %.3f  pur %.3f  F1 %.3f"
          % (name, len(rs), len(p), len(f), e, u, 2 * e * u / max(e + u, 1e-9)))
    return len(rs), len(p), len(f), e, u


print("\n=== 1. ONE AXIS AT A TIME (is each part of the definition doing work?) ===")
score(rows, "everything")
for lab, key in (("vertex ON charge  (vgap<=5)", 'ok_v'),
                 ("vertex NOT re-seated later", 'ok_m'),
                 ("event NOT busy    (ncand<=3)", 'ok_b')):
    score([r for r in rows if r[key]], lab + "  PASS")
    score([r for r in rows if not r[key]], lab + "  FAIL")

print("\n=== 2. THE STRATA ===")
cl = score([r for r in rows if r['clean']], "CLEAN  (all three)")
hd = score([r for r in rows if not r['clean']], "HARD   (any one fails)")
print("  NOTE the efficiencies above are WITHIN-STRATUM (positives of that")
print("  stratum), which is the owner's question -- 'when the situation is clean,")
print("  do we get decent results'.  They are NOT what a veto would deliver on")
print("  the whole population; section 3b prices that separately.")

print("\n=== 3. THE FREE TEST: do the owner's own flags land in HARD? ===")
flagged = [r for r in rows
           if re.search(r'incorrect.*vertex|busy|not sure', r['comment'], re.I)
           or r['conf'] == 'low']
print("  objects the owner flagged (wrong vertex / busy / low confidence): %d" % len(flagged))
inh = sum(1 for r in flagged if not r['clean'])
print("  of those, landing in HARD by the a-priori definition: %d (%.0f%%)"
      % (inh, 100 * inh / max(len(flagged), 1)))
for r in flagged:
    print("    evt%-8d node%-8d %-8s %-5s vgap %6.1f dv %6.2f ncand %2d -> %s"
          % (r['event'], r['node'], r['verdict'], 'HARD' if not r['clean'] else 'CLEAN',
             r['vgap'], r['dv'], r['ncand'], (r['comment'][:52] or '(low confidence)')))

print("\n=== 3b. A VETO ON vgap IS A DIAL, NOT A FREE FILTER ===")
print("  Every false fire has vgap >= 13.4 cm -- but so do 20 of the 33 correct")
print("  cuts (they run to 231.7 cm).  So vgap does not SEPARATE the two; it is a")
print("  purity/efficiency dial with a measured price, and the price is steep.")
pos_all = [r for r in rows if r['pos']]
fires_all = [r for r in rows if r['fired']]
print("  %-9s %6s %6s %6s %7s %7s" % ('vgap<=', 'fires', 'right', 'wrong', 'eff', 'pur'))
for t in (3, 5, 8, 10, 13, 15, 20, 30, 1e9):
    ff = [r for r in fires_all if r['vgap'] <= t]
    rr = [r for r in ff if r['pos']]
    print("  %-9s %6d %6d %6d %7.3f %7.3f"
          % (('%g' % t) if t < 1e8 else 'no veto', len(ff), len(rr), len(ff) - len(rr),
             len(rr) / max(len(pos_all), 1), len(rr) / max(len(ff), 1)))
print("  Read the LAST TWO COLUMNS against the owner's instruction to balance:")
print("  a veto at 13 cm buys purity 0.805 -> 1.000 and pays efficiency")
print("  0.767 -> 0.372.  That is not obviously a good trade and is his call.")

print("\n=== 4. THE FALSE FIRES, by stratum -- where does the cost actually sit? ===")
ff = [r for r in rows if r['fired'] and not r['pos']]
print("  %d false fires: %d CLEAN, %d HARD"
      % (len(ff), sum(1 for r in ff if r['clean']), sum(1 for r in ff if not r['clean'])))
for r in sorted(ff, key=lambda r: (r['clean'], -r['vgap'])):
    print("    evt%-8d node%-8d %-6s %-5s vgap %6.1f ncand %2d valley %.3f  %s"
          % (r['event'], r['node'], r['verdict'], 'HARD' if not r['clean'] else 'CLEAN',
             r['vgap'], r['ncand'], r['valley'], r['comment'][:44]))

print("\n=== 5. THE MISSES, by stratum ===")
ms = [r for r in rows if r['pos'] and not r['fired']]
print("  %d misses: %d CLEAN, %d HARD"
      % (len(ms), sum(1 for r in ms if r['clean']), sum(1 for r in ms if not r['clean'])))
for r in sorted(ms, key=lambda r: -r['valley']):
    print("    evt%-8d node%-8d %-8s %-5s valley %.3f n_seed %d  %s"
          % (r['event'], r['node'], r['verdict'], 'HARD' if not r['clean'] else 'CLEAN',
             r['valley'], r['n_seed'], r['comment'][:40]))

with open(args.tsv, 'w') as f:
    w = csv.writer(f, delimiter='\t')
    f.write("# doc pr/138 -- a-priori cleanliness strata (vgap<=%.0f, |dv|<%.0f, ncand<=%d)\n"
            % (VGAP_MAX, DV_MAX, NCAND_MAX))
    w.writerow(['event', 'node', 'owner', 'positive', 'fired', 'clean',
                'vgap_cm', 'dv_cm', 'n_cand', 'valley_best', 'n_seed', 'confidence'])
    for r in rows:
        w.writerow([r['event'], r['node'], r['verdict'], int(r['pos']), r['fired'],
                    int(r['clean']), '%.2f' % r['vgap'], '%.3f' % r['dv'], r['ncand'],
                    '%.4f' % r['valley'], r['n_seed'], r['conf']])
print("\nwrote %s" % args.tsv)

if args.png:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    # (a) valley_best vs vgap, marker = verdict, colour = fired
    a = ax[0][0]
    for pos, mk, nm in ((True, 'o', 'owner SPLIT'), (False, 'x', 'owner KEEP/TRIM')):
        sel = [r for r in rows if r['pos'] == pos]
        a.scatter([max(r['vgap'], 0.05) for r in sel], [r['valley'] for r in sel],
                  c=['#d62728' if r['fired'] else '#1f77b4' for r in sel],
                  marker=mk, s=34, alpha=0.85, label=nm)
    a.axhline(0.95, color='k', ls='--', lw=1)
    a.axvline(VGAP_MAX, color='g', ls=':', lw=1.5)
    a.set_xscale('log'); a.set_xlabel('vertex-to-nearest-charge  vgap [cm]')
    a.set_ylabel('valley_best   (red = the trigger fired)')
    a.set_title('(a) the trigger vs vertex health\ndashed = the 0.95 cut, dotted = the CLEAN bound')
    a.legend(fontsize=8)
    # (b) efficiency / purity per stratum
    b = ax[0][1]
    labels, effs, purs = [], [], []
    for nm, sel in (('all 164', rows),
                    ('vertex ok', [r for r in rows if r['ok_v']]),
                    ('not re-seated', [r for r in rows if r['ok_m']]),
                    ('not busy', [r for r in rows if r['ok_b']]),
                    ('CLEAN', [r for r in rows if r['clean']]),
                    ('HARD', [r for r in rows if not r['clean']])):
        f_ = [r for r in sel if r['fired']]; p_ = [r for r in sel if r['pos']]
        t_ = [r for r in f_ if r['pos']]
        labels.append('%s\nn=%d' % (nm, len(sel)))
        effs.append(len(t_) / max(len(p_), 1)); purs.append(len(t_) / max(len(f_), 1))
    x = np.arange(len(labels))
    b.bar(x - 0.2, effs, 0.4, label='efficiency', color='#1f77b4')
    b.bar(x + 0.2, purs, 0.4, label='purity', color='#d62728')
    b.set_xticks(x); b.set_xticklabels(labels, fontsize=7)
    b.set_ylim(0, 1.05); b.axhline(1.0, color='k', lw=0.5)
    b.set_title('(b) does the splitter do better when the situation is clean?')
    b.legend(fontsize=8)
    # (c) busyness
    c = ax[1][0]
    bins = [0, 1, 2, 3, 5, 8, 100]
    for pos, col, nm in ((True, '#2ca02c', 'owner SPLIT'), (False, '#7f7f7f', 'owner KEEP/TRIM')):
        h = np.histogram([r['ncand'] for r in rows if r['pos'] == pos], bins=bins)[0]
        c.step(bins[:-1], h, where='post', color=col, lw=2, label=nm)
    hf = np.histogram([r['ncand'] for r in rows if r['fired'] and not r['pos']], bins=bins)[0]
    c.step(bins[:-1], hf, where='post', color='#d62728', lw=2, ls='--', label='FALSE fires')
    c.axvline(NCAND_MAX + 0.5, color='g', ls=':', lw=1.5)
    c.set_xlabel('candidates above the floors in the event  (event busyness)')
    c.set_ylabel('objects'); c.set_title('(c) the owner\'s "very busy events"')
    c.legend(fontsize=8)
    # (d) the trade the owner asked to balance
    d = ax[1][1]
    d.axis('off')
    txt = ["THE OWNER'S BALANCE, per stratum", ""]
    for nm, sel in (('CLEAN', [r for r in rows if r['clean']]),
                    ('HARD', [r for r in rows if not r['clean']])):
        f_ = [r for r in sel if r['fired']]; p_ = [r for r in sel if r['pos']]
        t_ = [r for r in f_ if r['pos']]
        w_ = [r for r in f_ if not r['pos']]
        txt += ["%-6s  n=%-4d positives %-3d" % (nm, len(sel), len(p_)),
                "        right cuts %-3d   wrong cuts %-3d" % (len(t_), len(w_)),
                "        eff %.3f  pur %.3f" % (len(t_) / max(len(p_), 1),
                                                len(t_) / max(len(f_), 1)), ""]
    txt += ["a-priori CLEAN = vgap<=%.0fcm AND |dv|<%.0fcm AND n_cand<=%d" % (VGAP_MAX, DV_MAX, NCAND_MAX),
            "no label or outcome was used to define it"]
    d.text(0.02, 0.98, "\n".join(txt), va='top', family='monospace', fontsize=9)
    plt.tight_layout()
    out = 'docs/pr/pr138-clean-strata.png'
    plt.savefig(out, dpi=115)
    print("wrote %s" % out)
