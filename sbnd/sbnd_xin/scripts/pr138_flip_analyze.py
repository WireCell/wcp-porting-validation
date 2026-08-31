#!/usr/bin/env python3
# doc pr/138 -- "should shower_split be turned ON for SBND running?"
"""Adjudicate the flip on the owner's own instruments, at BOTH operating points.

TWO PAIRS, and the distinction is the whole reason this is not one number:

  poff / pon        PRODUCTION config.  q_extra is already at its floor (6.9 %),
                    so a splitter here can only take charge OUT of showers.  This
                    is the SAFETY gate -- "does turning it on break anything".
  c90off / c90on    the pass-4 angle_v1 escape ON (doc pr/136's onV1c90).  The
                    escape buys q_miss and PAYS q_extra, and giving that q_extra
                    back is the job doc pr/138 sec C1 designed the splitter for.
                    This is the EFFICACY test.

THE RULE, from the owner (2026-08-31): "a bit q_miss is OK, we want to balance
the q_miss vs. q_extra."  So this prints the trade rather than applying a
threshold -- the balance is his call, not the script's.

TWO MEASUREMENT CAVEATS THAT MUST TRAVEL WITH THE NUMBERS, both proven here:

  1. The ABSOLUTE q_miss depends on whether the arm ran WCT_SHOWER_CONTENT_DEBUG.
     em117_score prefers a probe-built sidecar and falls back to the dump's
     single-valued segments[].shower_id.  Crossing pr/136's own f086probe DUMPS
     with this round's (probe-less) prepdir reproduces 15.1 %, not the published
     14.0 % -- so the offset is the SIDECAR, not the arm's physics.  Every arm in
     this round is scored the same way, so the OFF->ON deltas are like-for-like;
     the absolutes are not comparable to doc pr/136's.  Reported lossiness of the
     fallback join is 0 members on all four scores.
  2. The completeness instrument CANNOT grade a splitter on its own.  Its target
     is `(members | marked-in) - marked-out` from the 2026-08-27/28 attribution
     scan, which called several of these objects ONE shower; the 2026-09-01 split
     scan says they are three to five.  So a CORRECT cut scores as q_miss.  The
     per-object decomposition below separates that from real cost.

Repro:
    python3 scripts/pr138_flip_analyze.py --png
"""
import os, sys, re, csv, json, glob, argparse, collections

ap = argparse.ArgumentParser()
ap.add_argument('--pairs', default='poff:pon,c90off:c90on')
ap.add_argument('--tsv', default='docs/pr/pr138-flip-decision.tsv')
ap.add_argument('--png', action='store_true')
args = ap.parse_args()
SPL = ('SPLIT2', 'SPLIT3', 'SPLIT4+')


def comp(tag):
    f = 'docs/pr/pr138-completeness-%s.tsv' % tag
    if not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f), delimiter='\t'):
        out[(int(r['event']), int(r['shower']))] = r
    return out


def totals(d):
    m = sum(float(r['q_miss'] or 0) for r in d.values())
    e = sum(float(r['q_extra'] or 0) for r in d.values())
    t = sum(float(r['q_target'] or 0) for r in d.values())
    return 100 * m / t, 100 * e / t, t


def census(tag):
    p = '/home/xqian/tmp/pr138-census-%s.txt' % tag
    if not os.path.exists(p):
        return None, None
    ex = sh = None
    for ln in open(p):
        m = re.search(r'exact\s+(\d+)\s+([\d.]+) %', ln)
        if m:
            ex = int(m.group(1))
        m = re.search(r'sharing a gamma (\d+) %', ln)
        if m:
            sh = int(m.group(1))
    return ex, sh


def closure(tag):
    f = 'docs/pr/pr138-closure-%s.tsv' % tag
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader((l for l in open(f) if not l.startswith('#')), delimiter='\t'))
    imp = sum(1 for r in rows if r['R_prod'] and float(r['R_prod']) < 1)
    over = sum(1 for r in rows if r['m_prod'] and float(r['m_prod']) > 160)
    return dict(n=len(rows), impossible=imp, over=over,
                rows={(r['event'], r['sample']): r for r in rows})


fires = {}
if os.path.exists('docs/pr/pr138-predict-delta.tsv'):
    for r in csv.DictReader((l for l in open('docs/pr/pr138-predict-delta.tsv')
                             if not l.startswith('#')), delimiter='\t'):
        fires[(int(r['event']), int(r['node']))] = r

print(__doc__.split('Repro:')[0].rstrip())
print("=" * 78)

summary = []
for pair in args.pairs.split(','):
    off, on = pair.split(':')
    A, B = comp(off), comp(on)
    if A is None or B is None:
        print("\n### %s -> %s : NOT AVAILABLE YET" % (off, on))
        continue
    (mo, eo, tgt), (mn, en, _) = totals(A), totals(B)
    exo, sho = census(off)
    exn, shn = census(on)
    co, cn = closure(off), closure(on)
    print("\n### %s  ->  %s" % (off.upper(), on.upper()))
    print("  %-34s %10s %10s %10s" % ("instrument", "OFF", "ON", "delta"))
    print("  %-34s %9.1f%% %9.1f%% %+9.1f" % ("q_miss (hand-scan attribution)", mo, mn, mn - mo))
    print("  %-34s %9.1f%% %9.1f%% %+9.1f" % ("q_extra", eo, en, en - eo))
    if exo is not None:
        print("  %-34s %10s %10s %+10d" % ("pi0 census EXACT (of 66)", exo, exn, exn - exo))
        print("  %-34s %9d%% %9d%% %+9d" % ("pairs sharing a gamma", sho, shn, shn - sho))
    if co:
        print("  %-34s %6d/%-3d %6d/%-3d %+10d"
              % ("pi0 pairs KINEMATICALLY IMPOSSIBLE", co['impossible'], co['n'],
                 cn['impossible'], cn['n'], cn['impossible'] - co['impossible']))
        print("  %-34s %10d %10d %+10d" % ("over-clustering class (m>160)", co['over'], cn['over'],
                                           cn['over'] - co['over']))
    # per-object decomposition of both deltas
    agg = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for k in sorted(set(A) & set(B)):
        dm = float(B[k]['q_miss'] or 0) - float(A[k]['q_miss'] or 0)
        de = float(B[k]['q_extra'] or 0) - float(A[k]['q_extra'] or 0)
        if abs(dm) < 1 and abs(de) < 1:
            continue
        f = fires.get(k)
        v = (f or {}).get('owner_verdict')
        cls = ('the split scan says SPLIT' if v in SPL else
               'the split scan says KEEP' if v in ('KEEP', 'TRIM') else
               'fired, no split label' if f else 'no fire on this object')
        agg[cls][0] += dm; agg[cls][1] += de; agg[cls][2] += 1
    print("\n  WHERE THE TWO DELTAS COME FROM  (%.4g of q_target over 90 marked showers)" % tgt)
    print("  %-28s %4s %12s %8s %12s %8s" % ('class', 'n', 'd q_miss', 'pt', 'd q_extra', 'pt'))
    TM = TE = 0.0
    for c, (dm, de, n) in sorted(agg.items(), key=lambda t: -abs(t[1][0])):
        TM += dm; TE += de
        print("  %-28s %4d %12.4g %+8.2f %12.4g %+8.2f"
              % (c, n, dm, 100 * dm / tgt, de, 100 * de / tgt))
    print("  %-28s %4s %12.4g %+8.2f %12.4g %+8.2f" % ('TOTAL', '', TM, 100 * TM / tgt,
                                                        TE, 100 * TE / tgt))
    sp = agg['the split scan says SPLIT']; kp = agg['the split scan says KEEP']
    if TM:
        print("\n  SYMMETRIC READING -- the two hand scans disagree in BOTH directions:")
        print("    %.0f%% of the q_miss RISE sits on objects the split scan says MUST be"
              % (100 * sp[0] / TM))
        print("    split -- the attribution scan called them one shower, so a correct cut")
        print("    scores as a miss.  And %.0f%% of the q_extra GAIN sits on objects the"
              % (100 * kp[1] / TE if TE else 0))
        print("    split scan calls KEEP -- there the attribution scan agrees the charge")
        print("    did not belong.  Neither scan alone adjudicates a split.")
        print("    Cost where BOTH scans agree (the KEEP class): q_miss %+.2f pt for"
              % (100 * kp[0] / tgt))
        print("    q_extra %+.2f pt." % (100 * kp[1] / tgt))
    # IS THE CENSUS'S 32 THE SAME 32?  "no change" and "two gains cancelling two
    # losses" are very different results and the headline count cannot tell them
    # apart.
    ca, cb = ('docs/pr/pr138-census-%s.tsv' % t for t in (off, on))
    if os.path.exists(ca) and os.path.exists(cb):
        def cen(f):
            return {(r['setname'], r['sample'], r['event']): r
                    for r in csv.DictReader((l for l in open(f) if not l.startswith('#')),
                                            delimiter='\t')}
        CA, CB = cen(ca), cen(cb)
        ch = [(k, CA[k]['match'], CB[k]['match']) for k in sorted(set(CA) & set(CB))
              if CA[k]['match'] != CB[k]['match']]
        ea = {k for k, r in CA.items() if r['match'] == 'exact'}
        eb = {k for k, r in CB.items() if r['match'] == 'exact'}
        print("\n  THE CENSUS, UNPACKED -- same count is not the same set")
        print("    exact OFF %d, exact ON %d, identical set: %s"
              % (len(ea), len(eb), ea == eb))
        print("    events whose match class changed: %d" % len(ch))
        for k, a, b in ch:
            arrow = 'BETTER' if ('exact', b) == ('exact', 'exact') or (a, b) in (
                ('no-group', 'exact'), ('no-group', 'partial'), ('partial', 'exact')) else 'WORSE'
            print("      evt%-9s %-10s -> %-10s  %s" % (k[2], a, b, arrow))

    # the pi0 pairs that moved
    if co:
        moved = [(k, co['rows'][k], cn['rows'][k]) for k in sorted(set(co['rows']) & set(cn['rows']))
                 if co['rows'][k]['R_prod'] != cn['rows'][k]['R_prod']]
        print("\n  THE pi0 PAIRS THAT MOVED (%d of %d)" % (len(moved), co['n']))
        print("  %-9s %-7s %8s %8s %8s %8s   %s" % ('event', 'sample', 'm OFF', 'm ON',
                                                     'R OFF', 'R ON', 'verdict'))
        for k, a, b in moved:
            ra, rb = float(a['R_prod']), float(b['R_prod'])
            tag = ('CROSSED INTO IMPOSSIBLE' if ra >= 1 > rb else
                   'rescued' if rb >= 1 > ra else
                   'closer to 135' if abs(float(b['m_prod']) - 135) < abs(float(a['m_prod']) - 135)
                   else 'further from 135')
            print("  %-9s %-7s %8.1f %8.1f %8.3f %8.3f   %s"
                  % (k[0], k[1], float(a['m_prod']), float(b['m_prod']), ra, rb, tag))
    summary.append(dict(pair=pair, q_miss=(mo, mn), q_extra=(eo, en),
                        census=(exo, exn), impossible=(co['impossible'] if co else None,
                                                        cn['impossible'] if cn else None)))

if args.png:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

    # (a) the trade the owner asked to balance, at both operating points
    a = ax[0][0]
    xs, w = np.arange(len(summary)), 0.35
    a.bar(xs - w / 2, [s['q_miss'][1] - s['q_miss'][0] for s in summary], w,
          color='#d62728', label='delta q_miss  (cost)')
    a.bar(xs + w / 2, [s['q_extra'][1] - s['q_extra'][0] for s in summary], w,
          color='#2ca02c', label='delta q_extra (gain)')
    a.axhline(0, color='k', lw=0.8)
    a.set_xticks(xs)
    a.set_xticklabels(['%s\nOFF %.1f/%.1f' % (s['pair'].split(':')[0],
                                               s['q_miss'][0], s['q_extra'][0])
                       for s in summary], fontsize=8)
    a.set_ylabel('percentage points of q_target')
    a.set_title('(a) the balance the owner asked for\n'
                'most of the q_miss rise is the label-epoch artefact, see (d)')
    a.legend(fontsize=8)

    # (b) the pi0 instruments
    b = ax[0][1]
    labs, offs, ons = [], [], []
    for s in summary:
        if s['census'][0] is not None:
            labs.append('%s\ncensus exact' % s['pair'].split(':')[0])
            offs.append(s['census'][0]); ons.append(s['census'][1])
        if s['impossible'][0] is not None:
            labs.append('%s\nimpossible pairs' % s['pair'].split(':')[0])
            offs.append(s['impossible'][0]); ons.append(s['impossible'][1])
    xs = np.arange(len(labs))
    b.bar(xs - 0.2, offs, 0.4, color='#7f7f7f', label='knob OFF')
    b.bar(xs + 0.2, ons, 0.4, color='#1f77b4', label='knob ON')
    b.set_xticks(xs); b.set_xticklabels(labs, fontsize=7)
    b.set_title('(b) pi0 reconstruction: census exact (higher better)\n'
                'and kinematically impossible pairs (LOWER better)')
    b.legend(fontsize=8)

    # (c) the pi0 pairs that moved
    c = ax[1][0]
    co_, cn_ = closure('poff'), closure('pon')
    if co_ and cn_:
        mv = [(k, co_['rows'][k], cn_['rows'][k])
              for k in sorted(set(co_['rows']) & set(cn_['rows']))
              if co_['rows'][k]['R_prod'] != cn_['rows'][k]['R_prod']]
        for i, (k, u, v) in enumerate(mv):
            m0, m1 = float(u['m_prod']), float(v['m_prod'])
            good = abs(m1 - 135) < abs(m0 - 135)
            c.annotate('', xy=(m1, i), xytext=(m0, i),
                       arrowprops=dict(arrowstyle='->', lw=2,
                                       color='#2ca02c' if good else '#d62728'))
            c.text(max(m0, m1) + 8, i, k[0], va='center', fontsize=8)
        c.axvline(135, color='k', ls='--', lw=1.2)
        c.set_yticks(range(len(mv))); c.set_yticklabels([])
        c.set_xlim(40, 700); c.set_xscale('log')
        c.set_xlabel('reconstructed pi0 mass [MeV]   (dashed = 135)')
        c.set_title('(c) the 8 hand pi0 pairs the splitter moved\n'
                    'green = toward 135, red = away')

    # (d) where the deltas come from
    d = ax[1][1]
    d.axis('off')
    txt = ["WHY THE RAW q_miss NUMBER OVERSTATES THE COST", "",
           "The completeness target comes from the 2026-08-27/28",
           "ATTRIBUTION scan, which called several of these objects",
           "ONE shower.  The 2026-09-01 SPLIT scan says they are 3-5.",
           "So a CORRECT cut is scored as a miss.", "",
           "  production pair, decomposed:",
           "    the split scan says SPLIT   q_miss +4.01 pt  q_extra -0.90 pt",
           "    the split scan says KEEP    q_miss +0.29 pt  q_extra -0.41 pt",
           "    fired, unlabelled           q_miss  0.00 pt  q_extra -0.19 pt",
           "    TOTAL                       q_miss +4.30 pt  q_extra -1.50 pt", "",
           "93% of the rise is the first line -- the instrument",
           "penalising the splitter for obeying the later scan.",
           "28% of the GAIN is the second -- the attribution scan",
           "agreeing that charge did not belong on objects the split",
           "scan calls KEEP.  THE TWO SCANS DISAGREE BOTH WAYS.", "",
           "Absolute q_miss here is the no-sidecar variant (15.1% vs",
           "doc pr/136's 14.0%); proven to be the PREPDIR, not the arm,",
           "by scoring pr/136's own dumps with this round's prepdir."]
    d.text(0.01, 0.99, "\n".join(txt), va='top', family='monospace', fontsize=8.2)
    plt.tight_layout()
    out = 'docs/pr/pr138-flip-decision.png'
    plt.savefig(out, dpi=115)
    print("wrote %s" % out)

with open(args.tsv, 'w') as f:
    w = csv.writer(f, delimiter='\t')
    f.write("# doc pr/138 -- the flip decision, both operating points\n")
    w.writerow(['pair', 'q_miss_off', 'q_miss_on', 'q_extra_off', 'q_extra_on',
                'census_off', 'census_on', 'impossible_off', 'impossible_on'])
    for s in summary:
        w.writerow([s['pair'], '%.2f' % s['q_miss'][0], '%.2f' % s['q_miss'][1],
                    '%.2f' % s['q_extra'][0], '%.2f' % s['q_extra'][1],
                    s['census'][0], s['census'][1], s['impossible'][0], s['impossible'][1]])
print("\nwrote %s" % args.tsv)
