#!/usr/bin/env python3
"""doc 97 -- per-BUNDLE A/B between two PR arms, keyed on the FLASH.

Why not doc94r2_flip_report.py.  That tool keys bundles on `main_id` from
nusel-evt<ID>.tsv, which is correct for a PR-stage knob: the Q/L input is
byte-identical, so cluster ids are too.  This round's knob is a CLUSTERING
knob.  When ClusteringSeparate splits a member the tree renumbers every
cluster after it (`cluster_id_order: 'tree'`, doc 96 sec 1.1), so on a firing
event EVERY bundle changes main_id and a main_id-keyed diff reports the whole
event as one-arm-only.

The flash is the stable handle: a bundle is a (cluster, flash) match, and the
flash inventory is an input to Q/L matching, not an output of clustering.
Measured on 272-2-30 (the one event where the knob does the most): all 23 OFF
bundles keep their (flash_apa, flash_gid, flash_time_us) in the ON arm, while
23 of 23 main_ids move.

Reports, per sample:
  * bundles matched on both sides and their field flips
    (in_beam, tgm, stm, fc, lm, label) plus main-size changes;
  * bundles that exist on one side only (a split frees a cluster onto its own
    flash: that is a NEW bundle, not a lost one -- both directions are listed);
  * the population counts the owner asked for: in-beam bundles, nu-candidates,
    TGM, STM, FC, LM.

  Usage: d97_flip_report.py <on-suffix> [off-suffix] [sample ...]
         d97_flip_report.py d97onpr r3entry
Read-only.
"""
import glob, os, sys
from collections import Counter

ON = sys.argv[1] if len(sys.argv) > 1 else 'd97onpr'
OFF = sys.argv[2] if len(sys.argv) > 2 else 'r3entry'
SAMPLES = sys.argv[3:] or ['ncpi0', 'nuecc48', 'mcp1k', 'mcp2k']

FIELDS = ('in_beam', 'tgm', 'stm', 'fc', 'lm', 'label')
SIZE = ('npts_main', 'len_main_cm', 'n_bundle', 'n_frag')


def rows(path):
    """{(flash_apa, flash_gid): [row-dict, ...]} for one nusel table, or None.

    (flash_apa, flash_gid) is NOT unique: one flash can be matched to a bundle
    GROUP holding several mains -- nueCC48 18255-1-10550 has a 131.8 cm
    nu-candidate and a 374.1 cm TGM sharing flash 1/1000002.  Keying a dict on
    the pair alone silently drops one of them and understates every count, so
    the value is a LIST and pairing inside a flash is done by size below.
    """
    if not os.path.exists(path):
        return None
    raw = [l.split() for l in open(path) if l.strip()]
    if len(raw) < 2:
        return {}
    head = raw[0]
    out = {}
    for r in raw[1:]:
        d = dict(zip(head, r))
        out.setdefault((d['flash_apa'], d['flash_gid']), []).append(d)
    return out


def pair_within_flash(off_list, on_list):
    """Greedily pair two bundle lists sharing one flash, closest npts_main first.

    Returns (pairs, off_only, on_only).  With one bundle a side -- the case for
    99.9% of flashes -- this is the identity pairing.  Where a flash carries
    several mains their sizes differ by orders of magnitude (2853 vs 9519 points
    in the case above), so nearest-size is unambiguous; a split that halves one
    of them still pairs to itself rather than to its sibling.
    """
    if len(off_list) == 1 and len(on_list) == 1:
        return [(off_list[0], on_list[0])], [], []
    cand = sorted(((abs(int(o['npts_main']) - int(n['npts_main'])), i, j)
                   for i, o in enumerate(off_list) for j, n in enumerate(on_list)))
    used_o, used_n, pairs = set(), set(), []
    for _d, i, j in cand:
        if i in used_o or j in used_n:
            continue
        used_o.add(i); used_n.add(j)
        pairs.append((off_list[i], on_list[j]))
    return (pairs,
            [o for i, o in enumerate(off_list) if i not in used_o],
            [n for j, n in enumerate(on_list) if j not in used_n])


tot = Counter()
flips, only_on, only_off, resized = [], [], [], []


def one_pair(smp, evt, k, off_r, on_r):
    tot['bundles'] += 1
    df = [f for f in FIELDS if on_r[f] != off_r[f]]
    ds = [f for f in SIZE if on_r[f] != off_r[f]]
    in_beam = off_r['in_beam'] == '1' or on_r['in_beam'] == '1'
    if df:
        flips.append((smp, evt, k, df, off_r, on_r)); tot['bundle-flip'] += 1
        if in_beam:
            tot['bundle-flip-in-beam'] += 1
    elif ds:
        tot['bundle-resize-only'] += 1
        # DIRECTION matters and no verdict field records it.  An IN-BEAM main
        # that got materially LONGER is a neutrino fused into a cosmic -- the
        # one regression mode that flips no field and, if the PR chain's
        # unmerge undoes the grouping, moves no calib physics either.  Both
        # other instruments are blind to it; this bucket is not.
        if in_beam:
            resized.append((smp, evt, k, off_r, on_r))
            tot['bundle-resize-in-beam'] += 1

for smp in SAMPLES:
    for on_path in sorted(glob.glob(f'work-{smp}-{ON}/pr_evt*/nusel-evt*.tsv')):
        evt = on_path.split('pr_evt')[1].split('/')[0]
        a = rows(on_path)
        b = rows(on_path.replace(f'-{ON}/', f'-{OFF}/'))
        if b is None:
            tot['event-missing-off'] += 1
            continue
        tot['events'] += 1
        for k in sorted(set(a) | set(b)):
            off_l, on_l = b.get(k, []), a.get(k, [])
            if len(off_l) > 1 or len(on_l) > 1:
                tot['flash-with-several-mains'] += 1
            pairs, o_only, n_only = pair_within_flash(off_l, on_l)
            for r in n_only:
                only_on.append((smp, evt, k, r)); tot['bundle-only-on'] += 1
            for r in o_only:
                only_off.append((smp, evt, k, r)); tot['bundle-only-off'] += 1
            for off_r, on_r in pairs:
                one_pair(smp, evt, k, off_r, on_r)
        for tag, arm in (('on', a), ('off', b)):
            for r in [x for lst in arm.values() for x in lst]:
                if r['in_beam'] == '1':
                    tot[f'{tag}:in_beam'] += 1
                    tot[f"{tag}:label={r['label']}"] += 1
                    for f in ('tgm', 'stm', 'fc'):
                        if r[f] == '1':
                            tot[f'{tag}:{f}'] += 1
                    if r['lm'] != '0':
                        tot[f'{tag}:lm'] += 1

print(f'ON = work-*-{ON}   OFF = work-*-{OFF}   samples: {" ".join(SAMPLES)}')
print(f'events {tot["events"]}  bundles matched on flash {tot["bundles"]}  '
      f'field flips {tot["bundle-flip"]} (in-beam {tot["bundle-flip-in-beam"]})  '
      f'size-only changes {tot["bundle-resize-only"]}')
print(f'bundles only in ON {tot["bundle-only-on"]}   only in OFF {tot["bundle-only-off"]}'
      f'   events missing from OFF {tot["event-missing-off"]}')
print()
print('in-beam population (the number a clustering knob moves first):')
keys = sorted({k.split(':', 1)[1] for k in tot if k.startswith(('on:', 'off:'))})
print(f'  {"quantity":<28}{"OFF":>8}{"ON":>8}{"delta":>8}')
for q in keys:
    o, n = tot.get(f'off:{q}', 0), tot.get(f'on:{q}', 0)
    if o or n:
        print(f'  {q:<28}{o:>8}{n:>8}{n - o:>+8}')

if flips:
    print(f'\nFIELD FLIPS ({len(flips)}):')
    print(f'  {"sample":<8}{"event":>8}  {"flash":<14}{"beam":>5} {"field":<10}{"off":<14}{"on":<14}'
          f'{"len off":>9}{"len on":>9}')
    for smp, evt, k, df, o, n in flips:
        # in_beam is what makes a flip a physics result: an out-of-time bundle
        # is a cosmic that no analysis reads.  Print it on every row.
        beam = f'{o["in_beam"]}/{n["in_beam"]}'
        for f in df:
            print(f'  {smp:<8}{evt:>8}  {k[0]+"/"+k[1]:<14}{beam:>5} {f:<10}{o[f]:<14}{n[f]:<14}'
                  f'{float(o["len_main_cm"]):>9.1f}{float(n["len_main_cm"]):>9.1f}')
if resized:
    grew = [r for r in resized if float(r[4]['len_main_cm']) > float(r[3]['len_main_cm']) + 1.0]
    shrank = [r for r in resized if float(r[4]['len_main_cm']) < float(r[3]['len_main_cm']) - 1.0]
    print(f'\nIN-BEAM MAINS THAT CHANGED SIZE BUT KEPT EVERY VERDICT '
          f'({len(resized)}; grew by >1 cm: {len(grew)}, shrank by >1 cm: {len(shrank)}):')
    print(f'  {"sample":<8}{"event":>8}  {"flash":<14}{"len off":>9}{"len on":>9}'
          f'{"d cm":>9}{"pts off":>9}{"pts on":>9}  {"label":<14}')
    for smp, evt, k, o, n in sorted(resized,
                                    key=lambda r: -(float(r[4]['len_main_cm'])
                                                    - float(r[3]['len_main_cm']))):
        lo, ln = float(o['len_main_cm']), float(n['len_main_cm'])
        print(f'  {smp:<8}{evt:>8}  {k[0]+"/"+k[1]:<14}{lo:>9.1f}{ln:>9.1f}{ln - lo:>+9.1f}'
              f'{o["npts_main"]:>9}{n["npts_main"]:>9}  {n["label"]:<14}')

for name, lst in (('ONLY IN ON (a split freed a cluster onto its own flash)', only_on),
                  ('ONLY IN OFF (a bundle disappeared)', only_off)):
    if lst:
        print(f'\n{name} ({len(lst)}):')
        for smp, evt, k, r in lst[:60]:
            print(f'  {smp:<8}{evt:>8}  flash {k[0]}/{k[1]:<10} t={float(r["flash_time_us"]):>10.1f}us  '
                  f'main {r["npts_main"]:>6} pts {float(r["len_main_cm"]):>7.1f} cm  '
                  f'in_beam={r["in_beam"]} {r["label"]}')
        if len(lst) > 60:
            print(f'  ... and {len(lst) - 60} more')
