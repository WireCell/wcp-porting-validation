#!/usr/bin/env python3
"""Doc 66 sec 11: score the owner's d66flip hand scan of the 11 diffusion flips.

The owner scanned the NEW (4.0/8.8) arm on :5011 with --tag d66flip and the OLD
arm as --prev, i.e. every bundle he looked at was flagged as "this verdict
changed because of the diffusion".  His verdict is recorded as free text in the
bundle `comment` field (scan_labels were not used), so this script does NOT try
to be clever: it prints the raw comment next to the code's verdict and applies
one explicit keyword rule, stated here so it can be checked by eye ---

    a comment containing "not a stm" (case-insensitive), or "nu candidate"
    without "stm is fine", means the OWNER's verdict is NOT-STM (= nu-candidate,
    doc 61's key: the verdict space is STM | nu only).

Every classification is printed with the comment that produced it.  Anything the
rule cannot decide is reported UNCLEAR and counted separately rather than
guessed.

Usage: ./d66_scan_score.py [--tag d66flip] [--root .]
"""
import argparse
import glob
import json
import os

HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

# the 11 flips of doc 66 sec 4a: event -> (main_id, direction, doc-62 owner verdict)
FLIPS = {
    '281632': (8,  'STM->nu', 'STM'),
    '283463': (14, 'STM->nu', 'STM'),
    '315849': (10, 'STM->nu', 'STM'),
    '319809': (20, 'nu->STM', 'not-STM'),
    '321107': (13, 'STM->nu', 'not-STM'),
    '58345':  (7,  'STM->nu', None),
    '58755':  (21, 'nu->STM', None),
    '63163':  (6,  'STM->nu', None),
    '289295': (15, 'nu->STM', None),
    '317543': (15, 'nu->STM', None),
    '390864': (16, 'nu->STM', None),
}


def owner_verdict(comment):
    """-> 'not-STM' | 'STM' | None (unclear).  One explicit rule, see docstring."""
    c = (comment or '').strip().lower()
    if not c:
        return None
    if 'not a stm' in c or 'not stm' in c:
        return 'not-STM'
    if 'nu candidate' in c and 'stm is fine' not in c:
        return 'not-STM'
    if 'stm is fine' in c or c in ('stm', 'is a stm', 'good stm'):
        return 'STM'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')   # 2026-09-01: the d66 arm is retired; the label store now lives at ./nusel_labels/d66flip (doc 91)
    ap.add_argument('--tag', default='d66flip')
    a = ap.parse_args()
    root = a.root if os.path.isabs(a.root) else os.path.join(HERE, a.root)
    d = os.path.join(root, 'nusel_labels', a.tag)

    labelled = {os.path.basename(p)[len('nusel-labels-evt'):-len('.json')]
                for p in glob.glob(os.path.join(d, 'nusel-labels-evt*.json'))}
    visited = {os.path.basename(p)[len('.scan_state-evt'):-len('.json')]
               for p in glob.glob(os.path.join(d, '.scan_state-evt*.json'))}

    print(f"tag {a.tag} under {root}")
    print(f"  flips to scan : {len(FLIPS)}")
    print(f"  with labels   : {len(labelled & set(FLIPS))}")
    print(f"  visited only  : {sorted(visited - labelled)}   <- no verdict saved")
    print(f"  never visited : {sorted(set(FLIPS) - visited)}")
    print()

    hdr = (f"{'bundle':<13} {'direction':<9} {'code(new)':<13} {'owner':<9} "
           f"{'code ok?':<9} {'doc62':<8} comment")
    print(hdr)
    print('-' * len(hdr))
    tally = {'correct': [], 'wrong': [], 'unclear': [], 'unscanned': []}
    for evt, (main, direction, d62) in sorted(FLIPS.items(), key=lambda kv: kv[1][2] or 'zz'):
        p = os.path.join(d, f'nusel-labels-evt{evt}.json')
        key = f'{evt}:{main}'
        if not os.path.exists(p):
            print(f"{key:<13} {direction:<9} {'-':<13} {'-':<9} {'NOT SCANNED':<9} "
                  f"{str(d62):<8}")
            tally['unscanned'].append(key)
            continue
        doc = json.load(open(p))
        row = next((b for b in doc['bundles'] if b['main_id'] == main), None)
        if row is None:
            print(f"{key:<13} bundle absent from label file"); continue
        code = row['auto']['label']            # 'STM' | 'nu-candidate'
        code_stm = row['auto']['stm'] == 1
        com = row.get('comment', '')
        ov = owner_verdict(com)
        if ov is None:
            verdict, bucket = 'UNCLEAR', 'unclear'
        else:
            ok = (ov == 'STM') == code_stm
            verdict, bucket = ('YES' if ok else 'NO'), ('correct' if ok else 'wrong')
        tally[bucket].append(key)
        print(f"{key:<13} {direction:<9} {code:<13} {ov or '?':<9} {verdict:<9} "
              f"{str(d62):<8} {com!r}")

    print()
    n = len(tally['correct']) + len(tally['wrong'])
    print(f"scored bundles      : {n}")
    print(f"  code CORRECT      : {len(tally['correct'])}  {tally['correct']}")
    print(f"  code WRONG        : {len(tally['wrong'])}  {tally['wrong']}")
    print(f"  unclear comment   : {len(tally['unclear'])}  {tally['unclear']}")
    print(f"  not scanned       : {len(tally['unscanned'])}  {tally['unscanned']}")

    # the directional split is the headline -- report it explicitly
    print()
    for direction in ('STM->nu', 'nu->STM'):
        c = [k for k in tally['correct'] if FLIPS[k.split(':')[0]][1] == direction]
        w = [k for k in tally['wrong'] if FLIPS[k.split(':')[0]][1] == direction]
        print(f"  {direction}: {len(c)} correct, {len(w)} wrong"
              f"{'   -> ' + str(w) if w else ''}")

    # --- OLD arm vs NEW arm on the same truth -------------------------------
    # A flipped bundle's OLD verdict is by definition the opposite of its NEW
    # one, so the owner's verdict scores both arms with no extra input.  Doing
    # it mechanically avoids the arithmetic slip of scoring only the subset that
    # happens to sit in the doc-62 baseline.
    print()
    print("old arm vs new arm, on the owner's verdicts, over the flipped bundles only")
    print("(a flip's OLD verdict is the opposite of its NEW one by construction)")
    old_err = [k for k in tally['correct']]     # new correct => old wrong
    new_err = [k for k in tally['wrong']]       # new wrong   => old correct
    print(f"  OLD arm wrong : {len(old_err)}  {sorted(old_err)}")
    print(f"  NEW arm wrong : {len(new_err)}  {sorted(new_err)}")
    verdict = ("NEW better" if len(new_err) < len(old_err)
               else "OLD better" if len(new_err) > len(old_err) else "a wash")
    print(f"  => {verdict} by {abs(len(new_err) - len(old_err))} on these "
          f"{len(old_err) + len(new_err)} bundles")

    # split by whether the bundle was already in the doc-62 truth set, because
    # the two subsets point opposite ways and quoting only one is misleading
    print()
    for lbl, sel in (("in the doc-62 baseline", lambda k: FLIPS[k.split(':')[0]][2] is not None),
                     ("newly adjudicated here", lambda k: FLIPS[k.split(':')[0]][2] is None)):
        oe = [k for k in old_err if sel(k)]
        ne = [k for k in new_err if sel(k)]
        print(f"  {lbl:<24}: OLD wrong {len(oe)}, NEW wrong {len(ne)}")


if __name__ == '__main__':
    main()
