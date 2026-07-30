#!/usr/bin/env python3
"""Prototype-fidelity comparison between two qlport sweep arms.

For each event, parse the wire-cell-uboone-tagger-compare category table in
tagger_<EV>.log (each arm's toolkit output vs the SAME prototype reference)
and report:
  - total n_diff_branches (all categories) per arm
  - the *_flag_dir_weak branches that differ vs prototype, per arm

Usage: fidelity_compare.py <off_label_dir> <on_label_dir>
"""
import os, re, sys, glob

CAT_ROW = re.compile(r'^(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+')
# verbose per-branch row: name  ndiff/ncmp  maxdiff ...
BR_ROW = re.compile(r'^(\S*flag_dir_weak\S*)\s+(\d+)/(\d+)\s+')

def parse_log(path):
    total = None
    dw = []          # (branch, proto, tool) for differing dir_weak branches
    if not os.path.isfile(path):
        return None, None
    lines = open(path, errors='replace').read().splitlines()
    in_table = False
    for i, ln in enumerate(lines):
        if ln.startswith('category') and 'n_diff_branches' in ln:
            in_table = True
            total = 0
            continue
        if in_table:
            if ln.startswith('-'):
                continue
            m = CAT_ROW.match(ln)
            if m:
                total += int(m.group(4))
                continue
            if ln.strip() == '':
                in_table = False
                continue
        m = BR_ROW.match(ln)
        if m:
            # scalar row has "maxdiff proto tool default" on same line;
            # vector row is followed by proto/toolkit lines
            rest = ln[m.end():].split()
            if 'default=[]' in ln:
                proto = lines[i+1].split(':',1)[1].strip() if i+1 < len(lines) else '?'
                tool  = lines[i+2].split(':',1)[1].strip() if i+2 < len(lines) else '?'
            else:
                proto, tool = (rest[1], rest[2]) if len(rest) >= 3 else ('?','?')
            dw.append((m.group(1), proto, tool))
    return total, dw

def arm_events(d):
    out = {}
    for ed in sorted(glob.glob(os.path.join(d, '*_*/')),
                     key=lambda p: int(os.path.basename(p.rstrip('/')).split('_')[0])):
        b = os.path.basename(ed.rstrip('/'))
        idx, ev = b.split('_')
        logs = glob.glob(os.path.join(ed, 'tagger_*.log'))
        out[(int(idx), ev)] = logs[0] if logs else None
    return out

off_dir, on_dir = sys.argv[1], sys.argv[2]
off, on = arm_events(off_dir), arm_events(on_dir)

print(f"{'idx':>3} {'ev':>6} {'off_diffs':>9} {'on_diffs':>8} {'delta':>6}  dir_weak off -> on")
tot_off = tot_on = 0
n_better = n_worse = n_same = 0
dw_off_tot = dw_on_tot = 0
for key in sorted(off):
    idx, ev = key
    t_off, dw_off = parse_log(off[key]) if off[key] else (None, None)
    t_on,  dw_on  = parse_log(on.get(key)) if on.get(key) else (None, None)
    if t_off is None or t_on is None:
        print(f"{idx:>3} {ev:>6}   MISSING LOG (off={t_off} on={t_on})")
        continue
    d = t_on - t_off
    tot_off += t_off; tot_on += t_on
    n_better += d < 0; n_worse += d > 0; n_same += d == 0
    dw_off_tot += len(dw_off); dw_on_tot += len(dw_on)
    note = ''
    if dw_off or dw_on:
        offs = {b for b,_,_ in dw_off}; ons = {b for b,_,_ in dw_on}
        fixed = offs - ons; new = ons - offs; still = offs & ons
        parts = []
        if fixed: parts.append('FIXED:' + ','.join(sorted(fixed)))
        if new:   parts.append('NEW:'   + ','.join(sorted(new)))
        if still: parts.append('still:' + ','.join(sorted(still)))
        note = ' ; '.join(parts)
    print(f"{idx:>3} {ev:>6} {t_off:>9} {t_on:>8} {d:>+6}  {note}")

print(f"\nTOTAL diff-branches vs prototype: off={tot_off} on={tot_on} (delta {tot_on-tot_off:+d})")
print(f"events improved={n_better} worsened={n_worse} unchanged={n_same}")
print(f"flag_dir_weak branches differing vs prototype: off={dw_off_tot} on={dw_on_tot}")
