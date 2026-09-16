#!/usr/bin/env python3
"""doc pr/149: the Stage-3 cell tie-break of 149_pred.txt sec 6 ("fewer Q2 adjudicated degradations").

The count per cell, Stage 1 + Stage 2, read from the committed compare printouts:
  event_label migrations + nu_evaluated flips + STM/FC bundle flips (TGM is 0 everywhere)
  + labelled ISO events whose vertex moved > 1 cm AWAY from the click (q1_verdict.py 'away')
  + pr127 sentinel FAILs on the Stage-1 arms
The definition was written after the Q1 table was read (the pre-registration names the rule,
not the count), so it is printed term by term.

Usage: tiebreak.py cs:s2cs csq2000:s2csq2000 ...
"""
import re, subprocess, sys
F = 'docs/pr/149_figs/'
def q2(path):
    t = open(path).read().split('## Q2')[1]
    lab = int(re.search(r'event_label migrations: (\d+)', t).group(1))
    nue = int(re.search(r'nu_evaluated flips: (\d+)', t).group(1))
    m = re.search(r'TGM (\d+)\s+STM (\d+)\s+FC (\d+)', t)
    return lab, nue, int(m.group(2)) + int(m.group(3))
V = open(F + '149_q1_verdict.txt').read()
for cell in sys.argv[1:]:
    c1, c2 = cell.split(':')
    s1 = q2(F + f'149_s1_compare_{c1}.txt') if c1 != 'cs' else q2(F + '149_s1_compare_cs.txt')
    s2 = q2(F + f'149_s2_compare_{c2}.txt')
    away = [int(m.group(1)) for m in re.finditer(rf'^{c1}\s+stage[12]: .*? away (\d+)', V, re.M)]
    sent = subprocess.run([sys.executable, 'scripts/pr127_sentinels.py', '--arms', f'work-*-pr149{c1}'],
                          capture_output=True, text=True).stdout
    nfail = int(re.search(r'(\d+) FAIL', sent.strip().splitlines()[-1]).group(1))
    total = sum(s1) + sum(s2) + sum(away) + nfail
    print(f'{c1:8s} stage1 label/nu_eval/tagflips {s1}  stage2 {s2}  ISO away s1+s2 {away}  sentinel FAIL {nfail}  => {total}')
