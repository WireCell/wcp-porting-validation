# doc pr/106 lockbox read -- 2026-08-21T19:26:24Z -- ONE read, thetas pre-declared on the tuning set

events 349 (carried 293, frozen 56) skipped {'empty-cloud': 1, 'no-harvest-dump': 1}; harv-base/topo3 snapped-row mismatch 0
live-pick resolution onto the pre-DL cloud: {'accept-id': 1676, 'final-id': 1018, 'pos<=1.5': 62, 'pos<=3-unambiguous': 29, 'unresolved': 11}

d_target (click -> nearest pre-DL candidate), per sample [<=1 | 1-3 | 3-10 | >10]:
  nuecc48    n=  15    14     1     0     0
  ncpi0      n=   7     6     1     0     0
  mcp1k      n= 133   110    20     3     0
  mcp2k      n= 194   156    31     6     1
  human      n= 254   212    33     8     1
  ai-scanner n=  95    74    20     1     0
  ALL        n= 349   286    53     9     1

primary universe (d_target <= 3): 339 ; secondary band: 9 ; candidate-missing: 1
admission (target among snapped rows): top5 231/339, top10 260/339 ; reject-outcome coverage 336/339

closure (offline decision == live route/winner), all label events:
  harv-base   mismatches 0
  base        mismatches 0
  harv-topo3  mismatches 0
  topo3       mismatches 0
  ma4         mismatches 0
  topk10      mismatches 0
  dlonly      mismatches 4  e.g. [(53881, 'dl-legacy-reject', None, ('accept', 12002)), (105946, 'dl-legacy-reject', None, ('accept', 14002)), (168869, 'dl-legacy-accept', 16007, ('accept', 16006)), (399118, 'dl-legacy-reject', None, ('accept', 16014))]

### target-hit table, primary universe (n=339)

| method | ALL | nueCC48 | NCpi0 | mcp1k | mcp2k | IPW | human | AI |
|---|---|---|---|---|---|---|---|---|
| M1 DL alone (offline, legacy rule) | 195/339 (57.5%) | 12/15 | 4/7 | 70/130 | 109/187 | 57.1% | 122/245 | 73/94 |
| M1 DL alone (live dlonly arm) | 201/339 (59.3%) | 12/15 | 4/7 | 73/130 | 112/187 | 58.6% | 127/245 | 74/94 |
| M2 re-rank + topo (w=3) | 254/339 (74.9%) | 12/15 | 5/7 | 100/130 | 137/187 | 76.3% | 171/245 | 83/94 |
| M3 re-rank, no topo = PRODUCTION | 259/339 (76.4%) | 12/15 | 5/7 | 101/130 | 141/187 | 77.6% | 174/245 | 85/94 |
| production (live base arm) | 259/339 (76.4%) | 12/15 | 5/7 | 101/130 | 141/187 | 77.6% | 174/245 | 85/94 |
| min_accept 4 (live ma4) | 229/339 (67.6%) | 12/15 | 5/7 | 90/130 | 122/187 | 66.1% | 151/245 | 78/94 |
| top_k 10 (live topk10) | 259/339 (76.4%) | 12/15 | 5/7 | 101/130 | 141/187 | 77.6% | 174/245 | 85/94 |
| no DL, traditional (live trad) | 244/339 (72.0%) | 9/15 | 3/7 | 97/130 | 135/187 | 74.1% | 165/245 | 79/94 |

M3 classes: {'hit': 259, 'reject-to-wrong': 50, 'wrong-accept': 10, 'not-admitted': 17, 'veto-to-wrong': 3}

### target-hit table, secondary band 3-10 cm (n=9)

| method | ALL | nueCC48 | NCpi0 | mcp1k | mcp2k | IPW | human | AI |
|---|---|---|---|---|---|---|---|---|
| M1 DL alone (offline, legacy rule) | 2/9 (22.2%) | 0/0 | 0/0 | 1/3 | 1/6 | 21.2% | 2/8 | 0/1 |
| M1 DL alone (live dlonly arm) | 3/9 (33.3%) | 0/0 | 0/0 | 2/3 | 1/6 | 31.8% | 3/8 | 0/1 |
| M2 re-rank + topo (w=3) | 4/9 (44.4%) | 0/0 | 0/0 | 2/3 | 2/6 | 47.0% | 4/8 | 0/1 |
| M3 re-rank, no topo = PRODUCTION | 3/9 (33.3%) | 0/0 | 0/0 | 2/3 | 1/6 | 31.8% | 3/8 | 0/1 |
| production (live base arm) | 3/9 (33.3%) | 0/0 | 0/0 | 2/3 | 1/6 | 31.8% | 3/8 | 0/1 |
| min_accept 4 (live ma4) | 5/9 (55.6%) | 0/0 | 0/0 | 2/3 | 3/6 | 57.6% | 5/8 | 0/1 |
| top_k 10 (live topk10) | 3/9 (33.3%) | 0/0 | 0/0 | 2/3 | 1/6 | 31.8% | 3/8 | 0/1 |
| no DL, traditional (live trad) | 2/9 (22.2%) | 0/0 | 0/0 | 1/3 | 1/6 | 21.2% | 2/8 | 0/1 |

M3 classes: {'hit': 3, 'not-admitted': 1, 'reject-to-wrong': 1, 'uncovered': 2, 'wrong-accept': 2}
events tsv: docs/pr/106_events-lockbox.tsv

## C1 best search theta (tuning 521/673)
eval {'w_snap': 3.0, 'w_fwd_z': 1.0, 'w_clen': 1.0, 'w_isol': 3.0, 'w_main': 1.0, 'w_fv': 0.5, 'w_topo': 5.0, 'center': 0.0, 'min_accept': 12.0, 'scale': 500.0}
| method | ALL | nueCC48 | NCpi0 | mcp1k | mcp2k | IPW | human | AI |
|---|---|---|---|---|---|---|---|---|
| theta | 267/339 (78.8%) | 12/15 | 5/7 | 103/130 | 147/187 | 81.1% | 181/245 | 86/94 |
classes {'hit': 267, 'reject-to-wrong': 57, 'not-admitted': 10, 'wrong-accept': 5}

## C2 scale=500 (tuning 514)
eval {'w_snap': 1.0, 'w_fwd_z': 1.0, 'w_clen': 1.0, 'w_isol': 1.0, 'w_main': 1.0, 'w_fv': 1.0, 'w_topo': 0.0, 'center': 0.0, 'min_accept': 10.0, 'scale': 500.0}
| method | ALL | nueCC48 | NCpi0 | mcp1k | mcp2k | IPW | human | AI |
|---|---|---|---|---|---|---|---|---|
| theta | 267/339 (78.8%) | 12/15 | 5/7 | 103/130 | 147/187 | 81.1% | 181/245 | 86/94 |
classes {'hit': 267, 'reject-to-wrong': 57, 'not-admitted': 10, 'veto-to-wrong': 1, 'wrong-accept': 4}

## C3 min_accept=15 (tuning 513)
eval {'w_snap': 1.0, 'w_fwd_z': 1.0, 'w_clen': 1.0, 'w_isol': 1.0, 'w_main': 1.0, 'w_fv': 1.0, 'w_topo': 0.0, 'center': 0.0, 'min_accept': 15.0, 'scale': 1000.0}
| method | ALL | nueCC48 | NCpi0 | mcp1k | mcp2k | IPW | human | AI |
|---|---|---|---|---|---|---|---|---|
| theta | 266/339 (78.5%) | 12/15 | 5/7 | 103/130 | 146/187 | 80.8% | 180/245 | 86/94 |
classes {'hit': 266, 'reject-to-wrong': 57, 'not-admitted': 11, 'veto-to-wrong': 1, 'wrong-accept': 4}
