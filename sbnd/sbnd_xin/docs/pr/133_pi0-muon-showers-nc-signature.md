# doc pr/133 — pi0 round 2: mu-typed gamma readmission (K20) + the owner NC signature (K21) + the K16@120 Bee call

**Status: OPEN (2026-08-30). Owner orders: "Can you proceed for the next
ronund 1. K20, for K16, please provide me a bee link to examine for 3. The
key for NC pi0 without vertex is that a) we are inside an EM shower b),
there is at least another major object looks like SM shower, and the
direction is not aligned in this case, we can go into the NCpi0 reco case.
We can sasume the 2 gamma (1 pi0) case. We do not want to touch the earlier
part (e.g. upstream imaging)." Then: "You should continue on this with
another 3 iterations follow your own judgement and recommendations. Note,
the key is the metric in comparing with the hand scan results."**

Follow-on to `docs/pr/132_pi0-reco-round1.md` (rounds 1-10 + the r9-verdict
typing census).  Baselines: census 31 exact / 16 partial / 1 none / 18
no-group (82 groups, nueCC-counter 2); gamma ledger 120/132 = 90.9% OK.
Production = fudge 0.84 + K7/K8 + K3=28.

Repro block (iteration 1):
```
# toolkit @ <hash below>; wcbuild; freshness proof; wcdoctest-clus 2585 PASS
# K16 Bee (existing r5off/r5cmw arms, no new run):
python3 scripts/bee/make_pr_bee.py -q work-mcp1k-grp0825 -q work-mcp2k-grp0825 \
  -p work-pr132-r5off-mcp1k -p work-pr132-r5off-mcp2k -o bee/pr133k16/pr133k16-off.zip 54332 54341 99838 165157 347824
# (same with r5cmw arms -> -on.zip); ./upload-to-bee.sh each
# smoke (12 specimens, probes+both knobs):
#   SBND_PI0_ADMIT_MU=1 SBND_PI0_BP_VERTEX=8 SBND_PI0_NC_SIG_ANGLE=15 WCT_PI0_PAIR_DEBUG=1
# OFF gate:
PR_JOBS=12 bash scripts/pr133_arms.sh 98 off 0 && PR_JOBS=12 bash scripts/pr133_arms.sh 141 off 0
for s in mcp1k mcp2k ncpi0 nuecc48; do python3 scripts/pr85_hash_gate.py work-pr132-r10off-$s work-pr133-off-$s; echo $s rc=$?; done
# ON arms + census/ledger/movers: see sections below
```

## 1. K16@120 — the Bee package for the owner call (round-5 item, delivered)

OFF (production point, `work-pr132-r5off`):
<https://www.phy.bnl.gov/twister/bee/set/dabd9eb5-47ea-49f3-8568-a28c8d0f1e1e/event/list/>
ON (K16@120, `work-pr132-r5cmw` = `SBND_EM_COLLINEAR_DEG=10 SBND_EM_COLLINEAR_DIS=120`):
<https://www.phy.bnl.gov/twister/bee/set/23579411-90ac-47aa-92c5-a9f6fd8a1b87/event/list/>

| bee_idx | event | status |
|---|---|---|
| 0 | 54332 | RESCUED: g1 charge 0.73→1.04 of label; census partial→exact (the +1) |
| 1 | 54341 | fix-direction: g1 0.68→0.82 of label |
| 2 | 99838 | ADVERSE: g1 OK→OVER 1.41 of label |
| 3 | 165157 | ADVERSE: g2 OK→OVER 1.51 of label |
| 4 | 347824 | ADVERSE: g2 OK→OVER 1.36 of label |

Index + urls: `bee/pr133k16/` (sidecars committed).  The call stays with the
owner: +1 exact vs 3 gammas gaining 28-51% spurious in-window charge.

## 2. K20 `pi0_admit_muon_showers` — DEFAULT OFF

A shower-topology object typed pdg ±13 is hard-excluded from every pi0 pool
with no readmit path (K7 covers only the A5/211 tag).  The r9-verdict typing
census: 3 of the 132 hand gammas are in this class (348691 sh 51080 79.8 MeV
= the owner's "missing" shower, 5.4 cm from the click; 166870 g2 38.6 MeV;
54341/71872 g2 typed 211 — those two are K7 territory).  Knob-on admission =
`get_flag_shower()` (topology/trajectory) AND start segment not in the long
muon; accepted members re-stamped EM (K7/K8 machinery).  Sites: both path-1
pools, both path-2 ray pools, the K19 bp candidate pool.
`id_pi0_with_vertex` now takes `segments_in_long_muon` (threading only).

## 3. K21 `pi0_nc_sig_angle_deg` — the owner NC signature — DEFAULT OFF

Gate v4 for the K19 back-projection proposer (needs `pi0_bp_vertex_miss_cm
> 0`): ALL main-vertex prongs inside showers (the 406125 lesson: a p+mu
track vertex passed v3) and none in the long muon; pairing restricted to
(vertex-hosting shower) × (other major EM object misaligned by > knob deg);
then the K19 mechanics unchanged (miss cap, back-projection t-range, 1-100
cm shift, (100,160)+offset window, best |delta|).  0 (default) = v3 gate =
r9 behavior byte-identical.

## 4. Iteration 1 — measurements

(filled below as arms land)
