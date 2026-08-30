# doc pr/133 — pi0 round 2: mu-typed gamma readmission (K20) + the owner NC signature (K21) + the K16@120 Bee call

**Status: iterations 1-3 CLOSED (2026-08-30, autonomous per owner nap order).
K20 (mu-typed gamma readmission) + K21 (owner NC signature) built, gated
(478/478), measured: K20 clears the last admission blocker (166870 true pair
m=109.1 accepted, zero collateral); K21 keeps both NC rescues with ZERO of
K19's 7 ADVERSE.  Ranking front measured DEAD by tape simulation (208/208
calibration; mainfirst trades 166870 for 397630 whose true pair lives at a
non-main vertex).  K3 28->29 measured: removes the last real nueCC-fake
accept (176502) with ZERO other change in 239 events (fake counter 2->1,
census/ledger/movers all flat) -- the closed interval is (28.7, 29.6), and
30 would veto 76346's true NC pair.  K16@120 Bee package delivered (sec 1).
Recommendations for owner: sec 7.**

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

## 4. Iteration 1 — measurements (K20 and K21, separate arms)

Gates first: `work-pr133-off` vs `work-pr132-r10off` **PASS 4/4 (132+212+38+96
= 478 archives byte-identical)**; wcdoctest-clus 2585 PASS; compiled-config
proofs (keys present in the dbg1 ON configs, absent in the off configs).
Toolkit commit `8432b7af` (pushed).

### 4.1 K20 admission needed a v2 — the flag was self-defeating

dbg2 tape: both target showers refuse `why=notflag` — `get_flag_shower()` is
FALSE for the mistyped class (that is exactly why the typer called them 13).
v2 admission = not-in-long-muon AND (flag OR the file's own shower-ish-muon
idiom: `len < 40 cm && seg_dir_weak`).  dbg3: both admitted; 166870 gains an
accepted in-window pair m=109.1 with both members re-stamped EM.

### 4.2 Arm results vs the hand scan (98+141 manifests, 239 events each arm)

| arm | census (exact/partial/none/no-group) | ledger | movers (vtx105) |
|---|---|---|---|
| `off`  | 31 / 16 / 1 / 18 | 120/132 = 90.9% OK | — |
| `k20`  (`SBND_PI0_ADMIT_MU=1`) | 31 / 16 / 1 / 18 — one row moved: 166870 partial, blocker `g2:pdg=13` CLEARED | 90.9% | 0 x4 |
| `k21`  (`SBND_PI0_BP_VERTEX=8 SBND_PI0_NC_SIG_ANGLE=15`) | identical, 0 rows moved | 90.9% | 1: 76346 moved 4.29 cm (30.27→33.73 to click) — the r9 flat-mass ambiguity, no other mover |

K21 vs raw K19 (r9): the same two NC pair rescues fire (76346 true pair
m=129.6; 116962 m=125.8, and it now picks γ1=21072 232 MeV over the off-arm's
21070) — with ZERO of K19's 7 ADVERSE (v4 refuses 406125's p+mu vertex,
506746, 486687; smoke-verified).  The r9 +1 exact (165157) does not fire
under v4 — that event is not NC-signature.  76346's vertex still cannot be
fixed from pi0 observables (the flat-mass finding, confirmed third time).

### 4.3 Why the census headline cannot move from admission knobs anymore

The 35 non-exact rows: ~16 have the LABEL pair mass outside (100,160) — and
their RECO pair masses are far worse (54341: label 101, reco pair masses
22-25 — the charge/angle deficit is upstream, owner-scoped out).  The
54341/71872 "boundary" reading is an illusion: no offset knob reaches a reco
mass of 22.  8 empty-why partials = wrong-partner/fragmentation (K12/K16/K18
territory, measured).  K20 closed the LAST admission-class blocker (166870).

### 4.4 The one new live defect: wrong-partner-by-RANKING (166870)

With K20 the true pair (85045 x 87058, m=114.6, in-window) EXISTS in the
recorded pair list and LOSES the greedy ranking to (10013 x 87058, m=123.8):
the rank key is |m-125| minus a 6 MeV BONUS for ct2+ct2 pairs — fake scores
1.2, truth 4.4.  Iteration 2 = a full-tape probe arm (`k2021p` = K20+K21+
probes = also the combined production-candidate measurement) + an OFFLINE
ranking-policy simulation over all recorded pairs before spending any knob
(the round-10 tape-first method).

## 6. Iteration 3 — K3 28→29 measured (`k3v29`)

`work-pr133-k3v29` (98+141, 239 events rc=0, `SBND_PI0_ATTACH_MIN_MEV=29`,
compiled-config proof in the per-event config): census **0 rows moved**
(31/16/1/18 identical), ledger **90.9% flat**, movers **0 x4**, fake counter
**2 -> 1** — exactly the tape prediction: the only change in the entire
sample is 176502's fake group (attached 149.6 MeV + 28.7 MeV partner, no
hand label) disappearing.  76346's true NC pair (partner 29.6 MeV) survives.

## 7. Recommendations (owner decisions)

| item | recommendation | evidence |
|---|---|---|
| **K3 28→29** (production value change) | FLIP — my top recommendation | closed population: kills the last real fake, zero other change in 239 events; interval (28.7, 29.6); 30 would kill 76346's true pair |
| **K20 `pi0_admit_muon_showers`** | FLIP-clean by measurement | 166870 true pair accepted + restamped EM; zero collateral (census/ledger/movers flat elsewhere); the mistyped-mu class is 3/132 hand gammas |
| **K21 `pi0_nc_sig_angle_deg=15` (+bp 8)** | owner call | does exactly what the signature specifies: both NC pair rescues, none of K19's 7 ADVERSE; but the proposed vertex is eye-validated only — the census cannot see it, and 76346's click distance worsens 30.3→33.7 (flat-mass, third confirmation); also cleans the 76346 fake-counter misflag |
| **K16@120** | owner call (sec 1 Bee links) | +1 exact (54332) vs 3 gammas +28-51% spurious charge |

Residual census misses (35 events) are measured upstream (label-mass-outside-
window with reco masses far below label: charge/angle deficits) or locally
degenerate (ranking/fragmentation) — consistent with rounds 5-10; no further
in-scope finder knob is expected to move the headline.

Repro (iterations 2-3):
```
PR_JOBS=12 bash scripts/pr133_arms.sh 98 k2021p 1 SBND_PI0_ADMIT_MU=1 SBND_PI0_BP_VERTEX=8 SBND_PI0_NC_SIG_ANGLE=15  # + 141
python3 scripts/pr133_rank_sim.py --manifest98 em117-133k2021p98-manifest.tsv --manifest141 em114c-133k2021p141-manifest.tsv --overlay-tag pi0scan-0829-agent
PR_JOBS=12 bash scripts/pr133_arms.sh 98 k3v29 0 SBND_PI0_ATTACH_MIN_MEV=29  # + 141
python3 scripts/pr132_pi0_census.py --manifest98 em117-133k3v2998-manifest.tsv --manifest141 em114c-133k3v29141-manifest.tsv --fudge 0.84 --overlay-tag pi0scan-0829-agent --tsv docs/pr/pr133-census-k3v29.tsv
```
