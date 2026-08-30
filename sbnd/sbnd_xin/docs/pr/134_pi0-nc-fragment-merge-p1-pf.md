# doc pr/134 — the NC merged-complex back-projection (116962) + the P1 PF satellite absorb

**Status: ROUND CLOSED 2026-08-30 (awaiting owner Bee verdict).**  K22
`pi0_nc_frag_merge` + K23 `pi0_pf_assoc_deg`, both DEFAULT OFF.  116962
fixed: the merged-complex back-projection lands the NC vertex 44.3 cm
upstream and the event reduces to exactly two showers (owner: "clearly a
two shower event").  76346 preserved byte-for-byte; fires exactly
{76346, 116962} in 239 events; OFF gate PASS 478/478; census flat 32
exact; movers ADVERSE 0.  K23 measured ledger NET -1 (one heal, two
overfills) -> recommended OFF.  Recommendation: flip K22 with the NC
chain (sec 6).  Bee: OFF `84e89e06` / ON `94f7e232` (sec 7).

**Owner brief (2026-08-30, on the v2.2 Bee)**: "The scan is quite good.
Note, the evt 116962 is still not good, the nu vertex is not at the right
place. Can you work according to your recommendation for the next round?"
The pr/133 §12.4 recommendation ranked (1) the P1 PF-association port and
(2) the NC left-complex fragment merge + re-back-projection; the owner's
116962 verdict makes (2) the lead front.  Both are this round.

## Repro block

```
# build + tests
direnv exec /home/xqian/toolkit-dev wcbuild
direnv exec /home/xqian/toolkit-dev ./build/clus/wcdoctest-clus     # 2593 assertions

# smoke (specimens, probes on)
WCT_PI0_PAIR_DEBUG=1 PR_JOBS=6 PR_EXTRA_STAGES=pr_display \
SBND_PI0_BP_VERTEX=8 SBND_PI0_NC_SIG_ANGLE=15 SBND_PI0_NC_FLOOR=5 SBND_PI0_NC_PF_ASSOC=20 \
SBND_PI0_NC_FRAG_MERGE=1 SBND_PI0_PF_ASSOC=20 \
./run_pr_chain_batch.sh work-nuecc48-grp0825 work-pr134-dbg1-nuecc48 data 116962 342199

# full arms
PR_JOBS=12 bash scripts/pr134_arms.sh 98 off6 0 && PR_JOBS=12 bash scripts/pr134_arms.sh 141 off6 0
PR_JOBS=12 bash scripts/pr134_arms.sh 98 k2223 0 SBND_PI0_BP_VERTEX=8 SBND_PI0_NC_SIG_ANGLE=15 \
    SBND_PI0_NC_FLOOR=5 SBND_PI0_NC_PF_ASSOC=20 SBND_PI0_NC_FRAG_MERGE=1 SBND_PI0_PF_ASSOC=20 \
    && PR_JOBS=12 bash scripts/pr134_arms.sh 141 k2223 0 SBND_PI0_BP_VERTEX=8 ...

# gates / census / ledger / movers
python3 scripts/pr85_hash_gate.py work-pr133-flipchk-<s> work-pr134-off6-<s>   # x4, echo rc=$?
bash scripts/pr134_manifests.sh k2223
python3 scripts/pr132_pi0_census.py --manifest98 em_display/em117-134k222398-manifest.tsv \
    --manifest141 em_display/em114c-134k2223141-manifest.tsv --fudge 0.84 \
    --overlay-tag pi0scan-0829-agent --tsv docs/pr/pr134-census-k2223.tsv
python3 scripts/pr132_gamma_ledger.py  (same manifests) --tsv docs/pr/pr134-gamma-ledger-k2223.tsv
python3 scripts/pr90_movers.py work-pr133-flipchk-<s> work-pr134-k2223-<s> --tags vtx105  # x4
```

## 1. The 116962 diagnosis — why fragment-level rays cannot work

Production baseline (`work-pr133-flipchk-nuecc48`): the owner's "clearly a
two shower event" is reconstructed as SIX shower fragments seated at the
wrong main vertex (-69.7,-79.1,155.9): right gamma = 21072 (232.2) + 21070
(111.6) + 21073 (78.3 MeV, all conn-1 co-started at the seat); left gamma =
22025 (112.4) + 58035 (118.8) + 67056 (54.9), plus satellite 55030 (31.0).

Two measured facts kill every fragment-level approach (K21 v2.2 tape,
`work-pr133-dbg8-nuecc48`):

1. **The false pair is in-window at fragment charges**: 21072 x 55030,
   miss 1.5, m=125.8 -> accepted, vertex moved only 12 cm (the v2.2 state
   the owner rejected).  With the host complex MERGED (E=422.1) the same
   crossing is m=166.1 — out of window.  *Merging kills the false fire by
   honest charge accounting.*
2. **The true crossing is out-of-window at fragment charges**: 21072 x
   67056 crosses at miss 4.9 but m=33.1 (232 x 55 MeV) -> rejected.  With
   merged charges the same geometry gives m ~ 146 — in window.  *Merging
   admits the true fire through the SAME legacy (100,160) window.*

A third fact fixes the ray recipe: each fragment's own 15 cm local dir is a
fragment artifact (58035's own axis points at the seat, 8.6 deg from the
right trunk).  The left gamma's true direction only emerges from the MERGED
non-host cloud: the local dir at 58035's start over {22025,58035,67056}
points into the left complex, and its crossing with the host back-ray lands
at (-70.1,-79.7,111.7) — 44.3 cm upstream, miss 1.5 cm, both conversion
legs behind (u=-44.3,-7.3).

## 2. K22 `pi0_nc_frag_merge` (bool, DEFAULT OFF) — complex-level pairing

Active only inside the K21 NC-signature gate (`pi0_nc_sig_angle_deg` > 0).
Replaces the fragment-level v2 pair loop with:

- **host complex** = the `nc_sig_hosts` (every main-vertex prong's shower),
  E_host = their summed kine_charge; host head = highest-E member;
- **partner rays** = each non-host candidate's start/end anchor with the
  local dir computed over the merged NON-HOST cloud (the multi-shower
  analogue of `shower_cal_dir_3vector`, unweighted fit-point centroid in
  15 cm);
- crossing math, u-range, miss cap, 1-100 cm shift, opening angle: verbatim
  v4; ranking miss-first among in-window combos: verbatim v2.1;
- **merged mass**: E_host x E_partner, where the partner anchor fragment
  always belongs to the partner (its charge is the conversion stem) and
  every other non-host candidate joins the side whose vertex ray its cloud
  centroid sits closer to.  The window stays the legacy (100,160)+offset —
  no new mass parameter;
- **on accept**: host members fold into the host head, assigned partner
  members into the partner anchor (K12/K18 splice precedent), charges
  refresh via `cal_kine_charge`, the registered pio mass refreshes at the
  accept-time opening angle; then the existing v2.2 PF block mops the
  < 35 MeV satellites.

Offline final-sim (this doc's derivation, run on flipchk dumps before any
C++ was written) predicted: 116962 FIRE at (-70.1,-79.7,111.7) m=146.4;
76346 FIRE at its unchanged v2.2 vertex m=134.0; 21073 / 282979 / 175896 /
176502 / 275011 / 499423 all NO FIRE (282979's best crossing m=38.2,
175896's m=31.2, both window-rejected).

**Smoke (work-pr134-dbg1/2)**: in-code reproduces the sim exactly —
116962: `bp accept sh1=21072 sh2=58035 m=146.4 shift=44.3`, absorbs
21070+21073 -> 21072 and 22025+67056 -> 58035, PF adds 55030; final state
**two showers** 21072=468.7 + 58035=294.3 MeV, main vertex
(-70.1,-79.7,111.7).  76346: byte-same accept (14059 x 49048 m=134.0 shift
59.8), 14058 absorbed by the merge (was PF), final 243.8/83.3 identical to
v2.2.  21073: merge path evaluated (hosts=1), no in-window crossing, vertex
byte-identical to baseline.

## 3. K23 `pi0_pf_assoc_deg` (deg, DEFAULT OFF 0) — the P1 satellite absorb

The with-vertex-finder port of the owner's v2.2 PF directive ("The other
low-energy gammas can be associated with the pi0 gammas").  After the P1
selection loop completes, each accepted pair gamma absorbs its DETACHED
(conn-type != 1) satellites: E < 35 MeV, start within (0.5,120] cm of the
gamma start and inside the knob cone of the gamma's 15 cm axis (the
pr132 gamma-ledger sibling criterion), assigned to the closer-in-angle
gamma; long-muon members and pair members excluded.  Charges refresh and
the registered pair mass rescales by the charge ratio (angle unchanged).

Design note: the first build guarded against absorbing any shower with a
surviving recorded pairing — that guard is structurally wrong (the
selection loop runs until no acceptable pair remains, so every surviving
pairing is one the selector REJECTED; on 105946 all five sub-35 satellites
own rejected pairings and the guard made K23 a no-op).  Removed.

**Specimen findings (dbg1)**:
- 105946 (P1 accept 53030 x 55063 m=107.3): the K23 heal target — g1
  55063=477.9 vs label 604.3 with ~46 MeV in sub-35 satellites.
- 342199: NO P1 accept exists (all pio=-1) — its two label gammas' pair is
  window-rejected at fragment charges (56.7 x 64.2; healed energies 80.3 x
  121.8 would give m ~ 131).  K23 cannot act post-accept; this is a
  merge-BEFORE-accept case (K12 `pi0_collinear_merge_deg` domain) — queued,
  not this round's knob.
- 52044: P1 accepted 18004 x 58029 (m=116.1) but the label g2 is 24035
  (38.8 MeV >= the 35 cap, and a wrong-partner case, not a satellite) —
  K23 correctly leaves it alone.

## 4. Full-arm round 1 (k2223 v1) — two measured defects, both fixed

Arms `work-pr134-{off6,k2223}-*` (binary v1, 239 events x 2).  OFF gate
off6-vs-flipchk: **PASS 478/478** (132+212+38+96, all four rc=0).  Census
flat at 32 exact.  But the arm caught two defects the specimens could not:

1. **ADVERSE 499423 (the EM-host hole).**  A 211-typed attached prong
   passed the v4 gate as a "shower" host (the offline sim's pdg==11
   candidate filter hid this case — in-code `nc_sig_hosts` has no pdg
   test), and the MERGED partner charges lifted its false pair into the
   window: vertex moved 4.7 cm off the hand click (`pr90_movers` mcp2k:
   ADVERSE 1).  Fix: the EM-host guard — every host-complex member must be
   EM-typed (pdg +-11, or +-13 through the K20 admit); a non-EM host falls
   back to the fragment-level loop, i.e. exactly the k21v5 behavior the
   owner accepted (499423: no fire, vertex byte-identical).  dbg4:
   `bp mrg refuse host=11045 pdg=211`.
2. **The K23 cal_kine_charge refresh clobbered stored charges.**  Ledger
   k21v5 -> k2223 v1: OK 120 -> 113, OVER 4 -> 11.  Diffing the rows: the
   damage came from the post-absorb `cal_kine_charge` RE-DERIVATION, not
   from absorbed satellite charge — 47212 g1 dropped 221.9 -> 130.1 with
   no absorb into it, 314838 g2 inflated +107 MeV against ~10 MeV of
   satellites.  The stored charges are kine_best-derived; re-deriving
   loses that.  Fix: charges update arithmetically (gamma + its absorbed
   satellites), mirroring the ledger's own heal criterion; no refresh.
   (The K22/NC-path refresh stays — it is the owner-accepted v2.2 recipe
   on bp fires, verified on 76346: 243.8 vs label 246.7.)

**K23 cone sweep (offline, arithmetic charges, off6 dumps, cones
20/15/10/7/5)**: exactly two ledger movers exist at 20 deg — the 105946 g1
heal (+11.6 MeV from 39013 at 17.9 deg -> ratio 0.810, OK) and a marginal
47212 g2 overfill (65.1 -> 82.2, ratio 1.263 vs the 1.25 line; its 13.4 MeV
absorb sits at 4.5 deg, unexcludable by any cone).  No cone value nets
positive: K23 at 20 deg is LEDGER NET-ZERO (+1/-1).  It ships DEFAULT OFF
as PF-cleanliness machinery (the with-vertex mirror of the owner's v2.2
directive); the flip is the owner's call on the Bee evidence (105946 and
47212 packaged).

## 5. Final arms (k2223f, binary v3) — the round's numbers

Arms `work-pr134-{off7,k2223f}-*` (both manifests, 239 events x 2, all
rc=0; the earlier v1/v2 arms `{off6,k2223}` are the defect-discovery record
of sec 4).

- **OFF gate**: off7 vs work-pr133-flipchk, `pr85_hash_gate.py`:
  **PASS 478/478** (132+212+38+96; rc=0 x4).  Knobs-off byte-identity of
  the final binary.
- **Fires**: main vertex moves in EXACTLY {76346 (59.8 cm), 116962
  (44.3 cm)} of 239 — 499423 stays put under the EM-host guard.
- **Movers** (`pr90_movers.py --tags vtx105`, x4): 1 mover total, 76346
  *toward* the click (30.27 -> 29.54 cm).  **ADVERSE 0.**
- **Census**: 32 exact / 15 partial / 1 none / 18 no-group — byte-flat vs
  k21v5 production.  nueCC-fake counter 0.
- **Gamma ledger**: OK 119/132 (90.2%) vs 120 (90.9%) at k21v5.
  Decomposition: K22 is ledger-neutral (its two fires carry no label
  gammas' charges beyond the v2.2-identical 76346); K23 nets **-1**:
  +1 heal (105946 g1 UNDER+SIBS -> OK, 477.9 -> 489.6 of label 604.3) vs
  -2 overfills (47212 g2 65.1 -> 82.3 = ratio 1.263 marginally past the
  1.25 line; 506746 g2 74.6 -> 113.0 = 1.44 — an absorb chain the off6
  offline sweep did not predict).  ~20 other gammas drift OK->OK
  (satellites absorbed, several better-centered: 57709 g1 0.88 -> 0.94).

### 5.1 K22-only insurance batch

K23 absorbs run in P1, BEFORE the bp proposer — in principle an absorbed
satellite could have fed or starved a K22 crossing.  All 20 K23-active
events (the ledger charge-drift rows) plus 76346/116962/499423 re-ran with
`SBND_PI0_NC_FRAG_MERGE=1` and K23 OFF (`work-pr134-k22only-*`): main
vertex moves in exactly {76346 (59.8), 116962 (44.3)} — no new fires, no
lost fires.  The K22-only production point behaves identically to the
measured k2223f arm on every K22-observable.

## 6. Recommendation

- **K22 `pi0_nc_frag_merge` -> FLIP CANDIDATE** with the standing NC chain
  (`pi0_bp_vertex_miss_cm=8, pi0_nc_sig_angle_deg=15, pi0_nc_floor_mev=5,
  pi0_nc_pf_assoc_deg=20, pi0_nc_frag_merge=true`): it is the owner's
  116962 fix (vertex 44.3 cm upstream, the event reduced to exactly two
  showers 468.7 + 294.3 MeV), preserves 76346 byte-for-byte, and has zero
  collateral on 239 events after the EM-host guard.
- **K23 `pi0_pf_assoc_deg` -> STAYS OFF** (measured net -1 ledger OK).
  The machinery is shipped and correct (arithmetic charges); the blind
  20-deg cone trades one heal for two overfills.  Owner call on the Bee
  evidence; a future targeted variant (e.g. absorb only when the pair
  mass sits below the window center, or a tighter satellite distance)
  can revisit.
- K22-only insurance (sec 5.1): the K23-active events re-run with K23 off
  — no new fires, so the K22-only flip point equals the measured k2223f
  behavior on every K22-observable.

## 7. Bee package (uploaded 2026-08-30)

- OFF (baseline, off7): https://www.phy.bnl.gov/twister/bee/set/84e89e06-3b39-4fb3-a3e7-3b765247cecf/event/list/
- ON  (k2223f):         https://www.phy.bnl.gov/twister/bee/set/94f7e232-b3a6-47bd-b873-48673c60913a/event/list/

7 events (annotated index `bee/pr134k22/pr134k22.index.txt`): 0=116962 the
K22 rescue, 1=76346 unchanged-proof, 2=105946 the K23 heal, 3=47212 +
4=506746 the K23 overfill adjudications, 5=21073 dead-on preserved,
6=499423 the EM-host-guard refusal.  Vertex plot:
`docs/pr/pr134-vtx-116962.png`.

## 8. Queued (unchanged from pr/133 sec 12, minus this round)

1. 342199-class merge-BEFORE-accept in P1 (K12 `pi0_collinear_merge_deg`
   domain: the label pair is window-rejected at fragment charges 56.7 x
   64.2 but healed energies give m ~ 131).  2. 52044-class wrong-partner
   (label g2 = 24035, 38.8 MeV, loses to 58029).  3. Stub-prong NC
   signature variant (180801/259542).  4. Mass peak refit at the new
   production point.  5. PARK: outside-window (upstream).
