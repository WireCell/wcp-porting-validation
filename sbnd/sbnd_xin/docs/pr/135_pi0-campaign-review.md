# doc pr/135 — the π⁰ clustering campaign review: where we are, what is left, how to wrap up

**Status: review written 2026-08-30 at the owner's request ("we have made a
lot of progress, and should wrap up at some points. Can you review the
progress and see what are the major improvements that we should look
into?").  Covers docs pr/126 → 132 → 133 → 134 (round 2 of pr/134 in
flight when this was written; its numbers are folded into §1/§2 once the
arms land).  This doc changes no code and runs no arms.**

## Repro (provenance of every number)

```
# All numbers below are quoted from the closed round docs; each cites its arm labels there.
# docs/pr/126_pi0-audit-and-em-charge-scale.md   (audit + charge scale, 50 hand pi0)
# docs/pr/132_pi0-reco-round1.md                 (rounds 1-10, 66 hand pi0, gamma ledger)
# docs/pr/133_pi0-muon-showers-nc-signature.md   (K20/K21, NC signature, sec 12 review)
# docs/pr/134_pi0-nc-fragment-merge-p1-pf.md     (K22/K23/K24, NC merged-complex bp)
# Metric machinery: scripts/pr132_pi0_census.py, pr132_gamma_ledger.py, pr90_movers.py
# Label sets: em_labels/{emscan-0827,emscan-0828-agent5,pi0scan-0829-agent} (66 hand pi0, 132 gammas)
```

## 1. The trajectory

| point | exact | partial | none | no-group | nueCC fakes | γ ledger OK | doc |
|---|---|---|---|---|---|---|---|
| pr/126 baseline (0.80 scale, 50 hand π⁰) | 26 | 10 | 2 | 12 | 2 | — | 126 |
| pr/132 rounds 1–10 (0.84 + K7/K8 + K3=28, 66 hand π⁰) | 31 | 16 | 1 | 18 | 2 | 120/132 = 90.9% | 132 |
| pr/133 flips (K3=29 + K20 + K16@120) — **current production** | **32** | 15 | 1 | 18 | **1**¹ | 90.9% | 133 |
| pr/134 r1 measured point (+ NC chain + K22, OFF) | 32 | 15 | 1 | 18 | 1 | 119/132² | 134 |
| pr/134 r2 measured point (+ K24 v8, OFF, §7) | **33** | 15 | 1 | 17 | 0³ | 90.9% | 134 |

¹ the survivor is the 76346 misflag, which the NC chain itself clears.
² −1 is the accepted 116962 re-partition, not a regression.
³ the accepted-group fake counter at the k24b point.

Exact-match rate 26/50 (52%) → 32/66 (48.5%) on a denominator that grew by
16 harder hand π⁰; on the original 50 the campaign is well above 60%.
Sharing-a-gamma (exact+partial) = 71%. Movers ADVERSE = 0 at every shipped
point. Every flip is owner-adjudicated on a Bee A/B; every knob shipped
DEFAULT OFF behind a 478/478 byte-identical gate.

**Production ON today:** `kine_shower_fudge_factor=0.84`, K7/K8 (211-typed
readmission), K3=29 (attached-partner floor), K16@120 (EM collinear merge),
K20 (μ-typed shower admission), plus the sccc/K5 sentinels.

**Built, measured, DEFAULT OFF, pending owner:** the NC chain
(`pi0_bp_vertex=8, nc_sig_angle=15, nc_floor=5, nc_pf_assoc=20`) + K22 v2
(`pi0_nc_frag_merge`) + K24 (`pi0_prefer_main_vertex`) — the pr/134 round-2
operating point; K23 (`pi0_pf_assoc_deg`) measured net −1 → recommended
stays OFF.

## 2. What the campaign fixed (by mechanism, one line each)

- **Charge scale**: mass peak aligned to 135 MeV (PEAK fit, not median;
  fudge 0.80→0.84, pr/126 §4g).
- **Typing holes**: 211-typed (K7/K8) and 13-typed (K20) γs readmitted to
  the π⁰ pools with EM restamp on accept — closed the "invisible gamma"
  class (166870, 348691).
- **nueCC fake suppression**: attached-partner floor K3=29 kills the
  primary-e + tiny-γ fake with a measured closed interval (28.7, 29.6).
- **NC π⁰ vertex**: the owner's NC signature (K21) + end-agnostic
  back-projection + merged-complex pairing (K22 v2) moves 116962's vertex
  83.5 cm upstream with the owner's exact grouping (58035→21072), 76346 to
  1.3 cm — where K19's naive re-vertexer cost 7 ADVERSE for the same two
  rescues.
- **ν-vertex preference** (K24, measured §7): direction ambiguity resolves
  to the main vertex (owner rule) — 105946 off the proton end, 47212's
  both π⁰s off the pion stub.
- **Reliability**: every fire class is enumerated per arm (fires, census,
  ledger, movers); the deficit-biased internal shower direction is
  measured and documented (50–75° off flight line — the missing charge IS
  the downstream tail), which is why conversion-displacement rays replaced
  local dirs in every new pairing site.

## 3. Measured DEAD — do not re-run (the negative results are results)

Geometry merges (K12 cone, K16-v1@60, K17 back-tube); acceptance-aware
merge K18 (both variants); the P2 knob family for NC (K14/K4/K5/K13 —
ordering lock, P1 consumes the γs first); ALL ranking policies except
K24's main-first (208/208 calibrated tape sim, pr/133 §5.1); centroid rays
for mass (pr/132 round 10); mass-window/offset tuning (K1 closed at 10,
K11); ten admission-time features (pr/130); K19 naive re-vertexing; K23
satellite cones (no cone value nets positive); the wrong-vertex ranking
front (owner-scoped out).  Each has its mechanism written down in the
round docs — a future "why not just…" starts there.

## 4. The residual misses, by measured cause (34 events, 13 ledger γs)

| class | n | live lever? |
|---|---|---|
| label mass outside (100,160) — angle compression + charge deficit | 14 | NO finder-level lever (3 attempts dead); upstream imaging/shower-building |
| wrong-partner / fragmentation in P1 | 16 | partially — 342199-class merge-before-accept + 52044 wrong-partner remain the only unmeasured P1 structural items |
| mistyped γ, window-blocked even if readmitted | 2 | dead until charges move (same upstream deficit) |
| label γ absent / era artifact | 2 | no |

Ledger: UNDER+SIBS 3 (mergeable siblings exist), UNDER-nosibs 3 (charge
simply missing), OVER 4 (3 = the accepted K16 trade), ABSENT 2.

The pattern: **more than half of everything left traces to one upstream
root — the reconstructed shower is missing its downstream tail charge**,
which compresses opening angles (mass below window), bends internal
directions (wrong vertex seats), and starves fragment charges (window
rejections at true pairings).  The finder-level campaign has extracted
what geometry and ranking can give.

## 5. The major improvements to look into (ranked — this is the guide)

1. **Close the open owner decisions and freeze the production point
   (wrap-up item, no new code).**  (a) NC chain + K22 v2 + K24 flip on the
   round-2 Bee; (b) the 397630 adjudication — the one measured trade of
   the main-vertex preference (its TRUE pair sits 77.3 cm from main behind
   a 78.7 cm pion, structurally identical to 105946's fake seat; no local
   discriminator exists, so it is a policy call, or a track-length bound
   at overfit risk); (c) K23 stays OFF (net −1, machinery kept).
2. **π⁰ mass peak refit at the final production point (cheap, analysis
   only, do before freezing).**  The 0.84 fudge was fit at the pr/126
   point; K16@120 + K20 + K3=29 + the NC chain slightly changed the paired
   population (3 OVER γs).  One refit confirms or trims the fudge and
   feeds every mass window.  This is the natural "wrap-up seal".
3. **The upstream charge/direction deficit (the big one — a NEW campaign,
   not a π⁰ round).**  §4's dominant root cause.  Owner has so far scoped
   upstream imaging out; if the π⁰ numbers are to move materially beyond
   ~32/66, this is where the charge is: shower tail recovery / start
   re-derivation / the pr/130 q_miss mechanism (17/17 scan-confirmed,
   parked).  Recommend: treat as its own campaign with the γ ledger as the
   metric, decoupled from the finder knobs.
4. **The last two P1 structural items (one bounded round, +1–2 exact
   ceiling).**  342199-class merge-BEFORE-accept (label pair
   window-rejected at fragment charges 56.7×64.2, healed m≈131 — K12
   domain revisited with the K22-v2 merged-cloud machinery that did not
   exist when K12 died) and 52044 wrong-partner.  Worth exactly one round;
   stop on the first dead measurement.
5. **Non-π⁰ NC vertex discriminator (owner-level design, park unless
   wanted).**  Confirmed by eye: π⁰ observables are FLAT across the vertex
   ambiguity; separating a real secondary-interaction vertex from a
   track-end seat needs vertex dQ/dx, stub topology, or imaging — and any
   future NC gate must veto track activity (the 406125 lesson).  Also
   covers the stub-prong variant (180801/259542, 2 specimens).
6. **PARK explicitly: the outside-window 14.**  Not a finder problem;
   folds into item 3.

## 6. Wrap-up recommendation

Do items 1–2 (decisions + peak refit), write the production point into the
config with flip-equivalence gates, and declare the finder-level campaign
CLOSED — the measured-dead ledger (§3) is the fence that keeps it closed.
Item 4 is optional polish if one more round is wanted.  Items 3 and 5 are
new campaigns with different metrics and should be chartered separately,
not appended to this one.

## 7. The pr/134 round-2 close-out (lands after this review was drafted)

The measured point for §5 item 1(a): OFF gate PASS 478/478 (off9 vs
flipchk); fires exactly {76346, 116962}; census 32/15/1/18 -> **33/15/1/17
(exact 50.0%)** — movers +105946 exact (owner pair at the nu vertex),
+166870 exact, +71872 partial, −397630 (the main-preference trade, the
§5-1(b) adjudication); ledger 120/132 = 90.9% flat; vertex movers ADVERSE
0; pi0 group spread +3 (84 -> 87, was +19 before the K24 v8 admission
rules).  Bee A/B: OFF `473bf8e6-3d34-407a-9fc9-017462c4c2ee` / ON
`e0908476-322d-49c9-ba3f-c68626b45a1e` (12 events).  Details: doc pr/134
secs 11.1-13.  With the owner's flip verdict on this Bee, §5 item 1 closes
and item 2 (the peak refit) becomes the last act of the wrap-up.
