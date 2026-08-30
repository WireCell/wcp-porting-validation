# doc pr/132 — π⁰ reconstruction round 1: the EM scale flip to 0.84, five finder knobs, and the pairing pass

**Status: round 1 CLOSED (2026-08-29). Fudge 0.84 SBND PRODUCTION ON; K1-K5 DEFAULT OFF, measured; K3 recommended for the next flip.**
Follow-on to the pr/126 audit; implements its owner-decided items. Owner brief,
verbatim: *"1. adjust the EM charge scaling factor to 0.84, so that the pi0
mass is aligned to 135 MeV. 2. improve the pi0 reconstruction for both the
with vertex and without vertex … the hand scan results … help improve the pi0
reconstruction in the particle flow … 3. For the T_KINE, we want to make sure
that the pi0 reconstruction are not biased by the pi0 mass … keep it this way
… not all pi0 are from the main neutrino vertex … for the pi0 without vertex
(e.g. NC pi0), we will need to update the neutrino vertex … use the metric of
reconstructed pi0 accuracy … it is possible that our hand scan does not
contain all the pi0 … where we have nueCC, and the large electron shower is
paired with a tiny energy gamma shower to form an pi0. if the electron is
directly connected to the main vertex, we should minimize this kind of
reconstruction."*

Scope decisions (owner, via session Q&A): new doc (this one) + a pointer in
pr/126; a model pairing pass over the top ~40 rescan events under a NEW label
tag; production flip of `kine_shower_fudge_factor` to 0.84 NOW, all other
knobs DEFAULT OFF this round — measured, recommended, flipped only after owner
adjudication.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# sec 2 -- baseline re-point at the pr/131 production point (toolkit 95346dc5)
python3 scripts/pr126_pi0_census.py \
    --manifest141 em114c-132denom141-manifest.tsv \
    --manifest98  em117-132denom98-manifest.tsv --tsv <out>

# sec 3 -- the probe arms (miss attribution; WCT_PI0_PAIR_DEBUG)
./scripts/pr132_arms.sh dbg98  dbgpi098  1
./scripts/pr132_arms.sh dbg141 dbgpi0141 1
grep -h PI0_PAIR work-pr132-dbgpi0*/pr_evt*/*.log

# sec 5 -- OFF gate (new binary, no env) vs the pr131-denom baseline
./scripts/pr132_arms.sh 98  off98  0 && ./scripts/pr132_arms.sh 141 off141 0
python3 scripts/pr85_hash_gate.py work-pr131-denom98-<s>  work-pr132-off98-<s>   # x4
python3 scripts/pr85_hash_gate.py work-pr131-denom141-<s> work-pr132-off141-<s>  # x2

# sec 6 -- ON arms (env -> TLA; see run_pr_chain_batch.sh "doc pr/132" block)
./scripts/pr132_arms.sh 98 onfudge98 0 SBND_KINE_SHOWER_FUDGE=0.84   # + 141
./scripts/pr132_arms.sh 98 oneff98   1 SBND_KINE_SHOWER_FUDGE=0.84 \
    SBND_PI0_ASSOC_ANGLE=40 SBND_PI0_NV_ALLOW_TYPE2=1 SBND_PI0_NV_MAX_PRONGS=5  # + 141
./scripts/pr132_arms.sh 98 onguard98 0 SBND_KINE_SHOWER_FUDGE=0.84 SBND_PI0_ATTACH_MIN_MEV=20  # + 141
./scripts/pr132_arms.sh 98 onoff098  0 SBND_KINE_SHOWER_FUDGE=0.84 SBND_PI0_MASS_OFFSET=0      # + 141

# the census fork (scales the hand-mass blocker to the arm's fudge/offset)
python3 scripts/pr132_pi0_census.py --manifest141 <tsv> --manifest98 <tsv> \
    --fudge 0.84 --offset 10 --overlay-tag pi0scan-0829-agent --tsv <out>

# sec 4 -- the pairing pass (top-40 of docs/pr/pr126-pi0-rescan.tsv)
python3 scripts/pr132_pi0_pair.py --packet <EV>      # adjudication packet
python3 scripts/pr132_pi0_pair.py --write <EV> --g1 <ID> --g2 <ID> ...
```

Toolkit point: knobs + probe on top of `95346dc5` (this round's commits).
Baseline arms: `work-pr131-denom{98,141}-*` (239 events, dumps + logs on disk);
manifests written this round: `em_display/em117-132denom98-manifest.tsv`,
`em_display/em114c-132denom141-manifest.tsv`.

---

# 1. What ships in this round

| piece | state |
|---|---|
| `kine_shower_fudge_factor` 0.80 → **0.84** | SBND PRODUCTION ON (owner order; pr/126 §4g: peak floor 0.829, prototype ×0.95 ⇒ 0.842) |
| K1 `pi0_mass_offset` (MeV, default 10) | default OFF (=10, legacy); measured at 0 |
| K2 `pi0_assoc_angle_deg` (default 30) | default OFF (=30); measured at 40 |
| K3 `pi0_attached_partner_min_mev` (default 0) | default OFF; measured at 20 |
| K4 `pi0_nv_allow_type2` (default false) | default OFF; measured ON |
| K5 `pi0_nv_max_prongs` (default 2) + GATE2 companion | default OFF (=2); measured at 5 |
| `WCT_PI0_PAIR_DEBUG` | byte-neutral env probe (stderr only), both finders |
| pairing pass | `em_labels/pi0scan-0829-agent/` (~40 events; NEW tag, M13) |
| K6 `pi0_nv_skip_paired` | **NOT implemented** — probe trigger census first (§3) |

The T_KINE guarantee, stated once: neither the fudge flip nor any knob adds a
mass window to the `pio_kine` scans (`NeutrinoShowerClustering.cxx` T_KINE
blocks). K1/K3 act ONLY in the PF admission loops; K2/K4/K5 enlarge the shared
candidate pool, so `kine_pio_*` values can move when those flip — but the
selection stays max-energy, mass-blind, exactly as the owner requires so that
the T_KINE π⁰ mass can calibrate energy. Known indirect couplings (pi0
acceptance → `calculate_kinematics` recompute → `kine_charge`; the flag=2 scan
seeded from flag=1) are pre-existing and documented in pr/126 §2.6/§4f.

Knob plumbing is the house 7-step idiom; default-lock rows added to
`clus/test/doctest_clus_knob_defaults.cxx` (2544 assertions PASS). The bare
`wcsonnet` compile of `wct-pr-perevt.jsonnet` at defaults is **byte-identical
to HEAD** (knob declarations are fully key-suppressed), and the compiled-config
proof was taken from a real per-event config: `.wct-cfg-evt37112.json` of the
smoke arm carries all six keys at their TLA values (the pr/129 lesson — a bare
wcsonnet proof is vacuous for the PR chain, the per-event `.wct-cfg` is not).

# 2. The baseline moved under us: pr/127–131 cost the π⁰ census 6 exact matches

Re-pointing the pr/126 census at the **pr/131 production point** (the flips
`sccc_max_gap=10`, `shower_satellite_absorb`, `shower_pass4_prox_guard_len=50`,
`shower_pass3_backfill_guard_len=15`, `stem_backfill_back_dvtx=45`, …):

| | pr/125 flipchk (pr/126 §4f refresh) | pr/131-denom (this round's baseline) |
|---|---|---|
| exact | 26 (52 %) | **20 (40 %)** |
| partial | 10 | 11 |
| none | 2 | 2 |
| no-group | 12 | **17** |
| accepted groups (239 evts) | 75 | **71** |
| with / without vertex | 74 / 1 | 70 / 1 |

Control: the same script on the pr/125 arms still returns 26/10/2/12, so this
is production drift, not a script artifact. The movers:

| event | 125→131 | mechanism (probe, §3) |
|---|---|---|
| 141-71872 | exact→no-group | γ2 re-typed pdg 211 (PID drift) |
| 141-99782 | exact→no-group | reco mass 161.2, 1.2 MeV above the (100,160) window |
| 141-403023 | exact→no-group | mass 162.2 |
| 141-74123 | exact→partial | hand pair m=148.1 loses the 125-centred ranking to a wrong pair at 111.4 |
| 98-278794 | exact→no-group | mass 164.0 |
| 98-359980 | exact→no-group | mass 188.9 |

**Mechanism**: the pr/127–131 guard releases moved charge INTO showers, so the
same pairs' masses drifted a few MeV up — over the window's 160 MeV edge. The
π⁰ census was not a gate in those rounds (the pr/127 lesson again: a metric
nobody watches, drifts). Two consequences: (a) this round's OFF baseline is
20/11/2/17, and (b) the fudge flip (×0.952 on every mass) is itself the
principal rescue for this class — 161.2→153.5, 162.2→154.5, 164.0→156.2 all
come back inside (100,160); 171.2 (103798) and 188.9 (359980) do not.

# 3. Miss attribution from the probe (WCT_PI0_PAIR_DEBUG)

Probe arms `work-pr132-dbgpi0{98,141}-*` (24 targeted events: the 17 no-group
+ 2 none + the §2 movers + rescan heads 37112/142421/415278/176502). One line
per association verdict, recorded pair, window/veto verdict, path-2 gate.
Classes over the misses:

| class | events | evidence (one specimen) | lever |
|---|---|---|---|
| mass just above window top | 99782, 403023, 285443, 278794, 168432(F), 103798, 359980 | 99782: only pair m=161.2, `winreject` | fudge flip (§2); K1 offset for the stragglers |
| 30° association too tight | 37112 (35.2°), 347129 (36.0°), 54341 (38.2° partial), 56243 (40.7°) | 37112: `assoc vtx=84097 sh=67048 E=797 angle=35.2 acc=0` | K2 = 40 |
| ranking pulled to 125 | 74123 | hand pair 148.1 recorded, 111.4 accepted | K1 = 0 recentres to 135 |
| γ typed as track (PID) | 71872, 415278(γ2), 142421(γ2 side), 47212, +8 pr/126 blockers | 71872: pool has no second γ | recognition thread (deferred, pr/126 item 1) |
| second γ merged / crumbs only | 176502, 506746, 281485, 285567 | 176502: best partner for the 722 MeV γ is a 30 MeV crumb | upstream clustering; partially the pairing pass |
| path-2 pre-gates | gate1 (nsegs 3–5): 176502, 415278, 142421, 37112, 285567, 506746; gate2: 8 events; gate3: 3 | `P2 return=gate1 nsegs=3` | K5 = 5 (+ GATE2 companion) |

K6 trigger census (`P2 return=already_paired … unpaired_ct1_left=N`): the only
firing in the 24 probe events is 74123 with `unpaired_ct1_left=0` — the
early-return never blocks a live second candidate there. **K6 stays
unimplemented** (the reviewer's probe-first condition was not met).

Smoke proof of every knob (arm `work-pr132-smoke1-*`, all knobs + fudge, evt
37112): K2=40 flips the 35.2° association to `acc=1`; K3=20 emits `P1 veto` on
the attached+17.7 MeV pairs; K4 puts the 759 MeV conn-2 shower into the path-2
ray pool (`ray src=other ct=2`); K5 passes gate1 at nsegs=4; every kine_charge
is ×0.952 (797.3→759.4) — the fudge TLA is live.

# 4. The pairing pass: 40 events adjudicated, 16 new pairs, and the peak refreshed

pr/126 §4h item 0, executed as a MODEL pass (like the 141-set's emscan-0828-
agent5) over the top 40 rows of `docs/pr/pr126-pi0-rescan.tsv`, under the NEW
tag `em_labels/pi0scan-0829-agent/` (M13: base scan tags untouched). Tool:
`scripts/pr132_pi0_pair.py` (--packet adjudication view over the pr131-denom
dumps via em_geom.py; --write/--nopair emits a pio block schema-compatible
with `pr126_pi0_select.py`). Pairing rules were fixed before any event was
looked at, and rule 1 is the one that protects the calibration: **pair on
topology only — never prefer or reject a pairing for its mass**; nueCC
primary electrons are never paired (the owner's fake mode); ambiguity ⇒ an
explicit no-pair record.

Result: **16 pairs + 24 explicit no-pairs.** The no-pair notes cluster into
exactly the classes the probe found: owner-noted over-clustering /
wrong-vertex events (176502, 281567, 463565, 30504, 38856, 180801), primary-
electron-only nueCC events (163543, 444187, 75954, 423981, 64409, 176533,
284206), and collinear-fragment "pairs" that are one split EM system (98844,
100222, 410008, 283515, 287830, 71642, 116962). A recurring pairing-side
finding worth naming: the second gamma is often present but FRAGMENTED into
collinear pieces (56982, 259542, 71642) — the under/over-clustering thread
seen from the pi0 side.

### The refreshed peak (`scripts/pr132_pi0_peak_refresh.py`, estimator = pr/126 §4g unchanged)

| cell | n | n_in | peak | CI68 | implied fudge | peak after the 0.84 flip |
|---|---|---|---|---|---|---|
| base pooled (pr/126 §4h) | 45 | 34 | 140.6 | [136.3, 144.3] | 0.833 [0.808, 0.855] | 133.9 |
| overlay (pi0scan) | 12 | 6 | 144.3 | [126.8, 185.0] | 0.855 | 137.4 |
| **union** | **57** | 40 | **140.8** | **[136.6, 144.6]** | **0.835 [0.810, 0.857]** | **134.1** |

TSV: `docs/pr/pr132-pi0-peak.tsv`. Reading: the extended sample confirms the
pr/126 §4g direction and magnitude — 0.84 sits inside the union CI, and after
the flip the fitted peak lands at **134.1 MeV**, consistent with 134.98 given
that the §4g toys put the fitted peak a few MeV LOW against a heavy low tail.
The overlay's own cell is wide (n_in = 6 — most model pairs sit above the fit
window, the over-merge epidemic again) and is corroboration, not load-bearing.

# 5. OFF gate: PASS 6/6

New binary (probe + K1-K5 in, all knobs at defaults, no env) vs the
`work-pr131-denom*` baseline, `scripts/pr85_hash_gate.py` per sample-arm
(member-content hashes, exit-code discipline):

| pair | archives | verdict |
|---|---|---|
| denom98-mcp1k vs off98-mcp1k | 28 | PASS rc=0 |
| denom98-mcp2k vs off98-mcp2k | 34 | PASS rc=0 |
| denom98-ncpi0 vs off98-ncpi0 | 38 | PASS rc=0 |
| denom98-nuecc48 vs off98-nuecc48 | 96 | PASS rc=0 |
| denom141-mcp1k vs off141-mcp1k | 104 | PASS rc=0 |
| denom141-mcp2k vs off141-mcp2k | 178 | PASS rc=0 |

Every archive of all 239 events byte-identical: the five knobs and the probe
are inert when off.  Gate logs: `/home/xqian/tmp/pr132-gate-*.log`; arms kept
as `work-pr132-off{98,141}-*`.

# 6. Knob-ON arms and the census

Four ON families, all 239 events each, all on top of `SBND_KINE_SHOWER_FUDGE=0.84`
(arm rc=0 throughout; census = `pr132_pi0_census.py --fudge 0.84
--overlay-tag pi0scan-0829-agent`). "base-50" = the original hand pi0;
"overlay-16" = the pairing-pass pairs (never before visible to any census).

| arm | env on top of fudge | base-50 e/p/n/ng | overlay-16 e/p | groups (239) | P2 acc | fakes(<30) | rescan cov |
|---|---|---|---|---|---|---|---|
| pr131-denom baseline | — (fudge 0.80) | 20/11/2/17 | 0/? | 71 | 1 | 8 | 30/109 |
| **onfudge** | — | **24/8/2/16** | **2/5** | 69 | 1 | 10 | 29/109 |
| oneff | K2=40, K4=on, K5=5 | 24/8/2/16 | 2/6 | 76 | **4** | 11 | 34/109 |
| onguard | K3=20 | 24/8/2/16 | 2/5 | 64 | 1 | **5** | 29/109 |
| onoff0 | K1=0 | 24/7/2/17 | 3/5 | 63 | 1 | 8 | 29/109 |

### 6.1 The fudge flip alone (`onfudge`) — the production candidate

Base-50 movers, every one mechanism-attributed:

* **+5 exact**: 99782 (161.2→153.5), 403023 (162.2→154.5), 285443
  (161.7→154.0), 278794 (164.0→156.2) — the §2/§3 window-edge class scaled
  back inside (100,160) — and **74123 partial→exact** (the hand pair's ranking
  restored once masses recentre).
* **−1 exact**: 64591 — its accepted group (52.9+303.3 MeV, mass 101.9) scales
  to 97.0, below the window floor. This is pr/126 §4e's predicted
  one-confirmed-loss, now named. It is a *window-edge* casualty, not a scale
  error; K1/offset work (round 2) is the lever.
* −2 partial→no-group: 347824 (γ1 renamed between arms — the pr/126 rename
  case, not a physics change), 409634 (mass 83→79, was outside anyway).
* **Overlay: +2 exact +5 partial** — the reconstruction already reproduces 2 of
  the 16 model pairs exactly (314838, 397630) and shares a γ on 5 more.
* Vertex movers: **0 ADVERSE across all six samples** (`pr90_movers.py
  --tags vtx105`, off vs onfudge). Net census at the new production point:
  **26 exact / 66** (24 base + 2 overlay), vs 20/50 before.

### 6.2 K3 (`onguard`, attached-partner < 20 MeV) — works exactly as designed

Zero change on the hand pi0 (base-50 and overlay identical to onfudge),
**fake topologies 10 → 5**, groups 69 → 64. The five survivors all have
partner energies 20.5–29.6 MeV (54095, 76346, 76350, 176502, 268784) — a
threshold of 30 would remove all five, at the price of touching 76346's
partial group. **Recommendation: flip K3 at 20 (conservative) or 25 after a
one-look Bee check of the five survivors.**

### 6.3 K2/K4/K5 (`oneff`) — measured HARMFUL as configured, do not flip

On the hand pi0: nothing (base-50 identical to onfudge; overlay +1 partial =
415278). Globally: +7 groups, path-2 acceptances 1 → 4 (396222 + NEW 116962,
122660, 171143), rescan coverage +5. But the three new path-2 acceptances are
the whole story: **two are ADVERSE vertex movers** — 122660 moved 23.0 cm off
the truth click (nue score 3.91→4.28 on a nuecc event whose pairing pass
verdict was "radiated fragments, no pair") and 171143 moved 5.78 cm.
`id_pi0_without_vertex`'s vertex mutation (§pr/126 2.6) does exactly what it
says when woken. K2's own admissions (37112's 35.2° association now accepted)
did not convert: the admitted pair's mass (322.7·0.952 ≈ 307) is far outside
any window — the 37112 failure is over-merged charge, not the angle.
**Verdict: K2/K4/K5 stay OFF. Round 2 must gate path-2 acceptance on pair
quality (both members detached-EM-like, partner floor a la K3, vertex-move
cap) before revival is safe.**

### 6.4 K1 (`onoff0`, offset 0 ⇒ windows (110,170)/(75,195)) — a trade, owner's call

Rescues the two high-mass stragglers 103798 (163.0 in) and 56243, and 3
overlay pairs land exact; but the raised floor kills 283713 (90.4) and 292524
(**109.0 vs 110 — by one MeV**), plus 2 partials; and it ADMITS a new fake
(30504: 628 MeV attached e + 11.5 MeV crumb — K3 would veto it). Net exact
27/66 vs onfudge 26/66 with a worse partial column. **Recommendation: not
this round.  The †offset-5 compromise (windows (105,165)) plus K3 is the
round-2 arm worth running; scale and offset remain one degree of freedom
(pr/126 §4e) and the owner should adjudicate the moved groups on Bee.**

# 7. The production flip, and its equivalence proof

`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` `kine_shower_fudge_factor
= null` → `0.84` (the only value change; comment block carries the pr/126 §4g
+ §4 provenance). Owner order on record: *"adjust the EM charge scaling factor
to 0.84, so that the pi0 mass is aligned to 135 MeV."*

* Compiled-config proof, per-event (the pr/129 vacuity lesson — a bare
  `wcsonnet` compile does not even emit TaggerCheckNeutrino):
  `work-pr132-flipchk98-ncpi0/pr_evt37112/.wct-cfg-evt37112.json` carries
  `kine_shower_fudge_factor: 0.84` and **no** `pi0_*` knob key — the five new
  knobs stay fully suppressed at their defaults.
* **Flip-equivalence gate PASS 6/6**: `work-pr132-flipchk{98,141}-*` (post-flip
  cfg, NO env) hash-identical to `work-pr132-onfudge{98,141}-*` (pre-flip cfg,
  `SBND_KINE_SHOWER_FUDGE=0.84`) on every archive of all 239 events
  (`pr85_hash_gate.py`, rc=0 ×6; logs `/home/xqian/tmp/pr132-flipeq-*.log`).
  The production flip is byte-equal to the operating point every §6 number was
  measured on.

Vertex movers off→flip: 0 ADVERSE (§6.1). The nusel tables of the flipchk arms
are the new production reference.

# 8. Recommendations and the round-2 queue

Shipped this round: the 0.84 flip (owner order), K1–K5 default OFF, the
probe, the census fork, the pairing pass, this doc.

| item | recommendation |
|---|---|
| K3 attached-partner guard | **flip at 20** next round (zero hand-pi0 cost, −5 fakes); consider 25–30 after a Bee look at the five 20–30 MeV survivors |
| K1 offset | run the offset-5 + K3 joint arm; owner adjudicates the edge groups (64591, 283713, 292524 are the named specimens) |
| K2 assoc angle | keep OFF; re-test after the over-merge thread — the probe shows the 30° cut is rarely the binding constraint once charge is right |
| K4/K5 path-2 revival | keep OFF; redesign with an acceptance-quality gate (ADVERSE movers 122660/171143 are the counterexamples); the owner's NC-vertex-update goal stands, the current acceptance test is too loose for busy vertices |
| γ-typed-as-track (8+ events: 71872, 415278, 47212, 169626, 285567, 506746, 54341, 52044…) | the single largest remaining efficiency block — recognition-thread round (pr/126 item 1), not pi0 code |
| second-γ fragmentation (56982, 259542, 71642) + over-merge (176502, 463565, 37112) | the pairing pass's view of the under/over-clustering thread; feeds the same round-2 front |
| pairing pass, rows 41–109 | extend when more truth is wanted; the top-40 yielded 16/40 — the tail will yield less (ordered by 2nd-γ energy) |
| pr/127–131 regression guard | `pr132_pi0_census.py` against the current production manifests should join the sentinel suite so the pi0 census cannot drift silently again (the pr/127 lesson) |

**What is NOT claimed.** The 16 model pairs are one model's topology-only
adjudications, not owner scans; census gains on them are agreement with the
model, and they are reported separately (labelsrc column) so they can never
silently inflate the base-50 numbers. The T_KINE pio_kine block remains
mass-window-free; its values move with the EM scale (that is the calibration)
and with any future pool-enlarging knob (documented, mass-blind).
