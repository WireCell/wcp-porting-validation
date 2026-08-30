# doc pr/132 — π⁰ reconstruction round 1: the EM scale flip to 0.84, five finder knobs, and the pairing pass

**Status: rounds 1+2+3 CLOSED (2026-08-30). Fudge 0.84 + K7+K8 (track-gamma rescue) + K3=28 SBND PRODUCTION ON; K1/K2/K4/K5/K9/K10/K11 + round-3 K12 DEFAULT OFF, measured; over-merge adjudicated UPSTREAM by the substructure probe. Round 1 = secs 1-8; round 2 = sec 9; round 3 = sec 10.**
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

Round 2 (sec 9):
```
# arms (each dir holds BOTH manifests' events; census/gates are manifest-driven)
PR_JOBS=12 bash scripts/pr132_arms.sh 98 r2off 0    && PR_JOBS=12 bash scripts/pr132_arms.sh 141 r2off 0
PR_JOBS=12 bash scripts/pr132_arms.sh 98 r2resc 1 SBND_PI0_READMIT_RETYPED=1 SBND_PI0_ADMIT_TYPE3=1 SBND_PI0_CRUMB_MEV=30  && ...141...
PR_JOBS=12 bash scripts/pr132_arms.sh 98 r2o5 1 SBND_PI0_MASS_OFFSET=5  && ...141...
PR_JOBS=12 bash scripts/pr132_arms.sh 98 r2p2 1 SBND_PI0_NV_ALLOW_TYPE2=1 SBND_PI0_NV_MAX_PRONGS=5 SBND_PI0_NV_MASS_WIN=30  && ...141...
# gate (K3-flip equivalence + round-2 code neutrality, one proof)
for s in mcp1k mcp2k ncpi0 nuecc48; do python3 scripts/pr85_hash_gate.py work-pr132-onguard98-$s work-pr132-r2off-$s; done   # + onguard141-{mcp1k,mcp2k}; rc=0 x6
# manifests + census + movers
bash scripts/pr132_r2_manifests.sh r2off   # (r2resc r2o5 r2p2)
python3 scripts/pr132_pi0_census.py --manifest98 em117-132r2off98-manifest.tsv --manifest141 em114c-132r2off141-manifest.tsv \
    --fudge 0.84 --overlay-tag pi0scan-0829-agent --tsv docs/pr/pr132-census-r2off.tsv   # (--offset 5 on r2o5)
python3 scripts/pr90_movers.py work-pr132-r2off-$s work-pr132-r2resc-$s --tags vtx105   # 0 movers x4; same vs r2p2
```

Round 3 (sec 10):
```
# arms (merged-dir convention as round 2); binary = toolkit round-3 HEAD (K12 + substruct probe)
PR_JOBS=16 bash scripts/pr132_arms.sh 98 r3off 0    && PR_JOBS=16 bash scripts/pr132_arms.sh 141 r3off 0
PR_JOBS=16 bash scripts/pr132_arms.sh 98 r3flip 1 SBND_PI0_READMIT_RETYPED=1 SBND_PI0_ADMIT_TYPE3=1 SBND_PI0_ATTACH_MIN_MEV=28  && ...141...
PR_JOBS=10 bash scripts/pr132_arms.sh 98 r3cm 1 SBND_PI0_READMIT_RETYPED=1 SBND_PI0_ADMIT_TYPE3=1 SBND_PI0_ATTACH_MIN_MEV=28 SBND_PI0_COLLINEAR_DEG=20  && ...141...
PR_JOBS=6  bash scripts/pr132_arms.sh dbg98 r3sub 1 SBND_PI0_READMIT_RETYPED=1 SBND_PI0_ADMIT_TYPE3=1 SBND_PI0_ATTACH_MIN_MEV=28 WCT_PI0_SUBSTRUCT_DEBUG=1  && ...dbg141... (+ r3sub2 = evt 281567 alone)
# OFF gate (K12 + substruct-probe byte-neutrality)
for s in mcp1k mcp2k ncpi0 nuecc48; do python3 scripts/pr85_hash_gate.py work-pr132-r2off-$s work-pr132-r3off-$s; done  # rc=0 x4, 478 archives
# census + movers (baseline = the r2off TSV)
bash scripts/pr132_r2_manifests.sh r3flip   # (r3cm)
python3 scripts/pr132_pi0_census.py --manifest98 em117-132r3flip98-manifest.tsv --manifest141 em114c-132r3flip141-manifest.tsv \
    --fudge 0.84 --overlay-tag pi0scan-0829-agent --tsv docs/pr/pr132-census-r3flip.tsv
python3 scripts/pr90_movers.py work-pr132-r2off-$s work-pr132-r3flip-$s --tags vtx105   # 0 movers x4; same r3flip vs r3cm
# flip-equivalence (post-flip cfg, NO env)
PR_JOBS=20 bash scripts/pr132_arms.sh 98 r3flipchk 0 && bash scripts/pr132_arms.sh 141 r3flipchk 0
for s in mcp1k mcp2k ncpi0 nuecc48; do python3 scripts/pr85_hash_gate.py work-pr132-r3flip-$s work-pr132-r3flipchk-$s; done
```

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

# 9. Round 2 — K3 flip, the track-rescue family, the path-2 quality gate (2026-08-30)

**Owner brief** (2026-08-30): *"Let's proceed to 1"* (= the round-1
recommendation: flip K3 at 20 + the offset-5+K3 joint arm); on the
γ-typed-as-track class: *"I assume those track-like are isolated clusters. In
this case, I think the code can look at them, and update them to an EM shower
in the pi0 reconstruction code"*; *"Perform path-2 acceptance-quality gate
design"*; ideas wanted on second-γ fragmentation/over-merge.

## 9.1 What the data said about the γ-typed-as-track class

The owner's isolated-cluster reading is confirmed and sharpened: joining the
hand labels to the dumps shows every labeled "track" γ is ALREADY a WCShower
record in the shower maps — pdg 211/2212, mostly conn_type 2 (detached,
isolated cluster), some ct 1/3.  The finder excludes them by TYPE, not by
absence.  Three distinct exclusion mechanisms (probe evidence, round-1 dbg
arms):

| event | labeled γ (pdg, ct, E MeV) | exclusion mechanism |
|---|---|---|
| 285567 | 8107 (211, ct1, 102) | pr/99 A5 `m_hadronic_retyped_shower_ids` veto |
| 506746 | 21056 (211, ct1, 78) | A5 veto |
| 52044 | 18004 (211, ct1, 104) | A5 veto |
| 169626 | 22034 (211, ct1, 108) | A5 veto in path-2 good_showers — its true pair (m=145.8) FORMED in path 2 and died only on `good=0` |
| 47212 | 70038 (2212, ct3, 65) | conn_type 3 is in NO path-1 pool |
| 71872 | 64044 (211, ct2, 23) | in the pool, but the 23 MeV crumb's PCA direction fails association at 73° |

## 9.2 The round-2 knob family (all DEFAULT OFF)

| knob | default (legacy) | what ON does |
|---|---|---|
| K7 `pi0_readmit_retyped` | false | readmit A5-retyped showers into all four pi0 pool sites; accepted pair members still track-typed are re-stamped EM (`pi0_restamp_shower_em`: segment pdg→11 + 4-mom, shower type→11, `set_kine_best(0)` restores the EM best-energy fall-through, id erased from the A5 set) |
| K8 `pi0_admit_type3` | false | with-vertex disconnected pool also admits conn_type==3 |
| K9 `pi0_crumb_assoc_mev` | 0 = off | below this energy a disconnected shower skips the association-angle test (PCA direction of a crumb is noise) |
| K10 `pi0_nv_max_vtx_shift_cm` | 0 = off | path-2 selection skips pairs whose decay point is farther than this from the current main vertex |
| K11 `pi0_nv_mass_window_mev` | 60 = legacy | path-2 acceptance half-window |m−135+offset| |

The re-stamp is the owner's "update them to an EM shower", executed at
acceptance time so only pairing-confirmed objects are re-typed; the mass
window + greedy selection + the K3 guard carry the fake control.

## 9.3 Smoke evidence (arms `work-pr132-r2smkresc-*`, `r2smkp2*`)

K7+K8+K9=30 on the specimen events converts four to their HAND pairs with
re-stamp: 47212 accept 70038+109100 m=132.8 (K8), 285567 accept 121096+8107
m=139.0 (K7; the flipped K3 vetoed 25 crumb pairings against 8107 on the
way), 506746 accept 21056+69124 m=147.2 (K7), 169626 path-2 accept
22034+53069 m=138.9 (K7 good_showers).  52044 re-stamps its γ (18004) but
pairs it with the wrong partner (58029, m=116.1; the true partner 24035 fails
association at 126°) — a partial.  71872 does not convert (K9 admits the
23 MeV crumb everywhere but no in-window pair forms; 323 winrejects of crumb
combinatorics, all held by the window).  54341 unchanged (its failure is
under-counted charge, m=68.7 — the fragmentation thread).

## 9.4 The path-2 acceptance-quality gate: shift does NOT discriminate, mass does

The round-1 design sketch proposed a vertex-shift cap.  Measured (probe now
prints the decay-point shift), K4+K5 smoke:

| acceptance | m (MeV) | shift (cm) | verdict |
|---|---|---|---|
| 122660 (ADVERSE r1) | 85.2 | 23.0 | fake — radiated fragments |
| 171143 (ADVERSE r1) | 75.5 | 5.8 | fake |
| 396222 (legacy fire) | 133.5 | 14.5 | good |
| 169626 (K7 smoke) | 138.9 | 59.6 | true pair |

A shift cap separates NOTHING (the good ones sit at 14.5 and 59.6 cm, one
fake at 5.8).  The pi0 MASS does: fakes 75–85, good 133–139.  So the gate
shipped as K11 (path-2 window 60 → 30 ⇒ acceptance band (95,155) at legacy
offset): kills both fakes, keeps both good pairs.  K10 (the shift cap) is
implemented, measured non-discriminating, and stays a documented OFF knob.

## 9.5 The K3 flip and the round-2 OFF gate — one proof, both claims

`wct-pr-perevt.jsonnet` `pi0_attached_partner_min_mev = 0` -> `20` (owner:
"Let's proceed to 1").  Gate: `work-pr132-r2off-*` (round-2 binary, post-flip
production config, NO env) hash-identical to `work-pr132-onguard{98,141}-*`
(round-1 binary, env fudge=0.84 + K3=20) on every archive of all 239 events
-- **PASS 6/6, 478 archives** (`pr85_hash_gate.py` rc=0 x6, logs
`/home/xqian/tmp/pr132r2-gate-*.log`).  One gate proves both: the K3 flip
equals the round-1-validated operating point, and the K7-K11 code is
byte-neutral at defaults.  Compiled-config proof (per-event, pr/129 lesson):
`work-pr132-r2smkp2b-nuecc48/pr_evt122660/.wct-cfg-evt122660.json` carries
`kine_shower_fudge_factor 0.84` + `pi0_attached_partner_min_mev 20` and no
round-2 key when suppressed.

## 9.6 The rescue arm (`r2resc` = K7+K8+K9=30) — census 26 -> 31 exact

`pr132_pi0_census.py --fudge 0.84 --overlay-tag pi0scan-0829-agent`,
manifests `em117-132r2{off,resc}98` / `em114c-132r2{off,resc}141`:

| class | r2off (new production) | r2resc | movers |
|---|---|---|---|
| exact | 26 | **31** | +47212 (K8, m=132.8), +169626 (K7 path-2, m=138.9), +285567 (K7, m=139.0), +506746 (K7, m=147.2), +392901 overlay (K8, m=129.1) |
| partial | 13 | 16 | +52044 (K7, γ re-stamped, wrong partner), +347824 (K7, m=138.3), +486907 overlay (K9, 18.3 MeV crumb) |
| none | 2 | 1 | 47212 upgraded out |
| no-group | 25 | 18 | |
| nueCC fakes (E) | 5 | 8 | +116962 (partner 27.0), +282909 (23.8), +282979 (24.5) — all inside a K3=25-30 veto |
| accepted groups (239 evts) | 64 | 91 | rescan coverage 26 -> 37 of 109 |

Zero class downgrades.  Vertex movers off->resc: **0 movers > 0.05 cm on all
4 samples** (`pr90_movers.py --tags vtx105`).  169626 (unlabeled in vtx105)
moved 59.6 cm BY the path-2 acceptance -- onto the scanner's own
back-projected decay vertex: hand `vertex [3.748, 208.544, 414.675]
(vertex_how=backproject)` vs new main vertex `(3.75, 208.54, 414.68)` --
sub-mm agreement with hand truth.  The pdg=211 blocker column drops 6 -> 2,
pdg=2212 2 -> 0.

## 9.7 The offset-5 + K3 joint arm (`r2o5`) — the trade, sharpened, for the owner

Windows (105,165)/(70,190) at fudge 0.84, K3=20 in-config.  Census 26 -> 28
exact, fakes unchanged at 5 (K3 holds the floor — the offset-0 fake
admission of round 1 does not recur).  Movers: **+56243, +103798 exact** (the
round-1 high-mass stragglers, as predicted), +168432 partial, +486907
overlay exact; **−283713** (exact -> no-group; scaled mass 96 vs the new 105
floor — the round-1-named floor casualty) and −506114 (partial -> no-group).
292524, round 1's by-1-MeV casualty at offset 0, SURVIVES at offset 5.  Net
+2 exact / −1 partial with zero fake cost: a real trade, better than offset
0 on every axis, still the owner's call (Bee specimens: 283713, 506114 lost;
56243, 103798 gained).

## 9.8 The path-2 revival with the gate (`r2p2` = K4+K5+K11=30) — safe now, but empty here

Census identical to r2off on the hand pi0 (26 exact, 5 fakes).  Vertex
movers off->p2: **0 on every labeled event, 0 ADVERSE** — K11 blocks both
round-1 ADVERSE acceptances (122660 m=85.2 acc=0, 171143 m=75.5 acc=0) while
keeping the legacy fire (396222 m=133.5).  Path-2 acceptances 1 -> 2: the
one NEW acceptance is 116962 (55030 E=31 + 21072 E=232, m=124.8, shift
12.0 cm) — a nueCC event whose pairing-pass verdict was "collinear fragments
of the primary electron" and whose scan label carries note "incorrect
vertex" with the label vertex = the scan-time reco position (0.01 cm from
the off arm), so the move is not adjudicable from labels.  Verdict: the
GATE works (that was this round's design goal); the REVIVAL (K4/K5) buys
nothing on these 239 events and admits one unresolved acceptance — K4/K5
stay OFF; a path-2 partner floor (K3 analog; 116962's partner is 31 MeV)
is the round-3 refinement if NC-pi0 revival is still wanted on a sample
where it can show gains (the current manifests hold no reachable NC pi0
with a mis-seated vertex).

## 9.9 Second-γ fragmentation / over-merge: population + round-3 design

Population, measured from the round-1 no-pair adjudications + probe classes:
of the 25 pairing-pass no-pair events with notes, **15 are
fragmentation-shaped** (collinear splits of one EM system: 100222, 116962,
163543, 175896, 180801, 283515, 284206, 386948, 410008, 423981, 444187,
463565, 54453, 71642, 98844), 2 are hard over-merge (176502, 281567; owner
scan notes), 6 primary-electron-only, 1 other.  On the hand-pi0 side, 54341's
miss is under-counted second-γ charge (pair mass 68.7 at the right topology),
and 37112's is the mirror image (over-merged charge, pair mass ~307).

The two directions are asymmetric and want different fixes:

1. **Fragmentation (second γ split into collinear pieces).**  Round-3
   candidate knob `pi0_collinear_merge_deg`: at pairing time, before the mass
   computation, greedily absorb into each disconnected candidate any OTHER
   disconnected shower whose start lies within a cone around the candidate's
   vertex ray (axis angle < knob AND same side), summing kine_charge and
   keeping the leading fragment's direction.  A virtual merge only -- shower
   objects untouched unless the pair is ACCEPTED (then absorb, following the
   K7 re-stamp precedent).  This directly targets the 54341 class and the
   E-sum deficit the peak fit sees (most overlay pairs sit above the fit
   window).  It must NOT fire on primary-electron fragments: keep the K3
   guard + require the merged system detached (ct!=1) or partner above the
   K3 floor.
2. **Over-merge (both γ in one shower).**  A pi0-side split is NOT
   recommended: the round-1 K2 evidence (37112: association widened, pair
   formed, mass ~307 -- charge is simply merged) shows the defect lives
   upstream in shower building (the pr/123-125 over-clustering threads).
   Proposal: a byte-neutral probe first (two-axis substructure test on
   accepted single showers > 300 MeV: PCA split of the associate-points
   cloud, report the two-cluster mass), to measure how many over-merged
   events are even recoverable before any knob is designed.
3. **The crumb ceiling.**  K9 shows admission alone is not enough when the
   crumb's charge is a fraction of the true γ (71872: in-pool at every
   vertex, no in-window pair).  Fragment SUMMING (idea 1) and crumb
   admission (K9) compose: the merged crumb system carries the full γ
   energy and lands in the window.

## 9.10 Round-2 recommendations and the round-3 queue

Shipped this round: the K3=20 production flip (owner order), knobs K7-K11
DEFAULT OFF + measured, the re-stamp helper, this section.
Toolkit commits: 651ba9a0 (K7-K11 + re-stamp + probe fields, DEFAULT OFF),
c477210c (K3=20 SBND PRODUCTION ON).

| item | recommendation |
|---|---|
| K7+K8 (readmit-retyped + ct3) | **flip next round**: +5 exact +2 partial on the hand pi0, zero downgrades, zero vertex movers; the only regenerated fake is 116962 (K7-readmitted electron fragment + 27.0 MeV partner) — a K3 bump to 28-30 covers it |
| K3 threshold | bump 20 -> 28-30 WITH the rescue flip (covers 116962 at 27.0; round-1 note: 30 touches 76346's partial group) — one joint owner Bee look: 116962, 76346, 54095, 268784, 176502 |
| K9 crumb assoc (30 MeV) | keep OFF: its ledger is 1 overlay partial (486907) vs 2 regenerated fakes (282909 partner 23.8, 282979 partner 24.5, both crumb-admitted at 59-64 deg) + combinatoric noise (71872: 323 window rejects, no gain); revisit only WITH the round-3 fragment-summing — they compose |
| K1 offset 5 | +2/−1 exact, no fake cost; owner adjudicates (Bee: 283713, 506114 vs 56243, 103798) |
| K10 shift cap | measured non-discriminating; keep OFF, keep documented |
| K11 p2 window 30 | the working acceptance-quality gate; as a solo flip it is a no-op on current production data (legacy P2 fires once, in-window) — flip it WITH any future path-2 work |
| K4/K5 | stay OFF (sec 9.8) |
| fragmentation | round-3 front: virtual collinear merge at pairing time (sec 9.9 design); population 15/25 no-pair notes + 54341/71872 |
| over-merge | upstream thread; byte-neutral substructure probe first (sec 9.9) |
| census sentinel | run `pr132_pi0_census.py --fudge 0.84` vs the r2off manifests each round (baseline: 26 exact / 5 fakes / 64 groups) |

**What is NOT claimed.** Overlay gains are agreement with the round-1 model
pairing pass, reported separately (labelsrc).  T_KINE `pio_kine` remains
mass-window-free; the rescue knobs enlarge its candidate pool (flag=0 events
84 -> 12 on r2resc), a documented mass-blind drift.  The r2 arms hold BOTH
manifests' events in one dir per sample (`work-pr132-r2<tag>-<sample>`);
census/gates are manifest-driven so the merge is inert.

# 10. Round 3 — the K7+K8+K3=28 production flip, K12 measured, over-merge adjudicated upstream (2026-08-30)

Owner order, verbatim: *"Let's execute according to your recommendations '1.
Flip K7+K8 (the track-γ rescue) together with a K3 bump to 28–30 …'"* — the
full round-2 recommendation list (sec 9.10), with *"please continue use the
metric to track improvements."*  Executed here: item 1 (the flip, measured
then shipped), item 3 (the virtual collinear merge, implemented + measured),
item 4 (the over-merge substructure probe, run and adjudicated), item 6 (the
census sentinel, run on every arm).  Item 2 (offset-5) and the round-2 Bee
looks remain owner-adjudication items; item 5 (path-2 revival) stays parked
for a sample with reachable NC π⁰.

## 10.1 What ships

- **`pi0_readmit_retyped = true` + `pi0_admit_type3 = true` +
  `pi0_attached_partner_min_mev = 28` SBND PRODUCTION ON**
  (`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`).  28 is the safe end
  of the owner's 28–30 range: it kills the K7-regenerated fake 116962
  (partner 27.0 MeV) and the three round-2 fake-topology survivors
  54095/76350/268784 (26.1/20.5/22.0 MeV) while keeping 76346's
  partial-group partner (29.6) and 176502 (28.7).  30 would also kill those
  two — that residual pair stays the owner Bee adjudication item.
- **K12 `pi0_collinear_merge_deg` DEFAULT OFF** (C++ 0 = off), implemented
  per the sec 9.9 design and **measured net-negative** (sec 10.3).
- **`WCT_PI0_SUBSTRUCT_DEBUG`**, the over-merge substructure tape
  (byte-neutral, stderr only): PCA two-axis split of every >300 MeV shower's
  associate-points cloud, count-weighted split energies, implied two-gamma
  mass (sec 10.4).

## 10.2 The flip, measured (`r3flip` = K7+K8 via env + K3=28)

Census vs the r2off baseline (26 exact / 13 partial / 2 none / 25 no-group;
5 fakes; 64 groups; 26/109 rescan coverage), same 66-π⁰ denominator,
`--fudge 0.84`:

| metric | r2off | r3flip |
|---|---|---|
| exact | 26 (39.4%) | **31 (47.0%)** |
| partial | 13 | **16** |
| none / no-group | 2 / 25 | 1 / 18 |
| downgrades | — | **0** |
| nueCC-fake topologies (sec E) | 5 | **2** (76346 @29.6, 176502 @28.7) |
| accepted groups | 64 | 82 |
| rescan coverage | 26/109 | 32/109 |
| vertex movers (vtx105, x4 samples) | — | **0, 0 ADVERSE** |

Upgrades, all eight: 169626, 47212 (none→exact), 285567, 506746, 392901 →
exact; 486907, 347824, 52044 → partial.  47212's γ2 is the K8 (conn-type-3
pool) specimen; the rest are the K7 readmissions with accept-time EM
re-stamp.  486907 — attributed to K9 in the round-2 r2resc arm — arrives
without K9: the crumb knob stays OFF with nothing lost.  T_KINE drift
(mass-blind, documented): kine_pio_flag 1/2/0 = 152/3/84 → 202/2/35.

**Gates.**  OFF gate: `r3off` (round-3 binary, no env, pre-flip cfg) vs
`r2off` — **PASS 4/4 dirs, 478/478 archives byte-identical** (132 mcp1k +
212 mcp2k + 38 ncpi0 + 96 nuecc48; the merged dirs hold both manifests'
events, so 4 gates cover all 239) — proving K12 + both probes byte-neutral
off.  Flip-equivalence gate: `r3flipchk` (post-flip cfg, NO env) vs
`r3flip` — result quoted in sec 10.5.  Freshness proof done;
`wcdoctest-clus` 2556/2556 (incl. the K12 default-lock row).
Compiled-config proofs: r3off per-event JSON has NO `pi0_collinear_merge_deg`
key; r3cm per-event JSON carries `pi0_collinear_merge_deg: 20`,
`pi0_readmit_retyped/admit_type3: true`, `pi0_attached_partner_min_mev: 28`.

## 10.3 K12, the virtual collinear merge — implemented, measured, dead in v1 form

Implementation (sec 9.9 idea 1, `id_pi0_with_vertex`): at each candidate
vertex the detached (conn-type≠1) pool is greedily re-clustered leading-first;
fragments within the vertex-ray cone are summed into the leading fragment for
the pair-mass computation only; on ACCEPT the fragments are truly absorbed
(`add_shower` + kinematics + `update_shower_maps`, the P2 precedent) and
their stale pairings retired.  Attached showers neither host nor get
absorbed (the primary-electron guard); the K3 floor still tests the leading
fragment's own charge.

Measurement (`r3cm` = r3flip env + 20°), census vs r3flip:

| event | move | probe attribution |
|---|---|---|
| 284235 | exact → **no-group** | a 9.2° merge added a 14.9 MeV crumb to the true γ2 (84.6→99.4 MeV), pushing the pair to m=160.5 — 0.5 MeV past the window ceiling |
| 399118 | exact → partial | a 9.5° merge of 61033 (22.6 MeV) into 16016; the accepted m=114.6 group no longer matches the label pair |
| 269774 | partial → no-group | 11–13.5° merges reshuffle the pool |
| 103798 | no-group → partial | gains need 13.6–18.8° merges |
| 409634 | no-group → partial | gain needs the 17.2° merge |

Net −2 exact, +1 partial — and the geometry is unfixable by tuning: **the
kills fire at 9.2–13.5°, the gains need 17.2–18.8°** — every cone admitting
the gains admits the kills.  Worse, the design's own motivating specimen
54341 is NOT rescued: the merge fires (γ2 fragment system reaches 281 MeV)
but the pair still win-rejects at m≈44 — the missing charge is not in
detached siblings within any cone.  0 movers (r3flip vs r3cm, x4).
**Verdict: K12 stays DEFAULT OFF.**  A v2 would have to be acceptance-aware
(merge only where unmerged pairing produced NO in-window pair, protecting
284235/399118) — but with both v1 gains landing only at partial and the
target specimen unmoved, that refinement is queued behind better fronts
(sec 10.6).

## 10.4 The over-merge substructure probe: recovery at the π⁰ level is (almost) not there

Tape over the dbg subsets + 281567 (r3sub/r3sub2 arms, flip operating
point): 13 showers >300 MeV in 10 events.  Selected rows (ax = split axis,
ev01 = eigenvalue ratio λ2/λ1, m = implied two-gamma mass in MeV):

| event | shower E (MeV) | ev01 | best split m | reading |
|---|---|---|---|---|
| 37112 (the canonical over-merge, hand π⁰ miss) | 759.4 | 0.031 | 68.7 @10.5° | geometrically collinear — a split cannot reach the window |
| 176502 (owner-scan over-merge) | 2391.2 | 0.134 | 204.1 / 356.1 | transverse structure exists but count-weighted split misses |
| 281567 (owner-scan over-merge) | 623.2 | 0.187 | **114.3 @21.4°** | the only over-merge candidate near the window |
| 415278 | 1239.5 | 0.382 | 210.8 | substructure real, masses wrong |
| true single γs (169626, 47212, 506746, 342199, 359980, 142421) | 398–989 | 0.03–0.10 | 13–95 | probe correctly reports no π⁰-like substructure |

**Verdict: of the three known over-merge events, at most one (281567) is
even a candidate for π⁰-level recovery; the canonical 37112 is
irrecoverable by geometric splitting.**  The sec 9.9 prediction holds: the
defect lives upstream in shower building (the pr/123–125 over-clustering
threads).  No splitting knob will be designed; the probe stays as the
instrument for re-checking after any upstream fix.

## 10.5 Flip-equivalence and ship

`r3flipchk` (post-flip config, no env) hash-gated vs `r3flip` per sample:
**PASS 4/4 — 478/478 archives byte-identical** (132 mcp1k + 212 mcp2k +
38 ncpi0 + 96 nuecc48; logs `/home/xqian/tmp/pr132r3-gate-flipchk-*.log`).
The production flip is exactly the validated operating point.  Sentinel baseline for round 4 moves to the r3flip
manifests: **31 exact / 2 fakes / 82 groups / 32-of-109 rescan coverage**
(TSV `docs/pr/pr132-census-r3flip.tsv`).

## 10.6 Recommendations and the round-4 queue

| item | recommendation |
|---|---|
| K7+K8+K3=28 | SHIPPED this round (sec 10.2) |
| residual fake pair (76346 @29.6, 176502 @28.7) | K3 28→30 kills both; 30 also touches 76346's partial group — one owner Bee look settles it |
| K1 offset 5 | unchanged trade (+56243 +103798 / −283713 −506114), owner Bee call |
| K12 collinear merge | measured dead in v1 (sec 10.3); acceptance-aware v2 only if fragmentation stays the top blocker after the upstream threads |
| over-merge | CLOSED at the π⁰ level (sec 10.4): upstream thread owns it; re-probe after any shower-building fix |
| fragmentation (the 15/25 class) | the honest reading after 10.3+10.4: most of the miss population is upstream clustering shape, not pairing arithmetic — route to the pr/123–125 over/under-clustering threads with the π⁰ census as their new acceptance metric |
| path-2 partner floor / NC revival | parked until a sample with reachable NC π⁰ (sec 9.8) |
| census sentinel | every round: `pr132_pi0_census.py --fudge 0.84` vs the r3flip manifests, baseline 31 exact / 2 fakes / 82 groups |
| pairing pass rows 41–109 | still the cheap truth-set growth if round 4 needs statistics |

## 10.7 The owner Bee package (uploaded 2026-08-30, owner-requested)

`bee/pr132r3/` — annotated indexes `pr132r3.index.txt` /
`pr132r3-off5.index.txt` carry the per-event evidence.

- Flip A/B (13 events; idx 0-1 = the K3 28→30 adjudication pair, 2-4 = the
  K3=28 fake kills, 5-9 = the rescue exacts, 10-12 = the partial gains):
  OFF <https://www.phy.bnl.gov/twister/bee/set/0b40fe6e-ea4e-42d1-b89c-ad84bc65b128/event/list/>
  ON <https://www.phy.bnl.gov/twister/bee/set/109e5d8e-b900-43d2-8e4c-063714d61f83/event/list/>
- Offset-5 adjudication (5 events; 283713/506114 lost, 292524 survives,
  56243/103798 gained):
  A(off10) <https://www.phy.bnl.gov/twister/bee/set/a6bcbcdd-eb9c-4362-b9dc-ce0a2d1bbbf3/event/list/>
  B(off5) <https://www.phy.bnl.gov/twister/bee/set/bd9e48fb-238a-446e-bcee-44ccc65254d3/event/list/>

Set contents verified post-upload against the event lists (all 13 + all 5
present).

**What is NOT claimed.**  Same round-2 caveats: overlay gains are agreement
with the model pairing pass (labelsrc-separated); T_KINE stays
mass-window-free (the flip enlarges its candidate pool, a documented
mass-blind drift); the r3 arms use the merged-dir convention (98/141
disjoint, census/gates manifest-driven).
