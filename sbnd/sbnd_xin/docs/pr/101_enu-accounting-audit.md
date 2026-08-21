# doc pr/101 — Neutrino energy reconstruction: accounting audit round

**Why.** Owner request (2026-08-20, after the pr/99 round-3 Bee scan):
summarize the prototype's Enu recipe, compare/validate the toolkit's, hold
both to the paper principle

    E_nu^rec = sum_i ( K_i^rec + m_i + B_i )

(m only for mu±, pi±, e±; B = 8.6 MeV per proton; protons/neutrons never get
a rest mass; hadronic showers get KE from the dQ/dx→dE/dx sum and then mass
or binding by type), and **stop every remaining form of energy double
counting** — event 18259-37112's overlapping "proton shower" + "electron
shower" was the trigger.  In discussion the owner added: a long muon
(mu→mu→mu chain) takes **range** KE plus exactly one muon mass; no mass
double counting along a chain; no charge overlap EM↔EM, EM↔hadronic, or
shower↔track — and, explicitly, *"we do not like double counting, so we should
avoid that and improve"* beyond what the prototype does.

**Status.** SHIPPED, **SBND PRODUCTION ON** (toolkit `6bf0aafb`, 2026-08-20):
K1 `kine_charge_track_ctx`, K2 `kine_mass_rules`, K3 `kine_hadronic_dqdx`,
K4 `kine_long_muon_mode = 2`, K5 `kine_mainvtx_used_guard`.  Knob-off gate
234/234 archives PASS on three binary iterations; flip proofs 16/16 both
ways; zero nue selection flips, two NCπ0 numu flips attributed to K3 (§7.2).
Owner pre-authorised the flip ("these principles are all general").  Bee A/B
zips built (`bee/pr101/pr101x20-{off,on}.zip`, index `pr101.index.txt`),
upload on owner ask.

Companion rounds: pr/35 (energy-reco port fidelity audit, F1 `kine_shower_
pdg_live` ON), pr/99 round 3 (shower↔shower cell ownership `kine_charge_dedup`
+ `kine_charge_rebuild` + A5 hadronic re-type, ON).  This round is the next
layer on the same knob chain.

## Repro

```bash
# toolkit 6bf0aafb (code + flip), wcp-porting-img = the commit carrying this doc
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# event lists = the pr/99 r3 production arms' pr_evt dirs (48 + 19 + 35 + 15)
# knob-off gate (PR_JOBS=16 x2 concurrent arms = the owner's 32 CPUs)
PR_EXTRA_STAGES=pr_display PR_JOBS=16 ./run_pr_chain_batch.sh work-<s>-ql0819 work-pr101-off-<s> data <evts>
python3 scripts/pr85_hash_gate.py work-pr101-off-<s> work-pr99r3-onf-<s> --jobs 8
# all five knobs on
SBND_KINE_CHARGE_TRACK_CTX=1 SBND_KINE_MASS_RULES=1 SBND_KINE_HADRONIC_DQDX=1 \
SBND_KINE_LONG_MUON_MODE=2 SBND_KINE_MAINVTX_USED_GUARD=1 \
PR_EXTRA_STAGES=pr_display PR_JOBS=16 ./run_pr_chain_batch.sh work-<s>-ql0819 work-pr101-all-<s> data <evts>
# screens
python3 scripts/analysis/pr101/pr101_enu_census.py work-pr101-off-<s> work-pr101-all-<s> --out /home/xqian/tmp/pr101/census_<s>
python3 scripts/pr83r3_scores_ab.py work-pr101-off-<s> work-pr101-all-<s>
python3 scripts/pr93_shower_ab_diff.py work-pr101-off-<s> work-pr101-all-<s>
# compiled-config proof (knob off byte-identical vs HEAD cfg; knob on keys present)
#   /home/xqian/tmp/pr101/{head_off,work_off,work_on}.json -- see §6.0
```

## §1 The prototype recipe (read-only summary, `prototype_base/pid/src/`)

### §1.1 Three kinetic-energy estimators (per `ProtoSegment` / `WCShower`)

| method | code | what it does |
|---|---|---|
| range | `ProtoSegment::cal_kine_range` (ProtoSegment.cxx:1380-1418) | `TGraph::Eval(L/cm)` from `input_data_files/ave_range_to_kenergy.root` (graphs `electron muon pion kaon proton`, TPCParams.cxx:109-115).  No neutron/gamma table; unknown pdg falls back to muon (two overloads) or to a null graph (the `(double L)` overload used by showers).  The PSTAR provenance is not evidenced in the tree — only the binary ROOT file is. |
| dQ/dx sum | `cal_kine_dQdx` (:1316-1378) | per fit point: `dEdx = (exp(dQdx·23.6e-6·β/1.38/0.273) − α)/(β/1.38/0.273)`, α = 1.0, β = 0.255, clamp [0, 50] MeV/cm, `dQdx/43e3 > 1000 → 0`; `KE = Σ dEdx·dx`.  The ~6 mm "piece" is the fit spacing (`low_dis_limit = 0.6 cm`). |
| charge scaling | `NeutrinoID::cal_kine_charge` (NeutrinoID_energy_reco.h:44 shower, :275 segment) | scans the **whole event's** 2D charge maps, accepts a cell within 0.6 cm (2D projection) of the object's associated or fit cloud, per-plane sums → (U,V,W) weights 0.25/0.25/1 with the 0.04 asymmetry switch → `overall/recom/fudge·23.6/1e6 MeV`.  EM: recom 0.5 × fudge 0.8 ⇒ **×2.50** (the paper's factor); track 0.7×0.95; proton 0.35×0.95.  `fudge = 0.95` is the paper's data factor. |

### §1.2 Which estimator is used

* Track segment (`ProtoSegment::cal_4mom`, :1420-1433): `L < 4 cm` → dQ/dx;
  `flag_shower_trajectory` → dQ/dx; else range.  **No containment or
  exiting test exists anywhere** — the paper's "or any exiting particle"
  has no code counterpart; charge is never a track's best estimate.
* Shower (`WCShower::calculate_kinematics`, WCShower.cxx:339-527):
  connection type 1 (attached) → the 4 cm rule; other connection types
  with `flag_shower` → `kenergy_best = 0` ⇒ charge fallback
  (`fill_kine_tree` :47/:169/:218); disjoint pieces → charge fallback.
* Long muon (`calculate_kinematics_long_muon`, :288-336): dQ/dx
  **unconditionally** (:302, comment "should be improved in the future
  using range"); `vec_dQ/vec_dx` include the delta-ray members, `L` does
  not.

### §1.3 The event sum (`NeutrinoID_kine.h`)

`kine_reco_Enu = Σ kine_energy_particle + kine_reco_add_energy` (:250-257),
**no KE threshold, no fiducial test**; `kine_energy_included` is written but
never read.  Admission is graph reachability from the main vertex (main-
vertex pass :38-101, BFS :104-205) plus a trailing sweep of unreached showers
with `start_connection_type ≤ 3` (:209-247; type 4 = >80 cm detached ⇒
excluded entirely).  Shower members are pre-inserted into `used_segments`
(:25-27) so the BFS never emits them as separate particles (:130) — that is
the only anti-double-counting mechanism, and it is particle-level.

Rest terms:

| object reached how | pdg 2212 | pdg 11 | other (13, 211, …) |
|---|---|---|---|
| track, main-vertex pass / BFS (:93-97, :155-159) | +8.6 MeV | 0 | +mass |
| shower, main-vertex pass / BFS (:67-68, :187-188) | **+938 MeV** (bare `!= 11`) | 0 | +mass |
| shower, trailing sweep (:240-245) | +8.6 MeV only if start segment > 5 cm | 0 | **nothing** |

`flag_reduce` (:121-124, :194-200) refunds the previous segment's rest term
on a same-pdg or 211↔13 continuation — track→track only.

### §1.4 What the prototype does NOT protect

1. **Charge-level overlap**: `cal_kine_charge` has no cell ownership — two
   interleaved showers each collect the shared cells in full, and a proton
   track beside a shower is counted through its own range/dQdx entry **and**
   inside the shower's charge integral.
2. **Main-vertex gap**: the main-vertex track branch (:70-100) never
   checks `used_segments` (the BFS does at :130), so a shower member
   attached to the main vertex is emitted as a standalone track on top of
   its shower.
3. **Proton-typed showers** get the 938 MeV rest mass (table above).
4. Neutron (2112) is never assigned; no term exists for it.

## §2 The toolkit, side by side

| item | prototype | toolkit (HEAD 8573877f) | status |
|---|---|---|---|
| range / dQ/dx / charge arithmetic | §1.1 | `PRSegmentFunctions.cxx:2620-2648`, `cal_kine_dQdx` via `IRecombinationModel` (SBND `BoxRecombination` A 1.0 B 0.255 or `PowerBox`), `NeutrinoEnergyReco.cxx:47-190` | match (pr/35 §2) |
| estimator selection, tracks | `cal_4mom` | `segment_cal_4mom` :2847-2853 | match, no containment test either |
| estimator selection, showers | §1.2 | `PRShower.cxx:1471-1480` / `:1684-1696` | match |
| long muon | dQ/dx always | `PRShower.cxx:1771` | match (→ **K4**) |
| SBND operating point | 0.7/0.95, 0.5/0.8, 0.35 | recom 0.87, shower 0.58/0.8, proton 0.51 (`wct-pr-perevt.jsonnet:686-688`) | tuned (doc 68) |
| event sum | §1.3 | `NeutrinoKinematics.cxx:511-515` | match |
| proton shower +938 | :67-68 | `push_shower_kine` :154-161 | match (→ **K2**) |
| trailing-sweep 211/13 massless | :240-245 | `:451-457` | match (→ **K2**) |
| main-vertex `used_segments` gap | :70-100 | `:207-225` | match (→ **K5**) |
| shower↔shower charge overlap | none | `kine_charge_dedup` (pr/99 r3, ON) | toolkit ahead |
| shower↔track charge overlap | none | none | → **K1** |
| hadronic shower KE | charge / range by flags | same | → **K3** (owner principle) |
| `kine_reco_Enu` consumers | BDT | numu XGBoost var 69 + nue reader (pr/35) | every Enu move moves scores |

## §3 Event 37112 (NCπ0) — what the owner saw

Arms: `work-pr99r3-onf-ncpi0` (production) vs `work-pr99r2-off-ncpi0`
(pre-round-3; `kine` block identical to `work-pr99r3-off-ncpi0`, which has no
display dump).

* Shower 2 (`id 67048`, e, 16 segments, 114.6 cm): charge 635.2 → 504.6 MeV
  under the r3 dedup; its own dQ/dx sum is 356.8.
* Shower 3 (`id 9008`, **pdg 2212**, conn-2, 20 segments, 38.8 cm): a
  13.3 cm proton-like stem (ΣdQ 3.3e6, particle_score 0.35) plus ~18 EM
  fragments at z ≈ 92–140 cm.  Charge 386.1 → 242.6 MeV; its own dQ/dx sum
  is **496.8**.  It is the second π0 gamma with a proton-typed stem, not a
  proton.  Reached by the leftover pass ⇒ +8.6 MeV (not +938).
* Pair mass 135.8 → 95.9 MeV (opening angle 15.76°, bit-identical in both
  arms) ⇒ falls below the (100, 160) window (`NeutrinoShowerClustering.cxx:3858`)
  ⇒ pair dropped, `pio_id −1`, `f3_pi0 with:0`; `kine_pio_flag` stays 1
  because `pio_kine` is filled before the window.  Enu 1645.4 → 1307.7.
* Between the two showers, post-dedup double counting is **zero by
  construction** (winner-take-all).  What is *not* arbitrated: the 103 MeV
  proton track (range) and the 45 MeV pion track next to the pair — their
  cells within 0.6 cm of either shower cloud are still credited to the
  showers (§1.4 item 1 → K1).
* The 211→13 flip on the 45 MeV segment (−33.9 MeV = m_π − m_μ in
  `add_energy`) is not a PID pass: the pi0 incoming-track stamp
  (:3905-3935) only fires when a pair passes the window.  L0 log lines now
  make this visible.

## §4 Principles adopted for this round (owner, 2026-08-20)

P1 paper rule (m for μ/π/e, B = 8.6 per proton, nucleons never a mass);
P2 long muon: range over the chain + one muon mass (dQ/dx kept as the
documented fallback); P3 hadronic shower: Σ dE/dx + mass-or-binding by type,
**object-level** (PF/Bee/taggers see the same number — owner choice);
P4 no mass double counting along a chain; P5 no charge overlap EM↔EM (r3),
EM↔hadronic, shower↔track — ownership for **all non-member segments**
(owner choice); P6 37112 must close.

## §5 Knobs (all default OFF, `KineChargeOptions`, key-suppression idiom)

| knob | key | what ON does | anchors |
|---|---|---|---|
| K1 | `kine_charge_track_ctx` | every graph segment in no shower gets an ownership context (its `associate_points`/`fit` clouds) in the final owned scan; appended after the shower contexts (shower↔shower ties unchanged, shower/track equal-distance tie → shower); what a track wins is discarded (tracks stay range/dQdx valued).  Requires `kine_charge_dedup`. | `NeutrinoEnergyReco.cxx` `recompute_shower_kine_charge_final` (+`Graph&`) |
| K2 | `kine_mass_rules` | one rest-term table at all four add sites and the continuation refund: μ/π/K +mass, 2212/2112 +8.6, e 0; leftover (detached conn-2/3) showers: nucleons get binding behind the legacy 5 cm gate (2212 as before, 2112 added), μ/π stay massless as in legacy — the census showed every μ-typed leftover piece (12 numu events) sits beside an already-counted muon (§6.3) | `NeutrinoKinematics.cxx` `rest_term_rules` |
| K3 | `kine_hadronic_dqdx` | 2212/211/2112-typed shower-like objects (multi-segment or shower-flagged; a single-segment unflagged proton stub keeps the track rule), incl. A5 re-types, write `kenergy_best = kenergy_dQdx` (`Shower::set_kine_best`, new) | `apply_hadronic_dqdx_best`, called in `calculate_shower_kinematics` and after the A5 block |
| K4 | `kine_long_muon_mode` (+`_ratio_lo` 0.3, `_ratio_hi` 0.5) | 1 = range over the muon chain; 2 = range iff the far muon vertex is a graph dead-end and dQdx/range ∈ [0.7, 1.5], else dQ/dx | `PRShower.cxx` `calculate_kinematics_long_muon` (trailing params) |
| K5 | `kine_mainvtx_used_guard` | the main-vertex pass skips segments already owned by a shower (same guard as the BFS) | `NeutrinoKinematics.cxx` first pass |
| L0 | — | log-only: `pi0 window reject` / `pi0 incoming stamp` DEBUG lines | `NeutrinoShowerClustering.cxx` |

Census lines (DEBUG/INFO): `kine_track_ctx:`, `kine_mass_census:`,
`kine_mainvtx_guard:`, `kine_hadronic:`, `kine_long_muon:`; the existing
`kine final recompute:` line gains `track_ctx=`.  Note on K5's sibling
"shower continuation refund": already covered — `flag_reduce` is evaluated
on `curr_pdg` *before* the shower/track branch, and for a shower `curr_sg`
is its start segment, so no knob is needed.

Plumbing: `TaggerCheckNeutrino.{h,cxx}` (configure / default_configuration /
copy), `cfg/pgrapher/common/clus.jsonnet`, `sbnd/clus.jsonnet` (both
blocks ×2), `sbnd/wct-pr-perevt.jsonnet` TLAs, `run_pr_chain_batch.sh` env
hooks `SBND_KINE_CHARGE_TRACK_CTX / _MASS_RULES / _HADRONIC_DQDX /
_LONG_MUON_MODE / _LONG_MUON_RATIO_{LO,HI} / _MAINVTX_USED_GUARD`;
`doctest_clus_knob_defaults.cxx` +7 rows.

## §6 Gates and screens

### §6.0 Proofs that do not need an arm

* Compiled-config proof: `wcsonnet` of `wct-pr-perevt.jsonnet` with the
  production TLAs, HEAD cfg tree (`git archive 8573877f cfg`) vs the working
  tree, knob off: `cmp` **byte-identical** (`/home/xqian/tmp/pr101/{head_off,
  work_off}.json`); with the seven keys passed as TLAs all seven appear in the
  `tagger_check_neutrino` node (`work_on.json`).
* `./build/clus/wcdoctest-clus`: 2281 assertions PASS on every binary
  iteration (the 7 new CHECK_KNOB rows included).
* Freshness proof: `build/clus/libWireCellClus.so` mtime newer than the last
  source edit before every arm (wire-cell dlopens `build/<pkg>/`).

### §6.1 Knob-off gate (final binary)

Arms `work-pr101-off3-{nuecc48,ncpi0,mcp1k,mcp2k}` vs the production arms
`work-pr99r3-onf-*` (`pr85_hash_gate.py`, inner-member content hashes):

| sample | events | archives | result |
|---|---|---|---|
| nueCC48 | 48 | 96 | PASS byte-identical |
| NCπ0 | 19 | 38 | PASS byte-identical |
| numu50 / mcp1k | 35 | 70 | PASS byte-identical |
| numu50 / mcp2k | 15 | 30 | PASS byte-identical |


Three binary iterations were gated (off: first binary; off2: after the K3
scope + K2 5 cm gate refinement; off3: after the K2 leftover-μ/π rule) — all
PASS, 234 archives each.

### §6.2 All five knobs on (`work-pr101-all3-*` vs `work-pr101-off3-*`)

Per-sample Enu move (B − A, MeV) and screens (`pr101_enu_census.py`,
`pr83r3_scores_ab.py`; TSVs `/home/xqian/tmp/pr101/census3_<s>.tsv`,
`scores3_<s>.tsv`):

| sample | n | dEnu mean / median | q10 / q90 | min / max | moved (>1 MeV) | π0 pairs A→B | selection flips (numu 0.9 / nue 7) |
|---|---|---|---|---|---|---|---|
| nueCC48 | 48 | −33.3 / −7.8 | −91.9 / −0.6 | −248.2 / 0.0 | 42 | 9 → 9 | **none** |
| NCπ0 | 19 | −35.7 / −18.3 | −144.2 / 0.0 | −181.2 / +230.9 | 15 | 8 → 9 (+56982) | 285567, 506746 numu ↑ (§7.2) |
| mcp1k | 34 | −8.8 / −0.4 | −53.0 / +0.3 | −214.0 / +129.5 | 18 | 3 → 4 (+292643) | none |
| mcp2k | 13 | −13.1 / 0.0 | −57.9 / 0.0 | −71.7 / 0.0 | 6 | 1 → 1 | none |

* `kine_reco_add_energy` is **unchanged on every one of the 117 events**
  (K2 and K5 are latent here: 0 graph-reachable 2212 showers, 0 main-vertex
  guard skips; the leftover μ/π stay massless by the §7.3 rule).  So every
  Enu move is a kinetic-energy move: K1 charge ownership (down only), K3
  hadronic Σ dE/dx (down for A5 pion showers, up for 37112's proton-typed
  gamma), K4 long-muon range (down 2–19 %).
* Track contexts: 71–108 per sample; the charge a counted track wins
  divided by its own range/dQdx KE has median 0.69–0.77 (q10 0.33, q90
  1.1–1.2) — the exclusive-ownership charge estimator under-reads tracks by
  ~25 %, as expected for the track recom/fudge pair.
* Long muons: 5 genuine chains in numu50 — 278684 (L 167 cm, range 395,
  dQdx 448, ratio 1.13), 283713 (498 cm, 1139 / 1158, 1.02), 286191
  (111 cm, 274 / 340, 1.24) take range (dead-end far vertex); 54629 (125 cm,
  306 / 397, 1.30) has a degree-3 far vertex ⇒ dQ/dx; 281165 log line torn.
  Three further type-13 "showers" have an empty muon-segment set (L = 0)
  and keep dQ/dx.
* Hadronic showers written: 11 + 7 + 4 + 2; `dqdx/charge` 0.34–0.85 for
  the A5 pion objects, 1.29 for 37112's gamma, 1.35 for 292643, 0.74 for
  281837's 678 MeV conn-3 proton bundle.


### §6.3 Single-knob attribution (`work-pr101-{a,b2,c,d}-*`)

Each arm vs `work-pr101-off3-*` (same binary).  `b2` = K2+K5, `c` = K3,
`d` = K4 mode 2; `a` = K1 ran on the second binary against `off2` (K1 code
unchanged between the two).

| arm | knob(s) | moved events (of 114 with JSON) | dEnu range (MeV) | π0 pairs | selection flips |
|---|---|---|---|---|---|
| a | K1 track ownership | 39 + 15 + 13 + 5 | −214 … +17 (down only, bar the asym-switch few-MeV ups) | +56982, +292643 | none |
| b2 | K2 mass rules + K5 guard | **0** | 0 | — | none |
| c | K3 hadronic Σ dE/dx | 6 + 5 + 3 + 2 | −248 … +254 (37112 up) | — | 285567, 506746 numu ↑ |
| d | K4 long-muon range | 0 + 0 + 3 + 1 | −65.5 … 0 | — | none |

The all-knobs result is the sum of a + c + d to within the asym-switch
cross-terms; b2 is inert on this manifest (the +938 and main-vertex-member
classes do not occur in these 117 events — both knobs are correctness
guards that cost nothing ON).


### §6.4 Owner events

| event | what moved | A → B |
|---|---|---|
| **37112** (NCπ0, the trigger) | proton track (103 MeV, range) owns 59.6 MeV and the 45 MeV pion 22.4 MeV of charge the showers used to claim; shower 3 (2212-typed gamma) best energy charge 242.6 → Σ dE/dx 496.8; shower 2 504.6 → 511.3 (asym switch) | Enu 1307.7 → **1538.6**; pair mass (pairing reads charge) 95.9 → 96.6, still unpaired (§7.1); nue −2.83 → −2.50 |
| 168596 (pr/99 r3 trigger) | ownership vs the main-vertex tracks | Enu 2444.9 → 2402.2, nue kept |
| 315167 | A5 pion shower 405 (charge) → 304 (Σ dE/dx) … numu 2.23 → 2.33 | Enu 1442 → 1572 (the orphan proton's context returns charge) |
| 395148 | A5 pion shower 232 → 197 | Enu 870 → 979, numu 2.94 → 3.06 |
| 285567 | two A5 pion showers 156 → 63, 165 → 119 | Enu 2077 → 1936, numu 0.72 → **1.16** (flip ↑) |
| 506746 | A5 pion shower 102 → 57 | Enu 1971 → 1926, numu 0.52 → **1.11** (flip ↑) |
| 84229 | five pdg-11 non-member fragments (5–20 cm, counted as tracks) reclaim 176 MeV from the 991 MeV shower | Enu 1364 → 1183, nue 1.91 → 0.69 |
| 163543 | three A5 pion showers 455/144/320 → 291/68/136 | Enu 1497 → 1249 |
| 314838 | 3.9 cm conn-3 muon stub: would have earned 105.7 MeV before the K2 gate fix | Enu +2.9 only |



### §6.5 Production flip and proofs

`wct-pr-perevt.jsonnet`: `kine_charge_track_ctx = true`, `kine_mass_rules =
true`, `kine_hadronic_dqdx = true`, `kine_long_muon_mode = 2`,
`kine_mainvtx_used_guard = true` (ratio knobs at C++ defaults).  Proofs on
8 events (2 per sample: 37112 285567 / 168596 163543 / 315167 278684 /
70084 54629): bare config `work-pr101-flip-*` ≡ `work-pr101-all3-*` 16/16
archives; env-forced-off `work-pr101-floff-*` ≡ `work-pr101-off3-*` 16/16.
Compiled-config: bare compile carries the five keys; forced-off compile is
byte-identical to the HEAD-cfg compile.

## §7 Owner decisions / open items

1. **π0 pairing still reads `kine_charge`** (`id_pi0_with_vertex` caches
   `get_kine_charge()` at entry, prototype parity).  Under K3 the hadronic
   gamma in 37112 carries 497 MeV as its best energy but the pairing sees
   243 MeV ⇒ mass 96.6, pair still lost.  If the pairing read `kine_best`
   the pair would sit at m ≈ 137 MeV.  That is a further divergence (pairing
   energy ≠ charge) and is NOT implemented — owner call.  The same choice
   decides whether `kine_pio_energy_*` and the PF shower energy agree.
2. **A5-retyped pion showers under K3** lose 30–65 % of their energy
   (Σ dE/dx of a partly-EM cascade under the track recombination model,
   census `dqdx/charge` 0.34–0.85 over 20 objects).  Two NCπ0 events
   (285567, 506746) cross the numu 0.9 score threshold upward because of
   it.  Physically the paper's "hadronic shower = Σ dE/dx" rule; whether the
   A5 objects are hadronic enough for it is the EM-shower validation
   campaign's question.
3. **K2 leftover μ/π**: kept massless (legacy).  Twelve numu50 events carry
   a 30–119 cm conn-2 piece typed 13 beside an already-counted muon; giving
   it 105.7 MeV would be the P4 double count.  If the owner wants "every
   identified μ gets a mass", the gate to add is "no other 13-typed entry in
   the event", not a length.
4. **K5 and the +938 class are latent on these 117 events** (0 main-vertex
   guard skips, 0 graph-reachable 2212 showers).  Both knobs are correctness
   guards for topologies the samples do not contain; they cost nothing ON.
5. **K4 calibration** rests on 5 long muons (numu50 has few multi-segment
   muon chains): dead-end chains read dQdx/range 1.02–1.24 (range chosen),
   the one degree-3 far vertex (54629) falls back to dQ/dx; two further
   "long muons" have an empty muon-segment set (`L = 0`, nueCC48 30504 /
   235435, NCπ0 399860) and keep dQ/dx.  The [0.7, 1.5] window was not
   exercised at its edges.
6. **Production flip** — not done; §6 is the evidence for the owner's call.

