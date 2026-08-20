# doc pr/99 — owner scan triage: five events, three problem classes

Date: 2026-08-20.  Binary/config epoch: toolkit `f4b4d0ec` (fit_exclusion SBND
PRODUCTION ON, doc pr/98 §10).  All owner complaints below were raised against
the **post-flip** Bee sets built the same day (`scan-prodflip` 3617888b /
`pr96-prodflip` 059fcee2), i.e. against current production, including events
that earlier rounds marked FIXED.  This round is diagnosis + action items
only — **no production behavior change**; all reruns are log-only probes,
hash-gated against the production arms.

## Repro

```
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# probe reruns (log-only env probes; PASS byte-identical vs prodflip arms):
SBND_TRAJ_COVER_PROBE=1 WCT_PR96_REMSEG_DEBUG=1 WCT_SHOWER_ABSORB_DEBUG=1 \
WCT_SHOWER_CONTENT_DEBUG=1 WCT_DUP_STAGE_DEBUG=1 PR_JOBS=5 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp2k-ql0819 work-pr99-probe-mcp2k data 279955 70084
  ./run_pr_chain_batch.sh work-mcp1k-ql0819 work-pr99-probe-mcp1k data 395148 315167
  ./run_pr_chain_batch.sh work-ncpi0-ql0819 work-pr99-probe-ncpi0 data 285567
SBND_WCT_LOGLEVEL=trace WCT_SHOWER_CREATE_DEBUG=1 WCT_PR96_REMSEG_DEBUG=1 PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-mcp2k-ql0819 work-pr99-t70084 data 70084
# gates: python3 scripts/pr85_hash_gate.py work-pr96-prodflip-mcp2k work-pr99-probe-mcp2k   (PASS 4/4)
#        python3 scripts/pr85_hash_gate.py work-scan-prodflip-mcp1k work-pr99-probe-mcp1k   (PASS 4/4)
#        python3 scripts/pr85_hash_gate.py work-scan-prodflip-ncpi0 work-pr99-probe-ncpi0   (PASS 2/2)
# census:  python3 scripts/pr96_uncover_census.py <arm>/pr_evt<ID>
# panels:  python3 scripts/pr99_point_panels.py <armOFF> <armON> <outdir> <evt> <x> <y> <z> --box CM
# P3 scan: python3 scripts/analysis/pr99/pr99_transition_scan.py <arm> <evt> <shower_id>
```

Arms (A/B legs pre-existing, read-only): OFF = `work-{mcp2k,mcp1k,ncpi0}-prod0819`
(fit_exclusion=false), ON = `work-pr96-prodflip-mcp2k` /
`work-scan-prodflip-{mcp1k,ncpi0}` (post-flip production).

## 1. Scope and identities

| owner item | event (run/sample) | class |
|---|---|---|
| "missing a vertex track???" (-95.9, 136.2, 281.3) | 18255-**279955** (mcp2k) | P1 |
| "ISO PR weird" (-197.7, 14.5, 122.1) | 18255-**70084** (mcp2k; "0-" is slot notation) | P2 |
| "missing a track at vertex" (-153.0,-58.4,191.6), "two tracks merged to one", "295 MeV electron EM contains many tracks", ghost blue tracks ×2 | 18255-**395148** (mcp1k) | P1+P3 |
| "PR weird ... missing a track ISO" (52.6,-173.4,293.5) + (44.5,-168.2,314.9) | 18261-**285567** (ncpi0) | P1+P2 |
| "172 MeV electron hadronic shower" | 18255-**315167** (mcp1k) | P3 |

"28857" (the P2 "etc" example) — **owner confirmed it is 285567**; no sixth event.

## 2. Baseline: uncovered-charge census, OFF vs ON (pr96_uncover_census.py, production cuts)

| evt | OFF uncovered q% / track-like | ON uncovered q% / track-like | flagged group (ON) |
|---|---|---|---|
| 279955 | 13.9 / 1 | 15.2 / 1 | 21.7 cm, rms 0.43, dvtx 3.1, 5.2° to seg 16001 |
| 70084 | 19.1 / 1 | **3.4 / 0 (FIXED)** | — |
| 395148 | 7.8 / 1 | 7.8 / 1 | 13.0 cm, rms 0.74, dvtx 4.3, 10.9° to seg 21007 (pre-existed the flip) |
| 315167 | 16.8 / 1 | 16.6 / 1 | shower-region spread (P3 is classification, not coverage) |
| 285567 | 15.4 / 0 | **31.3 / 1 (flip REGRESSION)** | **52.0 cm**, 418 pts, rms 0.53, dvtx 3.6, **0.3° to seg 8105** |

## 3. 18261-285567 — P1: mvga op3.5 approach-collapse eats a two-prong V (NEW, flip regression)

**Symptom.** ON only: a 52 cm, 418-point, rms 0.53 uncovered track-like group
parallel (0.3°) to segment 8105 at 3.6 cm from the vertex; total uncovered
charge doubles (15.4% → 31.3%).  Panel
(`pr99_point_panels.py ... 285567 52.6 -173.4 293.5 --box 30`): the owner's
point sits on an OFF-arm fit; in ON both that corridor and the long parallel
corridor are bare charge.

**Root cause (probe-proven).**  `find_other_segments` round 2 *correctly*
finds the missing prong (`pr67 fos step8/9: group=2 npts=134 len=50.96 nnf=126
-> KEEP/SELECTED`), iso-snaps its near end to a junction 10.4 cm from the main
vertex — the graph then holds the true V.  Then `main_vertex_graph_audit`
**op3.5 approach-collapse** (doc pr/86 §15 R2) fires on that degree-2 junction
and replaces the two chain segments (16-fit stub + 96-fit corridor segment,
`PR96REMSEG` backtrace = `main_vertex_graph_audit <- TaggerCheckNeutrino::visit`)
with ONE straight Steiner chord:

```
mvga: op3.5 approach-collapse cluster=8 d=10.39cm chord=53.56cm npts=53
mvga: op3.5 approach-collapse cluster=8 d=6.18cm  chord=58.26cm npts=60   <- consumed its own first product
```

The second fire removes the first fire's product (`PR96REMSEG nfits=0` — a
fresh, not-yet-fitted segment) plus another stub: **op3.5 cascades on its own
creations** (they join `created`, which exempts them from op3's absorb but not
from op3.5 itself).  The final chord product is today's segment 8105; the op4
refit rides the chord line; the real dogleg prong's charge (sagging ~5 cm off
the chord mid-way) is left uncovered.  The straight-chain charge veto
(`straight_steiner_chain`, good_r=1.0 cm) passed because in this busy NCπ0
region the chord itself lies on (other prongs') charge — it never asks whether
the *removed path's* charge stays covered.

**Why it hid.**  op3.5's design envelope (pr/86 §15: "349945's 14 cm polyline
over 5.8 cm") is enforced only through the junction-to-vertex radius (15 cm).
There is **no cap on the chord it creates** and **no kink-angle test at the
junction**.  Population census (grep `op3.5 approach-collapse` over production
logs):

| arm | fires | med chord | p90 | max | >30 cm |
|---|---|---|---|---|---|
| work-mcp2k-prod0819 (1940 evts) | 46 | 15.7 | 33.4 | **338.5** | 8 |
| work-mcp1k-prod0819 (999) | 18 | 7.2 | 14.9 | 28.0 | 0 |
| work-nuecc48-prod0819 (48) | 8 | 14.0 | 24.9 | 28.8 | 0 |
| work-pr98-flip-nuecc48 (48) | 10 | 15.7 | 25.1 | 29.8 | 0 |
| scan/pr96-prodflip (9) | 4 | — | — | **146.2 (evt 315167!)** | 3 |

Top offenders: 242550 (mcp2k, **338.5 cm** chord, 395 pts), **315167 (146.2 cm
— the owner's P3 event, see §6)**, 475140 (138.2 cm).

**Status: RESIDUAL — action item A1.**

### 3b. 285567 second point (44.5,-168.2,314.9) — P2 "iso PR weird"

The junction at (44.0,-172.4,312.0) (a satellite vertex ~45 cm from the main
vertex) carries a micro-segment braid: 8099 (0.7 cm, 2 fit points), 8109
(1.9 cm), 8005 (4.7 cm) plus the pr54 keep-isolated 12.4 cm segment
(`pr54 keep-isolated: cluster 8 n_points=27 length=12.40`).  This is pr/51's
class-(c) micro-stub zoo at a junction **outside mvga's near-vertex scope**
(mvga_radius 15 cm; op3's satellite reach `mvga_satellite=3 cm` doesn't extend
here) — same structural gap as §5's triangle: the audits never look far from
the main vertex.  Cosmetic relative to §3; folded into A2's scope question.

## 4. 18255-395148 — P1: FOS 2D-shadow drop; P3: a charge-negative ghost + a sub-floor proton

### 4a. P1 "missing a track at vertex / two tracks merged to one"

The 13 cm track-like uncovered group at dvtx 4.3 cm, 10.9° to segment 21007,
**pre-exists the flip** (present in both arms, cen (-153.4,-64.4,191.6) ≈ the
owner's point).  Probe: FOS *does* see the residual, but what survives step-1
tagging is a 3-point crumb dropped as **`nnf0_2d_shadowed`**
(`pr67 fos step8: cluster=21 group=1 npts=3 len=4.72 nnf=0 -> DROP`):
a prong nearly parallel and close to the already-fitted main track has its 2D
wire projections shadowed in the views, so `number_not_faked` = 0 and the
point floor never even gets asked.  This is 279955's disease at a different
seat — the 2D-projection ambiguity again, with no 3D
does-anything-else-explain-this-charge test.  **Status: RESIDUAL — action
item A4** (and the same event feeds A2/A3 below).

### 4b. P3 "295 MeV electron EM contains many tracks"

Shower 21006 (E=295.6 MeV, L=51.9 cm, 4 members) decomposes
(calib + `T_rec_charge` per `real_cluster_id`):

| member | len | pdg (score) | qmed [e/pt] | frac(q≤0) | verdict |
|---|---|---|---|---|---|
| 21006 (start) | 11.9 | 11 (0.18) | 5722 | 0.00 | heavy stub (~2.4×MIP) |
| 21007 | 14.7 | **2212 (0.17)** | 8358 | 0.04 | **confident proton, under the 50 cm guard floor** |
| 21010 | 23.4 | 11 (100) | **−678** | **0.93** | **charge-negative projective ghost** (§4c) |
| 21026 | 1.9 | 13 (100) | 380 | 0.50 | crumb |

Two independent defects: (i) the pr/93 `shower_accept_pid_guard` /
vote-guard family is floored at `shower_pid_guard_min_len = 50 cm`, so the
14.7 cm confident proton counts into EM length/energy — exactly the gap
hypothesized when pr/93 shipped (its §6 regression history is why the floor
exists; see A5 for the safe alternative); (ii) 45% of the declared length is
the ghost 21010.  Note the leverage: remove the ghost and the *existing*
`update_particle_type` vote already flips the object non-EM
(track 16.6 cm vs shower 11.9 cm).  **Status: action items A2/A3 (+A5).**

### 4c. "Ghost blue tracks"

Point A (-139.4,-20.6,267.4): the pair is 21008 (65.7 cm μ, score 0.036,
qmed 2403) and 21010 (23.4 cm, shower member).  Neither is fit-in-void (all
fit points ≤0.6 cm from charge), but 21010 is the **textbook pr/83 §11.2
projective ghost**: 3-D overlap 0.28@1.4 cm, per-view 2D overlap
**V=1.00, W=1.00, U=0.25** (the divergence lives in one view), and it is
charge-starved to the point of *negative* median charge (−678 vs the muon's
+2403; pr/83's shipped discriminator `mvga_proj_dqdx_ratio=0.55` would kill it
instantly).  It survives because **op1-proj runs only inside mvga's
main-vertex scope** and this pair sits ~95 cm from the vertex — inside a
shower's membership, where no audit ever looks.  **Status: action item A3.**

Point B (-195.9,-111.9,243.1): only a 0.35 cm 2-point stub (cluster 43) plus
on-charge fit points — nothing ghost-like in the dump; likely a Bee rendering
of a satellite micro-segment.  No action.

## 5. 18255-70084 — P2 "ISO PR weird": a charge-starved chord closes a false triangle

Coverage is FIXED by the flip (19.1% → 3.4% q, pr/96-prodflip idx 1) — the
owner's residual complaint is topology, and it is real.  The ON graph around
the vertex is a **closed triangle**: vertex →20034(13.0 cm)→ J1 →20022(9.4 cm,
the recovered prong)→ J2 →20035(15.7 cm)→ back to the vertex.  Charge content:

| leg | qmed [e/pt] | frac(q≤0) | verdict |
|---|---|---|---|
| 20034 vertex→J1 | 2353 | 0.17 | real (MIP) |
| 20022 J1→J2 | 4879 | 0.18 | real (the pr/96/98 recovery) |
| 20020 J2 onward | 4981 | 0.00 | real |
| **20035 J2→vertex** | **440** | **0.41** | **charge-starved chord (~0.2 MIP)** |

The truth is one bent prong vertex→J1→J2→onward; 20035 is a straight
vertex→J2 shortcut carrying ~nothing.  The trace rerun (`work-pr99-t70084`)
shows the audit **did see it and declined**:

```
mvga: op1-post eval cluster=20 pair len 12.97/15.69cm npts 23/27 overlap=0.87
```

overlap 0.87 ≥ the 0.7 production gate, but the pair's chord opening angle
(~30°) trips the `mvga_dup_angle=20°` near-parallel guard — and had it merged,
the *length* rule would have deleted the shorter REAL prong 20034 and kept the
ghost.  op2 (charge-less bridge removal, the exact tool: ratio 440/MIP ≈ 0.18)
evaluated cluster 20 before the chord existed (its four `op2 eval` lines show
only pre-triangle segments; the chord appears between the vertex-activity
refit at J2 and shower creation, where 20034/20035 surface as shower start
segments).  The charge second-opinion is missing at the seat that does fire
(op1-post), and the seat that has the charge rule (op2) runs too early.
**Status: action item A2.**

## 6. 18255-315167 — P3 "172 MeV electron is hadronic": measured, and op3.5 strikes again

Shower 8016 (E=172.7 MeV, L=69 cm, 3 members, `trk_frac=0` — **no aggregate
track-vote scalar can catch it**, confirming pr/93 §3): stem 8016 15.7 cm
qmed 2783; trunk 8006 44.8 cm qmed 3227 (µ/π-like, flat); stub 8008 8.5 cm
qmed 6936 (proton-like) with pdg 11 (score 0.30).

**The trunk is an op3.5 product**: `mvga: op3.5 approach-collapse cluster=8
d=7.67cm chord=146.25cm npts=164` — a 250-fit-point curved trajectory
((119.7,184.7,364.6)→(138.1,160.5,227.5)) plus a 7.7 cm vertex stub replaced
by ONE straight 146 cm chord (§3's defect, same seat, #2 offender in the
population).  The event's 19 uncovered groups (16.6% q) are the real curved
path off the chord.

**Transition-scan measurement** (the standing pr/93 §3 "never done"
measurement, script `scripts/analysis/pr99/pr99_transition_scan.py`): walking
the shower's own trajectory in 3 cm bins with an 8 cm charge cylinder:

- **315167/8016**: in-cylinder charge population *shrinks* downstream
  (727→~150 pts/bin; first-30cm growth ≈ 0.2×) and the trunk ends in a
  **terminal Bragg rise** (dqdx_med 2930→4596→6716→10801 over the last 9 cm)
  — a stopping hadron, not an electron.
- **395148/21006**: growth ≈ 0.5×, plus an unmistakable charge-negative tail
  (the ghost member's bins all read qmed −258…−1020).
- Controls **168596/14153** (real 2016 MeV e⁻) and **360535/7060** (real
  1944 MeV e⁻): growth over the first 30 cm = 5.7× and 3.1× (300→1722,
  389→1210 pts/bin).
- Adverse control **506114/19016** (pr/93's named genuine gamma with an 11 cm
  trunk): growth 2.3× — correctly on the electron side, while its stem dQ/dx
  (5891, pair conversion) would have failed any dQ/dx-only gate, exactly as
  pr/93 warned.

So a **downstream-charge-growth scalar separates these hadronic showers from
real EM by ~5-10× with the named adverse control surviving** — the first
measured candidate for the standing pr/80 R4 "no hadronic-shower tag" gap.
n=5; needs the calibration campaign (A5).  **Status: action items A1
(the trunk), A5 (the tag).**

## 7. 18255-279955 — P1: unchanged, known residual (pr/96 §3)

Post-flip probe confirms the pr/96 mechanism verbatim: `mvga: op1 dup-merge
cluster=16 removed seg len=27.34cm sumdQ=1.1e+06 overlap=0.89@14.0mm vs
survivor len=220.41cm sumdQ=1.19e+07` — the prong's interior fit collapses
onto the 11×-brighter muon and op1 deletes it as a duplicate; census 13.9% →
15.2% q.  pr/96 §7's F3 (keep the loser) stays recommended-against; the new
angle this round adds is A2's *charge-aware replacement* (re-anchor the loser
onto the charge the survivor does not explain) rather than keep-vs-delete.
**Status: RESIDUAL — A2 is the (hard) lever; fit-follows-its-own-charge
remains the underlying disease (pr/96 §10.1).**

## 8. The unifying reading

Every seat in §§3-7 fails the same way: a graph edit (collapse, merge, drop,
decline) is decided from **geometry proxies only** — overlap fraction, chord
angle, point floors, 2D shadowing — and never asks the two questions the data
can answer directly: **(i) does the candidate/loser explain charge nothing
else explains, and (ii) does the survivor/product carry the charge it claims**
(qmed vs MIP, frac(q≤0)).  pr/96 §7 said this for admission/dup gates; this
round shows it also holds for op3.5's chord veto, op1-post's angle guard, and
shower membership.  Additionally, the audit family is **main-vertex-scoped
and runs once**, so late creations (vertex-activity chords, shower-stage
segments) and vertex-remote structures (shower members, satellite braids)
are never audited at all.

## 9. Action items (ranked)

| # | action | seat | reaches | risk / constraint | validation |
|---|---|---|---|---|---|
| **A1** | **op3.5 chord cap + junction-kink guard + no self-cascade.**  Default-OFF knob (e.g. `mvga_ac_chord_max`, suggested 2× `mvga_approach_collapse`; plus decline when the removed chain's kink angle at the junction exceeds a threshold, and skip members of `created`) | `NeutrinoGraphAudit.cxx:1058-1125` | 285567 P1 (52 cm loss), 315167 trunk (146 cm chord), 242550/475140-class; 46 fires in mcp2k | design case (349945, 5.8 cm chord) must keep firing; gate + census + panels on mcp2k top-offender list | knob-off byte-identical gate; knob-on census Δ on {285567, 315167, 242550, 475140} + full-sample score screen |
| **A2** | **Charge second-opinion at the audit ops.**  (a) op1/op1-post: when overlap ≥ gate but the angle guard declines, kill the member whose qmed/MIP ≤ ~0.25 (70084's 20035 at 0.18 dies; both-real pairs untouched); when merging, pick the winner by charge, not length.  (b) op3.5's veto: require the removed path's charge to stay covered by the product.  (c) 279955-class: instead of delete-vs-keep, re-anchor the op1 loser onto charge the survivor does not explain (pr/96 §3.1's chord read) | `NeutrinoGraphAudit.cxx` op1 :317-461, op1-post :1157-1250, op3.5 veto | 70084 P2 (triangle), 279955 (hard case), §3b braid | pr/96 F3 lesson: never *keep* a geometric duplicate — A2 only *deletes* charge-starved members or re-anchors, the safe direction.  MIP scale from `m_mip_dqdx_median` (op2 already owns it) | same gate + the pr/96 census on the 69-group population; hand-check 10 op1-post declines |
| **A3** | **Extend the ghost kill past the main-vertex scope.**  Apply the op1-proj discriminator (per-view 2D overlap + `mvga_proj_dqdx_ratio`) to shower *membership* at `update_particle_type` / absorb time, or run a vertex-remote dup audit over shower members; a cheap first version: a member with frac(q≤0) > 0.5 never counts toward shower length/energy | `PRShower.cxx:1086` vote + `NeutrinoShowerClustering.cxx` absorb sites | 395148 P3+ghost A (kills 21010 → the existing vote then flips the 295 MeV object non-EM on its own) | must not touch genuine low-dQ/dx shower fragments — gate on *negative*/starved medians, not merely low; screen with `pr93_shower_composition.py` on nueCC48+NCπ0 | knob-off gate; nueCC48 62-shower + NCπ0 35-shower regression screen (pr/93's own harness) |
| **A4** | **FOS 2D-shadow override for near-vertex track-like residuals.**  When a residual component fails `nnf` (2D-shadowed) but is large in 3D (≥N pts), straight (rms floor), and starts within ~15 cm of the main vertex, admit it through a 3D charge-explanation test instead of the 2D one | `NeutrinoOtherSegments.cxx` step-8 quality cut (:415-423) | 395148 P1 (13 cm prong), part of pr/96's 59-group population | nnf exists to kill isochronous fakes — scope tightly (dvtx, straightness, min pts); pr/67 §11.7's non-monotonicity warning applies to anything touching iso admission | knob-off gate; pr/96 census before/after on all four samples; iso_band/isochronous negative controls |
| **A5** | **Hadronic-shower tag from downstream charge growth.**  Calibrate the §6 scalar (first-30 cm in-cylinder growth ratio; + frac(q≤0) member veto; + optional terminal-Bragg check) over pr/93's 102-shower composition TSV + these cases; ship first as a *log-only diagnostic* on the nusel row, then as a guard predicate replacing the blunt 50 cm floor where a growth verdict is available | new predicate near `PRSegmentFunctions.h:491` family; scan script `scripts/analysis/pr99/pr99_transition_scan.py` | 315167, 395148 P3; the standing pr/80 R4 gap; the sub-50 cm confident-proton gap without re-opening pr/93 §6's regressions | pr/93 §6 history: un-floored PID guards regressed 23/48 nueCC48 — the growth scalar must be validated on exactly that set first; 506114-class gammas are the adverse control (passes at n=1 here) | scan TSV over all pr/93 rosters; blind re-run of the pr/93 §6 regression list |
| A6 | Bookkeeping: "28857" = 285567 (resolved); ghost point B benign; op2's early-run vs late-creation ordering noted in A2(b)'s design | — | — | — | — |

Recommended order: **A1 first** (small, self-contained, measurable on named
events, and it removes the flip regression on 285567), then A2(a)+A3 (same
discriminator, two seats), then A4, with A5's calibration running as the
background campaign.

## 10. Artifacts

- Probe arms: `work-pr99-probe-{mcp2k,mcp1k,ncpi0}` (log-only, hash-gated
  PASS vs prodflip arms: 4/4, 4/4, 2/2), `work-pr99-t70084` (trace-level).
- Scripts (this round): `scripts/pr99_point_panels.py` (owner-point A/B
  panels, fork of pr98_fit_panels.py), `scripts/analysis/pr99/
  {pr99_transition_scan.py, pr99_seg_near.py, pr99_ghost_check.py,
  pr99_seg_overlap2d.py}`.
- Panels: `/home/xqian/tmp/pr99/panels/pr99_evt*.png` (regenerate via Repro).
- Owner Bee sets referenced: OFF-epoch e1357e60 / post-flip scan-prodflip
  3617888b, pr96-prodflip 059fcee2 (built by the concurrent session, arms
  read-only here).
- Prior docs built on: pr/96 (§3 op1 deletion, census), pr/54 (isolated
  residual), pr/86 §15 (op3.5), pr/83 §11.2 (projective ghosts), pr/93
  (EM-vs-track guards + §3 open design), pr/94 §9.10 (395148's earlier
  symptom), pr/51 (285567 zoo), pr/98 (fit_exclusion flip).

---

## Round 2 (2026-08-20) — implementation: A1 + A2 + A3, validated, SBND PRODUCTION ON

Owner instruction: implement the round-1 action items (scope answer: A1+A2+A3),
validate on nueCC48 + NCπ0 + a new ~50-event numu manifest built from past
PR-doc events (cathode-crossing topic excluded), and flip ON if validation
passes.  All shipped as default-OFF C++ knobs; the flip lives only in
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`.

### Repro block

```
cd wcp-porting-img/sbnd/sbnd_xin           # all arms: PR_JOBS=32, PR_EXTRA_STAGES=pr_display, reality=data
# numu50 standing manifest: scripts/manifests/numu50.txt (35 mcp1k + 15 mcp2k)
# pre-edit baselines at HEAD f4b4d0ec (proven ≡ production prodflip arms):
#   work-pr99r2-base-{ncpi0,mcp1k,mcp2k}; nueCC48 baseline = work-pr98-flip-nuecc48
# knob-off gate arms (final binary): work-pr99r2-off3-{nuecc48,ncpi0,mcp1k,mcp2k}
python3 scripts/pr85_hash_gate.py work-pr99r2-off3-<s> <baseline>   # 96+38+70+30 all PASS
# knob-on arms (final operating point): work-pr99r2-on3-* with
#   SBND_MVGA_AC_CHORD_MAX=30 SBND_MVGA_AC_NO_CASCADE=1
#   SBND_MVGA_DUP_STARVED_ASYM=0.55 SBND_MVGA_DUP_STARVED_MIP=0.8 SBND_MVGA_DUP_STARVED_SPAN=0.5
#   SBND_SHOWER_GHOST_MEMBER_DROP=1
python3 scripts/pr83r3_scores_ab.py work-pr99r2-off2-<s> work-pr99r2-on3-<s>
python3 scripts/pr93_shower_ab_diff.py work-pr99r2-off2-<s> work-pr99r2-on3-<s>
# flip proofs (13-event subset): work-pr99r2-flip-* ≡ on3; work-pr99r2-floff-* ≡ baselines
# panels: scripts/pr99_point_panels.py work-pr99r2-base-<s> work-pr99r2-on3-<s> <out> <evt> <x> <y> <z>
```

### Knobs (all C++ defaults OFF = byte-identical; production values below)

| knob | seat | prod | mechanism |
|---|---|---|---|
| `mvga_ac_chord_max` | op3.5, `NeutrinoGraphAudit.cxx` | **30 cm** | decline a collapse whose replacement chord \|vtx1−vtx2\| exceeds the cap.  Kills the off-envelope population (285567 53.6/58.3, 315167 146.2, 242550 338.5 cm; design case 5.8 cm keeps firing) |
| `mvga_ac_no_cascade` | op3.5 | **true** | never collapse a candidate whose sg1/sg2 is a `created` product (285567's second fire consumed its own nfits=0 chord) |
| `mvga_ac_veto_radius` | op3.5 charge veto | **OFF** | dedicated collapse-chord is_good_point radius.  Measured ADVERSE at the prototype 0.2 cm — kills the 349945 design case and benign small collapses (359980 vtx moved 34 cm) — re-confirming pr/86 Stage A's deliberate 1.0 cm relax (M15: the divergence was documented).  Knob retained for future scans |
| `mvga_dup_starved_asym` | op1-post angle-decline | **0.55** | pair min/max median-dQ/dx asymmetry.  The refit SPLITS corridor charge across a duplicate pair (70084: 1.16/0.62 — an absolute MIP floor can never fire post-refit); asymmetry is the op1-proj-shaped discriminator (production op1-proj gate is also 0.55) |
| `mvga_dup_starved_mip` | op1-post | **0.8** | the same threshold separates the pair BOTH ways: loser ≤ 0.8 ≤ survivor (a proton+MIP V's muon reads ~1.0 → protected; a 0.61-ratio "survivor" carries no verdict) |
| `mvga_dup_starved_span` | op1-post | **0.5** | pair min/max LENGTH comparability: a projective duplicate shares its whole span (70084: 0.83); a track paired with its own Bragg stub does not (138009: 0.15) |
| `shower_ghost_member_drop` | new pass in `shower_clustering_with_nv`, before `id_pi0_with_vertex` | **true** | drop a charge-starved shower member (median ratio ≤ 0.25 OR frac(dQ≤0)>0.5, len ≥ 10 cm) whose fit points are 2D-shadowed in ≥2 of 3 wire views (ov[1] ≥ 0.7, tol = mvga_dup_tol) by a healthy (≥ 2×0.25) partner segment ANYWHERE in the graph, same (apa,face).  View removal (leaf-only strand guard, new `Shower::drop_ghost_member`, forked from detach_track_prefix) + graph deletion (leaves the PF/Bee display) + vote/kine/charge recompute + maps rebuild |
| `shower_ghost_{overlap_frac,dqdx_ratio,min_len}` | — | C++ defaults 0.7 / 0.25 / 10 cm | thresholds, inert while the bool is off |

Runner hooks: `SBND_MVGA_AC_{VETO_RADIUS,CHORD_MAX,NO_CASCADE}`,
`SBND_MVGA_DUP_STARVED_{ASYM,MIP,SPAN}`, `SBND_SHOWER_GHOST_{MEMBER_DROP,
OVERLAP_FRAC,DQDX_RATIO,MIN_LEN}` in `run_pr_chain_batch.sh`.

### Design corrections measured during the campaign (3 iterations)

1. **A1 veto-radius retreat.**  Campaign 1 set the collapse veto to the
   prototype's 0.2 cm ("prototype parity").  It killed the 349945 design
   case (enu 1109→419, numu 2.47→0.22) and benign small collapses (359980:
   a 9.3 cm collapse vetoed, vertex moved 34 cm) — exactly what pr/86
   Stage A measured when it deliberately relaxed the radius to 1.0 cm.
   Charge-veto declines log at TRACE, so the first firing census missed
   this entirely.  Production keeps the radius knob OFF; the chord cap +
   no-cascade alone remove every owner-scan monster.
2. **A2 absolute-MIP form dead on arrival.**  The round-1 diagnosis quoted
   qmed 440 ≈ 0.18 MIP for 70084's chord — that was DISPLAY charge
   (T_rec q).  In fit space the refit splits the corridor charge: the pair
   reads 1.16/0.62, so "starved ≤ 0.25" never fires.  Redesigned to pair
   asymmetry (0.53 ≤ 0.55).
3. **A2 span guard + survivor floor.**  The asym form alone fired on
   31/117 events and LOST two nueCC48 nue selections: 138009 (nue 4.3→−15)
   deleted a 21 cm MIP electron stem paired with its own 3.2 cm Bragg-peak
   stub (span 0.15), 489330 (4.3→−15) a 14.8 cm limb vs a 6.4 cm spur.
   `mvga_dup_starved_span=0.5` + the survivor floor (hi ≥ 0.8) restore
   both; final footprint 5/117 events, all span-comparable corridor pairs.
4. **A3 partner is NOT a member.**  395148's ghost 2D-shadows the 65.7 cm
   track seg 21008 (non-member, ratio 1.32) at views 1.00/1.00/0.25 — the
   member-pair-only scan never fires.  Partner pool widened to the whole
   graph; the candidate stays member-only.  Also: the ghost's fit-space
   signature is starved ratio 0.11 with frac(dQ≤0)=0.20 — the NEGATIVE
   display-charge reading lives in T_rec space; the starved-ratio arm of
   the test is what fires.

### Gates and proofs (final binary, labels)

- Baselines ≡ production: `work-pr96-prodflip-mcp2k` 4/4,
  `work-scan-prodflip-mcp1k` 10/10, `work-scan-prodflip-ncpi0` 2/2 PASS.
- Knob-off byte-identical: `work-pr99r2-off3-{nuecc48 96/96, ncpi0 38/38,
  mcp1k 70/70, mcp2k 30/30}` — ALL PASS (pr85_hash_gate vs the baselines).
- Compiled-config proofs (M6): off-arm `.wct-cfg` diff vs base = arm-path
  lines only; knob-on `.wct-cfg` carries exactly the passed keys.
- `wcdoctest-clus`: 215 cases / 2243 assertions PASS (knob-defaults doctest
  extended with all 10 new keys).
- Flip proofs (13-event subset: 138009 235435 168596 / 285567 / 315167
  395148 349945 55595 / 70084 279955 242550 475140 54629):
  flip-bare ≡ explicit-on3 PASS 6+2+8+10 archives; forced-off ≡ pre-flip
  baselines PASS 6+2+8+10 archives (work-pr99r2-flip-* / work-pr99r2-floff-*).

### Owner-event outcomes (off2 → on3)

| evt | knob(s) fired | outcome |
|---|---|---|
| 285567 | chord-cap ×1, no-cascade ×2 (2 small collapses still fire) | the 53.6/58.3 cm fake chords declined; census track-like uncovered group (52 cm, rms 0.53) clears (TRACKLIKE 1→0, q 31.3→30.5%).  Residual ~30% uncovered is diffuse shower spread + the 279955-class interior fit collapse (FOS still SELECTs the prong, its segment 8105 exists in BOTH arms; the ribbon rides 4–7 cm off the fit) — pr/96 §10.1 territory, not op3.5 |
| 315167 | chord-cap ×1 | the 146.25 cm chord (the fake 44.8 cm EM-trunk source) declined.  The 172 MeV object remains typed 11 (the vote only ever writes 11) — typing stays with A5, as designed.  enu 1180→1305 |
| 70084 | starved-override ×1 | the 15.69 cm chord (0.62 vs 1.16, overlap 0.87 @ 30°) deleted; triangle OPEN, real chain 20034→20022→20020 intact.  enu 598→543 |
| 395148 | ghost drop ×1 | ghost member (23.4 cm, ratio 0.11, views 1.00/1.00/0.30, partner 1.18) removed from shower AND graph — the owner's "ghost blue track" point A is gone from the display.  Shower 4→3 members, kine_best 295.6→87.6 MeV, kine_charge 295.6→231.8; enu 940→732.  Object keeps its 11 label (vote never unwrites) — the "many tracks" energy inflation is fixed, the label is A5's |
| 279955 | none | unchanged, as expected (pr/96 §10.1 blocker stands) |

Panels (pr99_point_panels.py, base vs on3): `docs/pr/99_r2_evt{285567_p1click,
285567_prong,70084_triangle,395148_ghostA,395148_p1,279955_p1}.png`.

Bee A/B (12 events, owner 5 first then movers; `bee/pr99r2/pr99r2.index.txt`):
OFF 0f1b95f2 <https://www.phy.bnl.gov/twister/bee/set/0f1b95f2-b3ed-4aa1-99b9-6964bd324017/event/list/> /
ON 70f2fdbc <https://www.phy.bnl.gov/twister/bee/set/70f2fdbc-7aae-400d-8f38-2d188e6346e2/event/list/>.

### Campaign screens (117 events; off2 vs on3)

- Firing footprint: chord-cap 5 evts, no-cascade 6, starved-override 5,
  ghost drop 1 (= 395148 only).
- Score movers 10 total, every one attributed:
  nueCC48 4 — 46363/433451/168596 (span-comparable starved-dup kills;
  nue selections unchanged; 168596 enu 2619→3533), 235435 (no-cascade ×10;
  nue already −4.2 pre-knob, → −15 sentinel; enu +67).
  **No nue selection lost** (campaign-1's 138009/489330 losses recovered).
  NCπ0 2 — 56982 (nue −3.6→−4.3, good direction), 285567 (nue −2.6→−1.6,
  still far from selection).
  numu50 4 — the owner events (315167, 395148, 70084) + 475140, which
  GAINS its numu selection (numu −0.62→1.25) after the 338 cm-class
  chord-cap decline.
- pr/93 shower harness: no shower disappears, no type flips; member-energy
  shifts of a few MeV confined to the dup-kill events; adverse gamma
  506114 untouched.

### numu50 — standing numu validation manifest

`scripts/manifests/numu50.txt`: 35 mcp1k (pr93r3 standing scan set + owner
events + 349945) + 15 mcp2k (owner events, op3.5 offenders, doc 90/94/96
cases).  Baselines `work-pr99r2-base-{mcp1k,mcp2k}` at production HEAD are
the standing comparison arms going forward.

### Production flip

`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`: `mvga_ac_chord_max=30`,
`mvga_ac_no_cascade=true`, `mvga_dup_starved_asym=0.55`,
`mvga_dup_starved_mip=0.8`, `mvga_dup_starved_span=0.5`,
`shower_ghost_member_drop=true`.  C++ defaults stay false/0.

## Round 3 (2026-08-20) -- 168596 Enu double count + A5 hadronic-shower tag

Owner (round-2 Bee scan): (1) *"for 168596, I feel there are double counting
on the electron EM shower"* -- confirmed and fixed; (2) *"many of the
hadronic shower, or particle flow are still labeled as electrons"* = A5,
implemented; (3) same validation as round 2, flip if passed.

### Repro

```
cd wcp-porting-img/sbnd/sbnd_xin     # toolkit at 8573877f; PR_JOBS=16 (shared box, see incident note)
# knob-off gates (vs round-2 production arms; PR_EXTRA_STAGES unset):
./run_pr_chain_batch.sh work-<s>-ql0819 work-pr99r3-off-<s> data <evts>      # s in {nuecc48,ncpi0,mcp1k,mcp2k}
python3 scripts/pr85_hash_gate.py work-pr99r3-off-<s> work-pr99r2-on3-<s> --jobs 8
# knob-on (final operating point) + displays:
SBND_KINE_CHARGE_DEDUP=1 SBND_KINE_CHARGE_REBUILD=1 SBND_SHOWER_HADRONIC_TAG=1 \
SBND_SHOWER_HADRONIC_GROWTH_MAX=0.7 SBND_SHOWER_HADRONIC_STEM_RATIO=2.8 \
PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-<s>-ql0819 work-pr99r3-onf-<s> data <evts>
# A5 calibration table (109 showers) + screens:
python3 scripts/analysis/pr99/pr99_hadronic_calib.py work-pr99r2-on3-{nuecc48:nuecc48,...}
python3 scripts/pr83r3_scores_ab.py  work-pr99r2-on3-<s> work-pr99r3-onf-<s>
python3 scripts/pr93_shower_ab_diff.py work-pr99r2-on3-<s> work-pr99r3-onf-<s> --sample <s>
```

### 1. 168596 -- the double count is the ENERGY ACCOUNTING, and it is generic

Anatomy (round-2 arms): the op1-post starved-dup deletions re-routed the
shower-seed BFS (`NeutrinoShowerClustering.cxx:582` mints conn-1 showers with
no spatial-overlap test at all), so one EM cascade became TWO conn-1 showers
-- 14153 (41 members) + NEW 14058 (20 members, start vertex 13 cm out,
stem ~0.2 MIP).  Memberships are disjoint and member length grew only +4.6%,
yet summed EM energy grew +46%: `kine_charge_from_maps`
(`NeutrinoEnergyReco.cxx:47-190`) credits a 2D cell's FULL charge to every
shower whose `associate_points`/`fit` cloud passes within 0.6 cm, with **no
cell ownership anywhere**, and the Enu sum
(`NeutrinoKinematics.cxx:511-515`) flat-adds every shower's kine_best
(`kine_energy_included` is written but never read).  14058's own trajectory
supports ~300 MeV (its kine_dQdx) but its kine_charge read 1265 -- the
cascade's charge absorbed by proximity.  Enu 2619 -> 3533.

Two structural facts established against `prototype_base`:
- **The prototype is equally ownership-free** (`NeutrinoID_energy_reco.h:44-248`,
  flat sum `NeutrinoID_kine.h:251-256`; no used-flags, no ownership map in
  either tree).  Its only protection is structural -- one cascade stays one
  WCShower -- and a conn-1/conn-1 pair is outside the domain of every merge
  gate in BOTH trees.  The dedup below is therefore a **deliberate
  divergence**, not a port fix.
- One genuine port gap: the prototype calls `shower->rebuild_point_clouds()`
  before every energy cloud read (`NeutrinoID_energy_reco.h:99`); the
  toolkit clouds are add-only merges, so departed members' stale points keep
  pulling charge in.  Closed by `kine_charge_rebuild` (ephemeral rebuilt
  clouds at the final recompute only -- stored clouds untouched, taggers and
  pi0 query them later).

OFF-arm mini-instances of the same disease exist (14059: an 11 cm stub
claiming 111 MeV; doc pr/91 measured a ~7 MeV case) -- the fix reclaims
those too, which is why the whole-sample screens below matter.

### 2. C1 -- `kine_charge_dedup` (+ C1b `kine_charge_rebuild`)

`recompute_shower_kine_charge_final` (NeutrinoEnergyReco.cxx), hooked in
`shower_clustering_with_nv` after ALL structure passes and BEFORE the pi0
finders (which cache `get_kine_charge()` at entry), so every mid-pipeline
gate ran on legacy values and every consumer downstream reads one consistent
set.  ONE scan of the three plane maps over per-shower contexts
(IndexedShowerSet order = stable creation-id order; `ChargeMap` is an
ordered map; no pointer-keyed iteration): per cell each context computes the
legacy acceptance (pcloud1 then pcloud2, `dis < 0.6 cm`), and the
**min-distance context wins the cell's full charge** (tie -> lowest shower
id).  A single-context input reproduces the legacy sum exactly; the
legacy `kine_charge_from_maps` is byte-untouched (fork-by-duplication).
Same kNN count as legacy (cells x k either way).  DEBUG census
`kine final recompute: ... old -> new` per shower.

**168596 measured (onf arms): 14153 1843->1433, 14058 1265->595; EM total
3108 -> 2028 (pre-split reconstruction read 2016+112); Enu 3533 -> 2445;
nue selection kept (4.30).**

### 3. A5 (`shower_hadronic_tag`) -- calibration forced two design corrections

Round-1's n=5 growth scalar did NOT survive the roster.  Calibration
(scripts/analysis/pr99/pr99_hadronic_calib.py, 109 conn-1 |11| showers over
the four on3 samples):

1. **Raw in-cylinder growth mislabels real electrons whose vertex sits in
   hadronic debris** -- other prongs' charge inflates the early bins (46363
   primary: 2365->335 pts/bin reads "shrinking"); 41/99 fired at the seed
   thresholds including nue-selected primaries.  Fix: **ownership filter** --
   an imaged point counts only if its nearest fit point over ALL graph
   segments belongs to a member (event-level all-fit cloud + row->segment
   map).  With it, all 36 nue-selected primaries read ends-ratio growth
   >= 2.32 while the misID'd hadrons read <= 0.7.  Margin 3.3x.
2. **395148 needs a proton-stem branch**: its ownership growth is 0.87
   (above any safe cut) but its stem reads 3.0 MIP -- a proton, while
   pair-conversion gamma stems read ~2 MIP.  Branch: stem >= 2.8 (C++
   median convention reads ~7% below the python np.median -- hence 2.8, not
   the python-basis 3.0) AND growth < 1.2.  On the full census exactly ONE
   additional shower fires (395148) and the only other high-stem object
   (268784's real 1075 MeV primary, stem 2.89) is protected by its growth
   2.84 >= the 1.2 ceiling.

Verdict: `smax>=10cm && (growth<0.7 || (bragg>=3.0 && growth<1.2) ||
(stem>=2.8 && growth<1.2))`.  On verdict the START segment is stamped
pdg 211 + mass + 4-mom refresh (the id_pi0 incoming-track recipe; durable
because `calculate_kinematics` re-copies the start segment's pdg) plus the
shower's cached type; NOT 13 (long-muon routing); re-typed shower ids guard
the pi0 finders' four collection points (empty set = byte-identical).
A DEBUG census line for EVERY evaluated shower makes each knob-on arm a
calibration sample.  Adverse gamma control 506114: growth 2.3, untouched.

### 4. Gates and proofs

- knob-off byte-identical vs round-2 production arms: **234 archives PASS**
  (work-pr99r3-off-{nuecc48,ncpi0,mcp1k,mcp2k} vs work-pr99r2-on3-*:
  96+38+70+30).
- `wcdoctest-clus` 2267 assertions PASS (12 new knob defaults).
- Compiled-config proofs: bare `wct-pr-perevt.jsonnet` with the production
  pipeline at HEAD == at round-3, byte-identical; knob-on keys present in
  the per-event `.wct-cfg-evt*.json` of every onf arm.

### 5. Knob-on screens (on3 vs onf, 117 events)

- **Selections: ZERO flips** -- nue (thr 3) and numu (thr 0.9) selections
  identical event-by-event across all four samples (lost=0 gained=0).
- Retype census: **20 re-types / 117 events** (11 nuecc48 satellite hadrons
  in nue events -- none a selected primary; 5 ncpi0; 4 numu incl. all four
  design objects).  116962: one 80 MeV re-typed shower was then dissolved to
  a pi+ TRACK by an existing production pass -- the intended endpoint; the
  only structural change in the whole shower A/B.
- Design events: 315167 pdg 211 + numu 1.77->2.23; 395148 pdg 211 + numu
  2.41->2.94; 285567 BOTH fakes 211, sham nue -1.65->-15; 70084 211;
  91653 numu 0.03->0.65.  Enu of re-typed events gains the pi rest mass
  (+139.6) per the existing non-EM kine convention.
- **NCpi0 pairing (the adjudicated cost)**: in-window pi0 pairs 10 -> 9.
  Lost: 37112 (135.8; gammas 635+386 -> 505+243 under dedup ALONE -- the
  pair genuinely shares halo charge and the mass is a product), 399860
  (138.4; the 26 MeV gamma of an asymmetric pair stripped to 3), 56982
  (121.9; its "gamma" stem reads -0.44 MIP, pathological), 359980 (93.4;
  A5-driven, pr/53-touchy event).  Gained: 314838 (145.3), 71372 (153.5).
  Attribution proven with a dedup-only probe arm
  (work-pr99r3-dduponly-ncpi0).  **Owner decision (this round): ship FULL
  winner-take-all.**  Noted: this is a deliberate divergence from the
  prototype's ownership-free accounting (both trees double-count), the
  shower fudge/recom calibration was historically tuned WITH shared-charge
  accounting, and -- owner: *"we have not done the EM shower clustering
  validation etc ... there is really no clear truth yet"* -- the pi0-mass
  shift is a number for that future campaign to adjudicate, not a truth
  claim either way.

### 6. Production flip

wct-pr-perevt.jsonnet: `kine_charge_dedup=true`, `kine_charge_rebuild=true`,
`shower_hadronic_tag=true`, `shower_hadronic_growth_max=0.7`,
`shower_hadronic_stem_ratio=2.8` (other A5 numerics = C++ defaults:
scan 30 / bin 3 / r_cyl 8 / r_core 1.2 / min_len 10 / growth_bragg 1.2 /
bragg_ratio 3.0).  Flip proofs (8-evt subset x 3 samples): flip-bare ==
onf 6+4+6 archives PASS (work-pr99r3-flip-* vs -onf-*); forced-off (env
bools=0 on the flipped cfg; the 0.7/2.8 numerics provably inert) == off
6+4+6 archives PASS (work-pr99r3-floff-* vs -off-*).  Toolkit commit
8573877f (PUSHED, ls-remote verified).

### 7. Bee A/B

bee/pr99r3/pr99r3x20-{off,on}.zip, 20 events (fixes 0-4, adjudicated pi0
losses 5-6, top Enu movers 7-12, pi0 gains 13-14, A5 retypes 15-19; index:
bee/pr99r3/pr99r3.index.txt).  OFF = round-2 production (on3)
001c22e1-9b17-4a2b-9cd5-24e3dba0d5c0 / ON = onf
90038859-db19-4c31-9937-986ec34121c6.

### Incident note (concurrent-session build race, 2026-08-20)

A 14:34 `./wcb build` while the peer session's vtx100-base batch was
launching jobs broke 4 of its events with "failed to load plugin:
WireCellRoot" -- wire-cell dlopens `build/<pkg>/`, NOT `local/lib` (the M1
correction memory is right).  Events re-run clean; all later relinks
coordinated by explicit pause/resume messages between the sessions, and the
two campaigns shared the box at PR_JOBS=16 each.  Rule: **no toolkit build
while ANY sbnd_xin batch (either session) is launching.**  Also: both
sessions independently forgot `PR_EXTRA_STAGES=pr_display` on their first
knob-on pass -- it is required from the start whenever a campaign reads
`calib-pr-evt*.json`.

### Arms (docs/work-tags.md)

work-pr99r3-off-* (gate arms), work-pr99r3-on-* (first knob-on, no display,
stem 3.0 -- superseded), work-pr99r3-ond-* (partial, aborted at the stem
correction -- releasable), work-pr99r3-onf-* (FINAL knob-on arms, keep until
owner scan), work-pr99r3-dduponly-ncpi0 (attribution probe).
