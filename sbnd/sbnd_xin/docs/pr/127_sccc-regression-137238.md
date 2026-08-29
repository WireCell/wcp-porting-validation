# doc pr/127 — 137238's cross-cluster muon: a shipped fix that fell out of tier

**Status: SHIPPED** 2026-08-29 — toolkit `d74c5524` (K5 flip) + `49754bf8`
(`sccc_max_gap` 6→10), wcp `27a8bfdf`, both pushed. Production point is now
`work-pr125r1-flipS{98,141}-*`. Follow-on to doc pr/125 (owner verdicts on
its production Bee pair) and doc pr/93 round 4 (the original fix for this
same event).

Owner directive (2026-08-29, scanning the pr/125 production Bee pair
`7f4ffdb1` / `cdf0749a`):

> "For item 2, things are good, there is one issue 137238, the electron 89
> MeV should connect to some thing, which is missing from the PF tree,
> should check and fix. item 3, flip on is fine."

Two deliverables, deliberately kept in **separate commits and separate
gates** because they carry two different authorizations:

1. **K5 flip** (`shower_satellite_absorb`, doc pr/125 §5) — "flip on is
   fine" = the operating point the owner scanned on the K5 decision pair
   (`b169a068` / `defaa224`), i.e. `max_mev=10 / host_mev=20`, not the
   cap-3 variant. §4 below.
2. **137238** — §2/§3 below. This is **not** a new symptom: it is the
   silent loss of a shipped, owner-approved fix.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# 1. the tape that diagnoses it (byte-neutral stderr; no rebuild):
WCT_SCCC_DEBUG=1 PR_EXTRA_STAGES=pr_display PR_JOBS=4 \
  ./run_pr_chain_batch.sh work-nuecc48-grp0825 work-pr127r1-sccc-nuecc48 data 137238
grep SCCC work-pr127r1-sccc-nuecc48/pr_evt137238/stdout.log
# 2. the one-value fix, same event, no rebuild:
WCT_SCCC_DEBUG=1 SBND_SCCC_MAX_GAP=9 PR_EXTRA_STAGES=pr_display PR_JOBS=4 \
  ./run_pr_chain_batch.sh work-nuecc48-grp0825 work-pr127r1-g9-nuecc48 data 137238
grep -E "sccc (demote|bridge)" work-pr127r1-g9-nuecc48/pr_evt137238/wct_pr_evt137238.log
# 3. manifest-wide candidate census from any tape-armed arm:
python3 scripts/pr127_sccc_census.py 'work-pr125r1-flipK598-*' 'work-pr125r1-flipK5141-*' \
    --tsv docs/pr/pr127-sccc-census.tsv
# 4. PF-tree history of one event across every arm on disk (regression finder):
python3 scripts/pr127_pf_history.py 137238
# 5. the arms (NOTE: scripts/pr125_arms.sh prefixes every arm work-pr125r1-):
./scripts/pr125_arms.sh 98  flipK598  0 WCT_SCCC_DEBUG=1   # K5 flip,  both fronts off
./scripts/pr125_arms.sh 141 flipK5141 0 WCT_SCCC_DEBUG=1
./scripts/pr125_arms.sh 98  flipS98   0 WCT_SCCC_DEBUG=1   # + sccc_max_gap=10
./scripts/pr125_arms.sh 141 flipS141  0 WCT_SCCC_DEBUG=1
python3 scripts/pr85_hash_gate.py work-pr125r1-flipK598-nuecc48 work-pr125r1-flipS98-nuecc48
# 6. sentinels (doc sec 5.1) -- at the pre-fix point 137238 FAILs, after it PASSes:
python3 scripts/pr127_sentinels.py --arms 'work-pr125r1-flipK5*'   # 9 PASS 1 FAIL 1 SKIP
python3 scripts/pr127_sentinels.py --arms 'work-pr125r1-flipS*'    # 10 PASS 0 FAIL 1 SKIP
```

## 1. Symptom

Production PF tree for 137238 (arm `work-pr125r1-flipchk98-nuecc48`, the
point the owner scanned):

```
nu 0
  e-  89 MeV     id=143057  20.2 cm   <- the owner's node; nothing below it
  mu- 88 MeV     id=143052  23.6 cm
    gamma 9 -> e- 9 | pi+ 53 | mu- 58 -> (e- 11, e- 370 -> gamma 5 -> e- 5)
```

`kine_reco_Enu = 688.1 MeV`.

What the 89 MeV e- "should connect to" is in the calib dump but in no tree:
its far vertex (`7004`, at −109.0, 33.1, 412.3) is the start of segment
**7008 — a 79.3 cm pdg-13 track in cluster 7**, which continues into
segments 7007/7009 (a 19.4 cm, 51.1 MeV EM shower `7009`, conn-4). The
89 MeV shower itself already spans the cluster boundary (members 7003/7005/
7006 are cluster-7 segments), so the reconstruction has the whole ~118 cm
chain — the PF tree just cannot see past cluster 143, and neither can the
kine tree (`kine_energy_particle` has no entry for the muon or shower 7009).

## 2. Root cause — the pr/93 round-4 fix fell out of tier

This exact event and this exact object were adjudicated by the owner on
2026-08-18 (doc pr/93 §7, target 5):

> "the 152 MeV electron is actually a muon ... it has a delta ray, which may
> have tricked the track/shower separation, but it is really long, so
> clearly a muon." Follow-up: "since the angle is very aligned, maybe use
> this condition with a larger allowed gap."

The fix shipped that day, SBND PRODUCTION ON: `straight_cont_cross_cluster`
+ `sccc_bridge_body`, with the base tier **retuned to 6 cm / 18°** because
137238's muon-body candidate measured **g = 5.68 cm, K = 17.0°** (doc pr/93
§7 "iteration history", round A). Result then: `e- 152` → `mu- 60` (stem) →
bridge → `mu- 211` (81 cm body) → `mu- 65`, Enu 1087.1 → 1101.4.

Today, same knobs, same event, `WCT_SCCC_DEBUG` tape (§Repro step 1):

```
SCCC pass: main_vtx_gidx=41 gap=6.0/12.0cm kink=18.0/7.5deg bridge=1
SCCC stem-cand: len=14.3cm traj=1 topo=0        <- the stem is still a candidate
SCCC cand g=2.20cm K=79.2 k_tan=12.7 tier_ok=0  <- delta-ray stubs, correctly out
SCCC cand g=4.43cm K=69.5 k_tan=38.7 tier_ok=0
SCCC cand g=1.17cm K=40.2 k_tan=18.2 tier_ok=0
SCCC cand g=8.00cm K=14.0 k_tan=10.2 tier_ok=0  <- THE MUON BODY
```

The muon body now measures **g = 8.00 cm, K = 14.0°**: 2.3 cm further away
than when the tier was cut to fit it, and 3° straighter. It fails the base
tier on gap (8.00 > 6) and the aligned tier on kink (14.0 > 7.5) — it falls
in the notch between the two tiers. Nothing else about the pass changed: the
stem is still trajectory-flagged, still degree-1 at its tip, the body is
still a 79.3 cm pdg-13 track.

**Why the geometry moved — NOT ESTABLISHED.** The body is 79.3 cm today vs
81.2 cm in pr/93 r4 (~1.9 cm shorter at its near end) and the stem 14.3 vs
14.6 cm. Two candidate causes, and the data on disk cannot separate them:

- the Q/L era moved (`work-nuecc48-cb0805` at pr/93 r4 →
  `work-nuecc48-ql0819` by 2026-08-21 → `work-nuecc48-grp0825` today — the
  earliest broken arm's own compiled config names `ql0819` as its input);
- a PR-stage code change landed in the 2026-08-18 → 08-21 window
  (pr/94–99), moving the fit or the pass.

Neither `cb0805` nor `ql0819` survives on disk for nuecc48, so the working
state cannot be re-run and the attribution stays open. It does not affect
the fix — that rests on today's measurement — but it does affect the blast
radius: an input drift is one event's bad luck, whereas a code change could
have killed the *other* fixes pr/93 r4 shipped. §5.3 checks them.

**Why it hid**: the loss is invisible to every gate we run. It is not a
byte-diff of a knob A/B (both arms have the fix "on"); it moved no label
(137238 is unlabeled in vtx105); and in the pr/125 production Bee pair it
sits at idx 7 marked *"no-op verified"* — which was true of the flip and
said nothing about the event. The earliest surviving arm on disk
(`work-vtx105-base-nuecc48`, 2026-08-21) already shows the broken tree, so
the fix has been dead for **~10 days** in production.

## 3. Fix — one production value, no C++

`sccc_max_gap` 6 → 9 cm restores the pr/93 r4 structure exactly
(§Repro step 2, arm `work-pr127r1-g9-nuecc48`):

```
sccc demote: seg cluster=143 len_cm=14.3 pdg 11 -> 13 sib_cluster=7 sib_len_cm=79.3
sccc bridge: cluster 7 -> main 143 main_vtx_idx=51 far_vtx_idx=4 bridge=OK
```

```
nu 0
  mu-  88 MeV
    (unchanged sub-tree)
  mu-  60 MeV   id=143057  12.3 cm      <- was "e- 89 MeV"
    0 0 MeV     id=143070   8.0 cm      <- zero-charge bridge node
      e-  6 MeV  -> e- 9 MeV            <- the delta ray, as EM leaves
      e- 17 MeV
      mu- 207 MeV id=7008 76.6 cm       <- the body the owner is pointing at
        mu- 66 MeV id=7007 14.4 cm
        e-   7 MeV id=7009  2.5 cm
```

`kine_reco_Enu` 688.1 → **764.4 MeV** (+76.3). Structure matches pr/93 r4's
recorded result (`mu- 60` → bridge → `mu- 211` → `mu- 65`) to within the
era's own drift (207 vs 211, 66 vs 65).

Value choice: 9 cm leaves only 1.0 cm of margin over a measurement that has
already drifted 2.3 cm once. §3.1 picks the final value from the
manifest-wide candidate census (`pr127_sccc_census.py`), preferring the
largest gap that admits no candidate we cannot adjudicate.

### 3.1 Manifest-wide candidate census → `sccc_max_gap = 10`

Tape-armed arms `work-pr125r1-flipK5{98,141}-*` (the new production point,
K5 on), 239 events, **107 cross-cluster candidates** recorded in the
g ≤ 12 cm window (`docs/pr/pr127-sccc-census.tsv`):

| gate | candidates passing | events | newly admitted vs production |
|---|---|---|---|
| production 6/18 ∪ 12/7.5 | 1 | 1 | — |
| base gap 9 / 10 / 12 (kink 18) | 2 | 2 | **1 (137238 only)** |
| mid tier 9/15 or 10/15 | 2 | 2 | 1 (137238 only) |
| linear taper (6,18)→(12,7.5) | 2 | 2 | 1 (137238 only) |

Every proposal admits **exactly one** new candidate across both manifests,
and it is the object the owner is pointing at. The kink axis does all the
separating: 137238's body sits at **K = 14.0°**, and the next-lowest
candidate anywhere in 239 events is **28.6°** (163543, g = 4.31 cm) — a
14.6° margin below the 18° limit. Nothing between 6 and 12 cm of gap is
near the gate.

Two facts worth recording alongside:

- The one candidate that passes the production tiers today (**499577**,
  g = 3.01 cm, K = 9.8°) does **not** demote — it is stopped by a later
  gate (EM-leakage bound or `far_ok`). Grepping the arms for
  `sccc demote:` returns **zero fires in 239 events**: the pass that
  shipped SBND ON in doc pr/93 r4 is currently dormant everywhere, not
  just on its own target event.
- `tier_ok` is an upper bound on fires, not a fire count — the census
  answers "what could this gate let through", and the ON arm answers
  "what did it do".

Value chosen: **10 cm** (SBND cfg; C++ default stays 5). It is 2× the
margin over today's measured 8.00 cm, keeps the two-tier structure
meaningful (10 cm at ≤18°, 12 cm at ≤7.5°), and the census shows 12 cm
would have been equally collateral-free — that is the fallback if the
geometry drifts again, and §5's sentinel is what would tell us.

### 3.2 Validation of the flip (`sccc_max_gap = 10`, toolkit cfg)

Arms `work-pr125r1-flipS{98,141}-*` (post-flip config, no env) vs the
production point `work-pr125r1-flipK5{98,141}-*`:

| manifest | archives | verdict |
|---|---|---|
| 98 (mcp1k/mcp2k/ncpi0/nuecc48) | 196 | 195 identical; **1 differs: 137238 `mabc-pr.zip`** |
| 141 (mcp1k/mcp2k) | 282 | PASS all byte-identical |

478 archives, one moved event — the intended one; `pctree-pr` identical
everywhere including 137238.

- **Control**: 137238 rerun at *this* config with `SBND_SCCC_MAX_GAP=6`
  (`work-pr127r1-g6ctrl-nuecc48`) reproduces production byte-identically
  (PASS 2/2) — the only thing the cfg change does is the gap value.
- **Movers**: 0 on all six arm pairs (`--tags vtx105`, ADVERSE 0).
- **Q/L selection table**: `nusel-evt*.tsv` 239/239 byte-identical.
- **Label metric**: unmovable by construction — 137238 carries an
  `emscan-0827` record with **no marks and no verdict**, and every other
  event is byte-identical, so no qF1/q_extra number can change.
- **Sentinels** (§5.1): 10 PASS / 0 FAIL / 1 SKIP at the fixed point; the
  same suite reports 137238 FAIL at the pre-fix production point.

**Tagger scalars on the one moved event** (report, not a tune — CLAUDE.md
§5.7). 137238's PR-level scalars move as the structure moves:

| scalar | before | after |
|---|---|---|
| `kine_reco_Enu` | 706.4 | 768.0 |
| `kine_reco_add_energy` | 105.7 | 211.3 |
| `nue_score` | −4.30 | −15.00 |
| `numu_score` | 1.10 | 0.82 |
| `kine_pio_flag` | 0 | 1 (mass 3.25 MeV) |
| `cosmict_6_filled` | 0.0 | 1.0 |

Two readings that matter:

1. **−15.0 is this variable's floor**: exactly −15.0 is both the *minimum*
   and the *median* nue_score over all 239 events (125 events, 52 %, sit
   there); the maximum anywhere in the manifest is 4.30. What is *proven*
   here is that the **Q/L selection table is byte-identical** — and that
   table (`nusel-evt*.tsv`: flash, tgm/stm/fc/lm, label) does **not**
   contain nue_score, so it says nothing about a νe selection. Whether the
   −4.30 → −15.00 move crosses a νe acceptance cut is **not established in
   this doc** — no such cut was measured. Flagging it that way deliberately:
   137238 comes from the νe CC sample, and the event losing its (fake)
   vertex electron and gaining a vertex muon is exactly the structure the
   owner adjudicated in pr/93 r4, but "the structure is right" is not the
   same claim as "the selection is unaffected".
2. **A junk π⁰ pairing appears** and is worth handing to the π⁰ work
   (doc pr/126): Path 1 now pairs the 379.6 MeV electron with a **1.29 MeV**
   crumb 47.3 cm away, 9.3° apart → `kine_pio_mass = 3.25 MeV`. The
   reconstruction's π⁰ finder has no minimum-partner-energy floor, while
   pr/126's own hand-scan selection required min(E) > 15 MeV. This is a
   pre-existing finder property that the structure change exposed, not
   something this knob does.

## 4. K5 flip (`shower_satellite_absorb`) — owner "flip on is fine"

Flipped in `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` at the
operating point the owner scanned on the K5 decision pair (`b169a068` /
`defaa224`): `max_mev = 10`, `host_mev = 20` — **not** the cap-3 variant.
Toolkit `d74c5524` (pushed). The effect and its accepted cost are doc
pr/125 §5's table; nothing about them changed.

**Compiled-config proof.** Same event, same TLAs, arm names normalized:
the pre-flip (`work-pr125r1-flipchk98-mcp1k`) and post-flip
(`work-pr125r1-flipK598-mcp1k`) compiled JSONs differ by exactly one line —
`"shower_satellite_absorb": true`. `max_mev`/`host_mev` stay suppressed
(they equal the C++ defaults).

**Flip-equivalence, by per-event decomposition.** New arms
`work-pr125r1-flipK5{98,141}-*` (post-flip config, no env) vs the
validated env-driven `onM{98,141}d` arms:

| arm pair | archives | verdict |
|---|---|---|
| 98 mcp1k | 28 | PASS byte-identical |
| 98 mcp2k | 34 | 2 differ: 396222, 415278 |
| 98 ncpi0 | 38 | PASS |
| 98 nuecc48 | 96 | PASS |
| 141 mcp1k | 104 | PASS |
| 141 mcp2k | 178 | 4 differ: 52693, 77328, 94392, 173819 |

478 archives, 472 byte-identical; the 6 that differ are **exactly** the
pass3-guard footprint recorded in doc pr/125 §5.2 — the `onM*d` arms
predate that flip, so the difference is the guard, not K5.

**Control (closes the composition question).** The two guard-moved events
rerun at *this* config with `SBND_SHOWER_SATELLITE_ABSORB=0`
(`work-pr127r1-k5off2-mcp2k`) reproduce production byte-identically,
PASS 4/4: the cfg change alters nothing but the K5 knob, including where
the guard is active.

**Physics checks at the new production point** (flipchk* → flipK5*):
0 vertex movers (`pr90_movers.py --tags vtx105`) on all six arm pairs
(159 labeled events compared, ADVERSE 0); `nusel-evt*.tsv` **239/239
byte-identical**.

## 5. Standing exposure this round exposes

Every knob in this family was tuned to one event's *measured* geometry and
carries the same failure mode — a silent, gate-invisible death when the
upstream era moves the number: `sccc_max_gap=6` (g=5.68),
`sccc_kink_max=18` (K=17.0), `long_muon_cathode_bridge_lever=15`,
`shower_pass3_cone_guard_len=15`, `shower_samevtx_absorb_min_len=5`.
Proposal in §5.1: a per-fix **sentinel assertion** (named event → expected PF
signature / Enu / fire log line) run every round, so a dead fix reports
itself instead of waiting for the owner to spot it in a Bee scan.

### 5.1 Sentinels — implemented this round

`scripts/pr127_sentinels.py` holds one entry per shipped fix: the event it
was shipped for, and assertions on the *current production arm*. Energy
assertions are thresholds placed **between** the measured pre-fix and
post-fix values, so a few-MeV drift is tolerated and a structural loss is
not. Run it at the end of every round:

```bash
python3 scripts/pr127_sentinels.py --arms 'work-<round>-<prod arm glob>'
# exit 0 = all applicable sentinels PASS, 1 = at least one FAIL
```

Seed registry (11 entries, 6 shipped fixes, 4 docs):

| event | fix | assertion |
|---|---|---|
| 137238 | pr/93 r4 sccc + pr/127 | a PF `mu-` ≥ 150 MeV exists; log has `sccc demote` |
| 37112 | pr/125 K3 samevtx absorb | max EM shower ≥ 700 MeV; no `proton` ≥ 400 MeV |
| 69314 | pr/125 K5 satellite absorb | 14 ≤ showers ≤ 20 |
| 94392 / 52693 / 77328 / 173819 | pr/125 pass3 guard | no PF `e-` ≥ 250 / 175 / 130 / 200 MeV |
| 171572 / 393505 | pr/123 r2 guard-freed pickup | log has `pf-orphan-guard-freed` (+ `mu-` ≥ 250) |
| 348471 | pr/93 r4 detach + pr/121 | no `proton` ≥ 600 MeV; max EM shower ≥ 300 MeV |
| 315167 | pr/93 r4 orphan track | `proton` ≥ 500 MeV — **SKIP**, not in the manifests |

Measured behaviour: at the **pre-fix** production point
(`work-pr125r1-flipK5*`) the suite reports **9 PASS / 1 FAIL / 1 SKIP** —
the FAIL is 137238, i.e. it reproduces today's discovery from arms that
already existed. At the fixed point (`work-pr125r1-flipS*`) it reports
**10 PASS / 0 FAIL / 1 SKIP**.

Companion: `scripts/pr127_pf_history.py <event>` prints an event's PF-tree
signature across every arm on disk in mtime order, marking each change —
that is how the ~2026-08-21 turnover was located, and it is the tool to
reach for when a sentinel fails.

### 5.3 The siblings: are the OTHER pr/93 r4 fixes still alive?

Because the attribution above is open (§2), the blast radius had to be
checked: pr/93 round 4 shipped four fixes, and 137238 was one of them. The
other three, re-run at today's production config
(`work-pr127r1-r4check-mcp1k`, Q/L root `work-mcp1k-grp0825`):

| event | shipped 2026-08-18 | today | verdict |
|---|---|---|---|
| 348471 | `proton 719` aggregate → `proton 308` + π⁰ 113 + γ | `proton 310`, max EM shower 414 | **alive** (sentinel PASS) |
| 315167 | orphan machinery emits the 150.7 cm `proton 595` root | `proton 613` root, Enu 1705.1 — but `pf-orphan-audit: 0 unclaimed`, i.e. the ordinary track BFS now reaches it; the knob fires 0× in 98 events | **outcome healthy, mechanism dormant** |
| 292643 | `pi+ 162` aggregate → `pi+ 88` → `mu- 58` → 4 γ, Enu 1073.6 | no `pi+` at all: head is `e- 227` with γ 65/8/5 + `mu- 59`, plus `mu- 441`; Enu 858.5; `detach_track_stem` does not fire here (it fires in 69 of the 98-manifest events, so the knob is alive) | **drifted — owner look wanted** |

Single look-at Bee set (no A/B is possible — the cb0805/ql0819 roots are
gone): `5a253c3b-328e-4795-949a-57fe1425aa3a`, idx 0 = 292643, idx 1 =
315167, annotated `bee/pr127r1sib/pr127r1sib.index.txt`.

The pattern worth naming: of four fixes from one round, one died silently
(137238), one drifted to a different structure (292643), one is satisfied by
a different mechanism than the one that shipped (315167), and one is intact
(348471). None of that was visible from any gate — which is the argument for
§5.1 rather than a one-off repair.

### 5.2 What is still open / worth a round (measured, not speculative)

Owner-side decisions:

1. **π⁰ Path 2 disable knob** (doc pr/125 §5.3, 396222) — the "item 3"
   verdict answered the satellite absorb, so this one is still open. Path 2
   MUTATES the ν vertex (NSC:5675-5676, never SBND-validated), and the
   pass3-guard flip woke it for the first time on SBND.
2. **EM charge scale** `kine_shower_fudge_factor` 0.80 → 0.84 (peer's
   pr/126 peak fit). Must be its own round — never combined with a
   clustering change.

Named residual classes carried forward (each already measured):

3. `pass3_cone` **second** absorber — 20 labeled-OUT marks still absorbed
   after the guard (pr/123/124).
4. Contiguous far chains (278420: first link 23.9 cm, below the tier-2
   prune's G) and 179369's backward cluster (pr/123).
5. The **score-100 sentinel** (pr/124 §B.1): the trajectory branch stamps
   pdg-11 / score-100 unconditionally, which is (a) the upstream defect
   behind the twice-measured-dead "root is wrong" recognition family and
   (b) exactly what stamped 137238's muon stem an electron in the first
   place. Candidate knob: treat `score==100 && pdg!=0 && |pdg|!=11` as
   confident — **owner call**, because it changes PID semantics broadly.
6. π⁰ collinear split (314838, 142421) — owner-gated.
7. PID as the top π⁰ blocker (pr/126 §2) — and this round's junk 3.25 MeV
   pairing (§3.2) says the finder also wants a minimum-partner-energy floor.

Structural blind spots this round exposed (measurement first, no knob yet):

8. **Cross-cluster invisibility**: every PF orphan pool *and* the
   `pr65 pf-orphan-audit` log line is `same_cluster`-gated
   (MultiAlgBlobClustering.cxx:1852, :2331, :2398), so an object in another
   cluster that is not nv-bridged is neither displayed nor **counted**. The
   cheap next step is to widen the AUDIT LOG only (byte-neutral, no
   emission) and count how many events hide a cross-cluster track — turning
   an unknown into a number before anyone proposes a fix.
9. **conn-4 showers are dropped from PF and kine** (`conn4_skip_segs`).
   137238's own 51.1 MeV Michel-like shower 7009 is one of them. Same
   treatment: count them first.

Measured **dead**, do not reopen: P2 / charge-continuity merge (pr/118),
the expel predicate (pr/119), recognition-by-features (pr/122, pr/124
front B).
