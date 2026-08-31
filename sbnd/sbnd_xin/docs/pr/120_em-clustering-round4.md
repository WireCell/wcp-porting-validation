# doc pr/120 — EM clustering round 4: backward-admission guards (stem_backfill / pass3_cone / examine_shower_1 walk)

**Status: COMPLETE, OWNER-REVIEWED — `stem_backfill_back_guard` SBND
PRODUCTION ON 2026-08-28 (flip-equivalence PASS 196/196);
`shower_ex1_walk_em_track_guard` ships OFF, not selected (measured no-op,
§5a). Bee A/B reviewed by the owner (OFF 6d6714c1 / ON 3b8c1b82, idx
0=47212 RESCUED, 1=281567 mover): "The scan looks OK" (2026-08-28).**

Owner directive (2026-08-28): "Please proceed with pr/120, same validation as
previous rounds" — following the pr/119 close-out recommendation. Same bar as
pr/117–119: byte-neutral probe → truth-joined census → default-OFF knob →
OFF gates + scored ON arm + flip pre-authorized if validation passes, Bee
links for any doubt.

## 0. Repro block

```bash
cd /home/xqian/toolkit-dev && direnv exec . bash -c \
  'cd toolkit && ./wcb build --notests -p && ./wcb install --notests -p'
ls -la local/lib/libWireCellClus.so && toolkit/build/clus/wcdoctest-clus

/home/xqian/tmp/pr120_arms.sh off0 0     # new binary, no env
/home/xqian/tmp/pr120_arms.sh dbgA 1     # WCT_SHOWER_{ABSORB,CONTENT}_DEBUG=1

python3 scripts/pr85_hash_gate.py work-pr118r1-flipchk-<s> work-pr120r1-off0-<s>; echo rc=$?
python3 scripts/pr85_hash_gate.py work-pr120r1-off0-<s>   work-pr120r1-dbgA-<s>; echo rc=$?

python3 em_display/prep_pr117.py --tag 120dbgA work-pr120r1-dbgA-{mcp1k,mcp2k,ncpi0,nuecc48}
./scripts/pr120_absorb_census.py --prepdir em_display/emprep-120dbgA \
    --members-tsv docs/pr/pr120-absorb-members.tsv \
    work-pr120r1-dbgA-mcp1k work-pr120r1-dbgA-mcp2k \
    work-pr120r1-dbgA-ncpi0 work-pr120r1-dbgA-nuecc48
```

Baselines unchanged from pr/119: Q/L roots `work-*-grp0825` (98 events),
production PR arms `work-pr118r1-flipchk-*`, score baseline
`pr118-onT-score.tsv` + `emprep-118onT`.

## 1. The measured opening (why these three sites)

Label census over all marks (angle = scan display's final-frame angle:
shower dir15 vs start→closest member point; `marks_detail.absorbed_by` from
the SHOWER_ABSORB probe, viewer's last-record-per-segment convention):

| admitting site | OUT marks (angle°) | IN marks (angle°) |
|---|---|---|
| stem_backfill (direct) | 2 at 146.3–147.9 (evt47212 seg 2103, marked out of both holders) | 0 |
| examine_shower_1_tmp (walk) | 1 at 141.6 (evt54332 seg 16014, 32.3 cm track) | 0 |
| pass3_cone (direct) | 3 at 113.3–133.5 (evt76346) | 24, med 7.3, **max 101.5** |
| pass4_angle (direct) | 13 at 2.7–28.6 | 89, med 9.7, max 64.6 — NOT separable |
| from_vertices (walk) | 7 at 6.4–79.2 | 28 at 0.6–53.5 — NOT separable |
| in_other_clusters_* | 0 | 31, of which **22 above 110°** — backward is legitimate there |

The three targeted sites cover 6 of the 29 OUT marks — exactly the
anchor-root/track events (47212, 54332, 76346) that pr/119 measured as
unreachable by any post-hoc membership edit — with zero IN casualties in the
labeled sample above 110°. Known caveats going in:

- **Frame mismatch**: the discriminating angle is the FINAL shower frame;
  pass3_cone's own admission gate caps accepted angles at ≤ 25–30°, and
  stem_backfill has no angle test at all (its chain walks toward the main
  vertex — backwardness is the mechanism, and it re-seats the start onto the
  absorbed stem). Whether a guard can sit at admission time or must audit
  late is a measurement, not an assumption — the P120 probe prints both
  frames.
- **Unmarked-admission exposure**: only marked segments carry scan angles;
  the false-positive rate over all admissions in 98 events is unmeasured —
  that is the census's main job.
- **evt54332 is separately a guard hole**: the walk's
  `absorb_track_guard` exempts `pdg==11` from the straight-long-track
  exclusion (PRShower.cxx guard_excludes), and 16014 is a 32.3 cm
  `straight=1` segment mis-PID'd as e−. The existing ADD probe lines carry
  `pdg/len_cm/straight`, so this discriminator is censused with no new code.

Recognition-miss events noted for this round's diagnosis ride-along:
evt166870 (85045 reco'd as a 4-segment µ, should be a γ of a π0 — labels
even carry the pair at m_γγ ≈ 116 MeV with the OTHER gamma 87058) and
evt235435 (whole-event EM shower never assembled; zero marks — note-only).

## 2. Phase A — probe (toolkit, byte-neutral)

Under the existing `WCT_SHOWER_ABSORB_DEBUG` env (so the standard probe arm
covers it), two new stderr lines in `clus/src/NeutrinoShowerClustering.cxx`:

- `SHOWER_ABSORB P120_STEM …` at the stem_backfill decision (printed for
  every chain candidate, accepted or rejected): pdg, len, dQ/dx ratio, ok,
  and the scan-equivalent admission-frame angles ang15/ang60 (shower
  dir15/dir60 at the current pre-re-seat start vs start→closest stem point).
- `SHOWER_ABSORB P120_P3CONE …` for every winning cone pair: the site's own
  angle/dist (identical arithmetic to the gate) plus scan-equivalent
  ang15/ang60 at the same start point, and the angle_offset.

Census `scripts/pr120_absorb_census.py` (imports pr119's truth machinery):
per-(shower, member) FINAL angle computed offline exactly as the viewer did
(content-probe dir15 + dump-shower start + closest dump point), absorbed_by
from the probe stream, truth via charge-weighted label matching. Reports:
per-site final-angle distributions by truth class, the guard sweep
(site ∈ {stem_backfill, pass3_cone, examine_shower_1_tmp} ∧ final_ang > θ),
admission-frame vs final-frame comparison, and the straight-long-electron
walk census.

### 2.1 Gate ledger (Phase A)

| gate | arms | result |
|---|---|---|
| binary-off (probe code in, env unset) | work-pr118r1-flipchk-* vs work-pr120r1-off0-* | PASS 196/196 (28+34+38+96) |
| probe byte-neutrality (env set) | work-pr120r1-off0-* vs work-pr120r1-dbgA-* | PASS 196/196 (28+34+38+96) |
| doctests | build/clus/wcdoctest-clus | 2442/2442 (probe build), 2450/2450 (knob build) |
| compiled-config, knobs off | git-HEAD compile vs worktree compile, full tagger pipeline_names + `dl_weights=` | `cmp` rc=0 (byte-identical) |
| compiled-config, knobs on | same + 4 TLAs | all 4 keys present in the tagger node data |

## 3. Phase A measurement

98 events: 1598 content headers, 4072 absorb records, 1699 P120_P3CONE +
12 P120_STEM admission lines, 1288 walk adds
(`docs/pr/pr120-absorb-members.tsv`, 5716 member rows; census log in §0).

**3a. The final-frame "backward member" separation of §1 is largely a
scanner-start-override artifact — pass3_cone is measured CLEAN.** Computed
against the *reco* final frame (content-probe dir15 + dump start), the
guard sweep over the three sites fires ZERO members at any θ ∈ 100–140°.
evt76346's three pass3_cone marks sit at 13.0–23.6° in every reco frame
(admission site angle 12.8–24.8°, admission scan-equivalent 13.1–23.6°,
final 13.0–23.6°) — their 113–134° label angles were measured against the
scanner's *relocated* start point, i.e. they encode the wrong-owner
diagnosis (the pieces belong to the other gamma), not a backward geometry
any reco-side test can see. pass3_cone's whole admitted population: final
angle max 56.8° over 1536 HOLD + 134 IN members. **No pass3_cone guard
ships**; evt76346 stays in the wrong-owner ledger (doc 118 §7 / doc 119 §7
route 3 caveat: that class needs the future cascade/π0 context, not local
geometry).

**3b. stem_backfill: both measurable-angle absorbs in 98 events are
scanner-condemned.** The full admission population is 12 chain candidates;
5 accepted. The three degenerate-angle acceptances (ang15 = −1: conn-1
showers whose start already sits on the chain, dist = 0) are trunk
extensions in "good"-note events (172230 seg 5027, 239794 seg 2069, 437699
seg 11030). The two measurable-angle acceptances both develop backward and
both are the scan's over-clustering complaints:

| event | seg | conn | ang15/ang60 (°) | verdict |
|---|---|---|---|---|
| 47212 | 2103 | 2 | 150.2 / 152.2 | OUT-marked (of both holders); q_extra 5.5e5 |
| 281567 | 95128 | 2 | 150.4 / 155.9 | scan note: "95128 has an overclustering issue of an EM shower isolated with a main cluster segment" |

The four MIP-window rejections (105946/281165/282909/37112, ang 91–163°)
are already stopped by the existing gate. **Measured operating point:
decline when the scan-equivalent angle is measurable and > 110°** — fires
exactly on the two condemned absorbs, keeps all three trunk extensions.

**3c. The evt54332 hole is exactly one firing wide.** Walk-adds of
straight=1, pdg==11 segments: 23 at ≥10 cm (5 truth-IN — a 10 cm floor
would cost real shower content), **3 at ≥20 cm**: evt54332 seg 16014
(32.3 cm, OUT, site examine_shower_1_tmp — the target) and two HOLD
members at site in_main_cluster (30504 seg 11010 at 42.5 cm, 42280 seg
8040 at 20.6 cm) which a *site-scoped* guard never touches. **Measured
operating point: em-straight floor 20 cm, passed only by the
examine_shower_1 call site.**

## 4. Phase B — knobs (toolkit; all default OFF, thresholds from §3)

| knob | default | fires on (measured) |
|---|---|---|
| `stem_backfill_back_guard` | false | evt47212 seg 2103, evt281567 seg 95128 |
| `stem_backfill_back_ang` | 110.0° | inert while guard off |
| `shower_ex1_walk_em_track_guard` | false | evt54332 seg 16014 |
| `shower_ex1_walk_em_track_len` | 20 cm | inert while guard off |

Mechanics: (1) in `stem_backfill`, after the MIP window accepts, compute
the scan-equivalent angle (shower dir15 at current pre-re-seat start vs
start→closest stem point — the same arithmetic as the P120_STEM probe) and
`break` the chain when measurable and beyond the cut. (2)
`Shower::complete_structure_with_start_segment` gains a defaulted
`em_straight_min_len = 0` parameter (all nine call sites byte-identical at
the default); `guard_excludes`' pdg==11 early-out now falls through to
`segment_is_straight_long_track` when the segment is longer than the floor;
only the `examine_shower_1` call site passes the floor, knob-gated. Seats:
the standard 7 (NeutrinoPatternBase.h / TaggerCheckNeutrino.h/.cxx ×3 /
doctest pins / sbnd jsonnet decl + key-suppression) + runner env→TLA block
(`SBND_STEM_BACKFILL_BACK_GUARD`, `SBND_STEM_BACKFILL_BACK_ANG`,
`SBND_SHOWER_EX1_WALK_EM_TRACK_GUARD`, `SBND_SHOWER_EX1_WALK_EM_TRACK_LEN`).

## 5. Validation

- **Knob-off gate**: `work-pr120r1-off1-*` (knob binary, no env) vs
  `work-pr120r1-off0-*`: **PASS 196/196**.
- **ON arm** `work-pr120r1-on2-*` (both knobs via env): exactly **2 events
  change**, both mcp2k, both stem-guard fires with the predicted sentinel
  (`pr120 stem_backfill_back_guard: decline … ang15=150.2/150.4 > 110.0`);
  the other 96 events are archive-byte-identical. Only `mabc-pr.zip`
  differs on the two (`pctree` identical).
- **Scores** (probe-grade sidecars `emprep-120dbgon`, 3-event ON probe arm
  hash-identical to on2, diffstat vs `emprep-118onT`): 47212 — seg 2103
  released, **qF1 0.965 → 1.000** (n_extra 1→0, q_extra 5.5e5→0; the
  cross-run match moves to node 105072 and seg 2103 renders as a plain
  pi+ PF track, `shower_id −1` — no PF vanishing). 281567 — seg 95128
  (1.9e5) released from shower 9025, exactly the scan note's complaint;
  unlabeled, so adjudicated by Bee review. 54332 — **unchanged** (§5a).
  Note: a first whole-manifest score comparison against dump-derived
  sidecars showed 4th-decimal drift + two spurious drops (84229, 314838) —
  artifacts of the lossy single-valued `shower_id` dump join on arms run
  without the content probe; the hash gate (96/98 identical) is the
  authoritative no-change proof, and the probe-grade 3-event rescore shows
  the true deltas.
- **nusel**: byte-identical **98/98** (including the 2 changed events —
  the released stems don't move any tagger).
- **Vertex movers** (`pr90_movers.py`, mcp2k): compared 12, movers 0,
  ADVERSE 0.
- **Bee A/B** (owner review, pre-authorized "for any doubts, send me the
  bee links"): OFF `6d6714c1-d2d5-4e8e-996a-69b8a4016455` / ON
  `3b8c1b82-1ef0-430a-823b-148ebc13f07f`, idx 0=47212 (RESCUED), 1=281567
  (mover, owner adjudication requested). Index: `bee/pr120r1/`.
- **Flip** (owner: "same validation as previous rounds" = pre-authorized):
  `stem_backfill_back_guard = true` SBND PRODUCTION ON 2026-08-28;
  compiled-config proof: key present, ex1 keys suppressed.
  Flip-equivalence: `work-pr120r1-flipchk-*` (post-flip config, no env) vs
  `work-pr120r1-on2-*`: **PASS 196/196** (28+34+38+96) — also re-proving
  the ex1 guard's no-op status (flipchk runs without it, on2 ran with it).

### 5a. shower_ex1_walk_em_track_guard ships OFF — "not selected"

The guard works as designed (single-event probe run: `SHOWER_ABSORB
EXCLUDE shower_start_seg=16006 seg=16014` fires) but is a measured
**global no-op**: in the current production chain evt54332's seg 16014 is
not walk-absorbed anywhere — it carries a `kShowerTopology` flag
(`flags=-T-`) and is **seeded as the shower's own root** by the
in_main_cluster pass (`site=in_main_cluster shower_start_seg=16014`,
`walk_begin start_seg=16014`), with pass3_cluster_map adding 16015–16018.
The scan-era `absorbed_by = examine_shower_1_tmp` blame was the viewer's
last-record-per-segment convention over the *prod0825-era* probe stream —
that admission route no longer operates on this event. The real defect is
upstream: the track/shower separation flagged a straight 32.3 cm track as
shower topology. 54332 therefore moves to the **recognition thread** (§6).

## 6. What stays open

- **The recognition thread** (the owner's "EM clustering was not correctly
  recognized"), now with two measured cases: evt54332 (straight 32.3 cm
  track kShowerTopology-flagged → seeded as a shower root; the admission
  guards cannot reach a seed) and evt166870 (node 85045: a real 4-segment,
  20.5 cm EM object clustered fine but PID-voted µ⁻ at 38.6 MeV — the
  labels even carry its π⁰ pair with shower 87058 at m_γγ ≈ 116 MeV).
  Both live in `separate_track_shower`/`update_particle_type` territory,
  not in absorption geometry. evt235435 (whole-event shower never
  assembled, note-only, conn-3) is the third member.
  **→ Measured in doc [pr/122](122_recognition-round.md)** (all three die or
  defer: seed features interleave with real stems across both manifests and
  long-seed flags are inherited un-revalidated; 166870's µ⁻ is the pr/40 r9
  sfv_straight_guard by design, n=1 bad fire of 17, routed to the π⁰ thread;
  235435 is fragmentation, i.e. the merge thread).
- **evt76346 / pass3_cone**: measured clean in every reco frame (§3a) —
  confirms the wrong-owner class needs relocated-start/cascade context
  (doc 118 §7, doc 119 §7); no admission guard can see it.
- pass4_angle (13 OUT marks at 2.7–28.6°, overlapped with 89 IN) and
  from_vertices (7 OUT, overlapped) stay unseparable by angle — unchanged
  from the §1 table.
- Everything in doc 119 §7 (π⁰-hypothesis split owner-gated, c-set scan,
  under-clustering residuals).

## 7. Files

- toolkit (`apply-pointcloud`): `clus/src/NeutrinoShowerClustering.cxx`
  (P120_STEM / P120_P3CONE probes; stem_backfill backward guard;
  examine_shower_1 em-floor pass-through),
  `clus/src/PRShower.cxx` + `clus/inc/WireCellClus/PRShower.h`
  (`complete_structure_with_start_segment` gains defaulted
  `em_straight_min_len`; guard_excludes em-branch),
  `clus/inc/WireCellClus/NeutrinoPatternBase.h`,
  `clus/inc/WireCellClus/TaggerCheckNeutrino.h`,
  `clus/src/TaggerCheckNeutrino.cxx`,
  `clus/test/doctest_clus_knob_defaults.cxx` (2450 assertions),
  `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` (decl +
  key-suppression + production flip).
- wcp-porting-img (`main`): this doc; `scripts/pr120_absorb_census.py`;
  `docs/pr/pr120-absorb-members.tsv` (5716 rows) +
  `docs/pr/pr120-on2-score.tsv`; `run_pr_chain_batch.sh` pr/120 env block;
  `bee/pr120r1/` (index + prid-maps + urls; zips untracked);
  `em_display/emprep-120{dbgA,on2,dbgon}` + manifests.
- Arms (untracked): `work-pr120r1-{off0,dbgA,off1,on2,dbg54332,dbgon,flipchk}-*`;
  launcher `/home/xqian/tmp/pr120_arms.sh`; census log
  `/home/xqian/tmp/pr120_census1.log`.
