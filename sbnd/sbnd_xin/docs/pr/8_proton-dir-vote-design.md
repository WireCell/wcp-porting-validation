# 8 — design: proton-template direction vote (`proton_dir_vote`)

Status: DESIGN (owner-requested 2026-07-30, "OK to be different from the
prototype — this is an improvement"). Not implemented. Successor to
pr/5 §6.1; builds on the pr/7 §6 anatomy. Owner sign-off points in §8.

## 0. Repro block (design inputs)

```bash
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
# decision machinery this design extends:
sed -n '1407,1489p' $TK/clus/src/PRSegmentFunctions.cxx    # segment_do_track_pid
sed -n '1291,1359p' $TK/clus/src/PRSegmentFunctions.cxx    # do_track_comp scores
sed -n '957,966p'  $TK/clus/src/PRSegmentFunctions.cxx     # eval_ks_ratio gate
# evt 172230 numbers: fwd ks_mu=0.0746 ks_flat=0.0494 (abstain both ways);
# proton score fwd 0.135 vs bwd 0.193 (doc pr/5 replay).
```

## 1. Problem statement

`segment_do_track_pid` decides direction ONLY from the muon-vs-flat rows
(`eval_ks_ratio`); when both orientations fail — the generic outcome for a
short Bragg proton (pr/7 §6: recombination + diffusion make it nearly flat
in shape space, KS_flat ≈ 0.05 < KS_mu ≈ 0.075) — it returns
`{false, 0, 0, 100}` and the proton row, which is both well-fitting and
directional, is never read. Design goal: let the proton hypothesis declare
the direction in exactly this abstention region, without touching any
decision the current logic already makes.

## 2. Raw material already computed (no new physics needed)

`do_track_comp` per orientation returns:
`[0]` muon-vs-flat direction bit; `[1]` score_mu; `[2]` score_p;
`[3]` score_e — each score `sqrt(ks² + (ratio−1)²)` with
`ratio = Σ(template)/Σ(measured)`, template evaluated at residual range
`end_L − L` (Bragg at the far end of the orientation). Crucially the score
mixes SHAPE and MAGNITUDE, so a MIP-like or shower track cannot fake a
small proton score (its ratio_p ≈ 2–3 ⇒ score_p ≳ 1). The type
competition (`:1424-1445`) already encodes the mu/p/e priority rules
including the <20 cm electron eligibility.

Internal precedent: the `flag_force` branch (`:1477-1488`) already forces
direction+type from the best overall score when both orientations fail the
gate — but the prototype enables it only for isolated segments
(`start_n==1 && end_n==1 && npoints>=15`, `segment_determine_dir_track:1581-1583`),
where no topology fallback exists. Blind forcing for attached segments
would be dangerous (a flat muon's fwd/bwd scores differ by noise ⇒
coin-flip directions fed to vertexing). This design is the guarded,
proton-only extension of that mechanism to attached segments.

## 3. The decision rule

New knob-gated block in `segment_do_track_pid`, entered ONLY at the
current failure path (`flag_forward==0 && flag_backward==0 && !flag_force`
— i.e. where today it abstains; every currently-successful decision is
untouched):

```cpp
// Improvement beyond prototype (doc pr/8): proton-template direction vote.
if (opts.proton_dir_vote) {
    const double sp_f = result_forward.at(2), sp_b = result_backward.at(2);
    const bool fwd_wins = sp_f <= sp_b;
    const double sp_best  = fwd_wins ? sp_f : sp_b;
    const double sp_other = fwd_wins ? sp_b : sp_f;
    const int    win_type = fwd_wins ? forward_particle_type : backward_particle_type;
    const bool   free_end = fwd_wins ? (opts.end_n == 1) : (opts.start_n == 1);
    if (win_type == 2212                                  // G2: proton beats mu AND e in that orientation
        && sp_best < opts.proton_dir_score_max            // G1: good absolute proton fit (shape+magnitude)
        && sp_other > opts.proton_dir_asym_min * sp_best  // G3: significant fwd/bwd asymmetry
        && free_end) {                                    // G4: Bragg (stop) end is a free end
        return std::make_tuple(true, fwd_wins ? 1 : -1, 2212, sp_best);
    }
}
```

Guard rationale:
- **G1** `score_max` — absolute proton consistency. Kills flat muons
  (score_p ≥ 1), showers, MIP stubs. Evt 172230: 0.135.
- **G2** winning orientation's competition must already prefer proton —
  reuses the prototype's own mu/p/e priority (incl. the 20 cm electron
  rule); no new type logic.
- **G3** `asym_min` — the direction claim needs real Bragg asymmetry, not
  noise. Evt 172230: 0.193/0.135 = 1.43.
- **G4** a stopping proton's stop end must be topologically free; also
  guarantees the result persists through the existing store condition
  (`:1679` requires dirsign toward a free end), decoupling this
  improvement from the pr/7 persistence bug decision. Evt 172230:
  fwd + end_n==1 ✓.
- No separate median-dQ/dx guard: magnitude already lives in the score
  (ratio term); fewer parameters to validate.

Score/weakness semantics: `particle_score = sp_best` stored honestly. With
`dir_weak_use_score` ON, `is_dir_weak` for protons reads score >0.13
(≥5 cm) — evt 172230 at 0.135 lands *weak but directed*; sharper Bragg
protons (score <0.13) come out strong. This is intentional: the knob
declares the direction, the existing weakness machinery grades it.

Initial parameter values (to be CALIBRATED before freezing, §6):
`proton_dir_score_max = 0.25`, `proton_dir_asym_min = 1.3`.

## 4. What this deliberately does NOT do

- Never overrides a muon-vs-flat success (any branch that returns today
  still returns identically) — including both-pass ambiguity handling.
- No muon or electron direction recovery: an abstaining flat muon carries
  no direction information by construction; electron direction from dQ/dx
  is unreliable. Proton-only, where the Bragg signature is real.
- Does not touch the `flag_force` isolated-segment path.
- Does not by itself fix the pr/7 §3 persistence bug: median-catch
  protons/muons failing G1–G4 still lose their type on exit. That parity
  restoration remains a separate owner decision (pr/7 §5) — recommended,
  complementary, independently knobbed.
- Kaons: a stopping kaon may pass as 2212 (no kaon template in this
  competition; prototype has the same ambiguity). Accepted.

## 5. Threading and config (mirrors `dir_weak_use_score`, doc pr/6)

- `struct TrackPidOptions { bool proton_dir_vote=false; double
  proton_dir_score_max=0.25; double proton_dir_asym_min=1.3; int
  start_n=1; int end_n=1; };` appended as defaulted last parameter of
  `segment_do_track_pid` and `segment_determine_dir_track` (source- and
  behavior-compatible; default = legacy = byte-identical).
  `segment_determine_dir_track` fills start_n/end_n per segment and
  forwards the knobs to both its 35 cm and 15 cm attempts.
- `PatternAlgorithms` members `m_proton_dir_vote` etc.;
  `NeutrinoTrackShowerSep.cxx:139` builds the options from them. The
  internal call inside `segment_determine_shower_direction_trajectory`
  passes defaults (OFF) — a proton verdict there is wiped to electron
  anyway (prototype `:1662-1665` parity).
- `TaggerCheckNeutrino` config keys `proton_dir_vote`,
  `proton_dir_score_max`, `proton_dir_asym_min`, round-tripped in
  `default_configuration`, copied into pattern_algos next to
  `m_dir_weak_use_score`.
- jsonnet: `cm.tagger_check_neutrino(..., proton_dir_vote=false, ...)`
  with key-suppression (all three keys omitted when the bool is off ⇒
  compiled JSON byte-identical). SBND threading through `clus_pr`/`pr`;
  default OFF until §6/§7 validation, then owner decides SBND ON.
  uBooNE call site untouched initially (`DIR_WEAK`-style TLA can be added
  for the study arm).

## 6. Calibration before freezing thresholds

Blind-tune risk is real (G1/G3 chosen from one event). Plan:
1. Temporary TRACE dump (knob-independent, log-only) in the failure path:
   `(sp_f, sp_b, score_mu_win, win_type, length, start_n, end_n)` for
   every abstaining Track segment.
2. Run the 45 nu-candidate manifest + the doc-62 STM proton set (13
   hand-labeled protons) + evt 172230/444187. Tabulate score_p and
   asymmetry distributions for (a) hand-identifiable protons,
   (b) everything else.
3. Pick `score_max`/`asym_min` with margin between the populations;
   record the table in this doc; then remove the temporary dump
   (instrumentation reverted, same discipline as pr/5).

## 7. Validation matrix (after implementation)

- [ ] `wcdoctest-clus` pass; new focused doctest: synthetic rising-dQ/dx
      profile → recovery fires fwd; flat profile → abstains; reversed →
      bwd. (Guards G1–G4 each exercised.)
- [ ] Compile proofs: 16/16 configs byte-identical knob-off; PR job diff =
      exactly the new keys when on.
- [ ] Off-gates: uBooNE 35-evt ZIPS vs `dirweakon_ub` (DIR_WEAK default);
      SBND evt 172230 + 444187 bit-identical to current arms.
- [ ] Knob-on SBND: evt 172230 → expect pdg 2212, dirsign fwd, segment
      survives as track, cluster leaves all-showers mode; vertex end
      MEASURED (not assumed — direction is weak at score 0.135, so the
      traditional vertexer's behavior is an open measurement).
- [ ] 45-candidate before/after table: pdg flips, vertex moves, hand-scan
      the changed ones.
- [ ] uBooNE knob-on study arm: `fidelity_compare.py` + tagger diffs.
      NOTE this is designed divergence — fidelity vs prototype may
      *worsen* in particle-type columns while physics improves; judge
      changed events by hand, not by the diff count alone.
- [ ] Determinism: pure arithmetic on per-segment vectors, no
      pointer-order dependence introduced.

## 8. Owner sign-off points

1. Guards G1–G4 and the recovery scope (failure path only) — §3.
2. Initial thresholds 0.25/1.3 pending §6 calibration — or calibrate
   first, then freeze.
3. Whether the pr/7 §5 persistence parity fix rides along (separate knob)
   or stays deferred.
4. SBND default ON after validation; uBooNE default after its study.
