# pr/10 — energy calibration round: SBND recombination model, muon dQ/dx-vs-L envelope, kine_* factors

Status: DONE (2026-07-30).  Toolkit `405a0f9a` (gen model) + `21c31439`
(knob plumbing, byte-identical OFF) + `db625c81` (SBND adoption: muon
curve + 3 kine recombination factors).  Staged awaiting owner review:
`use_power_recomb`, `sp_dedx_use_recomb_model`/`sp_mean_dedx_cut` (sec 7).
Closes the plumbing of doc pr/2 §2e(i) row 3 (SinglePhoton inline inverse
Box) and the §2e(iv) muon-curve row (corrected here from 3 to **9** sites);
supplies the first SBND values for the §2e(iii) `kine_*` recombination
factors.  Follow-on to pr/2 (worklist), doc 55 §7g (the SBND recombination
fit), doc 48 (dQ/dx tables + MIP anchors).

## Repro block

```
# gen model unit tests (4 cases / 100 assertions):
cd toolkit && ./build/gen/wcdoctest-gen -tc='*powerbox*'

# knob-off byte-identity, uBooNE PR chain (gate 1 = ZIPS is the gate;
# gate 2 tagger-compare is known non-discriminating, A/A gives 33/35 DIFF):
cd wcp-porting-img/qlport/scripts
./sweep_5384.sh energyoff_ub 6 && ./ab_check.sh energyoff_ub geomoff_ub
#   -> ZIPS: 35/35 content-identical   (baseline geomoff_ub = toolkit cbd78820)

# knob-off compiled-config byte-identity (SBND PR job, all keys unset):
#   base = HEAD versions of {common,sbnd}/clus.jsonnet + wct-pr-perevt.jsonnet
#   cmp base.json new_off.json -> identical, 252680 bytes

# the two derivations (tables from energy_loss/pion_travel/):
cd wcp-porting-img/sbnd/sbnd_xin/dqdx_rr_sample
python3 fit_muon_length_curve.py      # -> muon_length_curve.{tsv,png}
python3 derive_kine_recom_factors.py

# SBND before/after arms (3 nuecc48 events x base/adopt/power), scratch:
#   pr_arm.sh = work-nuecc48-prsmoke2/run_pr3_evt.sh + EXTRA TLAs; see sec 7.
```

## 1. What this round is

Owner request (2026-07-30): now that the `kine_*` constants are config-fed
(pr/2 §2e(iii-a)), do the *energy-related* calibration itself:

1. **dQ/dx → dE/dx via the SBND model inverse** — the doc-55 §7g free-power
   Modified Box (normalization C + "special model"), which needs a numeric
   inverse;
2. **the muon dQ/dx-vs-length cut** — compare MicroBooNE and SBND from the
   stopping tables and derive an SBND version;
3. **the recombination survival factors** — move the uBooNE 0.7/0.5/0.35 by
   the uBooNE→SBND recombination difference.

Also answered here (owner question): *are there other places converting
dQ/dx to dE/dx with an inverse recombination model?*  Census in section 2.

## 2. Census: every dQ/dx→dE/dx conversion in the chain

- **Model-driven (the clean path)** — `clus/src/PRSegmentFunctions.cxx:1243,
  1274` (`segment_cal_kine_dQdx` / `cal_kine_dQdx`) call
  `recomb_model->dE(dQ,dx)` on the component's configured
  `IRecombinationModel` (`sbnd_box_recomb` on SBND, A=1.0/B=0.255 at
  0.5 kV/cm).  Feeds `segment_cal_4mom`, shower kinematics, SSM energies.
- **The one remaining inline inverse** —
  `clus/src/NeutrinoTaggerSinglePhoton.cxx` (was :1499-1507): float inverse
  Modified-Box with A=1.0, B=0.255, ρ=1.38 **frozen at uBooNE's
  0.273 kV/cm**, bypassing the configured model.  Computes the shower-stem
  `shw_sp_vec_{median,mean}_dedx` BDT features and feeds one hard cut,
  `mean_dedx < 2.3 MeV/cm`.  Closed this round behind
  `sp_dedx_use_recomb_model` (section 5).
- **Never converts** — the Bragg-peak PID (`do_track_comp`, STM proton
  checks) compares in dQ/dx space against the `ParticleDataSet` tables;
  recombination is baked into the tables, no inverse anywhere.
- **Flat factors, not a model** — `NeutrinoEnergyReco.cxx` divides summed
  charge by the `kine_*` survival factors (section 6).
- Residual literals, recorded not fixed: section 8.

## 3. `PowerBoxRecombination` — the SBND fitted model, invertible

The doc-55 §7g canonical fit (stored in `nusel_display/stm_ref_dqdx.json`)
is

    R = ln(A + u)/u,   u = k (dEdx / 2.1 MeV/cm)^p
    A = 0.93, k = 0.282371, p = 1.362179, C = 0.855175   (chi2/ndf 0.82)

With p ≠ 1 there is no closed-form inverse, so the toolkit gains a fourth
`IRecombinationModel` in `gen/RecombinationModels.{h,cxx}`:
**`PowerBoxRecombination`**, forward `dQ = C·(R/Wi)·dE`, inverse by
**fixed-count bisection** (60 iterations — deterministic, ASLR-safe) on the
monotone branch `dEdx ∈ (0, dedx_max]`.

Two properties of the function worth knowing (both verified numerically and
covered by `gen/test/doctest_powerbox_recombination.cxx`, 4 cases /
100 assertions):

- the forward **peaks at ~77.4 MeV/cm** at the canonical parameters, so the
  inverse is defined up to `dedx_max = 77` and saturates above (the clus
  callers clamp at 50 MeV/cm anyway);
- below **~0.75 MeV/cm** the forward goes negative (an A<1 artifact far
  below physical LAr dE/dx); the inverse maps non-positive charge to 0.
- round-trip accuracy over [0.8, 50] MeV/cm: < 1e-9 relative (double
  precision floor ~4e-13 in the python reference).

Unlike the three older models, its constructor defaults follow the
*practical-unit* convention the jsonnet actually uses — an unconfigured
instance is the canonical SBND fit, not silently broken (section 8, F1).

Config side: `sbnd_power_recomb` is defined next to `sbnd_box_recomb` in
`cfg/pgrapher/experiment/sbnd/clus.jsonnet` and selected by
`use_power_recomb` (TLA-reachable; default **false** = byte-identical).
Flipping it swaps the model under BOTH `TaggerCheckSTM` and
`TaggerCheckNeutrino`.

**The C question, decided:** C = 0.855175 is kept **inside the model** (it is
part of the fit that the forward reproduces data with), but it is
deliberately **excluded** from the kine-factor ratio (section 6): there it is
degenerate with gain/lifetime calibration, i.e. with `kine_fudge_factor`.

## 4. Muon median-dQ/dx-vs-length envelope: 9 sites, one knob, an SBND fit

The empirical uBooNE cut `0.8866 + 0.9533·(18cm/L)^0.4234` (× the
`mip_dqdx_median` scale) appears at **nine** sites, not the three pr/2
§2e(iv) listed: `NeutrinoTaggerNuMu.cxx:198,296`,
`NeutrinoVertexFinder.cxx:3394`, `NeutrinoTaggerNuE.cxx:402,759,1721,3448`,
`NeutrinoTaggerSSM.cxx:1034`, `NeutrinoTaggerCosmic.cxx:558`.  What it
encodes: the *median* dQ/dx of a stopping muon of length L is Bragg-inflated
at small L and relaxes toward the plateau at large L; the prototype source
still carries the predecessor `0.85+0.95·sqrt(25/L)` in comments at every
site — this is a refit of stopping-muon data, not first principles.

**Plumbing:** one `PatternAlgorithms::m_muon_dqdx_curve` array
`{c0, c1, pivot, power}` + inline `muon_dqdx_cut(length)` /
`muon_dqdx_cut_cm(length)` helpers shaped exactly like the literal
expression (pivot/units-conversion arithmetic chosen so the defaults are
**bit-identical** — 18·cm/L and 18.0/L both reproduce exactly).  Config key
`muon_dqdx_length_curve`→`muon_dqdx_curve` on `TaggerCheckNeutrino`,
TLA-reachable.

**Derivation** (`dqdx_rr_sample/fit_muon_length_curve.py`, outputs
`muon_length_curve.{tsv,png}`):

1. *Method validation on uBooNE*: the median of the uBooNE table dQ/dx over
   rr∈(0,L], normalized by 43e3, tracks the empirical envelope with a mild
   margin ratio g(L) = 1.16–1.32 (mean 1.25) over L = 4–120 cm — the
   uBooNE tune is the table median plus ~25 % acceptance headroom.
2. *Margin-preserving transfer*: SBND table median (0.5 kV/cm, doc-48
   regeneration), normalized by the production `mip_dqdx_median` = 48000,
   times the same g(L), refit with the same form (pivot fixed at 18 cm —
   degenerate with c1):

       muon_dqdx_curve = [0.8826, 1.0587, 18, 0.4745]
       (fit rms 1.5e-3, max deviation 4.5e-3)

   | L (cm) | 5 | 10 | 18 | 30 | 50 | 100 |
   |---|---|---|---|---|---|---|
   | uBooNE envelope | 2.526 | 2.109 | 1.840 | 1.654 | 1.505 | 1.348 |
   | SBND fit        | 2.827 | 2.282 | 1.941 | 1.713 | 1.535 | 1.352 |

   Physics of the difference: at 0.5 kV/cm recombination flattens less of
   the Bragg rise, so short-muon *medians* sit relatively higher; the two
   envelopes converge at long L.

Caveats: 48000 is itself the §2e(ii-a) placeholder (the whole curve scales
as 1/mip_dqdx_median); the tables end at rr = 59.5 cm (fit stops at 120 cm);
the tables' ×0.85 normalization cancels only inside g(L).

**Adopted in SBND production** (owner decision 2026-07-30): section 7.

## 5. SinglePhoton stem dE/dx behind `sp_dedx_use_recomb_model`

Default **false** = the inline float inverse Box, byte-identical.  When
true, `shw_sp_vec_{median,mean}_dedx` go through
`m_recomb_model->dE()` — the same configured model as the rest of the
chain (the model pointer now rides on `PatternAlgorithms`, set in
`TaggerCheckNeutrino::visit`).  The [0,50] MeV/cm clamps are unchanged.

The coupled hard cut is now `sp_mean_dedx_cut` (default 2.3; stored as a
float member so the default compares **bit-identically** to the legacy
`2.3f`).  Threshold transfer (computed in `derive_kine_recom_factors.py`):
2.3 MeV/cm on the inline scale = 58 768 e/cm = **2.23 MeV/cm** on the
official-Box physical scale — i.e. with a model-consistent reconstruction
the equivalent cut is `sp_mean_dedx_cut = 2.23`.  The uBooNE inline scale
was only ~3 % off physical here, because at ~2 MeV/cm the (1.0, 0.255) and
(0.93, 0.212) parameterizations nearly cross.

NOT flipped in production this round — staged with `use_power_recomb`
(section 7).

## 6. `kine_*` recombination factors: table-integrated ratio transfer

The 0.7 (track) / 0.5 (shower) / 0.35 (proton) survival factors are
empirical averages at 0.273 kV/cm (0.7 ≠ pointwise Box R(MIP) ≈ 0.63), so
the transfer keeps their empirical content and moves only the recombination
physics: scale each by

    ratio = R_eff^SBND / R_eff^uB,
    R_eff(L) = Σ R(dEdx)·dE / Σ dE      (energy-weighted mean survival
                                          = exactly Q·W/E_true, the factor
                                          the estimator needs)

integrated over the class dE/dx profile from `stopping.root` (NIST/PDG).
uBooNE side: official Box (0.93, 0.212, ρ=1.38) at 0.273 — the
parameterization the uBooNE calibration is anchored to.  SBND side: the
free-power fit, **C excluded** (degenerate with gain/fudge; the fudge
factors deliberately stay at their uBooNE values).  Profile choice is
second-order because only the ratio enters.

`derive_kine_recom_factors.py` output:

| class | profile | ratio (free-power) | spread | ratio (Box@0.5 x-check) | 0.273-value → SBND |
|---|---|---|---|---|---|
| track  | muon L=10–100 cm     | 1.249 | 1.156–1.369 | 1.183 | 0.70 → **0.87** |
| shower | electron rr<15 cm    | 1.169 | —           | 1.142 | 0.50 → **0.58** |
| proton | proton L=3–30 cm     | 1.453 | 1.437–1.475 | 1.359 | 0.35 → **0.51** |

Adopted values rounded to two decimals in the spirit of the originals
(which are 1–2 significant figures).  The official-Box cross-check lands
3–7 % lower — the free-power model's flatter high-dE/dx survival is a real
feature of the SBND fit, not an artifact.  The track ratio's L-dependence
(±9 %) is the dominant systematic and is quoted, not hidden.

Verification limits, restated from §2e(iii-a): `kine_charge` is **not
persisted** by the SBND standalone PR job — these values change SBND BDT
features (kine-derived ones) but no persisted energy until a tagger-tree
dump exists.  The compiled-config proof + the uBooNE gate are the
regression evidence.

## 7. Adoption and before/after

Production flips this round (each its own commit, separately revertible):

1. `muon_dqdx_curve = [0.8826, 1.0587, 18, 0.4745]` — SBND jobs only.
   NOT bit-identical: nine tagger cuts move.
2. `kine_recom_factor = 0.87`, `kine_shower_recom_factor = 0.58`,
   `kine_proton_recom_factor = 0.51` — SBND jobs only.  Changes
   kine-derived BDT features; nothing persisted moves (see section 6).

Staged, config-ready, NOT flipped (owner review pending):

3. `use_power_recomb = true` (+ `sp_dedx_use_recomb_model = true`,
   `sp_mean_dedx_cut = 2.23`) — the before/after evidence below is the
   review material.

Before/after on nuecc48 events (18253/1/{172230, 235435, 444187}; arms =
the run_pr3_evt.sh command + extra TLAs, all `setarch -R`, `dl_weights=''`;
scores from `tracking-pr.root` T_tagger, Enu from T_kine):

| evt | arm | numu_score | nue_score | kine_reco_Enu (MeV) | mabc hash vs base |
|---|---|---|---|---|---|
| 172230 | base            | -0.455 | 4.301  | 1924.8 | — |
| 172230 | adopt (1+2)     | 0.121  | 4.301  | 1691.9 | DIFF |
| 172230 | power (3)       | -0.455 | 4.301  | 1927.2 | DIFF |
| 235435 | base            | 0.426  | -15.0  | 898.2  | — |
| 235435 | adopt (1+2)     | -0.123 | -15.0  | 774.3  | DIFF |
| 235435 | power (3)       | 0.426  | -15.0  | 898.2  | **identical** |
| 444187 | base            | 0.358  | 4.301  | 1470.1 | — |
| 444187 | adopt (1+2)     | 0.889  | 4.301  | 1367.9 | DIFF |
| 444187 | power (3)       | 0.213  | 4.301  | 1471.0 | DIFF |

Readings:
- **adopt**: Enu drops 12–14 % (charge divided by the larger survival
  factors — the blend of 0.87/0.58/0.51 vs 0.7/0.5/0.35 per event
  composition); numu_score moves with the looser envelope; **no verdict
  flips** (cosmict/numu_cc flags unchanged) on these three events.
- **power**: Enu moves ≤ 0.15 % — consistent, since `kine_charge` uses the
  flat factors, not the model; the model reaches the dQ/dx-integrated
  energies (`cal_kine_dQdx`) and PID, hence the small persisted diffs and
  one shifted numu_score.  One event is bit-identical end-to-end.
- **bit-identity of the 9-site refactor**: a fourth arm passing the uBooNE
  defaults *explicitly* (`muon_dqdx_curve=[0.8866,0.9533,18,0.4234]`)
  hashes identical to base (`8c3b44dc…`), proving the member-based
  expression is arithmetic-for-arithmetic the old literal.
- **adoption-equality proof**: after flipping the defaults in
  `sbnd/clus.jsonnet` + `wct-pr-perevt.jsonnet`, the no-TLA compile is
  `cmp`-identical to the pre-edit compile with the adoption TLAs — the
  production flip is exactly the measured arm, nothing more.

**Correction to a §2e(iii-a) caveat**: `kine_charge` IS persisted on SBND —
in the `T_kine` tree of the pr/3-style `tracking-pr.root` (via
`tagger_output`).  Only `mabc-pr.zip` and the saved pctree are insensitive.
Future kine tuning can regress against T_kine directly.

## 8. Recorded, not fixed (owner-decision list)

- **F1, silent-zero trap**: an *unconfigured* `BoxRecombination` uses
  unit-bearing constructor defaults whose `Wi` comes out negative under
  `dE()`'s practical-unit convention — every energy silently becomes 0.
  Any detector that names the type without a `data:` block hits this.
  The new class avoids the convention; the old one is untouched
  (behavior-preserving rule).
- **Default-argument holes**: `PRSegmentFunctions.cxx:558,568` (in
  `break_segment`) and `:2456` take header-default MIP scales, ignoring
  SBND's configured 56000/48000 (already flagged in §2e(ii-a) residuals).
- **Duplicate `mip_dqdx`**: `TaggerCheckSTM.cxx:606` holds its own copy,
  synced by hand with `TaggerCheckNeutrino`'s.
- **Inert clamps**: `PRSegmentFunctions.cxx:1240,1271` keep literal `43e3`
  in the 1000×MIP outlier rejection.
- **Prototype SSM divergence (porting-dictionary item)**: the prototype's
  `NeutrinoID_ssm_tagger.h:743` writes `18/length/units::cm` — with
  CLHEP cm=10 that is 0.18/L_cm, a **factor-100** error tightening the SSM
  muon cut ~1.8×.  The toolkit port (`NeutrinoTaggerSSM.cxx:1034`,
  `18.0/length` with length already in cm) silently *fixed* it.  The
  toolkit is right; the divergence was undocumented until now.
- **Three Box parameterizations coexist**: PID templates
  (0.93/0.212 ×0.85 via `particle_dataset.jsonnet`), runtime
  `sbnd_box_recomb` (1.0/0.255), gen ctor defaults (0.93/0.212,
  unit-bearing).  The `use_power_recomb` flip would put the runtime model
  on the same fit family as the doc-55 measurement; the PID tables remain
  the doc-55 open item (its §9).

## 9. Commits and gates

Toolkit (`WireCell/wire-cell-toolkit` `apply-pointcloud`):

- `405a0f9a` — gen: `PowerBoxRecombination` + doctest.  Purely additive.
- `21c31439` — clus/cfg: `muon_dqdx_curve` (9 sites), `sp_dedx_use_recomb_model`
  + `sp_mean_dedx_cut`, `sbnd_power_recomb`/`use_power_recomb` threading.
  All defaults OFF ⇒ byte-identical (gates below).
- `db625c81` — sbnd/cfg: production adoption of the muon curve
  `[0.8826, 1.0587, 18, 0.4745]` and the kine recombination factors
  0.87/0.58/0.51.  NOT bit-identical by design; the no-TLA compile is
  cmp-identical to the measured TLA arm.

This repo: the two derivation scripts + `muon_length_curve.{tsv,png}` in
`dqdx_rr_sample/`, this doc, and the pr/2 §2e updates
((i-c)/(iii-b)/(iv-b), the 9-site row correction, tuning-protocol update).

Gates (all reported above, section-level):
- uBooNE qlport ZIPS **35/35 content-identical**, `sweep/energyoff_ub` vs
  `sweep/geomoff_ub` (baseline at toolkit `cbd78820`).  Gate 2: 3/35
  identical = its known non-discriminating behavior, not a gate.
- SBND compiled config, all knobs unset: `cmp` rc=0 vs HEAD (252680 bytes).
- Knobs-on compiled config emits exactly the set keys
  (`PowerBoxRecombination:sbnd_power_recomb` under both taggers,
  `muon_dqdx_curve`, `sp_*`).
- `wcdoctest-gen` 8 cases / 131 assertions (4/100 new);
  `wcdoctest-clus` 49 cases / 565 assertions.
- Freshness: `local/lib/libWireCellClus.so` 16:31 > last source edit 16:30;
  `libWireCellGen.so` 16:25 > 16:21.
