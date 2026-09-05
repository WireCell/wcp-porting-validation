# PDHD up to the STM tagger: the pattern-recognition chain, its physics inputs derived from the calibrated drift velocity, and whether the track/dQ/dx fit needs an effective transverse smearing

**Status (2026-09-05):** PDHD had no pattern-recognition tail at all — no
`pr.jsonnet`, no track-fitting parameter file, no dQ/dx reference tables, no way
to persist the point-cloud tree the PR job reads. This round builds that chain up
to and including `TaggerCheckSTM`, following the PDVD recipe (docs pdvd/25, 29,
42, 44), derives every physics input from PDHD's own calibrated drift velocity,
runs it on 30 events, and answers the question it was built to answer.

**The answer: PDHD is like PDVD, not like SBND — and more so.** The effective
transverse smearing of the charge the fit is compared to is measured at
**c = 3.23 ± 0.14 / 2.55 ± 0.19 / 2.82 ± 0.29 mm** (U/V/W, share-matched joint
fit) against **1.054 / 1.756 / 0.054 mm** configured from the signal-processing
closed form — a factor **3.1 / 1.45 / 52**. Those are the largest effective
widths measured on any of the three detectors, and unlike PDVD and SBND the worst
plane is the **collection** plane. Every 2-D charge metric is correspondingly
worse than PDVD's pre-fix state (footprint bias −0.19/−0.22/−0.31 against PDVD's
−0.23/−0.22/−0.10, χ²/N 23/22/64 against 9.6/6.7/18.7).

**Nothing is flipped.** `pdhd_track_fitting.json` ships with the closed-form seed
values, not the measured ones, because §9 found a first-order defect upstream of
the fit — PDHD's induction planes are **wrapped** (65 % of the wires on an imaging
U/V plane are `segment > 0` continuations, against PDVD's 11.3 %) and the Steiner
retiler's channel lookup does not handle that. Turning the existing
`retile_wrapped_channel_activity` knob on multiplies the Steiner terminal count by
**3.2**, raises the accepted-pass count 175 → 207, and cuts the fit's χ²/N by
28–37 % — but it moves the measured smearing constants by only −1 % / +2 % / −6 %
(§9.1), so the verdict above is not an artefact of the starved Steiner graph.
Settling the wrapped-strip question is item 1 of §10; re-deriving the constants
and proposing them for production is item 2.

---

## 0. Repro

```bash
# pinned binary: /home/xqian/tmp/pdhdstm_libpin (libWireCellClus.so fd273dc8f000780f...,
# = toolkit HEAD 869a554c).  Export LD_LIBRARY_PATH=/home/xqian/tmp/pdhdstm_libpin for every run.
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd

# --- A. the analytic inputs (no reconstruction) -----------------------------
python3 stm/pdhd_transport.py --tsv stm/pdhd_transport.tsv          # sec 4: E, D_L, D_T, add_sigma_L
cd ../../energy_loss/pion_travel                                    # sec 6: the dQ/dx tables
root -l -b -q 'convert_field.C(0.4959, "stopping_ave_dQ_dx_pdhd0496.root", true)'
cd ../docs && python3 emit_jsonnet_dedx.py ../pion_travel/stopping_ave_dQ_dx_pdhd0496.root
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd
python3 docs/scripts/d42_make_ref_dqdx.py                           # + its self-gate

# --- B. the arm: clustering with the pctree, then the PR/STM tail -----------
# work/029107_<evt>_stm0 hold symlinks to the existing imaging + opflash of work/029107_<evt>
for e in $(seq 0 29); do ./run_clus_evt.sh -s stm0 -save-pctree 029107 $e; done
for e in $(seq 0 29); do ./run_pr_evt.sh   -s stm0 -stm -stm-fit    029107 $e; done
# the knob-on arm of sec 9 (same pctrees, retiler fix on)
for e in $(seq 0 29); do PDHD_PR_TLA="-S retile_wrapped_channel_activity=true" \
                         ./run_pr_evt.sh -s stmw -stm -stm-fit 029107 $e; done

# --- C. the measurement ------------------------------------------------------
R="work/029107_*_stm0/tracking-stm.root"
python3 docs/scripts/d44_sigma_fit.py --det pdhd --max-advance 0.436 --nboot 200 \
        --split run,length,rr,foff,advance --out docs/figs/pdhd_sigma $R      # sec 8
for adv in 0.25 0.30 0.60; do python3 docs/scripts/d44_sigma_fit.py --det pdhd \
        --max-advance $adv --nboot 120 --out /home/xqian/tmp/pdhdstm/ctl2/adv$adv $R; done
for hw in 2 4;   do python3 docs/scripts/d44_sigma_fit.py --det pdhd --max-advance 0.436 \
        --halfwidth $hw --nboot 120 --out /home/xqian/tmp/pdhdstm/ctl2/hw$hw $R; done
python3 docs/scripts/d42_proj2d_resid.py --det pdhd --out /home/xqian/tmp/pdhdstm/ana2/resid_stm0 $R
python3 docs/scripts/d42_shape_diag.py   --det pdhd --out /home/xqian/tmp/pdhdstm/ana2/diag_stm0  $R
python3 docs/scripts/d42_ring_frame.py   --det pdhd $R
python3 docs/scripts/d42_dqdx_rr.py --det pdhd --ref stm/pdhd_ref_dqdx.json --ref-key MuonDeDx \
        --out /home/xqian/tmp/pdhdstm/ana2/dqdx_stm0 $R
python3 docs/scripts/pdhd_sigma_plots.py --bins docs/figs/pdhd_sigma_bins.tsv \
        --fit docs/figs/pdhd_sigma_fit.tsv --out docs/figs/pdhd_sigma
# POSITIVE CONTROL (sec 8.1): the same fork must reproduce doc pdvd/44's published PDVD numbers
git -C ../../toolkit show 869a554c^:cfg/pgrapher/experiment/protodunevd/pdvd_track_fitting.json \
        > /home/xqian/tmp/pdhdstm/ctl2/pdvd_tf_preflip.json
python3 docs/scripts/d44_sigma_fit.py --det pdvd --nboot 200 \
        --model-json /home/xqian/tmp/pdhdstm/ctl2/pdvd_tf_preflip.json \
        --out /home/xqian/tmp/pdhdstm/ctl2/pdvd_repro_preflip ../pdvd/work/*_d42fit/tracking-stm.root
```

Committed products: this doc; `docs/scripts/{d44_sigma_fit,d42_proj2d_resid,d42_shape_diag,d42_ring_frame,d42_dqdx_rr,d42_make_ref_dqdx,pdhd_sigma_plots}.py`;
`docs/figs/pdhd_sigma_{bins,fit,shape}.tsv` + the two PNGs; `stm/pdhd_transport.py`,
`stm/pdhd_transport.tsv`, `stm/pdhd_ref_dqdx.json`; `wct-pr-perevt.jsonnet`,
`run_pr_evt.sh`, and the `-save-pctree` additions to `wct-clustering.jsonnet` /
`run_clus_evt.sh`. In the toolkit: `cfg/pgrapher/experiment/pdhd/{pr.jsonnet,
particle_dataset.jsonnet,pdhd_track_fitting.json}` and the `clus.jsonnet`
additions. Regenerated, not committed: the arms `work/029107_*_{stm0,stmw}` and
the analysis products under `/home/xqian/tmp/pdhdstm/`.

---

## 1. What was missing

PDHD ran imaging → clustering → Q/L matching and stopped. Everything downstream —
the Steiner graph, the trajectory + dQ/dx fit, `TaggerCheckTGM/STM/FC` — existed
in `clus/` but had no PDHD binding. Concretely, none of these existed:

| | PDVD | PDHD before | PDHD now |
|---|---|---|---|
| persist the post-Q/L point-cloud tree | `run_clus_evt.sh -save-pctree` | — | §3 |
| PR builder | `protodunevd/pr.jsonnet` | — | `pdhd/pr.jsonnet` (§7) |
| per-event PR job | `pdvd/wct-pr-perevt.jsonnet` | — | `pdhd/wct-pr-perevt.jsonnet` (§7) |
| runner | `pdvd/run_pr_evt.sh` | — | `pdhd/run_pr_evt.sh` (§7) |
| TrackFitting parameters | `pdvd_track_fitting.json` | — | `pdhd_track_fitting.json` (§4, §5) |
| dQ/dx-vs-range reference | `protodunevd/particle_dataset.jsonnet` | — | `pdhd/particle_dataset.jsonnet` (§6) |
| tagger fiducial volume | `pdvd_pr_fv` | — | `pdhd_pr_fv` (§7.1) |

Every one of these is **additive**: no file that any existing PDHD job compiles
was changed in a way that moves its output. The two exceptions are gated in §3
and §7.3.

---

## 2. Forks, and what was deliberately not carried over

Following the house rule (CLAUDE.md §2 Code / M10), everything is a fork **by
duplication** from the PDVD files; `protodunevd/` and `sbnd/` are untouched. Four
PDVD features were deliberately **not** carried into the PDHD copies:

1. **The per-crate drift speed.** PDVD sets one speed per crate and its PR job
   must be handed both, or `add_sigma_L` is mis-scaled by 6 % (doc pdvd/25 §7b).
   PDHD has ONE drift speed and it lives in `params.jsonnet`; there is no TLA and
   no way for the PR job to disagree with the Q/L job. `run_pr_evt.sh` still
   *checks* it against the pctree sidecar (§3), because a `params.jsonnet` edit
   between the two jobs would be silent otherwise.
2. **The curved fiducial surface** (docs pdvd/41, 43). It is a *measured*
   space-charge map from 120 PDVD cosmic events. PDHD has no such measurement, so
   `pdhd/pr.jsonnet` carries only the flat box and the four margin knobs. The
   place to port it back from is named in the file.
3. **`unmerge_assoc`** (doc pdvd/39 round 2). It undoes the `cm.isolated()`
   physical merge the PDVD clustering performs. PDHD's clustering runs no such
   merge, so the stage would be inert; it is absent from `run_pr_evt.sh`'s
   pipelines rather than present-and-silently-doing-nothing.
4. **`tagger_check_neutrino` and its tail** (`tracking_visitor`, `tagger_output`,
   `pr_display`'s dump). The ~700 TLAs that drive it are carried verbatim so a
   later round can turn them on, but they are out of the default pipeline and
   **nothing in that tail has been graded on PDHD**. `-nu` runs it; the output is
   not part of this round.

What was **kept and is now known to matter** is the wrapped-strip family — see
§9. PDVD's own comment says "PDHD runs no Steiner stage", which was true only
because PDHD had no PR chain. It does now, and the knobs are live.

---

## 3. Persisting the point-cloud tree, and the round-trip gate

The PR job reloads the tree the Q/L job wrote. Three additions, all default-off:

- `cfg/pgrapher/experiment/pdhd/clus.jsonnet`: `clus_all_tpc(..., tensor_outname='')`,
  threaded to the all-TPC `TensorFileSink`. `''` keeps the inert `dump_mode` sink
  writing `trash-all-apa.tar.gz`; a path turns the *same* sink into a real
  TensorDM writer. The module also now exports the PR primitives
  (`pc_transforms`, `live_sampler`, `scope_coords`, `t0cor_coords`) as hidden
  fields, so nothing reaches a compiled clustering config.
- `pdhd/wct-clustering.jsonnet`: a `save_tensors` TLA.
- `pdhd/run_clus_evt.sh`: `-save-pctree`, writing
  `work/<run6>_<evt>/pctree-evt<ID>.tar.gz` plus a `.tlas` sidecar (run/subrun/
  event, trigger offset, readout ticks, qlmatch flag, and — read out of the
  *compiled* config so it cannot disagree with what ran — the wires file and the
  drift speed).

**Gate T0 (compiled-config diff-to-zero).** The compiled PDHD clustering job with
`-save-pctree` off, before and after all of the above:

```
md5  ed61bc688d114f84d39eea2c47ef7f6d   (HEAD cfg + HEAD wct-clustering.jsonnet)
md5  ed61bc688d114f84d39eea2c47ef7f6d   (this round's cfg + wct-clustering.jsonnet)
```
`cmp` rc=0 on 117 482 bytes. Re-run and still passing after the §9 knob was added.
**Knob-on proof:** with `save_tensors` set, the sink compiles to
`{dump_mode: false, outname: work/.../pctree-evt0.tar.gz, prefix: clustering_}`.

**Gate T1 (round trip).** `run_pr_evt.sh -pipe switch_scope` on 029107 evt 0
against the Q/L job's own `mabc-all-apa.zip`, comparing the `clustering-global`
Bee layer:

| | result |
|---|---|
| number of points | 109 221 = 109 221 |
| `q`, `y`, `z` arrays | **byte-identical** |
| point→cluster partition | **identical**, 98 clusters, same membership |
| `cluster_id` labels | permuted |
| `x` | max \|Δ\| **0.101 mm**, constant per cluster on 94 of 98 |

The id permutation is a *choice*, not a defect: PDHD's clustering job leaves
`cluster_id_order` at the C++ default and the forked PR job sets `'tree'`, exactly
as PDVD's pair does. Fork fidelity was kept over an exact gate; if a later round
wants the gate exact, set `cluster_id_order` the same way in both. The x residual
is 1/47 of a wire pitch (0.064 µs of t0) and changes no conclusion in this doc.
Anything joining PR output to clustering output must join on (y, z), never on the
cluster number — the same rule as `feedback_retile_ident_is_not_bee_cluster_id`.

---

## 4. The drift field and the diffusion coefficients, from the calibrated velocity

PDHD's drift speed is calibrated: **1.576 mm/µs**, from the four evt-983
cathode-crossing tracks (`cfg/pgrapher/experiment/pdhd/params.jsonnet`
`lar.drift_speed`, `clus.jsonnet`'s `local drift_speed`, and
`pdhd/docs/clustering-algorithm.md`). Unlike PDVD — where the production Q/L speed
(1.48073) and `params.jsonnet` (1.568) disagree and neither inverts to the adopted
field (doc pdvd/29 §3) — PDHD's two live sites agree, so the chain below is
self-consistent.

`stm/pdhd_transport.py` (fork of `pdvd/stm/pdvd_transport.py`) inverts that speed
through the BNL LAr-properties mobility parameterisation and evaluates the
transport coefficients at the resulting field:

```
mu(E,T)    = (a0 + a1 E + a2 E^1.5 + a3 E^2.5)/(1 + (a1/a0) E + a4 E^2 + a5 E^3) * (T/89)^-3/2
eps_L(E,T) = (b0 + b1 E + b2 E^2)/(1 + (b1/b0) E + b3 E^2) * (T/87)
D_L = mu * eps_L        D_T = D_L / (1 + (E/mu) dmu/dE)
```

| input | value | source |
|---|---|---|
| v_drift | 1.576 mm/µs | PDHD calibration (params.jsonnet) |
| T | 87.68 K | dunecore `protodunehd_detproperties.Temperature` |
| **E (BNL)** | **0.4959 kV/cm** | inverted |
| E (LArSoft Walkowiak/ICARUS branch) | 0.4942 kV/cm | cross-check, 0.3 % apart |
| **D_L** | **4.1207 cm²/s** | |
| **D_T** | **8.2017 cm²/s** | |

The dunecore **nominal** PDHD field is 0.4867 kV/cm, at which the same
parameterisation gives v = 1.5612 mm/µs — so the calibration sits ~1 % high in
velocity and ~1.9 % high in field. Temperature is a soft input: at the BNL site
default 87.3 K the same velocity gives E = 0.4895, D_L 4.129, D_T 8.177, i.e.
**< 0.4 % on D_T**. Full table in `stm/pdhd_transport.tsv`.

> **Flagged, not fixed.** `cfg/pgrapher/experiment/pdhd/params.jsonnet` still
> inherits the *generic* base defaults `lar.DL 7.2 / lar.DT 12.0 cm²/s /
> lar.lifetime 8 ms` from `pgrapher/common/params.jsonnet`. Those have no
> connection to PDHD's field — but unlike PDVD, where they were inert, PDHD's
> **simulation consumes them** through `pgrapher/common/sim/nodes.jsonnet`
> (`pdhd/sim.jsonnet` imports it). Setting them would move every PDHD sim job's
> compiled config, which is an owner decision (CLAUDE.md §5 rule 1), so this
> round records the physical values in `pdhd_track_fitting.json` and leaves
> `params.jsonnet` alone. dunecore also records an electron lifetime of 35 ms for
> PDHD against the inherited 8 ms.

---

## 5. The smearing terms that come from the signal-processing chain

`TrackFitting` builds the predicted charge of a trajectory point as a 2-D Gaussian
whose widths are

```
sigma_L   = hypot( sqrt(2 D_L t_drift), add_sigma_L )      / tick_width
sigma_T_p = hypot( sqrt(2 D_T t_drift), c_p )              / pitch_p        p = U, V, W
```
with `t_drift = max(50 µs, |x - x_anode| / v)` (`TrackFitting.cxx:7283-7315`;
`x_anode` is the collection plane's own x, `iface->planes()[2]->wires().front()->center().x()`).
`add_sigma_L` and the three `c_p` are the **non-diffusion** terms, and they are
where the signal processing enters.

**Longitudinal — a clean derivation.** `add_sigma_L = 1/(2π·σ_Gaus_wide)·v`.
`Gaus_wide` is the one SP time filter `OmnibusSigProc` consumes (`Gaus_tight` is
dead config). PDHD's is 0.12 MHz, the same as PDVD's:

```
add_sigma_L = 1/(2π · 0.12 MHz) × 1.576 mm/µs = 1.32629 µs × 1.576 = 2.0902 mm
```
(PDVD 1.9639, SBND 2.4876 — SBND's filter is 0.10 MHz and its drift 1.563.)

**Transverse — a seed, not a derivation.** The historical closed form is
`c_p = (1/√π)/Wire_p × pitch_p × {0.3 U, 0.5 V, 0.2 W}`, from
`pdhd/sp-filters.jsonnet`'s `wf('Wire_ind', σ = (1/√π)·0.75)` and
`wf('Wire_col', σ = (1/√π)·10.0)`:

| | PDHD | PDVD | SBND |
|---|---|---|---|
| `Wire_ind` / `Wire_col` (filter σ multiplier) | **0.75** / 10.0 | 5.0 / 10.0 | 1.05 / 3.60 |
| true spatial σ of the wire filter, induction | **1.756 mm (0.376 pitch)** | 0.432 mm (0.056) | 0.806 mm (0.269) |
| true spatial σ of the wire filter, collection | 0.135 mm (0.028 pitch) | 0.144 mm (0.028) | 0.235 mm (0.078) |
| closed-form seed c_U / c_V / c_W [mm] | **1.054 / 1.756 / 0.054** | 0.259 / 0.432 / 0.058 | 0.484 / 0.806 / 0.094 |

(The `0.5` factor in the closed form is exactly the true filter σ; `0.3` and `0.2`
are ad-hoc reductions of it. Note that a *larger* `wf` σ means a *narrower*
kernel in wire space — `feedback_frequency_sigma_is_not_a_width`. PDHD's induction
filter is therefore the **widest** of the three, and its collection filter as
narrow as PDVD's.)

**That closed form is WITHDRAWN as a derivation** (doc pdvd/42 §8.2/8.6/8.10,
doc pdvd/44): these keys are structurally the *entire* non-diffusion transverse
width of the charge the fit is compared to, not the filter term, and the closed
form breaks above Nyquist. It is used here only to give PDHD a defined starting
point for the first arm — the same footing SBND and PDVD started from — and §8
measures what it should be. `pdhd_track_fitting.json` says so in its own comment.

---

## 6. dQ/dx vs residual range: the reference tables at the derived field

`cfg/pgrapher/experiment/pdhd/particle_dataset.jsonnet` (new) carries the five
`*DeDx` LinterpFunctions regenerated at **E = 0.4959 kV/cm** with the repo's own
recipe, following doc pdvd/25 §7c / doc pdvd/29 §4 exactly and changing only the
field argument:

```
root -l -b -q 'convert_field.C(0.4959, "stopping_ave_dQ_dx_pdhd0496.root", true)'
python3 emit_jsonnet_dedx.py ../pion_travel/stopping_ave_dQ_dx_pdhd0496.root
```

Modified Box, unchanged parameters (α 0.93, β 0.212, ρ 1.38, W_ion 23.6 eV, and
the undocumented `fudge = 0.85` retained per the file header); only
`β' = β/(ρE)` moves. The five `*Range` tables are detector-independent and are
copied from the PDVD/SBND file unchanged.

| quantity | PDHD @ 0.4959 | PDVD @ 0.45 | at PDHD's nominal 0.4867 |
|---|---|---|---|
| muon plateau (rr = 59.5 cm) | 54 609.2 e/cm | 53 965.5 | 54 423 (−0.34 %) |
| muon Bragg bin (rr = 0.5 cm) | 167 509 e/cm | 159 982 | — |
| MIP (dE/dx = 2.1 MeV/cm) | 53 222.5 e/cm | 52 635.2 | — |

**Generator gate (`d42_make_ref_dqdx.py`):** the tables read back from the ROOT
file equal the *compiled* `pdhd/particle_dataset.jsonnet` to max relative
1 × 10⁻⁵ (the 6-significant-figure print of `emit_jsonnet_dedx.py`) before
`stm/pdhd_ref_dqdx.json` is written. PASS.

**`mip_dqdx` / `mip_dqdx_median`.** The SBND rule is `mip_dqdx = plateau ×
1.02456`, rounded to the nearest 1000: `54 609.2 × 1.02456 = 55 951 → 56 000`,
and the median threshold scales as `48 000 × 54 609.2/54 657.7 = 47 957 → 48 000`.
These *coincidentally* equal SBND's raw values — the arithmetic is shown so it is
not read as a copy. (PDVD's plateau is 1.2 % lower and lands on 55 000/47 000.)

**Against data: derived and consistent, NOT confirmed.** `d42_dqdx_rr.py` on the
30-event arm:

| tier | n tracks | n points | k (population) | χ²/11 |
|---|---|---|---|---|
| all status-0 passes | 170 | 52 163 | 0.888 | 3275 |
| contrast ≥ 2 | 17 | 6 941 | 0.784 | 145 |
| doc-55 muon cuts | **1** | 258 | 1.028 | 35.1 |

The clean-stopping-muon tier is **n = 1** against PDVD's n = 45, and the
`all_status0` tier is a mixed population whose k should not be quoted as a charge
scale. The honest statement is that the field is *derived* from the velocity and
*self-consistent* with the recombination model and the tables, and that
confirming it on data needs many more imaged PDHD events — a sample question,
separate from the smearing question this round set out to answer. See §10.

---

## 7. The PR builder, the driver, and the fiducial volume

### 7.1 The fiducial volume is the Q/L fiducial volume, by construction

`pdhd/pr.jsonnet` builds one `BoxFiducial` spanning both drift volumes (so a
cathode crosser is not an "exiter" at x = 0), with the **active** bounds, and puts
the insets back through the taggers' `fv_tolerance`. The arithmetic closes exactly
on `pdhd/clus.jsonnet`'s `dvm`, which is what the Q/L matching step uses — the
same construction PDVD adopted in doc pdvd/35:

| axis | `pdhd_pr_fv` box | margin | net | `dvm` ± its own margin |
|---|---|---|---|---|
| x | ± 357.985 cm (dvm `a0f0pA` FV_x) | 2 | ± 355.985 | 357.985 − FV_x_margin 2 ✓ |
| y | 7.61 … 606.0 cm (active) | 17.5 = 15 + 2.5 | 25.11 … 588.50 | FV_y 22.61…591.0 ∓ 2.5 ✓ |
| z | 0.234 … 462.297 cm (active) | 18 = 15 + 3 | 18.234 … 444.297 | FV_z 15.234…447.297 ∓ 3 ✓ |

The 15 cm y/z inset is the space-charge allowance the clustering FV already
carries. `tgm_fv_zmax_margin_interior = 0`, so "contained" has one meaning across
the endpoint and interior tests. The one deviation from PDVD: PDVD uses an x
margin of 2.5 cm (SBND's value); PDHD uses 2, its own `FV_x_margin`, so the
closure is exact on all three axes. Compiled proof: `TaggerCheckSTM`'s
`fv_tolerance` is `[-20, -20, -175, -175, -180, -180]` (WCT mm).

### 7.2 The rest of the operating point

`pdhd/wct-pr-perevt.jsonnet` sets it explicitly; the pr() defaults keep the
SBND/PDVD values as documentation.

| knob | PDHD | why |
|---|---|---|
| `anode_indices` | 0–3 | 4 APAs |
| `readout_window_ticks` | 6000 (runner passes the real 5999) | PDHD readout |
| `trackfitting_config` | `pgrapher/experiment/pdhd/pdhd_track_fitting.json` | §4/§5. The path **is** WIRECELL_PATH-resolved (`TaggerCheckSTM::load_trackfitting_config`); `sbnd_track_fitting.json`'s comment saying otherwise is stale |
| `mip_dqdx` / `mip_dqdx_median` | 56000 / 48000 | §6 |
| `cathode_x` | 0 | PDHD cathode |
| `pr_y_top` | 606.0 cm | top of the active volume (PDVD 336.4, SBND 200.0) |
| `beam_window_us` / `beam_window_only` / `nu_per_bundle` | readout-wide / true / true | PDVD parity: evaluate **every** matched flash bundle, not one arbitrary cosmic |
| every `stm_*_guard` | off | PDVD parity: STM-tagged is the SIGNAL here, not a veto; each guard must be re-derived on PDHD hand scans |
| `dl_weights` | `''` | geometric vertex only; the uBooNE-trained SCN net is ungraded on PDHD and is not bit-stable (CLAUDE.md M4). PDVD flipped it on 2026-09-04; pass the path (and preload libpython) to try it |
| `steiner_terminal_charge` | unset ⇒ C++ 4000 e | SBND's operating point. PDVD needed 500 e for its 7.65 mm pitch; PDHD's 4.67 mm pitch carries ~1.6× SBND's charge per point, so 4000 is if anything easier to pass. A PDHD doc-37 round is owed (§10) |
| `ctpc_aniso_metric` | off | PDHD's pitch/slice-step ratio is 1.48/1.48/1.52, between SBND's 0.96 and MicroBooNE's 1.36, far from PDVD's 2.58 that motivated the metric (doc pdvd/34, 36) |
| `wrapped_channel_charge`, `retile_wrapped_channel_activity` | off | §9 — the single largest known gap |

`run_pr_evt.sh` modes: `-stm` (default, the chain of this round), `-nu`
(ungraded), `-empty` (the §3 gate), `-pipe`. `-stm-fit` adds the
`tracking-stm.root` writer. Note `pr_display` sits in the `-stm` pipeline for PDVD
parity but is **inert** there — it warns `no TrackFitting in grouping 'live'` and
writes no `calib-pr-evt*.json`, because the dump reads the fit
`TaggerCheckNeutrino` builds.

### 7.3 Gates on the PR side

| gate | what | result |
|---|---|---|
| P1 | compiled PDHD PR job with `wrapped_channel_charge`/`retile_wrapped_channel_activity` unset, before and after they were added to the fork | `cmp` rc=0 — the key-suppression idiom holds, and the arm already running was unaffected |
| P2 | knob-on compiled proof | `BlobSampler.wrapped_channel_charge: true` on all 8 samplers, `ImproveCluster_2.wrapped_channel_activity: true` |
| P3 | the whole chain runs | 30/30 events, rc=0, 18 s and 2.2 GB peak per event, 503 STM fit passes, **175 accepted (status 0)**, 98 STM + 98 TGM verdict lines on evt 0 |

`pdhd_track_fitting.json` is read at **runtime**, so a compiled-config hash proves
nothing about it; it is graded on outputs (§8).

---

## 8. The measurement: does the PDHD fit need an effective transverse smearing?

Arm `stm0`: run 029107, events 0–29, 175 accepted STM fit passes, 46 256
profiles. The estimator is doc pdvd/44 §2.1 unchanged — per (block, plane, time
slice) of an accepted pass, the ±3-wire window around the fitted trajectory,
prolonged segments only, dead channels excluded, resolved in the fit's own drift
time, with σ per drift bin matched to the **share** of charge in the wire nearest
the measured centroid (the rms-matched variant is reported beside it).

Two things had to be re-derived for PDHD before any of it could be read:

- **The channel→plane map is not the LArSoft channel number.**
  `PdvdMagnifyTrackingVisitor` — the visitor `pdhd/pr.jsonnet` binds — writes
  `pu/pv/pw` and `T_proj_data.channel` as `base[plane] + the RANK of the wire's
  channel among that plane's channels over the whole detector`
  (`PdvdMagnifyTrackingVisitor.h` `ChanScheme`). PDHD has 3200 U, 3200 V and 3840
  W channels, so the thresholds are **(3200, 6400)** — exactly as PDVD's
  (3808, 7616) are its per-plane counts. Reading them as raw LArSoft channel ids
  (U 0–799, V 800–1599, W 1600–2559 per 2560-channel APA block) *looks* plausible
  — it produces a sane-looking 59k/52k/182k plane split — and mislabels every
  plane. This cost one full analysis pass.
- **The advance cut.** doc 44 keeps "prolonged" segments at
  |Δwire/Δslice| < 0.25 on PDVD. That is a wires-per-slice number, so the same
  value means a different *angle* on a detector with a different pitch and drift
  step. The geometric equivalent of PDVD's cut on PDHD is
  `0.25 × (7.65/4.6693) × (3.152/2.9615) = 0.436`, used as primary; 0.25, 0.30
  and 0.60 are run as controls.

### 8.1 Positive control: the fork reproduces doc pdvd/44

The PDHD copy of `d44_sigma_fit.py`, run with `--det pdvd` on PDVD's own `d42fit`
arm with the **pre-flip** model (`git show 869a554c^:…/pdvd_track_fitting.json`,
DT 7.9135, c 0.259/0.4316/0.0575):

| estimator | D_T,eff [cm²/s] | c_U | c_V | c_W [mm] | χ²/ndf |
|---|---|---|---|---|---|
| this fork, rms-matched | 2.78 ± 1.26 | 2.90 | 2.87 | 2.03 | 12.4/14 |
| **doc pdvd/44 §2.2, rms-matched** | **2.78 ± 1.26** | **2.90** | **2.87** | **2.03** | **12.4/14** |
| this fork, share-matched | 4.87 ± 0.55 | 2.30 | 2.30 | 1.18 | 12.9/14 |
| **doc pdvd/44 §2.2, share-matched** | **4.87 ± 0.55** | **2.30** | **2.30** | **1.18** | **12.9/14** |

Digit for digit. Every PDHD number below rests on this.

### 8.2 PDHD, per drift bin

`docs/figs/pdhd_sigma_bins.tsv`; six equal-population drift bins per plane,
t from ~200 to ~2070 µs (PDHD's lever arm is longer than PDVD's 187→1831).

| plane | t bin [µs] | rms meas / pred [mm] | σ_model [mm] | σ_eff rms [mm] | σ_eff share [mm] | centre share |
|---|---|---|---|---|---|---|
| U | 198 → 2068 | 3.82–4.55 / 2.31–3.02 | 1.20 → 2.12 | 3.26 ± 0.26 → 3.78 ± 0.14 | 3.08 ± 0.21 → 3.45 ± 0.15 | 0.473 → 0.433 |
| V | 276 → 2019 | 3.75–4.20 / 2.74–3.54 | 1.88 → 2.53 | 3.17 ± 0.20 → 3.31 ± 0.28 | 2.60 ± 0.28 → 1.96 ± 0.53 | 0.521 → 0.523 |
| W | 362 → 2045 | 4.31–5.24 / 1.78–2.53 | 0.77 → 1.83 | 4.85 ± 0.62 → 4.56 ± 0.57 | 3.23 ± 0.67 → 3.11 ± 0.56 | 0.487 → 0.496 |

![](figs/pdhd_sigma_fit.png)

### 8.3 The joint fits

`σ_eff² = 2 D_T,eff · t + c²`, one D_T shared by the three planes and three c's
(`docs/figs/pdhd_sigma_fit.tsv`):

| estimator | D_T,eff [cm²/s] | c_U [mm] | c_V [mm] | c_W [mm] | χ²/ndf |
|---|---|---|---|---|---|
| rms-matched | 6.72 ± 3.25 | 3.44 ± 0.15 | 3.19 ± 0.14 | 4.37 ± 0.27 | 6.4/14 |
| **share-matched (primary)** | **3.68 ± 3.03** | **3.23 ± 0.14** | **2.55 ± 0.19** | **2.82 ± 0.29** | **8.6/14** |
| configured | 8.2017 | 1.054 | 1.756 | 0.054 | — |

Three readings.

1. **The constant dominates, and it is the largest measured on any detector.**
   3.23 / 2.55 / 2.82 mm against 1.05 / 1.76 / 0.05 configured — factors of
   **3.1 / 1.45 / 52**.
2. **D_T,eff is consistent with zero and with the physical 8.20 within ~1.5 σ.**
   With c that large the width barely grows over the whole drift, so the drift
   axis carries almost no information here. The honest statement is that the
   width is **drift-independent within errors and the joint-fit c is the number** —
   the same conclusion doc 44 reached on PDVD. Do not read D_T,eff as physics.
3. **The two estimators disagree by 6–55 %**, and the disagreement is worst on W:
   that is the shape check saying the W profile is not a Gaussian (§8.4).

### 8.4 The shape check

Ring shares of the stacked, centroid-aligned prolonged profiles
(`docs/figs/pdhd_sigma_shape.tsv`):

| | centre | ±1 | ±2 | beyond | Σ\|meas − Gaussian\| |
|---|---|---|---|---|---|
| PDHD U measured | 0.433 | 0.470 | 0.084 | 0.013 | — |
| Gaussian, configured | 0.572 | 0.418 | 0.010 | 0.000 | 0.278 |
| Gaussian, rms-matched | 0.410 | 0.486 | 0.097 | 0.006 | 0.059 |
| Gaussian, share-matched | 0.433 | 0.484 | 0.080 | 0.003 | **0.029** |
| PDHD V measured | 0.506 | 0.413 | 0.067 | 0.014 | (cfg 0.124 / rms 0.155 / share **0.089**) |
| PDHD W measured | 0.511 | 0.353 | 0.097 | 0.039 | (cfg 0.392 / rms 0.321 / share **0.200**) |

![](figs/pdhd_sigma_shape.png)

Share-matched wins on every plane, as on PDVD, so it is the primary set. But note
the absolute quality: 0.029 on U and 0.089 on V are PDVD-class agreement, while
**0.200 on W is not** — PDHD's collection profile puts 3.9 % of its charge beyond
±2 wires where a Gaussian of the same core puts 0.0 %, against PDVD's 0.2 %. The
W profile is genuinely non-Gaussian, and no single σ will describe it.

### 8.5 Stability

Joint share-matched c [mm], all on the same 175 blocks:

| occasion | c_U | c_V | c_W |
|---|---|---|---|
| **primary, advance < 0.436** | **3.23** | **2.55** | **2.82** |
| advance < 0.25 (PDVD's literal cut) | 3.32 | 2.81 | 3.01 |
| advance < 0.30 | 3.29 | 2.78 | 2.96 |
| advance < 0.60 | 3.14 | 2.56 | 2.67 |
| window ±2 | 3.05 | 2.48 | 2.30 |
| window ±4 | 3.36 | 2.71 | 3.29 |
| advance sub-bins < 0.10 / 0.10–0.25 / 0.25–0.5 | 3.37 / 3.21 / 2.98 | 2.89 / 2.79 / 2.20 | 3.91 / 2.80 / 2.25 |
| track length < 100 cm / ≥ 100 cm | 3.24 / 3.13 | 1.76 / 2.51 | 3.29 / 2.57 |
| plateau rr > 30 cm / Bragg rr < 10 cm | 3.20 / 2.85 | 2.56 / 2.15 | 2.67 / 3.07 |

c_U holds to ±3 % across the advance cut and ±5 % across the window — PDVD-class
stability. c_V and c_W are looser: ±5 % / ±6 % across the advance cut but **±18 %
on W across the window**, against PDVD's ±3 %. That is the same non-Gaussian W
tail as §8.4 showing up as a window dependence, and it is the reason no constant
is proposed for production here even before §9.

### 8.6 The 2-D pixel charge, and the comparison that answers the question

Medians over the 175 accepted passes, footprint = ±1 cell of the trajectory
(`d42_proj2d_resid.py`), beside doc pdvd/44's published PDVD and SBND columns:

| metric | **PDHD** | PDVD before its flip | PDVD after | SBND |
|---|---|---|---|---|
| B_foot U / V / W | **−0.191 / −0.220 / −0.305** | −0.228 / −0.219 / −0.100 | −0.166 / −0.175 / −0.093 | −0.074 / −0.073 / −0.091 |
| U_foot U / V / W | **0.529 / 0.570 / 0.501** | 0.363 / 0.370 / 0.242 | 0.313 / 0.340 / 0.220 | 0.232 / 0.270 / 0.186 |
| χ²/N U / V / W | **23.1 / 21.6 / 63.7** | 9.56 / 6.74 / 18.7 | 7.94 / 5.95 / 14.7 | 6.80 / 6.56 / 11.0 |
| pull rms U / V / W | **3.62 / 3.52 / 5.55** | 2.18 / 1.74 / 2.82 | 1.86 / 1.65 / 2.42 | 1.86 / 1.68 / 2.42 |
| f_off U / V / W | **0.773 / 0.761 / 0.944** | 0.343 / 0.342 / — | 0.342 / 0.342 / — | 0.059 / 0.058 / 0.046 |

And the first-neighbour share the model predicts against the measured one
(`d42_ring_frame.py`, r = 1st/(centre + 1st)):

| | U | V | W |
|---|---|---|---|
| PDHD measured | 0.324 | 0.314 | 0.285 |
| PDHD predicted | 0.255 | 0.263 | 0.204 |
| gap | **+0.069** | **+0.051** | **+0.081** |
| PDVD gap before its flip | +0.109 | +0.091 | +0.065 |
| PDVD gap after | +0.031 | +0.015 | +0.032 |

**Verdict.** PDHD behaves like PDVD, not like SBND:

- the model under-predicts the first-neighbour share on every plane, by 0.05–0.08
  — the same sign and comparable size to PDVD's pre-fix 0.07–0.11, and nothing
  like SBND, whose own derived constants moved nothing outside noise;
- the measured effective width exceeds the configured one by 3.1× / 1.45× / 52×;
- every 2-D metric is worse than PDVD's *pre-fix* state, and W is worse than any
  plane on either detector.

The plane ordering is inverted relative to PDVD, and §5's table says why: PDHD's
`Wire_ind` of 0.75 makes its induction seed the largest of the three detectors
(already at SBND's *measured* level on V), while `Wire_col` 10.0 leaves the
collection seed as small as PDVD's. So the induction planes start close and the
collection plane starts 52× short. That much of the SP-filter-width story survives
the measurement. What does **not** survive is any attempt to predict the absolute
c from the filter: measured/true-filter-σ is 1.8/1.5/21 on PDHD, 5.3/5.3/8.2 on
PDVD and a uniform ~1.6 on SBND. There is no law here — the effective width is a
property of the whole SP + imaging + charge-solving chain, which is exactly doc
44's conclusion, now with a third detector behind it.

One number in the table above is unlike anything on PDVD or SBND and is not a
smearing statement: **f_off** — the fraction of the block's charge lying beyond a
Chebyshev-2 window of the fitted trajectory — is 0.77/0.76/0.94 on PDHD against
0.34 on PDVD and 0.06 on SBND. Three quarters of the charge in an accepted STM
block is nowhere near the trajectory that was fitted to it. That is not a kernel
width; it is the fit tracking a fragment of its cluster. §9 is about why.

---

## 9. The wrapped induction planes, and why no constant is proposed yet

**PDHD APAs wrap their induction planes.** Read straight out of
`protodunehd-wires-larsoft-v1.json.bz2`: every (anode, face, U or V) plane holds
**1148 wires on 800 channels** — 400 `segment 0`, 400 `segment 1`, 348
`segment 2`. So **65 % of the wires on an imaging induction plane are
continuations**, against PDVD's 11.3 % and SBND's and MicroBooNE's zero.

`IWirePlane::channels()` omits those channels (`AnodePlane.cxx:244-247`;
`feedback_channel_list_is_not_a_wire_lookup`), so any code that indexes that list
by *wire* index reads the wrong channel. Doc pdvd/31 found this at three sites and
knobbed two of them; PDVD flipped both to production. PDVD's own comment says
"PDHD runs no Steiner stage" — true only until this round. Both knobs are now
carried in `pdhd/pr.jsonnet` and `pdhd/wct-pr-perevt.jsonnet`, **default off**,
key-suppressed (gate P1/P2 in §7.3).

**Knob-on smoke run**, `retile_wrapped_channel_activity=true`, same pctree, same
binary, 029107 evt 0 (`work/029107_0_stmw`):

| | off (`stm0`) | on (`stmw`) |
|---|---|---|
| Steiner terminals (Bee layer points) | 2 657 | **8 515** (×3.2) |
| Steiner graph points | 13 419 | **42 996** (×3.2) |
| `stm_fit` trajectory points | 4 407 | **14 200** (×3.2) |
| STM fit passes | 14 | 30 |
| accepted (status 0) | 4 | 3 |
| wall / peak RSS | 18 s / 2.2 GB | 39 s / 4.2 GB |

The Steiner terminal finder is recovering 3.2× more terminals with the fix. This
is not a footnote: the trajectory the §8 profiles are measured around is built
from that graph, and f_off = 0.77–0.94 says the trajectory is not following the
charge.

**Why the two knobs are not symmetric.** `retile_wrapped_channel_activity` acts
inside `ImproveCluster_2`, which is new on PDHD — there is no legacy to preserve
and nothing upstream to stay consistent with. `wrapped_channel_charge` acts in
`BlobSampler`, and the PR job's samplers must sample the pctree the *Q/L job*
wrote the same way that job did; no PDHD Q/L job has ever run with it on, so
turning it on in the PR job alone would make the two disagree. Fixing that one
properly means re-running PDHD clustering with it, which changes PDHD's
production clustering and Q/L output — an owner decision, not this round's.

**Consequence for §8.** The constants in §8.3 are measured on a chain whose
Steiner input is demonstrably starved. They are reported, and they are enough to
answer the question that was asked — PDHD is PDVD-like, not SBND-like, by a wide
margin and on three independent metrics — but they are **not** proposed for
`pdhd_track_fitting.json`, which ships with the §5 seeds. §9.1 measures how much
they move under the retiler fix.

### 9.1 The same derivation on the retiler-fixed arm — the answer does not move

Arm `stmw`: the SAME 30 pctrees, the same pinned binary, `-stm -stm-fit` with
`retile_wrapped_channel_activity=true`. 30/30 rc=0. STM fit passes 503 → **841**,
accepted (status 0) 175 → **207**, prolonged profiles 12 533 → **19 749**.

| joint share-matched | D_T,eff [cm²/s] | c_U [mm] | c_V [mm] | c_W [mm] | χ²/ndf |
|---|---|---|---|---|---|
| `stm0` (retiler fix off) | 3.68 ± 3.03 | 3.23 ± 0.14 | 2.55 ± 0.19 | 2.82 ± 0.29 | 8.6/14 |
| `stmw` (retiler fix on) | 4.03 ± 2.48 | **3.20 ± 0.09** | **2.60 ± 0.15** | **2.64 ± 0.24** | 6.1/14 |
| shift | — | −1 % | +2 % | −6 % | |
| rms-matched, for completeness | 4.18 ± 2.90 | 3.47 | 3.32 | 4.35 | 7.9/14 |

**The constants move by less than the §8.5 systematic on every plane**, on a
sample 18 % larger and with a better χ². So the §8 verdict is not an artefact of
the starved Steiner graph: PDHD's effective transverse smearing really is ~3.2 /
2.6 / 2.7 mm, 3× / 1.5× / 50× the configured values.

The fit *quality* does improve, and substantially — which is the reason to pursue
the knob on its own merits, not because it changes the smearing answer:

| median over accepted passes | `stm0` U / V / W | `stmw` U / V / W |
|---|---|---|
| B_foot | −0.191 / −0.220 / −0.305 | −0.161 / −0.192 / −0.298 |
| U_foot | 0.529 / 0.570 / 0.501 | 0.535 / 0.619 / 0.470 |
| χ²/N | 23.1 / 21.6 / 63.7 | **16.7 / 14.4 / 40.2** |
| pull rms | 3.62 / 3.52 / 5.55 | **2.96 / 3.02 / 4.32** |
| f_off | 0.773 / 0.761 / 0.944 | 0.676 / 0.677 / 0.959 |
| dQ/dx k, all status-0 (n) | 0.888 (170) | 0.927 (205) |
| dQ/dx k, contrast ≥ 2 (n) | 0.784 (17) | 0.907 (24) |
| ring-share gap meas − pred | +0.069 / +0.051 / +0.081 | +0.082 / +0.058 / +0.072 |

χ²/N falls by 28–37 % and the pulls narrow on every plane; the collection-plane
bias and the ring-share gap do not close, exactly as one expects if the residual
is a kernel-width problem rather than a terminal-finding one. The knob is
therefore a real improvement to the PR chain **and** independent of the smearing
question — which is the cleanest possible outcome for both.

(Both of these are knob-on smoke measurements on one arm, not graded verdict
scans. §10 item 1 is still what is owed.)

---

## 10. What is not done, and what to do next

**Recommendation, in order.**

1. **Settle the wrapped-strip question first.** It is upstream of everything else
   in this doc. Concretely: (a) grade `retile_wrapped_channel_activity` on a hand
   scan of the STM verdicts it moves, the way doc pdvd/31 round 6 did; (b) take
   `wrapped_channel_charge` to the owner as a PDHD *clustering* change, since it
   moves production Q/L output. Only then re-derive §8 and propose constants.
2. **Then flip the transverse constants**, re-derived, with the doc-44 grading
   (2-D metrics, ring shares, dQ/dx, STM verdict churn) as the acceptance bar.
   The measurement machinery is committed and validated (§8.1), so that round is
   a re-run, not a rebuild.
3. **A PDHD Steiner-terminal round** (the doc pdvd/37 analogue). This round takes
   SBND's operating point (4000 e, `wire_tol` 1, `adjacent_slice` true,
   `terminal_min_separation` 0) on the argument that PDHD's 4.67 mm pitch carries
   ~1.6× SBND's charge per point. That is an argument, not a measurement.
4. **More imaged events.** 30 events give 175 accepted STM passes — plenty for
   §8 — but only **one** track passing the doc-55 stopping-muon cuts, so §6's
   comparison against data cannot be made. PDHD has ~100 more events with input
   data and no imaging.

**Explicitly out of scope and ungraded:** `tagger_check_neutrino` and its whole
tail (PID, Michel finder, the PF/kine trees); the DL/SCN vertex; the STM accept
guards, every one of which is off; `ctpc_aniso_metric`; the curved fiducial
volume. Each is reachable from `wct-pr-perevt.jsonnet` and none has a PDHD number
behind it.

**Flagged, not fixed** (§4): `params.jsonnet` still inherits generic
`lar.DL 7.2 / DT 12.0 / lifetime 8 ms`, which PDHD's *simulation* consumes.
Reconciling them with §4's 4.1207 / 8.2017 / 35 ms moves every PDHD sim job's
compiled config and is an owner decision.

---

## 11. Files

**Toolkit** (`cfg/pgrapher/experiment/pdhd/`, branch `apply-pointcloud`):
`pr.jsonnet` (new, fork of `protodunevd/pr.jsonnet`), `particle_dataset.jsonnet`
(new), `pdhd_track_fitting.json` (new), `clus.jsonnet` (`tensor_outname`, the
`wrapped_channel_charge` sampler knob, the exported PR primitives — all
default-off, gate T0).

**Working repo** (`wcp-porting-img/pdhd/`, branch `main`): this doc;
`wct-pr-perevt.jsonnet` (new), `run_pr_evt.sh` (new), `wct-clustering.jsonnet` and
`run_clus_evt.sh` (the `-save-pctree` path); `stm/pdhd_transport.py`,
`stm/pdhd_transport.tsv`, `stm/pdhd_ref_dqdx.json`; `docs/scripts/*.py`;
`docs/figs/pdhd_sigma_*`.

**Not committed:** the arms `work/029107_{0..29}_{stm0,stmw}`, the pinned binary
`/home/xqian/tmp/pdhdstm_libpin`, and the analysis products under
`/home/xqian/tmp/pdhdstm/{ana2,ctl2}`.
