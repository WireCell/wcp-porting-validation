# doc pr/21 — The cathode dead-gap dQ/dx notch: impact assessment and the MicroBooNE comparison

**Status: ANALYSIS ONLY (2026-08-02).** No code was changed, nothing is scheduled for
implementation. This answers the owner's question about doc pr/12 §7's dQ/dx notch:
what is its impact, how does it compare to the MicroBooNE dead-channel dead volume,
and should anything be done? **Recommendation up front: no new work now** — the one
demonstrated harm route is already closed by the doc pr/20 B0 cathode-kink veto (shipped
SBND DEFAULT ON at toolkit `fe6b7d90`), the decision-point guards that matter already
exist and see the gap, and the remaining exposures are watch items blocked on evidence
that does not currently exist (§6).

## 0. Repro block

The notch numbers are doc pr/12 §7 (census `12_cathode-census-pr11v3.tsv`, arms
`work-ccfeat300pr`, plots `scripts/analysis/cathode/cathode_plots.py` → `pics/pr12_dqdx_gap.png`). The code facts
below reproduce at toolkit HEAD `fe6b7d90` (doc pr/20 B0/A1/A2 already landed):

```
grep -n "dead_ind_weight\|dead_col_weight"    clus/src/TrackFitting.cxx        # dQ/dx-fit dead-plane regularization
sed -n '809,877p'                             clus/src/TrackFitting.cxx        # dead-channel 2D charge fill (blob redistribution)
sed -n '35,76p;123,227p'                      clus/src/FiducialUtils.cxx       # inside_dead_region / check_dead_volume
sed -n '164,181p;2025,2047p'                  clus/src/TaggerCheckSTM.cxx      # doc-63 cathode_guard (DEFAULT ON)
sed -n '300,360p;426,460p'                    clus/src/PRSegmentFunctions.cxx  # kink finder's two dQ/dx entries
sed -n '2694,2800p'                           clus/src/NeutrinoTaggerNuE.cxx   # gap_identification dead-chs exemption
sed -n '1,42p' prototype_base/pid/src/NeutrinoID_energy_reco.h                 # uBooNE fill_2d_charge_dead_chs
grep -n "cpa_left\|cpa_right" cfg/pgrapher/experiment/sbnd/params.jsonnet      # active volume ends at x = ±0.45 cm
```

## 1. The measurement being assessed (doc pr/12 §7, restated)

On the 44 clean spanned cathode crossers of the mcp1k census: the pooled dQ/dx profile
(each event normalised to its own 3–15 cm median) is flat at 1.0 from ~3 cm outward and
drops to **0.43 in the 0–1 cm bin**; the per-event ratio has median **0.80**, 13/44 below
0.6, worst 0.356. The notch is confined to |x| < 3 cm, centred on the inactive
|x| < 0.45 cm band.

## 2. Why the notch is "smaller than expected" — sampling mechanics, not charge recovery

A naive reading of a 0.9 cm all-plane hole is dQ/dx → 0 at its centre. That is not what
the fit can show, for two mechanical reasons:

1. **Fit points do not land inside the hole.** Trajectory points are seeded from charge;
   pr/12 §4 measures the nearest cross-cathode pair at 2.03 cm with the two points at
   |x| ≈ 0.66 cm — i.e. the sampled points sit at the gap *edges*. The 0–1 cm histogram
   bin is populated by edge points, not by points inside the inactive band.
2. **Edge points average live and empty path.** A gap-edge point's dQ is collected from
   the 2D measurements its ±nσ Gaussian window reaches (`cal_gaus_integral`,
   `TrackFitting.cxx:5145`) — the ticks past the CPA surface are live channels holding
   *genuine zeros*, and its dx spans halfway toward the far neighbour, i.e. across the
   hole. Charge from the live side divided by a dx that is roughly half dead gives a
   ratio of ~0.4–0.6 — which is exactly the measured 0.43.

So the notch size is what the geometry predicts for a 0.9 cm hole sampled from its edges.
Nothing anomalous is being recovered; the missing charge is simply smeared over the
neighbouring live points instead of appearing as a zero. The per-event ratios (median
0.80) are milder for the same reason — their window includes mostly-live points.

The physical hole itself: SBND's active volumes end at the CPA surfaces
x = ±0.45 cm (`params.jsonnet:24-25`, from GDML). Ionization deposited between them is
never drifted or read out — in **all three planes at once**. This matters for the
MicroBooNE comparison below.

## 3. Inventory — the dead-volume machinery we already have, and what sees the gap

The toolkit carries the full uBooNE dead-channel machinery, ported. Per mechanism,
whether it is *aware* of the cathode gap:

| # | mechanism | where | sees the cathode gap? |
|---|---|---|---|
| 1 | dead-channel 2D charge fill: blob charge redistributed into bad-plane cells (flag 0) before fits and energy sums | `TrackFitting::prepare_data` (`TrackFitting.cxx:809-877`); prototype `fill_2d_charge_dead_chs` (`NeutrinoID_energy_reco.h:33-35`) | **blind** — no blobs exist in the gap (no plane saw the charge), so there is nothing to redistribute |
| 2 | charge_err inflation ×5 (ind) / ×2.5 (col) for live pixels adjacent to dead channels | `TrackFitting.cxx:880-909`; prototype `PR3DCluster_trajectory_fit.h:1813-1857` | **blind** — keyed on flag-0 (dead-channel) measurements |
| 3 | dQ/dx-fit dead-plane regularization (`dead_ind_weight`/`dead_col_weight` smooth a dead-plane point toward its neighbours) | `TrackFitting.cxx:6736-6754` | **blind** — the gap's ticks are *live* channels holding real zeros (flag ≠ 0), so they act as genuine zero-charge measurements. This is *correct*: the charge really is absent |
| 4 | channel-keyed decision queries: `inside_dead_region` (`is_wire_dead` per plane), `get_closest_dead_chs` | `FiducialUtils.cxx:35-76`; nuE `gap_identification` (`NeutrinoTaggerNuE.cxx:2694-2792`); STM 2D-kink dead veto (`TaggerCheckSTM.cxx:1448,1532`); `PRSegmentFunctions.cxx:2155-2169` | **blind** — the gap is a drift-coordinate hole, not a channel hole; no dead-channel entry exists for it |
| 5 | volume-containment walkers: `check_dead_volume` / `check_signal_processing` count `contained_by() < 0` as dead | `FiducialUtils.cxx:123-227`, used by STM/TGM boundary logic | **SEES it** — the active volumes end at x = ±0.45 cm, so every step inside the gap is "not in any detector volume" and counts as dead path |
| 6 | STM `cathode_guard` + `deficit_guard` (doc-63 rounds 3/5a, **DEFAULT ON**): a fitted stop within `guard_cathode_cm` of the CPA with no Bragg rise is a truncation, not a stop | `TaggerCheckSTM.cxx:164-181, 2025-2047` | **explicitly cathode-aware** — SBND-specific work already shipped |

The other SBND fixed defect — the 6-channel W-plane dead column
(`cfg/.../sbnd/dead_regions.jsonnet`) — is channel-keyed and therefore *is* covered by
mechanisms 1–4 plus its own dead-gap registry. The cathode gap is the one hole the
channel-keyed machinery cannot express.

## 4. The MicroBooNE comparison

The premise "in MicroBooNE we did not do anything special about dead volume" needs one
correction: uBooNE **did not correct dQ/dx**, but it did quite a lot else. The WCP
response to ~10% dead channels was mechanisms 1–4 above: fill the dead-plane 2D cells
from the blob charge the live planes measured, inflate errors next door, regularize the
dQ/dx fit across dead-plane points, and consult the dead-region map at *decision points*
(STM/TGM boundary and kink logic, the nuE MIP/gap checks). What uBooNE never did is
flag or scale the *fitted dQ/dx values* themselves — consumers were instead made robust
(segment medians, peak-based Bragg windows, dead-region exemptions where a check would
otherwise misfire).

The two dead volumes differ in kind:

| | uBooNE dead channels | SBND cathode gap |
|---|---|---|
| what is lost | the *signal* in 1–2 planes; the ionization drifts and the other plane(s) see it | the *ionization* itself, in all three planes at once |
| recoverable? | yes — blob charge from live planes fills the dead cells (mechanism 1) | no — there is nothing to redistribute |
| geometry | scattered wire ranges, all drift times, ~10% of channels | one 0.9 cm slab at an exactly known, fixed location (~0.5% of the drift, ~1.5% counting the 3 cm notch skirt) |
| tracks cross into more active volume? | n/a — single drift volume, cathode at the boundary | yes — every cathode crosser; uBooNE never faced this geometry |

So the correct reading of the uBooNE precedent is: **compensate where charge is
recoverable, guard decision points where it is not, never repaint dQ/dx.** At the SBND
cathode the charge is unrecoverable, and the decision-point guards that matter either
already see the gap (mechanism 5) or were already built for it (mechanism 6).

## 5. Consumer-by-consumer impact

Every consumer of dQ/dx shape, its exposure, and the evidence:

1. **Trajectory fit** — crosses the gap as one segment in 44/45 (pr/12 §4). No action.
2. **Neutrino vertex** — not attracted to the gap (7 observed near-cathode vs 5.5
   expected from the null, pr/12 §7a). No action.
3. **PID and track/shower** — consume segment *medians*
   (`segment_median_dQ_dx`, `PRSegmentFunctions.cxx:864`): a few depressed points out of
   hundreds cannot move an `nth_element` median. Measured: 0/44 `particle_id` flips,
   0/44 `flag_shower` flips (pr/12 §7c). No action.
4. **The kink finder** (`segment_search_kink`) — the notch enters twice, with opposite
   signs:
   - the high-dQ/dx accept branch (`PRSegmentFunctions.cxx:307-356`:
     `max > 1.5×` and `ave > 1.0× mip_dqdx_median` = 48000 e/cm) is *suppressed* in the
     notch — a genuine high-dQ/dx kink sitting exactly on the CPA could be missed. That
     inefficiency is irreducible (the charge is gone) and no case has been observed.
   - the post-kink local-charge test (`:426-460`): once the *angle* criteria have found
     a kink, a ±2-point window below `25/43 × 48000 ≈ 27.9k e/cm` (0.58 MIP) flips the
     returned `flag_continue`, which changes how `proto_extend_point` walks and hence
     where the break point lands. A window straddling the gap edges (bin ratio 0.43)
     can cross that threshold *because of the gap alone* — this modulated the doc pr/20
     class-B cathode breaks (which fire on the angle criteria, `refl` 30.8/27.4 vs cuts
     30/27).
   **Verdict: real, and already closed.** The doc pr/20 **B0** cathode-kink veto
   (`PRSegmentFunctions.cxx:297-305`, shipped SBND DEFAULT ON at `fe6b7d90` with
   `cathode_kink_xcut = 5 cm` — wider than the 3 cm notch skirt) skips the accept tests
   for every fit index within the band, so neither the angle criteria nor either dQ/dx
   branch runs there. No separate dQ/dx treatment is needed; this analysis is a
   post-hoc confirmation that B0-at-the-source was the right shape (it removed the
   notch's only demonstrated consumer along with the geometric one).
5. **STM** — the Bragg/eval windows are peak- and max-based over 5–15 cm (robust to one
   low bin), and the near-cathode failure modes are *already* handled by the shipped
   doc-63 `cathode_guard`/`deficit_guard` (mechanism 6). No action.
6. **TGM / containment** — boundary walkers count the gap as dead path via
   `contained_by() < 0` (mechanism 5). No action.
7. **nuE `gap_identification` / MIP quality** — the one *blind* decision point with a
   plausible failure mode. A stem sub-point is "bad" unless a plane shows signal, dead
   channels, or prolongation (`NeutrinoTaggerNuE.cxx:2694-2792`); the dead-channel
   exemption cannot excuse the cathode gap. A genuine nue whose vertex sits within
   ~2.4 cm of the CPA with the stem crossing it would accrue unexcused bad sub-points →
   `flag_gap` → bad-reconstruction rejection. Exposure bound: |x_vtx| < 2.4 cm is
   ~1.2% of the drift, times the fraction of stems oriented across the CPA; the only
   near-cathode nueCC48 candidate (267597) has its vertex 80 cm away. **No observed
   case. Watch item** (§6, deferred option b).
8. **Energy (`cal_kine_charge` / kine_reco_Enu)** — 2D charge sums; the missing charge
   is physically absent, and mechanism 1 correctly has nothing to fill. Scale for a MIP
   crossing: per-event deficit ≈ 0.2 × 6 cm ≈ 1.2 cm-MIP ≈ 6.7e4 e (at
   `mip_dqdx = 56000 e/cm`), i.e. **~2–3 MeV** through the kine factors — calibration
   noise against typical kine_reco_Enu. The only way to lose more is a shower *core*
   straddling the CPA, which is boundary clipping of the same kind as any active-edge
   loss, not something a dQ/dx flag can restore. Doc pr/20 B2 already records the
   calibration-side aspect. No action.
9. **Direction votes** (`is_dir_weak`, proton template vote) — end-window and
   whole-segment template quantities; only a vertex-at-the-CPA geometry is exposed, no
   observed case. No action.

## 6. Should we do something? — Recommendation

**Do nothing new now.** The instinct "we did nothing special in MicroBooNE, so we may
not need anything here" reaches the right conclusion, with the corrected reasoning:
uBooNE compensated dead volume *where the charge was recoverable* and guarded *decision
points*; it never repainted dQ/dx. At the SBND cathode the charge is unrecoverable, the
relevant decision points are guarded (STM `cathode_guard` ON; the containment walkers see
the gap through the volume bounds), and the single consumer with a demonstrated pathology
— the kink finder minting cathode vertices/breaks — is already fixed by the shipped
doc pr/20 **B0** veto (SBND DEFAULT ON, `fe6b7d90`), which this analysis retroactively
strengthens.

**Explicitly rejected:** filling in or scaling up dQ/dx inside the notch. It would
fabricate charge, silently shift every downstream consumer (a maximal violation of the
byte-identical discipline), diverge from prototype behavior with no demonstrated failure
to justify it, and — per §2 — "correct" values that are already the geometrically
truthful description of a real hole.

**Deferred options, each blocked on evidence that does not currently exist:**

- *(a)* a fit-validity flag on points with |x| < 3 cm (pr/12 §10 item 4), so Bragg/MIP
  logic can skip rather than fit through them — implement only if a mis-tag is ever
  traced to a dQ/dx window straddling the CPA. The STM guards make this unlikely.
- *(b)* a cathode-band exemption in nuE `gap_identification`/MIP quality, mirroring its
  existing dead-channel exemption (config-gated, default OFF) — implement only if a nue
  MC event is found rejected for a cathode-straddling stem. Cheap first step when the
  next nueCC MC sample is scanned: count candidates with |x_vtx| < 3 cm and check their
  `flag_gap`/MIP-quality outcome. n = 0 so far.

## Related

- doc pr/12 §7 (the measurement) and §10 item 4 (the flag proposal deferred here).
- doc pr/20 Part II (B0 cathode-kink veto, shipped SBND DEFAULT ON `fe6b7d90` — closes
  §5.4; B2 records the calibration aspect).
- docs/63 rounds 3/5a (STM cathode_guard + deficit_guard, shipped DEFAULT ON).
- `cfg/pgrapher/experiment/sbnd/dead_regions.jsonnet` (the *channel-keyed* SBND dead gap,
  which the uBooNE machinery does handle).
- doc pr/2 §7.3 (provenance of the 25/43 kink charge threshold).
