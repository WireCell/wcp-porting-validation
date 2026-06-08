# SP for the APA0 anomalous collection plane (plane 2) — ProtoDUNE-HD

This document is a focused deep-dive into how the WireCell signal-processing
(SP) chain treats **APA0, plane 2** — the nominal collection (W/Z) plane that,
because of a hardware fault, behaves like an induction plane with much weaker
signals. It complements the general [sp.md](sp.md) (which documents the whole
chain) and [nf.md](nf.md). For the broader workflow see
[nf_sp_workflow.md](nf_sp_workflow.md).

**Status**: understanding / reference. No config or code is changed by this
document; the tuning ideas in the last section are *suggestions* to be applied
later as togglable, default-off knobs so production stays bit-identical.

---

## 1. The anomaly

In a DUNE-HD APA the three wire planes are **U** (induction, plane index 0),
**V** (induction, plane index 1) and **W / Z** (collection, plane index 2).
Charge drifts through U and V inducing **bipolar** signals, then is **collected**
on W, producing a strong **unipolar** signal.

On **APA0** the drift field near the wire planes was not properly established.
The practical consequence is a **V↔W role swap**:

- the nominal **collection** plane (plane 2, W) no longer fully collects — charge
  largely passes it, so it **induces** a weak, induction-like (≈bipolar) signal;
- the nominal **V** induction plane instead **collects** more like a collection
  plane.

This is the plane this document is about: **plane 2 now looks like an induction
plane and carries much weaker charge** than a healthy collection plane.

### Why two different code comments describe the "same" plane oppositely

The configuration encodes the swap with `plane2layer = [0, 2, 1]`
(`cfg/pgrapher/experiment/pdhd/sp.jsonnet:112`). This maps **plane index →
Garfield field-response layer**:

| plane index | physical wires | → layer (field response used) |
|-------------|----------------|-------------------------------|
| 0 | U | 0 (U / induction) |
| 1 | V | **2 (W / collection)** |
| 2 | W | **1 (V / induction)** |

So it is a **V↔W** swap, *not* a U↔V swap. From this single fact, two existing
comments that look contradictory are actually two views of the same event:

- `chndb-base.jsonnet:132–137` says *"APA 0 (n==0) V plane is hardware-faulty and
  behaves as a collection plane"* — i.e. **V now collects** (it zeros V's FR⊗ER
  kernel and `response_offset` so `PDHD::SignalProtection`'s deconvolution gate
  falls through on V).
- This document's framing — **W (plane 2) now induces, weak** — is the same swap
  seen from W's side, and is the plane we want to recover.

> **Note on a stale label:** `sp.md` (rows at `sp.md:34` and `sp.md:37`) describes
> `plane2layer=[0,2,1]` as *"swaps U and V"*. That is inaccurate — `[0,2,1]` swaps
> **V and W**. The mislabel is noted here for the record; `sp.md` is not edited by
> this document.

---

## 2. How the chain processes plane 2 today

APA0 (ident 0) is special-cased throughout `make_sigproc`
(`cfg/pgrapher/experiment/pdhd/sp.jsonnet`). Inventory:

| Element | APA0 | APAs 1–3 | Where |
|---------|------|----------|-------|
| Field response (PIR) | `np04hd-garfield-6paths-mcmc-bestfit.json.bz2` | `dune-garfield-1d565.json.bz2` | `sp.jsonnet:59`; `params.jsonnet:169–173` |
| `filter_responses_tn` (extra per-plane FR correction) | `[FilterResponse:plane0, FilterResponse:plane2, FilterResponse:plane1]` | `[]` | `sp.jsonnet:60–63`; consumed `OmnibusSigProc.cxx:1860–1862, 2045` |
| `plane2layer` | `[0, 2, 1]` (V↔W) | `[0, 1, 2]` | `sp.jsonnet:112` |
| `Wiener_tight_filters` (ROI kernels) | `[Wiener_tight_U_APA1, Wiener_tight_W_APA1, Wiener_tight_V_APA1]` | `[Wiener_tight_U, Wiener_tight_V, Wiener_tight_W]` | `sp.jsonnet:114–116`; `sp-filters.jsonnet:58–67` |
| `r_th_factor` (refine threshold) | `2.5` | `3.0` | `sp.jsonnet:78` |
| L1SPFilterPD planes | `[0]` (U only) | `[0, 1]` (U+V) | `sp.jsonnet:47–48` |

Two things matter for plane 2 specifically:

1. **The deconvolution kernel shape** that plane 2 is divided by is now the
   **V/induction** field response (via the `plane2layer` swap) plus the
   `filter_responses_tn[2]` = `FilterResponse:plane1` correction
   (`protodunehd-field-response-filters.json.bz2`). This is *why* the deconvolved
   plane-2 waveform "looks like induction": it is deconvolved with an induction
   kernel.

2. **Plane 2 takes the collection ROI branch** in the per-plane loop
   (`OmnibusSigProc.cxx:1890–1928`), which is structurally *shorter* than the
   induction branch:

   ```
   if (iplane != 2) {            // induction planes 0,1
       decon_2D_tighterROI();    // tighter-LF ROI decon
       decon_2D_tightROI();      // tight-LF ROI decon
       find_ROI_by_decon_itself(iplane, tight, tighter);   // two-decon compare
       ... decon_2D_looseROI(); find_ROI_loose(); decon_2D_ROI_refine();
   } else {                      // collection plane 2
       decon_2D_tightROI();      // single tight pass, NO LF filter (cxx:1482-1486)
       find_ROI_by_decon_itself(iplane, tight);            // single-decon
       // loose-ROI block skipped entirely (cxx:1914 `if (iplane != 2)`)
   }
   ```

   Note `decon_2D_looseROI` itself early-returns for plane 2
   (`OmnibusSigProc.cxx:1550–1552`, *"don't filter colleciton"*). So on plane 2
   the **only** ROI-admission pass is the single tight-ROI decon, with **no LF
   filter** applied and **no loose-ROI extension or ROI-refine decon**. After
   that the common `roi_refine` clean-up/merge stage runs for all planes.

A wire-domain smoothing filter is also applied during `decon_2D_init`: plane 2
uses `Wire_col` (σ = 10/√π ≈ 5.64, `sp-filters.jsonnet:84`), much wider than the
induction `Wire_ind` (σ ≈ 0.42). This widens charge across collection wires; for
a now-inducting plane 2 it is worth revisiting (see §5).

---

## 3. The two kernels: ROI-finding vs charge deconvolution

The chain deliberately uses **separate kernels** for *finding where signal is*
(ROI) and for *measuring the charge* inside those ROIs. For plane 2:

```
        raw ADC (plane 2, weak, induction-like)
                  │
                  ▼   decon_2D_init  (cxx:1141)
   divide by (V-layer field response ⊗ elec) , × Wire_col wire filter
                  │   ──► m_c_data[2]  (deconvolved frequency spectrum)
        ┌─────────┴───────────────────────────────┐
        ▼                                          ▼
  ROI-FINDING KERNEL                        CHARGE KERNEL
  decon_2D_tightROI(2)  (cxx:1482)          decon_2D_charge(2) (cxx:1755)
  × Wiener_tight_filters[2]                 × Gaus_wide  (σ=0.12 MHz)
    = Wiener_tight_V_APA1                      (same for all planes)
    (σ=0.160 MHz, power=3.55)                       │
  (NO LF filter on collection)                      ▼
        │                                    charge inside ROIs
        ▼                                    → gauss2 output (imaging input)
  find_ROI_by_decon_itself(2)               (Wiener_wide path → wiener2)
  threshold = troi_col_th_factor (5.0)
        │
        ▼
  admitted ROIs (where signal is allowed)
```

- **ROI-finding kernel** = `Wiener_tight_filters[2]` = `Wiener_tight_V_APA1`
  (`sp-filters.jsonnet:61–63`). Wiener filters are tuned to the *signal+noise*
  spectrum and are deliberately aggressive; their job is purely to decide **which
  (wire, tick) regions are signal**. The threshold applied to this decon is
  `troi_col_th_factor = 5.0` (`sp.jsonnet:71`), in units of local noise RMS.

- **Charge-deconvolution kernel** = `Gaus_wide` (σ = 0.12 MHz, `sp-filters.jsonnet:46`),
  applied in `decon_2D_charge` (`OmnibusSigProc.cxx:1755–1786`), identical for all
  three planes. It extracts the actual charge **inside** the admitted ROIs and is
  written to the `gauss2` tag that imaging consumes. A parallel `Wiener_wide`
  decon (`decon_2D_hits`) fills the `wiener2` tag.

The separation is the key lever for the user's goal: **efficiency (how much weak
signal is admitted) is controlled by the ROI kernel + its threshold; the charge
measurement kernel is left untouched.**

---

## 4. Live vs dead knobs for plane 2

Because plane 2 takes the short collection branch, several knobs that the general
`sp.md` tuning table lists are **bypassed** for plane 2. Tuning a dead knob has
no effect.

| Knob | Plane 2? | Why |
|------|----------|-----|
| `troi_col_th_factor` (`sp.jsonnet:71`) | **LIVE** | Threshold on the tight-ROI decon — the primary admission gate. |
| `Wiener_tight_filters[2]` (`sp.jsonnet:114–116`) | **LIVE** | The ROI-finding kernel for plane 2 (`Wiener_tight_V_APA1`). |
| `field_response` + `filter_responses_tn[2]` (`sp.jsonnet:59–63`) | **LIVE** | Shape the deconvolution kernel itself. |
| `Gaus_wide` / `Wiener_wide` (`sp-filters.jsonnet:46,78`) | **LIVE** | Charge / wiener extraction kernels. |
| `Wire_col` (`sp-filters.jsonnet:84`) | **LIVE** | Wire-domain smoothing applied in `decon_2D_init`. |
| `r_th_factor` (`sp.jsonnet:78`) | **PARTIAL** | Used by `roi_refine` clean-up/merge, which *does* run for plane 2. |
| `ROI_tight_lf`, `ROI_tighter_lf` (`sp-filters.jsonnet:42–43`) | **DEAD** | Collection tight-ROI branch applies no LF filter (`cxx:1482–1486`). |
| `lroi_rebin`, `lroi_th_factor`, `lroi_th_factor1`, `lroi_jump_one_bin` (`sp.jsonnet:73–76`) | **DEAD** | Loose-ROI block is skipped for plane 2 (`cxx:1914`); `decon_2D_looseROI` early-returns (`cxx:1550`). |
| `r_fake_signal_low_th` / `_high_th` (`sp.jsonnet:79–82`) | **VERIFY** | Reach plane 2 only via the refine/break machinery; confirm against `roi_refine` before relying on them. |

---

## 5. Suggestions: higher efficiency / lower purity on plane 2

Goal: **admit more of plane 2's weak signal now and let downstream
imaging/clustering reject the extra fakes.** All ideas below should land as
new jsonnet knobs gated on `anode.data.ident == 0` (and defaulting to today's
values) so non-APA0 anodes and existing production configs stay bit-identical —
the same pattern already used for `r_th_factor`, `plane2layer`, and the `_APA1`
Wiener set.

### Tier 1 — config-only, plane-2-targeted, no C++ change

In the `make_sigproc` instance for APA0, **plane 2 is the only collection-classed
plane**, so the collection knobs hit plane 2 alone.

1. **Lower `troi_col_th_factor`** (currently `5.0`, `sp.jsonnet:71`) for APA0.
   This is the *primary* admission lever: it directly lowers the noise-RMS
   multiple a tight ROI must clear. Going from 5.0 toward, say, 3.0–3.5 admits
   weaker plane-2 signal at the cost of more fake ROIs. Add it as an
   `ident==0`-branched knob (e.g. `troi_col_th_factor: if ident==0 then <new> else 5.0`).

2. **Make the ROI-finding kernel `Wiener_tight_filters[2]` more sensitive.**
   Today plane 2 uses `Wiener_tight_V_APA1` (σ=0.160 MHz, power=3.55,
   `sp-filters.jsonnet:61–63`). A weaker/broader plane-2 pulse benefits from a
   filter that passes more low-amplitude, broader structure: try a new
   `Wiener_tight_W_APA0weak` instance with **lower power** (slower roll-off) and a
   modestly **wider σ**, and point `Wiener_tight_filters[2]` at it for APA0. Keep
   it a separate named filter (the SP C++ hard-codes filter instance *names*, so
   add a new name rather than mutating an existing one — see `sp-filters.jsonnet:1–4`).

3. **Lower `r_th_factor` for APA0** further (currently `2.5`, `sp.jsonnet:78`).
   This is the refine-stage threshold and `roi_refine` runs for plane 2, so a
   smaller value keeps more marginal ROIs through clean-up/merge.

Recommended order to try: (1) first — biggest, cleanest effect; then (2) to
recover sub-threshold broad signal; (3) last, as it also loosens U/V refine.

### Tier 2 — deeper, needs C++ (flag as exploratory)

4. **Route plane 2 through the induction 3-step ROI path.** Now that plane 2
   *inducts*, the richer induction admission (tighter+tight compare + loose-ROI
   extension + ROI-refine decon) may recover more than the single collection
   pass. This is currently impossible from config: the branch keys on the **raw**
   `iplane == 2` (`OmnibusSigProc.cxx:1891, 1914`; `decon_2D_looseROI` guard at
   `:1550`), independent of `plane2layer`. It would need a config-selectable
   "treat-as-induction" flag for that plane.

5. **Extend L1SPFilterPD to plane 2 (research, not a drop-in).** L1SP corrects
   unipolar-on-induction signals using pre-built bipolar+unipolar bases
   (`sp.md:302–347`), and is currently U-only on APA0. Plane 2's anomaly is the
   *reverse* polarity (a collection plane gone induction-like), so the existing
   kernel bases do not directly apply; new bases tuned to plane 2's response
   would be required. Treat as a study, not a quick win.

### What to leave alone

- The **charge kernel** (`Gaus_wide`) — efficiency should come from ROI
  admission, not from changing the charge measurement.
- The **dead knobs** in §4 (`ROI_*_lf`, `lroi_*`) — no effect on plane 2.
- **Non-APA0 anodes** — every change must be `ident==0`-gated.

---

## 6. Source reference index

| File | Lines | Role |
|------|-------|------|
| `cfg/pgrapher/experiment/pdhd/sp.jsonnet` | 47–48, 59–63, 71, 78, 112, 114–116 | APA0 special-casing: L1SP planes, field/filter responses, troi/r thresholds, plane2layer, Wiener set |
| `cfg/pgrapher/experiment/pdhd/sp-filters.jsonnet` | 42–46, 58–67, 78, 84 | LF/Gauss filters; `_APA1` Wiener-tight set; wide Wiener; `Wire_col` |
| `cfg/pgrapher/experiment/pdhd/chndb-base.jsonnet` | 132–137 | APA0 V-plane FR⊗ER zeroing (the other side of the V↔W swap) |
| `cfg/pgrapher/experiment/pdhd/params.jsonnet` | 169–176 | Field-response file list; `fltresp` filter-response file |
| `sigproc/src/OmnibusSigProc.cxx` | 1457–1500 | `decon_2D_tightROI` (ROI kernel; plane-2 = no LF, `else` at 1482–1486) |
| `sigproc/src/OmnibusSigProc.cxx` | 1548–1552 | `decon_2D_looseROI` early-return for plane 2 |
| `sigproc/src/OmnibusSigProc.cxx` | 1755–1786 | `decon_2D_charge` (Gaus_wide charge kernel, all planes) |
| `sigproc/src/OmnibusSigProc.cxx` | 1860–1862, 2045 | `filter_responses_tn` consumption |
| `sigproc/src/OmnibusSigProc.cxx` | 1890–1928 | Per-plane ROI branch (collection vs induction path) |
| `pdhd/docs/sp.md` | whole | General SP chain (note the `plane2layer` "U/V" mislabel at 34/37) |
