# QLMatching low-PE detection-inefficiency error model (run 29107)

## The problem

Hand scans of PDHD run 29107 (evts 983/991/999/1007, 124 human-confirmed
flash↔cluster matches) show that **when a photon detector's predicted PE is modest it
frequently measures nothing at all** — real low-light detection inefficiency, far
stronger than Poisson:

| predicted PE | fraction of channels measuring 0 |
|---|---|
| 1–2  | 60% |
| 2–5  | 49% |
| 5–10 | 31% |
| 10–30 | 13% |
| 30–100 | 6% |
| 100+ | 3% |

(Poisson would predict P(0\|pred=3)≈5%, not ~49%.) These channels are counted in the
bundle chi2 (`sigma² = meas + perr²`, `chi2_j = (pred−meas)²/sigma²`). Under PDHD's
prior *measured*-based error a "predicted few PE, measured 0" channel gets
`perr = floor = 0.3`, so `chi2_j ≈ pred²/0.09 ≈ 100`; many such channels per bundle
pile up and wreck the chi2 of otherwise-correct matches.

## The fix

Grow the per-opdet *relative* error as the predicted PE falls, computed from the
**predicted** pe (so a measured-zero channel can still be told it was expected to be
dim). In `match/src/TimingTPCBundle.cxx` (`per_opdet_perr`), active when
`pe_err_on_pred` and `pe_err_lowpe_frac ≥ 0`:

```
rel(pred) = pe_err_frac + (pe_err_lowpe_frac − pe_err_frac) · exp(−pred / pe_err_lowpe_knee)
perr      = sqrt( (rel(pred)·pred)² + pe_err_floor² )
```

- High pred: `rel → pe_err_frac` (0.44) — unchanged tight error.
- Low pred: `rel → pe_err_lowpe_frac` (~1.55) — "predicted a few PE, measured zero"
  costs ~1 in chi2 instead of ~100.

`pe_err_lowpe_frac < 0` ⇒ disabled (the floor/frac/knee branch, bit-identical). New C++
knobs default off; PDHD `cfg/.../pdhd/qlmatching.jsonnet` enables it.

## Calibration

`pdhd/ql_light_calib/fit_lowpe.py` reads the 4 label files, applies the chi2 ndf gate
(drop channels with meas<1 and pred<1), and picks `lowpe_frac`/`lowpe_knee` so a typical
measured-zero channel contributes ~1 to chi2. With `pe_err_frac = 0.44`:

```
pe_err_on_pred:    true
pe_err_lowpe_frac: 1.55
pe_err_lowpe_knee: 5.5
```

meas==0 subset median chi2 **5.2 → 1.0**; `pred ≥ 50` channels unchanged
(sum chi2 16026 → 16025).

## Validation (no re-run needed)

The change touches only the chi2 denominator, so the existing calib bundles can be
re-scored directly (`/home/xqian/tmp/validate_lowpe.py`):

- **Closure** against the stored measured-based chi2: max |Δ| = 2e-13 (FP roundoff) —
  the masking / `chi2_relax` replication is faithful.
- Median chi2/ndf of the human-confirmed matches (old → new):
  983 **51.9 → 2.5**, 991 **30.2 → 4.9**, 999 **42.8 → 2.8**, 1007 **30.8 → 5.5**.
- Times the labeled cluster is the chi2-best candidate for its flash:
  983 0→8, 991 2→6, 999 1→8, 1007 3→5.

The correct matches become chi2-consistent. End-to-end A/B via
`run_clus_evt.sh -calib 29107 <evt>` (regenerate calib, re-check `auto_selected` vs the
labels) is the remaining gold-standard check.

## Scope

This addresses **only** the low-pred / measured-zero case (the inefficiency the scans
reported). The separate measured≫predicted heavy tail (the dominant chi2 contributor
overall) is *not* addressed here; it would need a robust/Huber per-PD cap and is left
for a later decision.
