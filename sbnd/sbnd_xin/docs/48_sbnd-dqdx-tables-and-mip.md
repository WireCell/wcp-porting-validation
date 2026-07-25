# 48 — SBND dQ/dx reference tables at 0.5 kV/cm, and a configurable flat-MIP scale

Follow-up to [47_stm-bragg-reference-sbnd-retune.md](47_stm-bragg-reference-sbnd-retune.md),
which inventoried the problem. This doc *acts* on §8's first two proposals: it replaces the
five `*DeDx` reference curves with SBND versions and makes the hard-coded flat MIP dQ/dx a
config knob.

**Status: NOT bit-identical for SBND.** uBooNE untouched. Every number below is measured, and
§4 reports a real effect on STM verdicts that needs a hand scan before this is called good.

---

## Repro

```bash
# 1. regenerate the curves (energy_loss repo, remote lastgeorge/energy_loss)
cd /home/xqian/work/scratch_wcgpu1/toolkit-dev/energy_loss/pion_travel
root -l -b -q 'convert_field.C(0.5, "stopping_ave_dQ_dx_sbnd.root", true)'
# regression: this must reproduce the committed convert.C output bit-for-bit
root -l -b -q 'convert_field.C(0.273, "/tmp/repro.root", false)'
cd .. && python3 docs/emit_jsonnet_dedx.py pion_travel/stopping_ave_dQ_dx_sbnd.root
python3 docs/plot_sbnd_dedx_tables.py          # docs/sbnd_dedx_tables.png

# 2. the 3-way scan A/B (30 events, ~6 s/event)
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
for tag in dq48base dq48tab dq48; do
  for suff in mcp10 mcp1000 mcp1000b; do
    mkdir -p work-$suff-$tag
    for d in work-$suff-fvxy/ql_evt* work-$suff-fvxy/evt*; do
      ln -sfn $PWD/$d work-$suff-$tag/$(basename $d); done
  done
done
F="-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5 -fvy 3"
# dq48base: git-stash particle_dataset.jsonnet to the uBooNE tables, then -mip 50000
# dq48tab : new tables, -mip 50000     (isolates the table change)
# dq48    : new tables, -mip 56000     (the full change)
S=$PWD/input_files_reco1/staged-mcp2025c-1000evt
SBND_MAX_JOBS=5 SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10-dq48 ./run_nusel_evt.sh data all $F -mip 56000
for e in $(seq 10 19); do SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000-dq48  ./run_nusel_evt.sh data 1 $F -mip 56000; done
for e in $(seq 20 29); do SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000b-dq48 ./run_nusel_evt.sh data 1 $F -mip 56000; done

# 3. the per-window ks1/ratio1 evidence in §3 (evt 290135, e28 shard)
SBND_MAX_JOBS=1 SBND_INPUT_DIR=$S/e28 SBND_WORK_ROOT=$PWD/work-mcp1000b-evnew \
  ./run_nusel_evt.sh data all $F -mip 56000 -stm-fit
# then read T_stm_eval from work-mcp1000b-evnew/nusel_evt290135/tracking-stm.root
```

Scan tags: `work-{mcp10,mcp1000,mcp1000b}-{dq48base,dq48tab,dq48}`; per-window dumps
`work-mcp1000b-{evbase,evnew}` (evt 290135 only). All fresh tags — nothing pre-existing was
written into (M13).

---

## 0. What changed, in one table

| # | thing | was | now | where |
|---|---|---|---|---|
| 1 | five `*DeDx` dQ/dx curves | Modified Box at **0.273 kV/cm** ×0.85 | same formula at **0.5 kV/cm** ×0.85 | `sbnd/particle_dataset.jsonnet` |
| 2 | flat MIP dQ/dx | hard-coded `50e3` in 35 places | knob `mip_dqdx`, C++ default `50e3`, SBND config **56000** | `clus/src/TaggerCheckSTM.cxx`, `cfg/pgrapher/common/clus.jsonnet`, `cfg/.../sbnd/clus.jsonnet` |
| 3 | electron curve stopping end | synthetic 260000 @ rr −0.5 cm, hand-set 100000 @ 0.5 cm, then a *drop* to 47000 | **flat** into rr → 0 at the first physical value | `sbnd/particle_dataset.jsonnet` |
| 4 | viewer reference dump | uBooNE `MuonDeDx`/`ProtonDeDx` | regenerated | `nusel_display/stm_ref_dqdx.json` |

What did **not** change, deliberately: `alpha = 0.93`, `beta = 0.212`, `rho = 1.38`,
`W_ion = 23.6 eV` (ArgoNeuT / MicroBooNE eq. 3.1), and the **undocumented ×0.85** — per owner
instruction, only `E` moved. The five `*Range` tables are genuinely detector-agnostic and are
byte-identical. `search_range` / `time_tick_cut` untouched (doc 47).

---

## 1. The generator

`convert.C` in the `energy_loss` repo is the original producer of
`stopping_ave_dQ_dx.root`, and it hard-codes the field inside the Modified-Box expression:

```cpp
y = log(0.93 + 0.212/1.38/0.273*y)/(23.6e-6*0.212/1.38/0.273)*fudge_factor;  // fudge = 0.85
```

Rather than edit it (it is the provenance of the shipped MicroBooNE tables), it was duplicated
as `pion_travel/convert_field.C` with `E` as an argument:

```
dQ/dx = ln(alpha + beta' * dE/dx) / (beta' * W_ion) * 0.85      beta' = beta/(rho*E)
```

**Regression proof.** `convert_field(0.273, ..., electron_flat_end=false)` reproduces the
committed `stopping_ave_dQ_dx.root` to double round-off — max relative deviation
**5.4e-16** across all five graphs, and identical point counts and x-coordinates:

| graph | N | max &#124;dx&#124; | max rel dy |
|---|---:|---:|---:|
| muon | 60 | 0 | 5.4e-16 |
| pion | 60 | 0 | 4.6e-16 |
| kaon | 60 | 0 | 4.5e-16 |
| proton | 60 | 0 | 4.2e-16 |
| electron | 61 | 0 | 2.7e-16 |

So the SBND curves differ from the shipped ones by exactly one changed constant, plus the
electron end treatment of §5. MIP sanity anchor at dE/dx = 2.1 MeV/cm: **47843.7** e/cm at
0.273 (matches the value `energy_loss/docs/energy_loss_overview.md` §4.1 quotes) →
**53266.2** e/cm at 0.5.

---

## 2. The new curves

![SBND vs MicroBooNE dQ/dx tables](../../../../energy_loss/docs/sbnd_dedx_tables.png)

*(figure lives at `energy_loss/docs/sbnd_dedx_tables.png`)*

Muon, per residual range:

| rr (cm) | uBooNE (e/cm) | SBND (e/cm) | ratio |
|---:|---:|---:|---:|
| 0.5 | 123417.0 | 168151.0 | 1.362 |
| 1.5 | 92909.8 | 118770.9 | 1.278 |
| 2.5 | 82815.2 | 103292.2 | 1.247 |
| 5.5 | 69711.4 | 83853.0 | 1.203 |
| 10.5 | 61186.1 | 71636.5 | 1.171 |
| 20.5 | 54513.4 | 62330.4 | 1.143 |
| 40.5 | 50379.1 | 56683.1 | 1.125 |
| 59.5 | 48879.4 | 54657.7 | 1.118 |

Peak and plateau for all five:

| particle | peak rr=0.5 uB → SBND | ratio | plateau rr=59.5 uB → SBND | ratio |
|---|---|---:|---|---:|
| muon | 123417 → 168151 | 1.362 | 48879.4 → 54657.7 | 1.118 |
| pion | 130057 → 179121 | 1.377 | 50018.1 → 56194.5 | 1.123 |
| kaon | 161542 → 232354 | 1.438 | 59000.8 → 68563.2 | 1.162 |
| proton | 178414 → 261588 | 1.466 | 66964.7 → 79877.4 | 1.193 |
| electron | (hand-set) → 52137.8 | — | 59051.0 → 68633.5 | 1.162 |

**The correction is residual-range dependent, so this is not a rescale.** The muon Bragg
*contrast* — peak(0.5 cm) / plateau(59.5 cm) — moves **2.525 → 3.076**, and the proton's
**2.664 → 3.275**. Because `kslike_compare` is area-normalized (`util/src/KSTest.cxx:216-238`),
a pure scale factor would leave `ks1` untouched and move only `ratio1`. A contrast change moves
`ks1` too. That is why doc 47 §4 concluded the curve had to be *replaced*, and §3 below measures
both effects separately.

---

## 3. Measured effect on the STM evaluator (evt 290135)

This is the direct instrument: `save_stm_fit` persists a `stm_eval` cluster PC per
(window, offset, com_range) trial, which `SbndMagnifyTrackingVisitor` writes as `T_stm_eval`.
Both runs are the same event, same fit, same pctree — only the tables and `mip_dqdx` differ.

Cluster 7 (the bundle that **lost** its STM tag), all trial windows:

| peak | off | com | ks1 uB → SBND | ratio1 uB → SBND | ratio2 uB → SBND | ks2 |
|---:|---:|---:|---|---|---|---:|
| 40 | 0 | 35 | 0.0995 → 0.1173 | 1.0584 → **1.2479** (×1.179) | 0.8632 → 0.9668 (×1.120) | 0.0248 both |
| 40 | 3 | 35 | 0.0700 → 0.0799 | 0.9861 → **1.1404** (×1.156) | 0.8632 → 0.9668 (×1.120) | 0.0248 both |
| 40 | 0 | 15 | 0.0643 → 0.0802 | 1.2881 → 1.5663 (×1.216) | 0.9069 → 1.0157 (×1.120) | 0.0532 both |
| 40 | 3 | 15 | 0.0427 → 0.0420 | 1.1388 → **1.3422** (×1.179) | 0.9069 → 1.0157 (×1.120) | 0.0532 both |

Four readings, all confirmed rather than assumed:

1. **`ratio1` scales by the table ratio** (×1.16–1.22 on this track), exactly as
   `ratio1 = Σref_muon / Σdata` requires. Both `flag_strong_check` accept branches
   (`TaggerCheckSTM.cxx:1598,1600` pre-change) gate on `fabs(ratio1-1) < 0.1`, so a track whose
   reconstructed charge matched the *old* reference now fails that gate. In the base run the
   `(40, 3, 15)` window returned `verdict=1`; with the new tables **every** window returns
   `verdict=0`, and the window loop then goes on to try the `peak=20` set (4 extra rows).
2. **`ratio2` scales by exactly the `mip_dqdx` ratio, ×1.120 = 56000/50000**, to four digits, on
   every window. That is the knob doing precisely and only what it should.
3. **`ks2` is bit-identical** (0.0248 / 0.0532 in both runs). `kslike_compare` is
   area-normalized and `ref_flat` is constant, so the flat-reference KS statistic is invariant
   under `mip_dqdx`. **`mip_dqdx` cannot affect `ks2` at all** — it reaches the verdict only
   through `ratio2` and through the ~33 threshold divisors of §6.
4. **`ks1` moves only via the Bragg contrast**, and only slightly (≤ +0.016), in both
   directions (0.0427 → 0.0420 goes *down*). Consistent with §2's contrast argument.

And the change is **not** uniformly adverse — cluster 5 in the same event moved *toward*
acceptance: `ratio1` 0.8540 → **1.0059**. Whether the new reference helps or hurts a given track
depends on where that track's own reconstructed charge sits, which is the confound doc 47 §5
flagged and this change does not resolve.

---

## 4. Scan A/B over 30 events — 4 of 11 STM tags lost

Same 30-event MCP2025C set as docs 36–39, same flags, three configurations. Baseline is a
**fresh** run (not doc 39's `-fvxy` tag) because the DL/DT diffusion change of doc 47 §6a also
moved the fit; comparing to `-fvxy` would conflate three things.

| config | tables | `mip_dqdx` | bundles | STM | TGM | nu-candidate | not-tagged | LM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `dq48base` | uBooNE | 50000 | 329 | **11** | 119 | 21 | 176 | 1 |
| `dq48tab` | SBND | 50000 | 329 | **7** | 119 | 21 | 180 | 1 |
| `dq48` | SBND | 56000 | 329 | **7** | 119 | 21 | 180 | 1 |

**The table change costs 4 of 11 STM tags (36%). The `mip_dqdx` change costs none.**
TGM, nu-candidate and LM counts are untouched by either.

The four bundles that flipped STM → not-tagged:

| event | main | len_main (cm) |
|---|---:|---:|
| 284657 | 14 | 190.8 |
| 288947 | 3 | 98.2 |
| 289849 | 14 | 397.5 |
| 290135 | 7 | 90.9 |

`mip_dqdx` 50000 → 56000 changes **no label on any of the 329 bundles**. It does flip the
`stm` bookkeeping column between 0 and −1 on 7 bundles (i.e. whether the evaluator reached a
decision at all, via the §6 threshold divisors) with the label unchanged in every case.

**This is doc 47 §4a's falsifiable prediction coming true, and it is not a bug.** The reference
is now on the true e/cm scale at SBND's field, while the data is still MCP2025C **uncalibrated**
— no gain and no electron-lifetime correction (doc 42 §0; and the lifetime correction is not
even ported, doc 47 / `7_notes.md`). Doc 42 measured the SBND plateau at 52.9 ke/cm, ~8 % above
the old uBooNE table, not the ~12 % the new table now assumes. The strict
`fabs(ratio1-1) < 0.1` gate has no tolerance for that residual.

**Not tuned to make this go away** (escalation rule 7). What it means:

- The 4 lost tags are **not** demonstrated false negatives. Nor are they demonstrated
  corrections of former false positives. Deciding that needs a hand scan of the four events —
  that is the open item, not a parameter to twist.
- The honest summary is that the tables are now *physically* right and *calibration-wise*
  premature, exactly as the user framed it ("current best educated guess, once we have more
  results from data we can do a more reliable calibration").
- If the 4 turn out to be real stopping muons, the cheap correct fix is a charge calibration
  upstream of the fit, **not** loosening the `ratio1` gate and **not** reverting the field.

---

## 5. The electron curve

The shipped `ElectronDeDx` had a rise into the stopping end that was entirely artificial —
both leading points were hand-set, one of them commented `// hack ..` in `convert.C`:

| rr (cm) | uBooNE as shipped | provenance | SBND now |
|---:|---:|---|---:|
| −0.5 | 260000 | **invented** during transcription to fit `LinterpFunction`'s regular grid | *(point removed)* |
| 0.5 | 100000 | `convert.C`: `if (i==0) y = 100e3` | **52137.8** (= the 1.5 cm value) |
| 1.5 | 46999.4 | first genuinely evaluated point | 55095.7 |
| 15.5+ | 59051 (44 of 60 entries) | the `length > 15 → Eval(15)` clamp | 68633.5 |

Per owner instruction the rise is gone: the curve is held **flat** into rr → 0 at its first
physical value. Toolkit `LinterpFunction` clamps below the first node
(`util/inc/WireCellUtil/Interpolate.h`: `if (x <= m_le) return m_dat.front()`), so a first bin
equal to the second extends flat to rr = 0 with no synthetic negative-rr anchor. The grid
therefore moves `start: -0.5` → `start: 0.5`, matching the other four tables.

Note the shipped curve's shape was not merely "rising too steeply" — it *dipped*, 260000 →
100000 → 46999, then climbed back to 59051. The flat version removes a discontinuity, not just
a slope.

**Unchanged and still wrong:** everything above ~15 cm is the `convert.C` clamp, because
`ele_travel/ele1.dat` stops at 15.3 cm / 60 MeV. **44 of 60 entries remain one constant.** Any
`electron_ref` comparison beyond ~15 cm residual range is against a flat line, not electron
physics. Fixing that needs `ele1.dat` extended (ESTAR data in `argon.dat` runs to 1 GeV, so the
60 MeV cap is a loop bound, not a data limit) — out of scope here, listed in §8.

---

## 6. The `mip_dqdx` knob — one number, two roles

`TaggerCheckSTM.cxx` held `50e3` / `50000` in **35 places**. All 35 are the same physical
constant in **electrons per cm** (every site either divides a dQ/dx already multiplied by
`units::cm`, or builds a reference vector compared against such values). Note this differs from
`PRSegmentFunctions.h`, whose `MIP_dQdx` default arguments are written `50000/units::cm`.

All 35 now read the member `m_mip_dqdx`, but they are **not one role**:

| role | sites | effect of raising the number |
|---|---:|---|
| flat "MIP-like, not stopping" **reference curve** — `detect_proton`'s `const_ref`, `eval_stm_core`'s `ref_flat` | 2 | rescales the reference: moves `ratio2`, leaves `ks2` invariant (§3 reading 3) |
| **divisor** normalizing measured dQ/dx into MIP units for hard-coded thresholds (`ave_res_dQ_dx/mip > 0.9`, `> 1.2`, `> 4.5`, `dQ_dx[max_bin]/mip > 2.3`, `< 1.5` …) | 33 | **tightens every one of those cuts by 12 % on unchanged measured charge** |

The second row is worth stating plainly: those thresholds were tuned against 50000 on uBooNE
data, and raising the divisor is only coherent in the limit where SBND's reconstructed dQ/dx is
also 12 % higher — i.e. after the calibration §4 says has not happened. Measured consequence on
the 30-event set: **zero label changes**, 7 bookkeeping flips. So it is currently harmless, but
it is a knob with two jobs.

**One knob drives both roles by design, not oversight.** If the reference should be rescaled
*without* retuning the cuts, that needs a second knob — deliberately not built uninvited.

### Choosing 56000

`50000` was never the dE/dx = 2.1 MeV/cm value of the uBooNE table (47844); it was a round
number just above that table's **plateau**. `ref_flat` is the *not-stopping* hypothesis, so it
belongs where a through-going muon sits — the muon table's plateau — not at the ionization
minimum. Preserving that relationship:

| | uBooNE | SBND |
|---|---:|---:|
| muon table plateau (rr = 59.5 cm) | 48879.4 | 54657.7 |
| flat MIP constant | 50000 | **56000** |
| ratio | 1.02293 | 1.02456 |

Headroom preserved to **0.16 %**, and 56000 is as round as 50000 was. Alternatives considered
and rejected: 55911 (exact plateau ratio, 0.2 % away, not round), 53266 (the dE/dx = 2.1 value —
5 % lower, and anchored to the wrong quantity for a not-stopping reference).

### Byte-identical when unset

C++ default is `50e3`, and the jsonnet builder omits the key entirely when the parameter is
null:

```
toff -> KEY ABSENT
ton  -> present=56000
```

So `TaggerCheckSTM`'s compiled config is unchanged for any caller that does not pass it. SBND is
the only caller in the tree (uBooNE's is commented out at `qlport/uboone-mabc.jsonnet:1252`).

---

## 7. Verification

| gate | result |
|---|---|
| generator regression vs `convert.C` | **PASS**, max rel dev 5.4e-16 (§1) |
| `wcbuild` + M1 freshness proof | `local/lib/libWireCellClus.so` 09:19:08 > sources 09:16 |
| `./build/clus/wcdoctest-clus` | **PASS** 41 cases / 518 assertions |
| `./build/aux/wcdoctest-aux` | **PASS** 17 cases / 88473 assertions |
| compiled-config proof, knob on | `TaggerCheckSTM.data.mip_dqdx = 56000`; `MuonDeDx[0] = 168151`, `[-1] = 54657.7`; `ElectronDeDx.start = 0.5`, `values[0:2] = [52137.8, 52137.8]` |
| key-suppression proof, knob unset | key absent from `data` |
| uBooNE tables untouched | `qlport/uboone-mabc.jsonnet` clean in `git status`; still `MuonDeDx[0] = 123417` |
| `*Range` tables untouched | `particle_dataset.jsonnet` identical from line 96 to EOF; only 5 value arrays + electron `start` changed |
| only SBND config touched | `cfg/pgrapher/experiment/sbnd/clus.jsonnet` + the shared builder in `cfg/pgrapher/common/clus.jsonnet` (key-suppressed) |
| 30-event scan A/B | §4 — 4 STM tags lost to the tables, 0 labels moved by `mip_dqdx` |

**Not bit-identical for SBND**: the docs 41–46 STM/PR numbers, and doc 47 §6b's smoke check,
predate both this and the diffusion change.

---

## 8. Open items — none of these were done here

1. **Hand-scan the 4 lost STM events** (284657 m14, 288947 m3, 289849 m14, 290135 m7). This is
   the gate on whether §0 item 1 was a net gain. Until then the tables are "physically right,
   calibration-wise premature".
2. **The ×0.85 is still undocumented and still in the curve.** Retained per owner instruction.
   It is a flat −15 % and is degenerate with the missing lifetime correction
   (`energy_loss/docs/dqdx_consistency_check.md` §6.2 reading 3) — one uncalibrated event cannot
   separate them. It should die with a real charge calibration, not by fitting.
3. **Charge calibration upstream of the fit** (gain + electron lifetime). The lifetime
   correction is not ported at all — `clus/src/PRSegmentFunctions.cxx:1196-1198` says callers
   must pre-apply it, and nobody does. This is the actual fix for §4.
4. **Two MIP constants now live in the same SBND event.** `TaggerCheckSTM` runs at 56000;
   `PRSegmentFunctions.h`'s defaults are still the uBooNE values and are reached from
   `NeutrinoTaggerSSM.cxx:64,65,96` — `50000/units::cm` at `:91` (`do_track_comp`), `:93`
   (`segment_do_track_pid`), `:111` (`segment_cal_4mom`), `:116`
   (`segment_is_shower_trajectory`), and `43000/units::cm` at `:27` (`segment_search_kink`),
   `:104`, `:117`, `:119`, `:120` (direction / shower routines). Flagged, not fixed — widening
   this change to cover them was out of scope.
5. **The table change reaches past the STM tagger.** `do_track_comp` builds
   `muon_ref`/`proton_ref`/`electron_ref` from these same tables and is called from
   `NeutrinoTaggerSSM`, so SSM/PID sees the new curves too. Not separately validated here.
6. **`BoxRecombination` in `cfg/.../sbnd/clus.jsonnet:465` still uses A/B = 1.0/0.255**, which
   matches no published MicroBooNE set (`energy_loss/docs/dqdx_consistency_check.md` §6.3). The
   tables use the official 0.93/0.212. Two Modified-Box parameter sets remain live in one
   pipeline; `cal_kine_dQdx` inverts one while `do_track_comp` compares against the other.
7. **Electron table above ~15 cm is still 44 identical entries** (§5).
8. **Get SBND's own `ModBoxA`/`ModBoxB`** if SBND has measured them; only `E` was updated here.

---

Companion docs: [47](47_stm-bragg-reference-sbnd-retune.md) (the inventory and the field-anchor
proof), `energy_loss/docs/energy_loss_overview.md` (generator chain, `.dat` formats, known
defects), `energy_loss/docs/dqdx_consistency_check.md` (provenance proof, the two parameter
sets, and the §6.1 forward-looking risk this doc realized).
