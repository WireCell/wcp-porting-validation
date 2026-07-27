# 66 — Diffusion reverted to DL = 4.0 / DT = 8.8, and what it does to 1000 data events

**Status: DONE and SHIPPED. NOT bit-identical — this is a deliberate physics
constant change, not a knob.** The SBND diffusion coefficients are back to
sbndcode's production pair on both the simulation side and the STM/PR
track-fitting side, reverting the 2026-07-25 retune of
[47](47_stm-bragg-reference-sbnd-retune.md) §6a on owner instruction after he
clarified the numbers with colleagues.

| | before 2026-07-25 | 2026-07-25 → -27 | **now** |
|---|---|---|---|
| `DL` | 4.0 cm²/s | 6.5781 | **4.0** |
| `DT` | 8.8 cm²/s | 13.1349 | **8.8** |

Source of truth: `sbndcode/sbndcode/WireCell/wcsimsp_sbnd.fcl`, which sets
`DL: 4.0` / `DT: 8.8` at three sites. (9.8 is uBooNE's `DT`, not SBND's; the
uBooNE presets in `TrackFitting.h:37-38` and `qlport/uboone_track_fitting.json`
are deliberately **untouched**.)

**Headline on 1000 data events: the tagging barely moves, and where it moves it
is mostly right.** Over 11 426 bundles the bundle set is identical, TGM / FC / LM
do not flip once, and **STM flips 11 times (0.10 %)**, all in-beam.

The owner hand-scanned the 11 (§11, tag `d66flip`, 8 of 11 with a saved verdict)
and the result is **perfectly directional**:

| direction | scanned | code correct | code wrong |
|---|---|---|---|
| `STM → nu` — the revert drops a tag | 4 | **4** | 0 |
| `nu → STM` — the revert adds a tag | 4 | 0 | **4** |

So the revert's only mistakes are the four newly-*added* STM tags. Its dropped
tags are right every time — including **283463:14 and 315849:10, whose doc-62
labels the owner revised to not-STM**, which retires two of the four
"regressions" §5 reported against the old baseline (§11.1). §5's "3 → 6 errors"
headline therefore **does not stand**; a final doc-63 number needs the three
unscanned bundles (281632:8, 319809:20, 321107:13).

This also inverts §6a/§9's proposed follow-up: the guard that now *rejects*
283463:14 on a 3.6 cm charge desert is producing the verdict the owner wants, so
loosening it would break it. The open problem is the four `nu→STM` bundles where
a veto that used to fire has **stopped** (§11.2).

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit

# 1. sec 1 -- DL/DT are inert in every compiled reco config (the pctree-reuse licence)
for j in wct-pr-perevt wct-clus-matching-perevt wct-clustering; do
  wcsonnet --tla-code DL=6.5781 --tla-code DT=13.1349 pgrapher/experiment/sbnd/$j.jsonnet > /tmp/old.json
  wcsonnet --tla-code DL=4.0    --tla-code DT=8.8      pgrapher/experiment/sbnd/$j.jsonnet > /tmp/new.json
  echo "$j difflines=$(diff /tmp/old.json /tmp/new.json | wc -l) DLkeys=$(grep -c '"DL"' /tmp/new.json)"
done
# and the sim side DOES carry them:
echo 'local p=import "pgrapher/experiment/sbnd/simparams.jsonnet"; {DL:p.lar.DL, DT:p.lar.DT}' > /tmp/sp.jsonnet
wcsonnet /tmp/sp.jsonnet          # -> 3.9999999999999998e-07 / 8.8000000000000015e-07

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# 2. sec 4 -- the two 1000-event arms.  ONE binary; only the fit JSON differs.
ls -d work-mcp1kall-d59k/nusel_evt* | sed 's#.*nusel_evt##' | sort -n > /home/xqian/tmp/d66-events.txt
EV="$(tr '\n' ' ' < /home/xqian/tmp/d66-events.txt)"
SBND_TRACKFIT_JSON=$PWD/stm_campaign/sbnd_track_fitting_d47.json \
  STM_EVENTS="$EV" NJOBS=8 ./stm_campaign/run_round.sh d66old     # 6.5781 / 13.1349
STM_EVENTS="$EV" NJOBS=6 ./stm_campaign/run_round.sh d66new       # 4.0 / 8.8  (in-tree default)

# 3. secs 4-5 -- the three reports
python3 d66_flip_report.py work-stmcamp-d66old work-stmcamp-d66new
python3 d66_flip_report.py work-stmcamp-d66old work-stmcamp-d66new --beam-only
python3 stm_campaign/score_round.py --round work-stmcamp-d66new --ref work-stmcamp-d66old
python3 stm_campaign/score_full.py  --round work-stmcamp-d66new --ref work-stmcamp-d66old
./d60_ab_report.py work-stmcamp-d66old work-stmcamp-d66new

# 4. sec 6 -- per-bundle fit diagnostics behind the flips
for t in d66old d66new; do
  python3 stm_campaign/extract_stm_diag.py --root work-stmcamp-$t --out /home/xqian/tmp/d66-diag/$t
done

# 5. sec 10 -- the 11 dQ/dx-vs-residual-range overlay figures
./d66_flip_plots.sh                        # -> pics/d66/ + overlay-numbers.txt

# 6. sec 11 -- serve the 11 for hand scan, OLD arm as the --prev baseline so the
#    amber change flag marks exactly the diffusion-induced flips.  ABSOLUTE paths.
SB=$PWD
nusel_display/serve_nusel_scan.sh 5011 --tag d66flip --charge-src pr \
  --prev $SB/work-stmcamp-d66old:d66old \
  --prev $SB/work-mcp1kall-d59k:s61mis \
  --prev $SB/work-mcp1kall-d59k:s59k \
  $(for e in 58345 58755 63163 281632 283463 289295 315849 317543 319809 321107 390864; do \
      echo $SB/work-stmcamp-d66new/nusel_evt$e/nusel-evt$e.tsv; done)

# 7. sec 11 -- score the scan (prints the owner's raw comment beside each verdict)
./d66_scan_score.py                        # -> pics/d66/scan-score-d66flip.txt

# 8. Bee sets for the 11 (outward-facing; owner asked).  img-global +
#    clustering-global + stm_fit-global, one set per arm -- the two differ ONLY
#    in stm_fit-global, verified by member-content hashing.
./make_scan_bee.sh bee-d66 work-stmcamp-d66new bee-d66/flip11-new.txt
./make_scan_bee.sh bee-d66 work-stmcamp-d66old bee-d66/flip11-old.txt
```

Bee sets (2026-07-27): **new 4.0/8.8** →
`beb8075f-3a39-4a38-af7d-98e08df90ff2`, **old 6.5781/13.1349** →
`35ee52b3-4614-4d8f-9373-7f8ec8731515`, both under
`https://www.phy.bnl.gov/twister/bee/set/<uuid>/event/list/`. Bee index *i* =
line *i+1* of `bee-d66/flip11-*.txt`; PR→img cluster-id map in
`bee-d66/flip11-*.zip.stmid-map.txt`.

Labels: `work-stmcamp-d66old` (6.5781/13.1349) and `work-stmcamp-d66new`
(4.0/8.8), both 1000 events, both `rc=0` on 1000/1000. Doc 55 §12 re-fits its
three bundles from the same two roots.

---

## 0. Why this change was made

The retune was made on 2026-07-25 because 6.2/9.8 were acknowledged placeholders
inherited from the Q/L chain and uBooNE (doc 47 §6a). It replaced them with
owner-supplied "physical" values on both the fit and the simulation. Doc 47 §6a
recorded the problem that survived it, and it is the reason for this revert:

> **CAVEAT: official MC already on disk (mcp2025c reco1) was simulated with
> 4.0/8.8, so the fit over-smears those waveforms by ~15 % transverse until they
> are re-simulated.**

Every sample this chain actually runs on — including the 1000-event MCP2025C
**data** sample — comes from SBND production at 4.0/8.8. The fit was assuming a
transverse footprint the waveforms do not have. The owner confirmed 4.0/8.8 with
colleagues, so both sides go back and fit, our sim, and the samples on disk now
share one diffusion model.

**The case for 4.0/8.8 was provenance, not fit quality** — the fit should assume
what the simulation used, which is now true and was not before. That remains the
primary argument, and no χ² measurement supports either pair (doc 55 §12.3; §9
item 5).

**The hand scan has since added a physics argument, though.** §11: on the 8
flips the owner scanned, 4.0/8.8 is right on all 4 where it *drops* an STM tag
(including two bundles whose doc-62 labels he revised) and wrong on all 4 where
it *adds* one. So the revert is a net improvement on the STM side, not merely a
provenance fix — with one specific failure mode left to chase (§11.2).

### 0a. What MicroBooNE uses, and a field-consistency check

uBooNE carries **one** pair everywhere — sim and fit agree, and both trace to the
WCP prototype:

| where | `DL` | `DT` |
|---|---|---|
| `cfg/pgrapher/experiment/uboone/params.jsonnet:9-10` — simulation | 6.4 | 9.8 |
| `qlport/uboone_track_fitting.json:3-4` — track fit | 6.4e-7 | 9.8e-7 |
| `clus/inc/WireCellClus/TrackFitting.h:37-38` — C++ presets | 6.4 | 9.8 |
| `prototype_base/pid/src/PR3DCluster_multi_dQ_dx_fit.h:43-44` (+2 more) | 6.4 | 9.8 |

That is where SBND's placeholders came from: `TrackFitting.h` labels 6.4/9.8
"LArTPC standard values", and they were never an SBND measurement. (One
inconsistency worth knowing: the prototype's older generative-sim path uses a
*different* pair, `DL = 5.3`, `DT = 12.8` — `prototype_base/nav/src/GenerativeFDS.cxx:97-98`,
`nav/src/DetGenerativeFDS.cxx:151-152`, `signal/src/DetGenerativeFDS.cxx:175-176`.
Only 6.4/9.8 reached the toolkit.)

**A field-consistency argument, stated as an argument and not a measurement.**
uBooNE drifts at 0.273 kV/cm (`drift_speed = 1.098 mm/µs`; doc 47 §3); SBND is at
0.5 kV/cm (`1.563 mm/µs`). Diffusion coefficients fall with increasing drift
field, so at nearly double the field SBND's pair should sit *below* uBooNE's.
4.0/8.8 does (both below 6.4/9.8). The retuned pair did not: **13.1349 would put
SBND's transverse diffusion 34 % above uBooNE's at twice the field**, which is
the wrong direction. This is a sanity check on the sign of the change, not a
derivation of either number — the magnitude depends on the electron-energy
dependence of the transverse coefficient, which is not evaluated here. But it
does point the same way as §11's scan: the §5 regressions are more likely to be
tagger-threshold artifacts than evidence that 4.0/8.8 is wrong, and §11
subsequently retired two of the four outright.

## 1. The compiled configs do not change — and why that proves nothing about the fit

`DL`/`DT` appear as TLAs on four jobs. Compiling each at both pairs:

| job | diff lines | `"DL"` keys emitted |
|---|---|---|
| `wct-pr-perevt.jsonnet` | **0** | 0 |
| `wct-clus-matching-perevt.jsonnet` | **0** | 0 |
| `wct-clustering.jsonnet` | **0** | 0 |
| `sbnd_xin/ql_dump_scalar.jsonnet` | **0** | 0 |

So on the reco side these TLAs are inert: they only ever reach a sim `Drifter`,
and none of these jobs builds one. This re-measures doc 47 §6a's claim at current
HEAD (after the `origin/master` merge that rewrote `pnode()`/`unique_list`, and
after doc 64's config promotion) rather than trusting it, and it is the licence
for §4's shortcut: the stored Q/L pctrees are still valid inputs, so only the PR
tagger tail needs re-running.

> **The trap.** `sbnd_track_fitting.json` is read at **runtime** by
> `TrackFitting` via `trackfitting_config_file` — with a plain `std::ifstream`,
> never through the compiled jsonnet. A byte-identical compiled config therefore
> says **nothing** about whether the fit changed. In this change the compiled
> configs are identical *and* the fit is different, which is exactly the
> combination that would let a careless gate report "byte-identical".

The **simulation** side does carry them, and there the compiled config moves as
intended: `simparams.lar.DL` = `3.9999999999999998e-07`, `DT` =
`8.8000000000000015e-07` (internal units, cm²/s = 1e-7), and the three
`fhicl/*.fcl` `structs` read `DL: 4.0` / `DT: 8.8`.

## 2. Sites changed

**Fit side — the only live consumer for data reco**

| file | change |
|---|---|
| `cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json` | `DL` 6.5781e-07 → **4.0e-07**, `DT` 1.31349e-06 → **8.8e-07**, and `_comment_diffusion` rewritten (the MC caveat inverts) |

**Simulation side — the five sites doc 47 §6a lists**

| file | role |
|---|---|
| `cfg/.../sbnd/simparams.jsonnet` | jsonnet `lar.DL`/`lar.DT` → the `Drifter` |
| `cfg/.../sbnd/fhicl/standard_detsim_sbnd.fcl` | LArSoft `structs.DL`/`DT` |
| `cfg/.../sbnd/fhicl/wirecell_pgrapher_detsim_sbnd.fcl` | ditto |
| `cfg/.../sbnd/fhicl/wirecell_tbb_deposet_detsim_sbnd.fcl` | ditto |
| `cfg/.../sbnd/wct-clustering.jsonnet` | sim-side clustering TLA defaults |

**Inert TLA defaults, moved for consistency** —
`cfg/.../sbnd/wct-pr-perevt.jsonnet`, `wct-clus-matching-perevt.jsonnet`,
`sbnd_xin/ql_dump_scalar.jsonnet`, and the runners `run_nusel_evt.sh`,
`run_pr_evt.sh`, `run_ql_evt.sh`, `run_clus_evt.sh`, `run_clust_QL_evt.sh`,
`profile_pr65.sh`.

**Deliberately NOT touched** — `clus/inc/WireCellClus/TrackFitting.h:37-38`
(uBooNE 6.4/9.8 C++ presets), `qlport/uboone_track_fitting.json`, and
`sbndcode/` (a read-only reference checkout; our values now *agree* with it, so
there is nothing to propagate). **No C++ changed, so no rebuild** — verified by
`grep -rn "6\.5781\|13\.1349" clus/ match/ flash/` returning nothing.

### 2a. A/B plumbing: `SBND_TRACKFIT_JSON`

`run_nusel_evt.sh` hardcoded the fit JSON path, which would have forced a
time-ordered "run baseline, edit, run new" A/B that can never be re-run. It now
reads

```bash
TFJSON=${SBND_TRACKFIT_JSON:-$SBND_DIR/sbnd_track_fitting.json}
```

and echoes `trackfitting_config=<path> "DL":… "DT":…` into every per-event log,
so an arm's diffusion is recoverable from its own output afterwards. The
pre-revert file is kept at
`sbnd_xin/stm_campaign/sbnd_track_fitting_d47.json`. Provenance check on the two
arms — exactly **one** unique line each:

```
d66old: stm_campaign/sbnd_track_fitting_d47.json "DL":6.5781e-07,"DT":1.31349e-06,
d66new: sbnd_track_fitting.json                  "DL":4.0e-07,"DT":8.8e-07,
```

## 3. What the change does to the fit's predicted footprint

`TrackFitting` (`clus/src/TrackFitting.cxx:6120-6131`) predicts each fitted
point's charge footprint as

```
σ_L         = hypot( sqrt(2·DL·t_drift), add_sigma_L ) / tick_width
σ_T,{u,v,w} = hypot( sqrt(2·DT·t_drift), {ind,ind,col}_sigma_*_T ) / pitch
```

At SBND's `v_drift = 1.563 mm/µs`, `add_sigma_L = 2.4876 mm`,
`col_sigma_w_T = 0.09403 mm`, `ind_sigma_u_T = 0.48359 mm`, `t_drift` floored at
50 µs:

| drift x [cm] | t_drift [µs] | σ_T,W old → new [mm] | ratio | σ_T,U old → new | ratio | σ_L old → new | ratio |
|---|---|---|---|---|---|---|---|
| 0 (floored) | 50.0 | 0.3744 → 0.3112 | **0.831** | 0.6043 → 0.5673 | 0.939 | 2.5008 → 2.4956 | 0.998 |
| 50 | 319.9 | 0.9215 → 0.7562 | **0.821** | 1.0364 → 0.8927 | 0.861 | 2.5708 → 2.5385 | 0.987 |
| 100 | 639.8 | 1.2998 → 1.0653 | **0.820** | 1.3837 → 1.1661 | 0.843 | 2.6514 → 2.5884 | 0.976 |
| 200 (full) | 1279.6 | 1.8358 → 1.5036 | **0.819** | 1.8961 → 1.5767 | 0.832 | 2.8056 → 2.6855 | 0.957 |

**Transverse σ narrows ~18 % at every drift distance; longitudinal narrows
0.2–4.3 %.** Note this is a *larger* step than doc 47 §6a measured — that table
compares 6.2/9.8 → 6.5781/13.1349 (σ_T ×1.157), whereas this revert is
6.5781/13.1349 → 4.0/8.8 in the opposite direction. Do not read §6a's ratios as
the revert's effect.

Two consequences, in opposite directions:

- **Good:** doc 47 §6a worried that a *wider* σ_T against an unchanged
  `search_range = 10` / `time_tick_cut = 20` window was "exactly the shape of a
  silently truncated footprint". Narrowing removes that risk entirely.
- **The cost:** a narrower predicted footprint apportions measured charge over
  fewer wires/ticks per fitted point, so per-point `dQ` moves — and low-charge
  points move *down*. That is the mechanism behind every flip in §5–6.

## 4. The 1000-event A/B

Two arms, **same binary**, same production flag set (`run_full1k_nusel.sh`'s
`NUF` verbatim: `-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm
-main-pair-real -fvx 2.5 -fvy 3 -stm-fit -mip 56000 -unmerge-assoc`, which since
doc 63 means all seven STM guards default ON), same input pctrees symlinked
read-only from `work-mcp1kall-d59k` (M11/M13 — nothing under d59k was written).
Only the fit JSON differs.

Both arms: **1000/1000 `rc=0`**. Notably evt 278794 — the doc-59 deterministic
crash — completes in both, so doc 60's fix holds at this HEAD.

### 4a. Verdict flips, all 11 426 bundles

```
arm OLD work-stmcamp-d66old: 1000 events, 11426 bundles
arm NEW work-stmcamp-d66new: 1000 events, 11426 bundles
all bundles: 11426 present in both arms; 0 only-OLD, 0 only-NEW

tagger    unchanged  flipped  transitions (old->new)
tgm           11426        0  none
stm           11415       11  0->1:5, 1->0:6
fc            11426        0  none
lm            11426        0  none
label        11415       11  STM->nu-candidate: 6, nu-candidate->STM: 5

bundles with ANY tagger flip: 11 / 11426 (0.10 %)
```

- **The bundle set is identical** — 0 bundles in only one arm. The un-merge and
  the whole clustering-scope machinery are unaffected, so every comparison below
  is bundle-for-bundle and nothing is being silently dropped.
- **TGM, FC and LM do not flip once.** They do not consume the track fit.
- **Every flip is STM**, and every flipped bundle is `in_beam=1`, so the
  restriction to the 923 in-beam bundles gives the same 11 flips — i.e. **11 of
  923 in-beam bundles (1.2 %)** in the population the hand scan actually judges.

The 11:

| event:main | len [cm] | old → new |
|---|---|---|
| 58345:7 | 97.0 | STM → nu-candidate |
| 58755:21 | 401.1 | nu-candidate → STM |
| 63163:6 | 198.8 | STM → nu-candidate |
| 281632:8 | 54.5 | STM → nu-candidate |
| 283463:14 | 86.2 | STM → nu-candidate |
| 289295:15 | 75.2 | nu-candidate → STM |
| 315849:10 | 91.3 | STM → nu-candidate |
| 317543:15 | 149.6 | nu-candidate → STM |
| 319809:20 | 73.6 | nu-candidate → STM |
| 321107:13 | 226.9 | STM → nu-candidate |
| 390864:16 | 174.3 | nu-candidate → STM |

## 5. Against the doc-62 owner baseline: 3 → 6 errors — SUPERSEDED by §11

> **Read §11 first.** The owner hand-scanned all 11 flips on 2026-07-27 and
> **revised the doc-62 verdict on 283463:14 and 315849:10 to not-STM**, which
> means the code is correct on both and two of the four "regressions" below are
> not regressions at all — the doc-62 labels were stale. The "3 → 6 errors"
> headline of this section **does not stand**; §11.1 has the revised accounting
> and explains why a final number needs three more bundles scanned. The section
> is kept as written because it is the doc-62-baseline measurement, which is the
> input to that revision.

This is the check that answers "should the final tagging change much", because
the doc-62 set is the only place with owner truth. Doc 63's shipped state was
**3 errors on 72 owner-adjudicated bundles**, measured with 6.5781/13.1349.

```
== work-stmcamp-d66new vs owner baseline (72 bundles) ==
correct: 66  still-wrong: 2  FIXED: 1  REGRESSED: 4  missing: 0
  FIX  321107:13  (code-FALSE-STM)
  REG  281632:8   (code-STM-correct)
  REG  283463:14  (code-STM-correct)
  REG  315849:10  (code-STM-correct)
  REG  319809:20  (code-not-STM-correct)
still wrong: 48895:17 278662:1
```

**Net: 3 errors → 6.** Doc 63's acceptance standard for a round was "fixes ≥ 1
error and regresses **0** of the 56 correct verdicts"; this change regresses 4 of
them. It is not a doc-63 round and was not held to that bar — the owner directed
the constant — but the bar is the right yardstick for reporting the cost, and the
cost is four owner-confirmed verdicts.

The remaining 6 flips are on non-adjudicated bundles, where the doc-61 AI scan is
the only reference (`score_full.py`, 923 in-beam bundles compared):

| bundle | flip | AI-scan verdict | t [µs] | len [cm] |
|---|---|---|---|---|
| 58345:7 | STM → not | STM | 1.568 | 97.0 |
| 58755:21 | not → STM | nu | 1.988 | 401.1 |
| 63163:6 | STM → not | nu | 1.881 | 198.8 |
| 289295:15 | not → STM | nu | 1.338 | 75.2 |
| 317543:15 | not → STM | nu | 1.463 | 149.6 |
| 390864:16 | not → STM | nu | 0.778 | 174.3 |

Five of the six move *away* from the AI scan's reading (four `nu → STM` where the
scan said nu, one `STM → not` where the scan said STM), and one (63163:6) moves
toward it. These are not truth, they are unadjudicated — listed so the owner can
judge them, per doc 63's protocol, not scored.

## 6. Why each baseline verdict flipped

All five reach the dQ/dx evaluation (`stmfit=eval` in both arms), so none is a
pre-fit exit. The exit status tells the rest (statuses per doc 63 §1: 0 accepted,
2 long leftover, 3 dQ/dx eval, 5 detect_proton, 7 accept_guards):

| bundle | owner | status old → new | mechanism |
|---|---|---|---|
| 281632:8 | STM | 0 → **3** | dQ/dx eval now rejects |
| 283463:14 | STM | 0 → **7** | `accept_guards`: *"fit bridges a 3.6 cm charge desert (one-objectness)"* |
| 315849:10 | STM | 2,0 → 2,**3** | dQ/dx eval now rejects on pass 2 |
| 319809:20 | not-STM | **5** → 0 | `detect_proton` stopped firing → accepted |
| 321107:13 | not-STM | 0 → **3** | dQ/dx eval now rejects (**the fix**) |

### 6a. The charge-desert threshold has lost its margin

283463:14 is the clearest and the most concerning. Doc 63 §1.1 introduced the
desert veto with:

> longest contiguous stretch of the fitted muon segment with per-point
> dQ/dx < 0.02 MIP: 278794:7 = 23.4 cm, 285443:7 = 6.7 cm, **every other
> accepted bundle (false or correct) = 0.0 cm.** A ≥ 3 cm veto separates with
> infinite margin on this set.

Under 4.0/8.8 that same bundle develops a **3.6 cm** desert and trips the 3 cm
veto. The mechanism is exactly §3's: a narrower predicted footprint pushes
low-charge points lower, so more of them fall under the `0.02·MIP` floor. The
"infinite margin" was a property of the 6.5781/13.1349 fits, not of the
algorithm. **Every doc-63 threshold expressed in MIP-relative dQ/dx terms is in
the same position** and would need re-deriving at 4.0/8.8 to recover doc 63's
score: the desert floor (`0.02 MIP`, `≥3 cm`), the `ratio2 > 2` cap, the
`spike > 3.2 / shoulder < 1.8 MIP` pair, `deficit_guard`'s
`end5 med < 0.6 / last15 max < 2 MIP`, and `vertex_kink_guard`'s `post > 2.2 MIP`.

That is a doc-63-scale retuning campaign against the owner baseline, and it is
**not done here** — this doc changes a constant and measures the consequence.

### 6b. How much the fit actually moved

Across the 65 baseline bundles with an eval record in both arms, **zero have
bit-identical eval quantities** — but the median shift is small:

| quantity | median Δ | median \|Δ\| | median relative | p90 \|relative\| |
|---|---|---|---|---|
| `ks1` | +0.0008 | 0.0028 | +1.7 % | 34.2 % |
| `ks2` | +0.0003 | 0.0025 | +0.4 % | 12.2 % |
| `ratio1` | +0.0000 | 0.0110 | +0.0 % | 8.1 % |
| `ratio2` | +0.0000 | 0.0080 | +0.0 % | 7.1 % |
| `ave_res_mip` | −0.0100 | 0.0300 | −0.9 % | 6.8 % |
| `res_len_cm` | +0.0000 | 0.0000 | +0.0 % | 8.5 % |

So the typical bundle's fit is perturbed at the sub-percent-to-few-percent level
on the charge ratios, with the Kolmogorov statistics noisier (p90 34 % on `ks1`).
**That is the honest reading of "the tagging should not change much": it does
not, in the median.** The 11 flips are bundles that were already sitting within a
few percent of a cut, plus 283463:14 crossing a hard geometric threshold.

`ratio2` for 283463:14 illustrates it: 1.477 → 1.658, both comfortably under the
`> 2` cap doc 63 shipped — so the cap is *not* what killed it, and doc 63's
"highest correct accepting call is ratio2 = 1.48" is reproduced exactly (1.477)
in the old arm.

## 7. Archive-level comparison

`d60_ab_report.py` compares both arms' `mabc-pr.zip` and
`pctree-pr-evt<ID>.tar.gz` by **member content** (`abtest/hash_archive.py`, never
raw `cmp`/`md5sum` — M2) and the `nusel-evt<ID>.tsv` tables by direct hash;
`tracking-stm.root` is skipped as ROOT files embed a creation timestamp.
Confirmed first that both arms write their *own* PR products (regular files, not
symlinks resolving to one shared d59k copy, which would have reported a spurious
IDENTICAL).

**Expected and observed: DIFFER**, and the shape of the difference is itself a
consistency check.

```
arm A: work-stmcamp-d66old
arm B: work-stmcamp-d66new
events compared : 1000
  identical     : 665
  DIFFERENT     : 335
  only in A     : 0
  only in B     : 0
```

Member-level breakdown over the 335 differing events:

| member | events differing | what it means |
|---|---|---|
| `mabc-pr.zip` | 335 | the Bee layers carrying the fit |
| `pctree-pr-evt<ID>.tar.gz` | 335 | the persisted `stm_fit`/`stm_eval`/`stm_pass` PCs |
| `nusel-evt<ID>.tsv` | **11** | the per-bundle verdict table |

Three things follow, none of which the flip report could show on its own:

1. **665 events are byte-identical.** Those are events where the STM fit never
   ran at all — no in-beam bundle, or the in-beam bundle was TGM/LM-tagged before
   the STM stage — so nothing consumed `DL`/`DT`. That is a useful negative
   control: the change reaches exactly the code path it should and no other.
2. **335 events have a changed fit but an unchanged verdict.** Every one of the
   335 differs in *both* archives, i.e. the fit genuinely moved on a third of the
   sample, and on 324 of them the tagger still reached the same conclusion.
   That is the archive-level version of §6b's "the median bundle barely moves".
3. **Exactly 11 events differ in the verdict table** — and they are precisely the
   11 bundles §4a found. Two independent instruments (member-content hashing of
   the tables vs. parsing them into a confusion matrix) agree on the same 11
   events, so the flip count is not a parsing artifact.

Runtime note: this comparison hashes ~4000 archives member-by-member over NFS and
took **~50 min**. It prints nothing until the whole loop finishes, so a 0-byte
output file mid-run is expected and not a hang.

## 8. Doc 55's three bundles

[55](55_dqdx-vs-rr-three-bundles.md) §12 re-fits the doc-55 dQ/dx-vs-residual-range
bundles from these same two arms. Summary: `npts`, fitted path length and median
`dx` are **identical to the printed digit** (`dx` is geometric — `DL`/`DT` do not
enter it), the proton candidate holds at **1.91×** the muon curve, and the two
muons stay on the muon curve (0.98 / 1.04 vs a published ±2 %). `reduced_chi2`
moves in both directions, so those three bundles favour neither pair.

## 9. What is left open

1. **Why did four `nu→STM` vetoes stop firing?** ~~The doc-63 guard thresholds
   are stale at 4.0/8.8 (§6a) and need re-deriving.~~ **Reframed by §11.2:** the
   guards that now *reject* are producing the verdicts the owner wants, so
   loosening `guard_desert_cm` — §6a's "clean one-knob fix" — would make things
   worse, and is now explicitly **not** recommended. The real defect is the four
   bundles that gained a wrong STM tag because a veto stopped firing:
   319809:20 (`detect_proton`, §6), 58755:21, 289295:15, 317543:15, 390864:16.
   Their `stm_pass` status went to 0 (accepted) in the new arm; the first step is
   finding which veto each one used to trip, which for `detect_proton` needs
   `ks3` / `dQ_dx[max_bin]` at TRACE level (not in the persisted record).
2. **Three flips still unscanned** — 281632:8, 319809:20, 321107:13 (§11).
   Until they carry a verdict the doc-63 score cannot be restated. 319809:20 is
   the important one: it is `nu→STM`, the direction that is 0/4 so far, and it is
   the owner-confirmed proton of §6.
3. **Doc 55 §§6–11** (the curated 12-muon/1-proton sample, the 12-proton
   population, and the §§7–10 recombination fits plus
   `nusel_display/stm_ref_dqdx.json`) are left at their published values; §12.4
   explains why that is defensible and what redoing them would cost.
4. **Locally-generated simulation is not bit-identical either** — simulated
   waveforms change. Nothing in flight depends on locally-generated SBND sim, so
   no sample was regenerated here.
5. **No unbiased fit-quality comparison exists.** §10.1 notes `reduced_chi2`
   dropping on 9 of the 11 flipped bundles, but those 11 were selected *because*
   their fit moved, so the sample is biased and the observation cannot support
   "4.0/8.8 fits SBND data better". The clean version is median `reduced_chi2`
   old vs new over all 923 in-beam bundles, which needs `tracking-stm.root`
   parsed for every bundle in both arms (the `stmfit_particle_overlay.py` path,
   looped) — not run. §11 has since answered the *verdict* version of that
   question directly from the owner's eyes, so this is now a supporting
   measurement rather than the deciding one.

## 10. Per-bundle dQ/dx vs residual range for all 11 flips

One figure per flipped bundle, **old and new fit of the same bundle overlaid** on
the SBND `box` per-particle reference curves, for hand investigation before a
scan.

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./d66_flip_plots.sh                    # -> pics/d66/d66_evt<ID>_main<M>.png (11)
                                       #    + pics/d66/overlay-numbers.txt
```

`d66_flip_plots.sh` pins the **deciding pass** per bundle — the pass whose status
differs between the arms, read from each arm's
`persist_stm_fit: cluster N stmfit pass=P status=S`. Ten bundles decide on
pass 0; **315849:10 is the only two-pass case** (pass 0 exits `status=2` in both
arms, pass 1 is where the verdict moves), so its block is 101, not 100. Getting
that wrong would plot a pass that never flipped.

**The trajectories are identical.** Every one of the 11 has the same `npts`,
`kink`, `exit_L` and `left_L` in both arms — verified from the
`persist_stm_fit` lines. So in each figure the two traces sample *the same
points at the same residual ranges*, and differ **only** in the charge
apportioned to each. That is what makes these readable as "what the diffusion
change did to dQ/dx" rather than as two different fits.

| bundle | flip | vs owner truth | npts | median `reduced_chi2` old → new | median fit/MuonBox old → new | new/old | owner (§11) | code right? | figure |
|---|---|---|---|---|---|---|---|---|---|
| 281632:8 | STM → nu | **REGRESSION** (owner STM) | 78 | 1.67 → **1.53** | 1.52 → 1.50 | 0.987 | — | *not scanned* | [png](../pics/d66/d66_evt281632_main8.png) |
| 283463:14 | STM → nu | **REGRESSION** (owner STM) | 40 | 3.60 → 3.58 | 0.46 → **0.51** | **1.109** | not-STM | **YES** | [png](../pics/d66/d66_evt283463_main14.png) |
| 315849:10 | STM → nu | **REGRESSION** (owner STM) | 115 | 0.98 → **0.94** | 0.93 → 0.91 | 0.978 | not-STM | **YES** | [png](../pics/d66/d66_evt315849_main10.png) |
| 319809:20 | nu → STM | **REGRESSION** (owner not-STM) | 115 | 1.82 → **1.78** | 0.98 → 0.99 | 1.010 | — | *not scanned* | [png](../pics/d66/d66_evt319809_main20.png) |
| 321107:13 | STM → nu | **FIX** (owner not-STM) | 370 | 2.14 → 2.15 | 0.87 → 0.87 | 1.000 | — | *not scanned* | [png](../pics/d66/d66_evt321107_main13.png) |
| 58345:7 | STM → nu | unadjudicated (AI: STM) | 74 | 1.79 → **1.71** | 0.84 → 0.84 | 1.000 | not-STM | **YES** | [png](../pics/d66/d66_evt58345_main7.png) |
| 58755:21 | nu → STM | unadjudicated (AI: nu) | 635 | 2.03 → **1.96** | 0.89 → 0.92 | 1.034 | not-STM | **NO** | [png](../pics/d66/d66_evt58755_main21.png) |
| 63163:6 | STM → nu | unadjudicated (AI: nu) | 285 | 1.85 → 1.84 | 0.90 → 0.91 | 1.011 | not-STM | **YES** | [png](../pics/d66/d66_evt63163_main6.png) |
| 289295:15 | nu → STM | unadjudicated (AI: nu) | 52 | 1.43 → **1.22** | 0.87 → 0.86 | 0.989 | not-STM | **NO** | [png](../pics/d66/d66_evt289295_main15.png) |
| 317543:15 | nu → STM | unadjudicated (AI: nu) | 169 | 2.26 → 2.33 | 0.85 → 0.86 | 1.012 | not-STM | **NO** | [png](../pics/d66/d66_evt317543_main15.png) |
| 390864:16 | nu → STM | unadjudicated (AI: nu) | 278 | 1.31 → **1.21** | 0.92 → 0.92 | 1.000 | not-STM | **NO** | [png](../pics/d66/d66_evt390864_main16.png) |

Per-bundle binned dQ/dx tables, ratios against all five reference curves, and the
`/MIP` column for both arms are in
[`pics/d66/overlay-numbers.txt`](../pics/d66/overlay-numbers.txt).

### 10.1 What the numbers say before the scan

**The overall charge level barely moves.** `median fit/MuonBox` changes by
< 2 % on 10 of the 11 bundles. The single exception is **283463:14 at +11 %**
(0.46 → 0.51) — the bundle that also crossed the charge-desert threshold (§6a),
and the one whose fit is worst-behaved in both arms (`reduced_chi2` 3.6, far above
every other bundle here). So the verdict flips are **not** driven by a
normalisation shift; they are threshold crossings by bundles that were already
marginal, which is what §6b's median-level statistics said from the other
direction.

**`reduced_chi2` mostly improves.** 9 of 11 are lower or flat in the new arm
(largest drops: 289295:15 1.43 → 1.22, 390864:16 1.31 → 1.21, 281632:8
1.67 → 1.53); only 321107:13 (2.14 → 2.15) and 317543:15 (2.26 → 2.33) worsen.

> **Do not read that as evidence 4.0/8.8 fits better.** These 11 bundles were
> selected *because* they flipped, i.e. because their fit moved most — a biased
> sample by construction. Doc 55 §12.3, on three bundles chosen for other
> reasons, found `reduced_chi2` moving in both directions with no systematic
> direction. An unbiased χ² comparison over all 923 in-beam bundles has **not**
> been run; §9 item 5.

**Where to look in each figure.** The `STM → nu` cases (281632:8, 283463:14,
315849:10, 321107:13, 58345:7, 63163:6) lost their tag, so the question is
whether the Bragg rise near rr → 0 still looks like a stopping muon in the new
trace. The `nu → STM` cases (319809:20, 58755:21, 289295:15, 317543:15,
390864:16) gained one, so the question is the opposite: does the new trace's end
region look like a genuine stop, or like a ν-vertex spike / proton that the old
fit correctly vetoed? 319809:20 is the one to start with — it is the
owner-confirmed **proton** whose `detect_proton` veto stopped firing (§6, and the
one flip whose mechanism §6 could not pin down from stored data).

## 11. The owner's hand scan of the 11 flips (2026-07-27) — and what it overturns

The owner scanned the flips on :5011 against the OLD arm as `--prev` (so every
bundle he saw was flagged "this verdict moved because of the diffusion"), tag
`d66flip`. **8 of the 11 carry a saved verdict**; 281632:8, 319809:20 and
321107:13 were visited (`.scan_state` written) but no verdict was saved.

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./d66_scan_score.py                      # --tag d66flip --root work-stmcamp-d66new
```

Verdicts are free text in the bundle `comment` (the label buttons were not
used), so `d66_scan_score.py` applies **one** stated keyword rule — "not a stm" /
"nu candidate" ⇒ owner verdict not-STM — and always prints the raw comment beside
it so the classification is checkable by eye. All 8 were unambiguous; none
needed a judgment call.

| bundle | direction | code (new) | owner | code right? | doc-62 said | owner's comment |
|---|---|---|---|---|---|---|
| 283463:14 | STM→nu | nu-candidate | not-STM | **YES** | STM | *"nu candidate is OK"* |
| 315849:10 | STM→nu | nu-candidate | not-STM | **YES** | STM | *"not a STM"* |
| 58345:7 | STM→nu | nu-candidate | not-STM | **YES** | — | *"Nu candidate, not a STM"* |
| 63163:6 | STM→nu | nu-candidate | not-STM | **YES** | — | *"nu candidate, not a STM, vertex activity"* |
| 58755:21 | nu→STM | STM | not-STM | **NO** | — | *"Not a STM, it should be a TGM though"* |
| 289295:15 | nu→STM | STM | not-STM | **NO** | — | *"Not a STM"* |
| 317543:15 | nu→STM | STM | not-STM | **NO** | — | *"Not a STM"* |
| 390864:16 | nu→STM | STM | not-STM | **NO** | — | *"not a STM, "* |
| 281632:8 | STM→nu | — | — | *not scanned* | STM | — |
| 319809:20 | nu→STM | — | — | *not scanned* | not-STM | — |
| 321107:13 | STM→nu | — | — | *not scanned* | not-STM | — |

**4 correct, 4 wrong — and the split is perfectly directional:**

| direction | n scanned | code correct | code wrong |
|---|---|---|---|
| `STM → nu` (the revert drops a tag) | 4 | **4** | 0 |
| `nu → STM` (the revert adds a tag) | 4 | 0 | **4** |

Every flip where 4.0/8.8 *removed* an STM tag was the right call. Every flip
where it *added* one was wrong. The owner's verdict was not-STM on all 8 bundles
he scanned, so the revert's only mistakes are the four newly-added tags.

### 11.1 This overturns two of §5's four "regressions"

§5 scored the change against the **doc-62** baseline and reported 4 regressions,
because doc 62 recorded owner=STM for 283463:14 and 315849:10. **The owner has
now revised both**, looking at the same bundles with the new fit in front of
him — "nu candidate is OK" and "not a STM". So the new code is *right* on both,
and what §5 counted as damage was the old label being stale, not the tagger
getting worse. Revised state of the five doc-62 bundles:

| bundle | §5 called it | after the d66flip scan |
|---|---|---|
| 283463:14 | REGRESSION | **not a regression** — doc-62 label revised to not-STM, code correct |
| 315849:10 | REGRESSION | **not a regression** — doc-62 label revised to not-STM, code correct |
| 281632:8 | REGRESSION | **unresolved** — not scanned (`STM→nu`, so the §11 pattern predicts it is also fine) |
| 319809:20 | REGRESSION | **unresolved** — not scanned (`nu→STM`, the direction that is 0/4) |
| 321107:13 | FIX | **unresolved** — not scanned (`STM→nu`; doc 62 said not-STM, so still a fix on the old label) |

So **§5's "3 → 6 errors" no longer stands as written.** With the two revisions
folded in, only one of the four claimed regressions (319809:20) is still a
plausible regression, and it is unscanned. The doc-63 score cannot be restated
until 281632:8, 319809:20 and 321107:13 are scanned — that is the one thing
blocking a final number, and it is three bundles of work.

### 11.2 What this says about §6a's guard-threshold worry

§6a argued that the doc-63 guards are stale at the new diffusion, using
283463:14's 3.6 cm charge desert crossing the 3 cm veto as the evidence, and §9
item 1 proposed re-deriving the thresholds to recover doc 63's score. **The scan
removes the motivation for exactly that case**: the desert veto firing on
283463:14 produced the verdict the owner wants. Raising `guard_desert_cm` to 4.0
(the §6a "clean one-knob fix") would *break* it.

That inverts the priority. The problem is not the guards that now *reject*; it is
the four `nu→STM` bundles where a veto that used to fire has stopped —
319809:20's `detect_proton` (§6) is the documented example, and 58755:21,
289295:15, 317543:15, 390864:16 are four more. The work worth doing is finding
why those vetoes stopped firing, not loosening the ones that did.

### 11.3 Two observations outside the diffusion question

- **58755:21 should be TGM, not nu-candidate** — *"Not a STM, it should be a TGM
  though"*. TGM did not flip on any of the 11 (§4a), so this is a **pre-existing
  TGM miss** on a 401 cm through-going candidate, unrelated to the diffusion. It
  deserves its own look; it is not a regression of this change.
- **58345:7 contradicts the doc-61 AI scan.** §5 noted the AI scan called it STM
  and the code moved away from it; the owner says *"Nu candidate, not a STM"*, so
  the code was right and the AI scan was wrong there. Of the six §5 flips
  measured against the AI scan, the owner has now overruled the AI scan on the
  one case where the two disagreed in that direction.

### 11.4 Labels

`work-stmcamp-d66new/nusel_labels/d66flip/` — 8 label files + 11 `.scan_state`.
Fresh tag, nothing pre-existing written (M13). The three unscanned bundles can be
added under the same tag without disturbing the eight.

---

Companion docs: [47](47_stm-bragg-reference-sbnd-retune.md) §6a (the retune this
reverts, and the σ derivation), [63](63_stm-improvement-campaign.md) (the guards
whose thresholds this unsettles), [62](62_stm-baseline-and-protons.md) (the owner
truth set), [59](59_full1k-production-scan.md) (the 1000-event sample and the
production chain), [55](55_dqdx-vs-rr-three-bundles.md) §12 (the three re-fitted
bundles), [2](2_geometry-and-timing.md) / [3](3_scripts.md) (the constant tables
this syncs).
