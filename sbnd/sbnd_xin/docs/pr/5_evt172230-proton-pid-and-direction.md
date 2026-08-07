# 5 — evt 18253/1/172230: the vertex proton called "e− 29 MeV", and why the geometric vertex chose the Bragg end

Status: INVESTIGATION COMPLETE for the mechanism (owner-reported, 2026-07-30);
plus ONE display bug FIXED (§4, `SbndPrMagnifyTrackingVisitor` half-channel
truncation). No reconstruction behavior changed — the fixed run's `mabc-pr.zip`
is bit-identical to the pr/4 DL arm (`5b4e8158…`). The systematic PID/direction
fix is scoped in §6, not implemented.

Owner symptoms: (1) particle-flow `cluster = 5030` at (−47.4, −84.0, 22.7) is
labeled electron but is a proton; (2) the traditional neutrino-vertex finder
ignored this proton's clear Bragg peak. (3, added mid-investigation) the
Magnify tracking display shows the fitted track offset from the 2D measured
charge, while 3D dQ/dx looks right.

Companion docs: `pr/4` (DL vertex adoption, same event), `pr/2` §2e (uBooNE
constants worklist), doc 57 (dQ/dx constants audit), doc 48 (SBND dQ/dx
tables).

## 0. Repro block

```bash
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
cd $SX/work-nuecc48-prsmoke2
# standard DL arm (pr/4): nupr_evt172230_defaultdl/
# trace rerun used for the pin (per-logger trace; note -L debug would clobber it, §3c):
#   ... -l "$OUT/wct_trace.log:trace" -L clus.NeutrinoPattern:trace ...  -> nupr_evt172230_dl_trace2/
# post-fix verification run:                                             -> nupr_evt172230_pufix/
python3 ../../abtest/hash_archive.py nupr_evt172230_pufix/mabc-pr.zip   # 5b4e8158... (= pr/4 DL arm)
```

The dQ/dx numbers below come from `tracking-pr.root:T_rec_charge` with
`real_cluster_id==5030`: **dQ/dx = (q+1000)·10/nq** (q is the Bee-scaled
per-point charge `dQ·0.1−1000`; `nq` is the per-point dx in cm). The Bee
`track_fit` JSON has the same `q` but no dx — dividing by nothing
understates dQ/dx by the ~0.6 cm point spacing (a mistake made and corrected
during this investigation; the Magnify GUI was always right).

## 1. What the two arms produced

DL arm (`nupr_evt172230_defaultdl`, production default since pr/4):

```
id=5030  e-  29 MeV     start=(-54.7,-87.5,19.9)  end=(-46.1,-84.2,22.9)   <- the proton
id=5032  e-  1649 MeV   start=(-54.7,-87.5,19.9)  end=(-108.7,-105.4,169.1)
```

Vertex right (DL), stub direction right, **species and energy wrong**.

Geometric arm (`nupr_evt172230`): the proton is not even a separate particle —
vertex at the Bragg end, the stub absorbed as the first ~10 cm of
"e− 1686 MeV". Both symptoms are one event seen from two vertices.

## 2. The measured dQ/dx — a textbook proton at the correct absolute scale

| rr (cm) | dQ/dx (e/cm) | / SBND ProtonDeDx |
|---|---|---|
| 9.15 | 128,128 | 0.98 |
| 6.75 | 132,129 | 0.94 |
| 4.35 | 155,464 | 0.99 |
| 1.95 | 168,407 | 0.89 |
| 1.35 | 191,809 | 0.92 |
| 0.67 | 191,180 | 0.76 (Bragg-tip smearing) |

The profile IS the doc-48 SBND proton table (ratio 0.88–0.99 until the final
smeared points), rising monotonically to the Bragg peak at the end where the
geometric finder put the "vertex". Direction is in the data; the absolute
charge scale is right. (The Magnify GUI screenshot, docs/pics
`Screenshot 2026-07-30 at 10.45.44 AM.png`, shows the same 120→195k e/cm.)
Plot: `docs/pics/pr5_seg5030_dqdx_corrected.png`.

Retraction for the record: an earlier draft of this doc claimed the profile
was "0.54× the proton table" and that "the whole event reads low". That was
the missing ÷dx in the Bee-JSON conversion — wrong, withdrawn. The event's
charge calibration is consistent with the doc-48 tables.

The "29 MeV" is `cal_kine_range(9.725 cm, e)` — the range table under the
electron hypothesis (e 29.75 / µ 46.77 / **p 113.33** MeV). Energy error is
purely a consequence of the PID error.

## 3. Root cause, pinned empirically

Three independent probes agree (trace rerun `_dl_trace2`; a temporary
log-only instrumentation of the 14 pdg-11 assignment sites in
`NeutrinoTrackShowerSep.cxx`, reverted after one run; and a numerical replay
of the PID arithmetic):

### 3a. Step 1 — the track PID abstains despite a perfect proton signal

`determine_direction` trace: `Track nfits=17 len=9.75cm dirsign=0 pdg=0`.

Replaying `segment_do_track_pid` (`PRSegmentFunctions.cxx:1392`) with the
exact in-code inputs (16 non-vertex fit points, SBND tables, flat MIP 50k):

| hypothesis | forward (true dir) | backward |
|---|---|---|
| ks muon / ks flat | 0.075 / 0.049 | 0.154 / 0.049 |
| ratio muon / flat | 0.66 / 0.34 | 0.67 / 0.34 |
| **score proton** | **0.135** (ks 0.058, ratio 1.12) | 0.193 |
| score muon / electron | 0.35 / 0.61 | 0.37 / 0.61 |

The direction bit is `eval_ks_ratio(ks_mu, ks_flat, r_mu, r_flat)`
(`PRSegmentFunctions.cxx:1353` → `:957`), whose first line is
`if (ks1-ks2 >= 0) return false`. The smeared proton rise (1.5× over 9.7 cm)
matches a **flat line better than the muon-with-tip template** in KS shape
distance, forward AND backward, at both 35 and 15 cm windows → the gate can
never fire → `dirsign=0, pdg=0`. Meanwhile the proton template fits
beautifully and **is directional** (0.135 vs 0.193) — computed in
`result[2]`, never consulted for direction. Prototype parity
(`ProtoSegment.cxx:1174`): a stopping proton's direction is decided by a muon
test that structurally cannot see it. It works in uBooNE more often because
sharper Bragg contrast there lets ks_mu beat ks_flat; here it lost by 0.026.

### 3b. Step 2 — electron coercion sweeps the orphan

With `dirsign=0, pdg=0` the segment enters the electron-coercion rules. The
instrumented run pins the actual site: **`examine_all_showers`**
(`NeutrinoTrackShowerSep.cxx:~1780-1875`) — once the cluster as a whole is
judged shower-like, every remaining non-shower segment is forced to pdg 11.
The 25.6 cm trunk had already been absorbed by
`improve_maps_shower_in_track_out` (`:767`), the small stubs at `:767/:792`.
Coercion sequence (instrumented run, cluster 5):

```
@125  (determine_direction, S_topo)        x many   — real shower fragments
@767  (improve_maps_shower_in_track_out)   len=25.57 — the trunk
@767/@792                                  len=2.51/2.42/2.62 — small stubs
@1861 (examine_all_showers)                len=9.75  — THE PROTON (last holdout)
```

Adjacent proton-eating rules found on the way (all uBooNE-scaled, listed for
the fix campaign): `judge_no_dir_tracks_close_to_showers` (0.6 cm 2D
proximity — did NOT fire here, proton is 12–26 cm from every shower fragment
in all 3 views); `improve_maps_no_dir_tracks` Cases C/D (`pdg==2212` +
`dQ_dx_rms > {1.0,0.75,0.4}×43e3` + anti-parallel to shower → electron) and
Case H (`pdg==0`, `len<12cm`, `median/(43e3)>1.2`, anti-parallel → electron).
`judge_no_dir_tracks_close_to_showers` and Cases C/D/H remain untouched;
Case E (a sibling rule in the same function, muon-topology demotion, not
listed here since it wasn't implicated in this event) got a dQ/dx guard in
doc pr/40 F2 (2026-08-06) after a separate owner report traced SBND evt 388
to it.

### 3c. Step 3 — the all-showers vertex path picks an end blindly

With zero tracks left anywhere, `determine_main_vertex` sets
`flag_save_only_showers=true` (`NeutrinoVertexFinder.cxx:2422-2441`; trace
confirms) and routes to `compare_main_vertices_all_showers` (`:352-535`) —
`compare_main_vertices`' proton/track scoring never runs. That chooser: PCA
axis of the whole cluster → the two extreme candidates along the axis → rough
path (148.8 cm) → shower-direction test → `dir=-1` picks `min_vtx`; for
showers >80 cm with Δz>40 cm an override (`:518-526`) takes the smaller-z
extreme. The proton is nearly perpendicular to the shower axis — its two ends
project within ~0.4 cm of each other — so the "extreme" is a numerical coin
flip and the Bragg end won: `selected vertex (-45.54,-84.03,23.02)
sg_length=148.84cm sg_dir=-1`. Both candidate ends were on the ballot
(including the true vertex at (−54.92,−87.67,19.72)); no dQ/dx information
was consulted.

In the DL arm the SCN rerank overrides step 3 (pr/4) — vertex fixed — but
steps 1–2 persist, hence "e− 29 MeV" survives.

Trace-logistics gotcha for the record: per-logger trace needs
`-L clus.NeutrinoPattern:trace` WITHOUT a global `-L debug` — `fill_levels`
(`util/src/Logging.cxx:177-198`, applied at `apps/src/Main.cxx:346`) applies
the global level as a prefix rule to every logger AFTER the per-logger set,
clobbering it. `nupr_evt172230_dl_trace/` is the exhibit of that failure;
`_dl_trace2/` is the good one.

### 3d. Systemic findings beyond this event

1. **The `is_dir_weak()` port divergence.** The prototype vertex logic calls
   score-thresholded `ProtoSegment::is_dir_weak()`
   (`prototype_base/pid/src/ProtoSegment.cxx:1291-1302`; proton: weak if
   score>0.13 at ≥5 cm). The toolkit reads the raw `dir_weak()` member at
   every ported site (`NeutrinoVertexFinder.cxx:314, 806, 832, 902, 1202,
   613-616`, `NeutrinoPatternBase.cxx:2056`; 83 reads in 7 files). The
   faithful port `segment_is_dir_weak()` (`PRSegmentFunctions.cxx:1064`)
   exists with ZERO callers (commit 7d494879). Raw is a strict subset ⇒ the
   toolkit calls directions strong that the prototype calls weak; in mixed
   clusters that deletes vertex candidates the prototype would keep
   (`examine_main_vertex_candidate` `flag_in`). Not what decided THIS event
   (the cluster went all-showers), but a real divergence — undocumented in
   `porting_dictionary.md`, and `NeutrinoTaggerNuE.cxx:38` even records the
   substitution as if equivalent. Needs an owner decision.
2. **The uBooNE constants** (pr/2 §2e, doc 57): the direction/PID path
   normalizes by hardcoded 43e3 (~102 sites) and 50e3 (flat-MIP reference)
   while the tables under them are SBND-regenerated — every ratio cut is
   ~30%/12% displaced. Secondary for this event (the shape gate failed, not a
   ratio cut), but Case C/D/H thresholds above are all in this class.
3. `examine_direction(flag_final=true)` (`NeutrinoVertexFinder.cxx:1202`)
   rewrites ALL directions outward from the chosen vertex and sets
   `dir_weak(true)` — post-hoc `dirsign` is a consequence of the vertex
   choice, not evidence. Diagnose upstream of it.

## 4. The Magnify 2D offset — FIXED (display-only)

Symptom (owner screenshot): fitted track drawn offset from the 2D measured
charge; 3D dQ/dx fine. Quantified from `tracking-pr.root` (track channel vs
per-slice charge centroid): **median −0.5 channel, rms 0.27, identical in U,
V, W** — the signature of `floor()` on a uniform fractional coordinate.

Root cause: `PR::Fit::pu/pv/pw` are fractional wire coordinates (double,
`clus/inc/WireCellClus/PRCommon.h:122`), but my pr/3 fork
`root/src/SbndPrMagnifyTrackingVisitor.cxx:334-336` pushed them through
`ChanScheme::global(int wire)` via `static_cast<int>` — truncating the
fraction. The uBooNE original keeps it (`fit.pu + kPlaneChOffset[0]`,
`UbooneMagnifyTrackingVisitor.cxx:478`).

Fix (this commit): compute `cs.base[plane] + apa*cs.nch[plane] + fit.pu` in
double. Verification (`nupr_evt172230_pufix/`): residuals →
U −0.09 / V +0.03 / W +0.03 ch (rms ≤0.20); `mabc-pr.zip` member hash
`5b4e8158…` identical to the pr/4 DL arm ⇒ reconstruction untouched.
`T_bad_ch`/`T_proj_data` keep integer channels (real channel IDs — correct).

## 5. Scope of evidence

One event, but the mechanism is generic: any short proton at a vertex
dominated by an EM shower can lose the muon-vs-flat direction gate, be
coerced to electron by the cluster-level sweep, and (geometric arm) turn the
vertex choice into a coin flip. The 45-event both-arms expansion (pr/4 §6)
will measure the rate.

## 6. Next steps (the systematic fix, owner decisions needed)

1. **Direction gate**: let the proton template vote — e.g. accept a direction
   when `score_p` is small and clearly asymmetric (here 0.135 vs 0.193), as a
   default-OFF knob (`nu_proton_dir` or similar), validated on the 45
   nu-candidates + the uBooNE Track-A gate (off ⇒ byte-identical).
2. **`is_dir_weak()` divergence** (§3d.1): wire `segment_is_dir_weak()` into
   the 83 sites behind a knob, or document the divergence as intentional.
   Owner call (CLAUDE.md M15).
3. **MIP normalization**: thread a `mip_dqdx` knob through
   `TaggerCheckNeutrino` → `PatternAlgorithms` (the TaggerCheckSTM
   `a*m_mip_dqdx` pattern, doc 48), covering the 43e3/50e3 sites — pr/2 §2e
   worklist (ii), unblocked but large (~102 sites; doc 57 taxonomy).
4. Re-hand-scan after (1): does the geometric vertex recover this event once
   the proton keeps its identity? (The all-showers path would no longer be
   taken.)
