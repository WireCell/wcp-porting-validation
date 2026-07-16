# PDVD U/V–vs–W wire-offset calibration from real tracks (run 39324)

Measured from real signal (2026-06-07). Analysis reproduces with
`pdvd/pdvd_uvw_offset.py`; the calibration is baked into a local **v5** wire file
(`protodunevd-wires-larsoft-v5.json.bz2`) and run 39324 was re-imaged with it. **Two things
to know up front:** (1) the shift is assigned **by registration type** ({0,2,5,7} vs
{1,3,4,6}), which **respects** the cathode mirror symmetry — see *The symmetry*; (2) on the
already-consistent wide-W calibration tracks the imaging change is **modest** (blob z is
pinned by the unchanged W plane), but a *wrong* (by-CRP) shift is destructive and the by-type
fix recovers it — see *The v5 wire file*.

## Why

PDVD imaging leaves **gaps**. A blob only forms where a U-wire, a V-wire and a
W-wire all cross at one transverse (Y,Z) point — three-plane consistency in
`RayGrid`/`GridTiling`. The prior file audit
(`img/docs/protodune-wire-geometry-channel-mapping-audit.md`,
`docs/16_pdvd-wire-geometry-v3-v4.md`) showed the wire **positions** are fine and
that **v4 = v3 channel map + v5 GDML positions**; it left **one thing unprovable
from the files alone**: the induction **U/V vs collection W registration** (no
upstream test pins the U/V channel↔wire pairing). This note measures that
registration directly from charge.

The check is exactly the one suggested: **at each time tick, use the wire
geometry to test whether the U and V hits are consistent with the W hit (to
within ±½ pitch), and find the offset that makes them match.**

## Tracks used

Run 39324, event 0 (DNN-ROI SP magnify files in `pdvd/work/039324_0/`), one clean
isolated track per CRP type:

| CRP | anode | face | U chan | V chan | W chan | ticks |
|---|---|---|---|---|---|---|
| bottom | 0 | 1 | 75–125 | 1010–1090 | 2020–2070 | 0–5000 |
| top    | 4 | 8 | 6530–6610 | 7460–7550 | 8480–8630 | 0–4500 |

W is collection (un-wrapped) so the W window fixes the face unambiguously; the
quoted U/V windows are the face-matching induction segments. Each track is a
single clean diagonal in all three planes (`pics/track-anode{0,4}.png`).

## Method

Per tick (charge from the `gauss` SP histograms `h{u,v,w}_gauss{anode}`), keeping
the **wire geometry fixed**:

1. Charge-weighted centroid channel for U, V, W (ticks with all three planes lit
   and a narrow, single-track profile; ~3500/event survive).
2. Map the U and V centroid channels to their **wire lines in (Y,Z)** and
   intersect them → crossing point `P = (y, z_cross)`.
3. Read the measured **W wire z**, `z_W`, for the W centroid channel (W wires are
   vertical → constant z).
4. **Mismatch** `ΔZ_W(t) = z_cross − z_W` — the distance, along the W pitch, by
   which the W plane would have to move to sit under the charge that U and V see.
   Consistent ⇔ `|ΔZ_W| < ½·W-pitch = 2.55 mm`.

The required **W-plane shift** (U,V untouched) is `median ΔZ_W`. Because the
induction planes are mirror-symmetric (`pdir_U`,`pdir_V` share their wire-direction
component and oppose along W-pitch), the predicted W depends only on `pU+pV`, so the
same correction can equivalently be made on the induction side as **one symmetric
U/V strip offset** `dp = ½·ΔZ_W` (a pitch shift applied equally to U and V; in
*channel* index it is opposite-signed on U vs V, since their numbering runs in
opposite pitch senses). Pitch: U=V=7.65 mm, W=5.10 mm.

## Result

Both CRPs are **grossly inconsistent today**: with the shipped geometry, **0–1 %
of ticks fall within ±½ W-pitch** — i.e. essentially every point of the track
fails three-plane closure → gaps. A single rigid shift recovers **83–90 %**.

| CRP / anode | geom | required W-plane shift ΔZ_W | in W-pitch | consistent ±½pitch (before→after) | equiv. symmetric U/V offset |
|---|---|---|---|---|---|
| bottom / 0 | **v4** | **−13.2 mm** | −2.59 | 0 % → 90 % | −6.6 mm (−0.86 strip) |
| bottom / 0 | v3 | −12.7 mm | −2.48 | 0 % → 90 % | −6.3 mm (−0.83 strip) |
| top / 4 | **v4** | **+9.8 mm** | +1.92 | 1 % → 83 % | +4.9 mm (+0.64 strip) |
| top / 4 | v3 | +9.3 mm | +1.81 | 1 % → 83 % | +4.6 mm (+0.60 strip) |

per-tick scatter (RMS) of ΔZ_W is 1.9 mm (bottom) / 2.2 mm (top) — **below** the
2.55 mm half-pitch, which is why centering on the median lands most ticks inside
tolerance. ΔZ_W is **flat in tick** (`pics/dzW-anode{0,4}.png`) even
though the track's local W-slope changes ~2× along its length — so this is a
**geometric** offset, **not** an induction-vs-collection timing skew (a timing
skew would scale with the local slope and would not be flat).

![bottom CRP per-tick W-plane mismatch](../pics/dzW-anode0.png)
*Bottom CRP (anode 0): per-tick ΔZ_W sits at −12 to −14 mm, far outside the green
±½ W-pitch consistency band; v3 and v4 nearly coincide.*

![top CRP per-tick W-plane mismatch](../pics/dzW-anode4.png)
*Top CRP (anode 4): ΔZ_W ≈ +9–10 mm, opposite sign.*

Equivalent **channel-index** statement of the symmetric U/V offset (v4):
bottom CRP **U −0.86 / V +0.86 channels**, top CRP **U −0.64 / V +0.64 channels**
— in both CRPs U decreases and V increases, opposite signs because the two
induction planes' channel numbering runs in opposite pitch directions. (Beware: the
channel-numbering *direction* also flips between top and bottom, so the same channel
sign does **not** mean the same physical shift — see *The symmetry*.)

### Recommended offset

Apply on top of **v4** (the better-positioned base; v3 differs only ~0.5 mm here),
**by registration type** (see *The symmetry*). v5 keeps W fixed and shifts U & V along z:

- **Type A (anodes 0, 2, 5, 7):** shift U & V by **+13.2 mm** along z (W-pitch direction)
  — equivalently move W by **ΔZ_W ≈ −13 mm**, or slide U & V by **dp ≈ −6.6 mm** along
  induction pitch (U −0.86 ch / V +0.86 ch). Measured on anode 0.
- **Type B (anodes 1, 3, 4, 6):** shift U & V by **−9.8 mm** along z — equivalently move W
  by **ΔZ_W ≈ +10 mm**, or slide U & V by **dp ≈ +4.9 mm** (U −0.64 ch / V +0.64 ch).
  Measured on anode 4.

The two measured anodes (0 and 4) need different shifts — but they are **different
registration *types***, not a top-vs-bottom difference. The first-wire-offset table in
`16_pdvd-wire-geometry-v3-v4.md` §1 groups the 8 anodes into two types, each appearing once
per CRP: **type A = {0, 2, 5, 7}**, **type B = {1, 3, 4, 6}**. Anode 0 is type A; anode 4
is type B — so their different corrections track the **type**, not the CRP. The shift for
each type's mirror partner in the *other* CRP follows from the cathode symmetry (next
section). The values are also **fractional**, not an integer number of wires — a
*continuous* sub-pitch U/V-vs-W registration error, not a clean off-by-N-wires bug.

## The symmetry — respected by assigning the shift by *type*

Earlier drafts compared anode 0 (+13.2 mm) directly to anode 4 (−9.8 mm), called them
"opposite sign", and concluded the correction *breaks* the cathode symmetry. **That was a
mistake: anode 0 and anode 4 are different registration *types*, not mirror partners**, so
their differing shifts say nothing about the symmetry.

The first-wire-offset table (`16_pdvd-wire-geometry-v3-v4.md` §1) groups the anodes by type,
and the cathode mirror pairs **same-type** anodes across the two CRPs (top = bottom with
U↔V swapped, z preserved):

| type | first-wire character | anodes (mirror partners) | measured on | U,V z-shift |
|---|---|---|---|---|
| **A** | small (~1.0–1.8 mm) | bottom {0,2} ↔ top {5,7} | anode 0 | **+13.2 mm** |
| **B** | large (3.5 / 11.7 mm) | bottom {1,3} ↔ top {4,6} | anode 4 | **−9.8 mm** |

**The mirror demands the same shift on partners — and the by-type assignment gives it.**
The correction is a *common-mode* shift (U and V move together, equally), and the cathode
mirror is `x→−x` with **z preserved** and U↔V swapped. A common-mode z-shift is invariant
under the U↔V swap, and z is preserved, so a type's two mirror-partner anodes must take the
**same signed z-shift**. Assigning **type A {0,2,5,7} = +13.2 mm** and
**type B {1,3,4,6} = −9.8 mm** therefore *respects* the symmetry by construction, rather
than breaking it. (The opposite *sign* between type A and type B is a genuine
type-dependent registration difference, fully allowed — it is not a symmetry violation.)

**Empirical corroboration.** Only anodes 0 (type A) and 4 (type B) were measured from
tracks; {1,3,5,7} are *deduced* by the symmetry. An earlier v5 that (wrongly) shifted
**by CRP** mis-corrected exactly those four — e.g. anode 1 (type B, true −9.8) got +13.2,
i.e. ΔZ_W ≈ +9.8+13.2 ≈ **+23 mm** (4.5 W-pitch, catastrophic) and its imaged blobs
collapsed 1917 → 608. The by-type v5 sends those four to ΔZ_W ≈ 0 and they **recover to
~v4 levels** (anode 1 608 → 1994; see the imaging table below) — converting the deduction
from "assumed" to "assumed **and** confirmed by imaging".

## The v5 wire file — built, validated, and imaged

The correction is now baked into a local **`protodunevd-wires-larsoft-v5.json.bz2`**
(in `wire-cell-data/`, kept local like v4 — not pushed to `WireCell/wire-cell-data`).

**Convention chosen: keep W fixed, move U & V along the W pitch direction.** W is the
collection plane and its absolute z is the trusted reference; shifting U/V (rather than
W) lands the blobs at the *collection* z. A W-shift gives identical *internal*
U∩V-vs-W consistency but would rigidly translate every blob by ΔZ_W — so U/V-shift is
the physically correct choice for absolute placement.

`pdvd/make_v5_uvwcal.py`: from v4, keep the W plane (ident 2) untouched and shift each
**U and V (ident 0,1) wire endpoint purely in z** (= the W pitch direction, since the
vertical collection wires have pitch along z) by `−ΔZ_W`, assigned **by registration
type** (so the cathode symmetry is respected — see *The symmetry* above):

| type | anodes | ΔZ_W (mismatch) | **U,V z-shift applied** | source |
|---|---|---|---|---|
| **A** | 0, 2, 5, 7 | −13.2 mm | **+13.2 mm** | measured anode 0; {2,5,7} by symmetry |
| **B** | 1, 3, 4, 6 | +9.8 mm | **−9.8 mm** | measured anode 4; {1,3,6} by symmetry |

> **Magnitude note.** This is the **full ΔZ_W** in pure z. A rigid z-translation `t` of
> *both* U and V lines moves the U∩V crossing by exactly `t`, so nulling
> `z_cross − z_W = ΔZ_W` needs `−ΔZ_W`. Do **not** confuse it with the `+3.3 / −2.45 mm`
> *induction-pitch* numbers quoted in the last section — those are the z-*component* of an
> induction-pitch slide (a different motion, with a y-component too); the crossing still
> moves the full ΔZ_W via `ΔZ_W = 2·dp`. "Along the **W** pitch direction" = pure z = the
> full ΔZ_W.

**Validated three independent ways** (`pdvd/validate_v5.py`):

1. **Geometry file diff v4→v5** — W tail/head points byte-identical (max |Δ|=0); U,V
   points moved *only* in z (max |Δx|,|Δy|=0), by exactly +13.2 / −9.8 mm (to 1e-13 mm).
2. **Offset remeasure on the real tracks** — v5 gives **median ΔZ_W = −0.01 / −0.00 mm**
   and ±½-pitch consistency **0.2 % → 89.9 %** (bottom) / **0.8 % → 83.1 %** (top).
3. **W-shift vs U/V-shift equivalence** — v5 (U/V-shift) and the W-shift file
   (`make_v4_uvwcal.py`) give **identical** median ΔZ_W and consistency, confirming the
   two framings are the same correction seen from either side.

**Imaging (the physics validation).** Run 39324 event 0 (art event 339850) was
**re-imaged and re-clustered with v5**, all 8 anodes (production config temporarily
pointed at v5, then reverted to v3). Per-anode Bee links (bee idx 0–7 = anode 0–7):

| geometry | Bee link |
|---|---|
| **v4 baseline** (positions correct, U/V-vs-W offset remains) | <https://www.phy.bnl.gov/twister/bee/set/fd21cf88-9936-4c38-8803-9b050ed63a2f/event/list/> |
| **v5** (by-type U/V-vs-W correction) | <https://www.phy.bnl.gov/twister/bee/set/251465d9-d2b5-434b-8852-68e774518324/event/list/> |

**Per-anode imaged-blob counts** (`pdvd/check_gap_closure.py` for coverage; counts below).
The middle column is the *earlier, wrong* by-CRP v5 — shown only to demonstrate that the
four symmetry-deduced anodes recover under the by-type fix:

| anode | type | v4 baseline | by-CRP v5 (wrong) | **by-type v5** |
|---|---|---|---|---|
| 0 | A | 2678 | 2693 | 2693 |
| 1 | B | 1917 | **608** | **1994** ✓ |
| 2 | A | 3376 | 3290 | 3290 |
| 3 | B | 1663 | 1312 | **1742** ✓ |
| 4 | B | 13572 | 13657 | 13657 |
| 5 | A | 4787 | 4693 | 4716 ✓ |
| 6 | B | 3187 | 3226 | 3226 |
| 7 | A | 5653 | 4744 | **5749** ✓ |
| **total** | | 36 833 | 34 223 | **37 067** |

Measured anodes 0,2,4,6 are unchanged (same shift in both v5s); the four *deduced* anodes
1,3,5,7 recover to ~v4 levels — the empirical confirmation of the symmetry deduction.

**What the imaging change looks like on the measured tracks (`check_z_residual.py`).** On
the *correctly-corrected* wide-W calibration tracks (anodes 0, 4) the change is **modest**:
drift-slice coverage in a 4 cm tube is unchanged (anode 0 97 %→98 %, anode 4 67 %→67 %) and
blob z moves only ~1.6 mm — because **blob z is pinned by the W collection plane** (vertical
wires measure z), which v5 leaves untouched. Shifting U/V along z fixes *which* W charge is
three-plane-consistent and *whether* a blob forms where the W ROI is narrow, but does **not**
translate the track in z. So a *correct* shift is gentle on already-consistent wide-W tracks,
whereas a *wrong* shift is destructive (the by-CRP anode 1, ΔZ_W ≈ +23 mm → 608 blobs).

**Bottom line:** v5 is the geometrically-correct, symmetry-respecting U/V-vs-W registration
(validated three ways + the recovery above). Its imaging effect is **track-topology-
dependent** — largest where the W ROI is narrow, small on wide-W tracks like anodes 0/4.
**Compare the v4-baseline and v5 Bee links on your own events** to judge the gaps you see.

### Equivalent W-shift recipe (alternative)

Instead of moving U/V, shift each CRP's W plane in z by ΔZ_W (`pdvd/make_v4_uvwcal.py`,
DZ = −13.2 bottom / +9.8 top). Same *internal* consistency (verified identical above),
but it displaces all blobs by ΔZ_W in z — use only if W's absolute z is not trusted.

## Can U and V take *separate* shifts? (separability study)

The v5 correction is **common-mode** — U and V move together by the same `−ΔZ_W`. A
natural follow-up: do U and V each need a *different* z-shift (still along the W-pitch /
z direction, but unequal)? Geometrically there are two degrees of freedom, but **these
data constrain only one of them — the sum `dzU+dzV`.** The U-vs-V *difference* is
unmeasurable from three-plane consistency. This was checked nine independent ways
(2026-06-08); all agree. Plots in `pics/`.

**The geometry — why only the sum is observable.** The only handle is three-plane
closure, and W is the vertical collection plane, so it pins only **z**. U and V are exact
mirror images about that vertical, so — measured directly from the v4 wire `pdir` vectors —

> `∂z_cross/∂dzU = ∂z_cross/∂dzV = 0.5000` *identically* (both anodes).

A pure-z shift of U and a pure-z shift of V move the U∩V crossing's z by the **same**
amount, so closure depends on `(dzU+dzV)` only. The difference `dzU−dzV` slides the
crossing purely in **y** (`Δy = (dzV−dzU)/(2·tanθ)`, the *same* constant for every point) —
i.e. it is a **rigid y-translation of the entire reconstructed image**, which no
self-contained U/V/W test can detect (the same reason a global z-shift is invisible). It
is a gauge symmetry, not a weak signal.

**What was tried — all land on "only the sum":**

| # | test | result | plot |
|---|---|---|---|
| 1 | single-track 2-D (dzU,dzV) scan (consistency + RMS) | optimum is a straight diagonal `dzU+dzV=const`, never a closed peak | `pdvd_uvw_2dscan_anode{0,4}.png` |
| 2 | coordinate descent (the "wiggle adds info" idea) | returns the symmetric point from every seed, incl. asymmetric ones | — |
| 3 | whole-plane multi-track triplets (pair every U·V, keep if a W peak is within ~2.5 pitch; 832 / 3886 triples) | same straight diagonal valley | `pdvd_wholeplane_anode{0,4}.png` |
| 4 | charge-weighted peak matching, fine ±3 mm window | score **flat to 0.000 %** along the difference at fixed sum | `pdvd_chargematch_anode{0,4}.png` |
| 5 | V-only fine fit (fix U & W, optimize V) | extra V = **+0.00 / +0.10 mm**; the mirror U-only fit gives the *identical* number → the split is a free gauge choice, not a measurement | — |
| 6 | all-tracks fine-tune of the sum | median agrees with the single track but is half-pitch-comb contaminated → no clean improvement | `pdvd_wholeplane_finetune.png`, `pdvd_cleanmultitrack.png` |
| 7 | V-residual diagnostic, single track (predict V from U∩W, compare to measured V) | centred on **0.00 mm**, RMS ~2 mm, 92–96 % within ½ V-pitch → **V has no room** | `pdvd_v_finetune_residual.png` |
| 8 | V-residual, all tracks (correct per-face mapping) | dominant central spike on **0** (confirms) + discrete half-W-pitch comb = integer-wire assignment artifact, *not* a continuous offset | `pdvd_v_finetune_residual_alltracks.png` |

The whole-plane attempts (#3, #6, #8) only ever sharpen or confirm the **sum**; pooling more
tracks adds combinatorial / integer-wire / wire-wrapping scatter (a comb at half-W-pitch),
never difference information. The clean single calibration track per type remains the most
reliable estimate. The full anode-0 plane (`anode0_fullplane.png`) shows the ~6 multi-angle
tracks used.

**Conclusion.** The common-mode v5 already captures everything these tracks can measure
(the sum, ≈ +13.2 / −9.8 mm per plane). A genuine U≠V differential may well exist in the
true geometry, but measuring it requires an **external y-reference** — MC-truth y for a
track, a track that starts/ends at a known-y boundary (CRP frame, cathode), or a LArSoft
3-D cross-check — because the difference lives entirely in the reconstructed absolute *y*,
which nothing internal to the U/V/W charge localises. **No separate U/V shift is applied;
v5 stays common-mode.** Scripts: `pdvd_uvw_2dscan.py`, `pdvd_uvw_wholeplane_scan.py`,
`pdvd_uvw_chargematch_scan.py`, `pdvd_uvw_vonly_scan.py`, `pdvd_uvw_wholeplane_finetune.py`,
`pdvd_uvw_cleanmultitrack.py`, `pdvd_uvw_vresidual.py`, `pdvd_uvw_vresidual_alltracks.py`.

## v3 vs v4

For *this* registration metric v3 and v4 are within ~0.5 mm (the v4 reposition
moves W by ~−5.5 mm on the bottom CRP but moves U+V by almost the same amount, so
the U/V-vs-W *combination* barely changes). v4 is the better geometry overall —
its wire **positions** come from the v5 GDML — but that improvement is in the
absolute placement this internal U∩V-vs-W test is largely blind to; **the ~2–2.6
W-pitch U/V-vs-W offset survives into v4 unchanged** and is exactly the residual
calibrated here. In other words: v4 fixes the positions, this fixes the U/V↔W
registration; they are independent and v4 still needs this.

## Shifting U and V instead of W — sync with the first-wire-offset doc

`16_pdvd-wire-geometry-v3-v4.md` describes the U/V-vs-W registration as the **first-wire
offset along +Z** (`U[0]−W[0]`, `V[0]−W[0]`). This note used a **W shift**. They are
the same correction seen from the two sides — applying *either* is verified to give
the same 90 % / 83 % consistency. The dictionary (v4 base):

Columns are the two registration *types* (the value carries to each type's mirror
partner in the other CRP, §*The symmetry*):

| apply on | parameter | type A (0,2,5,7) / meas. anode 0 | type B (1,3,4,6) / meas. anode 4 |
|---|---|---|---|
| **W** plane (keep U,V) | ΔZ_W along z | −13.2 mm | +9.8 mm |
| **U & V** planes, **along W pitch (z)** — *this is what v5 does* | −ΔZ_W | **+13.2 mm** | **−9.8 mm** |
| **U & V** planes (keep W) | dp along induction pitch | −6.6 mm (−0.86 strip) | +4.9 mm (+0.64 strip) |
| ⤷ same, as **+Z move of U[0],V[0]** (z-component of the pitch slide) | ΔZ_U = ΔZ_V | +3.3 mm | −2.45 mm |
| ⤷ same, as **channel** index | U / V | −0.86 / +0.86 ch | −0.64 / +0.64 ch |

Geometry-fixed relations for the *induction-pitch* slide: **ΔZ_W = 2·dp** and its z-component
**ΔZ_U = ΔZ_V = −dp/2 = −ΔZ_W/4** (the induction pitch makes 60° with the W pitch, so only the
cos 60° = ½ z-projection counts toward W). **v5 does not use this slide** — it uses the pure-z
shift (row 2, the full −ΔZ_W). The induction-pitch row is the equivalent alternative.

### Does this keep the bottom-{0,2} ↔ top-{5,7} symmetry? (Q1)

**Yes — when the shift is assigned by *type*, which is what v5 does.** Bottom {0,2} and top
{5,7} are the two halves of **type A** and are cathode-mirror partners (identical first-wire
offsets with U↔V swapped). v5 gives both the **same +13.2 mm** common-mode U/V z-shift; since
the mirror preserves z and a common-mode shift is U↔V-swap-invariant, the corrected geometry
is **still mirror-symmetric**. (Earlier this subsection answered "No" by comparing anode 0 to
anode 4 — *different types*, not partners. That was the mistake corrected in §*The symmetry*.)

One thing that *is* clean: my correction is a **common-mode** U=V shift (U and V move
the same way), which is a *different component* from the U≠V *differential* first-wire
offsets the other doc tabulates — those are a fixed geometric feature of the wrapping;
this rides on top of them.

## Caveats / next steps

- **One track per *type*.** Each type's shift is measured on a single track (anode 0 for
  type A, anode 4 for type B); the other three anodes of each type are set by the cathode
  symmetry and **corroborated** by the blob-count recovery (not independently fit). More
  tracks per type, the other face, and per-anode fine-tuning remain worthwhile.
- Only the `pU+pV` combination is constrained by W, so neither a single track *nor*
  all the tracks in the plane can separate an asymmetric U-only vs V-only component from
  the symmetric one; the symmetric (common-mode U=V) form is assumed. This is now
  established nine ways — see §*Can U and V take separate shifts?* — and is a gauge
  symmetry (the difference is a rigid y-translation), breakable only with an external
  y-reference.
- **Applied & imaged.** v5 is built (U/V shifted along z by type, W fixed) and run 39324
  evt 0 re-imaged with it (links + recovery table above). The two type magnitudes (+13.2 /
  −9.8 mm) come from one track each; a multi-track fit per type could refine them.

## Files

- `pdvd/pdvd_uvw_offset.py` — the analysis (parameterized over anode/geometry).
- `pdvd/make_v5_uvwcal.py` — **build the v5 file**: keep W, shift U/V along z (canonical).
- `pdvd/validate_v5.py` — three-way validation (file diff / offset remeasure / equivalence).
- `pdvd/make_v4_uvwcal.py` — equivalent W-shift recipe (alternative).
- `pdvd/build_peranode_bee_upload.sh` — per-anode Bee build+upload (used for both links).
- **Separability study** (§*Can U and V take separate shifts?*) — analyses:
  `pdvd/pdvd_uvw_2dscan.py` (single-track 2-D scan + coordinate descent),
  `pdvd/pdvd_uvw_wholeplane_scan.py` (whole-plane multi-track triplets),
  `pdvd/pdvd_uvw_chargematch_scan.py` (charge-weighted fine scan),
  `pdvd/pdvd_uvw_vonly_scan.py` (V-only fine fit),
  `pdvd/pdvd_uvw_wholeplane_finetune.py` + `pdvd/pdvd_uvw_cleanmultitrack.py` (all-tracks sum fine-tune),
  `pdvd/pdvd_uvw_vresidual.py` + `pdvd/pdvd_uvw_vresidual_alltracks.py` (V-residual diagnostic).
- `pdvd/pics/` — separability plots: `pdvd_uvw_2dscan_anode{0,4}.png`,
  `pdvd_wholeplane_anode{0,4}.png`, `pdvd_chargematch_anode{0,4}.png`,
  `pdvd_v_finetune_residual.png`, `pdvd_v_finetune_residual_alltracks.png`,
  `pdvd_wholeplane_finetune.png`, `pdvd_cleanmultitrack.png`, `anode0_fullplane.png`.
- `pdvd/docs/pics/` — `track-*`, `dzW-*` (per-tick mismatch +
  distribution), `yz-*` (U∩V crossings) for anode 0 and anode 4.
- `wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2` — the corrected file (local,
  regenerate with `make_v5_uvwcal.py`; not committed to `WireCell/wire-cell-data`).
