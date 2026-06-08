# PDVD U/V–vs–W wire-offset calibration from real tracks (run 39324)

Measured from real signal (2026-06-07). Analysis reproduces with
`pdvd/pdvd_uvw_offset.py`; the calibration is now baked into a local **v5** wire file
(`protodunevd-wires-larsoft-v5.json.bz2`) and confirmed by re-imaging — see
**The v5 wire file** below.

## Why

PDVD imaging leaves **gaps**. A blob only forms where a U-wire, a V-wire and a
W-wire all cross at one transverse (Y,Z) point — three-plane consistency in
`RayGrid`/`GridTiling`. The prior file audit
(`img/docs/protodune-wire-geometry-channel-mapping-audit.md`,
`docs/pdvd-wire-geometry-v3-v4.md`) showed the wire **positions** are fine and
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
single clean diagonal in all three planes (`wire-offset-figs/track-anode{0,4}.png`).

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
tolerance. ΔZ_W is **flat in tick** (`wire-offset-figs/dzW-anode{0,4}.png`) even
though the track's local W-slope changes ~2× along its length — so this is a
**geometric** offset, **not** an induction-vs-collection timing skew (a timing
skew would scale with the local slope and would not be flat).

![bottom CRP per-tick W-plane mismatch](wire-offset-figs/dzW-anode0.png)
*Bottom CRP (anode 0): per-tick ΔZ_W sits at −12 to −14 mm, far outside the green
±½ W-pitch consistency band; v3 and v4 nearly coincide.*

![top CRP per-tick W-plane mismatch](wire-offset-figs/dzW-anode4.png)
*Top CRP (anode 4): ΔZ_W ≈ +9–10 mm, opposite sign.*

Equivalent **channel-index** statement of the symmetric U/V offset (v4):
bottom CRP **U −0.86 / V +0.86 channels**, top CRP **U −0.64 / V +0.64 channels**
— in both CRPs U decreases and V increases, opposite signs because the two
induction planes' channel numbering runs in opposite pitch directions. (Beware: the
channel-numbering *direction* also flips between top and bottom, so the same channel
sign does **not** mean the same physical shift — see *Top vs bottom consistency*.)

### Recommended offset

Apply on top of **v4** (the better-positioned base; v3 differs only ~0.5 mm here):

- **Bottom CRP (anodes 0–3):** move the W collection wires by **ΔZ_W ≈ −13 mm**
  (≈ −2.6 W-pitch) along z — or, keeping W fixed, shift the U & V strips
  symmetrically by **dp ≈ −6.6 mm** (≈ −0.86 induction pitch; U −0.86 ch / V +0.86 ch).
- **Top CRP (anodes 4–7):** move the W wires by **ΔZ_W ≈ +10 mm** (≈ +1.9 W-pitch)
  — or shift U & V by **dp ≈ +4.9 mm** (≈ +0.64 induction pitch; U −0.64 ch / V +0.64 ch).

The offset is **different in sign and magnitude between top and bottom CRP**, so it
is *not* one global numbering convention; it tracks the CRP. The values are also
**fractional**, not an integer number of wires — so this is a *continuous*
U/V-vs-W transverse registration error (consistent with a relative plane
mis-placement / sub-pitch channel-map offset), not a clean off-by-N-wires bug.

## Top vs bottom consistency — and the symmetry question

The two CRPs need shifts that are **opposite in sign and unequal in magnitude**:

| CRP / anode | W shift ΔZ_W | +Z move of U[0],V[0] | in W-pitch |
|---|---|---|---|
| bottom / 0 | −13.2 mm | **+3.3 mm** | −2.6 |
| top / 4 | +9.8 mm | **−2.45 mm** | +1.9 |

**Applying these as-measured BREAKS the bottom-{0,2} ↔ top-{5,7} first-wire symmetry.**
The nominal geometry — and the `U[0]−W[0]`/`V[0]−W[0]` table in
`pdvd-wire-geometry-v3-v4.md` §1 — is **mirror-symmetric about the cathode**: bottom
{0,2} and top {5,7} carry the *same* value set with U↔V swapped, **same (positive)
sign**. The cathode mirror is `x→−x` only, so **z is preserved**; preserving that
symmetry therefore requires the **same +Z correction on both CRPs** (same sign *and*
magnitude). The measured corrections are **opposite sign** (+3.3 vs −2.45 mm) and ~30 %
apart, so bottom-{0,2} and top-{5,7} would no longer be mirror images.

**What the opposite sign means.** A genuinely mirror-symmetric cause — a wire-position
error, or a channel-map offset wired identically in both CRPs — would give the **same**
sign z-shift on top and bottom. An **opposite-sign** shift instead tracks something that
**flips between the two CRPs**, most naturally the **drift direction** (bottom drifts
+x, top −x). So the residual looks more like a CRP/drift-oriented z-offset than a
mirror-symmetric geometry/channel error. (It is still flat in tick — not a track-slope
timing skew.)

**Caveats that could inflate the asymmetry:** one track per CRP, measured on
*different* faces (anode 0 back face vs anode 4 front face), at different track angles.
The clean test is several tracks per CRP on both faces: if the true correction is
symmetric, more statistics should pull |bottom| and |top| together; if the opposite-sign
asymmetry persists, it is a real CRP/drift effect to model — **not** a simple symmetric
wire re-registration. **So: not yet consistent; the asymmetry is the main open question.**

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
vertical collection wires have pitch along z) by `−ΔZ_W`:

| CRP | anodes | ΔZ_W (mismatch) | **U,V z-shift applied** |
|---|---|---|---|
| bottom | 0–3 | −13.2 mm | **+13.2 mm** |
| top | 4–7 | +9.8 mm | **−9.8 mm** |

> **Magnitude note.** This is the **full ΔZ_W** in pure z. A rigid z-translation `t` of
> *both* U and V lines moves the U∩V crossing by exactly `t`, so nulling
> `z_cross − z_W = ΔZ_W` needs `−ΔZ_W`. Do **not** confuse this with the `+3.3 / −2.45 mm`
> in the table above — those are the z-*component* of an *induction-pitch* slide (a
> different motion, with a y-component too); the crossing still moves the full ΔZ_W via
> `ΔZ_W = 2·dp`. "Along the **W** pitch direction" = pure z = the full ΔZ_W.

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
| **v4 baseline** (positions correct, U/V-vs-W gap remains) | <https://www.phy.bnl.gov/twister/bee/set/fd21cf88-9936-4c38-8803-9b050ed63a2f/event/list/> |
| **v5** (U/V-vs-W corrected) | <https://www.phy.bnl.gov/twister/bee/set/0150ea98-9d26-4c23-bacd-37c26a98187d/event/list/> |

Compare the two links anode-by-anode: the v5 imaging closes the U/V-vs-W gaps that the
per-tick consistency above quantifies (0–1 % → 83–90 % of track ticks now satisfy
three-plane closure). Total imaged blobs are comparable (v4 36 833 / v5 34 223) — gap
closure re-forms blobs at the correct W rather than simply adding them.

### Equivalent W-shift recipe (alternative)

Instead of moving U/V, shift each CRP's W plane in z by ΔZ_W (`pdvd/make_v4_uvwcal.py`,
DZ = −13.2 bottom / +9.8 top). Same *internal* consistency (verified identical above),
but it displaces all blobs by ΔZ_W in z — use only if W's absolute z is not trusted.

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

`pdvd-wire-geometry-v3-v4.md` describes the U/V-vs-W registration as the **first-wire
offset along +Z** (`U[0]−W[0]`, `V[0]−W[0]`). This note used a **W shift**. They are
the same correction seen from the two sides — applying *either* is verified to give
the same 90 % / 83 % consistency. The dictionary (v4 base):

| apply on | parameter | bottom / anode 0 | top / anode 4 |
|---|---|---|---|
| **W** plane (keep U,V) | ΔZ_W along z | −13.2 mm | +9.8 mm |
| **U & V** planes (keep W) | dp along induction pitch | −6.6 mm (−0.86 strip) | +4.9 mm (+0.64 strip) |
| ⤷ same, as **+Z move of U[0],V[0]** | ΔZ_U = ΔZ_V | **+3.3 mm** | **−2.45 mm** |
| ⤷ same, as **channel** index | U / V | −0.86 / +0.86 ch | −0.64 / +0.64 ch |

Geometry-fixed relations: **ΔZ_W = 2·dp** and the **+Z move of the first U/V wires is
ΔZ_U = ΔZ_V = −dp/2 = −ΔZ_W/4** (the induction pitch makes 60° with the W pitch, so
only the cos 60° = ½ z-projection of an induction-pitch shift counts toward W; hence
the factor between `dp` and `ΔZ_{U,V}`).

**So, to shift the first U and V wires:** move the **whole U and V planes toward +Z by
3.3 mm on the bottom CRP, and toward −Z by 2.45 mm on the top CRP** (U and V move
together, by the *same* amount). Equivalently slide them −0.86 / +0.64 induction strip
along pitch. (The absolute `U[0]−W[0]` value in the other doc depends on its wire-0 /
front-back-face / +Z-sign labeling; the **change** quoted here is convention-free.)

### Does this keep the bottom-{0,2} ↔ top-{5,7} symmetry? (Q1)

**No — as measured it does not.** Bottom {0,2} and top {5,7} have identical first-wire
offsets with U↔V swapped (the other doc's table). Adding a common +Z shift keeps that
symmetry **only if the shift is the same on both CRPs**. The measured shifts are +3.3 mm
(bottom) and −2.45 mm (top) — opposite sign and unequal — so the symmetry is broken.
The full discussion (why opposite sign points to a drift-oriented cause, and what to
measure next) is in **Top vs bottom consistency** above.

One thing that *is* clean: my correction is a **common-mode** U=V shift (U and V move
the same way), which is a *different component* from the U≠V *differential* first-wire
offsets the other doc tabulates — those are a fixed geometric feature of the wrapping;
this rides on top of them.

## Caveats / next steps

- **One track, one event, one face per CRP.** The offset is robust *within* each
  track (flat ΔZ_W, v3≈v4) but should be confirmed on more tracks, the other face
  of each anode, and anodes 1–3 / 5–7 before baking it into a wire file.
- Only the `pU+pV` combination is constrained by W, so a *single* track cannot
  separate an asymmetric U-only vs V-only component from the symmetric one; the
  symmetric form is assumed (and is what "the same offset for U and V" means).
- **Applied & imaged.** v5 is built (U/V shifted along z, W fixed) and run 39324 evt 0
  re-imaged with it (links above). The shift baked in is the **single-track** value per
  CRP applied to all four anodes of that CRP — so per-anode/per-face fine-tuning, and the
  opposite-sign bottom/top asymmetry, remain open (see *Top vs bottom consistency*).

## Files

- `pdvd/pdvd_uvw_offset.py` — the analysis (parameterized over anode/geometry).
- `pdvd/make_v5_uvwcal.py` — **build the v5 file**: keep W, shift U/V along z (canonical).
- `pdvd/validate_v5.py` — three-way validation (file diff / offset remeasure / equivalence).
- `pdvd/make_v4_uvwcal.py` — equivalent W-shift recipe (alternative).
- `pdvd/build_peranode_bee_upload.sh` — per-anode Bee build+upload (used for both links).
- `pdvd/docs/wire-offset-figs/` — `track-*`, `dzW-*` (per-tick mismatch +
  distribution), `yz-*` (U∩V crossings) for anode 0 and anode 4.
- `wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2` — the corrected file (local,
  regenerate with `make_v5_uvwcal.py`; not committed to `WireCell/wire-cell-data`).
