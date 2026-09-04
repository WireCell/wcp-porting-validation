# PDVD doc 35 — the CRP gaps, measured; and the tagger fiducial made consistent with clustering

**Status (2026-09-04).** Three owner steers, executed:

1. **The 15 cm y/z inset in the clustering FV is space charge.** Recorded (§1);
   doc 33 §3.2 attributed it only to `JudgeSeparateDec_2` parity, which was
   incomplete. SBND's space charge is much smaller, which is why its inset is
   1 cm and why the difference is not a defect on either side.
2. **The cosmic-tagger FV is made consistent with the clustering FV**:
   x 2.5, y 15 + 2.5, z 15 + 3. Implemented, and measured over five 120-event
   PR arms against a repeat-arm noise band (§5).
3. **PDVD gets a composite structure-exclusion fiducial, measured from data.**
   Built (§4) on the CRP/CRU geometry (§2) at the widths §3 measures.

Config changes this doc ships:
`cfg/pgrapher/experiment/protodunevd/pr.jsonnet` (new `tgm_fv_zmin_margin`
knob, **compiled-config diff-to-zero at its default**), the new
`cfg/pgrapher/experiment/protodunevd/crp_gap_fiducial.jsonnet` (imported by
nothing yet — inert by construction), and the PDVD driver's margin flip, which
is **not** byte-identical and whose evidence is §5.

---

## 0. Repro

```bash
WCPI=/home/xqian/toolkit-dev/wcp-porting-img
cd $WCPI/pdvd

# (1) the gap measurement -- read-only, the 120-event d28dlfp arm already on disk
python3 docs/nf_sp_img_clus/scripts/fv_gap_measure.py work/*_d28dlfp \
    --sbnd ../../toolkit/sbnd_xin/work-ncpi0-doc25_r2post5/pr_evt* \
    --tsv /home/xqian/tmp/doc35/gap_widths.tsv

# (2) the constructed fiducial, and its test against the same arm
echo "local g = import 'pgrapher/experiment/protodunevd/crp_gap_fiducial.jsonnet'; g().configs" \
     > /home/xqian/tmp/doc35/gapfid_harness.jsonnet
wcsonnet -o /home/xqian/tmp/doc35/gapfid.json /home/xqian/tmp/doc35/gapfid_harness.jsonnet
python3 docs/nf_sp_img_clus/scripts/fv_gap_fiducial_check.py \
    /home/xqian/tmp/doc35/gapfid.json work/*_d28dlfp

# (3) the margin arms.  -S is wcsonnet's --tla-code, so the numbers go straight
#     through PDVD_PR_TLA and no driver edit is needed to measure.  Every arm
#     carries -S dl_weights='' (M4: the DL vertex is not bit-stable), a pinned
#     library snapshot, and a fresh tag (M13).
#     d34base 30/3/5/3 int 3 | d34x 2.5/3/5/3 int 3 | d34yz 30/17.5/18/18 int 0
#     d34all 2.5/17.5/18/18 int 0 | d34alli3 the same with int 3
#     d34rep  = d34base repeated verbatim -- the run-to-run noise band
for RUN IDX in <the 120 events of the d28dlfp arm>; do
  ./scripts/stage_pr_tag.sh $RUN $IDX <tag> d28dlfp
done
PDVD_PR_TLA="-S dl_weights='' -S tgm_fv_x_margin=2.5 -S tgm_fv_y_margin=17.5 \
             -S tgm_fv_zmax_margin=18 -S tgm_fv_zmin_margin=18 \
             -S tgm_fv_zmax_margin_interior=0" \
  ./run_pr_evt.sh -s d34all -stm-fit <run> all      # PDVD_MAX_JOBS <= 28

# (4) the per-cluster census
python3 docs/nf_sp_img_clus/scripts/fv_margin_census.py \
    d34base d34rep d34x d34yz d34all d34alli3
```

The arm tags below carry a **`d34`** prefix: they were staged while this was
doc 34, and a concurrent session had already taken that number
(`34_ctpc-anisotropic-distance-metric.md`, commit `28d827e4`). The document
renumbered; the tags did not, because a work tag is a record and records are not
renamed (M13).

Arms read read-only: `pdvd/work/*_d28dlfp` (the doc-28 §27 production arm).
Arms written, all fresh tags: `d34base d34rep d34x d34yz d34all d34alli3`, plus
three one-event tags left in place — `d34smoke` (039252/0, a plumbing check),
`d34dlrep` (039349/50, the flipped driver run end to end), and **`d34dlrep2`**
(039349/50, production margins with the DL vertex **on** under the current
library — this is the control §5.1 cites). Nothing under an existing tag,
`sweep/` or `abtest/snap/` was touched.

---

## 1. The three steers, and the arithmetic that closes the second one

The 15 cm y/z inset on PDVD's clustering FV is a **space-charge** allowance.
That reframes doc 33 §3.2, which explained it only as restoring the prototype's
"on the surface = within ~15 cm" operating point for `JudgeSeparateDec_2`. Both
are true; the physical reason is the one that matters, because it is what makes
the number transferable: any stage that asks "is this point safely inside the
detector" on PDVD must carry the same allowance, and SBND — whose space charge
is far smaller — legitimately does not.

The cosmic taggers were the stage that did not carry it. Making them consistent
is not an approximation; it closes **exactly**, in two places:

| axis | tagger box `pdvd_pr_fv` | margin | effective | clustering `dvm.overall` FV ∓ its own margin |
|---|---|---|---|---|
| y | ±336.4 | **17.5** | **±318.9** | 321.4 − 2.5 = **318.9** ✔ |
| z | [0.05, 299.25] | **18** | **[18.05, 281.25]** | [15.05 + 3, 284.25 − 3] = **[18.05, 281.25]** ✔ |
| x | ±339.91 | **2.5** | ±337.41 | the box spans the cathode, so only the outer face is tested |

Two independent exact matches. The y margin is 15 + 2.5 because the clustering
FV_y is the active volume inset 15 cm and then carries `FV_y*_margin` 2.5; the
z margin is 15 + 3 for the same reason with `FV_z*_margin` 3. There is no free
parameter here to tune.

The x margin drops 30 → 2.5. The 30 cm was doc 25 M3's near-CRP dQ/dx guard;
doc 33 §7 measured that the "rise over the last 30 cm" is not a near-CRP
feature at all but a smooth monotone drift gradient with no knee, and the
owner's steer is that the effect must be understood rather than paid for with
volume. §5 measures what the 27.5 cm buys back.

---

## 2. What the gaps actually are — CRP and CRU, not one thing

Every PR log prints the 16 `AnodePlane` sensitive boxes, so this is free
(`fv_gap_measure.py` parses them; no geometry number in this document or in the
new config was typed by hand):

```
x tiles: [-339.910,-3.000], [3.000,339.910]
y tiles: [-336.390,-168.510], [-168.490,-0.610], [0.610,168.490], [168.510,336.390]
z tiles: [0.813,149.520] / [0.813,149.600],  [149.700,298.435] / [149.780,298.435]
```

Read as structure: **each drift volume holds 2 CRPs** (one at y < 0, one at
y > 0), each **3357.8 × 2976.2 mm** — the 3.0 × 3.4 m² ProtoDUNE-VD CRP — and
**each CRP is 4 CRUs**, 2 in y × 2 in z. Doc 33 listed four "gap families"
without saying what they were; they are three different kinds of boundary and
they do not deserve one width:

| seam | separates | in WCT terms | mechanical gap |
|---|---|---|---|
| y = 0 | **CRP ↔ CRP** | anodes 0,1 ↔ 2,3 | **12.2 mm** |
| \|y\| = 168.50 | **CRU ↔ CRU** inside a CRP | face 0 ↔ face 1 of one anode | **0.2 mm** |
| z = 149.65 | **CRU ↔ CRU** inside a CRP | anode 0 ↔ anode 1 | **2.6 mm** bottom, **1.0 mm** top |
| \|x\| < 3.0 | the cathode slab | between drift volumes | **60 mm** |

SBND, for comparison, has exactly one: its CPA, \|x\| < 0.45 cm.

---

## 3. How wide each gap is to the reconstruction

### 3.1 The instrument, and why doc 33's was not good enough

Doc 33 §5 read widths off a pooled point-density profile in 1 mm bins. The
imaged points sit on the ctpc **lattice**, so at 1 mm the profile is a comb —
alternating 3.4 and 0.0 — not a density. Everything below is therefore either
integrated or binned coarser than the point pitch, and every number is quoted
against a **control** measured the same way at a plane with no structure:

- **R(w)** — points within \|d\| < w over the far-field expectation. 1.0 = no deficit.
- **A2, the per-crossing-track gap** — for clusters that straddle the plane by
  ≥ 5 cm on both sides with \|cos θ\| > 0.3 to the seam normal, the median empty
  interval along the normal. This is the width a track actually loses, which is
  what an exclusion volume is for. The same-axis control gives the point-pitch
  floor; **excess = gap − floor**, and half of that is the box half-width.
- **q@core** — median point charge inside the seam over the far field, which
  separates a gap that loses *points* from one that only loses *charge*.

120 events, 9,308,616 imaged points (289,079 no-t0 sentinel points dropped).

### 3.2 The result

| seam | geom h/w | R(0.25) | R(0.5) | R(1) | R(2) | R(3) | gap | floor | excess | **meas h/w** | q@core |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cathode | 3.000 | 0.25 | 0.23 | 0.26 | 0.28 | 0.33 | 8.45 | 0.30 | +8.15 | **4.075** | 0.92 |
| y = 0 (bot) | 0.610 | 0.00 | 0.10 | 0.32 | 0.79 | 0.84 | 1.51 | 0.29 | +1.21 | **0.606** | 0.90 |
| y = 0 (top) | 0.610 | 0.00 | 0.00 | 0.26 | 0.70 | 0.78 | 1.46 | 0.29 | +1.17 | **0.585** | 0.91 |
| \|y\|=168.5 (4 instances) | 0.010 | 0.39–0.56 | 0.58–0.91 | 0.67–0.74 | 0.79–0.87 | 0.81–0.88 | 0.36–0.59 | 0.29 | +0.06…+0.29 | 0.03–0.15 | 0.99–1.07 |
| z = 149.65 (bot) | 0.130 | 0.70 | 1.02 | 1.00 | 0.97 | 0.93 | 0.60 | 0.48 | +0.12 | 0.059 | 0.92 |
| z = 149.65 (top) | 0.050 | 0.66 | 1.37 | 1.06 | 1.00 | 0.94 | 0.39 | 0.48 | −0.09 | −0.046 | 0.96 |

Controls (no structure): R = 0.88–1.11 at every w, excess ±0.01. The
instrument is not manufacturing dips.

Three findings, and the second is a correction to doc 33:

1. **The cathode is the only family that needs dilating.** Measured half-width
   **4.08 cm against 3.00 geometric** — a crossing track loses 8.1 cm where the
   slab is 6.0 cm. And the region is not empty: R ≈ 0.25 inside, at *normal*
   charge (q@core 0.92), so ~25 % of nominal density is reconstructed in a
   volume `DetectorVolumes::contained()` calls "outside the detector". SBND's
   CPA hole, same instrument, is R = 0.00 at \|d\| < 0.25 — literally empty.
2. **The CRP ↔ CRP seam at y = 0 is exactly its mechanical width.** Measured
   half-width 0.585–0.606 cm against 0.610 geometric — agreement to 4 %, from
   two independently measured drift volumes. Doc 33 §5's conclusion that "the
   CRP y seam is real but far wider than the geometry says" was read off the
   comb and **does not survive**: what is wide is a shallow *density* shoulder
   (R still 0.78–0.84 at ±3 cm), not a gap. Tracks cross the shoulder; they do
   not cross the 1.2 cm core.
3. **The CRU ↔ CRU seams are at or below the measurement floor.** \|y\| = 168.5
   costs a crossing track 0.6–2.9 mm beyond the floor; z = 149.65 costs nothing
   measurable (the top instance is −0.9 mm, i.e. consistent with zero). Their
   charge is untouched (q@core 0.99–1.07).

The one asymmetry worth naming: the \|y\| = 168.5 seam has a **0.2 mm**
mechanical gap and shows a ~25 % density deficit out to ±2–3 cm, while
z = 149.65 has a **2.6 mm** gap — thirteen times wider — and shows nothing. The
difference is not mechanical. \|y\| = 168.5 is the boundary between the two
**faces** of one WCT anode, where the tiling and the wire pattern restart;
z = 149.65 is a boundary between two anodes, where they do not. So the y
deficit is a reconstruction edge effect, not a dead region — which is why it
shows up in density and not in the per-track gap. Naming the mechanism is not
needed to build the volume, and the volume is not built on it: a shoulder that
tracks cross is not something to exclude.

### 3.3 The walls, which is where the cushion question is settled

Density inward from each outer wall (2 mm bins) goes from 0 to full within
about 2 mm in y and 4–6 mm in z; the anode (shield) planes ramp more slowly,
0.73 → 0.95 over ~4 cm at the bottom and 0.5 → 1.0 over ~2 cm at the top. So
there is no soft edge in y/z for a cushion to cover — see §4.

---

## 4. The constructed fiducial

`cfg/pgrapher/experiment/protodunevd/crp_gap_fiducial.jsonnet` — a structural
sibling of `cfg/pgrapher/experiment/sbnd/cathode_fiducial.jsonnet`: same
`BoxFiducial` + `CompositeFiducial{logic:'or'}` primitives, same
`{ boxes, composite, tn, configs }` return, same per-axis cushion arguments.
The difference is the source of the widths. SBND models its CPA from
engineering drawings (pad 0.6 / tube 2.7 / knuckle 4.1 cm) and dilates by
0.5 cm to cover whatever the reconstruction adds. PDVD's two widths that matter
were measured *on the reconstruction's own output*, so they already contain
that, and **the cushions default to zero** — dilating a measured reconstruction
width by a reconstruction allowance would double-count it. §3.3 backs the same
conclusion from the walls: there is no soft edge to pad.

| box | half-width (cm) | basis |
|---|---|---|
| `pdvdgap-cathode` | **4.075** in x | measured (geometric 3.000) |
| `pdvdgap-crp_y0` | **0.610** in y | measured 0.585–0.606 ≡ geometric |
| `pdvdgap-cru_y_pos` / `_neg` | 0.010 in y | geometric; measured excess at the floor |
| `pdvdgap-cru_z_bot` | 0.130 in z, x < 0 | geometric; measured ≈ 0 |
| `pdvdgap-cru_z_top` | 0.050 in z, x > 0 | geometric; measured ≈ 0 |

The z seam is two boxes, split at the cathode, because the geometry itself
differs between the drifts (2.6 vs 1.0 mm) — one box would misstate one of them.

The four CRU boxes are kept (`include_cru=true`) even though the measurement
cannot resolve those seams, and the reason is that keeping them is free: they
are narrower than the ctpc point pitch, so the union's flagged share is
0.379 % with or without them. They record mechanics that exist; they claim no
measurement that does not. `include_cru=false` is there for a consumer that
would rather carry only what was measured.

**The test** (`fv_gap_fiducial_check.py`, which re-implements the OR-of-boxes
containment exactly as `aux/src/{Box,Composite}Fiducial.cxx` do) over the same
120-event arm:

| box | points flagged | share | in/out density |
|---|---|---|---|
| `pdvdgap-cathode` | 33,960 | 0.365 % | **0.42** |
| `pdvdgap-crp_y0` | 1,330 | 0.014 % | **0.10** |
| `pdvdgap-cru_y_pos` / `_neg` | 0 | 0.000 % | — |
| `pdvdgap-cru_z_bot` / `_top` | 26 / 0 | 0.000 % | — |
| **union (or)** | **35,299** | **0.379 %** | |

The two measured boxes sit on real deficits (0.42 and 0.10 of the density just
outside them). The four CRU boxes are **narrower than the ctpc point pitch and
so hold no points at all** — they are inert. That is the honest result, not a
bug: nothing was measured at those seams that would justify a wider box, and
padding them to make them "do something" would be inventing structure. They are
kept because they record the mechanics, and a consumer that wants margin has
the cushion arguments.

**Nothing imports this file** — `grep -rl crp_gap_fiducial --include='*.jsonnet' cfg/ pdvd/`
returns **nothing at all**, and that is the whole of its byte-identity argument:
a jsonnet module no config imports cannot change a compiled config. It is
exercised only by the §0 harness. Doc 33 R2 proposed Q/L matching as the first
consumer; that was written before reading `QLMatching.cxx:5149`, where
`m_cathode_fv` turns out to be consulted for the cathode-end `at_x_boundary`
flag and nothing else. PDVD's cathode is a flat slab, so a 3-D test there gains
nothing, and the CRU seams are invisible to that site. **Doc 33 R2's "first
consumer is Q/L, so the change is config-only" is withdrawn.** The consumer
this was built for is in §6.

---

## 5. The margin change, measured

### 5.1 What was run, and why the baseline is a fresh arm

Six 120-event PR arms off the same `d28dlfp` point-tree inputs, one pinned
library (`libWireCellClus.so` md5 `c094432d…`, re-verified byte-identical
against `local/lib` after the campaign), every arm with `-S dl_weights=''`.

| tag | x | y | z_max | z_min | interior |
|---|---|---|---|---|---|
| `d34base` | 30 | 3 | 5 | 3 | 3 |
| `d34rep` | 30 | 3 | 5 | 3 | 3 (verbatim repeat of `d34base`) |
| `d34x` | **2.5** | 3 | 5 | 3 | 3 |
| `d34yz` | 30 | **17.5** | **18** | **18** | 0 |
| `d34all` | **2.5** | **17.5** | **18** | **18** | 0 |
| `d34alli3` | 2.5 | 17.5 | 18 | 18 | **3** |

All six report **7,135 distinct clusters with identical ids**, so every
comparison is per-cluster.

**`d34rep` is identical to `d34base` on every metric — 0 flips of 7,135 on TGM,
STM and containment.** Under a pinned library and fixed TLAs the PR stage is
exactly reproducible, so every delta below is signal, with no noise band to
subtract.

That matters because the obvious baseline is wrong. The production `d28dlfp`
arm ran at 04:24 today; `bec1bd75` (the per-plane good-point pitch floor) and
`79235519` (`good_point_pitch_frac = 0.35` for PDVD production, doc 32 round 3b)
landed at 04:39 and 06:05, with `local/lib` rebuilt at 06:02. Across that
change **STM = 1 moves 663 → 669 with 390 per-cluster flips**, while TGM,
containment and the evaluated set are untouched — the pitch floor feeds the
strict good-point tests and so the STM fit, which is exactly where it shows up.
Comparing anything here against `d28dlfp` would have charged that round's effect
to this one. A one-event control settles the other candidate: `039349/50` rerun
at production margins with the **DL vertex ON** under the current library
(`d34dlrep2`) gives STM = 1 : 21, the same as `d34base` and `d34rep`, against
`d28dlfp`'s 27. That rules `dl_weights` out cleanly. It does not by itself
*demonstrate* the library: the remaining known difference is the 04:39–06:05
window, and the two commits in it act on the good-point pitch floor, which feeds
the strict good-point tests and so the STM fit. That is elimination plus a
mechanism in the right place, not a controlled test — nothing here was rerun
under a snapshot of the pre-06:02 libraries.

### 5.2 The census

| tag | TGM | STM = 1 | STM evaluated | fully contained |
|---|---|---|---|---|
| `d34base` | 1,592 | 669 | 5,543 | 2,942 |
| `d34rep` | 1,592 (**+0**) | 669 (**+0**) | 5,543 (**+0**) | 2,942 (**+0**) |
| `d34x` | 1,213 (**−379**) | 643 (−26) | 5,922 (+379) | 3,338 (**+396**) |
| `d34yz` | 2,501 (**+909**) | 708 (+39) | 4,634 (−909) | 2,018 (**−924**) |
| `d34all` | 2,185 (**+593**) | 688 (+19) | 4,950 (−593) | 2,290 (**−652**) |
| `d34alli3` | 2,231 (+639) | 670 (+1) | 4,904 (−639) | 2,290 (−652) |

Per-cluster flips against `d34base`:

| tag | TGM | STM = 1 | fully contained |
|---|---|---|---|
| `d34rep` | +0 / −0 | +0 / −0 | +0 / −0 |
| `d34x` | +22 / −401 | +65 / −91 | +396 / −0 |
| `d34yz` | +949 / −40 | +206 / −167 | +0 / −924 |
| `d34all` | +937 / −344 | +248 / −229 | +272 / −924 |
| `d34alli3` | +966 / −327 | +233 / −232 | +272 / −924 |

### 5.3 Reading it

**Decomposed by axis, because the net hides it.** The two flips push opposite
ways. The x flip *enlarges* the volume (30 → 2.5 cm of inset), so ends that were
exits become contained: TGM −379, fully-contained +396. The y/z flip *shrinks*
it by 15 cm on four faces: TGM +909, fully-contained −924. Reported only as the
combined arm this reads "TGM +593", which is true and nearly uninformative.

**The mechanism checks out on the cluster ends.** Over the base arm's 9,874 PCA
ends, the x flip **admits 630** ends that were outside at 30 cm and the y/z flip
**excludes 1,597** that were inside at 3/5/3 — against 401 clusters losing TGM
and 949 gaining it, i.e. ~1.6 ends per cluster flip, which is what it should be
when TGM needs both ends out. The census is the arithmetic, not a surprise.

**Doc 33 §6.3's prediction was wrong, and instructively so.** It estimated that
dropping the x margin would turn 232 clusters into stopping-muon candidates.
Measured: STM = 1 goes **down** 26 while fully-contained goes **up** 396. The
PCA-extreme model could only represent two-exit → one-exit conversions; the move
that dominates is one-exit → **zero**-exit — a stopping-muon candidate loses its
single exit and becomes fully contained. The flip is still right; the reason
doc 33 gave for it being cheap is not what happens.

**The interior knob does not matter at this scale.** `d34alli3` holds the CASE-A
interior-support tests at the legacy 3 cm z inset while the endpoints use 18.
Against `d34all` that is **46 clusters of 7,135** on TGM (2,231 vs 2,185) and
**zero** on the fully-contained population (2,290 either way). The doc-35
corner-clipper concern this knob exists for is a 2 cm-scale effect; at 15 cm it
is measured rather than assumed, and it is small. Production therefore ships
`tgm_fv_zmax_margin_interior = 0` — the key suppressed, so
`TaggerCheckTGM.cxx:692` falls back to `fv_tolerance` and there is exactly one
volume, which is the whole point of "consistent". Passing 3 restores the other.

**What the owner is accepting.** The largest consequence is that the
fully-contained population falls **2,942 → 2,290 (−22 %)** and TGM rises
**1,592 → 2,185 (+37 %)**. That is the intended meaning of the change: if
clustering already treats a point within 15 cm of a wall as "on the surface"
because space charge makes its position untrustworthy, a cosmic ending there
cannot be told from one that exits, and through-going is the consistent verdict.
It is still a large move in the population every neutrino selection starts from,
and it has not been hand-scanned. The events carrying the most TGM flips, for a
Bee scan, are `039253/9` (23), `039252/12` (21), `039252/6` (21), `039253/0`
(21), `039253/17` (20), `039252/2` (19).

### 5.4 What shipped

- `protodunevd/pr.jsonnet`: new `tgm_fv_zmin_margin`, default 3 — the literal
  that was hard-coded in both margin vectors. Compiled-config **diff-to-zero**
  at the default; at 18 the vector's last element reads −180. SBND's identical
  literal at `sbnd/clus.jsonnet:1969,1981` is deliberately **not** touched.
- `pdvd/wct-pr-perevt.jsonnet`: `tgm_fv_x_margin = 2.5`,
  `tgm_fv_y_margin = 17.5`, `tgm_fv_zmax_margin = 18`,
  `tgm_fv_zmin_margin = 18`, `tgm_fv_zmax_margin_interior = 0`. Compiles to
  `fv_tolerance = [-25, -25, -175, -175, -180, -180]` mm with the interior key
  absent — the `d34all` operating point exactly — and runs clean end to end
  (039349/50, rc = 0, no DL failure). **Not byte-identical**; §5.2 is its
  evidence.
- All four `*_consistent_fv` flags are on, so this one vector also moves
  `match_isFC` and the cosmic / nue / single-photon containment. That is the
  point of the change, and it is stated here rather than discovered later.

---

## 6. What this document does not establish

- **The 15 cm space-charge inset is adopted, not measured here.** §1 shows the
  tagger margins now close exactly on the clustering FV, which makes the two
  stages consistent; it says nothing about whether 15 cm is the right allowance.
  The measurement that would set it rather than assume it is the transverse
  displacement of cathode-crossing cosmics as a function of y and z — the same
  sample doc 25 already reconstructs.
- **The gap widths are imaged-point measurements, not fitted-trajectory ones.**
  A2 asks where the *points* stop; a track fit interpolates across a gap and
  would give a different, smaller number. For an exclusion volume the point
  measurement is the right one (it is what `contained()` will be asked about),
  but the two are not interchangeable.
- **The face-boundary explanation for the \|y\| = 168.5 density shoulder is a
  hypothesis.** What is measured is that a 0.2 mm mechanical gap shows a ~25 %
  density deficit over ±2–3 cm while a 2.6 mm gap thirteen times wider shows
  none, and that the first is a WCT face boundary and the second is not. No
  test isolating the tiling as the cause was run.
- **No angular or track-population control** is applied to the gap profiles; a
  seam crossed at 90° and one run along are not separated beyond the \|cos θ\|
  > 0.3 requirement A2 imposes.
- **The margin flip has not been hand-scanned.** §5 names the biggest movers;
  nobody has looked at them in Bee.

## 7. Next

1. **Grade `good_point_pitch_frac = 0.35` on STM.** §5.1 found in passing that
   the doc-32-round-3b production flip moves **390 of 7,135 STM verdicts**
   (663 → 669 net) while leaving TGM, containment and the evaluated set
   untouched. That is a real effect of somebody else's round, measured here only
   because it had to be separated from this one's; nobody has looked at whether
   those 390 moved the right way. The arms to do it with are on disk
   (`d28dlfp` is the pre-flip library, `d34base` the post-flip one, same inputs,
   same margins), so it costs a census, not a run.
2. **The gap fiducial's consumer**: `FiducialUtils` gains an optional
   `structure_fiducial`; a step inside it is scored **transparent** in
   `check_dead_volume` / `check_signal_processing` rather than
   `num_points_dead++`; then `MakeFiducialUtils` can be routed off `fiducial=dv`
   onto `pdvd_pr_fv` (doc 33 R4a/R4b). Worth stating explicitly, because it is
   easy to misread as live code: the `test_wpid.apa() < 0` branch in both walks
   is **unreachable today**. With `fiducial == dv`, the loop guard
   `fiducial->contained(p)` and the branch test `dv->contained_by(p).valid()`
   are the same predicate, so the walk exits at a gap instead of stepping into
   one. That is exactly why the transparency fix is a prerequisite for the
   volume swap and not an independent cleanup. `FiducialUtils` is bound by SBND
   and PDHD too, so that round owes gates on all three.
3. **Still owed from doc 33 §7**: fitted dQ/dx vs residual range in bins of
   drift distance on the doc-25 stopping-muon sample. The x margin has now been
   dropped to 2.5 cm by the consistency argument, which is the owner's decision;
   it does not answer what the near-CRP rise was, and that question is still
   open.

## 8. Related

`33_fiducial-volume-pdvd-vs-sbnd.md` (the five-FV map this executes, and the
three corrections this round wrote back into it),
`32_stm-trajectory-end-coverage-and-fiducial-volume.md` §6,
`25_pdvd-stopping-muon-michel-chain.md` M3 (the origin of the 30 cm x margin),
`28_pdvd-pr-perf-round1.md` §27 (the DL flip, hence `dl_weights=''` in every arm
here), `07_pdvd-tpc-geometry-fiducial.md` (the Q/L-side geometry),
`clus/docs/clustering-separate-fv.md` (the 15 cm inset),
`cfg/pgrapher/experiment/sbnd/cathode_fiducial.jsonnet` (the primitive §4
copies), `sbnd_xin/docs/49_stm-containment-fv-inconsistency.md` and
`docs/27_fc-tgm-consistent-fv.md` (the SBND precedents for sharing one fiducial
across TGM/FC/STM), `docs/35_*` (the endpoint-only z widening the
`tgm_fv_zmax_margin_interior` knob exists for).
