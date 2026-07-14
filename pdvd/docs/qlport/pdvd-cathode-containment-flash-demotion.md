# PDVD Q/L matching: cathode containment demotes bright cathode-crosser flashes

Run 039252 evt 298567 — why two cathode-crossing cosmics are matched to *dim*
flashes when a much *brighter* flash sits ~30–46 µs away.

**Status: analysis only — no toolkit code or config was changed.** All inputs are
our own pipeline products.

## Repro

```bash
cd pdvd
# §1-9 numbers + figure (reads one calib dump, no wire-cell run):
python3 docs/qlport/analyze_cathode_containment.py
#   input: work/039252_0/calib-evt298567.json   (Xe/175nm production dump)
#   figure: docs/qlport/cathode_containment_298567.png

# §10 applied study (2 cm anode pull + cathode cushion 2.0 cm), tag pull2c2.
# Produce the 18 tagged variant dumps (canonical dumps untouched):
for i in $(seq 0 17); do
  PDVD_QL_EXTRA_OFFSET_US=13.507 PDVD_QL_CATHODE_EXT1_CM=2.0 \
    ./run_clus_evt.sh -calib -s pull2c2 39252 $i
done
python3 docs/qlport/compare_pull2c2.py     # 18-event before/after census
```

Companion: `docs/qlmatch/pdvd-light-charge-time-chain.md` (the light↔charge time
chain and the residual-offset hunt this event is a symptom of). Sibling debug of
a *different* pair in the same event: `docs/pdvd-qlmatching-debug-evt298567.md`.

---

## 1. The claim

While hand-scanning evt 298567, two cathode-crossing pairs are matched to flashes
whose measured PE is far too low for a bright cathode crosser; a much brighter
flash sits a few cm (in drift) away and looks like the physical match:

| current (displayed) match | measured PE | suspected correct (bright) flash | its PE | Δt |
|---|---|---|---|---|
| flash **gid38** t=−1090.9 µs ← `top:60`,`bot:50` | **186.6** | flash **gid37** t=−1137.3 µs | **5466** | 46.4 µs ≈ 6.9 cm |
| flash **gid42** t=−965.1 µs ← `bot:8`,`top:63` | **50.2** | flash **gid41** t=−994.0 µs | **5679** | 28.9 µs ≈ 4.3 cm |

(`gid` = the ql_scan display's flash number; `top:NN`/`bot:NN` = the cluster
`ident` in the top/bottom volume. Bottom = apa 0, uid = ident; top = apa 4,
uid = 4000000+ident.)

**Hypothesis (confirmed below):** under the bright flash, one TPC half of the
crosser is pushed *past the cathode* and fails the cathode-containment cut, so
that flash is removed as a candidate for that half; because a crosser's two
halves must share one flash, the joint match falls back to a dim flash on which
both halves stay inside the box.

---

## 2. The mechanism: containment is a hard per-(flash,cluster) prefilter

`require_containment` (PDVD default **true**, `qlmatching.jsonnet:53,228`) is a
**hard gate, not a scoring term**. In the per-flash × per-cluster candidate loop:

```
match/src/QLMatching.cxx:1360   if (m_require_containment && !contained) continue;   // skips pre_bundles.insert (L1530)
```

An uncontained `(cluster half, flash)` pair never enters `pre_bundles` → never
enters the LASSO, the cross-TPC pairing, or the `xtpc_joint_pin` crosser rescue
(all fed from the contained pool, `QLMatching.cxx:3227-3232`). It is **deleted**,
not down-weighted. So containment is evaluated *independently for every flash*: a
half can be contained under flash A's T0 and dropped under flash B's.

The verdict (`QLMatching.cxx:3645-3649`) is taken in the per-TPC drift coordinate
`u = s·(x − anode_x)` (u = 0 at the anode, u = u_cathode at the cathode; the sign
`s` makes it outward-positive; `x` already carries the flash's `flash_x_offset`):

```
contained  <=>  first_u > anode_in − 1 cm  &&  last_u > 0  &&  last_u < cathode_in  &&  first_u < u_cathode
             with  cathode_in = u_cathode + cathode_ext1
```

"Too much into the cathode" = the (gap-trimmed) cathode-end `last_u` exceeds
`cathode_in`. The flash time enters entirely through `x`:
`flash_x_offset = sign_offset·(flash_time + trigger_offset)·drift_speed`
(`QLMatching.cxx:1297-1302`), per-crate: a bottom half uses the BDE clock, a top
half the TDE clock (dump `f["time"]` vs `f["time1"]`, commit e587f357).

### 2.1 The tolerance is +1.2 cm past the cathode — not ±3 cm

The containment overshoot budget is **`cathode_ext1 = +1.2 cm`**:

```
match/inc/WireCellMatch/QLMatching.h:238   double m_cathode_ext1{1.2 * units::cm};
```

PDVD's `qlmatching.jsonnet` sets only `cathode_ext2` (the *at-cathode flag*
window, −2 cm), never `cathode_ext1` — confirmed by grep — so by the key-
suppression idiom `cathode_ext1` keeps the C++ default +1.2 cm.

The **"3 cm"** in the question is a *different* quantity: the geometry places each
drift volume's active cathode-facing surface at **|x| = 3.0 cm** from the central
membrane (`params.jsonnet:33`, echoed `clus.jsonnet:629-630`). That 3 cm is what
*defines* `u_cathode` itself (u_cathode = 336.91 cm here); it is not the overshoot
tolerance. So the cut fires at **u_cathode + 1.2 cm = 338.11 cm**, and a half is
dropped once its cathode-end reaches there — only 1.2 cm past the cathode plane.

---

## 3. Case 1 — `top:60` + `bot:50` (airtight)

Cathode-end drift coordinate `u` of each half under the two flashes (v = 0.148073
cm/µs, per-crate clock; u_cathode = 336.91, cathode_in = 338.11):

| half | flash | PE | cathode-end u (rawmax) | past cathode face | past +1.2 tol | pts beyond | in dump? |
|---|---|---|---|---|---|---|---|
| `top:60` | gid38 (current) | 186.6 | 329.56 | −7.35 | −8.55 | 0 | **kept** |
| `bot:50` | gid38 (current) | 186.6 | 333.73 | −3.18 | −4.38 | 0 | **kept** |
| `top:60` | gid37 (bright) | 5466 | 336.43 | **−0.48** (at cathode) | −1.68 | 0 | **kept** |
| `bot:50` | gid37 (bright) | 5466 | **340.60** | **+3.69** | **+2.49** | 13 (2.0%) | **DROPPED** |

Under the bright flash gid37 the **top** half lands essentially *on* the cathode
(0.5 cm short) — the signature of a genuine cathode crosser — but the **bottom**
half is driven **+3.7 cm past the cathode** (13 points, 2 % of the cluster, beyond
`cathode_in`). That fails `last_u < cathode_in` → `require_containment` deletes
`(bot:50, gid37)`.

This is confirmed independently by the code's own candidate lists in the dump:
**`bot:50`'s candidate-flash list starts at gid38** (it has *no* gid37 bundle),
while `top:60` *does* have gid37. The crosser pair must share one flash; since
`bot:50` cannot use gid37, the joint match falls back to gid38 — where both halves
are contained but **neither reaches the cathode** (7.3 / 3.2 cm short), i.e. the
crosser looks wrongly-placed, and the matched flash is a dim 186 PE.

The +3.7 cm recomputed overshoot is large and dense (13 pts), so it is robust to
the code's exact endpoint definition (§6) — case 1 needs no caveat.

---

## 4. Case 2 — `bot:8` + `top:63` (same effect, marginal)

| half | flash | PE | cathode-end u (rawmax) | past cathode face | past +1.2 tol | pts beyond | in dump? |
|---|---|---|---|---|---|---|---|
| `bot:8` | gid42 (current) | 50.2 | 334.95 | −1.96 | −3.16 | 0 | **kept** |
| `top:63` | gid42 (current) | 50.2 | 334.04 | −2.87 | −4.07 | 0 | **kept** |
| `bot:8` | gid41 (bright) | 5679 | 339.23 | **+2.32** | **+1.12** | 8 (0.5%) | **DROPPED** |
| `top:63` | gid41 (bright) | 5679 | 338.32 | **+1.41** | **+0.21** | 2 (0.18%) | **DROPPED** |

Here the bright flash gid41 pushes **both** halves just past `cathode_in`
(bot:8 +1.1 cm past the tol, top:63 +0.2 cm past). Both `(cluster, gid41)` pairs
are absent from the dump; **both candidate-flash lists start at gid42**. So gid41
is dropped for the whole pair → falls back to dim gid42 (50 PE).

The overshoots here are at the tolerance edge, so this case is stated with the §6
caveat: the recomputed cm number and the bundle-absence agree, and the fact that
both halves' candidate lists begin *exactly* at gid42 (the first flash for which
they come inside the box) is the decisive evidence.

**Not a global drop.** Census from the dump: gid37 is a candidate for 22 distinct
clusters, gid38 for 24, gid41 for 26, gid42 for 29 — all four flashes are widely
used. So the missing halves are *cluster-specific*, and containment is the only
per-(flash,cluster) gate, which forces the attribution.

---

## 5. Direct answers to the two questions

**(1) Is this what's happening?** Yes. In both pairs the brighter/earlier flash
*is* a candidate the matcher considers, and it is removed by `require_containment`
because one (case 1: bottom) or both (case 2) TPC halves overshoot the cathode by
more than the +1.2 cm budget. With the bright flash gone for at least one half of
the crosser, the joint match settles on a dim flash that keeps both halves boxed.
The low measured PE Xin noticed is the *symptom* of that fallback, not the cause.

**(2) How much do they cut into the cathode — vs the "±3 cm"?** Reported against
both the +1.2 cm containment tolerance and the user's 3 cm cathode face:

| half (bright flash) | past cathode face | vs 3 cm face | past +1.2 cm tol | dropped? |
|---|---|---|---|---|
| case 1 `top:60` (gid37) | −0.5 cm | inside | −1.7 cm | no |
| case 1 `bot:50` (gid37) | **+3.7 cm** | **past 3 cm too** | **+2.5 cm** | **yes** |
| case 2 `bot:8` (gid41) | +2.3 cm | inside 3 cm | **+1.1 cm** | **yes** |
| case 2 `top:63` (gid41) | +1.4 cm | inside 3 cm | **+0.2 cm** | **yes** |

This localizes the whole effect to `cathode_ext1`:

- **If the boundary were 3 cm**, case 2 (both halves ≤ +2.3 cm) would *pass* and
  the pair would match its bright flash gid41. Case 1's bottom (+3.7 cm) would
  *still* fail even at 3 cm.
- At the actual **+1.2 cm** tolerance, all three overshooting halves are dropped.

So case 2 is entirely a "tolerance is 1.2 cm, not 3 cm" effect; case 1's bottom
half is genuinely ~3–4 cm mis-placed and would need a real T0/geometry fix, not a
looser tolerance.

---

## 6. Precision caveat (why case 2 is "marginal")

The number this script reports is the **per-point `rawmax`** over the cluster's
`x[]` array. The code's `last_u` is a **per-slice** value (one point per slice,
`pts.at(0).x()`, `QLMatching.cxx:3498-3506`) after an always-on gap-trim that
walks the deepest slices inward and snaps back an *isolated* straggle separated by
a >0.75 cm gap (`QLMatching.cxx:3547-3569`). The robust/charge trim
(`QLMatching.cxx:3583+`) is **OFF** for PDVD (`qlmatching.jsonnet:324`). For case 1
the +3.7 cm overshoot carries 13 contiguous points (2 % of the cluster) — far too
dense to be gap-trimmed, so the containment failure is method-independent. For
case 2 the overshoots (2 and 8 points, ≤ +1.1 cm past tol) are comparable to what
the slice-level definition could shift, so there the **bundle-absence + the
candidate-list boundary** (not the cm value) carry the verdict. Additionally the
dump applies `cross_side_mismatch_drop` on *contained* bundles
(`QLMatching.cxx:2831`), a second reason a contained bundle could be absent — so
absence alone is only conclusive when paired with the independent overshoot
recomputation, as done for case 1.

---

## 7. Why a dim flash wins once the bright one is gone

Nothing in the surviving-flash scoring rewards absolute brightness, so a 50 PE
flash is freely selected once the 5679 PE one is removed:

- Bundle chi2 (`TimingTPCBundle.cxx:139`) and the KS term normalize the predicted
  and measured PE patterns to unit sum — pure *shape* metrics, PE-scale-blind.
- The LASSO L1 weight (`QLMatching.cxx:1713-1718`) penalizes the *fractional* PE
  mismatch `|pred−meas|/meas`, floored at 0.3 — again not absolute PE.
- The single global QtoL normalization (`qlmatching.jsonnet:154`) is known to
  over-predict ~10×, so predicted light is not on an absolute scale anyway.

Hence the containment prefilter — a hard geometric gate evaluated per flash — is
decisive: it silently removes the correct bright flash for the overshooting half,
and the pattern-based fit then settles on a dim flash whose T0 boxes both halves.

---

## 8. Connection to the residual-offset hunt

This event is a concrete instance of the overall light↔charge shift suspected in
`docs/qlmatch/pdvd-light-charge-time-chain.md`. The overshoots that trip the gate
are *small* (case 2 ≈ +0.2…+1.1 cm past the tolerance; case 1 bottom +2.5 cm) —
the scale of a modest unaccounted time/position offset or a drift-velocity
mismatch, not a gross misassignment. A tight +1.2 cm containment budget turns such
a small over-drift into a *categorical* flash demotion.

One further observation, stated **tentatively**: under the bright flash gid37 the
two halves of pair 1 do not meet at the cathode — `top:60` lands at −0.5 cm,
`bot:50` at +3.7 cm, a ~4 cm top/bottom discrepancy that no single shared flash
removes (both halves move together). That is the shape a residual *per-crate*
(top/bottom) offset would produce. **But** the `top:60`+`bot:50` pairing is a
*human hand-scan label* — the auto-matcher put different clusters on gid38
(bot uid 80, top uid 4000130) — so this ~4 cm may equally mean these two are not a
validated same-track pair. It is a lead for the offset hunt, not evidence of an
offset on its own.

Related: `[[project_ql_tpc_containment]]`, `[[project_pdvd_qlmatching]]`,
`docs/pdvd-crosser-flash-mismatch.md`.

---

## 9. Next-checks (diagnostic — no default changed)

Any change to `cathode_ext1` alters production output unconditionally and is the
owner's call (this doc does not propose one). To isolate/confirm the effect:

1. **Re-run these two events with `require_containment=false`** (the driver
   already exposes this for trigger-offset diagnostics) and confirm the pair now
   lands on the bright flash gid37 / gid41 with a good chi2 — proving the gate,
   not the fit, caused the demotion.
2. **Census over-cathode depth across many crossers.** Extend
   `analyze_cathode_containment.py` over the standard manifest: for every
   contained-on-the-dim-flash crosser, measure the cathode-end overshoot on the
   nearby bright flash. If the distribution clusters just past +1.2 cm with a
   consistent sign (and a bottom-vs-top split), a systematic offset/velocity — not
   individual mis-clustering — is demoting bright flashes, and the fix belongs in
   the time chain (companion doc §7 / §8.14), not in the tolerance.
3. Only if (2) shows the overshoot is genuinely geometric slack (not a T0 error)
   would loosening `cathode_ext1` toward the 3 cm face be on the table — and that
   is a stop-and-ask, gated change, not something to tune here.

## 10. Applied study: 2 cm anode pull + cathode cushion 1.2 → 2.0 cm

> **Status: adopted as PDVD processing default 2026-07-14 (owner decision);
> NOT byte-identical; census + A/B still OPEN.** The two knobs were first run
> ONLY for the tagged study (tag `pull2c2`, knob-OFF compiled config proven
> byte-identical). On 2026-07-14 the owner made both the **runner** production
> default for **all** PDVD runs (`run_clus_evt.sh`: `PDVD_QL_EXTRA_OFFSET_US`
> defaults 13.507 µs, `PDVD_QL_CATHODE_EXT1_CM` defaults 2.0 cm — set to 0 /
> 1.2 to recover the pre-study path). The toolkit `qlmatching.jsonnet` default
> stays null (byte-identical) — the runner is where PDVD processing turns them
> on. **These remain TRIAL values fit to the two hand labels of §1, not
> census-pinned**: the pull magnitude should ultimately equal the
> census-measured offset (§9.2), and the wider cushion + global pull still owe
> a full-manifest A/B. Revisit the default once the census lands.

### 10.1 What was changed (two default-OFF knobs)

Both levers already existed in the flash-placement term
`flash_x_offset = sign·(flash_time + trigger_offset)·drift_speed`
and the containment cut `last_u < u_cathode + cathode_ext1`:

- **2 cm toward-anode pull** — an *additive* `+13.507 µs` on **both** crate
  trigger offsets (`2.0 / 0.148073`). `du/d(trigger) = −v` on both volumes, so a
  larger offset lowers `u` (toward the anode). Additive preserves the per-crate
  BDE/TDE skew (unlike the absolute `PDVD_TRIGGER_OFFSET_US`). New runner knob
  `PDVD_QL_EXTRA_OFFSET_US` (default 0).
- **cathode cushion 1.2 → 2.0 cm** — `cathode_ext1 = 2.0 cm`, moving the
  containment ceiling `u_cathode + cathode_ext1` from 338.11 → **338.91 cm**.
  New toolkit knob `cathode_ext1` in `qlmatching.jsonnet` (default null → key
  omitted → C++ +1.2), threaded through `wct-clustering.jsonnet`
  (`ql_cathode_ext1_cm`) and the runner (`PDVD_QL_CATHODE_EXT1_CM`).

**Byte-identical-OFF proof.** Compiled PDVD config (`do_qlmatch=true`, evt298567)
with both knobs off, new code vs pre-change code: `diff` empty. `cathode_ext1`
absent from the OFF config; present as `20` (2.0·wc.cm) only when on, with
`trigger_offsets = [−2503821, −2483677]` ns (both pulled +13.507 µs, the
~20.14 µs crate skew preserved) and `anode_ext*` still at C++ defaults.

### 10.2 The two target crossers are fixed

Re-running evt298567 with the knobs on, `bot:50`'s cathode end drops exactly
340.60 → **338.60 cm** (−2.0 cm), now inside the widened 338.91 ceiling (the
other three halves were already inside once pulled). All four bright-flash
halves are now **contained**, and QLMatching's own final selection
(`auto_selected`, the field the ql_scan viewer renders) converges both halves of
each crosser onto its single bright coincident flash:

| cluster | OFF `auto_selected` | ON `auto_selected` |
|---|---|---|
| `bot:50` | gid79  t=+666.3 PE=123  | **gid37  t=−1123.8 PE=5466** |
| `top:60` | gid55  t=−436.9 PE=864  | **gid37  t=−1123.8 PE=5466** |
| `bot:8`  | gid47  t=−813.4 PE=1688 | **gid41  t=−980.5 PE=5679** |
| `top:63` | gid70  t=+260.7 PE=5860 | **gid41  t=−980.5 PE=5679** |

(Note: the *current* canonical dump matched these halves to scattered flashes
79/55/47/70 — not the 38/42 seen in an earlier display state; the canonical set
was regenerated in the §8.14 shield-FV redo since. Either way the OFF matching
split each crosser across two flashes; the ON matching pairs both halves on one
bright flash — the physically-correct outcome for a cathode crosser.)

### 10.3 The global cost is large — this is a case-check, not a validation

The pull is a **global** registration change: shifting every cluster's x by
2 cm re-optimizes the coupled LASSO, so it re-matches far more than the two
targets. Across all 18 candle events (`compare_pull2c2.py`):

- **1152 of ~2100 auto_selected clusters (~55%) change their matched flash**;
  only **532** move to a *brighter* flash (brightness is not the quality metric
  — chi2/KS shape is — so this neither confirms nor refutes the re-matches).
- Total matched-bundle count and contained-bundle count barely move
  (sel ±3/event; contained ±20 of ~7000): the knobs **reshuffle** assignments,
  they don't add/remove many matches. The cushion's own effect on containment is
  small; the **pull dominates the churn**.

So the 18-event `pull2c2` display shows whether the *target* crossers land
correctly (they do) — but it does **not** establish that the other ~55% of
re-matches are improvements. That verdict needs the hand-scan of the display
plus the two things this study deliberately does not do:

1. **Disentangle** pull-only vs cushion-only (two more tagged runs) — is the
   churn the geometric cushion or the time pull?
2. The **census** (§9.2) to pin the *true* offset (the pull magnitude should
   equal it, not be sized to `bot:50`), the per-crate velocity split, and the
   residual cushion; then a **full-manifest byte-identical A/B** before any
   default changes. A pull larger than the true offset pushes anode-touching
   tracks past the −3.0 cm anode floor (up to ~19–26 clusters/event sit within
   3 cm of the anode), trading cathode-side gains for anode-side losses.

### 10.4 How to view

Canonical shield-FV dumps are untouched (`work/039252_<idx>/`); the variant is a
separate tagged set (`work/039252_<idx>_pull2c2/`, symlinked imaging inputs).
Serve it as its own display instance so its labels stay apart:

```
cd wcp-porting-img/pdvd
./ql_scan/serve_ql_scan.sh 5019 --tag pull2c2 work/039252_*_pull2c2/calib-evt*.json
# then from the laptop:  ssh -L 5019:localhost:5019 user@wcgpu1
#                        http://localhost:5019/ql_scan_viewer
```
