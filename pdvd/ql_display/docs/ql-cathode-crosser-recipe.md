# Finding PDVD cathode-crossing standard candles (recipe)

A **cathode crosser** is one physical cosmic track reconstructed as two
clusters — one in the bottom drift volume (apa 0 / BDE, uids < 4000000) and
one in the top volume (apa 4 / TDE, uids = 4000000+ident) — that at the TRUE
flash T0 **meet at the cathode plane (x ≈ 0)** with **aligned directions**.
Because both halves must simultaneously touch the cathode, the pair pins the
event T0 to a single flash with no anode/cathode ambiguity: these are the
best "standard candle" tracks for understanding the PDVD light detectors
(cathode X-ARAPUCAs see both halves; each membrane/PMT group sees its own
half at a known position).

This recipe was derived from 4 pairs hand-picked by the owner in run 039252
evt298567 (tag `crossers`, seeded from `claude-geomfix` scan state), then
refined and validated on evts 298581 and 298595. It has two forms: the
**script** (fast path) and the **manual criteria** (what the script encodes,
for viewer/render-based scanning by a human or model).

## Repro

```bash
cd pdvd/ql_display
# find pairs + write crossers-only decisions for make_labels.py:
python3 find_crossers.py ../work/039252_0/calib-evt298567.json --self-check
python3 find_crossers.py ../work/039252_N/calib-evt<ID>.json \
    --emit decisions-crossers/decisions-evt<ID>.jsonl
python3 make_labels.py ../work/039252_N/calib-evt<ID>.json \
    decisions-crossers/decisions-evt<ID>.jsonl --tag crossers --check
# review (tag `crossers` now covers all 10 scan events 039252_0..9):
../ql_scan/serve_ql_scan.sh 5018 --tag crossers ../work/039252_{0,1,2,3,4,5,6,7,8,9}/calib-evt*.json
# render one candidate flash group (viewer group = gid+1):
python3 render_groups.py ../work/039252_0/calib-evt298567.json \
    --outdir /home/xqian/tmp/png --context /home/xqian/tmp/ctx.jsonl \
    --mode none --groups 43
```

## Geometry background (calib-dump conventions)

- Cluster `x/y/z` are cm, **x uncorrected**. The per-flash T0 correction is
  `x_corr = x + sign_offset * (t_flash + trigger_offset) * drift_speed`
  with `sign_offset`, `drift_speed` from the dump (`trigger_offset` is 0 in
  current dumps — offsets are matcher-applied; NEVER re-apply them).
- Per-volume geometry: apa 0 anode at x = −335.8, cathode at −3.0; apa 4
  anode +335.8, cathode +3.0. The cathode plane is x ≈ 0; **each volume's
  active region ends ~3 cm short of it**, so even a perfect crosser's halves
  are ≳6 cm apart, and with reco + the top/bottom trigger-crate skew the
  true pairs meet at 10–22 cm.
- The viewer's "flash N" and the `.scan_state` group number are the flash
  **gid**. A pair candidate = two bundles sharing one `flash_gid`.

## The five criteria (with measured thresholds)

For a candidate pair (one apa-0 cluster + one apa-4 cluster sharing a flash
gid), computed in the T0-corrected frame of that flash:

1. **Both halves are real tracks**: 3-D span ≥ 50 cm each (the 17 validated
   pairs range 51–427 cm).
2. **Transverse continuity**: minimum point-pair (y,z) distance ≤ 25 cm.
   T0-independent. Validated pairs: 0.4–12.3 cm.
3. **Track-axis collinearity**: `min(global-PCA angle, local-end angle)` ≤
   10°, angles folded to [0,90]. Validated pairs: global 0.6–16.9°, min
   0.4–6.5°. Two traps (same lessons as the C++ `xtpc_joint_pin` gate):
   - do NOT use the connector (closest-point-to-closest-point) angles — a
     small transverse inter-volume shift makes the connector nearly
     perpendicular to a genuine crosser's axis;
   - do NOT trust the local end direction alone (end-curl/fans: evt298567
     c21+c4000097 reads 89° locally, 4.4° globally) nor the global PCA
     alone (bent halves: evt298581 c95+c4000208 reads 16.6° globally, 1.6°
     locally). Take the min of the two.
4. **They meet AT the cathode**: exact closest approach d between the full
   clusters, and the meeting midpoint `|x_mid| ≤ 10 cm`. Validated pairs:
   d = 3.8–21.8 cm, x_mid −7.1…+0.5 cm. The x_mid cut is what kills the
   dominant fake: two halves whose x-extents OVERLAP across the cathode at
   the tested T0 can come arbitrarily close (d ~ 8 cm) but meet deep inside
   a volume (evt298567 c109+c4000131: d = 8.1 cm, aG = 1.0°, x_mid =
   −47.7 cm ⇒ NOT a crosser at that flash).
5. **Flash choice = geometry first, amplitude as tie-break.** The same pair
   appears as bundles on up to ~50 flashes (the at-cathode T0 degeneracy);
   d(t) grows linearly (~0.16 cm/µs per side) away from the true T0, so
   pick the eligible flash minimizing d, and break near-ties (Δd ≤ 3 cm)
   toward the brighter flash (evt298567 c134+c4000062: gid60 d=9.4/479 PE
   vs gid61 d=10.3/3949 PE — the owner picked gid61).
   **Amplitude is NOT a veto**: three validated picks sit on 41–59 PE
   flashes with pred/meas ~ 10 (owner's own gid24 pick included). A dim
   flash under a bright pair is itself interesting light-detector physics
   (saturation veto, `remove_late_light`, masking) — keep it, flag it.

## Tag coverage (all 10 scan events)

The `crossers` viewer tag now spans all 10 hand-scan events of run 039252
(`039252_0..9` = evts 298567, 298581, 298595, 298609, 298623, 298637, 298651,
298665, 298679, 298693). The first 3 events hold the 17 render-validated pairs
(worked examples below); the remaining 7 events hold **finder candidates**
generated by `find_crossers.py` at the default thresholds (28 pairs) — these
are loaded for the owner to examine visually, not yet hand-confirmed. Each half
of every candidate is pre-selected in the scan state so the viewer opens on the
crossers only. Counts per event: 298609 7, 298623 4, 298637 4, 298651 3,
298665 4, 298679 3, 298693 3.

## Worked examples (all 17 validated pairs)

evt298567 (`work/039252_0`), owner's 4 + 2 found:

| gid | pair | len (cm) | d | x_mid | dyz | aG/aL (°) | flash PE | note |
|----:|------|---------|----:|-----:|----:|-----------|--------:|------|
| 24 | c21+c4000097 | 100/129 | 15.6 | −1.7 | 3.7 | 4.4/0.8 | 41 | owner pick; dim flash, overpredicted ~15× |
| 47 | c8+c4000069 | 254/177 | 17.2 | −1.1 | 7.1 | 5.0/2.1 | 1688 | owner pick |
| 61 | c134+c4000062 | 353/404 | 10.3 | −1.7 | 2.4 | 2.8/5.0 | 3949 | owner pick; tie-break over gid60 (479 PE) |
| 83 | c96+c4000076 | 401/327 | 11.0 | −1.3 | 5.2 | 4.9/13.1 | 5223 | owner pick |
| 42 | c52+c4000066 | 100/210 | 14.1 | +0.5 | 4.9 | 3.4/2.7 | 50 | NEW; textbook geometry, dim flash |
| 96 | c139+c4000104 | 147/266 | 19.2 | −6.0 | 12.3 | 4.9/29.9 | 2625 | NEW; local bend, global axes agree |

evt298581 (`work/039252_1`), 6 found:

| gid | pair | len | d | x_mid | dyz | aG/aL | PE | note |
|----:|------|-----|----:|-----:|----:|-------|---:|------|
| 26 | c183+c4000360 | 162/122 | 3.8 | −1.7 | 3.3 | 3.7/20.4 | 7862 | |
| 34 | c47+c4000054 | 175/77 | 6.7 | −0.8 | 4.4 | 2.2/21.0 | 5449 | |
| 23 | c95+c4000208 | 89/107 | 7.8 | −3.6 | 1.7 | 16.6/1.6 | 1196 | curved top half; local axis rescues |
| 72 | c118+c4000210 | 323/177 | 8.9 | −3.8 | 4.1 | 5.3/0.4 | 269 | |
| 58 | c172+c4000343 | 222/192 | 14.3 | −7.1 | 11.5 | 4.5/36.7 | 2575 | local bend |
| 173 | c9+c4000151 | 401/360 | 21.8 | −2.7 | 3.0 | 1.3/0.8 | 59 | 4 m track, 1.3° collinear, 59 PE flash(!) |

evt298595 (`work/039252_2`), 5 found:

| gid | pair | len | d | x_mid | dyz | aG/aL | PE | note |
|----:|------|-----|----:|-----:|----:|-------|---:|------|
| 65 | c4+c4000018 | 339/346 | 6.3 | −2.5 | 1.8 | 5.9/1.4 | 2640 | |
| 35 | c157+c4000433 | 282/213 | 6.5 | −1.7 | 5.2 | 4.6/2.1 | 1596 | both halves auto-matched already |
| 14 | c161+c4000453 | 51/52 | 10.3 | −2.4 | 9.5 | 0.6/6.5 | 1877 | short, shallow (near-cathode-plane) — weaker T0 lever |
| 111 | c6+c4000437 | 350/70 | 16.8 | −0.3 | 11.5 | 16.9/5.5 | 1941 | |
| 29 | c154+c4000393 | 120/145 | 17.2 | −3.3 | 0.4 | 3.2/4.6 | 341 | |

Counterexamples (correctly rejected):

| event | pair | fails on |
|-------|------|----------|
| 298567 | c109+c4000131 (d 8.1, aG 1.0) | x_mid −47.7: halves overlap in x at that T0 — wrong flash |
| 298567 | c141+c4000196 (d 9.4) | x_mid +53 AND aG 23.3 |
| 298581 | c169+c4000346 gid55 (d 3.3) | aG 58.7 / aL 16.7: accidental touch, not one track |

## Pitfalls

- **One-cluster-many-flashes**: every genuine pair also forms bundles on
  dozens of wrong flashes (d grows linearly with |Δt|). Judge the pair once
  at its best flash; do not keep duplicates.
- **uid 3999999**: ident = −1 pseudo-cluster (calib-dump artifact, can
  collide — two such clusters under one flash in evt298595). Never select.
- **Junk tails**: overclustered material can stick out past the cathode
  (evt298567 c4000062 has a ~45 cm tail). Closest-approach + x_mid metrics
  are robust to this; per-cluster "endpoint gap to cathode" is not.
- **window-truncated halves**: still fine as crosser halves (3 of the 4
  owner picks carry `wtrunc`); the C++ admits them via scenario 2.
- **Shallow crossers** (track nearly parallel to the cathode plane, e.g.
  evt298595 gid14): pass all cuts but d(t) varies slowly, so the flash
  choice has more uncertainty — mark lower confidence.

## Relation to the C++ xTPC flags

With the PDVD xTPC enable (toolkit commit on `apply-pointcloud`,
`cfg/pgrapher/experiment/protodunevd/qlmatching.jsonnet`: `xtpc_flag`,
`xtpc_dmax=25cm`, `xtpc_joint_pin`, plus `cathode_ext2=−12cm` for candidate
admission), the calib dump carries the matcher's own version of this test
per bundle: `xtpc_consistent` (geometric pair confirmed), `xtpc_scenario1`
(closest approach < 25 cm), `xtpc_pin` (pair bound to ONE flash, exempt from
the LASSO strength prune). Scanners can use these as a strong prior, but the
C++ pairs by flash-time coincidence and distance only at each flash — it
does not apply the x_mid cut per flash and its flash pick is light-driven
(min ks-sum among coincident pairings) — so still verify visually. The
`crossers` tag itself was built with the python finder above.

Validation after the enable (run 039252 reprocessed, toolkit 0017de8e):
the 17 pairs went from 4/17 auto-matched together (1/17 correct flash) to
17/17 together, 9/17 on the exact hand-scan flash; the other 8 pinned one
neighboring flash away (3.5-43 us) because the C++ pin flash choice is
ks-led (geometry only tie-breaks). When scanning with these flags, treat a
pinned pair on flash g as covering g +- ~2 flashes and verify the meeting
distance yourself.
