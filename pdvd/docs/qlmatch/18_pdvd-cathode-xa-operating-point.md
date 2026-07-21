# 18 — PDVD Q/L matching: the cathode-XA-anchored operating point

2026-07-16.  Status: **ADOPTED as the PDVD runner default** (toolkit knobs all
default OFF and byte-identical when off — gate in §4).  Follow-up to doc 17
(`17_pdvd-pd-family-reliability-evt298567.md`), which established that the
cathode X-ARAPUCAs are the only reliable PD family (full-stream readout, 100%
flash coverage, proportional response) while the membrane/wall XAs are bimodal.

## Repro

```bash
# threshold derivation (tables in §3)
cd pdvd/docs/qlmatch/scripts && python3 derive_cathode_operating_point.py

# knobs-OFF byte-identical gate vs the canonical _keep dumps (§4)
cd pdvd && mkdir -p work/039252_0_abgate && cd work/039252_0_abgate && \
  for f in ../039252_0_keep/clusters-apa-anode*-ms-*.tar.gz \
           ../039252_0_keep/protodune-sp-dnnroi-frames-anode*.tar.bz2; do ln -sf "$f" .; done
cd ../.. && PDVD_LIGHT_SUFFIX=_keep PDVD_QL_MASK_WALL_XA=0 PDVD_QL_WALL_FLAGS=1 \
  PDVD_QL_FLASH_SEL=0 PDVD_QL_REJECT_OVERPRED=0 ./run_clus_evt.sh -calib 039252 0 -s abgate
# then hash_archive.py field-1 compare of all mabc-*.zip + cmp of calib-evt298567.json

# 18-event production reprocess (§5) -- new knobs ON via runner defaults
for i in $(seq 0 17); do d=work/039252_${i}_cathxa; mkdir -p $d; \
  (cd $d && for f in ../039252_${i}_keep/clusters-apa-anode*-ms-*.tar.gz \
            ../039252_${i}_keep/protodune-sp-dnnroi-frames-anode*.tar.bz2; do ln -sf "$f" .; done); done
PDVD_MAX_JOBS=6 PDVD_LIGHT_SUFFIX=_keep PDVD_QL_ROBUST_TRIM=1 \
  PDVD_QL_ROBUST_WALK_FLOOR=1 PDVD_QL_XTPC_CATHODE_TOL_CM=10 \
  ./run_clus_evt.sh -calib 039252 all -s cathxa

# scan-pick carryover + display (§6)
python ql_display/remap_scan_state.py --calib work/039252_0_cathxa/calib-evt298567.json \
  --tag cathxa --labels work/ql_labels/wfresc/labels-evt298567.json
# (per-event --decisions form for the other 17; see remap_scan_state.py header)
./ql_scan/serve_ql_scan.sh 5020 --tag cathxa work/039252_*_cathxa/calib-evt*.json
```

Toolkit commits: `4331ac47` (xtpc cathode rescue, pre-existing dependency) and
`b9cb1561` (this operating point), branch `apply-pointcloud`.

## 1. The four changes

All four are toolkit knobs that **default to the legacy behaviour** (C++
defaults OFF, jsonnet keys suppressed when off); `run_clus_evt.sh` turns them
on for PDVD production, each individually revertible by env var:

| # | change | toolkit knob (protodunevd/qlmatching.jsonnet) | runner env (default) | production value |
|---|--------|----------------------------------------------|----------------------|------------------|
| 1 | wall XAs out of chi2/LASSO/KS | `mask_wall_xa` (ch_mask += [0,1,3,12,18,19]) | `PDVD_QL_MASK_WALL_XA` (1) | on |
| 2 | cathode-scoped flash admission | `flash_sel_cathode` + `flash_sel_minPE/min_fired/fired_pe` → C++ `flash_sel_channels=[4..11]` | `PDVD_QL_FLASH_SEL` (1), `_MINPE` (5), `_MINFIRED` (2), `_FIRED_PE` (1.0) | ≥5 PE and ≥2 ch @ ≥1 PE on cathode |
| 3 | ±y wall proximity flag OFF | `wall_flags=false` (empties `pd_wall_channels_{ylo,yhi}`) | `PDVD_QL_WALL_FLAGS` (0 = off) | off |
| 4 | cathode-scoped reject_overpred | `reject_overpred` + C++ `overpred_channels=[4..11]`, ceilings | `PDVD_QL_REJECT_OVERPRED` (1), `_TOTAL` (15), `_MAXCH` (50) | R_total ≤ 15, R_max ≤ 50 |

Notes:
- (1) `mask_ks` (`bundle_mask_ks`) is already on for PDVD, so masked wall XAs
  leave the KS shape test together with the chi2 and the LASSO rows.  The dim
  ch 2 and Ar-blind ch 13 were already in `ch_mask_base`.
- (2) is **on top of** the all-PD `flash_minPE=25` floor, which is unchanged.
- (3) With the wall XAs masked, the wall relax channels protected nothing; the
  wall `flag_close_to_PMT` otherwise only exempted wall-hugging junk from the
  overpred prefilter and inflated chi2 denominators for free.  Doc 17 found the
  wall flag helped only marginally, on the subdominant z-wall PMTs.  The
  bottom-anode proximity flag (`anode_pd_channels`) is untouched.
- (4) The channel scope is new C++ (`overpred_channels`): judged over all PDs, a
  self-trigger-silent PMT reading 0 fakes over-prediction on good bundles; the
  cathode XAs are always read out.  The production exemption
  (close_to_PMT | window_truncated | at_x_boundary) is unchanged.

## 2. Data used to tune

- 18-event keep-round dumps `work/039252_<0..17>_keep/calib-evt*.json`
  (canonical since 2026-07-16) with the keep-round confirmed candle scans
  `ql_display/decisions-{crossers,boundary}-keep/` — 297 confirmed candle
  flashes, 469 confirmed (flash, cluster) pairs, 4354 admitted flashes,
  92 014 non-exempt candidate bundles.
- The evt298567 full hand scan (44 matches / 84 rejected,
  `work/ql_labels/wfresc/labels-evt298567.json`) as an independent safety set.

## 3. Threshold derivation

### 3.1 Flash admission (cathode sum PE, n fired @ ≥1 PE)

| sum≥ | nf≥ | confirmed lost | admitted flashes cut |
|-----:|----:|---------------:|---------------------:|
| 1 | 1 | 2/297 | 0.8% |
| 2 | 1 | 3/297 | 1.8% |
| 5 | 1 | 6/297 | 5.5% |
| **5** | **2** | **8/297 (2.7%)** | **6.7%** |
| 10 | 2 | 12/297 | 9.4% |

Chosen: **sum ≥ 5 PE AND ≥ 2 fired @ ≥ 1 PE**.  The 8 lost confirmed candles
are all dim (27–160 PE total) wall-hugging boundary tracks / dim crosser
halves; two (evt298651 gid 134, evt298679 gid 142) have literally **zero**
cathode signal — with the wall XAs also masked, such flashes carry no usable
fit information anyway.  The evt298567 full-scan matched flashes clear the cut
with two orders of magnitude of margin (min cathode sum 1644 PE, min 7 fired).

### 3.2 Cathode-scoped overpred ceilings

| R_tot ≤ | R_max ≤ | confirmed culled | candidate bundles culled |
|--------:|--------:|-----------------:|-------------------------:|
| 10 | 30 | 5/33 | 18.6% |
| **15** | **50** | **3/33** | **15.8%** |
| 30 | 50 | 3/33 | 14.7% |

Chosen: **R_total ≤ 15, R_max ≤ 50**.  The 3 confirmed pairs above them are
wrong-amplitude picks (cathode meas 8–123 PE vs pred 1 000–19 500 — the
geometric crosser/boundary finder confirms by TIME coincidence, so a pair whose
true light was lost still gets confirmed): evt298749 (207, 4000222) R=137/247,
evt298777 (158, 4000005) R=639/3259, evt298665 (25, 74) R=60/1256.  Every
evt298567 full-scan match survives with wide margin (max R_total 1.43, max
R_max 4.97; even the human-rejected entries only reach 1.55/15.6 — the culling
power is against the 92k pre-fit candidate pool, not the auto-match list).

## 4. Byte-identical gate (knobs off) + build

- `wcbuild` clean; freshness proof done (libWireCellMatch.so newer than the
  source edit at A/B time); `./build/match/wcdoctest-match` 23/23 assertions.
- Gate label: **`work/039252_0_abgate`** (kept) — run with the new binary +
  new config and every new knob at legacy (`PDVD_QL_MASK_WALL_XA=0
  PDVD_QL_WALL_FLAGS=1 PDVD_QL_FLASH_SEL=0 PDVD_QL_REJECT_OVERPRED=0`), vs the
  canonical `work/039252_0_keep`:
  all 27 `mabc-*.zip` member-content identical (`hash_archive.py`), and
  `calib-evt298567.json` **byte-identical** (`cmp`).  This also re-proves the
  (previously uncommitted) xtpc cathode-rescue code path byte-identical when
  off, since the binary contains it.

## 5. 18-event reprocess — tag `cathxa`

`work/039252_<0..17>_cathxa/`, inputs symlinked from `_keep` (M11-clean),
light from `_light*_keep`.  Operating point = runner defaults (the four new
knobs ON) **plus the wfresc trial knobs** `PDVD_QL_ROBUST_TRIM=1
PDVD_QL_ROBUST_WALK_FLOOR=1 PDVD_QL_XTPC_CATHODE_TOL_CM=10`, so the first
event is directly comparable with its wfresc full-scan labels.  18/18 events
completed (batch summary ok: 18, failed: 0).

Knob-on smoke evidence (evt 0 = 298567):
- log sentinels present: `QLMatching flash_sel: admission over 8 channels ...`
  and `QLMatching reject_overpred scoped to 8 channels (R_total <= 15, R_max <= 50)`;
- flashes 198 → 192 (the cathode admission), auto-selected 113 → 107 (vs keep);
- wall XAs gone from the fit: dump `opdets[].active` = cathode + PMTs only,
  wall-XA `pred_pe` ≡ 0.

## 6. Scan-pick carryover + display (port 5020)

`flash_gid` is the flash's index in the admitted list, so the new flash cut
renumbers gids; `ql_display/remap_scan_state.py` (new) re-keys confirmed picks
onto the new dumps by flash TIME + (stable) cluster uid and writes the
viewer's `.scan_state` files into the fresh tag `work/ql_labels/cathxa/`
(never touching the old tags, M13).

- evt298567 (first event): the **full wfresc hand scan** — 44/44 picks map, 0
  lost.
- other 17 events: the keep-round **cathode-crosser + 2-boundary-track**
  confirmed picks (`decisions-{crossers,boundary}-keep`) — 425/436 unique
  picks map; the 11 lost picks sit on 8 flashes and are exactly the dim
  candles the flash admission cuts (§3.1); **none** lost to the overpred
  prefilter or any other stage (`bundle-missing = 0` everywhere).

Display: `serve_ql_scan.sh 5020 --tag cathxa work/039252_*_cathxa/calib-evt*.json`
(replaces the wfresc display that was on 5020; same SSH tunnel).  The old
wfresc/candles-keep label tags remain untouched on disk.

## 7. Caveats / follow-ups

- The 8 lost candle flashes (2.7%) are a real, quantified cost of the flash
  criteria.  If dim wall-hugger coverage matters for a future calibration,
  lower `PDVD_QL_FLASH_SEL_MINPE/_MINFIRED` (grid in §3.1) — 2 PE / 1 fired
  keeps 294/297 at 1.8% cull.
- The overpred ceilings were tuned on keep-round predictions; if QtoL or the
  efficiency scales are recalibrated, re-run
  `derive_cathode_operating_point.py`.
- The `cathxa` run inherits the **uncensused** wfresc trial knobs (robust trim
  + walk-to-floor + xtpc cathode rescue tol 10).  Their census is still open;
  reverting them does not affect the four changes documented here.
- Doc 17's own follow-ups remain open: 18-event census of the wall-XA verdict
  (this tag's rescans can provide it), cathode per-channel constants, dim
  channels {2,16,17,33} re-check with a reduced ch_mask.
- The wall XAs stay in the light *reconstruction* (OpFlash) and in the dumps
  (measured PE still recorded); the mask only removes them from the matching
  fit.  A future Xe/175 nm pass should revisit both the mask and the library.
