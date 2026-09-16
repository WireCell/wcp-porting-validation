# doc pdvd/109 — why the PDHD `stm_fit` Bee layer looked 5× busier than PDVD's, and the one-line config reason

**Status.** Question answered and fixed. The asymmetry is **not physics** — it is a display
scope that PDVD received on 2026-09-05 (doc pdvd/39 §17, owner decision) and PDHD never did.
PDHD's STM layers were still drawing the candidate pool the tagger *fitted*; PDVD's draw the
clusters it *tagged*. The PDHD config is now moved to the same round-3 scope, proven
display-only by a gate on 5 events. **No C++ change, no reconstruction change.**

## 0. Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img

# sec 1-3, 6 -- the census; read-only, from products already on disk
python3 pdvd/docs/nf_sp_img_clus/scripts/d109_stm_fit_census.py \
    > pdvd/docs/nf_sp_img_clus/figs/109_stm_fit_census.txt

# sec 4 -- the 5-event proof arm (pin = the doc-108 pin, clus md5 091e142b9481)
ARM=d109hstm DET=pdhd SRC=d108hflip JOBS=5 PIN=/home/xqian/tmp/d102/libpin_d102 \
  EVENTS="028084_3 028084_12 029107_5 029107_16 029107_18" \
  bash pdvd/docs/nf_sp_img_clus/scripts/d109_arms.sh

# sec 4 -- the display-only gate
python3 pdvd/docs/nf_sp_img_clus/scripts/d109_display_gate.py --a d108hflip --b d109hstm --det pdhd \
    > pdvd/docs/nf_sp_img_clus/figs/109_display_gate.txt

# sec 5 -- the compiled-config proof (for the 'pre' md5, restore pr.jsonnet from HEAD first)
bash pdvd/docs/nf_sp_img_clus/scripts/d102_compile_pr.sh pdhd d109post   # a870511c9b22
bash pdvd/docs/nf_sp_img_clus/scripts/d102_compile_pr.sh pdvd d109post   # 211a49a48229 (unchanged)

# sec 7 -- the two Bee sets
python3 pdvd/docs/nf_sp_img_clus/scripts/d109_bee_sets.py /home/xqian/tmp/d109/bee
```

Arms: PDHD `d108hflip` (61 evt, the flipped production config of doc pdvd/108),
PDVD `q29flip` (120 evt). Toolkit at the run: `b9ce8b4d`.

*Erratum (2026-09-16, doc pdvd/110 sec 1): the toolkit line holds for the PDHD arm only. `q29flip` ran on
2026-09-13, before the PDVD trajectory flip `8fc6070e`, so the PDVD set of sec 7 shows pre-flip trajectories
while the PDHD set shows production. The layer-scope finding is unaffected. A PDVD set on production is in
doc 110 sec 1.*

## 1. The question

Opening the same `stm_fit` layer on one PDHD event and one PDVD event, PDHD carries far more
trajectories. The owner's framing: *either PDHD accepts more STM, or PDVD accepts less; either
way there is an asymmetry worth understanding.*

Measured, the answer is **neither**. The two detectors accept STM at nearly the same rate. The
layers were showing **different populations**.

## 2. What each layer actually draws

`figs/109_stm_fit_census.txt` §1–2. Three populations are readable per event from the job log:
clusters the tagger **evaluated** (one `STM=` verdict line each), clusters it **fitted**
(`persist_stm_fit` records), and clusters it **tagged** (`STM=1`).

| | evaluated | fitted | tagged | drawn in `stm_fit` | drawn set is |
|---|---|---|---|---|---|
| PDHD `d108hflip` (61 evt) | 4041 | 1135 | 333 | **1135** | the **fitted** set |
| PDVD `q29flip` (120 evt) | 4733 | 1821 | 542 | **542** | the **tagged** set |

Not just in total — per event, `drawn == fitted` on **61/61** PDHD events and `drawn == tagged`
on **120/120** PDVD events. The identification is exact, not statistical.

Per event as displayed: PDHD 18.61 clusters / 6320 points, PDVD 4.52 / 1249 — **4.12× and
5.06×**. That is the ratio the owner saw.

**The acceptance rates are not the asymmetry.** Among the clusters each tagger fitted, the
fraction it tags is 29.3 % (PDHD) vs 29.8 % (PDVD).

## 3. The cause: a display scope that never reached PDHD

`TaggerCheckSTM::persist_stm_fit` writes the `stm_fit` cluster PC for **every evaluated main
that recorded a fit pass, whatever the verdict** (`TaggerCheckSTM.cxx:665`, unconditional on
`is_stm`). So an ungated Bee set on that PC draws the candidate pool. Restricting it is the
`require_flag` field on the Bee set (`MultiAlgBlobClustering.h:128`), a per-cluster filter.

Doc pdvd/39 §17 ("Round 3 — the STM Bee layers carry the VERDICT population", owner decision
**2026-09-05**) moved PDVD's five rows. PDHD was left on round-2 scope:

| Bee set | PDVD (since 2026-09-05) | PDHD (until this doc) |
|---|---|---|
| `stm_fit` | `require_flag:'STM'` | *(no gate)* |
| `stm` | `require_flag:'STM'` | `require_pc:'stm_fit'` |
| `steiner_graph` | `require_flag:'STM'` | `require_pc:'stm_fit'` |
| `steiner_terminals` | `require_flag:'STM'` | `require_pc:'stm_fit'` |
| `stm_tagged` | removed | present |

Round 2 was self-consistent — all four layers drew the fitted set, and a separate `stm_tagged`
layer carried the verdict. It was simply the other choice, and PDHD kept it by omission.

**The beam window is not involved.** Both detectors set `beam_window_us = [-10000, 10000]`, and
every event of both arms logs `0 out of window` on every tagger. The gate is inert by
construction; evaluating all matched pairs is the intended behavior on both detectors.

## 4. The fix and its gate

`cfg/pgrapher/experiment/pdhd/pr.jsonnet`, `bee_points_sets` only: all five rows moved to the
PDVD round-3 state. `require_flag` alone, not ANDed with `require_pc` — a cluster tagged after a
pass that recorded no fit would be hidden by `require_pc` (doc 39 §17.1). Measured on PDHD:
**0** such clusters in 61 events, so the AND is omitted for what it *would* do, not for what it
does today.

**Arm `d109hstm`**: the 5 events `028084_3/12`, `029107_5/16/18`, symlinking `d108hflip`'s
pctrees, on the doc-108 pin (`libWireCellClus.so` md5 `091e142b9481`, **unchanged before and
after**). Same binary, same input — only the compiled jsonnet differs.

`figs/109_display_gate.txt` — **GATE PASS**, four checks:

- **G1** every branch of `T_stm_michel`, `T_stm_michel_pts` (tracking-pr.root) and
  `T_stm_pass`, `T_stm_eval` (tracking-stm.root) element-wise identical;
- **G2** `calib-pr-evt*.json` identical as parsed JSON;
- **G3** every non-STM Bee layer (`clustering`, `track_fit`, `shower_track`, `vertices`, `mc`,
  the three dead-area layers) content-identical — arrays compared member-wise, never raw zip
  bytes (M2);
- **G4** the rescope itself: the new `stm_fit` equals the old one **restricted to the tagged
  clusters**, and the new `stm` equals the old `stm_tagged` on every array. That last equality
  is what makes dropping `stm_tagged` a rename rather than a loss — verified on PDHD rather
  than inherited from PDVD's measurement.

Per event, `stm_fit` before → after:

| event | points | clusters |
|---|---|---|
| 028084_3 | 5701 → 3114 | 20 → 10 |
| 028084_12 | 9422 → 2624 | 28 → 9 |
| 029107_5 | 6074 → 2162 | 20 → 11 |
| **029107_16** (the event the owner viewed) | 8418 → **1782** | 25 → **7** |
| 029107_18 | 12345 → 3091 | 31 → 9 |

## 5. Compiled-config proof

51 nodes before and after, same node key list. **Exactly one node differs** —
`MultiAlgBlobClustering/clus_pr` — in **exactly one field**, `bee_points_sets`, in exactly the
five rows of §3's table. No reconstruction component's config changes at all; this is the
structural reason G1–G2 pass. PDHD compiled md5 `87a86589c767` → `a870511c9b22`. **PDVD
compiles to `211a49a48229` either way** — the value recorded in doc pdvd/108, i.e. untouched.

## 6. What is genuinely different between the detectors

With PDHD gated, the two displays are close but not equal:

| per event | PDHD gated | PDVD | ratio |
|---|---|---|---|
| clusters in `stm_fit` | 5.46 | 4.52 | 1.21× |
| points in `stm_fit` | 1459 | 1249 | 1.17× |

So of the observed 5.06× in points, **4.33× was the missing gate** and **1.17× is real**. The
real part traces upstream of the tagger, to how many mains exist:

| per event | PDHD | PDVD |
|---|---|---|
| live clusters | 345.7 | 331.7 |
| matched flash groups | 58.1 | 47.3 |
| in-window mains | 96.8 | 60.8 |
| **mains per flash group** | **1.664** | **1.284** |
| mains per live cluster | 0.280 | 0.183 |

PDHD promotes ~1.3× more clusters to main per matched flash group. That is a QL-matching-level
statement about these manifests (61 evt / 2 runs vs 120 evt / 3 runs), **not** an attribution to
APA geometry or drift direction, which this doc does not measure.

One further contribution to the owner's impression is the specific PDVD event chosen:
`039349_2` carries **1** tagged cluster against a PDVD median of 4, so it sits well below its
own detector's typical occupancy.

The step that does differ between detectors is **fit reach**, not acceptance:
`fitted/evaluated` is 28.1 % (PDHD) vs 38.5 % (PDVD). Unexplained here; it does not reach the
display once the gate is in place, because it sits upstream of the verdict.

## 7. Bee sets for the owner

Built by `d109_bee_sets.py`; the Bee event index selects the physics event, recorded in the
`.index.txt` sidecar.

- **PDHD** (arm `d109hstm`, the rescoped config):
  <https://www.phy.bnl.gov/twister/bee/set/c3165b38-2a1d-483a-9f5b-af2d67e7ce1d/event/list/>
  — 0 `029107_16` (7 clusters), 1 `029107_5` (11), 2 `029107_18` (9), 3 `028084_3` (10),
  4 `028084_12` (9).
- **PDVD** (arm `q29flip`, already round-3):
  <https://www.phy.bnl.gov/twister/bee/set/51c1e410-d7f7-4ae2-bd84-8444a7d01c1f/event/list/>
  — 0 `039349_2` (1), 1 `039349_20` (8), 2 `039349_18` (10), 3 `039252_8` (10),
  4 `039253_6` (17).

Index 0 of each set is the event the owner viewed before the fix, so the pre-fix links remain
directly comparable.

## 8. Not concluded

- **Why PDHD's fit reach is lower** (28.1 % vs 38.5 % of evaluated mains record a pass). Not
  investigated; it is invisible in the display after this fix.
- **Why PDHD carries more mains per flash group** (1.66 vs 1.28). Measured, not explained; it
  belongs to QL matching, not to the STM tagger.
- **Whether PDHD's larger candidate pool costs anything in the verdict.** Out of scope — this
  doc changed only a display.
- The 1 PDVD event (`039349_30`) with 0 tagged clusters emits **no** `stm_fit` layer at all.
  PDHD has no such event on this manifest, but the gated layer can now be absent, and any
  consumer that assumes the member exists must tolerate that (`d109_stm_fit_census.py` does).

## 9. Files

| path | what |
|---|---|
| `scripts/d109_stm_fit_census.py` | §1–3, §6: the four measurements, read-only |
| `scripts/d109_arms.sh` | §4: subset arm runner (fork of `d102_run_arms.sh`) |
| `scripts/d109_display_gate.py` | §4: the display-only gate, G1–G4 |
| `scripts/d109_bee_sets.py` | §7: the two 5-event Bee sets (fork of `d36_build_bee_sets.py`) |
| `figs/109_stm_fit_census.txt` | census output |
| `figs/109_display_gate.txt` | gate output (PASS) |
| toolkit `cfg/pgrapher/experiment/pdhd/pr.jsonnet` | the fix, `bee_points_sets` only |
