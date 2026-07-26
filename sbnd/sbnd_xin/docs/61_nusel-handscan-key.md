# 61 — nusel hand-scan key: STM vs neutrino candidate (SBND d59k scan set)

**Status.** Scan key + sub-agent operating instructions. First 20 events of the
d59k scan set are scanned (`scan-d59k/handscan-first20.tsv`, tag `c59k`) and are
**awaiting owner validation** — do not extend the scan until the owner has signed
off on those 20, because every criterion below is calibrated on two of them.
The set the display serves has since been narrowed to **393 events** (§5a: FC
candidates dropped, the recovered crash event added, new Bee link), of which 10
of the first 20 are no longer in it.

No code and no configuration is changed by this document.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# the owner's own scan viewer (tag s59k) -- READ ONLY, never drive it.
# Relaunched 2026-07-26 with the scan overlay of section 4a:
#   nusel_display/serve_nusel_scan.sh 5011 --tag s59k --charge-src pr \
#       --ai-scan scan-d59k/handscan-first20.tsv \
#       --prev <3 d56bw roots> <648 TSV paths>
#   http://localhost:5011/nusel_scan_viewer     648 events

# your own instance (pick an unused port and YOUR OWN tag):
nusel_display/serve_nusel_scan.sh 5019 --tag <yourtag> --charge-src pr \
    $(sed 's|^|'"$PWD"'/work-mcp1kall-d59k/nusel_evt|; s|$|/nusel-evt&.tsv|' \
      scan-d59k/events-withmixed.txt)     # or a subset; see §5 for the real form

# grab the per-bundle evidence images for the events you are scanning
python3 nusel_display/grab_scan_shots.py --port 5019 \
        --out /home/xqian/tmp/scanshots --events evt48301 evt48367 ...
```

`grab_scan_shots.py` writes, per in-beam bundle, `<evt>-r<row>-ctx.png` (the 3
projections at full-detector range), `<evt>-r<row>-zoom.png` (the same 3
projections auto-zoomed to that bundle's bounding box) and
`<evt>-r<row>-dqdx.png` (the STM dQ/dx panel + the tagger's fit verdict text),
plus `info.json` with every text panel and the bbox. It only navigates — it
never clicks a label button, a comment box or Save — so it writes nothing into
any `nusel_labels/` tree.

## 1. What the scan decides

The scan set is the events whose in-beam bundles are only STM-tagged or
untagged and not fully contained (doc 59 + §5a: 393 events). For **each in-beam
bundle** decide whether it is a stopping muon. The tagger's `label` column is the thing under test, not an
input to the decision.

**The verdict has two values only** (owner, 2026-07-26): the set has already had
the TGM- and LM-tagged events removed, so a bundle that is *not* a stopping muon
stays in the **neutrino-candidate** pool. "Not STM" ⇒ `nu`; there is no
`cosmic` verdict.

| verdict | meaning |
|---|---|
| `STM` | a real stopping muon: enters through a boundary, stops inside, Bragg rise |
| `nu` | still a neutrino candidate — anything that is not a stopping muon |

Everything about *why* goes in a second, separate `quality` field. It qualifies
a verdict; it never replaces one:

| quality | meaning |
|---|---|
| `clean` | the verdict is well supported by the picture and the dQ/dx |
| `weak` | plausible but featureless (single contained track, no vertex activity) |
| `cosmic-like` | not a stopping muon, but the topology reads as a cosmic (through-going, or truncated with flat MIP dQ/dx) — a nu candidate the selection will probably still have to reject |
| `junk` | the cluster is not one physical object (sparse blobs / fragments) |
| `unclear` | a reconstruction pathology dominates; the owner should look |

Only in-beam bundles matter (`in_beam=1`, i.e. `cluster_t0` in
[0.2, 2.2) us — the doc-56 gate).

## 1a. Where a scan verdict is recorded

Three places, and the distinction matters:

* **`--ai-scan` TSV** (`scan-d59k/handscan-first20.tsv`) — the canonical
  machine-readable record of a *non-owner* scan, overlaid read-only on the
  owner's display (§4a). This is where a sub-agent writes.
* **`nusel_labels/<tag>/`** — a *scanner's own* record, written only by that
  scanner's own viewer instance under its own `--tag`. The verdict goes in the
  bundle **comment**; the four label buttons are TGM/STM/FC/LM only, so tick
  `STM` when you confirm an STM and leave the buttons alone otherwise (this is
  what the owner did on evt48895: no button, comment "Not STM"). The doc-61
  verdicts are also in `nusel_labels/c59k/`.
* **`nusel_labels/s59k/`** — the owner's record. Nothing else ever writes
  there, including the overlay (M13). Note that `--prev <root>:<tag>` would
  *copy* another tag's comments into the current tag on load — that is why the
  overlay exists instead of pointing `--prev` at `c59k`.

## 2. The physics criteria

The owner's rules, as applied:

1. **STM is a stopping muon.** One end must *cross a boundary* (the entry),
   the other end must *stop inside* with a Bragg rise. It may be followed by a
   Michel electron — a short low-dQ/dx spray or hook at the stopping end.
2. **Neutrino candidates have vertex structure**: kinks, two or more prongs
   from one point, a short high-dQ/dx stub (proton) plus a long MIP prong.
3. **FC ⇒ not STM.** A fully contained cluster (`fc=1`, `stmfit=contained`)
   cannot be a stopping muon, because a muon that stops inside had to enter
   from outside. Contained + vertex structure is the neutrino signature.
4. **Grazing ≠ entering.** A track running *parallel* to a wall that touches
   it (evt50787: 6 cm of z travel over a 49 cm track at z≈497) has not entered
   through that wall. A muon entering through a wall travels *across* it.
5. **Beam direction is evidence.** SBND's beam is +z. A long prong going +z is
   ν-like; a near-vertical track (|dy| dominant) with tiny dz is cosmic-shaped.

### The decisive panel is dQ/dx, not the picture

This is the single highest-value lesson from the calibration events. The
`dqdx` panel plots the fitted dQ/dx against residual range with the **muon
stopping expectation** (green) and the **flat MIP line** (grey, 56 ke/cm =
`mip_dqdx`, doc 48) overlaid:

* **data follows the green curve up to ~165 ke/cm at rr→0 ⇒ it really stopped.**
  evt48301 (ks1=0.010) and evt50765 (ks1=0.018) are the reference pictures.
* **data stays flat on the grey MIP line right down to rr=0 ⇒ nothing stopped
  there**; the track was cut off (broken cluster, out of readout, exiting).
  evt48895 is the reference picture, and it is exactly why the owner called
  that STM tag wrong even though the tagger's `eval` PASSed.
* **data spikes well ABOVE the green curve (250–300 ke/cm) in the last few
  cm ⇒ proton-like, not a muon.** evt51051's 10 cm stub at 3.24 MIP.

A Bragg peak alone does **not** mean STM — the muon of a contained νμ
interaction stops too. STM needs Bragg **and** a boundary-crossing entry.

## 3. Reading the display without being fooled

* **Zoom before judging containment.** The green dashed FV box is inset from
  the red detector box by only 2.0 cm in x, 2.5 cm in y, 3.0 cm in z
  (`FV_BOX` vs `DET_BOX`), so at full-detector range the two lines are one
  line. The `ctx` image says *where in the TPC*; only the `zoom` image says
  *inside or outside*.
* **Orange squares are merge fragments of the main cluster**, not a second
  track and not a kink (the viewer draws them distinctly on purpose; doc 50
  records that STM fits jump gaps inside their own main). Several bundles here
  have `n_frag` 3–6.
* **`--charge-src pr` vs `ql` matters and the two disagree by construction.**
  Both zips hold the *same* points; the PR zip has them **relabelled by the
  un-merge** (doc 45). So a bundle's pre-un-merge cluster can be far bigger
  than the post-un-merge main that the taggers actually fitted:

  | event | pts in main, `ql` | pts in main, `pr` |
  |---|---|---|
  | evt50743 | 94 | 26 |
  | evt51865 | 109 | 19 |
  | evt53713 | 145 | 21 |
  | evt52613 | 652 | 115 |
  | evt48895 | 4033 | 3971 |

  Un-merged-away pieces are drawn **grey** in `pr` mode, i.e. they look like
  unrelated context. Scan in `pr` mode (it is what the fit was computed on,
  and what port 5011 serves) but re-check any bundle whose drawn point count
  is far below the table's `npts_main` in `ql` mode before calling it junk.
  `grab_scan_shots.py`'s `info.json` gives you the drawn count per bundle.
* **Contained clusters have no dQ/dx panel at all** (`stmfit=contained` ⇒ the
  STM tagger skipped the fit, so `tracking-stm.root` has nothing). For those
  the verdict rests on topology alone — say so, and prefer `nu-weak` over `nu`
  when there is no vertex structure either.
* **A slope break exactly at x=0 is suspect** (evt52085): that is the cathode,
  where two clusters get joined (`cathode_connect`) and where the ~1.5 cm
  TPC0/TPC1 offset lives. Do not score it as a physics kink.
* **Constant-z tracks are badly measured** (evt52657: dz=7 cm over 165 cm).
  That is the prolonged-W geometry where collection ROIs break, so truncation
  is the default explanation, not containment.
* **Junk has a density signature.** Real tracks here run 5–10 points/cm of
  length; the six junk bundles are all ≲3 points/cm with the points in
  disconnected blobs spread over 100+ cm. `npts_main / len_main_cm` from the
  TSV flags them before you open an image.

## 4a. Seeing a scan in the display: the `--ai-scan` overlay

`serve_nusel_scan.sh … --ai-scan FILE.tsv` (repeatable, default off) adds three
read-only things, all clearly marked as coming from the overlay file:

* an **"AI scan" column** in the bundle table — verdict (STM red / nu green)
  plus the quality chip (`unclear` amber, `junk`/`cosmic-like` grey; `clean`
  is not drawn);
* an **"AI comment (focused, read-only)" box** in the widget column, directly
  above the scanner's own *bundle comment* field, holding verdict, quality,
  confidence, the full reasoning and the source filename. It is a `Div`, not a
  text input, precisely so it cannot be typed into or saved;
* an `AI scan (<file>)` / `AI reason` pair of lines at the bottom of the
  metrics panel.

Implementation notes (`nusel_scan_viewer.py`):

* Bundles are matched by **event label + `<main_id>:<flash_gid>`** — the same
  `row_key()` the scan state uses — so a verdict lands on the right row even
  when cluster idents shift, not by row order.
* It is a **default-off knob**, gated both ways:
  * **ON** (5011, 648 events): `AI scan` column present; evt48895's verdict
    lands on the in-beam row and only that row (t=1.442, main 17, auto STM →
    `nu cosmic-like`); evt55539 (unscanned) shows a blank cell, "no AI verdict
    for this bundle" in the box and no AI lines in the metrics; no JS errors.
  * **OFF** (a spare instance on 5021, same args minus `--ai-scan`): the
    column list is byte-identical to the pre-change 15
    (`# grp t(us) PE(grp) beam main clusters npts len(cm) tgm/stm/fc[/lm]`
    `stm fit prev auto label (+FC) scan ✎`), `DataTable.width` still 980, the
    AI comment box absent from the layout, `rebuild_table` fine across an
    event step (10 → 15 rows), no JS errors. So `TABLE_FIELDS` never carries
    an `ai` column the table has no home for.
* The overlay is **never written back**. It is not `--prev`: no carry-over, no
  seeding, nothing lands in `nusel_labels/s59k/`.
* TSV contract: header line with `event main_id flash_gid verdict [quality]
  [conf] [reason]`, tab-separated, `#` comment lines skipped. `event` may be
  bare (`48895`) or prefixed (`evt48895`).

So the workflow for a batch is: scan → write/extend the TSV → relaunch the
display with `--ai-scan` → the owner reads the AI column next to the taggers'
and records their own verdict in the comment as usual.

**Also changed on the owner's request (2026-07-26): the display now opens in
IN-BEAM-only mode.** `state["inbeam_only"]` and the `mode_btn` Toggle both start
on (the toggle is constructed `active=True, button_type="warning"` with the
"IN-BEAM only" label, because `on_mode` is wired up afterwards and so fires no
callback at construction). The scan only ever judges in-beam bundles, so the
old default cost one click per event. Toggling the button still shows all
bundles. This one is *not* byte-identical by design — it is a default change,
not a knob.

## 4. Results — first 20 events (tag `c59k`, awaiting validation)

Full reasons in `scan-d59k/handscan-first20.tsv`. 20 in-beam bundles in 20
events (evt52723 also has an in-beam flash with no bundle at all, which is not
a bundle and is not scanned).

| verdict | n | events |
|---|---|---|
| `STM` | 2 | 48301, 50765 |
| `nu` | 18 | everything else |

with the qualifiers:

| quality | n | events |
|---|---|---|
| `clean` | 7 | 48301, 50303, 50765, 50787, 51051, 52723, 53361 |
| `cosmic-like` | 3 | 48367, **48895**, 49951 |
| `junk` | 6 | 50743, 50831, 50919, 51865, 52613, 53713 |
| `unclear` | 3 | 52085, 52657, 53427 |
| `weak` | 1 | 51513 |

Calibration check: the two events the owner had already judged both reproduce.
evt48895 → not a stopping muon, on "flat MIP dQ/dx to rr=0, no stopping point"
(the owner's own comment on that bundle is exactly "Not STM", nothing more);
evt50787 → nu candidate, on "vertex + stub, wall contact is grazing" (the owner
gave "several kinks"). Those are the only two owner data points there are —
everything else below is unvalidated.

Two findings worth the owner's attention:

1. **Of the 3 STM-tagged bundles, 1 (48895) is not a stopping muon** — the STM
   `eval` PASSed on a track with no Bragg peak at all. Three bundles is not a
   rate; it is a reason to keep counting STM tags separately as the scan grows.
2. **Over half of the surviving nu-candidate pool is not usable physics.** Of
   the 18 `nu` bundles: 6 `junk` + 3 `unclear` = **9/18 = 50%**, and the 3
   `cosmic-like` take it to 12/18 = 67% that a real selection still has to
   reject. Any efficiency/purity number from this scan set has to state whether
   `junk` bundles are in the denominator; `npts_main/len_main_cm` would remove
   most of them up front.

Also noted in passing (not acted on): evt49951 enters the top wall and reaches
the anode plane with two arms, i.e. it looks through-going, but `tgm=0`.

## 5a. The scan set the display now serves: 393 events (FC cut + recovered event)

Owner request, 2026-07-26: drop the **fully-contained** neutrino candidates too,
add the recovered crash event, rebuild the Bee links, re-serve 5011.

**Why the FC cut.** An `fc=1` bundle cannot be a stopping muon (§2 rule 3) and
the STM tagger does not even fit it (`stmfit=contained`), so an STM-vs-nu scan
has nothing to decide there — it is a nu candidate by construction. All 10 of
the `fc=1` bundles in the first-20 batch had no dQ/dx panel at all, which is
what made this concrete.

**Keep rule** (`nusel_scan_filter.py --drop-fc --keep-mixed`), the owner's
wording: *an event stays in if at least one of its in-beam bundles does not
satisfy the filter condition.* Formally: keep the event iff ≥1 in-beam bundle
has label ∈ {STM, nu-candidate} **and** `fc≠1`. A kept event still displays its
other in-beam bundles (6 FC bundles ride along this way) — that is deliberate,
the scanner should see the whole beam window.

**Census, 1000 events** (`scan-d59k/census-nofc.tsv`):

| verdict | n | |
|---|---|---|
| `keep` | 387 | kept |
| `mixed` | 6 | kept (a cosmic-tagged *and* a keepable in-beam bundle) |
| `all-fc` | 256 | **new** — had STM/untagged in-beam bundles, all contained |
| `tgm` | 154 | dropped |
| `lm` | 10 | dropped |
| `no-inbeam-bundle` | 187 | dropped |

→ **393 events**, `scan-d59k/events-nofc.txt`, 400 keepable in-beam bundles,
all 153 events with an STM-tagged bundle retained (an STM tag needs a point
outside the FV, so no STM tag can be lost to an FC cut — verified, not assumed).

**Regression gate on the filter rewrite:** with `--drop-fc` off the census is
identical to the committed doc-59 `scan-d59k/census.tsv` line for line, the only
difference being the added `278794 keep 8 1 STM` row; likewise
`events-withmixed.txt`. So the FC logic changed nothing about the old cut.

**Recovered event.** evt278794 (entry 618) aborted during the doc-59 run;
`2a821fd2` (doc 60) fixed it. Re-run from its saved pctree — rc=0 in 2 s,
`assoc=4/28`, in-beam bundle main 7 at t=0.695 us, STM-tagged, `fc=0` → in the
set. Freshness proof first: `local/lib/libWireCellClus.so` (11:50) newer than
every `clus/` source, worktree clean at `2a821fd2`, and the only code commit
since the 999-event run is that guard (doc 60: 80/80 byte-identical), so its
table is comparable with the rest.

**Bee, all 393 events, one set** (166 MB):

> <https://www.phy.bnl.gov/twister/bee/set/b8d23796-23ab-42a5-85f3-1b0ae492a276/event/list/>

Layers per event: `img-global`, `clustering-global`, `op` (the Q/L match),
`channel-deadarea-*`, and `stm_fit-global` where a fit exists. Bee index *i* =
line *i+1* of `scan-d59k/bee/nofc393.index.txt`; PR→img cluster-id map in
`nofc393.stmid-map.txt`. Verified HTTP 200 with 393 distinct `/event/<i>/`
links. **GOTCHA:** plain `curl`/`urllib` cannot verify a Bee URL from this shell
(no CA bundle: `CERTIFICATE_VERIFY_FAILED`); use `curl -k`, which is what
`upload-to-bee.sh` itself does.

**5011 now serves those 393 events** — same `--tag s59k` (so the owner's
existing evt48895 comment is still there), `--charge-src pr`, the same three
d56bw `--prev` baselines, `--ai-scan scan-d59k/handscan-first20.tsv`, opening in
IN-BEAM-only mode. Verified: 393 in the dropdown, evt278794 present, AI column
live, sampled in-beam rows all `fc=0`, no JS errors. Ten of the first-20 batch
were FC and are therefore no longer in the display; their verdicts stay in the
TSV and in `nusel_labels/c59k/`.

**The FC cut demonstrably removes the low-information events.** Of the first-20
batch, the 10 it dropped are 4 of the 6 `junk`, **all 3** `unclear`, the 1 `weak`
and 2 `clean`; the 10 it kept are both `STM`, all 3 `cosmic-like` (the STM tags
and near-misses that the scan is actually about) and 5 more. §4's "half the nu
pool is junk or unclear" was measured before this cut and should be re-measured
on the new set — on the surviving 10 it is 2 `junk` and no `unclear`.

## 5. Instructions for a sub-agent extending this scan

1. **Never write into `work-mcp1kall-d59k/nusel_labels/s59k/`.** That is the
   owner's live scan record (M13). Serve your own viewer with your own
   `--tag`, on your own port. Do not drive the process on 5011.
2. Take the event list from **`scan-d59k/events-nofc.txt`** (393 events, the
   current port-5011 order and the current Bee set; `events-withmixed.txt` is
   the older 648-event cut before the FC filter, §5a). The viewer takes explicit
   TSV paths, so a subset needs no symlink farm:
   ```bash
   R=$PWD/work-mcp1kall-d59k
   for e in $(sed -n '21,60p' scan-d59k/events-nofc.txt); do
       echo $R/nusel_evt$e/nusel-evt$e.tsv; done > /home/xqian/tmp/tsvs.txt
   nusel_display/serve_nusel_scan.sh 5019 --tag <yourtag> --charge-src pr \
       $(cat /home/xqian/tmp/tsvs.txt)
   ```
   Use `/home/xqian/tmp/`, never `/tmp` (M16).
3. Grab evidence with `grab_scan_shots.py` (§Repro), then judge from the
   images: `zoom` for topology, `ctx` for position/containment, `dqdx` for
   stopping vs MIP. Do not decide from the `tgm/stm/fc/stmfit` columns — those
   are what the scan is testing. Use `npts_main/len_main_cm` only to
   pre-flag junk, and confirm it in the image.
4. Record every bundle with a one-line reason naming the *evidence*
   ("flat MIP to rr=0", "vertex + 3 cm stub", "1.4 pts/cm in 8 blobs"), in the
   same TSV shape as `scan-d59k/handscan-first20.tsv` — one file per batch,
   `handscan-<range>.tsv`. **`verdict` is `STM` or `nu`, nothing else**; put
   every other observation in `quality` (§1). An unknown verdict still
   displays, but in grey with no styling — that is a sign you invented one.
   Then hand the owner the relaunch line with every batch TSV passed as its own
   `--ai-scan` (§4a) so all batches show at once.
5. Scan in batches of ~20 and stop for validation. The criteria in §2–3 are
   calibrated on three events only; a systematic misreading would otherwise
   propagate over hundreds.
6. Report, do not fix. Tagger bugs (49951's missed TGM), clustering
   pathologies (52085's cathode kink) and junk clusters belong in the notes
   column, not in a code change.

## 6. Files

| path | what |
|---|---|
| `scan-d59k/handscan-first20.tsv` | the 20 verdicts + reasons (this round); the `--ai-scan` input now overlaid on 5011 |
| `nusel_display/nusel_scan_viewer.py` | `--ai-scan` overlay added (default off, §4a) + IN-BEAM-only is now the opening mode |
| `nusel_display/serve_nusel_scan.sh` | forwards `--ai-scan` |
| `nusel_display/grab_scan_shots.py` | headless evidence grabber (read-only) |
| `scan-d59k/events-nofc.txt` + `census-nofc.tsv` | **the current 393-event scan set** (§5a) |
| `scan-d59k/bee/nofc393.{url,index.txt,stmid-map.txt}` | its Bee set (166 MB zip beside them) |
| `nusel_scan_filter.py` | `--drop-fc` added (default off; old census reproduces) |
| `scan-d59k/events-withmixed.txt` | the older 648-event scan order (doc 59) |
| `work-mcp1kall-d59k/nusel_labels/c59k/` | the same verdicts as viewer labels |
| `work-mcp1kall-d59k/nusel_labels/s59k/` | **the owner's record — read only** |
