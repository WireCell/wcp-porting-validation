# 61 — nusel hand-scan key: STM vs neutrino candidate (SBND d59k scan set)

**Status.** Scan key + sub-agent operating instructions. The whole 393-event
display set is now scanned and overlaid on **:5011**, and **:5012** serves the
93 events worth a second pair of eyes: the 62 bundles where the scan contradicts
the STM tagger (§5c) plus the 44 pathology bundles (§5d). Two separately attributable batches — first 20 events (`scan-d59k/handscan-first20.tsv`, §4) and the
remaining 383 by 10 sub-agents (`scan-d59k/handscan-batch2.tsv`, §5b). **Both are
awaiting owner validation:** every criterion below is calibrated on *two*
owner-judged bundles (evt48895, evt50787), and the ten agents agreeing 60/60 on
the seeded controls shows the criteria are applied *consistently*, not that they
are right. The set the display serves was narrowed to **393 events** in §5a (FC
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
  the verdict rests on topology alone — say so, and use verdict `nu` with
  quality `weak` when there is no vertex structure either. (`nu-weak` is **not**
  a verdict: the verdict field takes `STM` or `nu` and nothing else, §1.)
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

## 5b. Batch 2: the remaining 383 events, scanned by 10 sub-agents (UNVALIDATED)

Owner request, 2026-07-26: scan the rest of the 393-event set with 10 sub-agents,
using the first-20 batch as the worked example, and put the result in the
display. Record: **`scan-d59k/handscan-batch2.tsv`**, 402 in-beam bundles in 383
events. Overlaid on 5011 as a second `--ai-scan` file, so the two batches stay
separately attributable.

```bash
# 1. keyed skeletons + 10 slices (the join key flash_gid is NOT on screen, so
#    the agents never type it -- they return event+main_id and it is joined back)
# 2. one private viewer per slice (own --tag, ports 5062-5072), grabbers 5 at a time
python3 nusel_display/grab_scan_shots.py --port 506X --out /home/xqian/tmp/nusel-b2/shots/sN
# 3. one sub-agent per slice, images only; 4. merge + gate; 5. relaunch 5011
```
Working tree `/home/xqian/tmp/nusel-b2/` (skeletons, 1343 PNGs, per-slice
verdicts, `merge_verdicts.py`, `AGENT_INSTRUCTIONS.md`, `regrab_verified.py`).

**Result.** 125 `STM` / 277 `nu`.

| quality | STM | nu | |
|---|---|---|---|
| `clean` | 104 | 58 | |
| `cosmic-like` | — | 79 | |
| `junk` | — | 69 | 17% of the batch |
| `unclear` | 9 | 54 | |
| `weak` | 12 | 17 | |

Confidence: 208 high / 147 med / 47 low. Per-slice STM rate 25–40 %, no outlier
slice. Coverage: **393 of 393 events** of the scan set now carry ≥1 overlaid
verdict (383 from this batch, 10 from the first-20).

**The scan disagrees with the STM tagger in both directions:**

| tagger | scan | n |
|---|---|---|
| STM | STM | 107 |
| STM | nu | **43** (29 % of the 150 STM tags) |
| nu-candidate | STM | **18** |
| nu-candidate | nu | 228 |
| TGM / LM | nu | 5 / 1 (the mixed events' cosmic bundles, §5a) |

18 of the 43 overturned STM tags are the evt48895 failure mode verbatim — flat
MIP dQ/dx to rr=0, no Bragg. The 18 promotions are mostly wall entry + a clean
Bragg that the fit rejected as "long leftover" / "extra tracks" on material away
from the stopping end. **Neither number is validated; both are the owner's call.**

**Inter-agent agreement: 60/60.** Six bundles were seeded into all ten slices and
removed from the examples file the agents were given: three the criteria are
calibrated on and named in §2 (48301, 48895, 50787), two **fully blind** (48367,
52723 — cited nowhere), and 51865, which §3 does name in the pr-vs-ql point-count
table though not with a verdict. All ten agents reproduced all six batch-1 calls
on **verdict *and* quality**. Weight them accordingly: 48367 is the strongest of
the six (a `cosmic-like` call on a track the tagger tagged STM), 51865 the weakest
(19 drawn points, an obvious `junk`).

This bounds *inconsistency*, and only that. Criteria calibrated on two
owner-judged bundles could be systematically misread the same way by all ten
agents and still give 60/60 — so it says the criteria were applied uniformly, not
that they are right. Owner validation is still the gate.

**A third owner data point turned up, and it backs a promotion.** The owner's
d56bw-era label record (`work-mcp1000b-d56bw/nusel_labels/d56bw/`, carried onto
5011 by `--prev`) holds one comment: evt290201 main 9, auto `nu-candidate` —
*"probably a STM, but there is a kink ... likely large angle MCS"*. The batch-2
scan independently promoted that exact bundle to **STM/clean/high** on "enters the
top wall travelling across it (dy=98 over 115 cm), textbook Bragg to 175–193
ke/cm at rr=0, dense blob at the stopping end = Michel candidate". So the one
owner-judged case that exists in the *promotion* class agrees with the scan, and
the owner's kink is explained as large-angle MCS in a stopping muon — the reading
that also decides several §5b hard rows. One case, but it is the class most in
need of validation.

**Note on the owner's `s59k` record (M13).** Verifying the overlay clicks table
rows on 5011, and that caused the viewer to persist a *state* file for one event:
`nusel_labels/s59k/.scan_state-evt290201.json`, holding `seen`/`seeded` and a
**seeded copy of the owner's own d56bw comment above** — no verdict of this
scan's, and no existing file altered (evt48895's two files are byte-identical,
md5 checked). Left in place rather than deleted, because deleting from that tree
is exactly what M13 forbids; remove that one path if it is unwanted. Any future
overlay check should run against a private instance instead of 5011.

**Two harness bugs found and fixed, both caught by an agent refusing to invent a
verdict rather than by the gate:**

1. **`grab_scan_shots.py` could not run at all** since the IN-BEAM-only default
   (§4a): it clicked `Mode: ALL bundles`, a button that no longer exists at
   startup, so the click blocked to timeout and died before the first
   screenshot. Now conditional. `info.json` is also rewritten after every event
   so a grabber that dies mid-slice still leaves join data.
2. **A row click can silently not take.** On evt62495 and evt400174 both rows'
   screenshots show the *last* row's bundle (byte-identical PNG triples), i.e.
   the display never focused row 0. 2 of 35 multi-row events. Detect it by
   comparing the drawn payload (`bbox` + fit text) across an event's rows, not
   the table row text — that differs even when the focus is wrong.
   `nusel_display/../regrab_verified.py` (scratch) re-grabs a named
   `evt:main_id` by clicking a *different* row first and asserting the info div
   names the wanted main. Both bundles were re-grabbed and judged by the main
   session; the two overrides are in `merge_verdicts.py`, not edited into an
   agent's file. evt400174 main 8 is a textbook STM (Bragg on the muon curve to
   190 ke/cm, top-wall entry, ks1=0.018) that would otherwise have been recorded
   `nu`.
   **Corrected en route:** the first reading of evt62495 — "draws no charge at
   all in `pr` mode" — was this focus bug, not the un-merge. With verified focus
   it draws 909 points in `pr` (1026 in `ql`).

**Reconstruction pathologies the batch surfaced** (reported, not fixed — §5
rule 6). The count is of *reasons that name it*, so it is a floor:

* **dQ/dx fits returning negative charge**, 15 bundles across 8 of the 10 slices
  (56243:9, 170808:14, 175896:17, 278794:7, 280598:15, 284200:6, 284657:5,
  285443:7, 285929:13, 286655:19, 347085:4, 394532:8, 399052:15, 400504:6,
  489327:19). Excursions reach −320 ke/cm. This is a fit bug, not physics, and
  it is the most reproducible finding in the batch.
* **Whole-track dQ/dx uniformly 1.5–2× the muon table with the correct shape**,
  concentrated on cathode-hugging and isochronous bundles — a dx/normalisation
  suspicion, and the discriminator two agents had to invent (does the far end
  return to 56 ke/cm?) decides STM-vs-proton on several rows. Worth an explicit
  owner ruling: 386838:16, 389962:5 were called STM on it, 389544:13 nu.
* **Isochronous (near-constant drift-x) bundles** produce ghost fans that inflate
  the point cloud: 18 bundles name it.
* **len_main_cm ≫ the drawn extent** on 10 bundles (e.g. 291345:12, 140 cm quoted
  vs 25 cm drawn) — the un-merge/`pr` effect of §3, but large enough to mislead
  a density-based junk cut.
* **Cosmic tags that leaked into the beam window**: 169598:2 is an unmistakable
  anode→top-wall through-goer labelled TGM, 70562:10 is LM-labelled, 399382:15
  carries "TGM lm" — all in-beam bundles of the 6 `mixed` events kept by §5a's
  rule, so they are expected to be here, but 173498:6 and 280972:7 are
  two-boundary crossers with **`tgm=0`**, i.e. TGM misses.
* **Upward-going Bragg** on 4 near-bottom-wall stubs (290729:12, 291768:18,
  291301:18, 68956:13): either a real sub-population or the fit mis-assigning
  which end is the exit on short near-wall objects.

**What the owner should re-check first.** Each agent returned its own hardest
rows; the union is ~60 bundles, all with `conf` in the TSV. The 47 `low`-conf
rows and the 18 promotions are the highest-value re-checks, then the 43
overturned STM tags.

## 5c. The claimed STM mistakes, on their own display (:5012)

Owner request, 2026-07-26: *"For all these mistakes STM misidentified or missed,
can you put them into a new display port in 5012? I want to check them myself."*

**`scan-d59k/stm-disagreements.tsv`** — 62 bundles in 62 events, both batches,
one row per claimed mistake with the tagger's label, the scan verdict, quality,
confidence and the full reason:

| direction | n | meaning |
|---|---|---|
| `misidentified` | 44 | tagger tagged **STM**, scan says not a stopping muon (43 from §5b + evt48895 from §4) |
| `missed` | 18 | tagger left it **untagged**, scan says it **is** a stopping muon |

Every one is in the 393-event set, so the Bee links of §5a cover them all.
Confidence on the 18 `missed`: 2 high, 14 med, 2 low — i.e. the promotions are
the *less* certain half, and 290201:9 (one of the two `high`) is the one the
owner's own d56bw comment already backs.

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
S=$PWD
nusel_display/serve_nusel_scan.sh 5012 --tag s61mis --charge-src pr \
    --ai-scan $S/scan-d59k/handscan-first20.tsv \
    --ai-scan $S/scan-d59k/handscan-batch2.tsv \
    --prev $S/work-mcp1kall-d59k:s59k \
    $(awk -F'\t' '!/^#/ && $1!="direction" {print $3}' scan-d59k/stm-disagreements.tsv \
      | sort -un | sed "s|^|$S/work-mcp1kall-d59k/nusel_evt|; s|$|/nusel-evt&.tsv|")
#   http://localhost:5012/nusel_scan_viewer      62 events
```

Two deliberate choices:

* **Its own tag `s61mis`**, not `s59k` — two viewer instances writing the same
  `nusel_labels/<tag>/` tree would race per event file, and this list is a
  focused re-check, not the main scan record.
* **`--prev …:s59k`** seeds the owner's existing `s59k` comments into the new tag
  so nothing already recorded is invisible here. `--prev` only ever *reads* the
  source tag (§1a), so `s59k` stays untouched.

Verified: 62 events in the dropdown, the `AI scan` column live and matching the
TSVs on sampled rows (48895 `nu cosmic-like`, 290201 `STM`, 62613 `STM`, 409458
`nu weak`), the AI comment box populated, no JS errors.

**Reading order suggestion.** The `missed` 18 are the cheaper check (each is a
single claim: does it enter a boundary *and* stop with a Bragg?), and they are
where the tagger would gain efficiency. Of the `misidentified` 44, the 18 that
cite flat-MIP-to-rr=0 are the same failure the owner already confirmed on
evt48895, so a spot-check of a few decides the class.

## 5d. The pathologies are on :5012 too — 93 events

Owner request, 2026-07-26: put the reconstruction/tagger pathologies of §5b on
5012 as well. **`scan-d59k/stm-pathologies.tsv`**, 38 bundles in 4 classes:

| class | n | what to look at |
|---|---|---|
| `negative-dqdx` | 15 | the STM dQ/dx fit returns **negative** charge somewhere on the track (to −320 ke/cm). Every row whose reason names it, so it is a floor. A fit bug, not physics — the most reproducible finding in the batch |
| `dqdx-normalisation` | 15 | the **whole** track sits 1.5–2× above the muon table *with the correct shape*, mostly cathode-hugging / drift-parallel. **Needs an owner ruling** |
| `upward-going-bragg` | 5 | near-bottom-wall objects whose Bragg forces an *upward*-going muon: a real sub-population, or the fit mis-assigning which end exits |
| `tgm-miss` | 3 | both ends on boundaries (through-going) but `tgm=0`: 49951:16, 173498:6, 280972:7 |

The `dqdx-normalisation` class is the one that changes verdicts. The scan had to
invent a discriminator — *does the far end return to the 56 ke/cm MIP line?* — and
it decides STM-vs-proton: 281632:8, 289343:9, 386838:16, 389962:5, 409084:12 were
called **STM** on it, 59377:7, 61313:18, 63603:13, 72828:7, 168388:6, 174488:3,
285366:13, 291345:12, 404684:9 **nu**, and 389544:13 is the counter-case (rises
*faster* than a muon ⇒ proton). Rule on the discriminator and up to 15 rows move
together. That class is curated by hand from the reason texts, not by a regex —
the file header says so; the other three classes are mechanical.

12 of the 38 are also claimed STM mistakes (e.g. 285443:7, 285366:13, 174488:3),
so those four classes took 5012 from the 62 of §5c to 88 events. Same command as §5c
with the pathology events appended to the TSV list. Verified: 88 in the dropdown,
`AI scan` matching on a sample from every class (489327 `nu unclear`, 386838
`STM`, 291301 `nu weak`, 280972 / 173498 `nu cosmic-like`, 48895 `nu
cosmic-like`), AI comment box populated, no JS errors.

**Added on the owner's request (same day): the cosmic-tagged in-beam bundles too**
— class `cosmic-tag-in-beam`, 6 bundles, taking 5012 to **93 events**. These are
*expected*, not leaks: each is the one cosmic-tagged in-beam bundle of one of the
6 `mixed` events that §5a's keep rule retains on purpose (an event stays if *any*
in-beam bundle is keepable). Listing them lets the class be eyeballed instead of
trusted.

| event:main | tag | t (us) | len | scan |
|---|---|---|---|---|
| 70562:10 | LM | 1.724 | 8.3 | nu junk |
| 169598:2 | TGM | 0.265 | 251.8 | nu cosmic-like |
| 174928:5 | TGM | 1.358 | 48.0 | nu junk |
| 280281:8 | TGM | 1.641 | 2.4 | nu junk |
| 282952:7 | TGM | 0.206 | 5.6 | nu junk |
| 399382:15 | TGM | 1.572 | 31.4 | nu junk |

That is the **complete** class — one per mixed event, exactly the 6 `mixed` rows
of `census-nofc.tsv` — not just the 3 the agents happened to name in their
reports (399382's tag renders as "TGM lm" in the display; the label column is
`TGM`). Five of the six events were new to 5012.

Verified on 5012: 93 events, and all 12 in-beam rows of the six events match the
overlay. The check incidentally shows the structure that makes these worth a look
— every one of the six pairs a cosmic-tagged bundle with a keepable one in the
same beam window, and two of those partners are the scan's own STM calls
(169598:18 `STM unclear`, 280281:14 `STM clean`).

**GOTCHA for anyone restarting these viewers:** `pkill -f 'tag s61mis'` matches
the *launch command's own* shell, so it kills the caller before the relaunch
runs; and `pgrep -f 'bokeh serve --port 5012'` self-matches the same way. Kill the
viewer by a pid obtained without the pattern appearing in your own command line.

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
5. The criteria in §2–3 are calibrated on two owner-judged bundles, so a
   systematic misreading propagates silently over hundreds of rows. Batch 1
   stopped at 20 for validation; batch 2 (§5b, at the owner's request) covered
   the remaining 383 in one pass and bought the missing check a different way:
   **seed the same handful of already-scanned bundles into every slice**, at
   least some of them cited nowhere in the criteria and cut out of the examples
   file, then report the agreement. Do that in any future batch — it costs ~3
   bundles per agent and it is the only number that distinguishes "the criteria
   were applied" from "the criteria are right".
6. Report, do not fix. Tagger bugs (49951's missed TGM), clustering
   pathologies (52085's cathode kink) and junk clusters belong in the notes
   column, not in a code change.

## 6. Files

| path | what |
|---|---|
| `scan-d59k/handscan-first20.tsv` | the first 20 verdicts + reasons (§4); `--ai-scan` input on 5011 |
| `scan-d59k/handscan-batch2.tsv` | **the other 402 bundles / 383 events (§5b)**, 10 sub-agents; second `--ai-scan` input on 5011 |
| `scan-d59k/stm-disagreements.tsv` | the 62 claimed STM mistakes (44 misidentified + 18 missed), served on **:5012** (§5c) |
| `scan-d59k/stm-pathologies.tsv` | the 44 pathology bundles in 5 classes, also on **:5012** (§5d) |
| `scan-d59k/batch2/` | raw per-slice agent returns + the merge gate + the agents' manual; see its README (the slices still contain the seeded controls) |
| `nusel_display/regrab_verified.py` | re-grab `evt:main_id` with focus verification (§5b bug 2) |
| `nusel_display/nusel_scan_viewer.py` | `--ai-scan` overlay added (default off, §4a) + IN-BEAM-only is now the opening mode |
| `nusel_display/serve_nusel_scan.sh` | forwards `--ai-scan` |
| `nusel_display/grab_scan_shots.py` | headless evidence grabber (read-only) |
| `scan-d59k/events-nofc.txt` + `census-nofc.tsv` | **the current 393-event scan set** (§5a) |
| `scan-d59k/bee/nofc393.{url,index.txt,stmid-map.txt}` | its Bee set (166 MB zip beside them) |
| `nusel_scan_filter.py` | `--drop-fc` added (default off; old census reproduces) |
| `scan-d59k/events-withmixed.txt` | the older 648-event scan order (doc 59) |
| `work-mcp1kall-d59k/nusel_labels/c59k/` | the same verdicts as viewer labels |
| `work-mcp1kall-d59k/nusel_labels/s59k/` | **the owner's record — read only** |
