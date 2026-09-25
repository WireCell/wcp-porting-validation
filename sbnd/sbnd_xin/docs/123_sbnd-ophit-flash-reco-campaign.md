# doc 123 — SBND flashes from OpHits: investigation and campaign design

2026-09-24.

**Status (2026-09-25): FLIPPED.** `flash_source=hits` is the standalone chain's production default
(§17, `ref/prod-2026-09-25`), the cathode rescue stays ON. §0–§9 are the 09-24 design as written;
§10–§16 the campaign that demonstrated it; §17 the flip and the follow-ups it leaves; §18 (round 6) the
`QLXTPC coincident` cull understood and fixed behind two knobs, and the rescue ruled on MC; §19 the second
flip: the light gate + ceiling are production (`ref/prod-2026-09-25b`), rescue still ON.

**Question.** SBND Q/L matching uses the reco1 `recob::OpFlash`. That flash merges or drops
flashes that are a few µs apart. Can we rebuild the flashes from the reco1 `recob::OpHit`,
the way the toolkit light reco does for PDHD/PDVD? If so, how should the new flashes enter
Q/L matching, given the geometric patches we already added to save clusters?

## Repro block

The first-hand numbers in §1 and §6 come from:

```sh
cd sbnd_xin
S=scripts/analysis/light
D2=/nfs/data/1/xning/wirecell-working/SBND/data/reco1_upstream/data_reco1      # mcp2k (symlinks into yuhw's area)
cut -d: -f1 docs/123_flash/123_rescue_census_detail.txt > ~/tmp/d123_events40.txt

# OpHits and OpFlashes for the 40 rescue-fired events, bare ROOT with the wire-cell-sbnd-reco1 mirror dictionary
root -l -b -q "$S/d123_ophit_probe.C(\"input_files_reco1/data_MCP2025C_reco1_frameshift_first1000ev.root\",\"$HOME/tmp/d123_events40.txt\",\"$HOME/tmp/mcp1k.tsv\",-20,30)"
for p in 1 2; do
  root -l -b -q "$S/d123_ophit_probe.C(\"$D2/data_MCP2025C_reco1_frameshift_2nd1k_part$p.root\",\"$HOME/tmp/d123_events40.txt\",\"$HOME/tmp/mcp2k_p$p.tsv\",-20,30)"
done

python3 $S/d123_rescue_vs_hits.py ~/tmp/{mcp1k,mcp2k_p1,mcp2k_p2}.tsv docs/123_flash/123_rescue_census_detail.txt \
    > docs/123_flash/123_rescue_vs_hits.tsv
python3 $S/d123_summary.py docs/123_flash/123_rescue_vs_hits.tsv ~/tmp/{mcp1k,mcp2k_p1,mcp2k_p2}.tsv \
    > docs/123_flash/123_rescue_onsets.tsv
python3 $S/d123_null.py /nfs/data/1/xning/wirecell-working/SBND/light/flash_from_hits/config/sbnd-pmt-channels.json \
    docs/123_flash/123_rescue_onsets.tsv ~/tmp/{mcp1k,mcp2k_p1,mcp2k_p2}.tsv > docs/123_flash/123_null_onsets.tsv
```

The probe scans `EventAuxiliary` over every entry; each 4 GB mcp2k file takes a few minutes.
All 40 events were found: 11 in mcp1k, 20 in mcp2k part1 and 9 in mcp2k part2.

**Where each claim comes from.** Every claim below carries one of these tags:

- **[V]** — I checked it first-hand this session: I ran the command or read the source line.
- **[C]** — xning reports it in `/nfs/data/1/xning/wirecell-working/SBND/light/*.md`. I have
  not re-run it.
- **[S]** — read from source code or fcl by the investigation (file and line given), without
  running anything.
- **[D]** — from an earlier doc in this tree (doc number given).

---

## 0. Answers in brief

| # | question | answer |
|---|---|---|
| 1 | Do reco1 files carry OpHits? | **Yes.** `recob::OpHits_ophitpmt__Reco1.` and the `OpFlash↔OpHit` Assns are in every SBND reco1 sample we use. PMT hits only: no waveforms, no X-ARAPUCA. **[V]** |
| 2 | How does LArSoft build OpFlash from OpHit? | `SBNDFlashFinder` + `SimpleFlashAlgo`, run separately per TPC. It takes the brightest 10 ns seed first, integrates **8 µs**, then vetoes **±8 µs** around it. So two flashes less than 8 µs apart cannot both survive in one TPC. **[V]** for fcl and code lines. |
| 3 | How does the toolkit separate close flashes? | `flash/OpFlashFinder` is a larana `OpFlashAlg` port: 1 µs dual accumulators, then a per-hit **width-tolerance refinement** (the step that separates), then late-light removal. For SBND, xning's `SBNDOpFlashFinder` adds a **hit-level pulse split**. **[S]**, **[C]** |
| 4 | Design | Use xning's `SBNDOpFlashFinder`, fed by an OpHit reader in `wire-cell-sbnd-reco1`. It writes the same `[nflash, 313]` opflash tensor, so it is a drop-in for the existing reader. Bring it in behind a default-OFF knob (§4). |
| 5 | Integration with Q/L and the geometric patches | Keep the cathode rescue ON while the new flashes are evaluated. Measure how often it still fires; that residual measures the inefficiency the new flashes leave behind. Only then decide whether to retire rescue knobs (§5). **The production path runs inside LArSoft and has no OpHit source: this is an owner decision (§5.1).** |
| 6 | Close-flash examples | The cathode-rescue census has 40 events and 41 moves. 22 moves have Δt0 inside the 8 µs veto; in **21 of them** the TPC without a reco1 flash has a clear hit-level pulse at that time (≥100 PE, ≥3× the preceding 200 ns). The same rule fires on **0 of 166** null probes 0.8–1.5 µs away. Three more moves (72759, 78242, 317427) have Δt0 outside the veto but still show the veto signature. The OpHits hold what reco1 threw away. **[V]** (§6) |

---

## 1. Reco1 files contain the OpHits

Branches, listed with uproot from the `Events` tree **[V]**:

```
recob::OpHits_ophitpmt__Reco1.
recob::OpFlashs_opflashtpc0__Reco1.            recob::OpFlashs_opflashtpc1__Reco1.
recob::OpFlashrecob::OpHitvoidart::Assns_opflashtpc{0,1}__Reco1.
```

| sample (our tag) | file | entries | OpHit branch |
|---|---|---|---|
| nueCC48 / fsprod | `input_files_reco1/data_filtered_decoded_reco1-fe6033f3-…_eventidfiltered_frameshift.root` | 48 | yes **[V]** |
| mcp1k | `input_files_reco1/data_MCP2025C_reco1_frameshift_first1000ev.root` → yuhw `round2-patrec/` | 1000 | yes **[V]** |
| mcp2k | `…/data_MCP2025C_reco1_frameshift_2nd1k_part{1,2}.root` (doc 102 sample table; xning's `data_reco1/` links to the same files) | 2×1000 | yes **[V]** (xning also checked) |
| MC | xning `reco1_upstream/mc_reco1/reco1-detsim-g4-gen-Gen2_2026-*.root`, 10 files | 104 | yes **[C]** (branch list), not re-checked here |

The `.obj` is stored unsplit, so uproot cannot read the hits. Bare ROOT with the mirror
dictionary already in `wire-cell-sbnd-reco1` can. That dictionary has carried
`recob::OpHit` v15 since the initial import (`inc/WireCellRoot/SBNDReco1Products.h:94`,
`dict/LinkDef.h:22-44`). `d123_ophit_probe.C` reads the hits **[V]**: 12.7–53.5 k hits per
event on the 40 events (for example 28 657 on evt 288952). Its field is
`t = fStartTime + fRiseTime`, the same input time `SBNDFlashFinder` uses
(`OpHitInputTime "RiseTime"`).

**What the hits are.** Most are single photoelectrons: the median width is 0.034 µs **[C]**.
The exceptions are the brightest PMTs of a large flash, where one hit can be up to 5.8 µs wide
and hold thousands of PE **[C]**.

Upstream they come from Wiener deconvolution (`opdecopmt`) followed by larana
`SlidingWindow` hit finding, with `SPEArea 200`, `HitThreshold 1` and per-channel ADC
thresholds **[S]** (`sbndcode/OpDetReco/OpDeconvolution/job/sbnd_ophitfinder_deco.fcl:72-105`).
23 PMT channels are masked in `ophit_finder_sbnd.fcl:192` **[S]**.

We can re-find flashes from the hits. We cannot re-find the hits themselves: there are no
waveforms in reco1.

## 2. How LArSoft turns OpHits into the OpFlash we use

The chain is `ophitpmt` → `opflashtpc0` and `opflashtpc1`. The module is `SBNDFlashFinder`
and the algorithm is `SimpleFlashAlgo` (the only one registered). Data: `reco1_data.fcl:25-32`.
MC: `workflow_reco1.fcl:21-32` **[S]**.

Parameters, from `sbndcode/OpDetReco/OpFlash/job/sbnd_flashalgo.fcl:5-10` **[V]**:

```
PEThreshold 20   MinPECoinc 6   MinMultCoinc 3   IntegralTime 8.   PreSample 0.1   VetoSize 8.   (µs)
```

**How `SimpleFlashAlgo` builds flashes** (`FlashFinder/SimpleFlashAlgo.cxx`):

1. **Per TPC.** The PMTs are split by OpDet x < 0 or x > 0, and each TPC is clustered on its
   own. The two collections are never reconciled with each other **[S]**.
2. **Seeds.** Hits are put in 10 ns bins. A bin is a seed if it has ≥ 6 PE and ≥ 3 hits.
   Seeds are processed **brightest single bin first**; the map is keyed by `1/pe` **[S]**.
3. **Veto.** A candidate whose start is within ±8 µs of an already accepted flash is skipped.
   These are the two `skip=true` branches at about L224 and L228 **[V]**.
4. **Accept.** The flash integrates PE and hits over `[seed − 0.1 µs, seed + 7.9 µs)` and is
   kept if that sum is ≥ 20 PE.
5. **Time.** The time is the seed bin. `SBNDFlashFinder_module.cc:~164-178` then refines it
   with `FlashT0SelectedChannels` and subtracts a light-propagation time taken from a PMT-ratio
   drift estimate **[V]**.

**So two real flashes Δt < 8 µs apart in one TPC become one flash** **[S]**:

- **The dimmer one comes later.** It is vetoed, and its PE falls inside the brighter flash's
  window. Its PE and per-PMT pattern are added to the brighter flash.
- **The dimmer one comes earlier.** It is vetoed, and its light is before the other flash's
  `−0.1 µs` pre-sample, so it is in no flash at all.

Because each TPC keeps whichever flash is brighter *in that TPC*, the two TPCs can keep
different members of the pair. This is the evt 59415 signature in
`toolkit/match/docs/sbnd-opdetreco-chain.md` §"Known failure mode": the pair is 4.79 µs apart
and each flash is missing on the other side. This doc agrees with that analysis and does not
redo it.

**Two more reco1 defects xning found.** Any comparison has to keep these separate from the
close-flash effect (§7 R2):

- **The 74 µs time bug** **[C]**. When the drift estimate fails it returns −999999. The
  propagation correction then shifts `fTime` 74.06 µs late. This affects 9.9 % of MC flashes
  and 0.59 % of data flashes, almost all small. On 14 MC events it moved 13 of 204 Q/L pairs.
  (`FLASH_DRIFT_POSITION.md`, `QL_MATCHING_IMPACT.md`.)
- **Dropped bright flashes** **[C]**. On a bright PMT, one giant hit can fail the ≥ 3-hits
  seed rule. A later, dimmer flash then vetoes the bright one. About 97 k PE was lost over 13
  MC events. xning's Python emulator of `SimpleFlashAlgo` reproduces all 449 MC and 94 data
  reco1 flashes exactly (`FLASH_FROM_HITS.md` §"SBND loses some bright flashes").

**Could LArSoft itself do better?** Two options exist, and neither is attractive **[S]**:

- **Shrink `VetoSize` and `IntegralTime` to 2–3 µs.** Every flash's PE integration changes,
  and `SimpleFlashAlgo` has no late-light rejection, so a bright flash's slow tail would seed
  false flashes.
- **Use larana `OpFlashFinder`.** It is configured (`opticaldetectormodules_sbnd.fcl:250`) but
  commented out of reco1. It has no per-TPC split, no t0 tool and no propagation tool, and it
  is not tuned for deconvolved SBND hits.

Either option is a change to the SBND production reco1, which we do not own. Hence the
toolkit route.

## 3. How the toolkit separates close flashes (PDHD/PDVD), and xning's SBND version

**The PDHD/PDVD chain** **[S]**, documented in `toolkit/flash/docs/design.md` and
`stage2-reconstruction.md`:

```
OpWaveformSource → OpDecon → OpHitFinder → [OpHitMerge] → OpFlashFinder → TensorFileSink
                                                          "opflash" [nflash, 1+nchan] + "flash_summary" + "ophits"
```

`flash/src/OpFlashFinder.cxx` ports larana `OpFlashAlg::RunFlashFinder`:

1. **Two accumulators.** The bins are 1 µs wide (`bin_width`), and the second set is offset
   by half a bin. A bin becomes a candidate when it crosses `flash_threshold`.
2. **Claim.** Candidates are processed largest first, and each claims the hits no other
   candidate has taken.
3. **Width refinement** (cxx:77-137). Inside each claimed group, a sub-flash grows from the
   largest hit. A hit joins it if `|t_hit − t_flash| ≤ 0.5·(half-width_hit + half-width_flash)`.
   **This is the step that separates two flashes inside one bin.**
4. **Late-light removal.** A later flash j is dropped if
   `(PE_j − hyp)/√hyp < 3`, with `hyp = PE_i·(w_j/w_i)·e^{−Δt/1.6 µs}`.
5. **Optional re-merges.** `flash_refine` (PDHD) and `flash_tail_merge` (PDVD production)
   merge back over-split tails.

There is no per-TPC veto. Flash time is the PE-weighted mean of hit peak times. The output
tensor feeds `aux/src/FlashTensorToOpticalPCs.cxx`, which feeds `QLMatching`.

**The SBND version: `SBNDOpFlashFinder`** **[C]**, plus a code read.

- **Where it lives.** All of it is **uncommitted** in xning's trees:
  - toolkit: `/nfs/data/1/xning/wirecell-working/toolkit/flash/{inc/WireCellFlash/SBNDOpFlashFinder.h, src/SBNDOpFlashFinder.cxx, test/doctest_sbndopflashfinder.cxx}`,
    on `apply-pointcloud` at b5a897f6;
  - reader: `SBNDReco1OpHitSource` and `SBNDReco1CafOffset.h`, plus edits to
    `SBNDReco1OpFlashSource.cxx`, in xning's `wire-cell-sbnd-reco1`;
  - jobs: `SBND/light/flash_from_hits/wct-reco1-flash-from-hits.jsonnet` and
    `SBND/light/wct-ql-from-hits.jsonnet`.
- **The algorithm, in order:**
  1. Two accumulators of 8 µs bins, offset by half a bin, 20 PE threshold. This deliberately
     reproduces SBND's grouping scale.
  2. Claim, as in the toolkit.
  3. **Pulse split** inside a claimed group. The group is scanned in 10 ns bins. A later
     spike of ≥ max(30 PE, 1 %) cuts the group when all of these hold:
     - it is ≥ 0.5 µs after the onset;
     - its 100 ns burst, with the brightest OpDet removed, is ≥ max(100 PE, 5 % of the
       onset burst);
     - it comes after a dip: the 500 ns before it hold ≤ 50 % of the burst.
  4. **Join**: over-split pieces are merged back.
  5. Late-light removal, as larana does it.
  6. At least 3 fired PDs.
  7. **Time**: SBND's prompt rule (top hits within −20/+10 ns of the brightest 10 ns bin,
     until 60 % of its PE).
- **What it leaves out.** There is no X estimate and no light-propagation correction, so the
  74 µs bug cannot happen, but times sit about 5–8 ns off reco1.
- **Output.** The same `opflash [nflash, 313]` tensor as `SBNDReco1OpFlashSource`, so
  `TensorFileSource → FlashTensorToOpticalPCs → QLMatching` reads it unchanged.

**xning's results** **[C]**, not re-run here:

| | MC (13 evt) | data (63 evt) |
|---|---|---|
| reco1 flashes recovered | 449/449 | 1782/1782 |
| one-to-one | 97 % | 91 % |
| PE ratio, median | 1.000 | 1.005 |
| extra flashes | 142 (median 36 PE) | **723 (median 51 PE)** |
| MC truth: flashes still holding a second pulse ≥ 20 PE | 13/589 (reco1: 17/405) | — |
| Q/L, MC: same flash as bug-fixed reco1 | 163/185 clusters | not run |

Their open items: no tail subtraction after a split, so the later piece carries 1.5–2× its
true PE; small second pulses on a bright tail are still merged; no Q/L on data yet.

## 4. Design: a new SBND flash reco

**Principle.** Reproduce reco1 first, then change one thing at a time. The new source must be
a drop-in producer of the existing `opflash_apa<N>` tensor sets, so nothing downstream of
`TensorFileSource` changes.

```
SBND reco1 art file
  └─ SBNDReco1OpHitSource  (wire-cell-sbnd-reco1, per TPC: channels = even/odd PMT OpChannels)
       └─ SBNDOpFlashFinder (toolkit flash/)  → opflash [nflash,313] (+ flash_summary, ophits with flash id)
            └─ md: run/subrun/event + frame_apply_at_caf       ← MUST be carried (see below)
                 └─ TensorFileSink opflash_apa{0,1}.tar.gz  ─→ unchanged run_ql_evt.sh / wct-clus-matching-perevt.jsonnet
```

- **Where the code goes.**
  - The OpHit reader belongs in `wire-cell-sbnd-reco1`, **not** toolkit `root/`. Its mirror
    dictionary clashes with LArSoft's (`wire-cell-sbnd-reco1/docs/DESIGN.md`, WCT issue #494).
  - `SBNDOpFlashFinder` in toolkit `flash/` is allowed. It depends only on `flash` and `aux`
    and adds no external dependency.
- **The flash-time offset is an acceptance item.**
  - Today `SBNDReco1OpFlashSource` writes `frame_apply_at_caf` into the tensor-set metadata,
    and `FlashTensorToOpticalPCs` adds it to every flash time (doc 21). It is 0.25–2.7 µs per
    event.
  - The new chain must carry it through the hit tensor into the flash tensor. xning's finder
    writes `offset_us` only to metadata and does not apply it. Check whether it passes
    `frame_apply_at_caf` through.
  - If the offset is dropped, every flash moves out of the 0.3–1.9 µs beam window and out of
    the rescue's 0.2–2.2 µs cut, with no warning.
  - Test: on the 48-event fsprod sample, at least 45 of 48 in-time flashes must sit in
    +0.3…1.9 µs, the doc 21 reference.
- **Per-TPC split.** Keep one flash collection per TPC, as reco1 does, so APA ports, gid
  offsets (+1000000 for APA1) and `flash_group_window` (80 ns) keep their meaning.
  - PMT parity: even OpChannel = TPC0, odd = TPC1. This comes from xning's
    `sbnd-pmt-channels.json` (60 + 60 PMTs), and the probe uses it. **[V]**: all 97 channels
    with hits in the 40 dumped events (179 932 hits; 120 PMTs minus the 23 masked) are in
    that list, 50 even in `tpc0` and 47 odd in `tpc1`, with no violations (`d123_null.py`).
  - A joint two-TPC finder, which would fix the cross-TPC disagreement in one step, is an
    option for a later round, not round 0.
- **Knob.** A default-OFF switch in the SBND job selects "reco1 OpFlash" (today) or "hit
  flashes" (new). Use key suppression so the compiled production config is byte-identical
  when off (§2 of the operating manual).

## 5. Integration with Q/L matching and the geometric patches

**What exists today (ON in SBND production):**

- **`ClusteringCathodeBundleRescue`** (`clus/src/clustering_cathode_bundle_rescue.cxx`; docs
  pr/14, pr/17, 73) **[D]**.
  - It takes a beam-window cluster with a tip within 5 cm of the cathode. It pairs it with a
    cluster on the other side that sits on a different flash with
    **Δt0 ∈ [−8, +13] µs**, which is essentially the reco1 veto. It also pairs it with a
    cluster that has no flash (`rescue_unmatched`).
  - It merges the pair, then an a/b/c/d length rule chooses which flash the merged object
    takes.
  - Round 2 and 3 knobs (`rescue_geom_first`, `rescue_pierce_test`, `rescue_allow_in_beam_far`, …)
    extend it to wrong-flash cases +28…+855 µs away.
- **`nu_bundle_flash_group`** (doc 109 r4) **[D]**. It merges neutrino bundles whose two TPC
  flashes fall in one 80 ns group and whose charge touches within 20 cm. Both flashes exist
  here, so it is not a flash-inefficiency patch.
- **`QLMatching` cross-TPC checks** **[D]**. `xtpc_flag` and `flash_group_window` 80 ns only
  pair flashes whose times agree to 80 ns. A pair split by the veto is never looked at as a
  pair (`match/docs/qlmatching-evt59415-xtpc-flash-split.md`).

**How the new flashes should meet these patches:**

1. **Rescue stays ON first.** Run the new flashes with production rescue on.
   - Count how often rescue still fires, and on which moves. Once hit flashes restore the
     partner flash, the 22 within-veto moves (§6) should stop firing: both halves find
     time-coincident flashes and `xtpc`/grouping pairs them.
   - Whatever still fires measures the inefficiency the new finder leaves.
2. **Only then try rescue OFF**, or retire individual rescue knobs, on the same events. The
   moves that are *not* flash losses must be checked separately, because they belong to the
   patch's other duties:
   - same-time moves;
   - 8–13 µs moves;
   - geom-first moves beyond 13 µs.
3. **Watch the extra small flashes.** In data, xning's finder adds about 11 flashes per event
   **[C]**, with a median of 51 PE, right at `flash_minPE` 50. That means many new candidates
   in the global LASSO fit. Measure how many clusters move to a < 100 PE flash before trusting
   any efficiency gain.
4. **Beam-flash PE changes.** Once a merged partner is split off, the beam flash loses the
   partner's PE. Q/L scores (KS, χ², over-prediction veto) then compare against a different
   PE pattern. Split-off tails also over-count by 1.5–2× **[C]**. Expect movers even on
   clean events. Adjudicate them by hand scan, not by the matched count.

### 5.1 The owner decision that gates production

SBND production runs Wire-Cell **inside LArSoft**:
`cfg/pgrapher/experiment/sbnd/wcls-img-clus-matching-xin.jsonnet:66-79` reads
`recob::OpFlash` through `wclsOpFlashSource`. **No wcls OpHit source exists.** I did not find
one in the local sbndcode, or in larwirecell v10_04_03 on cvmfs.

The standalone reco1 path is enough for development and for this whole campaign. To use hit
flashes in production, one of these is needed:

- **(a)** a new larwirecell OpHit source (`recob::OpHit` → the `ophits` tensor), with the same
  finder running in the wcls graph;
- **(b)** an sbndcode-side change: new reco1 flash settings, or a new flash module, which SBND
  owns;
- **(c)** keep hit flashes as a standalone-only improvement.

This does not block rounds R0–R4 below, but it decides where the campaign ends.

## 6. Close-flash examples from the cathode-rescue census

**Where the cases come from.** The census ran with all rescue knobs on over mcp1k + mcp2k
(3000 events, doc 73 / cbr3). It fired on 40 events and 41 moves; 56463 has both a rescue and
an unmatched-rescue move.

- The per-move log is now committed as `docs/123_flash/123_rescue_census_detail.txt`. It was
  only in `~/tmp/cbr3/`, which cleanup rounds sweep.
- For each move, the probe found the two reco1 flashes the halves were matched to, using the
  rescue's Δt0, which does not depend on the flash-time offset. The 39 two-flash moves were all
  matched this way within 30 ns. The 2 unmatched moves have no far flash, so for those the
  probe takes the near-side in-time flash with the most PE.

**The test.** For each move, look at each TPC at the time where it has no reco1 flash:

- A = far TPC at the near (beam-side) time;
- B = near TPC at the far time.

Sum the hit PE in `[t − 50, t + 150] ns` ("prompt") and in `[t − 350, t − 150] ns` ("pre",
same width). An **onset** is prompt ≥ 100 PE and prompt ≥ 3 × pre, i.e. a new pulse, not the
tail of an earlier flash.

Full tables: `docs/123_flash/123_rescue_vs_hits.tsv` and `123_rescue_onsets.tsv`. **[V]**

| class (by Δt0) | moves | reading |
|---|---|---|
| **within the 8 µs veto**, 0.05 < \|Δt0\| ≤ 8 µs | **22** | The veto signature: the near TPC's nearest reco1 flash to the far time is always the beam-side flash itself, at −Δt0 (22/22). The far TPC's nearest flash to the beam time is the partner at +Δt0 in 20/22; in 169824 and 288952 it is another flash 3.6 and 2.1 µs away. **21/22 show a hit-level onset** in a flashless TPC. The exception, 173450, has A = 301 PE with pre = 239. **These are the target of the new finder.** |
| far half unmatched (`rescue_unmatched`) | 2 | 50801: A = 110 PE (pre 4). 56463: A = 20 996 PE (pre 3), yet the far TPC's nearest reco1 flash is 4.56 µs away. Both are onsets, so both are hit-recoverable. |
| same time, \|Δt0\| < 0.05 µs | 8 | Both flashes exist in both TPCs. The rescue is doing cathode-crossing charge merging, not recovering light. Expected to keep firing. |
| 8–13 µs | 4 | In 164576, 397498 and 407798 both TPCs have a reco1 flash at both times, within 26 ns: a Q/L choice problem, not light loss. **72759 is a veto loss:** the far TPC's nearest reco1 flash to the beam time is 0.60 µs away, and there is a 124 PE onset (pre 7) at the beam time. |
| geom-first, > 13 µs | 5 | 65053, 78242, 281165, 317427, 319913. Three have a far-TPC reco1 flash at the beam time (within 0.1 µs; 65053's is 92 ns off, *outside* the 80 ns grouping window). **78242 and 317427 have no far-TPC beam flash:** the nearest is 2.05 and −2.39 µs away, with 3384 and 2855 PE onsets at the beam time. They are veto losses that Q/L then routed to a wrong flash hundreds of µs away. |

**Examples to use** (all run 18255/18259, data). *Missing* means the TPC and time where the
onset has no reco1 flash:

| evt | sample, entry | Δt0 (µs) | missing | onset PE (pre) | note |
|---|---|---|---|---|---|
| 56463 | mcp1k 599 | +4.56 | far TPC1 @ beam | 20 996 (3) | also the pr/17 founding event; the biggest loss |
| 78369 | mcp2k_p1 490 | +3.01 | near TPC0 @ far t | 23 648 (63) | beam flash 95 k PE absorbed the partner |
| 82608 | mcp2k_p1 645 | +3.79 | near TPC0 @ far t | 13 570 (13) | |
| 395148 | mcp1k 472 | −1.46 | near TPC0 @ far t | 16 399 (2) | earlier partner, so its light is **lost**, not absorbed |
| 169824 | mcp1k 809 | −4.47 | both TPCs | 3481 / 12 319 | each TPC kept a different member: the 59415 pattern |
| 179369 | mcp2k_p1 140 | +1.67 | both | 8277 / 2962 | |
| 288952 | mcp1k 227 | −6.50 | both | 1653 / 245 | pr/14 hand-scanned move |
| 392200 | mcp1k 453 | +2.43 | both | 140 / 2593 | pr/14 hand-scanned move |
| 78242 | mcp2k_p1 697 | +855 (geom) | far TPC1 @ beam | 3384 (0) | veto loss hidden behind geom-first |
| 70538 | mcp2k_p1 862 | +0.002 | — | — | **negative control:** same-time, both flashes exist |

**Null floor** **[V]**. Is 21/22 more than chance, given that the test times sit a few µs from
a bright flash in a TPC that is lit?

- For each of the 22 within-veto moves, `d123_null.py` re-runs the same onset rule on the same
  TPC at t_miss + δ, for δ ∈ {−1.5, −0.8, +0.8, +1.5} µs, on both sides.
- It skips any δ that lands within 0.3 µs of a reco1 flash on that side.
- Result: **0 of 166 null probes** fire, and 0 of 22 moves have any null onset
  (`docs/123_flash/123_null_onsets.tsv`).
- So the onsets at the vetoed times are real pulses, not the rule firing on a lit TPC.

**Caveats on this test.**

- The 100 PE / 3× thresholds are my choice. They were not tuned. A 200 ns box is cruder than
  xning's split rule.
- The pre-window control guards against a smooth tail, not against a second bright pulse just
  before t.
- The probe dumps hits in −20…+30 µs raw only. For the geom-first moves, the far time is
  outside that range, so their side-B prompt PE of 0 is an artefact of the window, not a
  measurement.
- The census sample is **biased**. It only sees close-flash cases whose charge crosses the
  cathode near the beam window. It says nothing about how often close flashes occur, or about
  close flashes inside one TPC with no cathode-crossing charge. That needs the unbiased census
  in R1.

---

## 7. Campaign rounds (for later sessions)

Each round follows the operating manual:

- every behaviour change is default-OFF;
- the knob-off config and outputs are byte-identical under `hash_archive.py` gates, with the
  labels reported;
- labels and tags are new, never overwritten.

**R0 — bring-in, no behaviour change.**

- xning commits `SBNDOpFlashFinder` (toolkit `flash/`, with their doctests) and
  `SBNDReco1OpHitSource` (`wire-cell-sbnd-reco1`). It is their code, so they author it.
- Add a runner or dump option to write hit-flash `opflash_apa{0,1}.tar.gz` into a *new*
  extracted tag.
- Add a default-OFF `flash_source` knob in the SBND per-event job.
- Gates:
  - knob-off compiled JSON byte-identical;
  - SBND QL gate (the evt 686 style of doc 21) member-hash identical;
  - `wcdoctest-flash` passes;
  - freshness proof on `libWireCellFlash.so`.
- Acceptance for the flash-time offset: at least 45 of 48 in-window on fsprod (§4).
- Resolve xning's open item that the Sep-18 `local/lib` install lacks the class (M1).

**R1 — unbiased close-flash census on the hits.**

- For every reco1 flash in mcp1k + mcp2k, and in the MC files with truth, scan its own
  associated hits (the Assns branch) for a second onset, using the §6 test and xning's split
  rule.
- Output: the rate and PE of second pulses versus Δt, per TPC and in both TPCs; the fraction
  of beam-window flashes carrying a partner; and cross-TPC disagreement (59415-type).
- Check that the hit-level second-pulse time reproduces the rescue Δt0 on the 22 + 2 moves.
- MC truth gives efficiency and purity of the split.

**R2 — attribution arms, so a gain is credited to the right fix.** The new finder changes
three things at once: it splits close flashes, it removes the 74 µs bug, and it keeps bright
flashes that reco1 drops. Arms, all from hits:

| arm | contents | purpose |
|---|---|---|
| A0 | xning's emulator at VetoSize 8 (≡ reco1, exact per §2) | null |
| A1 | A0 + 74 µs fix | isolates the bug fix |
| A2 | A1 + keep bright flashes | isolates the seed-rule fix |
| A3 | `SBNDOpFlashFinder` | adds the splitting |

- Run A0 against the stored reco1 OpFlash first; this is the null pair and it must be exact.
- Run Q/L on each arm, on nueCC48 plus a mcp subset.

**R3 — Q/L with the new flashes, rescue ON.**

- Full mcp1k + mcp2k + nueCC48 through the existing chain, on a new work root.
- Measure:
  - the rescue-firing census against the 41 moves in §6 (the prediction: the 22 within-veto moves, the 2
    unmatched moves and the 3 veto losses behind 72759, 78242 and 317427 stop firing);
  - neutrino-candidate count, and the beam flash per event;
  - movers, split by whether the matched flash is under 100 PE.
- Hand-scan the movers blind, with a fresh tag.

**R4 — the rescue decision.**

- On the R3 hit-flash arm, run rescue OFF, and each round-2/3 knob OFF in turn.
- Keep what still earns its place (same-time merging, geom-first); retire what the hit flashes
  make redundant.
- Also look at the 65053-type 92 ns offsets, where the partner exists but misses the 80 ns
  grouping window.
- Owner flip decision.

**R5 — the production path** (§5.1, owner decision). A larwirecell OpHit source, an SBND reco1
change, or standalone only.

**Later options, not scheduled:**

- a joint two-TPC finder, which fixes cross-TPC disagreement at the source;
- tail subtraction after a split, which fixes the 1.5–2× PE on split-off pieces;
- a MicroBooNE-style per-PMT KS pattern test to separate close flashes
  (`prototype_base/2dtoy/src/ToyLightReco.cxx`, per `sbnd-opdetreco-chain.md`).

## 8. Open questions for the owner

1. The production path for hit flashes, §5.1: (a), (b) or (c).
2. Bring-in form. Is xning's code accepted as the SBND variant in toolkit `flash/`, next to
   `OpFlashFinder`? Or should the existing `OpFlashFinder` get SBND knobs? The manual prefers
   duplication over in-place generalisation.
3. Which extra small flashes are acceptable in Q/L: keep `flash_minPE` 50, or raise it for hit
   flashes only.

## 9. Files

- `docs/123_sbnd-ophit-flash-reco-campaign.md` — this doc.
- `docs/123_flash/123_rescue_census_detail.txt` — copy of `~/tmp/cbr3/census-firing-detail.txt`:
  40 events, all rescue knobs on, mcp1k + mcp2k.
- `docs/123_flash/123_rescue_vs_hits.tsv` — per move: sample, entry, Δt0, class, matched reco1
  flashes, prompt PE, and the nearest reco1 flash on each side.
- `docs/123_flash/123_rescue_onsets.tsv` — per move: the onset test on sides A and B, and
  whether a hit-level split is expected.
- `scripts/analysis/light/d123_ophit_probe.C` — bare-ROOT OpHit/OpFlash dump. It needs
  `wire-cell-sbnd-reco1/install/lib/libWireCellSBNDReco1.so`.
- `scripts/analysis/light/d123_rescue_vs_hits.py`, `d123_summary.py` — the §6 analysis.
- `scripts/analysis/light/d123_null.py`, `docs/123_flash/123_null_onsets.tsv` — the §6 null
  floor and the PMT parity check.
- Raw hit dumps (6.8 MB) are not committed; the Repro block regenerates them.

**External references, read-only:**

- xning's docs: `/nfs/data/1/xning/wirecell-working/SBND/light/{README,FLASH_FROM_HITS,FLASH_DRIFT_POSITION,MC_FLASH_STUDY,QL_MATCHING_IMPACT}.md`;
- `toolkit/match/docs/sbnd-opdetreco-chain.md`, `qlmatching-evt59415-xtpc-flash-split.md`;
- `toolkit/root/docs/sbnd-reco1-source.md`; `wire-cell-sbnd-reco1/docs/DESIGN.md`;
- sbndcode `OpDetReco/OpFlash/{job/sbnd_flashalgo.fcl, FlashFinder/SimpleFlashAlgo.cxx, SBNDFlashFinder_module.cc}`.

---

# Execution log (2026-09-24 →)

Owner decisions before execution: bring-in commits owner-authored, crediting xning in the message;
R5 is a design note for a larwirecell OpHit source (no LArSoft build); full samples (data: nueCC48,
NCpi0, mcp1k, mcp2k; MC: round-3 cv, nuecc, beam-off); movers adjudicated from evidence, not by hand
scan; production arms may use up to 32 CPUs.

## 10. Round 0 — bring-in, gates, first look (2026-09-24)

### Repro

```bash
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
P=~/tmp/d123-libpin            # toolkit 69515f37 + the 3 flash files, reco1 85b7932 + the 7 files; libs.md5 inside
export LD_LIBRARY_PATH=$P:$P/reco1/lib:$LD_LIBRARY_PATH

# baseline (reco1 flashes), knob absent; then the two hit-flash arms on the SAME imaging
SBND_QL_KEEP_ICLUSTER=1 SBND_MAX_JOBS=3 ./run_chain_group.sh input_files_reco1/data_filtered_decoded_reco1-fe6033f3-*_frameshift.root work-nuecc48-d123base data --size 16 --layout perevt
JOBS=3 scripts/d123/hits_arm.sh work-nuecc48-d123base work-nuecc48-d123hits    data --ref
JOBS=3 scripts/d123/hits_arm.sh work-nuecc48-d123base work-nuecc48-d123nosplit data --ff '{"pulse_split":false}'

# gates
python3 scripts/multi/repro_cmp.py work-nuecc48-d123base work-nuecc48-prod0923 $(cat work-nuecc48-d123base/g*/events.txt)   # (a)
python3 scripts/d123/intime_gate.py work-nuecc48-d123hits                                                                     # (d)
python3 scripts/d123/r1_census.py work-nuecc48-d123hits --tsv /tmp/r1.tsv                                                     # first census
python3 scripts/d123/r3_ql_compare.py work-nuecc48-d123base work-nuecc48-d123hits --tsv /tmp/r3.tsv                          # first Q/L look
```

### 10.1 What was brought in

| repo | files | commit |
|---|---|---|
| toolkit (`apply-pointcloud`) | `flash/{inc/WireCellFlash/SBNDOpFlashFinder.h, src/SBNDOpFlashFinder.cxx, test/doctest_sbndopflashfinder.cxx}` verbatim from xning's tree; `cfg/pgrapher/experiment/sbnd/{sbnd-opdet-geom.json, sbnd-pmt-channels.json}` (regenerated from `wire-cell-data/sbnd/photodet/semi-analytical-sbnd.json` + sbndcode `sbnd_pds_mapping.json` with xning's `make_configs.py`: byte-identical); `flash/docs/sbnd-flash-from-hits.md` (provenance, sha256s) | see §10.6 |
| `wire-cell-sbnd-reco1` (`main`) | new `inc/WireCellRoot/SBNDReco1OpHitSource.h`, `src/SBNDReco1OpHitSource.cxx`, `src/SBNDReco1CafOffset.h`; modified `src/SBNDReco1OpFlashSource.cxx` (its CAF-offset code moved into the shared header), `CMakeLists.txt`, `test/check_factories.cmake`, `README.md` | see §10.6 |
| wcp `sbnd_xin/` | `wct-reco1-dump.jsonnet` knob (§10.2); `scripts/d123/{hits_arm.sh, intime_gate.py, r1_census.py, r3_ql_compare.py}`; this section | see §10.6 |

Build: `wcbuild` rc 0; `./build/flash/wcdoctest-flash` **32/32** (13 new `SBNDOpFlashFinder` cases,
through the factory); `md5sum local/lib/libWireCell*.so` before/after: **only `libWireCellFlash.so`
changed** (`ac64d0bd…` → `cb055020…`, mtime 09:38 > source); `nm -DC` finds the class. reco1: cmake build +
install rc 0, `ctest reco1_factories` 1/1 (3 factories). Lib pin `~/tmp/d123-libpin` (toolkit + reco1 `.so`,
`libs.md5`, `TOOLKIT_HEAD`); every arm below runs on it and records `.libs.md5.{start,end}` (all unchanged).

### 10.2 The knob: `flash_source` in `sbnd_xin/wct-reco1-dump.jsonnet`

New TLAs, all defaulting to today's graph: `flash_source='reco1'|'hits'`, `hit_time`, `hit_product`,
`ff` (code, merged over the finder config), `reco1_reference` (also writes `reco1flash_apa<N>.tar.gz`),
`with_frames`. With `hits`: `SBNDReco1OpHitSource:hits_tpc<N>` (channels from the committed
`sbnd-pmt-channels.json`) → `SBNDOpFlashFinder:ff_tpc<N>` (`nchan 312`, the committed geometry) → the
**same** `TensorFileSink:opflash_sink_apa<N>` and file name. `WireCellFlash` is appended to the plugin
list only when on. **The Q/L job, the PR job and `run_chain_group.sh` are byte-untouched.**

`scripts/d123/hits_arm.sh <base> <hits> <data|sim> [--ff JSON] [--ref] [--groups] [--hit-time]` builds
a hit-flash arm on a baseline's imaging: per `g<K>` it re-runs only the flash dump (same file, entry
range, CAF mode and frameshift product, read from the baseline's compiled dump config), checks the
event-id set, links `frames-dnn.tar.bz2` + `icluster-*.npz`, copies `events.txt`, and runs
`run_chain_group.sh --from ql --layout perevt`. Post-checks: `rse.json` identical, compiled Q/L config
identical after root-path normalisation, libs unchanged.

Size note: the hit-flash `opflash_apa<N>.tar.gz` also carries the finder's `flash_summary` and
`ophits` tensors (every hit with its flash id — what §11 reads), so it is ~7 MB per 16-event group
against 0.1 MB for reco1 flashes (a 48-event arm: 512 MB vs 779 MB total; the imaging dominates).

### 10.3 Gates — all PASS

| gate | what | result |
|---|---|---|
| (a) knob-off Q/L | `work-nuecc48-d123base` (new build, pin, knob absent) vs `work-nuecc48-prod0923` (toolkit 377119ee) | `repro_cmp.py`: **IDENTICAL same=192 differ=0** (48 events × pctree + 3 Bee zips) |
| (b) reco1 reader refactor | re-dump nueCC48 g0–g2 with the rebuilt plugin, `hash_archive.py` vs prod0923 | **9/9 archives identical** (`opflash_apa{0,1}` ×3 groups, `frames-dnn` ×3) |
| (c) compiled dump config | recorded `prod0923/g{0,1,2}/.wct-cfg-dump.json` vs recompiled with the new jsonnet; old-vs-new jsonnet for the data, `--fsproduct` and `--mc` TLA lists | **all identical**; knob on: finder ×2, hit source ×2 with 60 channels, `WireCellFlash`, no frame source, sinks unchanged |
| (d) flash-time offset carried | `intime_gate.py`: events with a flash in +0.3…1.9 µs after `frame_apply_at_caf` | hits **45/48**, no-split 45/48, reco1 45/48 — the same three events missing (111412, 131357, 214469) |
| (e) xning's comparison reproduced | 5 entries (0, 7, 19, 30, 44), their `compare_flashes.py` (time matching) | 148/152 reco1 flashes found; PE ratio ours/reco1 median **1.005** (5–95 %: 0.993–1.19); Δt median **+8.4 ns**; 194 ours vs 152 reco1 (xning, 63 entries: 91 % one-to-one, 1.005, +8.3 ns) |
| per-PMT pattern | brightest flash of evt 10550 TPC0, reco1 vs ours | 50/50 channels, corr **1.0000**, max per-channel Δ 21.5 PE of 7418 |

Post-check trap found and fixed: `run_chain_group.sh` writes the root path in the form it was given
(relative for the baseline, absolute from the arm script), so the compiled-config comparison
normalises both forms. A first version of `hits_arm.sh` used `GROUPS` as a variable name — bash's
own `GROUPS` array — and read the gid instead of the group list; renamed.

Arms: `work-{nuecc48,ncpi0}-d123{base,hits,nosplit}` (48 + 19 events); `work-mcp1k-d123base` started
(63 groups, 5 at a time).

### 10.4 First census on nueCC48 (48 events, `r1_census.py`, the §11 tool)

Per reco1 flash the nearest hit flash within 0.5 µs is its match; unmatched hit flashes ≥ 20 PE are
classed by where they sit against the reco1 flashes:

| | count | note |
|---|---|---|
| reco1 flashes / hit flashes | 1387 / 1780 | |
| matched | 1328 | Δt median −8 ns |
| reco1 flash with no hit flash within 0.5 µs | 59 | the finder joined or removed it (late-light rule); to inspect in §11 |
| **absorbed** (inside a reco1 flash's 8 µs integral, later) | 66 | Δt 6–7 µs dominates |
| **vetoed** (0.3–8 µs before a reco1 flash) | 94 | PE median 42, up to 8.7 k |
| prepulse (< 1 % of a bright flash, 0.3–4 µs before it) | 53 | 40 of the 87 beam-window flashes carry one at 20–100 PE: most likely deconvolution pre-ringing of the big pulse, not light — to confirm in §11 |
| piece (< 0.3 µs from a matched flash) | 21 | the same pulse cut in two |
| dropped (outside every reco1 window) | 218 | PE median 84 |

Beam window (+0.3…1.9 µs): 87 reco1 flashes, 22 with a vetoed partner ≥ 20 PE, 0 with an absorbed one.

### 10.5 First Q/L look on nueCC48 — the moves are NOT from the split

*(Numbers corrected in round 3: the first pass of `r3_ql_compare.py` keyed the matcher's two
anode-group blocks as one, so clusters of TPC0 and TPC1 with the same ident were compared with each
other; the corrected comparator labels the k-th bundle-map block of an event with the k-th
`anode N group-bbox` line. The picture did not change, the counts doubled.)*

`r3_ql_compare.py` keys matched clusters by (event, anode group, cluster ident) from the Q/L log
(null test base vs prod0923: 963/963 same) and calls two matches the same flash when their times
agree to 0.2 µs.

| arms | matched clusters | same | moved | of which to a < 100 PE flash | beam-window matches A → B |
|---|---|---|---|---|---|
| base → hits | 963 / 963 | 727 | **235** | 74 | 119 → 97 |
| no-split → hits | 963 / 963 | 954 | **9** | 1 | 98 → 97 |

So 24 % of the matched clusters change flash, and the pulse split accounts for 9 of the 235 moves:
the rest come from the finder's other differences — the recovered vetoed/dropped flashes (+28 %
candidates), the extra small flashes at `flash_minPE` 50, and the join/late-light rules. Two readings
of the beam-window drop 119 → 97:

- **the intended one** — evt 74544 anode 1: a 1972-PE-predicted cluster sat on the 32 k PE beam flash
  in base; with hit flashes it matches a **1101 PE flash at −495.5 µs that reco1 had vetoed** (2.8 µs
  before a 3895 PE flash) — predicted 1097 PE, a cosmic recovered from the beam bundle;
- **the one to watch** — evt 388 anode 1: two clusters leave a 489 PE beam flash for flashes that
  reco1 also had (−580 and −333 µs), i.e. the global fit re-balanced, not a new flash.

Flash times and `flash_x_offset` of the common flashes agree between the arms to the ns, and the
per-PMT pattern of a matched flash is identical (corr 1.0000), so the moves are in the candidate
set and the fit, not in a convention. Whether they are right is the question §13 answers with the
neutrino-candidate tables (PR stage) on 3.1 k data events and §14 with truth on 4 k MC events; the
no-split arm stays in the ladder as the attribution control (§12).

### 10.6 Commits

- toolkit `apply-pointcloud` **c2b578fe** (on 25edbe2a): the finder, its doctest, the two JSONs, `flash/docs/sbnd-flash-from-hits.md`.
- `wire-cell-sbnd-reco1` `main` **d114880**: the OpHit source, the shared CAF-offset header, the gated OpFlashSource refactor.
- wcp `sbnd_xin/`: the dump knob, `scripts/d123/`, this section (the commit that carries this text).

## 11. Round 1 — the close-flash census on the hits (2026-09-24)

### Repro

```bash
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 $SX/scripts/d123/r1_census.py $SX/work-mcp1k-d123hits --tsv ~/tmp/r1_mcp1k.tsv --summary docs/123_flash/123_r1_census_mcp1k.json
python3 $SX/scripts/d123/r1_rescue_xcheck.py ~/tmp/r1_mcp1k.tsv > docs/123_flash/123_r1_rescue_xcheck_mcp1k.tsv
```

Input: `work-mcp1k-d123hits/g*/{opflash,reco1flash}_apa{0,1}.tar.gz` (the hit flashes and SBND's reco1
flashes of the same 1000 MCP2025C events, §10.2). Every reco1 flash takes the nearest hit flash within
0.5 µs as its match; every unmatched hit flash ≥ 20 PE is classed by where it sits against the reco1
flashes: **absorbed** (0.3–8 µs after one, inside its integral), **vetoed** (0.3–8 µs before one, inside
its veto), **prepulse** (a vetoed flash under 1 % of the reco1 flash 0.3–4 µs after it), **piece**
(< 0.3 µs from a matched flash), **dropped** (outside every reco1 window). mcp2k is added below when
its arm lands; nueCC48 (§10.4) and NCpi0 (`123_r1_census_ncpi0.json`) show the same shares.
*mcp2k (2000 events, `123_r1_census_mcp2k.json`) reproduces every share: 57 057 reco1 / 74 864 hit
flashes, 95.5 % matched, per event 1.61 absorbed, 1.83 vetoed, 0.94 prepulse, 5.56 dropped; 2 570
beam-window reco1 flashes, 122 with an absorbed and 564 with a vetoed partner ≥ 20 PE.*

### 11.1 What reco1 loses, per 1000 data events

| | count | per event | ≥ 100 PE | note |
|---|---|---|---|---|
| reco1 flashes / hit flashes | 28 859 / 37 524 | 28.9 / 37.5 | | Δt of a match: median −8.8 ns (reco1 later: its light-travel term) |
| matched | 27 597 (95.6 %) | | | |
| reco1 flash with no hit flash | 1 262 (4.4 %) | 1.3 | | **734 sit 7.5–10.5 µs after a ≥ 2 k PE flash**: reco1's own 8 µs window ends and the remaining tail becomes a new "flash" (60–450 PE); the finder folds it back (late light) |
| **absorbed** | 1 653 | 1.65 | 657 | the ≥ 100 PE ones peak 3–5 µs after the seed (≈ 100 per µs bin); the < 100 PE ones pile up at 6–8 µs (tail pieces) |
| **vetoed** | 1 837 | 1.84 | 590 | the ≥ 100 PE ones are flat, 60–115 per µs bin, out to −8 µs; the < 100 PE ones sit at −2…0 µs (927 of 1 837, median 30 PE) |
| prepulse | 929 | 0.93 | 13 | |
| piece | 225 | | | |
| **dropped** | 5 283 | 5.28 | 2 365 | PE median 92, none ≥ 1 000: light reco1 put in no flash at all |

So in one TPC reco1 loses about **1.25 real pulses ≥ 100 PE per event** to its own window (657 absorbed +
590 vetoed) and never makes a flash for 2.4 more (dropped, 100–1000 PE). The prompt-PE rule behind the
dropped class: reco1 seeds on a 10 ns bin ≥ 20 PE, the finder on an 8 µs slice ≥ 20 PE, so light spread
over microseconds (dim, diffuse, or far-side) never seeds in reco1.

**The beam window (+0.3…1.9 µs), where Q/L matters:**

| | count |
|---|---|
| reco1 beam-window flashes | 1 330 |
| … carrying a ≥ 100 PE partner reco1 merged into them or vetoed | **47** (31 vetoed, 16 absorbed) |
| hit flashes ≥ 100 PE *inside* the beam window with no reco1 flash | **127 in 124 events** (78 dropped, 39 vetoed, 10 absorbed) |
| … the 78 dropped: PE 100–392, and in 73 of them the *other* TPC holds the reco1 beam flash | the dim half of a cathode-crossing beam-window activity — exactly what `cathode_rescue_unmatched` had to supply geometrically |
| events with reco1 beam flashes in both TPCs | 531: 499 agree to 80 ns, 32 disagree (hit flashes agree in 3 of the 32) |
| events with a reco1 beam flash in ONE TPC only | 268: hit flashes give an agreeing two-TPC pair in **62** of them |

The last row is the 59415-type gain: 62 of 1000 events go from a one-sided beam flash to a
time-coincident pair, which `xtpc` / `flash_group_window` can then treat as one flash.

### 11.2 The cathode-rescue moves are hit-level losses — confirmed on the flashes themselves

`r1_rescue_xcheck.py` looks, for every doc-123 §6 move of mcp1k, on both sides (far TPC at the near
cluster's t0, near TPC at the far cluster's t0) for a hit flash within 0.3 µs:

| move class (§6) | moves in mcp1k | restored by a hit flash reco1 did not have | reco1 already had it |
|---|---|---|---|
| within the 8 µs veto | 8 | **8** (PE 0.3 k – 26 k: 56463, 59003, 65289, 169824, 288952, 352365, 392200, 395148) | 0 |
| far half unmatched | 1 | **1** (56463) | 0 |
| same time | 2 | 0 | 2 (169758, 395060 — no light loss, as §6 said) |
| geom-first > 13 µs | 1 | 0 | 1 (65053 — the 92 ns pair, a grouping-window case) |

Every within-veto move of mcp1k has its missing partner in the hit flashes. With mcp2k, all 41 moves of
§6 (`123_r1_rescue_xcheck_mcp.tsv`):

| §6 class | moves | restored by a hit flash reco1 did not have | reco1 already had it | none |
|---|---|---|---|---|
| within the 8 µs veto | 22 | **19** (PE 0.3 k – 33 k) | 0 | 3 (70128, 173450, 396761 — Δt0 −0.4, −1.8, +1.0 µs; see below) |
| far half unmatched | 2 | 1 (56463) | 1 (50801) | 0 |
| 8–13 µs | 4 | **1 (72759** — the veto loss §6 predicted) | 3 | 0 |
| geom-first > 13 µs | 5 | **2 (78242** — predicted; **281165)** | 3 (317427 — §6 read it as a veto loss; its partner is a reco1 flash at 109.15 µs, so it is a choice case; 65053, 319913) | 0 |
| same time | 8 | 0 | 8 | 0 |

So 23 of the 41 rescue moves are light losses that the hit flashes repair, 15 are Q/L choices with both
flashes present (the same-time and most long-Δt0 cases), and 3 within-veto moves with |Δt0| ≤ 1.8 µs
have no separate hit flash within 0.3 µs of the expected partner time — at those separations the
finder's split (`split_min_gap_us` 0.5, dip test) does not always cut, which is the residual the
rescue still covers (§13.1: it fires 4× instead of 12× on mcp1k).

### 11.3 The small early flashes

The −2…0 µs pile-up (prepulse + small vetoed, ≈ 1.9 per event) is made of ~25 single-photoelectron
hits (median 1.1 PE per hit, 80 % below 1.5 PE) spread over ~1.3 µs on the PMTs that are brightest in the
following big flash (top-10 overlap 10/10). But *every* flash below 100 PE looks like that — matched ones
reco1 also has (1.1 PE/hit, 77 % SPE), dropped ones, prepulses — so the hit character does not separate
them; only the time correlation with a following bright flash does. Whether they are light or an
instrumental precursor (deconvolution pre-ringing, PMT pre-pulsing) needs waveforms, which reco1 does
not carry. For Q/L they are ~1 extra ≥ 50 PE candidate per event at −2…+0.4 µs; §13 measures whether
anything matches them, §15 has the knob if it does.

MC truth for the split itself is not re-derived here: xning's `split_truth_check.py` on the 13-event
7b1f file found 11 of 11 split points at a real new pulse and 0 wrong [C]; the round-3 MC arms (§14)
carry the truth-level check of what the recovered flashes do to the selection.

## 12. Round 2 — attribution: what the split changes, what the rest changes (2026-09-24)

Three arms per sample on one imaging: reco1 flashes (`base`), the finder without the pulse split
(`nosplit`, `ff={pulse_split:false}`), the finder (`hits`). `r3_ql_compare.py` per pair; the PR stage
(§13) on all three for the event-level view. mcp1k's `nosplit` arm is added when it lands.

| sample | matched clusters | base → hits moved | base → nosplit moved | **nosplit → hits moved** (= the split) |
|---|---|---|---|---|
| nueCC48 (48 evt) | 963 | 235 (24 %) | 230 | **9** |
| NCpi0 (19 evt) | 433 | 120 (28 %) | 118 | **5** |
| mcp1k (1000 evt) | 20 060 | 5 185 (26 %) | 5 051 | **292 (1.5 %)**; beam-window 41; **rescue firings base 12 / no-split 14 / hits 4** |

Event level (PR stage, `r3_pr_compare.py` on `pr_tables.sh` output, data, no truth): nueCC48
base → hits: 48/48 events keep a candidate; νμ > 0.9 passes 5 → 5 (one flip each way), νe > 7
passes 36 → 37 (2 lost, 3 gained), 2 vertices move > 5 cm; base → nosplit gives the same counts.
NCpi0 base → hits: 19/19 candidates, νμ > 0.9 3 → 2, νe > 4 2 → 1, 3 vertices move; nosplit identical.

So the **split is a 1–2 % effect at cluster level and invisible at event level on these samples**;
everything the hit flashes change comes from the recovered candidates (§11) and the different
candidate set in the global fit. **Except for the rescue trigger:** without the split the finder's own 8 µs
accumulator still merges the within-veto partners, and the cathode rescue fires 14 times on mcp1k (12
with reco1 flashes); with the split it fires 4 times (§13.1). The split is therefore small in the counts
but it is the piece that turns the §6 rescue cases into flash-level pairs (`123_r2_ql_nosplit_vs_hits_mcp1k.json`). This is what §7 R2 was for: a gain in §13 is not the split's, and a
loss is not the split's either. The split's own value is the within-veto rescue cases (§11.2), which
are rare (8 in 1000 events) but exactly the cathode-crossing neutrino topology the rescue was built
for.

## 13. Round 3 — the hit flashes in Q/L + PR, rescue ON: data (2026-09-24 →)

### 13.1 Q/L level, mcp1k (1000 events, `123_r3_ql_summary_mcp1k.json`, `123_r3_moves_mcp1k.json`)

| | base (reco1) | hits | note |
|---|---|---|---|
| matched clusters | 20 060 | 20 058 | 28 lost / 26 gained |
| same flash | | 14 847 (74 %) | |
| **moved** | | **5 185 (26 %)** | 1 362 to a < 100 PE flash |
| … destination is a flash reco1 did not have | | 3 248 (2 209 dropped, 486 vetoed, 432 absorbed, 95 prepulse, 26 piece) | ≥ 100 PE: 1 940 |
| … destination is a flash both arms had | | 1 937 | ≥ 100 PE: 1 883 — the global fit re-balanced |
| \|Δt\| of a move | | 461 < 8 µs, 718 8–100 µs, **4 006 > 100 µs** | small clusters (predicted light median 25 PE) jumping between cosmic flashes |
| beam-window matches | 1 738 | 1 743 | 332 leave the window, 335 enter; predicted light median 27 / 23 PE; ≥ 500 PE predicted: 21 leave, 23 enter |
| **cathode-rescue firings** | **12** (the 12 §6 moves of mcp1k) | **4** | see below |

**The rescue census, the prediction of §5:** in base the rescue fires exactly on the 12 §6 moves of
mcp1k (8 within-veto, 1 far-unmatched, 2 same-time, 1 geom-first). With hit flashes the 8 within-veto
moves, the far-unmatched move and the 65053 geom-first move **stop firing** — their partner is a flash
now (§11.2) and `xtpc` / the flash group pair the halves. What still fires: the 2 same-time moves
(169758, 395060: both halves matched to one flash time, the a/b/c/d merge rule — not a light loss,
as §6 said) and **2 new `unmatched rescue` adoptions** (49511, 409590) where a beam-window cluster adopts
a large unmatched cluster (198 cm / 358 cm). In both, the hit flashes *added* the beam-window flash
(reco1 had vetoed a 61 PE / 1033 PE pulse 5–8 µs before a bright flash), a small cluster matched it,
and the unmatched rule then attached the big cluster — the PR stage decides whether the adopted
object is right (§13.3).

The moves themselves are mostly the bookkeeping of small cosmic clusters among cosmic flashes: 77 %
jump more than 100 µs, their predicted light is ~25 PE, and the beam-window flow is balanced (332 out,
335 in). The event-level effect is what matters and is measured on the PR output (§13.2–13.3).

### 13.2 Event level, mcp1k (PR stage on both arms; `123_r3_pr_summary_mcp1k.json`, `123_r3_pr_flips_mcp1k.tsv`, `123_r3_pr_flip_classes_mcp1k.tsv`)

```bash
scripts/d123/stageB.sh work-mcp1k-d123base data; scripts/d123/stageB.sh work-mcp1k-d123hits data
scripts/d123/pr_tables.sh work-mcp1k-d123basepr products/d123/mcp1k_base;  scripts/d123/pr_tables.sh work-mcp1k-d123hitspr products/d123/mcp1k_hits
python3 scripts/d123/r3_pr_compare.py products/d123/mcp1k_base products/d123/mcp1k_hits --label base,hits --tsv flips.tsv
python3 scripts/d123/r3_evidence.py mcp1k <event> --r3 <r3.tsv> --r1 <r1.tsv>      # one event's evidence sheet
```

| | base (reco1) | hits | |
|---|---|---|---|
| events with a neutrino candidate | 465 | 470 | 453 in both; **12 only in base, 17 only in hits** |
| νμ > 0.9 | 271 | 271 | 10 flip each way |
| νe > 7 / > 4 | 1 / 1 | 1 / 1 | no flip |
| candidate vertex moves > 5 cm | | 21 | |
| **events changed at event level** | | **56 / 1000** | |

So the 26 % of clusters that change flash (§13.1) become a 5.6 % event-level perturbation with **zero net
change in the selected counts** — on data, where the truth is unknown, that is the strongest statement the
counts can make; §14 says which way the flips go on MC. What the 56 events are, from the Q/L moves in
the beam window (`r3_move_classes`-style mechanism per event, the table in `123_r3_pr_flip_classes_mcp1k.tsv`):

| mechanism (beam-window moves of clusters with ≥ 200 PE predicted light) | events | reading |
|---|---|---|
| a big cluster **joined a new beam-window flash** the hit finder recovered | 7 | reco1 had merged the beam pulse into a bright cosmic flash 2–5 µs away; 3 of the 7 become νμ candidates (280884, 281808, 390644: fully contained 150–190 cm tracks on 9–17 k PE flashes at their own time) |
| a big cluster **left the beam flash for a restored flash** | 5 | e.g. 74544-type (§10.5): a cosmic taken off the beam bundle; 2 of the 5 are rescue events (below) |
| a big cluster joined / left the beam flash with flashes both arms had (fit re-balance) | 6 / 4 (+1 both) | the global LASSO with a different candidate set |
| small clusters only (predicted light < 200 PE) | 33 | vertex moves, score drifts, candidates of a few cm gained or lost |

**The rescue events** (the 11 §6 events of mcp1k; `123_r3_pr_flips_mcp1k.tsv`):

| event | §6 class | base → hits |
|---|---|---|
| 56463, 65289, 395148 | within veto | νμ candidate kept (numu 4.67→4.85, 2.60→2.60, 3.63→3.63) |
| 288952, 352365, 392200 | within veto | no candidate in either (cosmic bundles), unchanged |
| 169758, 395060 | same time | unchanged (rescue still fires, as expected) |
| **169824** | within veto | **candidate lost** (numu 5.65, 325 cm, Enu 1059 MeV → none): TPC1 gained the restored −3.15 µs flash (14.9 k PE, the partner of TPC0's 39 k PE cosmic pulse), and the fit moved the TPC1 half from the beam flash (1.33 µs, 14.9 k PE) to it — both halves now sit on the −3.14 µs pulse, and the beam-window light (7.6 k + 14.9 k PE in the two TPCs) has no charge at all |
| **59003** | within veto | **candidate lost** (numu 3.20, 298 cm → none): TPC1 gained the restored 1.584 µs flash (9.2 k PE, absorbed by reco1 into the −0.75 µs 34 k flash); the `QLXTPC coincident` step now sees the TPC0 half (cluster 3, 2162 pts) coincident with TPC1 clusters at 1.59/1.58 µs, and the TPC0 half ends **unmatched** (a 624-PE cluster takes the beam flash, the bundle is TGM-tagged) while the TPC1 half stays on the −0.74 µs flash. In base no `coincident` line exists for that cluster (TPC1 had no flash there) and the rescue merged the halves geometrically |
| 65053 | geom-first (92 ns pair) | numu 2.88 → −0.31, vertex moves, a second 2 cm candidate appears: the 94 cm TPC0 cluster the rescue used to bring in now matches the beam flash itself, and the bundle is STM-tagged |

Reading: restoring the partner flash removes the rescue's *trigger* (it fires 4× instead of 12×, §13.1), and
in 6 of 8 within-veto cases Q/L then reaches the same event-level answer on its own. In the other two the
fit, given both flashes, assigns the cathode-crossing track differently from the rescue's geometry — and
in 169824 leaves 22 k PE of beam-window light without any charge, which is not a stable configuration
either. Whether the fit or the geometry is right in those two cannot be settled on data; the MC arms (§14)
carry the truth for exactly this topology, and the 59003 pattern (a coincident pair culling the TPC0
half instead of pairing it) is the first item for the round-4 look at `xtpc` (§15).

**The gains.** The 7 "new beam-window flash" events are the class §11 predicted: a beam-window pulse
reco1 folded into a cosmic flash 2–5 µs away, split out by the finder, matched by a contained ~150 cm
track. Three pass νμ > 0.9 (numu 4.40, 2.96, 2.83). On data they are either recovered neutrinos or
cosmics that happen to sit on a beam-window flash; the beam-off arm (§14) gives the rate of the latter.

### 13.3 mcp2k (2000 events): the same picture (`123_r3_ql_summary_mcp2k.json`, `123_r3_moves_mcp2k.json`, `123_r3_pr_summary_mcp2k.json`)

| | mcp1k (§13.1–13.2) | mcp2k |
|---|---|---|
| matched clusters, base / hits | 20 060 / 20 058 | 39 768 / 39 786 |
| moved | 5 185 (26 %) | 10 243 (26 %) |
| beam-window matches base → hits | 1 738 → 1 743 | 3 459 → 3 367 |
| rescue firings base → hits | 12 → 4 | **29 → 16** |
| PR level | 56 events changed; candidates 465 → 470; νμ 271 → 271 | 55 candidate-only events (27 base / 28 hits), 35 vertex moves; candidates 890 → 891; **νμ > 0.9: 513 → 513** (13 flips each way); νe > 7: 4 → 4 |

The 16 residual firings on mcp2k are, by the §11.2 classes: the 6 same-time merges (expected to stay),
the 3 within-veto pairs the finder did not split (70128, 173450, 396761), 4 cases where reco1 already had
the partner (50801, 164576, 319913 and 72759 — whose restored 250 PE partner did not take the far half),
and 3 new unmatched adoptions (73054, 74094, 79491). The 13 other §6 moves of mcp2k — all the ones the
census called *restored* — stop firing, as on mcp1k.

Over the 3000 mcp events the νμ > 0.9 count is **784 with reco1 flashes and 784 with hit flashes** (23 lost,
23 gained), the candidate count 1355 → 1361. The 29 mcp2k rescue events at event level
(`123_r3_pr_flips_mcp2k.tsv`): 24 unchanged; among the restored within-veto cases 179369 keeps its νμ
candidate with a higher score (2.21 → 5.08), **161725 loses its νμ pass (2.24 → 0.85)** — the third
fit-vs-geometry case with 169824 and 59003 — and 73324/287244 exchange a low-score candidate; 73727
(same-time) stays a pass at 2.45. So on 3000 data events, 3 of the 30 within-veto rescue events change
their νμ verdict once the partner flash exists, all three downward, against 3 + 5 new contained νμ
candidates on flashes reco1 had merged away (§13.2 for mcp1k; mcp2k has 5 more of the same class among
its 28 candidate-only-hits events).

## 14. Round 3 — MC and beam-off (2026-09-24 →)

### 14.1 Beam-off: the fake-candidate rate (1000 Run-1 off-beam gates, `work-r3off-d123{base,hits}pr`)

```bash
scripts/d123/mc_base.sh off; scripts/d123/mc_hits.sh off hits; scripts/d123/stageB.sh work-r3off-d123base data; scripts/d123/stageB.sh work-r3off-d123hits data
scripts/d123/pr_tables.sh work-r3off-d123basepr products/d123/r3off_base; scripts/d123/pr_tables.sh work-r3off-d123hitspr products/d123/r3off_hits
python3 scripts/d123/r3_pr_compare.py products/d123/r3off_base products/d123/r3off_hits --label base,hits
```

Off-beam gates hold cosmics only, so every candidate is a fake; this is the cost side of the extra
beam-window candidates (§11: hit flashes offer 824 beam-window flashes ≥ 20 PE against reco1's 647 here).

| per 1000 gates | base (reco1) | hits | |
|---|---|---|---|
| gates with a neutrino candidate | 85 (8.5 %) | 96 (9.6 %) | 80 in both, 5 only base, 16 only hits |
| νμ > 0.9 | **9** | **13** | 1 lost, 5 gained |
| νe > 7 / > 4 | 0 / 0 | 0 / 1 | |
| Q/L level (`123_r3_ql_summary_r3off.json`) | 19 273 matched clusters | 28 % moved | beam-window matches 880 → 895; rescue firings 13 → 5 |

The 5 gained νμ fakes are not small-flash accidents: they are 185–270 cm cosmic tracks (none contained)
that the global fit moves onto a 17–26 k PE beam-window flash *both arms had* (449681: from a −303 µs
11 k PE flash to the 0.94 µs 17.6 k one; 742853, 197600 alike), with νμ scores 0.8–2.0. So the fake rate
rises from 0.9 % to 1.3 % per gate through the fit re-balancing that the larger candidate set causes, the
same mechanism as the "both-had" moves of §13.1; the recovered small flashes and the prepulses play no
part in it. On beam-on data the νμ count stayed at 271 (§13.2) with 10 flips each way, so the beam-on
gains and the extra fakes are of the same order — the MC efficiency (§14.2) is what separates them.

### 14.2 MC truth: round-3 inclusive BNB (`mc-cv`, 2017 events, 557 true νμCC in the FV)

```bash
scripts/d123/mc_base.sh cv; scripts/d123/mc_hits.sh cv hits; scripts/d123/stageB.sh work-r3cv-d123base sim; scripts/d123/stageB.sh work-r3cv-d123hits sim
scripts/d123/pr_tables.sh work-r3cv-d123basepr products/d123/r3cv_base cv; scripts/d123/pr_tables.sh work-r3cv-d123hitspr products/d123/r3cv_hits cv
python3 d107_selection.py products/d123/r3cv_<arm> <figdir>            # docs/123_flash/sel/r3cv_<arm>_d107_selection.txt
python3 scripts/d123/r3_pr_compare.py products/d123/r3cv_base products/d123/r3cv_hits --label base,hits --tsv flips.tsv
```

Same imaging, same PR chain, the doc-107/115 definitions (FV 5 < |x| < 190, |y| < 190, 10 < z < 450 cm;
vertex match 5 cm; νμ score > 0.9, νe score > 7). The baseline at this pin reproduces doc 115
(70.0 % / 86.7 % against 69.5 % / 86.4 %).

| νμCC selection (true νμCC in FV: 557) | base (reco1 flashes) | **hits** |
|---|---|---|
| candidate vertex within 5 cm of the true vertex | 457 (82.0 %) | **472 (84.7 %)** |
| … and νμ score > 0.9 = **efficiency** | 390 (**70.0 %** [68.0, 71.9]) | **400 (71.8 %** [69.9, 73.7]) |
| selected candidates | 450 | 461 |
| **purity** | 390/450 = **86.7 %** | 400/461 = **86.8 %** |
| backgrounds: no true vertex within 5 cm / NC / out-of-FV / νe | 44 / 13 / 2 / 1 | 44 / 13 / 3 / 1 |
| event-level (no vertex requirement, doc 107 §5.7) efficiency / purity | 77.2 % / 93.8 % | 78.3 % / 93.5 % |
| νeCC (5 true in FV) efficiency / purity | 4/5, 4/5 | 4/5, 4/5 |

**+1.8 percentage points of νμCC efficiency at the same purity** (+10 selected signal events, +1
background), and +2.7 points in the vertex-matched candidate rate that feeds every downstream number.
The flips (`123_r3_pr_flips_r3cv.tsv`): 20 events pass νμ > 0.9 only with hit flashes, 10 only with
reco1 flashes. **11 of the 20 gains are events that had no candidate at all in base** (the neutrino's
beam-window light was merged into a cosmic flash or never seeded, §11), 8 of them true νμCC —
the mechanism of §13.2's "new beam-window flash" class, now with truth behind it. The 10 losses are
mostly score drifts across the 0.9 cut (0.9–1.9 → 0.6–0.85, vertex unchanged, 6 of 10) plus two
candidate re-assignments with a vertex jump (717/29/47, 719/81/47) — the fit-vs-geometry topology of
§13.2 (169824, 59003, 161725) costs ≈ 2 true νμCC per 2000 events, against the ≈ 10 it recovers.

Reading against the beam-off cost (§14.1: νμ fakes 0.9 % → 1.3 % per off-beam gate): on the inclusive
MC the extra cosmic-only fakes appear as +0 "no true vertex" backgrounds (44 → 44) — the MC has the
cosmic overlay, so this is measured, not assumed — and the purity does not move.

**The exclusive νe sample** (`mc-nuecc`, 2001 events, 1511 true νeCC in the FV; `123_r3_pr_summary_r3nue.json`,
`sel/r3nue_{base,hits}_d107_selection.txt`):

| νeCC selection (true νeCC in FV: 1511) | base (reco1 flashes) | **hits** |
|---|---|---|
| candidate vertex within 5 cm of the true vertex | 1077 (71.3 %) | **1097 (72.6 %)** |
| … and νe score > 7 = **efficiency** | 630 (**41.7 %**) | **631 (41.8 %)** |
| selected candidates / **purity** | 649 / **97.1 %** | 651 / **96.9 %** |
| events with a candidate | 1869 | 1894 (+25; 7 only base, 32 only hits) |
| νe > 7 flips (signal among them) | 90 only base (76) | 93 only hits (76) |
| candidate vertex moves > 5 cm | | 135 (6.7 %) |
| νμCC in this sample (90 true): efficiency | 47/90 = 52.2 % | 53/90 = 58.9 % |

The νe selection is flat in efficiency and purity (+1 / −0.2 points, inside the 68 % intervals) while the
vertex-matched candidate rate rises 1.3 points and 25 more events get a candidate — the same "invisible
neutrino recovered" mechanism as on cv. The νe flips are many (90 / 93) but symmetric and 76 signal on each
side: a shower's candidate is more sensitive than a track's to which flash the fit settles on (135 vertex
moves), and the hit flashes move it both ways in equal measure. Nothing here argues against the flip; the
inclusive sample is where the gain is, and the νe sample shows the cost is nil.

Bookkeeping: the hits arm's Q/L was stopped and restarted 8-wide when the machine freed up
(`mc_hits.sh` resume guard); two sub-roots (f061, f062) whose Q/L had been killed mid-way kept their
`ql_evt` dirs without pctrees and were skipped by the first guard (it counted dirs) — caught by the PR
count (1988 ≠ 2001), redone, and the guard now counts pctrees. The numbers above are the complete 2001.

## 15. Round 4 — the rescue under hit flashes (2026-09-24)

```bash
QLTLA=scripts/d123/tla/rescue_off.txt           scripts/d123/hits_arm.sh work-mcp1k-d123base work-mcp1k-d123hitsnr data   # whole rescue OFF
QLTLA=scripts/d123/tla/rescue_unmatched_off.txt scripts/d123/hits_arm.sh work-mcp1k-d123base work-mcp1k-d123hitsnu data   # unmatched-adoption rule OFF
scripts/d123/stageB.sh <arm> data; scripts/d123/pr_tables.sh <arm>pr products/d123/<name>; python3 scripts/d123/r3_pr_compare.py products/d123/mcp1k_hits products/d123/mcp1k_hitsnr
```

Same hit flashes, same imaging, the Q/L job with the cathode-bundle rescue removed by its own knobs
(`cathode_rescue=false`, `cathode_rescue_unmatched=false`: the compiled config has no
`ClusteringCathodeBundleRescue` node). The Q/L matches are identical by construction (the rescue runs after
the matching, inside the all-APA step); the PR stage is where it shows.

| arm (mcp1k, 1000 events) | events with a candidate | νμ > 0.9 | events changed vs `hits` |
|---|---|---|---|
| `hits` (rescue ON, production knobs) | 470 | 271 | — |
| `hitsnr` — whole rescue OFF | 472 | **271** | **3** (49511 and 395060 gain a low-score candidate that the rescue had merged away; 169758's score changes) |
| `hitsnu` — unmatched-adoption rule OFF only | 471 | **271** | 1 |
| nueCC48 / NCpi0, rescue OFF | 48 / 19 | 5 / 2 | 1 / 0 |

With reco1 flashes the rescue fires 12 times per 1000 events and its moves make the candidates of
§13.2; with hit flashes it fires 4 times (2 same-time merges, 2 unmatched adoptions) and switching it
off changes 3 events and no selection count. **The geometric patch is redundant once the light is
right** — its remaining cases are the same-time cross-TPC merge (which `xtpc`/the 80 ns flash group
should own) and the 3 within-veto pairs at |Δt0| ≤ 1.8 µs the finder does not split (§11.2).

**Recommendation for the flip (owner's decision):**

1. Flip the flash source: `flash_source=hits` as the standalone-chain default (runner + `ref/prod-<date>`),
   with the `nosplit` arm retired and the `hits_arm.sh` path becoming the dump's normal path.
2. Keep `cathode_rescue` / `cathode_rescue_unmatched` **ON** at the flip — they are inert (3/1000) and
   cover the residual; retire them in a later round once the MC (§14.2) has ruled on the two
   fit-vs-geometry events (169824, 59003), not before. The round-2/3 extensions (`rescue_geom_first`,
   `rescue_pierce_test`, `rescue_in_beam_far`, …) were not tested one by one: with the whole rescue
   inert there is nothing left for them to do on this sample.
3. Two follow-ups the flip exposes, neither blocking: the `QLXTPC coincident` culling of a coincident
   half (59003), and the finder's small early flashes (§11.3) — a `min_fired_pe`-type cut on SPE-only
   flashes if §14 shows any of them matched.
4. Cost side to state with the flip: the beam-off fake νμ rate 0.9 % → 1.3 % per gate (§14.1), from the fit
   re-balancing; on the inclusive MC (§14.2) the νμCC efficiency rises 70.0 → 71.8 % at unchanged purity
   (86.7 → 86.8 %), so the flip is a net gain on every measured axis.
## 16. Round 5 — the production path: a larwirecell OpHit source (design note, no build)

SBND production runs Wire-Cell inside LArSoft: `cfg/pgrapher/experiment/sbnd/wcls-img-clus-matching-xin.jsonnet:66-79`
reads `recob::OpFlash` per TPC through `wclsOpFlashSource` (`art_tag: std.extVar('opflash<N>_input_label')`),
which emits the same `[nflash, 313]` tensor set the standalone chain reads from `opflash_apa<N>.tar.gz`, and
`FlashTensorToOpticalPCs` (`aux/src/FlashTensorToOpticalPCs.cxx:90-97`) adds the set-metadata
`frame_apply_at_caf` to every flash time. Nothing in that graph reads `recob::OpHit`. The v10_04_03
larwirecell on cvmfs ships no source (`source/larwirecell` holds 9 files, none of them the components), so
the note below is written against the component's *interface* as the toolkit sees it, not against its code;
whoever implements it starts from `wclsOpFlashSource.cxx` in the larwirecell git repository.

**What is needed (option (a) of §5.1):**

1. **`wclsOpHitSource`** in larwirecell (`larwirecell/Components/`), an `IArtEventVisitor` +
   `ITensorSetSource` like `wclsOpFlashSource`, producing per event the tensor set `SBNDReco1OpHitSource`
   produces offline:
   - config: `art_tag` (the `recob::OpHit` product, `ophitpmt` in reco1), `channels` (the 60 PMT
     OpChannels of the TPC, from `sbnd-pmt-channels.json` in the jsonnet), `hit_time` (`rise` =
     `StartTime()+RiseTime()`, the SBNDFlashFinder convention), optional `frameshift_tag`
     (`sbnd::timing::FrameShiftInfo`, `frameshift`) whose `fFrameApplyAtCaf` (ns) becomes the
     metadata key `frame_apply_at_caf`, exactly as `wclsOpFlashSource` must already do for the reco1
     flashes it emits (data) and omitted on MC;
   - output: one tensor `"ophits"` f8 `[nhit, 9]` = `{OpChannel, time, width, area, amplitude, PE,
     start, -1, fast/total}` in ns / PE, `ident = event`, metadata `run`, `subrun`, `event`
     (+ `frame_apply_at_caf`);
   - ~150 lines; the only art dependency is `lardataobj/RecoBase/OpHit.h` (+ `sbnobj` for the
     FrameShiftInfo, already a dependency of the flash source).
2. **`SBNDOpFlashFinder`** is already in the toolkit (c2b578fe) and depends on nothing but `WireCellFlash`
   / `WireCellAux`; the wcls job adds `WireCellFlash` to its plugin list.
3. **Jsonnet**: in `wcls-img-clus-matching-xin.jsonnet`, behind a default-OFF `flash_source` extVar, replace
   the two `wclsOpFlashSource` pnodes by `wclsOpHitSource:tpc<N>` → `SBNDOpFlashFinder:tpc<N>` (the
   committed `sbnd-opdet-geom.json` as `geom_file`, `ff` overrides if any) and keep the edge into
   `flash_attach_apa<N>` port 1. Key suppression as in `sbnd_xin/wct-reco1-dump.jsonnet` (§10.2), so the
   compiled production config is byte-identical when off.
4. **fcl** (sbndcode `WireCell/`): `ophit0_input_label` / `ophit1_input_label` (both `ophitpmt`) and the
   `flash_source` extVar next to the existing `opflash<N>_input_label`.
5. **Gates**: the standalone and the LArSoft chain must agree flash for flash — dump the LArSoft-side
   hit-flash tensor with a `TensorFileSink` on the 48-event file and `hash_archive.py` it against
   `work-nuecc48-d123hits/g*/opflash_apa*.tar.gz` (member content; the `ophits` tensor included). Then the
   doc-118 two-chain gate (`scripts/cfg/two_chain_gate.py`) with the knob off.

**Option (b) instead** (sbndcode owns the flash): re-run `SBNDFlashFinder` with a shorter `IntegralTime` /
`VetoSize`, or port the split into `SimpleFlashAlgo`. That changes every SBND consumer of `opflashtpc<N>`
(CAF, other analyses) and cannot be gated by us; §11 shows what the veto costs, which is the argument to
bring to SBND, but (a) is the path that keeps the decision inside this chain.

**Cost**: (a) is one afternoon of larwirecell work plus an sbndcode fcl change and a release; the physics
evidence (§13–§15) does not wait on it. Until it exists, the standalone chain is the only place the hit
flashes run, i.e. option (c) is the de-facto state.

## 17. The flip — `flash_source=hits` is SBND standalone production (2026-09-25)

> "Let's flip this hits on for SBND production, and keep the cathode rescue on." — the owner, 2026-09-25

### Repro

```bash
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin; cd $SX
# proofs A/B on the compiled dump config (pre = git show 519e48ec:sbnd/sbnd_xin/wct-reco1-dump.jsonnet)
wcsonnet <runner TLAs> -A flash_source=hits  pre.jsonnet | diff - <(wcsonnet <runner TLAs> wct-reco1-dump.jsonnet)          # A: 0 lines
wcsonnet <runner TLAs>                       pre.jsonnet | diff - <(wcsonnet <runner TLAs> -A flash_source=reco1 wct-reco1-dump.jsonnet)  # B: 0 lines
# the production tripwire, now 28 artifacts
python3 scripts/cfg/prod_cfg_gate.py                       # PASS -- matches prod-2026-09-25
# output gate: the flipped runner on the PRODUCTION libs (no pin), then member-by-member against the campaign arms
SBND_QL_KEEP_ICLUSTER=1 SBND_MAX_JOBS=3 ./run_chain_group.sh input_files_reco1/data_filtered_decoded_reco1-fe6033f3-*_frameshift.root work-nuecc48-d123flip    data --size 16 --layout perevt
SBND_FLASH_SOURCE=reco1 SBND_QL_KEEP_ICLUSTER=1 SBND_MAX_JOBS=3 ./run_chain_group.sh <same file>                                        work-nuecc48-d123flipoff data --size 16 --layout perevt
SBND_MAX_JOBS=1 ./run_chain_group.sh $(awk -F'\t' 'NR==1{print $2}' products/d115/cv/files.lst) work-r3cv-d123flip/f000 sim --mc --size 1000 --layout perevt
PR_EXTRA_STAGES=pr_display PR_JOBS=8 ./run_pr_chain_batch.sh work-nuecc48-d123flip work-nuecc48-d123flippr data
python3 scripts/d123/flip_gate.py work-nuecc48-d123flip    work-nuecc48-d123hits            # stage A + B
python3 scripts/d123/flip_gate.py work-nuecc48-d123flipoff work-nuecc48-d123base  --no-pr   # the off path
python3 scripts/d123/flip_gate.py work-r3cv-d123flip/f000  work-r3cv-d123hits/f000 --no-pr  # MC, one file (sub-root vs sub-root)
```

### 17.1 What changed (wcp-porting-validation, this commit)

| file | change |
|---|---|
| `wct-reco1-dump.jsonnet` | `flash_source` default `'reco1'` → `'hits'`; the knob comment now states the flip, its date and the two proofs. Nothing else in the file moved (proof B: the reco1 path compiles byte-identically). |
| `run_chain_group.sh` | env `SBND_FLASH_SOURCE=reco1\|hits` appended to the dump TLAs when set; **unset ⇒ no TLA at all**, the jsonnet default. `SBND_FLASH_SOURCE=reco1` is the pre-flip graph — the off path is a real path, not an orphaned flag. |
| `run_reco1_dump.sh` | the same env passthrough for the per-event driver. |
| `scripts/cfg/compile_consumers.sh` | step **(i)**: the dump job compiled with the runner's exact data and `--mc` TLA lists → `sbnd_dump_data.json`, `sbnd_dump_mc.json`. The flash source was in **none** of the 26 artifacts — the (f)/(g)/(h) shape of hole: a production operating point the tripwire could not see. |
| `scripts/cfg/prod_cfg_gate.py` | the two dump artifacts join `KEEP_FULL` (4 KB each), so a flash-source or finder-setting drift is named by key. |
| `ref/prod-2026-09-25/` | the new generation: 28 artifacts, **0 of the previous 26 moved**, README states the flip and what it does not change. |
| `scripts/d123/flip_gate.py` | the output gate: two stage-A roots (and their `pr` roots) member by member — opflash, frames, icluster (npz arrays through numpy), pctree, Bee zips, PR pctree/Bee/calib/nusel. `tracking-pr.root` is not compared (UUID + timestamps in the header). |
| `scripts/d123/hits_arm.sh` | header note only; it stays the way to put a flash-only arm (other `--ff` settings, or a control) on an existing root's imaging. |

What did **not** change: the Q/L job, the PR job, every rescue knob (`cathode_rescue`,
`cathode_rescue_unmatched`, the round-2/3 extensions — all ON, as before), the fit JSONs, the
allocator block, the toolkit and reco1 plugin code (still `c2b578fe` / `d114880` from round 0), and
the LArSoft 1-step chain (§17.4).

### 17.2 Proofs on the compiled config

| proof | what is compared | data TLA list | `--fsproduct` list | `--mc` list |
|---|---|---|---|---|
| **A** (the flip runs what was measured) | pre-flip file + `flash_source=hits` vs flipped file bare | **0 lines** | **0 lines** | **0 lines** |
| **A′** (the measured arm exactly) | pre-flip file + the `hits_arm.sh` TLA (`flash_source=hits with_frames=false reco1_reference=true`) vs flipped file + that TLA minus `flash_source` | 0 lines | 0 lines | 0 lines |
| **B** (the off path is the old graph) | pre-flip file bare vs flipped file + `flash_source=reco1` | **0 lines** | **0 lines** | **0 lines** |
| flipped bare compile | 2 × `SBNDReco1OpHitSource` (60 channels each), 2 × `SBNDOpFlashFinder`, plugin `WireCellFlash`, **no** `SBNDReco1OpFlashSource` | ✓ | ✓ | ✓ |

`prod_cfg_gate.py` against `prod-2026-09-21d` before the refresh: **26/26 unchanged**, 2 NEW
(`sbnd_dump_data.json`, `sbnd_dump_mc.json`); after `--refresh --ref ref/prod-2026-09-25`: **PASS
28/28**. The flip therefore moved exactly the artifacts that did not exist before it, which is the
statement that it touched nothing the previous generation gated.

### 17.3 Output gate — the flipped runner on the production libraries

The campaign arms ran on the pin `~/tmp/d123-libpin` (toolkit 69515f37 + the finder); production
runs whatever the tree installs. Between the two, `libWireCellImg.so` (BlobCutting, `25be2a7e`, new
files only, gated byte-identical by its own round) and `libWireCellPytorch.so` (FMFeatureExtract)
changed; Flash, Clus, Match, Sio, Aux and the reco1 plugin are md5-identical to the pin. The gate
below runs the flipped runner **without the pin** — `toolkit/build/*` + `local/lib` as the direnv
environment loads them (md5-identical to each other, `~/tmp/d123/flip/libs.{start,end}.md5`) — so it
covers the config flip and the library drift in one go.

| arm (production libs, no pin) | against | stage A: opflash ×2, frames, icluster ×4 (per group) + pctree, 3 Bee zips (per event) | stage B: pctree-pr, mabc-pr, calib-pr, nusel (per event) | verdict |
|---|---|---|---|---|
| `work-nuecc48-d123flip` — the runner bare after the flip | `work-nuecc48-d123hits` (the campaign's measured arm, pin) | 3/3 groups, **48/48** events identical | **48/48** identical (`calib-pr` after dropping the one wall-clock key `vertex_scoreboard.dual_chain.off_ms`, the only key of 151 221 that differed) | **IDENTICAL** |
| `work-nuecc48-d123flipoff` — `SBND_FLASH_SOURCE=reco1` | `work-nuecc48-d123base` (reco1 flashes, pin) | 3/3 groups, **48/48** identical | — | **IDENTICAL** — the off path is the old production |
| `work-r3cv-d123flip/f000` — one MC file, `--mc` | `work-r3cv-d123hits/f000` | 1/1 group, **18/18** identical | — | **IDENTICAL** |

So the flipped runner reproduces, product for product, the arms every number of §13–§14 came from:
the flip changes nothing but the default, and the two library changes since the pin (Img, Pytorch)
are inert on this chain. Timings: stage A 48 events ≈ 5 min at 3 groups (the hit-flash dump is not
measurably slower than the reco1 one), stage B ≈ 3 min at `PR_JOBS=8`. Logs and md5 lists in
`~/tmp/d123/flip/` (`gate_*.txt`, `libs.{start,end}.md5`, `prodcfg_{pre,post}/`).

### 17.4 What the flip does not cover, and the follow-ups

1. **The LArSoft 1-step chain still matches to `recob::OpFlash`.** `wcls-img-clus-matching-xin.jsonnet`
   reads flashes through `wclsOpFlashSource`; there is no `wclsOpHitSource` in larwirecell (§16 is its
   design, one afternoon of larwirecell work + an sbndcode fcl change + a release). Until it exists the
   two chains run **different light** on purpose; `two_chain_gate.py` compares PR components and is
   blind to it by construction. This is the first follow-up and the only one that gates anything.
2. **The cathode rescue is ON and inert** (4 firings / 1000 events, 3 events change if it is switched off,
   no selection count moves — §15). Retire it, and the round-2/3 extensions with it, only after the two
   fit-vs-geometry events (169824, 59003) have a ruling on MC — not before; a later round, with the
   `hitsnr` arm pattern of §15 as its gate. **→ §18.4: ruled on MC (2026-09-25).**
3. **`QLXTPC coincident` culls a coincident half** (59003, §13.2): with both TPCs' flashes now present in
   the same 80 ns group, the coincident-flash rule can drop the half the charge belongs to. Small
   (1 event in 1000) but it is a bug in a production component, not a tuning; own round. **→ §18.1–18.3:
   understood and fixed behind two default-OFF knobs (2026-09-25).**
4. **SPE-only early flashes** (§11.3, ~1 per event, 1–3 µs before a bright flash): none matched in
   §13–§14, so no cut was added; a `min_fired_pe`-type threshold in the finder if a later census shows
   one adopted by a bundle.
5. **Beam-off fakes 0.9 → 1.3 % per gate** (§14.1) from the fit re-balancing under more flashes: the
   cost side of the flip, to watch in the next beam-off round rather than tune now.
6. **Housekeeping.** `work-*-d123*` stay KEEP until the owner has read §17; the hit-flash
   `opflash_apa*.tar.gz` (every hit with its flash id, 60× the reco1 ones) are the first to drop.
   `work-nuecc48-d123flip{,off,pr}` and `work-r3cv-d123flip` are the flip gate's record. The `nosplit`
   arms (§12) are closed.

## 18. Round 6 — the two follow-ups of §17.4: the `QLXTPC coincident` cull (59003) and the rescue on MC (2026-09-25)

```bash
# the light gate (an existing C++ knob, now a TLA of the Q/L job) and its new over-prediction ceiling, on the production hit flashes
QLTLA=scripts/d123/tla/xtpc_lg.txt   JOBS=4 scripts/d123/hits_arm.sh work-mcp1k-d123base work-mcp1k-d123lg   data   # ks/chi2 gate (pin ~/tmp/d123-libpin)
PIN=~/tmp/d123-libpin-r6 QLTLA=scripts/d123/tla/xtpc_lgop.txt JOBS=4 scripts/d123/hits_arm.sh work-mcp1k-d123base work-mcp1k-d123lgop data   # + overpred ceiling
scripts/d123/mc_hits.sh off lg; scripts/d123/mc_hits.sh cv nr; scripts/d123/mc_hits.sh cv lg; scripts/d123/mc_hits.sh cv lgop; scripts/d123/mc_hits.sh off lgop
scripts/d123/stageB.sh <arm> <data|sim>; scripts/d123/pr_tables.sh <arm>pr products/d123/<name> [cv]
python3 scripts/d123/r3_ql_compare.py work-mcp1k-d123hits work-mcp1k-d123lgop --tsv ql.tsv --events ev.tsv     # Q/L level
python3 scripts/d123/r3_pr_compare.py products/d123/mcp1k_hits products/d123/mcp1k_lgop --label hits,lgop      # selection level
python3 scripts/d123/r6_rescue_ruling.py work-r3cv-d123hits products/d115/cv/truth_base.tsv base=products/d123/r3cv_base hits=products/d123/r3cv_hits nr=products/d123/r3cv_nr
# knob-off gate of the new library: PIN=~/tmp/d123-libpin-r6 JOBS=3 scripts/d123/hits_arm.sh work-nuecc48-d123base work-nuecc48-d123r6off data
#                                   python3 scripts/d123/flip_gate.py work-nuecc48-d123r6off work-nuecc48-d123hits --no-pr   -> IDENTICAL 48/48
```

Owner (2026-09-25): "can you deal with 2 and 3 as you recommended" — §17.4's item 3 (the coincident cull
drops the half the charge belongs to) and item 2 (retire the rescue only after the two fit-vs-geometry
events have a ruling on MC). Both on the flipped production (hit flashes, `SBND_MAX_JOBS`-style lanes at
32 CPUs, the d123 lib pin, every arm on its baseline's imaging through `hits_arm.sh` + one `QLTLA` knob
file). The Bee sets of the beam-off events the owner asked for (§14.1, νμ > 0.9 in the FV in one arm only)
are in §18.5.

### 18.1 The mechanism, read from the logs (59003, `work-mcp1k-d123hits/g25/wct_ql.log`)

`QLMatching::cull_cross_tpc` pairs main-cluster bundles across the cathode whose flashes coincide within the
80 ns group window; a pair whose halves meet within `xtpc_dmax` 5 cm with axes within 20° ("scenario 1")
sets `flag_xtpc_scenario1` on **both** bundles, and `cull_inconsistent` then drops every other bundle of a
cluster that holds such a flag ("cluster kept xtpc scenario-1 crosser"). The flag is set from geometry
alone; the light of the pair is not consulted. In 59003:

| | TPC0 half (ident 3, 2162 pts) | TPC1 half (ident 11, 1799 pts) |
|---|---|---|
| bundles before the cull | beam flash 1.589 µs (meas 6467 PE, pred 5972, KS 0.079, χ²/ndf 1.19); a third flash (3638 PE, KS 0.38, χ²/ndf 17); **−0.733 µs (meas 164 PE, pred 5966, KS 0.13, χ²/ndf 14.6)** | 1.584 µs (9168 PE, pred 7019, KS 0.046, χ²/ndf 2.3); **−0.741 µs (25 331 PE, pred 7019, KS 0.56, χ²/ndf 272)** |
| coincident pair | 0/3 at −0.733 µs ↔ 1/11 at −0.741 µs: d = 2.30 cm, `sc1=true pass=true pin=true` | |
| production cull | drops the two beam-time bundles ("kept xtpc scenario-1 crosser") → cluster 3 ends **unmatched** | drops 1.584 µs → cluster 11 sits on the 25 k PE cosmic flash |
| `xtpc_sc1_light_gate` (KS ≤ 0.3, χ²/ndf ≤ 50 on the bundle's own light) | the −0.733 µs bundle **passes** (KS 0.13, χ²/ndf 14.6): the flash is only 164 PE against a 5966 PE prediction, and a 36× over-prediction is invisible to KS (a shape) and cheap in χ² (the error term scales with the prediction) — the production over-prediction prefilter would have removed it, but exempts `at_x_boundary` bundles, which a cathode-side crosser half always is | the −0.741 µs bundle **fails** (KS 0.56): cluster 11 keeps its beam bundle and the fit moves it to 1.584 µs |
| + `xtpc_sc1_overpred_max` 2.9 (new; deny the flags when pred > 2.9 × meas) | the −0.733 µs bundle fails (36×): no flag on cluster 3, `cull_inconsistent` keeps its high-consistent beam bundles, the fit puts it on 1.589 µs | as above |

The other fit-vs-geometry event, 169824 (§13.2: both halves pulled onto a −3.14 µs cosmic pulse, 22 k PE
of beam-window light left without charge), is the same shape with the light gate alone sufficient: the
TPC0 half at −3.134 µs (32 343 PE, pred 5437) and the TPC1 half at −3.148 µs (14 856 PE, pred 11 092) both
fail the KS gate, and both move to the 1.34 µs beam flash.

### 18.2 What was added (all default-OFF; production byte-identical)

| where | what | proof |
|---|---|---|
| `cfg/pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet` (toolkit) | TLAs `xtpc_sc1_light_gate`, `xtpc_sc1_ks_max`, `xtpc_sc1_c2n_max`, `xtpc_sc1_overpred_max`, all `null` = key omitted; threaded as the `extra` overlay of `qlm.matching()` and `qlm.matching_joint()` | `prod_cfg_gate.py` PASS 28/28 against `prod-2026-09-25` (twice: after the TLAs, after the ceiling); knob-on compile of the mcp1k g25 Q/L job differs from knob-off by exactly the emitted keys (`~/tmp/d123/r6/ql_{off,on,on2}.json`) |
| `match/src/QLMatching.cxx`, `.h` (toolkit) | `xtpc_sc1_overpred_max` (double, default 0 = not tested): inside `sc1_light_pass`, after the KS/χ² test, deny the flags when `total_pred_light > overpred_max × max(flash total PE, 1)`; config round-trip; the calib-dump `qp` key emitted only when > 0 | `wcdoctest-match` 10/10; knob-off byte gate on nueCC-48 with the new library, `work-nuecc48-d123r6off` (hit flashes, no knob, pin `~/tmp/d123-libpin-r6` = d123 pin + this `libWireCellMatch.so`) vs `work-nuecc48-d123hits`: `flip_gate.py --no-pr` **IDENTICAL 48/48** (pctree + 3 mabc zips), `~/tmp/d123/r6/gate_r6off_vs_hits.txt` |
| `sbnd_xin/scripts/d123/` (wcp) | `tla/xtpc_lg.txt`, `tla/xtpc_lgop.txt`; `mc_hits.sh` arms `lg`, `nr`, `lgop`; `r6_rescue_ruling.py` | — |

### 18.3 The light gate on data (mcp1k, 1000 events; beam-off, 1000 gates)

Q/L level (`r3_ql_compare.py`, matched clusters keyed by ident on the shared imaging):

| arm vs `hits` | matched | same | moved | lost / gained | beam-window moved / lost / gained | events changed | rescue firings |
|---|---|---|---|---|---|---|---|
| mcp1k `lg` (gate) | 20 058 → 20 064 | 19 798 | 255 | 5 / 11 | 12 / 0 / 0 | 179 | 4 → 2 |
| mcp1k `lgop` (gate + ceiling) | 20 058 → 20 079 | 19 798 | 258 | 2 / 23 | 13 / 0 / 1 | 185 | 4 → 2 |
| beam-off `lg` | 19 295 → 19 305 | 19 057 | 235 | 3 / 13 | 2 / 0 / 1 | 170 | 5 → 6 |
| beam-off `lgop` (gate + ceiling) | 19 295 → 19 318 | 19 057 | 237 | 1 / 24 | 2 / 0 / 1 | 173 | 5 → 4 |

The gate touches 1.3 % of matched clusters; scenario-1 cull lines fall 3897 → 2224 (mcp1k). The beam-window
changes of `lgop` (13 moved + 1 gained): the four halves of 59003 and 169824 arrive on their beam flash,
and ten clusters leave a beam-window flash for one far outside the window (−567, +511, +596 µs …) — in
production their sc1 flag made the beam-time bundle the *only* one `cull_inconsistent` kept; without the
flag the cull keeps their high-consistent bundles and the fit prefers the far flash. None of the ten changes a selected candidate (the selection compare below
names four events, all accounted for). The two unmatched-adoption rescue firings (49511, 409590) stop because
the halves are now matched; the two same-time merges (395060, 169758) still fire.

Selection level (`r3_pr_compare.py`, stage B on every arm):

| mcp1k | events with a candidate | νμ > 0.9 | changes vs `hits` |
|---|---|---|---|
| `hits` (production) | 470 | 271 | — |
| `lg` | 472 | **273** | +59003 (as its TPC1 half: 154 cm, 482 MeV, νμ 2.21), +169824 (325 cm, 1059 MeV, νμ 5.88 — the reco1-arm candidate back exactly); 49511 gains a 6.5 cm candidate at νμ −2.1, 314705 loses a 5.2 cm one at νμ −1.3 |
| `lgop` | 472 | **273** | the same four events; 59003 now **whole**: 298 cm, νμ 3.65, 819 MeV, vertex (42.6, −15.5, 210.2) — the reco1 arm had νμ 3.20, 821 MeV at the same vertex. `lg` vs `lgop`: 0 flips, 0 candidate changes |

| beam-off | any candidate | reco vertex in FV | νμ > 0.9 | νμ > 0.9 & FV | + Eν > 100 MeV |
|---|---|---|---|---|---|
| reco1 (`base`) | 85 | 22 | 9 | 4 | 4 |
| `hits` (production) | 96 | 24 | 13 | 6 | 6 |
| `lg` | 95 | 24 | 13 | **6** (same 6 events) | 6 |
| `lgop` | 96 | 25 | 14 | **7** (the 6 + 18358/111172) | 7 |

The one beam-off event the ceiling adds, 111172: a 211 cm track on a 93 k PE beam-window flash (1.15 µs)
that the reco1 arm had as a candidate at νμ −2.5 with its vertex out of the FV, and the `hits`/`lg` arms had
as no candidate; under the ceiling its TPC1 partner (ident 3, unmatched before) takes its own 8.1 µs flash,
and the TPC0 track alone reads as a contained candidate at νμ 2.28, vertex (−6, −35, 57) cm. So on data:
the gate alone recovers 169824 whole and 59003 as a half, at zero beam-off cost; the ceiling completes 59003
(Eν 482 → 819 MeV) at **+1 fake per 1000 off-beam gates** (0.6 → 0.7 %).

### 18.4 MC (round-3 inclusive BNB, 2017 events, 557 true νμCC in the FV)

**The rescue's ruling (item 2).** With hit flashes the rescue fires 6 times in 2017 events, all same-time
merges ("new-path-beam"). `work-r3cv-d123nr` is the same Q/L job with `cathode_rescue=false`,
`cathode_rescue_unmatched=false` (§15's `hitsnr` pattern), stage B, `pr_tables.sh … cv`, `d107_selection.py`:

| inclusive MC | `hits` (rescue ON, production) | `nr` (rescue OFF) |
|---|---|---|
| events with a candidate / νμ > 0.9 / flips | 965 / 563 / — | 965 / 563 / **0** |
| νμCC efficiency (true νμCC in FV: 557) | 400 = 71.8 % | 399 = 71.6 % |
| purity | 400/461 = 86.8 % | 399/460 = 86.7 % |
| vertex-matched FV candidates | 472 (84.7 %) | 471 |
| νe > 7 / νe > 4 | 6 / 11 | 6 / 11 |

Per firing (`r6_rescue_ruling.py`, `~/tmp/d123/r6/rescue_ruling_r3cv.txt`; "ON" = `hits`, "OFF" = `nr`):

| file / run / event | the merge | truth | ON | OFF | ruling |
|---|---|---|---|---|---|
| f015 719/1/2 | c11 (75.7 cm, 1.64 µs) + c9 (120.8 cm, on a 0.495 µs flash) | νμCC QE 745 MeV at T 1.49 µs | the 194 cm track, νμ 4.38, vertex 1.4 cm from truth, Eν 746 | only the 75.7 cm piece, νμ 0.66, vertex 108 cm off, Eν 308 | **rescue right**: the far half had taken a wrong flash; this is the −1 signal candidate of the table (the event still passes on its second νμCC) |
| f012 715/79/26 | c3 (34.5 cm) + c13 (11.7 cm), both at 0.70 µs | νμCC QE 753 MeV at T 0.56 µs | **no candidate** (a 1.5 cm fragment at νμ −1.9) | the 34.5 cm cluster, vertex 4.95 cm from truth, νμ −0.02, Eν 335 | **rescue wrong**: the merge takes the neutrino's cluster out of the candidate role (below the cut either way) |
| f120 713/98/36 | c2 (73.5 cm, 0.527 µs) + c15 (310 cm, 1.364 µs) | two νμCC: 1054 MeV at T 0.39 µs in the FV, 1976 MeV at T 1.20 µs at z = −415 cm (entering muon) | the FV one, νμ 2.20, vertex 0.16 cm, Eν 269 | the FV one, νμ 2.81, Eν 270, plus the 310 cm entering muon as a second candidate (z = 3.9 cm, out of FV) | **rescue wrong in physics** (two interactions 0.8 µs apart merged into one bundle), no selection effect |
| f066 711/20/13 | c2 (211 cm, 1.469 µs) + c9 (20 cm, **t0 787 µs**) | νμCC MEC 1262 MeV | νμ 0.85, Eν 1451 | νμ 0.74, Eν 1379 | wrong in physics (an out-of-time cluster merged into the beam bundle), +72 MeV, below the cut either way |
| f078 719/81/38 | c3 (16.4 cm) + c28 (177.5 cm), 1.51 µs | νμCC QE 735 MeV | νμ 5.22, Eν 693 | νμ 3.23, Eν 524 (the 172.6 cm piece on the TPC1 flash) | rescue right (energy) |
| f108 720/10/48 | c10 (113 cm, 0.758 µs) + c21 (5.4 cm, 0.965 µs) | νμCC RES 688 MeV | νμ 2.81, Eν 887 | νμ 3.11, Eν 1017 | neither near the truth (the reco1 arm had 704); rescue marginally better |

Two of six firings are the intended cathode re-join (f015, f078); two merge activity that does not belong
together (f120, f066); one removes the neutrino's own cluster (f012); one is noise. Net at the selection:
**+1 signal candidate in 2017 events for the rescue ON, purity unchanged** — inert at the counting level,
as on data (§15), and its geometry-only rule does produce wrong merges. This is the ruling the owner asked
for before retiring it: no measurable loss from retiring it on MC (−1/557 = −0.2 pt, inside the
[69.9, 73.7] interval), no measurable gain from keeping it.

**The light gate on MC (item 3).** `work-r3cv-d123lg` (gate) and `work-r3cv-d123lgop` (gate + ceiling, pin r6),
each vs the production `hits` arm:

| inclusive MC | `hits` | `lg` (gate) | `lgop` (gate + ceiling) |
|---|---|---|---|
| Q/L: matched clusters / moved / lost / gained | 35 062 | 35 068 / 623 / 5 / 11 | 35 075 / 630 / 3 / 16 |
| Q/L: beam-window moved / lost / gained; rescue firings | — | 12 / 0 / 0; 6 → 7 | 12 / 0 / 0; 6 → 7 |
| events with a candidate / νμ > 0.9 | 965 / 563 | 965 / 562 | 965 / 562 |
| flips at νμ > 0.9 (hits-only / arm-only) | — | 1 / 0 (713/67/7) | 1 / 0 |
| candidate vertex moved > 5 cm | — | 2 (the same two events) | 2: **719/1/34** νμCC DIS 3.4 GeV — a 43.6 k PE cosmic crosser that production kept in the neutrino's bundle (sc1 flag) leaves for its own 24 µs flash; the vertex goes from 169 cm off to **1.07 cm**, νμ 1.96 → 5.30, Eν 869 → 1417 MeV (**+1 signal**). **713/67/7** νμCC QE 605 MeV — a crosser pair that production had on two far flashes (−264, +250 µs) arrives on the beam-window flash pair (1.61 / 1.71 µs) and shares the neutrino's bundle; the vertex moves 10 cm off, νμ 1.30 → 0.46 (**−1 signal**) |
| νμCC efficiency (557 true in FV) | 400 = 71.8 % | 400 = 71.8 % | **400 = 71.8 %** |
| purity | 400/461 = 86.8 % | 400/460 = 87.0 % | **400/460 = 87.0 %** |
| νe > 7 | 6 | 6 | 6 |

On MC the gate is a wash at the efficiency (+1 −1 vertex-matched signal, both through a cosmic crosser
entering or leaving the neutrino's bundle) and −1 background at the purity; the cosmic-side re-assignments
(1.8 % of matched clusters) do not reach the selection otherwise. The MC sample holds no instance of the
59003/169824 topology that the gate recovers on data: the two §14.2 losses with a vertex jump (717/29/47,
719/81/47) are unchanged by the gate and by the rescue, so they are not this mechanism.

### 18.5 The beam-off events the owner asked to see (§14.1, νμ > 0.9 & FV in one arm only)

Same four gates in the same order in both sets, `make_pr_bee.py` (Q/L layers + the PR layers where the arm
selected a candidate):

| Bee index | run/event | reco1 arm | hits arm |
|---|---|---|---|
| 0 | 18482/444203 | νμ 3.07, 259 cm, vertex (−6, −11, 58) cm, flash 0.80 µs TPC0 | only a 1.8 cm candidate: the long track left the beam flash |
| 1 | 18503/179510 | same 198 cm cluster, νμ 0.64 | same cluster and flash, νμ 1.16 |
| 2 | 18269/197600 | no candidate | νμ 1.49, 185 cm, flash 2.03 µs TPC1 |
| 3 | 18390/742853 | no candidate | νμ 2.02, 256 cm, flash 1.50 µs TPC1 |

- reco1 arm (`work-r3off-d123basepr`): <https://www.phy.bnl.gov/twister/bee/set/d2b6785a-cf95-4512-8cac-332dbb49ef03/event/list/>
- hits arm (`work-r3off-d123hitspr`): <https://www.phy.bnl.gov/twister/bee/set/e2a0657d-4c5f-484d-9b6a-a30f53148dcd/event/list/>

Under the light gate (`work-r3off-d123lgpr`) and under the gate + ceiling (`work-r3off-d123lgoppr`) all four are unchanged.

### 18.6 Recommendation

**Item 3 — the coincident cull.** The defect is understood (§18.1) and closed behind two default-OFF knobs.
The configuration to adopt is the gate **with** its ceiling (`xtpc_sc1_light_gate=true`,
`xtpc_sc1_overpred_max=2.9`; the KS/χ² thresholds at their C++ defaults 0.3 / 50): on data it recovers both
§13.2 losses whole and costs nothing above the νμ cut on mcp1k (271 → 273); on MC it is a wash at the
efficiency (400 → 400) and +0.2 pt at the purity; the two knobs are byte-identical off (§18.2). `lg` and
`lgop` differ in two events only: 59003 (whole vs half, +337 MeV of the neutrino's energy) and the beam-off
fake 111172 (6 → 7 per 1000 gates); the MC does not separate them (0 flips). The ceiling is the physically
right rule (a flash of 164 PE cannot be the light of a 5966 PE prediction, boundary or not) and the choice
between the two is the owner's — I would take the ceiling. Flipping the
two TLA defaults in `wct-clus-matching-perevt.jsonnet` is a production change (the compiled Q/L job moves,
`ref/prod-<date>` must be refreshed, and the runner needs the off switch) — **the owner's decision**; until
then both are reachable from any arm through `QLTLA=scripts/d123/tla/xtpc_lgop.txt`. Cost to state with a
flip: 1.3 % (data) / 1.8 % (MC) of matched clusters change flash, all outside the selection except the
events named above; mcp2k, nueCC-48 and NCpi0 were not re-run in this round (mcp1k, beam-off and the
inclusive MC were), so a flip gate should include them.

**Item 2 — the rescue.** The MC ruling is in (§18.4): with hit flashes it fires 6 times in 2017 events, two
of them the intended cathode re-join, two merging activity that does not belong together, one removing the
neutrino's own cluster; net **+1 signal candidate for the rescue ON, purity unchanged**. It is inert at the
counting level on data (§15) and on MC, and it is not the mechanism behind the §14.2 vertex-jump losses.
Under the light gate it fires less on data (4 → 2: the two unmatched adoptions stop because the halves are
matched) and once more on MC (6 → 7). Recommendation: keep it ON through the light-gate flip (the owner's
standing decision, zero cost), then retire `cathode_rescue`, `cathode_rescue_unmatched` and the round-2/3
extensions together in the production refresh that follows, with one arm (`lgop` + `rescue_off.txt`) as the
gate — expected effect −1 signal per 2000 MC events, inside the interval, and two wrong merges fewer.

**Not done in this round:** `min_fired_pe` (§17.4 item 4, no candidate sits on an SPE flash, unchanged); the
larwirecell hit source (§16); the retire itself.


### 18.7 Bee sets for the colleagues: the νμ > 0.9 flips on the 3000 data events, old (reco1 flashes) vs new (hit flashes)

The 784 → 784 of §13.3 is 23 events lost and 23 gained (`123_r3_pr_flips_{mcp1k,mcp2k}.tsv`). Four sets,
`make_pr_bee.py` on the `d123base{,pr}` and `d123hits{,pr}` arms of mcp1k + mcp2k, the same events in the
same order in the old and new set of each pair (Q/L layers everywhere; PR layers where the arm selected a
candidate: 6 of the lost have none in the new arm, 9 of the gained none in the old). νμ −999 = no candidate.

**Lost with the flip** (pass at νμ > 0.9 with reco1 flashes only):
- old, reco1 flashes: <https://www.phy.bnl.gov/twister/bee/set/36c60c1f-dd1b-42b8-8077-9b8919fa9036/event/list/>
- new, hit flashes: <https://www.phy.bnl.gov/twister/bee/set/6a09d85a-c461-4155-8127-5dc02be26e83/event/list/>

| Bee index | sample run/event | reco1: νμ / Eν (MeV) | hits: νμ / Eν (MeV) | mechanism (§13.2 classes, mcp1k only) |
|---|---|---|---|---|
| 0 | mcp1k 18255/59003 | 3.195 / 820.7 | -999.000 / 0.0 | big cluster left beam (fit); big cluster joined beam (fit) |
| 1 | mcp1k 18255/59261 | 3.845 / 437.9 | 0.155 / 478.7 | small moves only (2 out/0 in) |
| 2 | mcp1k 18255/60933 | 3.749 / 712.9 | 0.826 / 756.1 | small moves only (0 out/0 in) |
| 3 | mcp1k 18255/62459 | 3.301 / 906.6 | 0.771 / 812.6 | small moves only (0 out/1 in) |
| 4 | mcp1k 18255/65053 | 2.877 / 512.7 | -0.315 / 608.6 | big cluster joined beam (fit) |
| 5 | mcp1k 18255/65999 | 2.103 / 275.6 | -1.138 / 147.6 | small moves only (1 out/1 in) |
| 6 | mcp1k 18259/169824 | 5.645 / 1059.0 | -999.000 / 0.0 | big cluster left beam for a restored flash |
| 7 | mcp1k 18259/174422 | 1.404 / 429.0 | -999.000 / 0.0 | big cluster left beam (fit) |
| 8 | mcp1k 18255/277298 | 1.282 / 686.9 | 0.894 / 555.5 | small moves only (1 out/0 in) |
| 9 | mcp1k 18255/281639 | 1.370 / 437.3 | 0.602 / 436.7 | small moves only (0 out/0 in) |
| 10 | mcp2k 18255/58006 | 0.987 / 469.0 | 0.737 / 468.4 |  |
| 11 | mcp2k 18255/71642 | 2.264 / 1522.4 | 0.636 / 1796.5 |  |
| 12 | mcp2k 18259/157215 | 1.476 / 753.5 | -999.000 / 0.0 |  |
| 13 | mcp2k 18259/158227 | 3.713 / 629.5 | -999.000 / 0.0 |  |
| 14 | mcp2k 18259/161725 | 2.241 / 586.0 | 0.854 / 588.7 |  |
| 15 | mcp2k 18255/281985 | 2.255 / 521.9 | 0.395 / 292.1 |  |
| 16 | mcp2k 18255/390743 | 1.793 / 502.2 | -2.170 / 63.1 |  |
| 17 | mcp2k 18255/391777 | 1.145 / 729.0 | 0.893 / 730.4 |  |
| 18 | mcp2k 18255/402297 | 1.284 / 929.4 | -999.000 / 0.0 |  |
| 19 | mcp2k 18255/410566 | 1.051 / 104.1 | 0.781 / 103.1 |  |
| 20 | mcp2k 18255/415564 | 1.879 / 476.5 | -2.294 / 19.6 |  |
| 21 | mcp2k 18255/480970 | 1.618 / 488.6 | 0.443 / 628.4 |  |
| 22 | mcp2k 18255/497399 | 2.080 / 1170.8 | -2.580 / 510.9 |  |

**Gained with the flip** (pass with hit flashes only):
- old, reco1 flashes: <https://www.phy.bnl.gov/twister/bee/set/4fa42488-5669-48ec-96a9-fbbf39db4bb0/event/list/>
- new, hit flashes: <https://www.phy.bnl.gov/twister/bee/set/87f3c3dc-0f0b-472a-8b14-9ab339848200/event/list/>

| Bee index | sample run/event | reco1: νμ / Eν (MeV) | hits: νμ / Eν (MeV) | mechanism (§13.2 classes, mcp1k only) |
|---|---|---|---|---|
| 0 | mcp1k 18255/59929 | -0.410 / 460.1 | 1.896 / 566.8 | small moves only (0 out/0 in) |
| 1 | mcp1k 18259/68428 | -999.000 / 0.0 | 2.539 / 818.3 | small moves only (0 out/0 in) |
| 2 | mcp1k 18259/170792 | 0.504 / 658.8 | 0.984 / 828.5 | small moves only (1 out/0 in) |
| 3 | mcp1k 18259/172832 | 0.635 / 775.5 | 3.098 / 633.5 | small moves only (1 out/0 in) |
| 4 | mcp1k 18255/280884 | -999.000 / 0.0 | 4.400 / 675.3 | big cluster joined a new beam flash |
| 5 | mcp1k 18255/281808 | -999.000 / 0.0 | 2.961 / 556.7 | big cluster joined a new beam flash |
| 6 | mcp1k 18255/284211 | -0.573 / 204.2 | 2.786 / 639.3 | small moves only (2 out/0 in) |
| 7 | mcp1k 18255/285531 | 0.467 / 1207.5 | 4.244 / 955.8 | small moves only (0 out/0 in) |
| 8 | mcp1k 18255/285665 | 0.550 / 418.9 | 1.026 / 427.8 | small moves only (2 out/0 in) |
| 9 | mcp1k 18255/390644 | -999.000 / 0.0 | 2.828 / 475.3 | big cluster joined a new beam flash |
| 10 | mcp2k 18255/53749 | -999.000 / 0.0 | 1.599 / 871.3 |  |
| 11 | mcp2k 18255/66944 | 0.034 / 355.7 | 1.774 / 414.2 |  |
| 12 | mcp2k 18255/67868 | 0.475 / 1089.7 | 3.252 / 1016.9 |  |
| 13 | mcp2k 18255/77846 | -0.559 / 156.7 | 3.175 / 639.8 |  |
| 14 | mcp2k 18255/105338 | -999.000 / 0.0 | 1.421 / 676.8 |  |
| 15 | mcp2k 18259/163595 | -999.000 / 0.0 | 4.003 / 718.8 |  |
| 16 | mcp2k 18259/165060 | -999.000 / 0.0 | 1.294 / 1107.4 |  |
| 17 | mcp2k 18259/169724 | 0.665 / 428.7 | 1.166 / 731.2 |  |
| 18 | mcp2k 18255/275385 | -0.305 / 240.9 | 4.216 / 481.5 |  |
| 19 | mcp2k 18255/293536 | -1.066 / 249.9 | 0.983 / 342.1 |  |
| 20 | mcp2k 18255/393538 | 0.363 / 332.9 | 1.082 / 725.3 |  |
| 21 | mcp2k 18255/396222 | 0.887 / 3565.5 | 0.931 / 3573.7 |  |
| 22 | mcp2k 18255/399963 | -999.000 / 0.0 | 2.591 / 698.5 |  |

Lost index 0 (59003) and 6 (169824) are the two the light gate of §18.3 gives back; under the recommended
knobs the new set would show them as in the old.

**The anatomy of these 46 flips is doc 124** (`docs/124_sbnd-hitflash-numu-flip-anatomy.md`): 17 are Q/L-driven
(9 recovered neutrinos, 3 cosmic candidates removed, 1 cosmic added, 4 real losses of which the light gate
recovers 59003 and 169824), 29 are a tie-sensitivity of the PR stage to inputs that should not matter (far specks
in the bundle, a 10 ns flash-time shift, cluster numbering) that is symmetric on MC truth.

## 19. The second flip — the QLXTPC scenario-1 light gate + ceiling ON in production (2026-09-25)

> "Flip the light gate on for production" — the owner, 2026-09-25

### Repro

```bash
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin; cd $SX
# proofs on the tripwire's Q/L TLA list: pre = git archive HEAD cfg (toolkit, before the flip)
T=(-A input=/x/clusters-apa.tar.gz -S "anode_indices=[0,1]" -A output_dir=/x -S run=18255 -S subrun=1 -S event=287517)
WIRECELL_PATH=<pre>/cfg:$DATA  wcsonnet "${T[@]}" -S xtpc_sc1_light_gate=true -S xtpc_sc1_overpred_max=2.9 -o preA.json wct-clus-matching-perevt.jsonnet
WIRECELL_PATH=$TK/cfg:$DATA    wcsonnet "${T[@]}" -o flip.json wct-clus-matching-perevt.jsonnet                     # A: diff preA flip = 0
WIRECELL_PATH=<pre>/cfg:$DATA  wcsonnet "${T[@]}" -o pre.json wct-clus-matching-perevt.jsonnet
WIRECELL_PATH=$TK/cfg:$DATA    wcsonnet "${T[@]}" -S xtpc_sc1_light_gate=null -S xtpc_sc1_overpred_max=null -o off.json wct-clus-matching-perevt.jsonnet   # B: diff pre off = 0
python3 scripts/cfg/prod_cfg_gate.py                                      # PASS 28/28 vs prod-2026-09-25b
# output gate: the flipped runner on the PRODUCTION libs (no pin), mcp1k groups 0, 25 (59003), 50 (169824), and one MC file
SBND_QL_KEEP_ICLUSTER=1 SBND_MAX_JOBS=3 ./run_chain_group.sh input_files_reco1/data_MCP2025C_reco1_frameshift_first1000ev.root work-mcp1k-d123lgflip data --size 16 --layout perevt --groups 0,25,50
SBND_XTPC_SC1_GATE=0 SBND_QL_KEEP_ICLUSTER=1 SBND_MAX_JOBS=3 ./run_chain_group.sh <same> work-mcp1k-d123lgflipoff data --size 16 --layout perevt --groups 0,25,50
SBND_QL_KEEP_ICLUSTER=1 SBND_MAX_JOBS=1 ./run_chain_group.sh $(awk -F'\t' 'NR==1{print $2}' products/d115/cv/files.lst) work-r3cv-d123lgflip/f000 sim --mc --size 1000 --layout perevt
PR_EXTRA_STAGES=pr_display PR_JOBS=12 ./run_pr_chain_batch.sh work-mcp1k-d123lgflip work-mcp1k-d123lgflippr data
python3 scripts/d123/flip_gate.py work-mcp1k-d123lgflip    ~/tmp/d125flip/view/mcp1k_lgop            # views = symlinks of g0/g25/g50 + their events
python3 scripts/d123/flip_gate.py work-mcp1k-d123lgflipoff ~/tmp/d125flip/view/mcp1k_hits --no-pr
python3 scripts/d123/flip_gate.py work-r3cv-d123lgflip/f000 work-r3cv-d123lgop/f000 --no-pr
# the samples never run with the gate: nueCC-48 and NCpi0 through the flipped production runner (~/tmp/d125flip/chain_small.sh)
python3 scripts/d123/r3_pr_compare.py products/d123/<s>_hits products/d123/<s>_lgflip --label hits,lgflip
```

### 19.1 What changed

| repo / file | change |
|---|---|
| toolkit `cfg/pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet` | TLA defaults `xtpc_sc1_light_gate` null → **true**, `xtpc_sc1_overpred_max` null → **2.9**. The KS/χ² thresholds stay null (the C++ 0.3 / 50 that §18 measured). The comment states the flip, the date and the escape. |
| wcp `run_chain_group.sh` | env `SBND_XTPC_SC1_GATE=0\|1`. Unset ⇒ no TLA (the jsonnet default, ON). `=0` passes `null` for both, so the keys are omitted and the pre-flip Q/L graph compiles byte for byte. `=1` passes the production values. `QL_EXTRA_TLA` still comes last and wins. |
| wcp `run_ql_evt.sh` | the same escape for the per-event driver (next to `SBND_CATHODE_RESCUE`). Syntax-checked; not exercised by a run in this round. |
| wcp `ref/prod-2026-09-25b/` | new tripwire generation. `prod-2026-09-25` is kept. |

The C++ defaults (`QLMatching`: gate false, ceiling 0) are unchanged, so PDVD and PDHD, which share the class,
are not touched. The production library is `libWireCellMatch.so` md5 d57af13f, the same file as the
measured pin `~/tmp/d123-libpin-r6` and `toolkit/build`, so no rebuild was needed. Not changed: the reco1 dump,
the PR job, every rescue knob (still ON, §18.6), and the LArSoft 1-step chain. That chain builds its Q/L node
from `qlmatching.jsonnet` directly; its tripwire artifact `sbnd_larsoft_1step.json` did not move.

### 19.2 Proofs on the compiled config

| proof | compared | result |
|---|---|---|
| **A** | pre-flip tree + the two knobs vs flipped tree bare (tripwire TLA list) | **0 lines** |
| **B** | pre-flip tree bare vs flipped tree + the escape (`null`, `null`) | **0 lines** |
| explicit `=1` | flipped + `true` / `2.9` vs flipped bare | 0 lines |
| flipped vs pre-flip bare | | exactly `"xtpc_sc1_light_gate": true` and `"xtpc_sc1_overpred_max": 2.9` on the QLMatching node |
| runner-precompiled `.wct-cfg-ql.json` (paths normalised) | flipped vs `work-mcp1k-d123lgop` (g0, g25, g50); MC f000 vs `work-r3cv-d123lgop/f000`; escape vs `work-mcp1k-d123hits` | **0 keys** each; flipped vs `hits` = the two knob keys |
| `prod_cfg_gate.py` vs `prod-2026-09-25` | 28 artifacts | **DRIFT: `sbnd_ql.json` only**; after `--refresh --ref ref/prod-2026-09-25b`: **PASS 28/28** (the manifests differ in that one line) |

### 19.3 Output gate — the flipped runner on the production libraries

Libraries (`local/lib` + `toolkit/build`) are md5-identical at start and end (`~/tmp/d125flip/libs.{start,end}.md5`).
The views `~/tmp/d125flip/view/mcp1k_{lgop,hits}{,pr}` are symlinks to the measured arms' groups g0, g25, g50
and their 48 events.

| arm (production libs, no pin) | against | stage A: opflash ×2, frames, icluster ×4 per group; pctree + 3 Bee zips per event | stage B: pctree-pr, mabc-pr, calib-pr, nusel per event | verdict |
|---|---|---|---|---|
| `work-mcp1k-d123lgflip`, the runner bare after the flip | `work-mcp1k-d123lgop` (the §18 measured arm, pin r6) | 3/3 groups, **48/48** events identical | pctree-pr, mabc-pr, nusel **48/48**; calib-pr **33/33** present identical, 15 absent in **both** (no-candidate events write none; `flip_gate.py` counts a both-absent file as "missing", so its verdict line reads DIFFER) | **IDENTICAL** |
| `work-mcp1k-d123lgflipoff`, `SBND_XTPC_SC1_GATE=0` | `work-mcp1k-d123hits` (the §17 production before this flip) | 3/3 groups, **48/48** identical | — | **IDENTICAL**: the off path is the previous production |
| `work-r3cv-d123lgflip/f000`, one MC file, `--mc` | `work-r3cv-d123lgop/f000` | 1/1 group, **18/18** identical | — | **IDENTICAL** |

The effect is visible where the knob acts: 59003 (g25) has no candidate with hit flashes and **νμ 3.65, 819 MeV**
after the flip, and 169824 (g50) has none and then **νμ 5.88, 1059 MeV**, as in §18.3.

### 19.4 The samples not run with the gate before: nueCC-48 and NCpi0

The flipped production runner, stage A + B, was set against the hit-flash arms of §13
(`work-{nuecc48,ncpi0}-d123hits{,pr}`, same imaging stage by construction). Files:
`docs/123_flash/123_r7_pr_{nuecc48,ncpi0}_hits_lgflip.json`.

| sample | Q/L: events with a changed match | beam-window matches lost / gained | selection, hits → flipped |
|---|---|---|---|
| nueCC-48 (48 events) | 9 | 0 / 0 | νμ > 0.9 5 → 5; **νe > 7: 37 → 38** (+350186: νe −5.9 → 9.5, vertex moved 31 cm), nothing lost |
| NCpi0 (19 events) | 7 | 0 / 0 | unchanged (νμ 2 → 2, νe 1 → 1); one candidate vertex moved (180801, same scores) |

The gate touches a handful of cluster matches per sample, none in the beam window, and changes no selected
event for the worse.

### 19.5 What the flip does not cover

- **The LArSoft 1-step chain** (`wcls-img-clus-matching-xin.jsonnet`) still runs reco1 flashes and no light gate (§16, §17.4).
- **The rescue** stays ON. Its retirement is the follow-up of §18.6: one arm with the gate and `rescue_off.txt` as its gate.
- **The PR stage's tie-sensitivity** (doc 124 §3, §7) is untouched by any flash change. It is the source of the
  symmetric ~1 % churn on the selected count.
