# 72 — In-beam tracks cut at the cathode: PR only gets one TPC's half

Owner request: find data events where a long track crosses the central cathode but
only the half in one TPC reaches pattern recognition, so the reconstructed track
stops at x = 0.

**Round 1 of this doc answered the wrong question and its numbers are withdrawn.**
It searched over *all* flashes and ranked by the displacement between the two halves.
That population is overwhelmingly cosmic, and the owner's exemplar check killed it:
for evt409546 the beam-flash-matched image sits at (−200.3, 47.5, 478.2), i.e. at the
anode, while the flagged track was a cosmic at t0 = −1111 µs. §A below is the correct
search; §B keeps the cosmic result as a separate, clearly-labelled finding.

**Doc number caveat**: next free top-level number on 2026-08-17; a concurrent session
may have claimed it (cf. the pr/83 collision).

---

## §A — The in-beam search

### Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 scripts/analysis/cathode/beam_cathode_split.py \
        --arm work-mcp1k-cb0805 --out products/beam-cathode-split-mcp1k.tsv --jobs 8
python3 scripts/analysis/cathode/beam_cathode_split.py \
        --arm work-mcp2k-cb0816 --out products/beam-cathode-split-mcp2k.tsv --jobs 8
python3 scripts/analysis/cathode/beam_cathode_split_plot.py \
        -w work-mcp1k-cb0805 -w work-mcp2k-cb0816 \
        -t products/beam-cathode-split-mcp1k.tsv \
        -t products/beam-cathode-split-mcp2k.tsv -o pics/beamsplit-top10.png
python3 scripts/bee/merge_mabc_bee.py -w work-mcp2k-cb0816 -w work-mcp1k-cb0805 \
        -o bee/beamsplit/beamsplit-top10.zip \
        398115 237798 65289 493439 78242 65053 51128 317427 319913 167964
./upload-to-bee.sh bee/beamsplit/beamsplit-top10.zip
```

Read-only; nothing written into any `work-*` dir.

### Why no T0 arithmetic appears anywhere

The beam window is `[0.2, 2.2) µs`
(`cfg/pgrapher/experiment/sbnd/clus.jsonnet:518`), so the T0 correction a beam-matched
cluster receives is `|dx| = v_drift·t0 ≤ 0.34 cm` — sub-blob. **The raw image already
shows one consistent track through the cathode**, which is exactly how the owner
reads it off Bee `img-global`. The whole search therefore runs in raw image
coordinates with no shift applied to anything. (Round 1's mirror shift, `disp` metric
and `other-flash`/`UNMATCHED` taxonomy were artefacts of the cosmic scope and do not
apply here.)

### What PR actually receives

The PR job runs `unmerge_bundle, unmerge_assoc` before Steiner (doc 53, doc 45), which
strips the associated clusters. So the object under test is the in-beam bundle's
**main**:

```
nusel-table.tsv row with in_beam=1 and label != 'no-bundle'  ->  main_id
main_id is a `clustering-global` cluster id
```

**Verified, not assumed.** evt409546: clustering-global clusters 6/7/9/11 hold
956/196/283/2708 points, and the nusel rows for main_id 6/7/9/11 carry npts_main
956/196/283/2708. Re-checked on 42 mcp2k in-beam mains: 42/42 exact.

`op_cluster_ids` is deliberately **not** used for bundle membership. It is a list of
*img* cluster ids and agrees with `n_bundle` on only 54 % of in-beam events (300
measured), so it would invent missing halves on events where Q/L did the right thing.

### Method

1. K = the in-beam main.
2. For **each** side of the cathode on which K has ≥ 50 points: fit a local axis at
   K's end nearest the cathode plane; require it track-like (residual RMS < 3 cm),
   within 10 cm of the plane, and heading across (`|dir_x| ≥ 0.3`). Extrapolate to
   x = 0 for the predicted piercing (y, z).
3. In the raw image, take every img cluster of the **other** TPC and ask whether it
   starts at the cathode (≤ 8 cm), pierces x = 0 within 15 cm of the prediction, and
   runs parallel within 20°.
4. A continuation already inside K is **`already-in-main`** — the beam crosser the
   chain handled correctly, and the control. One outside K is the defect.
5. Report its extent inside a 5 cm corridor around its own axis: the missing length.

Step 2 deliberately does **not** skip a main that merely *has* charge in both TPCs.
An in-beam main can be a scattered multi-fragment object (evt409546 main 9: 7
fragments, x from −201 to +167), so "spans both TPCs" ≠ "the continuation at this
cathode end is present". Taking it as such skipped 430 of 837 in-beam mains wholesale
and would have hidden a defect behind an unrelated fragment.

**One false-positive class was found and is filtered, not hand-waved.** If the same
main *does* have a correct continuation somewhere, it is not a one-sided track: its
other cathode end fit picked a different local fragment and the "missing" partner is
an unrelated track. evt407798 is the measured instance — main 25 carries
`already-in-main` on side 1 (gap 2.0 cm, 0.3°) and a spurious hit on side 0 (gap
13.0 cm, 13.2°). Such rows are demoted to `MISSING-main-also-crosses`, not deleted.

### Result

| sample | events | in-beam mains | candidate cathode ends | control `already-in-main` | **MISSING** |
|---|---|---|---|---|---|
| mcp1k | 1000 | 837 | 162 | 101 rows / 56 events | **2 events** |
| mcp2k | 2000 | 1671 | 337 | 200 rows / 106 events | **8 events** |

So **10 events in 3000**, against 162 events where the chain joined the two halves
correctly: roughly **6 % of in-beam cathode crossers lose a side**. This is a much
smaller population than round 1's — correctly, since it is one bundle per event rather
than every flash.

Also counted rather than dropped: 185 candidate ends were `isochronous`
(`|dir_x| < 0.3`, no lever arm, prediction meaningless) and 185 had
`no-continuation-found` — tracks that genuinely stop at the cathode, or whose
continuation fell in a dead region (see Limitations).

### The failure is never "no flash"

Every one of the 12 MISSING rows has the other half matched to *something*
(`p_matched = 1`). Two sub-classes:

* **8 events — matched to a cosmic flash.** Q/L put the other half in the wrong
  bundle, so it never joins the in-beam main.
* **2 events (398115, 237798) — matched to the beam flash, in a different cluster.**
  Q/L got the flash right and the two halves were *still* never joined into one
  cluster. On evt237798 both halves are separately in-beam mains (main 11 with 5842
  points and main 9 with 1505; main 9's points are exactly the partner main 11 points
  at) — so PR ran twice, each time on half a track. That is a `cathode_connect` /
  clustering failure, not a Q/L one.

### Top 10 — Bee set `e9a73b53-6124-4db9-841e-0bd831417ba1`

Index `bee/beamsplit/beamsplit-top10.index.txt`, panels `pics/beamsplit-top10.png`.

| bee | sample | event | label | missing cm | main cm | gap2d cm | angle | beam t µs | other half on |
|---|---|---|---|---|---|---|---|---|---|
| 0 | mcp2k | 398115 | nu-candidate | 325 | 117 | 5.8 | 2.0° | 1.86 | beam flash, other cluster |
| 1 | mcp2k | 237798 | nu-candidate | 141 | 251 | 0.5 | 4.7° | 1.57 | beam flash, other cluster |
| 2 | mcp1k | 65289 | nu-candidate | 88 | 187 | 6.3 | 13.2° | 1.75 | cosmic flash |
| 3 | mcp2k | 493439 | nu-candidate | 79 | 36 | 1.2 | 1.2° | 0.99 | cosmic flash |
| 4 | mcp2k | 78242 | nu-candidate | 74 | 168 | 2.1 | 1.5° | 1.91 | cosmic flash |
| 5 | mcp1k | 65053 | nu-candidate | 74 | 46 | 6.0 | 1.7° | 0.73 | cosmic flash |
| 6 | mcp2k | 51128 | nu-candidate | 54 | 109 | 2.7 | 9.0° | 1.67 | cosmic flash |
| 7 | mcp2k | 317427 | nu-candidate | 48 | 287 | 0.5 | 6.7° | 0.91 | cosmic flash |
| 8 | mcp2k | 319913 | nu-candidate | 9 | 54 | 3.0 | 15.6° | 0.74 | cosmic flash |
| 9 | mcp2k | 167964 | TGM | 8 | 226 | 2.6 | 6.3° | 2.03 | cosmic flash |

Rows 8 and 9 have under 10 cm of missing track and are marginal — kept in so the
owner can set the relevance floor rather than inherit one. On the panels, evt51128
(row 6) is the one whose two pieces are hardest to read as a single track by eye.

**Hand-check.** evt65289 and evt65053: the union of the main and the partner,
restricted to |x| < 60 cm, fits a single straight line with mean residual 2.1 / 2.4 cm
over 152 / 167 cm of track.

### Limitations

* `img-global` carries the **active** imaged blobs only. A continuation falling in a
  dead region lives in `icluster-apa*-masked.npz` and is invisible here — part of the
  185 `no-continuation-found` ends.
* Cuts are first-pass; every value is written per row in the TSV so the population can
  be re-cut offline without re-running.
* The in-beam bundle main is used as a proxy for "what PR received". It is the right
  object by construction (`unmerge_assoc`), but this was not cross-checked against
  the PR output tree itself.

---

## §B — The cosmic-scope result (round 1), retained separately

The same machinery over **all** flashes finds a different and much larger class: a
cathode crosser whose two halves land on different flashes, so the T0 correction
places one of them 20–190 cm from where it belongs. Scripts
`scripts/analysis/cathode/cathode_halfmatch{,_plot}.py`, TSVs
`products/cathode-halfmatch-mcp{1,2}k.tsv`, Bee set
`fb72bebd-5724-414d-9265-6d3fc51e918f`.

| sample | crossers (control) | `other-flash` | `UNMATCHED` | events, displacement ≥ 20 cm |
|---|---|---|---|---|
| mcp1k | 4571 | 47 | 3 | 36 (3.6 %) |
| mcp2k | 8943 | 117 | 6 | 101 (5.1 %) |

These are **not** the events the owner asked for — the beam flash is at t ≈ 0 and
these are cosmics — but the arithmetic there was verified whole-sample (`dx` vs
`SIGN[apa]·v_drift·op_t` over 1921 matched clusters: median residual 0.0001 cm) and
the class is real.

## Traps hit across both rounds

* **Three different cluster-id spaces.** `img-global` cids, the per-APA Bee layers'
  cids (each numbered **from 1**, so they overlap everything), and `clustering-global`
  cids (doc pr/55). Taking the per-APA ids at face value labels every cluster APA1,
  hides every crosser (control 82 → 0) and the search returns nothing. Join on `q` and
  take the majority.
* **A `(y,z,q)` join key returns zero matches**: the T0 correction also carries a
  transverse shift, and it **flips sign between the TPCs** (APA0 dy = −0.110,
  dz = +0.670; APA1 dy = +0.110, dz = −0.670; IQR 0.000 on all four, 150 events).
  Only `q` survives every stage.
* **`op_cluster_ids` is not a bundle list** (54 % agreement with `n_bundle`); it is an
  img-cid list for Bee's flash navigation.
* **Scope is the whole ballgame.** Round 1's cuts, control and hand-checks were all
  internally sound and still answered the wrong question. The exemplar the owner
  supplied — one coordinate — was what caught it.
