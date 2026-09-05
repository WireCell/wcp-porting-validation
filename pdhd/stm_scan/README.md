# PDHD stopping-muon hand scan — `retile_wrapped_channel_activity`

The gate doc `../docs/stm-tagger-chain.md` §10 item 1 asks for, now as a
dedicated display instead of a TSV you fill in by hand.

## Why this scan exists

§12 of that doc measured that PDHD's Steiner terminal gate is starved by the
wrapped induction planes: **`ncharge = 3` is exactly zero** over 922 469 sampled
points (PDVD: 0.462), and 78 % of points carry fewer than two plane charges, so
they can never be terminals at any threshold. `retile_wrapped_channel_activity`
raises that ceiling **3.6×** and is the only knob that moves the §8.6 coverage
number. It is still **default OFF**. This scan decides whether the STM verdicts
it moves actually get better — the last step before a flip can be proposed.

## Start it

```bash
cd wcp-porting-img/pdhd/stm_scan
./serve_stm_scan.sh 5017                  # --tag NAME to namespace a second pass
```

From a laptop:

```bash
ssh -L 5017:localhost:5017 user@wcgpu1
# then open  http://localhost:5017/stm_scan_viewer
```

Ports in use elsewhere: img_plot 5013, pd_plot 5014, ql_scan 5015,
wf_scan / pdvd ql_scan 5016. This app owns **5017**.

## What you see

Three projections of one cluster, all at **full detector extent** with the
active boundary drawn as a red dashed box:

| panel | axes |
|---|---|
| side view | Z (beam) vs Y (vertical) |
| top view | Z (beam) vs X (drift) |
| end view | X (drift) vs Y (vertical) |

- **Coloured** — the cluster in question, coloured by its charge.
- **Grey** — *all other charge in the event*. This is the most important thing
  on the screen after the cluster itself: a track that continues into a
  neighbouring cluster is a **fragment**, not a stopper, even though the
  coloured points appear to stop.
  - With **`Dense context near cluster`** on (the default), **every** grey point
    within **40 cm** of the cluster is drawn, and the rest of the event is
    thinned ~1-in-N. The whole `FRAG`-vs-`STM` call rests on that dense set: at
    global decimation alone a 300-point continuation renders as a handful of
    dots, which is indistinguishable from nothing.
  - Both selections are **purely geometric** over all other charge — no cluster
    ids, no tagger output, **not** "clusters the tagger considered", which would
    leak the answer. The self-test reproduces the dense set by brute force from
    coordinates alone and requires an exact match.
  - Outside 40 cm, sparse grey still means decimation, not absence of charge.
- Panels default to the whole detector on purpose. A cluster auto-zoomed to its
  own extent looks contained in every projection, which would invert
  `THRU`/`STM` judgements. `Zoom to cluster` is there when you want it.

## What to answer

From the charge alone, judge the **whole object** — the coloured cluster
*together with any grey charge that continues along the same trajectory*. Does
that object enter the detector and **stop** inside the active volume?

| button | the cluster is… | and the full object… |
|---|---|---|
| `STM` | the whole object | stops inside |
| `THRU` | the whole object | crosses / exits a face (anode and cathode count) |
| `FRAG → STM` | only **part** of the object (under-clustered) | stops inside |
| `FRAG → THRU` | only **part** of the object (under-clustered) | exits — e.g. the cluster is a piece of a TGM |
| `MESSY` | not one track at all — fused tracks, a shower | *ill-posed*: "does it stop" has no answer |
| `UNCLEAR` | — | you genuinely cannot tell |

**Why `FRAG` is two buttons and not one.** "The cluster ends but grey continues"
covers two opposite physics truths: a fragment of a **TGM** (an STM tag is
wrong) and a fragment of a **stopping muon** (an STM tag is right, just on a
wrong-sized object). Collapsing them into one category would make it
uninterpretable. So a `FRAG` button records the *same binary verdict* as its
plain counterpart, plus `partial: true`. Under-clustering therefore costs this
scan **no statistical power** — every fragment still scores — and the
under-clustering rate falls out as its own reportable number.

Keep the three escape hatches distinct, because they answer different questions:

- `FRAG` is about the **cluster** (it is a piece of something bigger).
- `MESSY` is about the **object** (the question does not apply to it).
- `UNCLEAR` is about **your confidence** (it might be any of the above).

Do not spend a fragment on `UNCLEAR` — that is the one substitution that loses
information the scan cannot recover.

Because `clustering-global` is byte-identical between the two arms, the fragment
judgement cannot be biased toward either one.

There is a free-text `notes` box; type the note *before* clicking the label,
since the click saves and advances.

Clicking a label **saves immediately** and jumps to the next unlabelled item.
Nothing is lost to a reload or a restart. `next unlabelled >>` resumes wherever
you left off.

## Why the blind is structural, not a request

The earlier version of this scan asked you not to switch on the Steiner and STM
layers. Now the app cannot show them: it opens **only** the `clustering-global`
and `channel-deadarea` members of `mabc-pr.zip`, and never names the others.
`selftest_stm_scan.py` asserts that — the strings `stm_fit`, `stm_tagged`,
`steiner_graph`, `steiner_terminals`, `stmw` do not occur in the viewer's code.

The two members it does read are **byte-identical between the knob-off and
knob-on arms** (zip-member SHA-256, also asserted by the self-test), so the
pixels cannot encode which arm produced them. The viewer never reads the answer
key. The `stratum` flag is deliberately absent from the UI: you can see the
point count, and calling a fragment `UNCLEAR` has to be your judgement, not a
nudge from a badge that says "small".

## The sample, and why it is stratified

224 clusters — every one whose STM verdict differs between the arms, 10.0 % of
2246 verdicts, 130 gaining the tag and 94 losing it. The `(event, cluster)` key
sets of the two arms are identical, so the knob changes no cluster partition,
only the Steiner stage.

The churn is **not** size-symmetric, and that shapes everything:

| stratum | n | knob gains tag | knob loses tag |
|---|---|---|---|
| A `npts ≥ 200` | 174 | 83 | 91 |
| **B `npts < 200`** | **50** | **47** | **3** |

**36 % of every STM tag this knob adds sits on a cluster with under 200 points.**
Dropping those — the obvious way to make the scan quicker — would have hidden
the effect most likely to argue *against* the knob. Stratum A asks whether real
tracks improve; stratum B asks whether the denser graph is manufacturing tags on
fragments. Tranche 1 (the first 63 items, shuffled with a fixed seed) is 40 from
A and all 23 available from B.

## Scoring, fixed before any label exists

```bash
python3 score_stm_scan.py            # --tag NAME to score another pass
```

That script reads the answer key; the viewer never does. It scores each arm
against your labels per stratum and per direction of change, reports the
unscored (`MESSY` + `UNCLEAR`) rate per stratum, and breaks down the composition
of the tags the knob **adds** — which is where a fragment-tagging knob shows up.
It refuses to run on a label it does not recognise rather than silently folding
it into `THRU`.

**Acceptance bar**, stated before the scan so it cannot be fitted afterwards —
and yours to overrule: flip only if the knob is net-positive in **stratum A**
*and* its **stratum-B gains** are not predominantly `THRU`/`UNCLEAR`. A knob
that fixes real tracks while inventing fragment tags is a different decision
from one that only does the first.

**Appended 2026-09-05** — the paragraph above is the original and is left
exactly as it was written. At the moment of this addition **exactly one label
existed**: item 1, event 21 cluster 125, `UNCLEAR`. The `FRAG` and `MESSY`
categories were added after the scanner reported that some clusters are
under-clustered pieces of a TGM. Two things follow, both stated here *before*
the remaining labels exist:

1. **The bar itself is unchanged.** `FRAG` rows carry the full object's binary
   verdict, so they count in "net-positive in stratum A" and in the stratum-B
   clause exactly as a plain `STM`/`THRU` would.
2. **If fragments dominate the knob's gains, the finding is about
   under-clustering, not about this knob** — and that is the more important
   result even if the binary comes out favourable. An offline proxy (other
   charge within 30 cm of a PCA extreme, measured before any labelling and kept
   off both the sheet and the display) puts an upper bound of ~⅓ of items, and
   nearly the same rate on gains (34 %) as on loses (32 %) — so fragments are
   not expected to bias one arm, but they are expected to be common enough to
   matter. Report it either way; the flip is the owner's call.

## Files

| file | |
|---|---|
| `stm_scan_viewer.py` | the app; fork by duplication of `../ql_scan/ql_scan_viewer.py`, which is untouched |
| `serve_stm_scan.sh` | fork of `../ql_scan/serve_ql_scan.sh` |
| `selftest_stm_scan.py` | 52 headless checks: the blind, the render path, the dense context vs a brute-force geometric reimplementation, full-volume default, every label round-trip, the scorer's rejection of an unknown label |
| `score_stm_scan.py` | scores labels against both arms |
| `../docs/scan/pdhd_retile_scan_sheet.tsv` | the item list (no verdicts) |
| `../docs/scan/pdhd_retile_scan_key.tsv` | the answer key — closed until labelling is done |
| `../work/stm_scan_labels/<tag>/labels.json` | your labels; a sibling of the per-event dirs so re-running an arm cannot delete them |

(You asked for "ql_display" — on PDHD the Bokeh scan app is `pdhd/ql_scan`;
`pdvd/ql_display` is a different, non-Bokeh analysis directory. This is forked
from `pdhd/ql_scan`.)
