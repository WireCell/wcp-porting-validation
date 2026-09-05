# Hand scan: `retile_wrapped_channel_activity` on PDHD

> **Superseded by the display.** Labelling is now done in the Bokeh app
> `pdhd/stm_scan/` (port 5017) rather than by editing the TSV by hand — the app
> renders the three projections and enforces the blind structurally. See
> `pdhd/stm_scan/README.md`. The two TSVs here remain the item list and the
> answer key that the app and its scorer read.

Gate for doc `stm-tagger-chain.md` §10 item 1. §12 measured that this knob
raises the Steiner terminal ceiling 3.6× and is the only knob that moves the
§8.6 coverage number; it is still **default OFF**. This scan decides whether the
STM verdicts it moves get better, and it is the last thing standing between the
measurement and a flip proposal.

## What you need to look at

Two arms, run 029107, 30 events, same pctree, same pinned binary
(`/home/xqian/tmp/pdhdstm_libpin`):

| arm | work dir | knob |
|---|---|---|
| off (current production) | `pdhd/work/029107_<evt>_stm0/mabc-pr.zip` | `retile_wrapped_channel_activity=false` |
| on | `pdhd/work/029107_<evt>_stmw/mabc-pr.zip` | `true` |

**For labelling, open either one** — the `clustering` and `channel-deadarea`
layers are byte-identical between them (verified by zip-member SHA-256), so raw
charge is blind. Keep `steiner_graph`, `steiner_terminals`, `stm_fit` and
`stm_tagged` switched **off** while labelling: those differ between arms and
would tell you which arm you are seeing.

## What to record

Fill the `label` column of `pdhd_retile_scan_sheet.tsv`, one row per cluster:

| label | meaning |
|---|---|
| `STM` | the track enters and **stops** inside the active volume |
| `THRU` | through-going, or exits any face (including anode / cathode) |
| `UNCLEAR` | fragment, too sparse, overlapping, genuinely ambiguous |

`UNCLEAR` is a real answer, not a failure — stratum B is expected to have many.

## Why it is stratified

The 224 changed verdicts split by cluster size, and the split is **not**
symmetric:

- **stratum A**, `npts >= 200` (174 clusters): does the knob improve verdicts on
  real tracks?
- **stratum B**, `npts < 200` (50 clusters): here the change is 47 gain / 3 lose
  — **36 % of every STM tag the knob adds is on a small cluster**. Are those
  real stoppers the starved skeleton was missing, or is the denser graph
  manufacturing tags on fragments? This is the half most likely to argue
  *against* the knob, which is exactly why it must be labelled.

Tranche 1 is rows 1–63 (A: 20 gain + 20 lose; B: 20 gain + all 3 lose), balanced
and shuffled with a fixed seed. Rows 64–224 are the rest, if you want more.

## Scoring

`pdhd_retile_scan_key.tsv` holds each cluster's verdict under both arms.
**Leave it closed until the labels are in** — it names the direction of change
per row and would bias the labelling. Scoring is then per stratum and per
direction: for each label, which arm agreed.

The acceptance bar is yours to set. What I would propose, stated before seeing
any label: the knob flips only if it is net-positive in stratum A **and** its
stratum-B gains are not predominantly `THRU`/`UNCLEAR`. A knob that fixes real
tracks while inventing fragment tags is a different decision from one that does
only the first.
