# EM shower clustering & π⁰ hand-scan display

A Bokeh display for **validating and improving** the post-vertex EM shower
clustering and π⁰ reconstruction: mark segments in or out of a shower, build a
π⁰ from scratch and watch the mass come out, and click through to Bee.

Full write-up: [`../docs/pr/114_em-pi0-handscan-display.md`](../docs/pr/114_em-pi0-handscan-display.md).
Sample and coverage audit it grew out of: [`../docs/pr/113_em-shower-pi0-long-muon-coverage-audit.md`](../docs/pr/113_em-shower-pi0-long-muon-coverage-audit.md).

## Quick start

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
./em_display/serve_em_display.sh 5021 --scan-tag myscan1
```

then from a laptop

```bash
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
    -L 5021:localhost:5021 <user>@wcgpu1.phy.bnl.gov
# open http://localhost:5021/em_display_viewer
```

The keepalive flags matter: a bare `ssh -L` gets reaped during the long pauses a
hand scan is made of, and Bokeh's JS does not auto-reconnect (doc pr/88).

## What is on the screen

**Row 1 — three projections** X-Y, Y-Z, X-Z, active volume and cathode in red.

**Row 2 left — the acceptance plot.** Angle to the shower axis vs distance from
the shower start, with the three pass-1 tiers drawn as dashed steps. Every
segment is a dot; a dot **under** a step is inside that tier.

This is deliberately *not* a cone drawn over the projections. A 3-D cone does not
project to a cone, so a wedge on X-Y would be decorative and would invite exactly
the wrong reading. Distance and angle are the two quantities
`NeutrinoShowerClustering.cxx:1310-1312` actually tests, so this panel is the
gate itself with nothing approximated.

**Row 3 — the shower table.** One row per `showers[]`. Click a row to select it.

**Then one of two panels**, chosen by the mode switch.

## EM mode

Select a shower, then mark any segment `IN` / `OUT` / `?` from the candidate
table or by selecting dots in the acceptance plot. The shower's axis is drawn as
an arrow, its members are haloed, and your marks are drawn green/red.

The candidate table's columns:

| column | meaning |
|---|---|
| dist / angle | to the selected shower's **start point** and **axis** |
| pass-1 | which of the three tiers accepts it — or `-` |
| ellip | the ellipsoidal ranking metric (40 cm long, 5 cm across) that decides which shower claims a segment several accept |
| in shower | which shower currently owns it |
| **absorbed by** | **which pass put it there**, from the code's own probe |

`absorbed by` is the column this display exists for. It is not inferred — it is
the `site=` tag the clustering itself printed at the moment it absorbed the
segment. Fifteen distinct values occur over the sample; the census is below.

**The pass-1 tiers are pass 1 only — read the plot in both directions.** The
gate the steps draw (`pass3_cone`) is the single largest absorber — **41 % of
4030 absorptions** over the 94-event sample — but not the only one:

| absorbing pass | share |
|---|---|
| `pass3_cone` (the steps) | 41.3 % |
| `pass4_angle` (`:1964-1967`, different constants) | 20.8 % |
| `in_main_cluster` | 17.7 % |
| `from_vertices` | 10.1 % |
| 11 others | 9.9 % |

So **above every step ≠ rejected** — it may have come in through another pass.
And **below a step ≠ absorbed**: `shower_cone_absorb_guard` is SBND-ON and
declines a confidently-PID'd non-electron straight track longer than 50 cm
(`:1336-1351`, pr/93 Cause D). The `absorbed by` column is the authority; the
plot is the geometry.

The **verdict** includes `vertex-bad (undecidable)`. Use it. If the neutrino
vertex is wrong the in/out question has no answer, and recording that is worth
more than a guess.

## π⁰ mode

1. Select a shower, hit **selected shower → gamma 1**; repeat for gamma 2.
2. Each gamma's start point defaults to the reco's `showers[].start`. To override:
   turn on **tap fills x/y/z**, tap the same point in **two** projections (each
   panel gives two of the three coordinates), then **snap start to nearest fit
   point** — which puts the start on a real fitted point rather than a free
   position in space.
3. Choose the decay vertex: the main vertex, a **back-projection** of the two
   gammas, or a manual point.
4. Read the mass.

Two masses are shown side by side, and that is the point:

- **axis convention** — the angle between the two showers' own axes;
- **vertex convention** — the angle between the two vertex→start chords.

The code itself uses different direction recipes for the mass it stores
(`:3771`) and the angle it stores (`:3830`), and they do not close. Seeing both
is how you tell a genuine π⁰ from a bookkeeping artefact.

### Two things the panel keeps apart on purpose

**`showers[].pio_id` is the pairing. `kine_pio_*` is a BDT feature.** They can
name *different pairs*. On ncpi0 evt21073 the accepted groups are
(60081+31023, 127.2 MeV) and (11008+63100, 111.2 MeV) — while `kine_pio_*`
reports E1 = 680.2 (from 60081) and E2 = 104.7 (from 63100), a third pairing no
reconstruction accepted, mass 207.25. `kine_pio_*` is filled by a separate
highest-energy scan over *all* candidate pairs, accepted or not. They are drawn
in separate blocks and never merged.

**The π⁰ vertex is usually already reconstructed.** When `id_pi0_without_vertex`
accepts a pair it *overwrites* the main vertex with the back-projected decay
point (`NeutrinoShowerClustering.cxx:4338-4340`, tell-tale `fit.dQ == 0`). So the
back-projection here mirrors code that already ran; the job is to judge it, not
to invent it. ("without vertex" means the gammas are **detached**, conn type 3 —
not that the event lacks a neutrino vertex; that case returns early at `:3961`.)

## Membership is 99 % the dump and 100 % the probe

`segments[].shower_id` stores **one** shower per segment, so when two showers
overlap the loser's members vanish from the join and it looks empty rather than
nested. Measured over the 94-event sample: **15 of 1567 showers (1.0 %)**, all of
them EM. Worst cases are ncpi0 evt84229 shower 69134 (43 of 50 joined, a 958 MeV
shower) and ncpi0 evt463565 shower 109073 (**0 of 5** — renders empty).

The display never guesses. It compares the join against the shower's own
`num_segments` and says so in the `joined` column and a banner. With a probe
sidecar loaded it uses the non-lossy membership instead and the banner says
`REPAIRED`. Over the 94-event sample: **15 lossy showers, 15 repaired,
1567/1567 exact.**

## Data it reads

| input | what for |
|---|---|
| `work-<arm>-prod0825/pr_evt<ID>/calib-pr-evt<ID>.json` | everything drawn |
| `em_display/emprep/emprep-evt<ID>.json` | non-lossy membership, `dir15` axis, `absorbed by` |
| `em_display/em114-manifest.tsv` | the event list, the Bee link, per-event stats |

Nothing here is ever written. Exactly one code path writes anything — `on_save` —
and only into `../em_labels/<tag>/labels-evt<ID>.json`.

### Regenerating

```bash
# stage 2: re-run the sample with the shower probes on (fresh arms, ~10 min)
./em_display/run_em114_probe.sh

# parse the probes + rebuild the manifest + resolve Bee links offline
python em_display/prep_em_scan.py --parse-probes

# self-tests
python em_display/selftest_repro.py        # reproduction + membership repair
python em_display/selftest_em_display.py   # drives the viewer's callbacks
```

The probes land in `pr_evt<ID>/stdout.log`, **not** in `wct_pr_evt<ID>.log`:
wire-cell's `-l` flags route only the spdlog logger, and a raw `fprintf(stderr)`
follows the subshell redirect in `run_pr_chain_batch.sh:1642`.

## Bee links

**All 94 events have a live Bee link** (uploaded 2026-08-27, `bee/em114/*.url`).
Two things about them worth knowing:

- **They point at the `em114` sets on purpose.** The same event exists in many
  older sets — `prod0813` among them — and those are *different reconstructions*.
  `bee_index(prefer="em114")` makes the matching epoch win every collision;
  without it, sorted-glob order silently sent 78 of 94 events to a Bee page whose
  clustering disagrees with the panels beside it.
- **A freshly-uploaded set can return 500 on the very first hit** while the
  server finishes unpacking it. Reload once. Verified: 94/94 return 200.

The mechanics, for the next round:  the
set UUID comes from `../bee/<round>/<tag>.url` and the per-event index from the
sibling `.index.txt`, and `/event/<n>/` is used directly as the on-disk directory
name server-side. For the other 16:

```bash
python em_display/prep_em_scan.py --bee-build bee/em114   # ALREADY DONE, see below
./upload-to-bee.sh bee/em114/em114-<arm>.zip     # <- YOURS to run, not mine
# save the printed URL as bee/em114/em114-<arm>.url, then re-run prep
```

`bee/em114/` is **already built**: four zips (mcp1k 10, mcp2k 17, ncpi0 19,
nuecc48 48 events; 51 MB total) with their `.index.txt` and `.prid-map.txt`
sidecars, each starting at `data/0/` as the server requires. Only the upload is
left.

*Gotcha the build exposed:* `make_pr_bee.py` decides "was this event evaluated"
by grepping the per-event **log**, and ncpi0 evt399860's prod0825 log is
truncated (22 KB vs a 207 KB median) so it refuses an event whose dump is fine.
`prep_em_scan.py` therefore points it at the **em114** arms, which carry the line
for 94 of 94 and are equal to prod0825 on every physics field.

The upload is outward-facing and owner-gated (CLAUDE.md §5.6). The build step is
local and touches no network.

## Labels

`../em_labels/<tag>/labels-evt<ID>.json`, one file per event per tag, written
tmp + `os.replace` so a record is never half-written. The `em` and `pio` blocks
upsert independently — scanning EM now and π⁰ later does not drop the first half.

Each record stores **the reconstruction's answer next to yours**: the shower's
membership and where it came from (`probe` or `dump-join`), its axis and which
branch produced it, the `pio_id` groups, the `kine_pio_*` block, and the vertex
with how it was obtained. A later tuning fit joins one file per event and never
has to re-read a dump.

> **A scan tag is a scientific record (CLAUDE.md M13).** Passing `--scan-tag`
> explicitly is consent to write into that set. Without it the viewer uses
> `emscan1` and **refuses to write** if that directory already holds labels.

## Known limits

- **The shower axis is the probe's `dir15`**, i.e. the C++'s own
  `shower_cal_dir_3vector(shower, start, 15 cm)`, for 1565 of 1567 showers. The
  probe prints components at `%.3f`, so it arrives with |v| in 0.9994–1.0004 and
  is re-normalised on read — about 0.06° on any angle, far below anything a scan
  turns on. The other 2 showers fall back to a Python mirror of `init_dir`
  (`PRShower.cxx:1552-1562` / `:1618-1640`); that mirror is exact for conn 1 and
  conn 2/3 and only approximate on the `shower_cal_dir_3vector` fallback branch.
- **`Σ dQ` in the impact line is fitted-point charge, not a calibrated energy.**
  It sizes the change; it does not price it in MeV.
- **No lasso.** Selection is by table row or by dot in the acceptance plot. A
  lasso over a 2-D projection selects everything along the third axis, and that
  ambiguity is not worth the mis-marks.
- The two `dir15`-less showers, and any event scanned without a probe sidecar,
  show an empty `absorbed by` column — the banner says so rather than leaving you
  to infer it.
