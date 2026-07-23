# 28 — TGM untagged-case log (bundles that look through-going but `tgm=0`)

Running log of matched bundles that a hand-scan reads as through-going muons but
`TaggerCheckTGM` leaves untagged, so we can accumulate statistics on the failure
modes **before** deciding whether/how to act. No code is changed by this doc.

Each case records the *mechanism* (which of `check_tgm`'s branches declined and
why), verified against the tagger's own extreme points — not a Bee-zip proxy,
because several of these turn on sub-millimetre margins.

## Repro (per-case diagnosis)

The verdict lines and the (env-guarded) per-extreme-point / CASE-B traces:

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild   # only if src changed
cd sbnd_xin
SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10 ./run_nusel_evt.sh data <idx>
L=work-mcp10/nusel_evt<ID>/wct_nusel_evt<ID>.log
grep "cluster <cid> → TGM" $L
```

Population-level scan for the Case-2 (readout-truncation) signature, which
needs no instrumentation — it reads only `nusel-table.tsv` and the per-event
`mabc-pr.zip`:

```bash
cd sbnd_xin && python3 tgm_readout_cut.py work-mcp10
```

The `check_tgm entry:` (all extreme groups + `inside_fv`), `check_tgm CASE-B:`
and `check_tgm P2DET:` (which CASE-B angle gates opened, and what
`check_signal_processing` / `check_dead_volume` each returned) traces are
**not** in the committed source; they were a temporary env-guarded
(`WCT_TGM_DEBUG`) instrumentation (mirroring the existing CASE-A `check_tgm dbg:`
block) used to produce the numbers below and then reverted. Re-add the same
block to reproduce. The permanent CASE-A `check_tgm dbg:` line already prints
under `WCT_TGM_DEBUG` and is what distinguishes "entered CASE A" from "never did".

Recap of `check_tgm` structure (see `clus/docs/tgm/`): a cluster is TGM when a
pair of extreme-point groups **both** exit the fiducial volume (CASE A), or when
an apparently-inside end is explained by a prolonged-signal artefact or a dead
region (CASE B). FV = `BoxFiducial:sbnd_pr_fv` inset by
`[-2,-2,-2.5,-2.5,-3,-3] cm` → floor `y = -199.312 + 2.5 = -196.812 cm`.

---

## Case 1 — evt284657, flash grp 6, main cluster 7 (`tgm=0`, hand-scan: TGM)

- Run/subrun/event 18255 / 1 / 284657, `flash_time = -715.547 µs` (well outside
  the beam window, so beam protection is irrelevant). `len_main = 401.2 cm`,
  `npts_main = 4416`, `n_frag = 1` — a single 4-m cluster.
- Bundle: `nusel-table.tsv` row `main_id 7 … flash_grp 6 … tgm 0 stm 0 fc 0
  not-tagged`.

### Mechanism — a knife-edge endpoint, NOT the dead channels

Cluster 7 is a **tilted vertical crosser**: y −196.8 → **+199.1**, x −114 → −47,
z 234 → 258. Its five extreme groups (tagger coordinates, T0-corrected scope):

| group | (x, y, z) cm | `inside_fv` |
|---|---|---|
| 0 (top) | (−112.6, **199.076**, 256.9) | **false → exits** |
| 1 (bottom) | (−47.5, **−196.785**, 249.5) | true (**0.027 cm** inside floor −196.812) |
| 2 | (−78.2, 33.0, 234.4) | true |
| 3 | (−47.5, −191.0, 247.3) | true |
| 4 | (−113.8, 193.2, 254.5) | true |

**Only the top group exits.** The bottom reconstructs ~2.5 cm short of the
physical floor (−199.312) and lands **0.027 cm inside** the 2.5-cm-inset FV
floor. So no extreme-point pair has *both* ends outside → **CASE A is never
entered** (no CASE-A `dbg` line for cluster 7, unlike the four clusters that do
tag TGM in this event). Every pair falls to CASE B, where the bottom's
`FiducialUtils::check_dead_volume` walks downward from −196.785 into **live**
detector and correctly returns "genuine inside" → the end stays inside → no tag.
CASE-B trace, pair (0,1): `p2 (-47.5,-196.785,249.5) in0 true inP true` — the
inside verdict survives the dead-volume check. Behavior is **self-consistent and
correct given the reconstructed endpoint.**

### The dead "middle bar" is a red herring

The dead channels the hand-scan noticed are real but **mid-track, not at an
endpoint**: the event's dead-area polygons overlapping the cluster sit at
**z ≈ 249.6–251.4 cm** (e.g. apa0 poly4 y[−68,−67]; further segments at
y = 88–100, 132, 154, 159), a vertical dead strip in z. They:

- are in the **middle** of the track's path, never reaching the TGM endpoint
  test;
- do **not** break the 3-D cluster — the y point-distribution is continuous
  top-to-bottom (the other two planes cover the dead wires); `n_frag = 1`;
- are absent at the bottom endpoint region (y ≈ −197, z ≈ 245), which is why
  `check_dead_volume` finds live detector there.

So the non-tag is entirely the **bottom-endpoint-vs-inset-margin knife-edge**,
independent of the dead region.

### Open — needs larger statistics before acting

The asymmetry is the real signal: the top exits cleanly (199.076, 0.2 cm from
the edge) while the bottom stops ~2.5 cm short at a **live** wall. Whether this
is imaging clipping a real exit or the muon genuinely ending there is not yet
established (a raw-charge check below y = −196.8 would settle it). Candidate
responses, **all** unconditional behavior changes ⇒ default-OFF knob + owner
sign-off + byte-identical gate (escalation rule 1) — deferred until we have a
population, not one event:

1. **Boundary-aware CASE B** — treat an "inside" endpoint within ~margin of the
   *physical* wall along its exit direction as exiting even when the region is
   live. Targets the actual mechanism.
2. **Tighten the y-margin** (< ~2.48 cm) — rejected as fragile: a 0.027 cm
   knife-edge would flip other events unpredictably.
3. **Accept as-is** — declining a track that reconstructs short of a live wall
   is defensible.

Recommendation pending statistics: option 1, if the population shows a clean
"stops a few cm short of a real wall" class that hand-scans call TGM.

---

## Case 2 — evt285185, flash grp 11, main cluster 18 (`tgm=0`, hand-scan: TGM)

- Run/subrun/event 18255 / 1 / 285185, `flash_time = -224.892 µs` (out of beam).
  `len_main = 394.2 cm`, `npts_main = 6515`, `n_frag = 2`.
- Bundle: `main_id 18 … flash_grp 11 … tgm 0 stm 0 fc 0 not-tagged`.

**This is a different mechanism from Case 1, and a quantitatively explained
one: the anode-side end of the track was never digitized because the cosmic is
early relative to the readout-window open time.**

### The cluster is one clean track; the merge fragment is inert

6501 of 6515 points (99.8%) lie within 10 cm of the grp0→grp1 axis (median
perpendicular distance 1.4 cm) — a straight 394.2 cm track. The "+1 merge
fragment (squares)" is an **8-point orphan** at (−76.1, 130.9, 490.4), 395 cm
off-axis. The fragment only adds a third extreme group, which is *inside* the
FV and so provides no second exit — it neither causes nor could cure the
non-tag.

Companion 13 is negligible either way: the bundle table records the whole
companion set as one extra point (`npts_bundle − npts_main = 6516 − 6515 = 1`),
and no cluster 13 appears in `0-clustering-global.json` at all (that dump holds
12 clusters, ids 1/5/8/9/12/15/16/17/18/19/20/21) — the usual bundle-table vs
Bee id-space mismatch, not a physical object. A ≤1-point companion cannot move
an extreme point. **The tagger only ever evaluates main cluster 18's own
extreme points**, and those already span the full 394 cm track, so neither the
fragment nor the companion is implicated.

### Mechanism — one end exits, the other is cut off by the readout window

Extreme groups (tagger coordinates, T0-corrected scope):

| group | (x, y, z) cm | `inside_fv` | |
|---|---|---|---|
| 0 | (154.552, 158.939, **0.230**) | **false → exits** | upstream z wall (z floor 0.85) |
| 1 | (**198.628**, −86.579, 305.480) | true | 0.42 cm inside inset x wall 199.05 |
| 2 | (−76.089, 130.920, 490.420) | true | the 8-point merge fragment |

Only grp 0 exits ⇒ **CASE A never entered** (no CASE-A `dbg` line for cluster
18). CASE B runs on pair (0,1) and declines — traced explicitly:

```
P2DET: cluster 18 pair (0,1) dir (0.092,-0.616,0.783) angU 57.0 angV 84.3 angW 83.3
       gate_sp false perp_main 89.0 gate_dv true
P2DET: cluster 18 check_dead_volume -> true
CASE-B: cluster 18 pair (0,1) ... p2 (198.628,-86.579,305.480) in0 true inP true
```

i.e. the prolonged-signal check never ran (all three wire angles above the
10/10/5 thresholds), and `check_dead_volume` ran and returned **true** — live
detector ahead. That verdict is *correct*: the region beyond the endpoint is
live. It simply **was not read out**.

### Proof: the endpoint sits on the readout-truncation plane

Near-anode charge has ~zero drift time, so it arrives at ≈ `t0`. For a cosmic
early enough that `t0` precedes the readout-window open time, the anode-side
end of the track is never digitized, and the reconstruction stops on a plane

> `|x_cut| = A + v·t0`

Fitting that line to the clipped endpoints of **11 main clusters across all 10
events** (t0 from −1176 to −225 µs, |x| from 50 to 199 cm) — `tgm_readout_cut.py`:

```bash
cd sbnd_xin && python3 tgm_readout_cut.py work-mcp10
```

> **A = 233.941 cm, v = 0.15634 cm/µs**, residual **rms 0.28 cm**, max 0.78 cm

The fitted slope reproduces the configured drift speed
(`run_nusel_evt.sh:43 DRIFTSPEED=1.563` mm/µs = 0.1563 cm/µs) **to 0.02%** —
the plane recedes at exactly the drift velocity, which is the signature of
readout truncation and not of anything geometric. The implied window-open time
is `t_start = (201.05 − A)/v ≈ −210 µs`.

For cluster 18 (`t0 = −224.892`): predicted cut 198.8 cm, observed max x
**198.94 cm**. Corroboration: across all 74 main clusters, **no cluster's
extreme x ever exceeds its own predicted cut on either side** (0 envelope
violations) — the plane is a hard upper bound, as a truncation must be — and
unclipped clusters do reach the anode (evt285185 clus 21 at x = +201.20,
clus 20 at x = −201.29). So the 2.4 cm deficit is specific to this cluster's
t0, not a detector edge.

The cleanest way to see it: the intercept `A` **is the readout boundary in the
raw (drift-time) frame**. Since `x_corr = x_raw − v·t0`, a boundary fixed at
`x_raw = A` maps to `x_corr = A + v·t0` — exactly the fitted line. And
`A ≈ 233.9 ≈ x_anode + v·|t_start| = 201.05 + 0.1563 × 210`. So every clipped
end sits at the **same, t0-independent** raw position; it is the T0 correction
that smears that one fixed boundary across the FV in corrected coordinates,
which is why it looks like a different "stopping point" for each cluster. That
the slope equals the configured drift speed to 0.02%, with zero envelope
violations in 74 clusters, is the proof — no geometric or physical
track-stopping effect is keyed to `t0` at the drift velocity.

Corroborating (not load-bearing): dQ/dx along the track is **flat (~150k/cm) to
the last bin — no Bragg peak**, so the muon did not stop; and the local
direction at the endpoint is only 5.3° off the anode plane, so the clip removes
the final ~26 cm of path.

### Same mechanism, second instance in the same event

Cluster **17** (`t0 = −342.783`, `len 370.9`, also `not-tagged`) has extreme
group 0 at (**180.514**, 37.003, 142.430) — predicted cut **180.50** — reading
`inside_fv true`, while its other end exits the y floor at y = −199.123. One
exit + one readout-clipped end, exactly as cluster 18.

### Population (this is the statistics Case 1 was waiting for)

Of **74 main clusters over 10 events, 11 have an endpoint on the readout cut**:

| label | n | of which long (>150 cm) |
|---|---|---|
| `not-tagged` | 7 | 7 |
| `TGM` | 3 | (tagged via other end pairs) |
| `STM` | 1 | |

The 7 not-tagged clipped clusters: evt284657/28 (154 cm), evt285185/17
(371 cm), evt285185/18 (394 cm), evt285999/20 (163 cm), evt286021/16 (403 cm),
evt286241/12 (320 cm), evt286241/14 (162 cm). **Every one is a long track**,
the population where a missed TGM matters most.

Note Case 1 (evt284657 cluster 7) is **not** in this list — its y-floor
knife-edge is a genuinely separate failure mode. The two cases should not be
fixed by the same knob.

### Open — candidate response

The tagger has **no concept of the readout-window boundary**. In the
T0-corrected frame the FV box is fixed in absolute x, but the *effective*
detector boundary on the anode side moves with `t0`; for `t0 ≲ −210 µs` it lies
strictly inside the FV. An endpoint sitting on that plane cannot be evidence
that the track stopped — charge beyond it could not exist in the data.

This is not a prototype divergence: the prototype tests `p.x - offset_x`
against a fixed boundary (`ToyFiducial.cxx inside_fiducial_volume`), i.e. the
same corrected-x-vs-fixed-FV convention. uBooNE's much wider readout relative
to its drift simply made the effect rare.

Candidate rule, sharper than Case 1's option 1 because it is derived from `t0`
rather than tuned: **treat an endpoint within ~1 cm of `x_cut(t0) = A + v·t0`,
with the track heading outward, as an exiting end.** `A` should come from the
readout-window open time and drift speed in config, not the fitted constant.

Still **unconditional behavior change ⇒ default-OFF knob + owner sign-off +
byte-identical gate** (escalation rule 1). Before implementing, worth
resolving:

- whether `A` is derivable from existing config (window open time × drift
  speed) or would have to be introduced as a new parameter;
- whether the 3 currently-`TGM` and 1 `STM` clipped clusters would change
  verdict (the STM one especially — an STM whose "stopping" end is really a
  readout clip is a *wrong* stopping-muon tag, arguably a worse error than the
  missed TGM);
- the same question for the FC tagger, which shares this FV and would call a
  readout-clipped end "contained".
