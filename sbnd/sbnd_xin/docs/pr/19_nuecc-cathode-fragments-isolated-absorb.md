# doc pr/19 — nueCC 444187: near-cathode shower fragments absorbed into a cosmic by the per-APA `clustering_isolated` small→big merge

**Status: INVESTIGATION — no code is changed.**  Owner report (2026-08-01):
in nueCC evt 444187, `(x,y,z) = (-77.8, 3.1, 246.0)` is on the main cosmic
(final Bee cluster 4) and `(x,y,z) = (-3.7, -16.3, 216.0)` is one of many dots
that belong to the nueCC (whose main body is in TPC1), yet the dots are
clustered with the cosmic.  This doc attributes the merge to a single pass,
quantifies the geometry, and lays out fix options.  Solution = owner decision;
nothing here alters any production path.

Bee set (rescue-ON arm QL output, `work-nuecc48-u17on/ql_evt444187/mabc-all-apa.zip`):
<https://www.phy.bnl.gov/twister/bee/set/90eb6542-5377-4e08-9191-468129c0dd33/event/list/>
(the 24k-pt merged cluster is flashless, so it appears ONLY in the raw
`img-global` layer — doc pr/13 / pr/17 §1.)

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# per-stage Bee trace of the per-APA clustering (fresh root; imaging symlinked
# read-only from the validated arm; input = the 48-evt nueCC set, idx 45)
ROOT=$PWD/work-oc444187-trace; mkdir -p $ROOT
ln -sfn $PWD/work-nuecc48-u17on/evt444187 $ROOT/evt444187
SBND_WORK_ROOT=$ROOT SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt \
    SBND_TRACE_BEE=1 ./run_ql_evt.sh data 45
# stage attribution (16 layers 0-tr00..0-tr15 in the per-APA zip)
unzip -q $ROOT/ql_evt444187/mabc-apa0-face0.zip -d /home/xqian/tmp/oc444187/trace-apa0
./oc_stage_trace.py /home/xqian/tmp/oc444187/trace-apa0 \
    "-77.8,3.1,246.0=cosmicA" "-75.0,-100.0,200.0=cosmicB" \
    "-3.7,-16.3,216.0=dotsNear" "-26.0,-22.0,220.0=dotsFar"
# quantify the isolated-stage absorption (44 clusters -> final cluster 4)
T=/home/xqian/tmp/oc444187/trace-apa0/data/0
./oc_stage_gap.py $T/0-tr13_ClusteringNeutrino-global.json \
    $T/0-tr14_ClusteringIsolated-global.json 4 55
```

State at the trace run: toolkit `c38b4c8e` (`apply-pointcloud`),
`libWireCellClus.so` md5 `83747b9fa765d7a12546ba3dd968a3cc` (the pr/17
validated binary), wcp-porting-img `1ba7a81`.  Trace output kept under the
fresh tag `work-oc444187-trace/` (M13).  The trace knob is `trace_bee`
(SBND_TRACE_BEE=1), default OFF ⇒ production zips byte-identical.

## 1. Symptom

Final clustering (`clustering-global` layer / pctree): cluster 4 ≡ pctree
ident 5 = **24422 points, 515.7 cm extent, all TPC0**, containing BOTH the
long cosmics and ~560 points of nueCC shower fragments at
x ∈ [−30, −0.3] cm, y ∈ [−26, −8], z ∈ [200, 235] (raw frame, i.e. genuinely
just on the TPC0 side of the cathode).  The merged monster gets **no flash**
(gid −1, sentinel t0) → invisible in every post-QL Bee layer and to the PR
chain, while the nueCC main body in TPC1 (final cluster 19, 3546 pts,
210.4 cm, t0 = 1.573 µs, pseudo-flash gid 1000000) is reconstructed without
its TPC0 ~26 cm of shower.  The pr/17 §7.4 "near-miss" (rescue_unmatched did
not fire here) is a downstream symptom of this same merge: the orphan the
rescue saw was the whole 24k-pt cosmic, not the nueCC fragments.

## 2. Raw-imaging anatomy (before any clustering)

Connected components of the apa0 `icluster` blob graph
(`icluster-apa0-active.npz`, b-b edges; x from slice time via
x = −234.09 cm + 1.563e−4 cm/ns · t, anchored on the cosmic at bee
x = −77.79):

- **comp 4183** — 1400 blobs, x[−100, −14], y[−13, +200], z[246, 423]: long
  cosmic through the owner's point (−77.8, 3.1, 246.0).
- **comp 3356** — 988 blobs, x[−79, −69] (near-isochronous), y[−198, −11],
  z[96, 255]: a second, distinct cosmic.
- **nueCC fragments** — ~17 separate tiny components (4–25 blobs each,
  0.7–6.8 cm long), x from −30 up to −0.3 (cathode), around (y, z) ≈
  (−22, 215): comps 4020/4186/4253/4365/4367/4366/4363/4356/4152/4206/4229/…
  Pairwise gaps 3–17 cm; **none touches either cosmic** (nearest big-comp
  approach ≥ tens of cm).
- Across the cathode, the apa1 (TPC1) nueCC charge begins ~5 cm from the
  dots reference point; the final cathode-tip gap between the fragment family
  and the TPC1 nueCC cluster is **1.88 cm** (pr/17 §7.4).

So imaging is fine: the nueCC fragments enter clustering as their own tiny
clusters.

## 3. Stage attribution (per-stage Bee trace, apa0)

`oc_stage_trace.py` on the 16 `0-tr*` layers; refs: cosmicA = comp 4183,
cosmicB = comp 3356, dotsNear = fragment at the cathode, dotsFar = fragment
at x ≈ −26:

```
00_ClusteringPointed          nclus=97  cosmicA=93 cosmicB=30 dotsNear=60 dotsFar=47
06_ClusteringClose            nclus=88  cosmicA=88 cosmicB=88 ...          MERGED: cosmicA+cosmicB
13_ClusteringNeutrino         nclus=69  cosmicA=55 cosmicB=55 dotsNear=64 dotsFar=59
14_ClusteringIsolated         nclus= 9  ALL = 4                            MERGED: everything
```

- Stage 06 `clustering_close` (1.2 cm cut) merges the two cosmics into one
  23186-pt cluster (55 at stage 13).  Cosmic–cosmic; not the reported
  problem, noted for completeness.
- Through stage 13, every nueCC fragment is **still separate** from the
  cosmics.
- **Stage 14 `clustering_isolated` collapses 69 clusters → 9**, absorbing
  44 pre-stage clusters into final cluster 4 — including the entire nueCC
  fragment family.

## 4. Mechanism (source)

`clus/src/clustering_isolated.cxx` (SBND per-APA instance:
`length_cut=15 cm`, `range_cut=150`, `cfg/.../sbnd/clus.jsonnet` line ~267):

1. Clusters split into **small** (wire/tick range max < 150 AND length
   < 15 cm) vs **big** (lines 177–253).  Every nueCC fragment (0.7–6.8 cm)
   is small; cluster 55 (the merged cosmics) is the dominant big cluster.
2. **Small→big absorb** (lines 267–296): each small cluster is merged to its
   *nearest* big cluster when the closest-point distance
   < `small_big_dis_cut = 80 cm` (hard-coded).  **No angle, no direction, no
   drift/cathode awareness — nearest big wins.**
3. Small→small chaining at 5 cm / second pass 50 cm (lines 299–352) pulls
   the rest of the family into the same group.

Measured gaps for the 15 nueCC-region fragments (stage 13 → cluster 55):
**46.6 – 75.9 cm — every one under the 80 cm cut, none under 40 cm.**
(For contrast, the legitimately-absorbed delta rays hug the cosmic at
4.8–10 cm.)  Table: `oc_stage_gap.py` output, §0.

Two structural aggravators:

- **Per-APA blindness.**  The pass runs per anode (before any cross-cathode
  context exists).  The fragments' true parent — the TPC1 nueCC cluster
  1.88 cm across the cathode — is not in the candidate list; the wrong
  parent 72 cm away is the only big cluster on offer.
- **Merge-before-matching.**  The isolated grouping is collapsed into one
  cluster before Q/L matching, so the fragments' flash identity is decided
  by the adopting cosmic.  Here the merged monster matched nothing →
  flashless orphan; the fragments became invisible along with it.  (The
  grouping IS recorded — `assoc_cluster_id/main` per-blob arrays,
  `save_assoc_id`, doc 52 — and the PR chain un-merges it for fitting, but
  the un-merged fragments stay in the cosmic's flashless group; nothing
  re-examines them.)

## 5. Why it hid

- On events whose neutrino is fully in one TPC, the nu main IS the nearest
  big cluster, and the same absorb helpfully gathers its own shower
  fragments — the pass is a net positive there.
- The failure needs a near-cathode vertex (fragments on the far side of the
  cathode from the nu main) *plus* a large cosmic within 80 cm in the same
  TPC — and the damage is then silent: the fragments simply vanish into a
  flashless cosmic, visible only in `img-global`.
- The earlier evt-11 gamma case (`docs/15_overclustering-evt11-gamma.md`)
  was the same absorb with a different driver (a 16 cm EM blob classified
  "small"); it was mitigated by tightening `length_cut` 20 → 15 cm.  That
  knob cannot help here — these fragments are genuinely tiny.  (That doc's
  fix note is also cited inline at the SBND `cm.isolated` call,
  `cfg/.../sbnd/clus.jsonnet` ~line 258.)

## 6. Fix options (owner decision; all would be default-OFF knobs)

**(a) Distance-tier the small→big absorb (per-APA, config knob).**
Make `small_big_dis_cut` configurable (keep 80 cm default); SBND could
tighten (e.g. 40 cm frees every fragment here while keeping the ≤10 cm
delta-ray absorbs).  Cheap, but blunt: any tightening also strands genuine
cosmic debris at 40–80 cm as isolated clusters, and picking the number is a
production-wide retune (needs the mcp1k + nueCC48 sweep).

**(b) Cathode guard on the absorb (per-APA, config knob).**
Skip the small→big merge when the small cluster reaches within X cm of the
cathode plane (raw |x| < X, e.g. 10–20 cm) and the big-cluster gap exceeds
some floor — exactly the fragments that plausibly belong to activity in the
other TPC.  Targeted (only near-cathode fragments change), but by itself it
only *protects*: the fragments stay isolated flashless clusters; the nueCC
still misses them unless something later adopts them.

**(c) All-APA cross-cathode fragment adoption (recovery; pr/14 family).**
After `cathode_connect`/`cathode_bundle_rescue`, adopt small flashless
fragments whose cathode tip sits within a few cm of a beam-window cluster's
cathode tip (here: 1.88 cm to the TPC1 nueCC, well inside every pr/14 pair
cut).  This is the pr/17 §7.4 follow-up, and would put the ~560 points INTO
the neutrino.  Requires (b) (or (a)) first — today the fragments are already
inside the 24k cosmic when the all-APA passes run, so there is nothing
separate to adopt.  Would also need to accept pseudo-flash (gid 1000000)
partners, since the nueCC main here carries the empty-flash-rescue gid.
Direction gate on the merge (fragments forced into-beam) as in
rescue_unmatched.

**(d) PR-side adoption via the assoc arrays (no clustering change).**
The `assoc_cluster_id/main` provenance already marks the fragments as
"associated" members absorbed by the cosmic main.  A PR-stage pass could
re-assign associated fragments near the nu vertex/cathode tip to the nu
cluster before fitting.  Least invasive to clustering, but acts after
matching (fragment charge still missing from the Q/L bundle prediction) and
adds PR complexity.

Recommendation: **(b) + (c)** as one campaign — (b) keeps near-cathode nu
charge out of cosmics, (c) recovers it into the beam cluster; both
default-OFF, validated pr/14-style (mcp1k sweep + nueCC48 + hand-scan of
every firing) before any SBND default flip.  (a) is a fallback if (b)'s
geometry proves too fiddly.  Not proposed: moving `clustering_isolated`
to the all-APA stage (raw x is not comparable across TPCs for out-of-time
activity, and the per-APA absorb is load-bearing for single-TPC events).

## 7. Artifacts

- Trace run (fresh tag): `work-oc444187-trace/ql_evt444187/` — per-APA zips
  carry the 16 `0-tr*` stage layers; all-APA zip as usual.
- Committed probes: `oc_stage_trace.py` (stage attribution),
  `oc_stage_gap.py` (per-cluster gap table for one stage's absorb).
- Scratch (session-local, not preserved): `/home/xqian/tmp/oc444187/`
  (raw-graph component analysis `probe_imaging_cc.py` / `probe_final.py`,
  layer probe `probe_layers.py`, stage-13 table `probe_stage13.py`).
- Related docs: pr/17 §7.4 (the rescue near-miss on this event), pr/13
  (Bee layer frames — why the flashless monster is only in `img-global`),
  doc 52 (assoc arrays / un-merge), `overclustering-evt11-gamma.md`
  (previous isolated-absorb overcluster, length_cut 20→15).
