# doc pr/19 — nueCC 444187: near-cathode shower fragments absorbed into a cosmic by the per-APA `clustering_isolated` small→big merge

**Status: CLOSED — FIX CAMPAIGN (b)+(c) SHIPPED default OFF; validation
DONE (§8): nueCC48 clean, 444187 recovered (+786 pts); mcp1k has 7/1000
knob-caused beam-verdict changes (§8.2).  OWNER DECISION (2026-08-02,
after the §8.3 Bee scan): keep the current default — the knobs stay OFF
for SBND production.  Production output remains byte-identical to pr/17;
the fix stays available per event via `SBND_ISO_CATHODE_GUARD=1
SBND_ADOPT_NU_FRAG=1` (or the `iso_cathode_guard`/`adopt_nu_fragments`
TLAs).**
Owner report (2026-08-01): in nueCC evt 444187, `(x,y,z) = (-77.8, 3.1,
246.0)` is on the main cosmic (final Bee cluster 4) and `(x,y,z) = (-3.7,
-16.3, 216.0)` is one of many dots that belong to the nueCC (whose main body
is in TPC1), yet the dots are clustered with the cosmic.  §§1-6 are the
original investigation (attribution to a single pass, geometry, options);
the owner then approved the recommended (b)+(c) pair, implemented in §7 and
validated in §8.

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

## 7. Implementation ((b)+(c), owner-approved; both default OFF in C++/common)

Toolkit changes (branch `apply-pointcloud`):

**(b) `clus/src/clustering_isolated.cxx` — cathode guard.**  New knobs
`cathode_guard_xcut` (default **0 = OFF**) and `cathode_x` (default 0).  When
`cathode_guard_xcut > 0`, a small cluster that (i) has a candidate big
absorber within 80 cm, (ii) reaches within `cathode_guard_xcut` of the
cathode plane, and (iii) is farther from that big cluster than from the
cathode, is **declined**: it is not absorbed, and it may not be chained into
an absorbed group by the 5 cm small-small pass (a guarded small skips any
pair there).  It remains a "remaining" small and may still merge with other
remaining smalls (the 50 cm pass) — in 444187 this consolidates the fragment
family into ONE isolated per-APA cluster.  Delta rays are unaffected: they
hug their track, so big-gap < cathode-distance fails (iii).  Each declined
absorb prints a raw-stdout census marker
`[iso-cathode-guard] declined absorb: small <len> cm / <npts> pts, cathode
<d> cm < big <D> cm` (batch-log greppable, pr/15 marker style).

**(c) `clus/src/clustering_cathode_bundle_rescue.cxx` — pass 3
(`adopt_nu_fragments`, default **false = OFF**).**  After the pr/14 bundle
pass and the pr/17 unmatched pass: adopt a small flashless fragment
(`adopt_min_npts` 5, length < `adopt_frag_max_length` 60 cm, cathode reach
< `adopt_xcut` 30 cm, evaluated in RAW scope — its x_t0cor is
sentinel-garbage) into a beam-window cluster (length ≥
`adopt_beam_min_length` 10 cm, pseudo-flash gids accepted) when the raw
closest approach under the beam-T0 hypothesis is < `adopt_dis` (C++ 10 cm;
SBND config passes 13 cm, covering the observed 444187 family hop spacing of
up to 12.1 cm).  One adoption per round, full re-enumeration (chaining), the
pass-2 merge/re-stamp/scope-repair machinery, ident-ordered determinism.
The length ceiling is the protection against swallowing a long flashless
cosmic on mere proximity — long orphans go through pass 2's strict
crossing-pair geometry instead.  Logs `fragment adopt round ...` per
adoption; summary only when count > 0 (house style — silence ≠ knob off).

**Config plumbing** (all key-suppressed; knobs-off compiled JSON proven
byte-identical to pre-knob HEAD):
- `cfg/pgrapher/common/clus.jsonnet`: `isolated(... cathode_guard_xcut,
  cathode_x)`, `cathode_bundle_rescue(... adopt_nu_fragments, adopt_dis,
  adopt_xcut, adopt_frag_max_length, adopt_min_npts, adopt_beam_min_length)`.
- `cfg/pgrapher/experiment/sbnd/clus.jsonnet`: `clus_per_face`/`per_apa` arg
  `iso_cathode_guard` (→ 30 cm when on); `clus_all_apa`/`all_apa` arg
  `adopt_nu_fragments` (→ `adopt_dis` 13 cm when on).
- `cfg/pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet`: TLAs
  `iso_cathode_guard` / `adopt_nu_fragments` (both default false pending
  validation).
- Runner `run_ql_evt.sh`: tri-state env `SBND_ISO_CATHODE_GUARD` /
  `SBND_ADOPT_NU_FRAG` (doc-68 convention: unset = inherit cfg).

## 8. Validation

Binaries (`local/lib/libWireCellClus.so`): pre-campaign `83747b9f…` (the
pr/17 validated binary); first campaign build `2149cc43…` (no dis_floor);
final `525a7c21…` (with `cathode_guard_dis_floor`).  wcdoctest-clus 565/565
on both builds.

### 8.1 Knobs-off byte-identity

- Compiled-config proof: `wcsonnet wct-clus-matching-perevt.jsonnet` (TLA
  defaults) at the campaign tree is `cmp`-identical to the same compile at
  pre-campaign HEAD (`83823f45`); with the TLAs on, the compiled JSON gains
  exactly `cathode_guard_xcut`×2 + `cathode_guard_dis_floor`×2 (per-APA
  isolated) and `adopt_nu_fragments` + `adopt_dis` (all-APA rescue).
- 444187 knob-off rerun at each campaign binary vs the validated pr/17 arm
  (`work-nuecc48-u17on`): mabc-all-apa.zip, both per-APA zips and the pctree
  all member-hash SAME (`abtest/hash_archive.py`; arms `work-oc444187-off2`
  binary 2149cc43, `work-oc444187-off3` binary 525a7c21).  Input caveat
  discovered on the way: the nueCC48 arms stage from
  `input_files_reco1/extracted-2025fall-48evt-fsprod` — the non-fsprod
  sibling has different opflash content and does NOT reproduce them.
- uBooNE chain (clustering_isolated is shared): qlport sweep `oc19_ub` vs
  base `isog2_ub`: **Bee ZIPS 35/35 content-identical**.  The
  tagger-compare half reported 32/35 log diffs, but a same-binary repeat
  (`oc19_ub2` vs `oc19_ub`, identical binary and config) reproduces the SAME
  32/35 pattern — the diffs are the documented T_tagger candidate-vector
  run-to-run flicker (value multisets identical, order permuted), not a
  behavior change.

### 8.2 ON-arm results (nueCC48 + mcp1k)

**444187 smoke (both knobs on, tags `work-oc444187-on`/`-on2`, binary
2149cc43 = pre-floor).**  The guard declines all the family absorbs (21
markers at this binary; 20 with the final floor — one *near* absorb,
big-gap 10.96 cm, correctly reverts to the legacy merge), the
50 cm remaining-small pass consolidates the fragments into one isolated
per-APA cluster, QL's own empty-flash rescue then matches it to the beam
pseudo-flash and the flash-gated all-APA merges join it to the nu.  Pass 3
(`adopt_nu_fragments`) fires **zero** times — the existing machinery
re-collects the freed fragments on its own; (c) is a pure backstop.
(b)-only (`work-oc444187-bonly`) is archive-identical to both-on.
Determinism: two independent both-on runs member-hash identical.

**nueCC48 v1 (guard without dis_floor, tag `work-nuecc48-oc19on`, binary
2149cc43).**  vs the validated pr/17 arm `work-nuecc48-u17on`: **0
nu-candidate / TGM / STM / LM / FC verdict changes**, 0 beam-row label
changes; 42/48 label tables differ only by extra `not-tagged` rows (freed
isolated smalls that no longer ride a cosmic) and nu-npts drift −236..+137;
444187 recovers +808 pts.  Guard fired on 43/48 events — footprint judged
too broad (it was also declining *near* absorbs, < 40 cm, where the legacy
merge is usually right), motivating the refinement:

**`cathode_guard_dis_floor` (C++ default 0 = no floor; SBND 40 cm).**  Only
declines absorbs whose big-gap exceeds the floor — nearby debris (delta
rays, vertex fragments) keeps the legacy merge; only *distant* (> 40 cm)
absorbs of cathode-hugging smalls are declined.  The 444187 family
(gaps 46.6–75.9 cm) is still fully freed.

**nueCC48 v2 (final operating point: guard 30 cm + floor 40 cm + adopt, tag
`work-nuecc48-oc19on2`, binary 525a7c21).**  vs `work-nuecc48-u17on`:

- **0 verdict-class label changes, 0 beam-row label changes** (48/48).
- 16/48 tables identical; 32 differ only by extra `not-tagged` rows
  (15 events, 1–2 rows each) and/or nu-npts drift.
- nu-candidate npts drift on 16 events: **−76..+30** (v1 was −236..+137)
  plus **444187: 3546 → 4332 (+786)** — the recovered fragment family
  (v1 gave +808; the floor returns one near fragment, big-gap 10.96 cm,
  to the legacy cosmic absorb).
- Guard fired on 32/48 events (v1: 43/48); 444187 has 20 declined absorbs.
- Pass 3 firings: **0** across all 48 events (backstop confirmed inert).
- Census script: `census_v2.py` logic — whitespace-split tsv parsing
  (space-separated tables; the first-pass tab-split census silently skipped
  every row and reported a false "48 identical").

**mcp1k (1000-event MCP2025C data, tag `work-mcp1kall-oc19on1k`, binary
525a7c21, both knobs on via `SBND_ISO_CATHODE_GUARD=1 SBND_ADOPT_NU_FRAG=1
TAG=oc19on1k ./run_full1k_nusel.sh 1000 8`; 1000/1000 rc=0, 1254 s)** vs the
pr/17 validated baseline `work-mcp1kall-u17on1kb`:

- Guard fired **1288 times on 564/1000 events**; pass 3 adoption **0/1000**
  (backstop confirmed inert at production scale too).
- Label tables: 437/1000 identical; 563 differ (matching the 564 firing
  events).  Global label counts 522→526 nu-candidate, 162→164 TGM,
  141→141 STM, 12→12 LM, 88→82 no-bundle, 10504→10680 not-tagged.
- **Verdict-class changes: 7 events** (census must lower-case the labels —
  a first pass with case-sensitive matching missed the two TGMs):
  - **+5 new nu-candidates** (56519, 62495, 314925, 348889, 407258): freed
    fragment groups get matched/adopted onto beam-window flashes that
    previously had nothing (62495's beam flash 13 was a `no-bundle` row;
    now a 43-pt contained candidate).  More scan candidates, no losses.
  - **315959 `no-bundle` → 30-pt TGM**: the in-beam flash (t = 0.877 µs)
    adopts a freed 2.3 cm debris pair which the taggers then call TGM.
    Cosmetic-scale object, but it is a beam-flash label.
  - **395060 nu-candidate → TGM**: the in-beam bundle changes 3207 → 3163
    pts / 179.9 → 177.0 cm and the TGM tagger now fires.  Borderline
    geometry moved by freed end fragments — the one potential efficiency
    LOSS in the sample; needs a hand scan.
- nu-candidate npts drift on 104 events, range **−4286..+252** with two
  large rearrangements beyond 395060:
  - **386948**: beam bundle 6063 → 1777 pts (174.6 → 86.4 cm), the freed
    material re-matches to two cosmic flashes (+476 µs cluster 1233 → 4371
    pts; +1131 µs cluster 902 → 2135 pts).  Label survives
    (nu-candidate/contained).  Either a genuine de-overclustering (the
    444187 failure mode in reverse — cosmic debris unstuck from the nu) or
    an over-eager guard; hand scan decides.
  - **65025**: beam candidate 506 → 13 pts (`nosteiner`), with a large Q/L
    rearrangement upstream (the −655 µs cosmic grows 3117 → 14166 pts by
    absorbing what was a separate 11061-pt cluster).  The freed smalls
    cascade into different flash-bundle pairings.  Hand scan needed.
  - (62495's apparent "1026→43" in the raw drift list is a census
    pairing artifact — both rows are labeled nu-candidate; the real drift
    there is 1026→1014 plus the new 43-pt candidate.)
- **Attribution (all 9 events above: 7 verdict + 386948 + 65025).**
  Same-binary pairs in fresh tags: knobs-OFF reruns
  (`work-mcp1kall-oc19attroff`) reproduce the baseline tables **9/9 SAME**,
  and ON reruns (`work-mcp1kall-oc19attron`) reproduce the ON-sweep tables
  **9/9 SAME**.  Every change is deterministic and knob-caused — none is
  the known QL run-to-run flicker.

### 8.3 Default-flip decision: NOT flipped — owner kept the legacy default

**Owner decision (2026-08-02): keep the current default; do not turn this
on for SBND production.**  Rationale context from the scan round below
(the campaign evidence was presented with old/new Bee sets); the knobs
remain in the code/config for per-event or future use.

The nueCC48 arm is clean (0 verdict changes, tight drift, 444187
recovered).  The mcp1k arm has 7/1000 beam-verdict changes and two large
bundle rearrangements, all deterministic and knob-caused.  Five changes are
new candidates (benign-to-good); 395060 (candidate→TGM), 386948 (−71%
bundle) and 65025 (506→13 pts) can each be either the intended
de-overclustering or collateral — that judgement needs the Bee/mabc hand
scan, so per the house rule ("physics number looks wrong ⇒ report, don't
tune") **both knobs stay default OFF**; the SBND production output is
byte-identical to pr/17.

To review: `work-mcp1kall-{u17on1kb,oc19on1k}/nusel_evt{395060,386948,
65025,315959,56519,62495,314925,348889,407258}/mabc-pr.zip` (baseline vs
ON), guard markers in the ON arm's `.log_e{471,956,546,318,257,532,308,
667,369}.log`.

Bee scan sets v2 (same event order in both, so index N compares directly;
layers per event: raw `img`, QL `clustering`, `op` flashes, plus the full
`-nu` PR-chain layers `track_fit` / `shower_track` / `vertices` / `mc`
particle flow where TaggerCheckNeutrino built a PR graph — fresh
`run_pr_evt.sh data -nu` runs on the sweep pctrees, roots
`work-oc19scan-{old,new}`):
- OLD (baseline `u17on1kb`):
  <https://www.phy.bnl.gov/twister/bee/set/29af12de-2cb2-43fb-b3d8-934000889c0d/event/list/>
- NEW (guard+adopt ON `oc19on1k`):
  <https://www.phy.bnl.gov/twister/bee/set/2ff42e09-f7e9-4bd6-86c8-086254f7fbb0/event/list/>
- Index map: 0=395060 (nu-cand→TGM), 1=386948 (bundle 6063→1777),
  2=65025 (506→13), 3=315959 (no-bundle→30-pt TGM), 4=56519, 5=62495,
  6=314925, 7=348889, 8=407258 (4-8 = new nu-candidates).
- PR-layer availability: idx 1 and 5 have the full PR layers in BOTH arms;
  idx 2 in OLD only (the NEW 13-pt candidate is `nosteiner`).  The tiny new
  candidates (idx 3,4,6,7,8) have no steiner points, hence no PR graph and
  no fit layers in either arm.  Idx 0 (395060) has no PR layers in EITHER
  arm: the standalone `-nu` pipeline (no un-merge visitors) TGM-tags its
  beam cluster even in the baseline (`nu_skip_cosmic` skip,
  TaggerCheckNeutrino.cxx:341) — the OLD nusel nu-candidate verdict there
  comes via the un-merge pipeline, i.e. the event is borderline TGM in
  BOTH arms.  (Superseded v1 sets: 50fbdda3…, e75fd523….)

444187 itself, same layer package (QL from the nueCC48 arms, `-nu` PR runs
in `work-oc19scan-{old,new}/pr_evt444187`; both arms select the nueCC main
— OLD fit L 171.5 cm, NEW 214.5 cm with the recovered fragments):
- OLD (`work-nuecc48-u17on`):
  <https://www.phy.bnl.gov/twister/bee/set/04bfaea1-ebaa-4ed0-a718-683fac80e80d/event/list/>
- NEW (`work-nuecc48-oc19on2`):
  <https://www.phy.bnl.gov/twister/bee/set/43237cfe-bcd1-4132-8a56-b9a224ed08e1/event/list/>  To flip after review: `iso_cathode_guard=true` /
`adopt_nu_fragments=true` TLA defaults in
`wct-clus-matching-perevt.jsonnet` (runner escapes
`SBND_ISO_CATHODE_GUARD=0` / `SBND_ADOPT_NU_FRAG=0` then force legacy).

## 9. Artifacts

- Trace run (fresh tag): `work-oc444187-trace/ql_evt444187/` — per-APA zips
  carry the 16 `0-tr*` stage layers; all-APA zip as usual.
- Committed probes: `oc_stage_trace.py` (stage attribution),
  `oc_stage_gap.py` (per-cluster gap table for one stage's absorb),
  `oc19_census_nuecc48.py` / `oc19_census_mcp1k.py` (the §8.2 censuses;
  whitespace-split parsing, lower-cased label classes).
- Campaign tags: knob-off gates `work-oc444187-off{,2,3}`; smoke
  `work-oc444187-{on,bonly,on2}`; sweeps `work-nuecc48-oc19on{,2}`,
  `work-mcp1kall-oc19on1k`; attribution `work-mcp1kall-oc19attr{off,on}`;
  qlport `sweep/oc19_ub{,2}`.
- Scratch (session-local, not preserved): `/home/xqian/tmp/oc444187/`
  (raw-graph component analysis `probe_imaging_cc.py` / `probe_final.py`,
  layer probe `probe_layers.py`, stage-13 table `probe_stage13.py`).
- Related docs: pr/17 §7.4 (the rescue near-miss on this event), pr/13
  (Bee layer frames — why the flashless monster is only in `img-global`),
  doc 52 (assoc arrays / un-merge), `overclustering-evt11-gamma.md`
  (previous isolated-absorb overcluster, length_cut 20→15).
