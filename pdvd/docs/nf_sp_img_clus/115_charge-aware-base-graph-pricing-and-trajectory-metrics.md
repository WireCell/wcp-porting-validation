# doc pdvd/115 — the interiors lever: a charge-aware pricing of the Steiner base graph before the Voronoi step, graded on the track trajectory and the dQ/dx fit together with doc 114's terminal knob

**Owner, 2026-09-18** (after doc 114): *"your idea of trying the interiors lever (a charge-aware base-graph weight
before the Voronoi step) sounds like a good way to give a try. Note, the main evaluation is on the track trajectory.
2D measurement vs. 3D track trajectory's projection. Metrics can also include the 3D track trajectory vs. 3D image. We
can worry less about the later STM, Michel efficiencies, since very likely they need retune to improve, which we can
do in a future session. What we want to evaluate is whether the previous round's knob and this round of knob can help
track trajectory and dQ/dx fit results."*

**Scope.**
- **Lever** (sec 1): one default-OFF knob (`base_weight_blank_alpha` + `base_weight_scope`, toolkit) that prices every
  base-graph edge by the number of zero-charge planes at its endpoints before the Voronoi step that admits the tree's
  interiors; a log-only base-graph edge dump (`WCT_STEINER_BASE_DUMP`) so the construction can be replayed offline.
- **Sizing** (sec 2): the Voronoi / bridge / back-walk construction replayed offline on 10 dumped events per
  detector under 8 pricing levels against a bar frozen before the replay ran; two levels built.
- **Instruments** (sec 3): two new metrics the owner asked for — the 2-D measurement against the projected 3-D
  trajectory per fitted row and wire plane (`d115_proj_resid.py`), and the dQ/dx fit result on the common accepted
  passes through doc 50's engine (`d115_dqdx_compare.py`) — beside the doc 111–114 3-D-vs-image metrics.
- **Grading** (sec 5): four levels — doc 114's `prefer3` (P3), this round's two (BW2 = α 2 scope tree, BWP1 = α 1
  scope tree+path) and their combination (P3BW2) — on PDHD (61 events) and PDVD (120) under a rule frozen before any knob-on arm, with the trajectory and
  dQ/dx clauses gating and the STM / Michel tags reported only. **Knob OFF is byte-identical. Knob ON is NOT
  bit-identical. No default is flipped.**

## Status: the answer

**Round 2 (sec 7, 2026-09-18 — the owner's follow-up):** `tree+path` pricing at α 0.25 / 0.5 / 1 *combined with*
`prefer3`, graded under a second frozen rule (`figs/115r2_pred.txt`, sha `c953f114…`) whose FIX clauses are
detour-scoped (DETOUR length, DETOUR + FIT length, the blank-carried sub-classes, the crawl class, the five spot fits,
`f_low`, shape) and whose arm-wide numbers are no-regression only. No code change; the doc-115 pin.

| level (all `prefer3` + `tree+path`) | DETOUR length H / V | DETOUR + FIT H / V | blank-carried stretches H / V | h1 / h2 / v3 fit | f_low H / V | wiggle H / V | verdict |
|---|---|---|---|---|---|---|---|
| base | 12.4 / 8.2 m | 15.2 / 11.2 m | 365 / 221 | 1.77 / 2.55 / 0.91 cm | 0.049 / 0.016 | 4.4 / 9.7 % | |
| α 0.25 | −52 % / −32 % | −40 % / −18 % | −70 % / −32 % | 0.47 / 0.69 / 1.07 | −5 % / −6 % | +0.1 / +0.9 % | FAIL (H: f_low, shape; V: DETOUR+FIT, blank-carried, v3) |
| **α 0.5** | −53 % / **−40 %** | −41 % / −4 % | −72 % / −40 % | 0.47 / 0.69 / 1.07 | −10 % / −7 % | +4 / +6 % | FAIL (H: shape +0.004; V: DETOUR+FIT, blank-carried −39.8, v3) |
| α 1 | **−61 %** / −33 % | −40 % / +6 % | **−81 %** / −43 % | 0.47 / 0.68 / 1.14 | −13 % / +7 % | +10 / **+16 %** | FAIL (V: DETOUR+FIT, v3, f_low, wiggle); **PDHD passes every clause** |

No level passes both detectors. The PDVD DETOUR + FIT miss at α ≥ 0.5 is **one cluster** (039349_61 cl 67): the
fit grows a 74 cm out-and-back spur along a painted retile stripe in the drift direction (doc 113's ghost class);
without it α 0.5 reads −19.9 % against −20. v3's fit (1.07) is `prefer3`'s cost at that spot (the pricing alone
gives 0.54). **Recommendation (sec 7.5): carry `prefer3` + `tree+path` α 0.5 (arms `d115?p3bwp05`) into the tagger /
PR retune as the working trajectory**; α 1 is the alternative on PDHD alone. The seed-side levers are done: what
remains is gaps (kept by design) and the fit leaving good seeds. Nothing is flipped.

---

**Round 1.** **The question was whether doc 114's terminal knob, this round's base-graph pricing, or both together help the track
trajectory and the dQ/dx fit.** Measured on the full sets (PDHD 61 events, PDVD 120) under the rule frozen before
the arms (`figs/115_pred.txt`, sha `5ca5eedc…`; amendment 1 renames the arm tags only), against this round's
knob-OFF arms, which equal doc 114's base byte for byte:

| level | seed off-ridge S1 (H / V) | fit rows > 1 cm P1 | DETOUR length U1 | W-plane rows > 1 wire off the charge R2D | h1 / v3 seed | h1 / v3 fit | median f_low D1 | median shape D2 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| base | 9.87 % / 7.82 % | 7.80 % / 5.09 % | 12.4 m / 8.2 m | 3.12 % / 1.73 % | 1.97 / 1.64 cm | 1.77 / 0.91 cm | 0.0556 / 0.0157 | 0.2985 / 0.1697 | |
| P3 `prefer3` (doc 114) | −15 % / −6 % | −7 % / −2 % | **−40 %** / −12 % | −2 % / −1 % | 0.68 / 1.64 | 0.47 / 1.14 | −13 % / +1 % | −5 % / −2 % | FAIL |
| BW2 α 2, scope tree | −9 % / −7 % | −4 % / −1 % | −20 % / **−31 %** | −3 % / −4 % | 1.63 / 1.53 | 1.59 / **0.54** | −4 % / −4 % | −2 % / −5 % | FAIL |
| BWP1 α 1, scope tree+path | −16 % / **−16 %** | −8 % / −3 % | **−51 %** / **−33 %** | −3 % / **−9 %** | 0.68 / 1.32 | 0.48 / **0.54** | −15 % / −4 % | +1 % / −2 % | FAIL (wiggle +14 % / +18 %) |
| P3BW2 both | −16 % / −7 % | −7 % / −2 % | **−43 %** / −27 % | −4 % / −5 % | 0.68 / 1.53 | 0.47 / 1.07 | **−19 %** / **−9 %** | **−4 %** / **−6 %** | FAIL |

(`figs/115_verdict.txt`; every clause per level and detector is in sec 5. The S1 / P1 / R2D bars were −20 / −15 /
−10 %, U1 −30 %, the spots ≤ 1.2 cm; the dQ/dx clauses are no-regression bars and pass on every level except P3's
`f_low` on PDVD, +0.0001.)

**1. Both knobs move the trajectory in the right direction, and the base-graph pricing reaches what the terminal
knob could not.** The DETOUR length of the doc-114 catalogue — the stretches where the image is continuous and the
trajectory left it, the class both levers aim at — falls 40–51 % on PDHD and 31–33 % on PDVD under the pricing
(`tree+path`) or the combination, against 12 % on PDVD for the terminal knob alone. Its `D-blank-int` sub-class,
the two-blank interiors of v3's kind that doc 114 named as out of the terminal knob's reach, falls 110 → 38 (PDHD)
and 107 → 51 (PDVD) under BWP1, where the terminal knob had *raised* it (→ 121 / 119). At v3 the fit comes from
0.91 to **0.54 cm** off the ridge under either pricing level (sec 5.3, `figs/115_spot_v3.png`); at h2, doc 111's
crawl spot, the fit comes from 2.55 to **0.69 cm** under BWP1. The pricing alone does not reach h1 (1.63 cm under
scope `tree`: the chord there is carried by one-blank *terminals*, which only the terminal knob or the `tree+path`
scope removes), and the two knobs are complementary: P3BW2 has the terminal knob's h1 and the pricing's DETOUR gain.

**2. The dQ/dx fit result improves where the trajectory does, and never regresses.** On the common accepted
passes, `f_low` — doc 50's charge completeness, the share of live points below 0.4 × the muon plateau — falls 13 %
under P3, 15 % under BWP1 and **19 % under P3BW2** on PDHD (0.0529 → 0.0430; paired 116 tracks better, 92 worse),
and 4–9 % on PDVD; the per-track shape rms against the muon table falls 2–6 %, the population χ² against the table
falls 1500 → 1225 (PDHD, P3BW2) with the 2–7 cm residual-range bins rising 0.57 → 0.61, 0.67 → 0.72, 0.71 → 0.76 of
the reference; `k_pop` is unchanged to 0.2 %. The Bee holes fall on every level but BW2 (PDHD +1 %) and BWP1
(PDVD +1 %).

**3. Why every level still FAILS the frozen rule.** The rule asked for −20 % on the seed, −15 % on the fit rows and
−10 % on the W-plane residual, arm-wide. The levels deliver 6–16 % on the seed, 1–8 % on the fit rows and 1–9 % on
the residual, because the detour class is 5–7 % of the unsupported length (doc 114): the arm-wide off-image share is
dominated by the gap jumps, which the levers keep by design (every gap clause Ga–Gd passes on every level; far
deviations unchanged; the dead-touching tags are lost *less* often than the others). The `tree+path` scope, the
strongest on the trajectory, also raises the seed wiggle 14 % / 18 % (its known cost, doc 111), and v3's seed stays
above the 1.2 cm spot bar (1.32–1.53 cm) although its fit is fixed.

**4. The tags (reported, not gating).** The STM / Michel purities and efficiencies move by −0.005 to −0.04 on PDHD
and the PDVD Michel efficiency by −0.05 to −0.09 under the pricing levels (`figs/115_grade_*.txt`, unlabelled
candidates bounded both ways); the candidate churn is 12–35 tags each way per level. As the owner anticipated, the
taggers were tuned on the old trajectories and will need a retune before any of this is flipped.

**Recommended next step.** The trajectory gains and the dQ/dx gains are real but sub-bar arm-wide, and the tags
move; the honest reading is that the levers are the right ones and the *levels* are not tuned: (a) the
`tree+path` pricing with a smaller α (0.25–0.5, which the sizing showed keeps most of the S1 gain with less
wiggle) combined with `prefer3`, sized on the dump and graded on the fit-side clauses (U1, R2D, D1–D2), which is
where the effect is; (b) the tagger retune the owner named, on the P3BW2 trajectories, before a flip is judged.
Both knobs stay in the tree default-OFF as the instruments.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdvd/docs/nf_sp_img_clus; S=$D/scripts; F=$D/figs
T=/home/xqian/toolkit-dev/toolkit; TR="WCT_STM_PATH_DEBUG=1 WCT_STEINER_GRAPH_DUMP=1"
# --- the build (toolkit 054b72d0), OFF proofs (sec 4)
(cd $T && wcbuild > /home/xqian/tmp/d115/build2.log 2>&1; echo rc=$?)         # rc=0; local/lib newer than the edits
(cd $T && ./build/clus/wcdoctest-clus | tail -3)                                # 442 / 442
mkdir -p /home/xqian/tmp/d115/libpin_d115 && cp -p $T/../local/lib/*.so* /home/xqian/tmp/d115/libpin_d115/   # clus 2efa7fa09325
for det in pdhd pdvd; do bash $S/d102_compile_pr.sh $det d115knoboff; \
  bash $S/d102_compile_pr.sh $det d115knobbw -S steiner_base_weight_blank_alpha=1.0; \
  bash $S/d102_compile_pr.sh $det d115knobbwp -S steiner_base_weight_blank_alpha=1.0 -S "steiner_base_weight_scope='tree+path'"; \
  bash $S/d102_compile_pr.sh $det d115knobboth -S steiner_base_weight_blank_alpha=1.0 -S "steiner_blank_plane_mode='prefer3'"; done   # figs/115_gate_config.txt
P=/home/xqian/tmp/d115/libpin_d115
ARM=d115hoff DET=pdhd SRC=d108hflip JOBS=12 PIN=$P LOGD=/home/xqian/tmp/d115/arm_d115hoff TRACE_ENV="$TR" bash $S/d111_run_arms.sh
ARM=d115voff DET=pdvd SRC=d103vflip JOBS=12 PIN=$P LOGD=/home/xqian/tmp/d115/arm_d115voff TRACE_ENV="$TR" bash $S/d111_run_arms.sh
python3 $S/d111_identity_gate.py --det pdhd --a d114hbase --b d115hoff > $F/115_gate_off_pdhd.txt   # 61 / 61
python3 $S/d111_identity_gate.py --det pdvd --a d114vbase --b d115voff > $F/115_gate_off_pdvd.txt   # 120 / 120
(cd $S && TAG=d115snew PIN=$P JOBS=3 bash d101_sbnd_arm.sh)
python3 $S/d111_sbnd_gate.py --old d114sold --new d115snew --ignore-branch Trun.toolkit_git --ignore-branch Trun.wcp_git > $F/115_gate_sbnd.txt
(cd $IMG/qlport/scripts && LD_LIBRARY_PATH=$P ./sweep_5384.sh d115ub 3 && ./ab_check.sh d115ub d114ub) > $F/115_gate_uboone.txt
# --- the base censuses on this round's OFF arms (== doc 114's base rows)
for x in h:pdhd v:pdvd; do det=${x#*:}; a=d115${x%%:*}off; \
  python3 $S/d113_steiner_census.py --det $det --arm $a --logd /home/xqian/tmp/d115/arm_$a --jobs 4 --out $F/115_steiner_$a; \
  python3 $S/d114_support_census.py --det $det --arm $a --logd /home/xqian/tmp/d115/arm_$a --jobs 4 --out $F/115_support_${det}_off; done
# --- sizing (sec 2): base-graph dump arms (10 events each), the frozen bar, the replay, the selection
(cd $F && sha256sum -c 115_sizing_rule.sha256)
ARM=d115hbd DET=pdhd SRC=d108hflip JOBS=4 PIN=$P LOGD=/home/xqian/tmp/d115/arm_d115hbd TRACE_ENV="$TR WCT_STEINER_BASE_DUMP=1" \
  EVENTS="029107_16 029107_18 028084_7 028084_11 029107_29 029107_10 028084_2 029107_12 028084_9 028084_10" bash $S/d111_run_arms.sh
ARM=d115vbd DET=pdvd SRC=d103vflip JOBS=4 PIN=$P LOGD=/home/xqian/tmp/d115/arm_d115vbd TRACE_ENV="$TR WCT_STEINER_BASE_DUMP=1" \
  EVENTS="039349_20 039252_5 039252_14 039349_7 039252_16 039252_9 039349_28 039349_31 039253_2 039252_12" bash $S/d111_run_arms.sh
python3 $S/d115_base_replay.py --det pdhd --arm d115hbd --jobs 4 --stretches $F/114_support_pdhd_stretches.tsv --out $F/115_replay_pdhd
python3 $S/d115_base_replay.py --det pdvd --arm d115vbd --jobs 5 --stretches $F/114_support_pdvd_stretches.tsv --out $F/115_replay_pdvd
python3 $S/d115_sizing_verdict.py > $F/115_sizing_verdict.txt
# --- the frozen rule, the arms, the clause scripts, the verdict (sec 5)
(cd $F && sha256sum -c 115_pred.sha256)      # 115_pred.txt + its amendment 1 (the arm tags; see sec 5)
DET=pdhd JOBS=4 LEVELS="bw2 bwp1 p3 p3bw2" bash $S/d115_run_levels.sh; DET=pdvd JOBS=4 LEVELS="bw2 bwp1 p3 p3bw2" bash $S/d115_run_levels.sh
DET=pdhd bash $S/d115_post_arms.sh; DET=pdvd bash $S/d115_post_arms.sh
python3 $S/d113_spot_figs.py --out $F/115_spot --logd-root /home/xqian/tmp/d115 \
    --arms pdhd=d115hoff:production,d115hp3:prefer3,d115hbw2:alpha2,d115hbwp1:alpha1+path,d115hp3bw2:prefer3+alpha2 \
    --arms pdvd=d115voff:production,d115vp3:prefer3,d115vbw2:alpha2,d115vbwp1:alpha1+path,d115vp3bw2:prefer3+alpha2
python3 $S/d115_verdict.py > $F/115_verdict.txt
# --- ROUND 2 (sec 7): no build; the same pin.  The sizing gap, the frozen rule, the six arms, the clause scripts, the verdict
for x in h:pdhd v:pdvd; do det=${x#*:}; python3 $S/d115_base_replay.py --det $det --arm d115${x%%:*}bd --jobs 4 \
  --stretches $F/114_support_${det}_stretches.tsv --levels 0:tree,0.25:tree+path,0.5:tree+path,1:tree+path --out $F/115r2_replay_$det; done
python3 $S/d115_sizing_verdict.py --replay-prefix 115r2_replay_ > $F/115r2_sizing.txt
for det in pdhd pdvd; do for L in p3bwp025:0.25 p3bwp05:0.5 p3bwp1:1.0; do bash $S/d102_compile_pr.sh $det d115r2_${L%%:*} \
  -S "steiner_blank_plane_mode='prefer3'" -S steiner_base_weight_blank_alpha=${L#*:} -S "steiner_base_weight_scope='tree+path'"; done; done   # figs/115r2_gate_config.txt
(cd $F && sha256sum -c 115r2_pred.sha256)     # c953f114...; the launcher refuses without it and checks the pin md5
DET=pdhd JOBS=5 LEVELS="p3bwp025 p3bwp05 p3bwp1" bash $S/d115r2_run_levels.sh; DET=pdvd JOBS=5 LEVELS="p3bwp025 p3bwp05 p3bwp1" bash $S/d115r2_run_levels.sh
DET=pdhd bash $S/d115r2_post_arms.sh; DET=pdvd bash $S/d115r2_post_arms.sh
python3 $S/d113_spot_figs.py --out $F/115r2_spot --logd-root /home/xqian/tmp/d115 \
    --arms pdhd=d115hoff:production,d115hp3:prefer3,d115hbwp1:alpha1+path,d115hp3bw2:prefer3+alpha2,d115hp3bwp025:prefer3+alpha0.25+path,d115hp3bwp05:prefer3+alpha0.5+path,d115hp3bwp1:prefer3+alpha1+path \
    --arms pdvd=d115voff:production,d115vp3:prefer3,d115vbwp1:alpha1+path,d115vp3bw2:prefer3+alpha2,d115vp3bwp025:prefer3+alpha0.25+path,d115vp3bwp05:prefer3+alpha0.5+path,d115vp3bwp1:prefer3+alpha1+path
python3 $S/d115r2_verdict.py --anchors > $F/115r2_verdict.txt
for det in pdhd pdvd; do for L in p3bwp025 p3bwp05 p3bwp1; do python3 $S/d115r2_adjudicate.py --base $F/115_support_${det}_off_stretches.tsv \
  --arm $F/115r2_support_${det}_${L}_stretches.tsv --out $F/115r2_newstretch_${det}_$L --top 6; done; done
python3 $S/d114_case_figs.py --stretches $F/115r2_newstretch_pdhd_p3bwp05_new_stretches.tsv $F/115r2_newstretch_pdvd_p3bwp05_new_stretches.tsv \
  --arm pdhd=d115hp3bwp05 --arm pdvd=d115vp3bwp05 --classes D-3live,FIT --top 4 --out $F/115r2_case
python3 $S/d115r2_cluster_fig.py --det pdvd --event 039349_61 --cluster 67 --xr 0 110 --out $F/115r2_case_pdvd_FIT_1_custom \
  --arms d115voff:production,d115vp3bwp025:prefer3+alpha0.25+path,d115vp3bwp05:prefer3+alpha0.5+path,d115vp3bwp1:prefer3+alpha1+path
```

All arms are new tags (M13); their pctrees are symlinks to the production arms' (`d108hflip`, `d103vflip`). Dump
logs are in `/home/xqian/tmp/d115/arm_<arm>/` (scratch, not committed). The instruments were checked on an arm
against itself (zero differences, identical `k_pop`) before use.

## 1. The lever

### 1.1 Where the interiors come from

`create_steiner_tree` (`clus/src/SteinerGrapher.cxx`) builds the Steiner tree on the cluster's base graph
`ctpc_ref_pid` (`CreateSteinerGraph.cxx:311`; the 0.8 / 0.9 same-blob edges added at `:322` for the rough path are
removed again at `:341`, so the graph the tree sees has none). Every base edge is priced by its Euclidean length in
all three build stages (`connect_graph_closely.cxx:609, 826`, `connect_graph_ctpc.cxx:838-870`,
`connect_graph.cxx:436-468`). `create_enhanced_steiner_graph` (`SteinerGrapher.cxx:1607-1797`) then:

1. runs a multi-source Dijkstra from every terminal on that graph (`Graphs.cxx:142-176`, the Mehlhorn
   construction: each vertex gets its nearest terminal, its distance, and the edge it was reached by);
2. keeps, per terminal pair, the cheapest bridging base edge by `dist[s] + w + dist[t]` (the first strict minimum in
   edge order);
3. walks both ends of every selected bridge back along the recorded edges to their terminals — **the vertices on
   those walks are the tree's interiors**, admitted with no charge test at all;
4. gives the reduced graph's edges the weight `length × (0.8 + 0.4 · mean Q0/(Q+Q0))`, Q0 = 10000, with the endpoint
   charges from `calc_charge_wcp` — a factor bounded to [0.8, 1.2] that can re-route among the admitted edges but
   cannot admit the on-image vertices the geometric Voronoi left out.

The prototype does exactly the same (`PR3DCluster_steiner.h:454-458`, `:615`; audit doc pr/29: the solver is
"character-for-character"), so a charge-aware base weight is a toolkit-only, deliberately knob-gated divergence
(M15) and not a port fix — the same footing as `steiner_gap_penalty` (`NeutrinoSteinerGapGraph.cxx:41-43`), which
reweights the *reduced* graph after the tree exists.

### 1.2 The pricing

Every base-graph edge weight becomes

    w' = w · (1 + α · ½ (nz(s) + nz(t)))        nz(v) = number of planes whose charge value at v is exactly 0 (0..3)

before step 1 (`WireCellClus/SteinerBaseWeight.h`, `reweight_base_graph`: a priced *copy* with the same vertex
indices and edge set; the base graph itself is untouched, and with α = 0 the copy is never built). A dead plane
counts as a zero plane: a pricing lever needs no dead-channel exemption to keep gap jumping, because inside a dead
or empty region every route is priced alike and Dijkstra still takes the only one — the penalty reorders routes only
where an on-image alternative exists. That is the difference from doc 114's admission lever, which removes
candidates and had to be blob-scoped to protect dead regions.

Two scopes: **`tree`** (the default when α > 0) uses the priced graph for steps 1–3 only; step 4 multiplies the
production charge factor onto the *geometric* length of the edge (looked up in the base graph), so only the tree's
topology moves and every consumer of `steiner_graph` sees the same pricing model. **`tree+path`** carries the priced
length into step 4 too, so the STM rough path (a Dijkstra on the reduced graph, `TaggerCheckSTM.cxx:1226`) avoids
blank interiors as well — doc 111's `P_paint` lower bound, now applied before the tree instead of after it.

### 1.3 Surface

- `Steiner::Grapher::Config::base_weight_blank_alpha` (C++ default 0 = off) and `base_weight_scope` (`"tree"` |
  `"tree+path"`; a typo is refused in `CreateSteinerGraph::configure`, as is a negative α); both round-trip in
  `default_configuration`.
- `create_enhanced_steiner_graph` gains two trailing defaulted parameters (`routing_graph = nullptr`,
  `priced_path = false`); with the null default every line is the historical one.
- jsonnet `cm.steiner(base_weight_blank_alpha=null, base_weight_scope=null)` with key suppression;
  `pdhd/pr.jsonnet` and `protodunevd/pr.jsonnet` TLAs `steiner_base_weight_blank_alpha`, `steiner_base_weight_scope`
  into both steiner passes; the runners' TLAs of the same names.
- Dump: a `STGW <ident> <alpha> <scope> <routed>` line in every graph dump, and under `WCT_STEINER_BASE_DUMP` (on top
  of `WCT_STEINER_GRAPH_DUMP`) one `STGA <ident> <a> <b> <w_cm> <provenance>` line per base edge — ~13 per point,
  8.0 M lines for PDHD 029107_16, hence a separate switch. `d111s_common.parse_trace(want_base=True)` reads them.
- Doctest `clus/test/doctest_steiner_base_weight.cxx` (3 cases, 61 assertions): the defaults and the parser, the
  factor, the priced copy (base untouched), and the routing effect at its public core (`Graphs::Weighted::voronoi`
  plus the bridge cost): on a 5-vertex toy with a chord through a two-blank vertex and a path through three-plane
  vertices, the historical graph makes the chord bridge cheaper (2 vs 3) and the priced graph the path (4 vs 3).

## 2. Sizing offline, before any knob-on arm

### 2.1 The replay and its check

`scripts/d115_base_replay.py` rebuilds, per cluster with a persisted STM record, the base graph from `STGA`, takes
the dumped terminal set, and replays steps 1–4 of sec 1.1 with scipy (multi-source Dijkstra with predecessors, the
first-minimum bridge per terminal pair in the dumped edge order, the back-walk, the reduced weights with the vertex
charges recomputed from `STGR` under the production `disable_dead_mix_cell = false`, the same-blob post-pass edges),
then replays every dumped STM rough walk on the result and scores it against the image ridge exactly as the
doc-113 census scores the C++ seed.

The bar was frozen first (`figs/115_sizing_rule.txt`, sha `92813a43…`). Its α = 0 check asked for ≥ 99 % of the
dumped tree edges reproduced; the first test event gave 96.1 %, and the cause is the lattice: many vertices sit at
*exactly* equal distance from two terminals, and the tree depends on which terminal such a vertex is assigned to.
Two replays of the same graph with the weights perturbed by a relative 1e-9 agree with each other on only 90.2 % of
their tree edges; the unperturbed replay agrees with the C++ tree on 96.1 %, better than any other tie order would.
The check was amended before the sizing table was produced (`figs/115_sizing_rule_amend1.txt`, sha in the same
file): edge agreement judged against that tie floor, and the walk cost within 0.5 % of the C++ graph's on ≥ 95 % of
walks. Every level is compared with the α = 0 *replay*, so the tie noise enters every level alike.

| detector | events | clusters | dumped tree edges reproduced | tie floor | walk cost within 0.5 % |
|---|---|---|---|---|---|
| PDHD | 10 | 221 | 96.09 % (762,548 / 793,550) | 90.2 % | 225 / 225 |
| PDVD | 10 | 200 | 95.66 % (881,426 / 921,455) | (PDHD floor) | 200 / 207 |

### 2.2 The table

`figs/115_replay_{pdhd,pdvd}.txt`; the walks are the STM rough walks of the 10 events (the doc-114 case events plus
the largest detour-length events per detector, with the spot events).

| level | S1 PDHD (rel) | S1 PDVD (rel) | far ×base H / V | wig ×base H / V | cov Δpt H / V | GAP stretches changed H / V | interiors nz1 / nz2+ PDHD | h1 seed | v3 seed | sizing |
|---|---|---|---|---|---|---|---|---|---|---|
| 0:tree | 11.17 % | 12.84 % | 1.000 / 1.000 | 1.000 / 1.000 | +0.00 / +0.00 | - | 110,771 (63 % off) / 63,270 (89 %) | 1.97 | 1.64 | base |
| 0.25:tree | 10.62 % (-4.9 %) | 12.46 % (-3.0 %) | 0.998 / 0.999 | 1.051 / 0.979 | +0.25 / -0.11 | 1.7 / 1.1 % | 100,724 (67 % off) / 49,338 (91 %) | 1.63 | 1.92 | no |
| 0.5:tree | 10.43 % (-6.7 %) | 12.43 % (-3.2 %) | 0.997 / 0.998 | 1.078 / 1.011 | +0.36 / -0.11 | 1.7 / 1.4 % | 95,012 (70 % off) / 45,604 (92 %) | 1.63 | 1.33 | no |
| 1:tree | 10.24 % (-8.4 %) | 12.34 % (-3.9 %) | 0.997 / 1.000 | 1.060 / 0.993 | +0.54 / -0.02 | 0.7 / 1.7 % | 90,097 (72 % off) / 43,230 (93 %) | 1.63 | 1.33 | no |
| 2:tree | 10.16 % (-9.0 %) | 12.20 % (-5.0 %) | 0.994 / 0.993 | 1.074 / 1.051 | +0.60 / +0.10 | 0.7 / 1.7 % | 86,174 (73 % off) / 41,145 (93 %) | 1.63 | 1.53 | no; built as BW2 (fallback) |
| 0.5:tree+path | 9.21 % (-17.5 %) | 11.57 % (-9.9 %) | 0.992 / 0.992 | 1.147 / 1.119 | +0.91 / +0.24 | 2.7 / 4.4 % | 95,012 (70 % off) / 45,604 (92 %) | 1.26 | 1.33 | no |
| 1:tree+path | 9.26 % (-17.1 %) | 11.43 % (-11.0 %) | 0.989 / 0.986 | 1.199 / 1.204 | +1.01 / +0.40 | 2.1 / 3.9 % | 90,097 (72 % off) / 43,230 (93 %) | 0.69 | 1.33 | no; built as BWP1 (fallback) |
| 2:tree+path | 9.24 % (-17.3 %) | 11.54 % (-10.1 %) | 0.973 / 0.979 | 1.274 / 1.290 | +1.13 / +0.60 | 3.1 / 9.4 % | 86,174 (73 % off) / 41,145 (93 %) | 0.69 | 1.15 | no |

The sizing bar (`figs/115_sizing_rule.txt`): Q1 replayed seed off-ridge length −20 %; Q2 far ≤ 1.10× and ≤ 5 % of
the catalogue GAP stretches with their in-ball seed length changed by > 20 % (a bridge abandoned shows as a detour or
a break); Q3 wiggle ≤ 1.10×; Q4 coverage ≥ base − 1 point. `figs/115_sizing_verdict.txt`:

- **No level qualifies on either detector: Q1 is missed by every level.** Scope `tree` saturates early — −8.4 / −9.0 %
  (PDHD, α 1 / 2) and −3.9 / −5.0 % (PDVD) — although the tree loses a third of its two-blank interiors (PDHD
  63,270 → 41,145; PDVD 93,049 → 60,542) and a fifth of its one-blank ones. Re-routing the tree is not the same as
  re-routing the seed: the rough path is a Dijkstra on the *reduced* graph, whose weights the `tree` scope leaves
  geometric, and the interiors that remain (nz1 still 63 → 73 % off-ridge) are the ones with no priced alternative.
- Scope `tree+path`, where the priced length also enters the reduced graph, gets −17 % (PDHD) / −10 to −11 % (PDVD) at
  α 0.5–2, and brings h1 to 0.69 cm — but it fails Q3: the seed wiggle rises 15–29 % (PDHD 4.00 → 4.59–5.10 %,
  PDVD 8.99 → 10.05–11.60 %), because a walk that pays for blank vertices threads between them (doc 111's jitter
  finding, in the other direction). At α 2 it also starts to change GAP-stretch routes on PDVD (9.4 %).
- **Gap safety holds** for every level up to α 1 on both scopes: far ≤ base, GAP-stretch routes changed on 0.7–4.4 %
  of the stretches, coverage flat or up.
- v3 moves under the pricing (1.64 → 1.33 cm at α ≥ 0.5) but not below the 1.2 cm spot bar; h1 moves only under
  `tree+path`.

Per the frozen fallback (best α by mean Q1 among the Q2-passing levels of a scope), **BW2 = α 2, scope `tree`**
(mean Q1 −7.0 %; fails Q1) and **BWP1 = α 1, scope `tree+path`** (mean Q1 −14.0 %; fails Q1 and Q3) were built as
*unqualified* levels: the arms then measure what the seed replay cannot — the fit rows, the 2-D residual and the
dQ/dx — and the prediction recorded in `figs/115_pred.txt` is that both miss S1's −20 % and that BWP1 fails W.

## 3. The instruments

### 3.1 2-D measurement vs the projected 3-D trajectory (`d115_proj_resid.py`)

The fit persists every row's projection into each plane as a *fractional* global wire rank (`pu`, `pv`, `pw`;
deliberately not truncated, `PdvdMagnifyTrackingVisitor.cxx`) and a drift slice (`pt`); the cluster's measured 2-D
cells are integer (channel rank, slice) with a charge (`T_proj_data`). Every earlier row-side test rounded first and
returned a boolean. This one keeps the fraction: per row and plane, `d_exact` = the projected wire minus the
cluster's nearest charged cell on the row's rounded slice (wire units, signed), `d_pm1` the same over the three
slices s−1..s+1, and *no cell on the slice* as its own state. A plane is dead at a row when the nearest dumped
Steiner vertex within 1.5 cm carries that plane's dead bit (the doc-114 recipe); dead planes are excluded.

W is the clean plane on both detectors (`pt` and `pw` close exactly, doc pdhd/24). U and V carry a known ~1-wire
systematic on PDHD (wrapped planes) and a rounding offset on PDVD (doc 110), so their shares are stated with it,
and the gating clause **R2D** is the W-plane share of live rows with |d_exact| > 1 wire. Also reported: rows off in
≥ 1 live plane (exact slice), the ±1-slice / ±1.5-wire form that equals the doc-111 P2, the `q < 0` rows, and the
paired per-record counts. On the doc-114 arms (3 PDHD events) `prefer3` moves R2D 4.41 → 4.00 % (−9 %), so the
metric sees what the ridge metric saw.

### 3.2 3-D trajectory vs the 3-D image

Unchanged from docs 111–114: the Steiner seed's off-ridge length (S1, `d113_steiner_census.py`), the fit rows' ridge
offset (P1, `d111_eval.py`), the image coverage (P4), the doc-114 support census (unsupported length by GAP /
DETOUR / FIT; the DETOUR length **U1** is this lever's target number), and the spot windows h1 / v3.

### 3.3 The dQ/dx fit result (`d115_dqdx_compare.py`)

Doc 50's engine (`d50_dqdx_rr_cross.py`: `load_ref`, `scale_and_shape`, the 11 residual-range bins, the 3 %
floor; the per-track block duplicated so it runs on one record) on the **common accepted passes** of base and arm
(status 0 in both, ≥ 10 rows): dQ/dx = ((q − offset)/scale)/dx in e/cm, rr re-anchored at the tagger's kink when a
Michel / leftover was cut, live points = finite dQ/dx > 0. Per track: `f_low` (share of live points below 0.40 × the
detector's muon plateau — doc 50's charge-completeness measure, the quantity a trajectory off the charge raises),
the muon-hypothesis free scale k and the scale-free *shape* rms over the populated bins, the Bragg contrast, the
`q < 0` rows and the Bee holes. Per population: `k_pop`, χ² against the detector's own muon table, per-bin ratios.
Paired per track: better / worse / same counts. Gating: **D1** median `f_low` ≤ base, **D2** median shape ≤ base +
0.01, **D3** `k_pop` within ±2 %, **D4** Bee holes ≤ base.

## 4. The knob's OFF gates

| gate | result |
|---|---|
| compiled PR configs with the TLAs unset (`d102_compile_pr.sh`) | md5 `a870511c9b22` (PDHD), `211a49a48229` (PDVD): **unchanged**; with the TLAs set the only differences are the two keys in the two `CreateSteinerGraph` nodes (`figs/115_gate_config.txt`) |
| `wcdoctest-clus` | **442 / 442** (439 before + 3) |
| PDHD 61 events, `d115hoff` (knob unset, dump on) vs `d114hbase` (`d111_identity_gate.py`) | **61 / 61 identical** (`figs/115_gate_off_pdhd.txt`) |
| PDVD 120 events, `d115voff` vs `d114vbase` | **120 / 120 identical** (`figs/115_gate_off_pdvd.txt`) |
| SBND 16 events (`d101_sbnd_arm.sh`, `d115snew`) vs `d114sold` | **BYTE-IDENTICAL, 80 files**, with the two provenance branches `Trun.toolkit_git` and `Trun.wcp_git` ignored — the first comparison differed on every event in `wcp_git` alone, the wcp-porting-img commit hash doc 109 records (`figs/115_gate_sbnd.txt`) |
| uBooNE 35 events (`sweep_5384.sh d115ub` vs `d114ub`) | **zips 35 / 35, tagger 35 / 35** (`figs/115_gate_uboone.txt`) |
| freshness | `local/lib/libWireCellClus.so` 11:30:09 > last edit 11:28:11; pin `libpin_d115` clus md5 `2efa7fa09325`, unchanged at every arm's end |
| G3 (sec 5): `d115?p3` vs `d114?p3` (the prior knob's path untouched by this build) | **61 / 61** and **120 / 120 identical** (`figs/115_gate_p3_{pdhd,pdvd}.txt`) |

## 5. Knob ON: the frozen rule, the arms and the verdict

### 5.1 The rule and the arms

`figs/115_pred.txt` (sha `5ca5eedc…`, frozen 11:56 after the sizing, before any knob-on output; amendment 1 at 12:00
renames the arm tags only: the first launch went out with placeholder TLAs, failed at config compile on every
event, and left output-less tag dirs `d115?bwa` / `d115?bwb` that the runner refuses to reuse — they are left in
place for the owner to remove). Levels, each on the doc-115 pin with the dump on: **P3** `prefer3` (doc 114's knob
re-run on this pin; G3 proves it equals doc 114's arm), **BW2** α 2 scope `tree`, **BWP1** α 1 scope `tree+path`,
**P3BW2** `prefer3` + α 2. All eight arms complete (PDHD 61, PDVD 120; 029107_17 under P3 and BWP1 is doc 114's
case: every output written, no STM candidate line). R ≤ 1.07 on every level.

FIX clauses: S1 seed off-ridge −20 %; P1 fit rows > 1 cm −15 %; U1 DETOUR length −30 %; R2D W-plane rows > 1 wire
off the measured cell −10 %; P5 / P5f spot seed ≤ 1.2 cm and fit ≤ base + 0.2 at h1 and v3; D1 median `f_low` ≤
base; D2 median shape ≤ base + 0.01; D3 `k_pop` within ±2 %; D4 Bee holes ≤ base. NO NEW ISSUES: W wiggle ≤ 1.10×,
P2, P4, P6', Ga–Gd (gap jumping), C, R, G2, G3. T (tags) reported.

### 5.2 Every clause (`figs/115_verdict.txt`)

| clause | P3 H / V | BW2 H / V | BWP1 H / V | P3BW2 H / V |
|---|---|---|---|---|
| S1 seed > 1 cm off (bar −20 %) | 8.37 (−15.2) / 7.35 (−6.1) **F/F** | 8.96 (−9.2) / 7.32 (−6.5) **F/F** | 8.32 (−15.8) / 6.61 (−15.5) **F/F** | 8.34 (−15.5) / 7.25 (−7.3) **F/F** |
| P1 fit rows > 1 cm (−15 %) | 7.22 (−7.4) / 4.99 (−2.0) **F/F** | 7.52 (−3.5) / 5.02 (−1.0) **F/F** | 7.20 (−7.6) / 4.94 (−2.9) **F/F** | 7.25 (−6.9) / 4.99 (−2.0) **F/F** |
| U1 DETOUR m (−30 %) | 7.5 (−39.5) / 7.2 (−12.2) P/**F** | 9.9 (−20.2) / 5.7 (−30.5) **F**/P | 6.1 (−50.8) / 5.5 (−32.9) P/P | 7.1 (−42.7) / 6.0 (−26.8) P/**F** |
| R2D W rows > 1 wire (−10 %) | 3.06 (−2.0) / 1.71 (−1.2) **F/F** | 3.03 (−3.2) / 1.65 (−4.4) **F/F** | 3.03 (−2.9) / 1.58 (−8.6) **F/F** | 3.01 (−3.6) / 1.63 (−5.3) **F/F** |
| P5 spot seed (≤ 1.2 cm) | 0.68 / 1.64 P/**F** | 1.63 / 1.53 **F/F** | 0.68 / 1.32 P/**F** | 0.68 / 1.53 P/**F** |
| P5f spot fit (≤ base + 0.2) | 0.47 / 1.14 P/**F** | 1.59 / 0.54 P/P | 0.48 / 0.54 P/P | 0.47 / 1.07 P/P |
| D1 median f_low (≤ base) | 0.0483 / 0.0158 P/**F** | 0.0539 / 0.0154 P/P | 0.0487 / 0.0152 P/P | 0.0430 / 0.0146 P/P |
| D2 median shape (≤ base + 0.01) | 0.2826 / 0.1668 P/P | 0.2785 / 0.1585 P/P | 0.2959 / 0.1613 P/P | 0.2740 / 0.1526 P/P |
| D3 k_pop (±2 %) | −0.00 / −0.05 % P/P | +0.06 / +0.01 % P/P | +0.12 / −0.15 % P/P | +0.03 / −0.02 % P/P |
| D4 Bee holes / 10 m (≤ base) | 7.79 / 5.74 P/P | 8.71 / 5.70 **F**/P | 8.05 / 5.94 P/**F** | 8.13 / 5.80 P/P |
| W wiggle (≤ 1.10×) | 4.45 / 9.51 P/P | 4.54 / 9.90 P/P | 5.07 / 11.45 **F/F** | 4.60 / 9.96 P/P |
| P2, P4, P6' | all pass | all pass | all pass | all pass |
| Ga truncated fits (≤ 4.48 / 3.37 %) | 1.39 / 0.76 | 1.39 / 0.76 | 1.74 / 0.81 | 1.39 / 0.71 |
| Gb far > 5 cm (≤ 1.10×) | 4.02 / 2.24 | 4.02 / 2.26 | 4.12 / 2.17 | 3.95 / 2.26 |
| Gc dead-touching tags lost vs others | 24.1 vs 28.3 / 15.6 vs 8.0 | 24.1 vs 28.3 / 18.3 vs 12.0 | 20.5 vs 28.3 / 19.1 vs 12.0 | 20.5 vs 30.4 / 17.5 vs 12.0 (all pass) |
| Gd clusters losing every fit (≤ 19 / 9) | 3 / 1 | 1 / 1 | 4 / 2 | 5 / 1 |
| C, R, G2, G3 | pass; R 1.000 / 1.013 | pass; 1.068 / 1.044 | pass; 1.025 / 1.036 | pass; 1.000 / 1.000 |
| T tags (reported) | UNDECIDED / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL |
| **verdict** | **FAIL** | **FAIL** | **FAIL** | **FAIL** |

### 5.3 What moved, and where

**The catalogue classes** (`figs/115_support_<det>_<level>.txt`, stretches):

| class (PDHD / PDVD) | base | P3 | BW2 | BWP1 | P3BW2 |
|---|---|---|---|---|---|
| D-blank-term (one-blank terminal on the seed) | 255 / 114 | 57 / 66 | 152 / 104 | 90 / 91 | 33 / 51 |
| D-blank-int (blank interiors only) | 110 / 107 | 121 / 119 | 72 / 66 | 38 / 51 | 87 / 102 |
| D-crawl | 104 / 99 | 64 / 77 | 82 / 53 | 50 / 42 | 61 / 65 |
| D-3live | 35 / 40 | 43 / 46 | 73 / 41 | 61 / 60 | 79 / 56 |
| FIT (seed supported, fit left it) | 83 / 110 | 92 / 117 | 92 / 135 | 107 / 142 | 115 / 131 |
| GAP-live | 591 / 924 | 596 / 909 | 605 / 918 | 596 / 915 | 603 / 920 |

The pricing does what it was built for — the blank *interiors* fall by a third (BW2) to two thirds (BWP1) — and
the terminal knob does not (it moves chords from terminals to interiors, 110 → 121). The two levers act on
different vertices and add up in P3BW2 (blank-term 33, the terminal knob's number; blank-int 87, between the two).
Two classes grow under the pricing: `D-3live` (a detour whose seed vertices all see three planes: the priced route
takes a charged but off-ridge path where the blank chord used to be) and `FIT` (a supported seed the fit still
leaves), which is where the next round's fit-side clauses should look. The GAP classes are flat within ±2 %.

**The spots** (`figs/115_spot_{h1,h2,v1,v2,v3}.png`, `115_spot.tsv`): h1 seed 1.97 → 0.68 under P3, BWP1 and
P3BW2 (fit 1.77 → 0.47), 1.63 under BW2 (the chord is carried by one-blank terminals, which scope `tree` prices
around but cannot avoid: the seed is a Dijkstra on a reduced graph whose weights stay geometric); h2 fit 2.55 →
0.69 under BWP1 and 0.70 under P3BW2 (2.11 under P3); v3 seed 1.64 → 1.53 / 1.32 (BW2 / BWP1) and fit 0.91 →
**0.54** under both — the S1 peak at s ≈ −5 cm (the two-blank interiors) shrinks and the fit follows the ridge —
while P3 leaves the seed and worsens the fit (1.14); v1 fit 0.53 → 0.38–0.40; v2 fit 1.27 → 1.42 (slightly worse
under every pricing level).

**The 2-D residual** (`figs/115_resid_*`): on both detectors the median |d| of a live row from the nearest charged
cell on its slice is 0.20–0.30 wire in every plane and unchanged by any level; the levers act on the tail. Under
BWP1 the W-plane share > 1 wire goes 3.13 → 3.03 % (PDHD) and 1.72 → 1.58 % (PDVD), U 3.10 → 2.79 / 3.56 → 3.41,
V 3.03 → 2.93 / 4.68 → 4.57; rows off in ≥ 1 live plane (exact slice) 9.08 → 8.54 / 11.56 → 11.25 %. The R2D bar
(−10 %) was set as if the W tail were the detour signal; it is mostly the width and the gap edges (rows with *no*
cell on the slice are counted separately, 2.4 / 1.7 %, and do not move).

**The dQ/dx** (`figs/115_dqdx_<det>_<level>.{txt,png}`): P3BW2 on PDHD — `f_low` 0.0529 → 0.0430, shape 0.2863 →
0.2740, χ² 1500 → 1225, the 0–1 / 2–3 / 3–5 / 5–7 cm bins 0.345 → 0.363, 0.571 → 0.606, 0.671 → 0.720, 0.712 →
0.755 of the reference (the PDHD low-rr deficit doc 50 measured shrinks by a tenth of its depth); on PDVD `f_low`
0.0160 → 0.0146, shape 0.1632 → 0.1526, the bins move within their errors. Paired per track the better / worse
counts are 116 / 92 (`f_low`) and 132 / 115 (shape) on PDHD, 188 / 175 and 235 / 226 on PDVD: a majority, not a
transformation, as expected of a lever that touches 5–7 % of the unsupported length.

**Nothing regresses the gap jumping**: far deviations flat, Ga ≤ 1.74 %, Gd ≤ 5 clusters, and on every level the
base tags whose Steiner graph touches a dead channel are lost *less* often than the other tags on PDHD (20–24 % vs
28–30 %); on PDVD the dead-touching share is 16–19 % against 8–12 % for the others but within the +10-point bar.

## 6. What this round does not reach, and what is not concluded

1. **The arm-wide bars.** S1, P1 and R2D are shares of *all* rows / seed length, 90 % of whose unsupported part is
   gap jumps (doc 114); a detour lever cannot move them by 15–20 % without touching the gaps. The next rule should
   gate on the detour-scoped numbers (U1, the DETOUR sub-classes, the spot fits) and on the dQ/dx (D1–D2), and keep
   S1 / P1 / R2D as no-regression clauses. *(Done in round 2, sec 7.)*
2. **The `tree+path` wiggle.** BWP1 buys its seed gain with 14–18 % more wiggle and a PDVD Bee-hole rise of 1 %;
   the sizing showed α 0.25–0.5 keeps most of the S1 gain with less of it, and that level was not run (the frozen
   selection took the best-Q1 α). *(Round 2: α 0.25 / 0.5 with `prefer3` keep the wiggle at +0.1…+6 %, sec 7.3.)*
3. **v3's seed** stays at 1.32–1.53 cm under the pricing although its fit is 0.54 cm: the S1 peak there is carried
   by two-blank interiors that keep no priced alternative inside the window (the image itself is sparse there).
4. **The `D-3live` and `FIT` growth** under the pricing (sec 5.3) is not adjudicated: whether those are wide-image
   routes (harmless) or new detours through charged noise needs the doc-114 case figures on the arm. *(Round 2,
   sec 7.4: fourteen of sixteen drawn cases are wide-image; one is a 1.79 m painted-stripe spur on PDVD.)*
5. **The tags.** Reported only, as the owner directed; the PDVD Michel efficiency moves −0.05 to −0.09 under the
   pricing levels, and the movers are not owner-reviewed (doc 103 showed such sets flip on review).
6. **The sizing check** reproduces the C++ tree to the tie floor, not exactly (sec 2.1); the levels were compared
   with the replay, not with the C++, and the arms confirm the replay's ordering (BWP1 > P3 ≈ P3BW2 > BW2 on S1).
7. **Housekeeping.** The output-less `d115?bwa` / `d115?bwb` event dirs of the placeholder launch (sec 5.1) are
   left under `work/` for the owner to remove; the dump arms `d115?bd` (10 events each, base-graph dump on) are
   kept in `/home/xqian/tmp/d115` for the next sizing.

## 7. Round 2 — `tree+path` at α 0.25–0.5 with `prefer3`, graded detour-scoped

**The owner's brief (2026-09-18, after round 1):** run the `tree+path` pricing at a smaller α (0.25–0.5) combined
with `prefer3`, graded on detour-scoped clauses (DETOUR length and its sub-classes, the spot fits, `f_low` and shape)
with the arm-wide numbers as no-regression only — the aim being to *finalize the track trajectory fit* so that the
tagger and the PR code can be retuned on it in the next sessions. No code changes: both knobs exist in `054b72d0`,
the doc-115 pin (clus md5 `2efa7fa09325`) is reused unchanged, so every round-1 OFF gate stands and `d115?off`
remains the base.

### 7.1 The levels and the rule

Three arms per detector, all `prefer3` + scope `tree+path`, α **0.25** (`d115?p3bwp025`), **0.5** (`d115?p3bwp05`)
and **1.0** (`d115?p3bwp1`, the anchor of the series: round 1's BWP1 pricing plus `prefer3`); launched together at
JOBS 5 after the compiled-config proof (`figs/115r2_gate_config.txt`: each TLA differs from the OFF config only in
the three keys of the two `CreateSteinerGraph` nodes, with the intended values) and after the rule was frozen
(`figs/115r2_pred.txt`, sha `c953f114…`; the launcher refuses without the sha and re-checks the pin md5; its
refusal before the sha existed was exercised: rc 2). The round-1 arms P3, BWP1 and P3BW2 are re-graded under the
same clauses as anchors (`d115r2_verdict.py --anchors`), so the α series reads against them directly.

FIX, detour-scoped, all required on both detectors: **U1** DETOUR length ≤ base − 30 %; **U1b** DETOUR + FIT length
≤ base − 20 % (the migration guard: a seed fixed but a fit that then leaves it does not count as a gain); **U2**
blank-carried detour stretches (`D-blank-term` + `D-blank-int`) ≤ base − 40 %; **U3** `D-crawl` ≤ base; **P5f**
spot fit maxima h1 ≤ 1.0, h2 ≤ 1.0, v3 ≤ 0.7 cm and every one of the five spots ≤ base + 0.2; **D1** median
`f_low` ≤ 0.90 × base (PDHD) / ≤ base (PDVD) with paired better ≥ worse; **D2** median shape ≤ base with paired
better ≥ worse; **D3** `k_pop` ± 2 %; **D4** Bee holes ≤ base. NO REGRESSION, arm-wide: S1, P1, R2D ≤ base (round
1's FIX numbers demoted), W ≤ 1.10×, P2, P4, P6', Ga–Gd, C, R, and G0 (the pin md5 before and after every arm).
Reported: the tags, the seed spot maxima, the class table, the FIT / `D-3live` provenance and case figures.
Recommendation rule: the smallest-α passing level; a larger passing α is the alternative only if it beats it by
> 10 points on U1 on both detectors; if none passes, the fewest-failing level, named unqualified.

### 7.2 The sizing gap (`figs/115r2_sizing.txt`, `115r2_replay_*`)

α 0.25 `tree+path` had never been replayed. On the same 10-event dumps (seed only, no `prefer3`, judged against the
tie floor as in sec 2):

| level | S1 seed off-ridge (H / V) | wiggle ratio | far ratio | GAP stretches changed | h1 / v3 seed max |
|---|---|---|---|---|---|
| 0.25:tree+path | −16.6 % / −9.0 % | 1.091 / 1.012 | 0.994 / 0.998 | 2.1 / 2.2 % | 1.21 / 1.15 cm |
| 0.5:tree+path | −17.5 % / −9.9 % | 1.147 / 1.119 | 0.992 / 0.992 | 2.7 / 4.4 % | 1.26 / 1.33 |
| 1:tree+path | −17.1 % / −11.0 % | 1.199 / 1.204 | 0.989 / 0.986 | 2.1 / 3.9 % | 0.69 / 1.33 |

The seed gain saturates already at α 0.25 (the priced route is a binary choice per chord: once the penalty exceeds
the geometric saving of the blank short-cut, a larger α changes nothing there), while the wiggle grows with α
(1.09 → 1.20 on PDHD). This is the prediction input; the sizing bar was not used for selection (the owner fixed the
range). The frozen predictions (`115r2_pred.txt`): U1 passes everywhere on PDHD and at α ≥ 0.5 on PDVD; **U1b is the
hard clause on PDVD** (the anchors under it: BWP1 +0.9 % with FIT 3.0 → 5.8 m, P3BW2 −8 %, against −20 %); W fails
at α 1 and passes at 0.25; h2 ≤ 1.0 needs the priced route (P3 alone leaves 2.11); v3 ≤ 0.7 is uncertain because
the pricing alone lowers it (0.54) while `prefer3` raises it (1.14, and 1.07 in P3BW2); a clean PASS on both
detectors was *not* predicted.

### 7.3 Every clause (`figs/115r2_verdict.txt`; H / V; P = pass, **F** = fail)

All six arms complete (PDHD 61, PDVD 120; pin md5 unchanged before and after every arm). Under α 0.5 one event per
detector (028084_13, 039252_17) has every output written but no STM / Michel candidate tagged — the round-1 pattern
of 029107_17 — so the runner's candidate-line check flags it; C passes.

| clause | P3BWP025 (α 0.25) | P3BWP05 (α 0.5) | P3BWP1 (α 1) |
|---|---|---|---|
| U1 DETOUR length (−30 %) | 5.9 m (−52) / 5.6 m (−32) P/P | 5.8 (−53) / 4.9 (**−40**) P/P | 4.8 (**−61**) / 5.5 (−33) P/P |
| U1b DETOUR + FIT length (−20 %) | 9.2 (−40) / 9.2 (−17.9) P/**F** | 8.9 (−41) / 10.8 (−3.6) P/**F** | 9.2 (−40) / 11.9 (+6.3) P/**F** |
| U2 blank-carried stretches (−40 %) | 111 (−70) / 150 (−32) P/**F** | 103 (−72) / 133 (−39.8) P/**F** | 69 (−81) / 127 (−42.5) P/P |
| U3 D-crawl (≤ base 104 / 99) | 57 / 58 P/P | 44 / 49 P/P | 37 / 47 P/P |
| P5f spot fits h1, h2 ≤ 1.0; v3 ≤ 0.7; all ≤ base + 0.2 | 0.47, 0.69 / v1 0.52, v2 1.43, **v3 1.07** P/**F** | 0.47, 0.69 / 0.52, 1.43, **1.07** P/**F** | 0.47, 0.68 / 0.52, 1.13, **1.14** P/**F** |
| D1 median f_low (≤ 0.90× H, ≤ 1× V; paired better ≥ worse) | −5.1 % (111/80) **F** / −6.1 % (203/180) P | **−10.2 %** (112/85) P / −6.7 % (189/189) P | **−12.8 %** (116/85) P / +7.1 % (189/192) **F** |
| D2 median shape rms (≤ base; paired) | 0.2730 → 0.2753 **F** / 0.1652 → **0.1534** P | 0.2730 → 0.2768 **F** / 0.1660 → 0.1583 P | 0.2839 → 0.2810 P / 0.1665 → 0.1640 P |
| D3 k_pop (± 2 %); χ² | +0.01 % P / −0.02 % P; 1385 → 1330 / 920 → **771** | −0.09 / −0.06 % P/P; 1257 → 1264 / 861 → 803 | −0.02 / −0.19 % P/P; 1339 → 1281 / 905 → 923 |
| D4 Bee holes / 10 m (≤ base) | 8.56 → 8.16 / 5.90 → 5.64 P/P | 8.57 → 8.04 / 5.89 → 5.78 P/P | 8.57 → 8.43 / 5.89 → 5.88 P/P |
| S1 seed off-ridge (≤ base) | −19.2 % / −14.5 % | **−20.1 %** / −15.3 % | −18.9 % / −15.2 % |
| P1 fit rows > 1 cm (≤ base) | −8.5 % / −4.1 % | −9.0 % / −3.9 % | −8.6 % / −4.5 % |
| R2D W rows > 1 wire (≤ base) | −4.0 % / −8.5 % | −4.9 % / −9.0 % | −3.8 % / −9.2 % |
| W wiggle (≤ 1.10×) | +0.1 % / +0.9 % P/P | +4.4 % / +5.8 % P/P | +9.5 % P / **+16.3 % F** |
| P2, P4, P6', Ga, Gb, Gc, Gd | all pass (Ga 1.57 / 0.76; Gd 7 / 1) | all pass (1.57 / 0.92; 6 / 0) | all pass (1.22 / 0.92; 7 / 2) |
| C, R, G0 | pass; R 0.975 / 0.968 | pass; 0.971 / 0.957 | pass; 0.953 / 0.983 |
| T tags (reported) | FAIL / FAIL | UNDECIDED / FAIL | UNDECIDED / FAIL |
| **verdict** | **FAIL** (H: D1, D2; V: U1b, U2, P5f) | **FAIL** (H: D2; V: U1b, U2, P5f) | **FAIL** (V: U1b, P5f, D1, W) |

The round-1 anchors under the same clauses: P3 fails H: P5f (h2 2.11); V: U1, U1b, U2, P5f, D1 — BWP1 fails H: D2, W; V: U1b,
U2, D1, D2, D4, W — P3BW2 fails V: U1, U1b, U2, P5f. **No level passes both detectors. On PDHD P3BWP1 passes every
clause and P3BWP05 misses only the shape median, by 0.004 (paired 123 better / 114 worse: noise level). On PDVD the
misses are U1b at every α, U2 at α ≤ 0.5 (−39.8 % at α 0.5 against −40), the v3 spot at every α, and at α 1 the wiggle
and `f_low`.** The rule's fewest-failing level is **P3BWP05**, named UNQUALIFIED (four clauses).

### 7.4 What moved, and where

**The class table** (`figs/115r2_support_*`, stretches; PDHD / PDVD; round-1 anchors for comparison):

| class | base | P3 | BWP1 | P3BW2 | P3BWP025 | P3BWP05 | P3BWP1 |
|---|---|---|---|---|---|---|---|
| D-blank-term | 255 / 114 | 57 / 66 | 90 / 91 | 33 / 51 | 37 / 56 | 39 / 56 | 28 / 51 |
| D-blank-int | 110 / 107 | 121 / 119 | 38 / 51 | 87 / 102 | 74 / 94 | 64 / 77 | 41 / 76 |
| D-crawl | 104 / 99 | 64 / 77 | 50 / 42 | 61 / 65 | 57 / 58 | 44 / 49 | 37 / 47 |
| D-3live | 35 / 40 | 43 / 46 | 61 / 60 | 79 / 56 | 63 / 50 | 71 / 56 | 78 / 66 |
| DETOUR length (m) | 12.4 / 8.2 | 7.5 / 7.2 | 6.1 / 5.5 | 7.1 / 6.0 | 5.9 / 5.6 | 5.8 / 4.9 | 4.8 / 5.5 |
| FIT stretches | 83 / 110 | 92 / 117 | 107 / 142 | 115 / 131 | 104 / 134 | 100 / 143 | 134 / 147 |
| FIT length (m) | 2.8 / 3.0 | 3.0 / 3.3 | 4.1 / 5.8 | 3.5 / 4.3 | 3.3 / 3.6 | 3.1 / 5.9 | 4.4 / 6.4 |

The combination does what round 1 predicted: the blank-carried detours fall by 70–81 % on PDHD (both levers add up:
`prefer3` takes the terminals, 255 → 28–39; the pricing the interiors, 110 → 41–74) and by 32–43 % on PDVD, where
`D-blank-int` stays at 76–94 (the two-blank interiors of v3's kind keep no priced alternative in a sparse image,
sec 6.3). The DETOUR length halves on PDHD (−52…−61 %) and falls a third on PDVD. What grows is `D-3live` (35 → 63–78 /
40 → 50–66: the priced route takes a charged but off-ridge path where the blank chord was) and, on PDVD at α ≥ 0.5,
the FIT length (3.0 → 5.9 / 6.4 m).

**Provenance of the FIT and `D-3live` stretches** (`figs/115r2_newstretch_<det>_<level>.txt`, matched by place
between the base and the arm catalogues, both directions). On PDHD at α 0.5 the 100 FIT stretches are 67 matched
to a base stretch (46 were FIT already, 7 were blank-terminal detours, 5 short gaps, 3 live gaps, 3 blank-interior
detours) and 33 new, 0.63 m in all, median max ridge offset 1.3 cm; 20 base FIT stretches (0.34 m) are gone. The 71
`D-3live` stretches are 54 matched (19 were `D-3live`, 14 were blank-terminal and 7 blank-interior detours: the
seed moved from a blank chord to a charged off-ridge route, the migration the rule expected) and 17 new (0.37 m).
On PDVD at α 0.25 the new FIT is 41 stretches, 0.84 m, diffuse (the five longest carry 47 %); at α 0.5 it is 47
stretches and **2.80 m, of which 1.79 m is one stretch** — event 039349_61, cluster 67, 68 rows, 17.8 cm from the
image — and at α 1 the same stretch out of 3.10 m.

**The case figures** (`figs/115r2_case_*`, the four longest new FIT and `D-3live` stretches per detector at α 0.5,
drawn with doc 114's tool): fourteen of the sixteen are 1.8–4.8 cm long with a max ridge offset of 1.1–2.4 cm and a
max distance to the nearest image point of 0.7–1.8 cm — the fit sits *inside* the image band a centimetre or two
off its PCA ridge (class (a), wide image; e.g. `115r2_case_pdhd_FIT_1_frame.png`, `115r2_case_pdvd_FIT_2_frame.png`).
One case failed to draw (the tool's window indexer; `pdhd_D-3live_3`). The sixteenth is the 1.79 m stretch, which
the doc-114 tool cannot window (no record within 12 cm of its centroid), drawn instead in the detector frame
(`figs/115r2_case_pdvd_FIT_1_custom.png`, `d115r2_cluster_fig.py`): **under α ≥ 0.5 with `prefer3` the fit of
cluster 67 grows a 74 cm out-and-back spur along the drift axis** at constant (y, z) ≈ (144, 20) cm from the
track's kink at x ≈ 0 to x ≈ 93, 40 cm (median) from every image point of the cluster; the fit has 85 rows in that
box against 11 for the base and for α 0.25 (699 → 777 rows, exit length 450 → 553 cm). The Steiner dump explains the
place, not the trigger: the retiled cloud holds 236 points along that line in every arm (a painted stripe along the
drift direction, 146 of them two-plane blank, 47 three-plane), with 58–61 Steiner vertices and 14 terminals on it
in every arm — the doc-111/113 painted-ghost class — and the seed's extent in the box is the same 14 points in every
arm; the fit's extension step walks the stripe only under `prefer3` + `tree+path` at α ≥ 0.5 (round 1's BWP1, α 1
without `prefer3`, and P3BW2 do not: 697 / 705 rows, no far row). One cluster, on a knife edge of the lattice ties
(sec 2.1). Without it, U1b at α 0.5 on PDVD reads −19.9 % (bar −20) and at α 1 −10 %; the rule is applied as
frozen and the level stays FAIL.

**The spots** (`figs/115r2_spot*`): h1 seed 1.97 → 0.68, fit 1.77 → 0.47 at every α (`prefer3`'s fix); h2 fit
2.55 → 0.69 at every α (the priced route's fix; `prefer3` alone left 2.11); v1 fit 0.53 → 0.52; v2 fit 1.27 → 1.43
(α ≤ 0.5) and 1.13 (α 1); **v3 fit 0.91 → 1.07 / 1.07 / 1.14** with seed 1.15 / 1.32 / 1.32. At v3 the two knobs
conflict: the pricing alone brings the fit to 0.54 (round 1, BW2 and BWP1) and `prefer3` alone takes it to 1.14 by
dropping 8 of the window's 14 one-blank terminals (doc 114); their combination inherits `prefer3`'s number at this
spot. The 2-D residual moves as in round 1: W-plane rows > 1 wire off the charge −4…−5 % (PDHD) and −8.5…−9.2 %
(PDVD), the medians unchanged.

**The dQ/dx** (`figs/115r2_dqdx_*`): on PDHD `f_low` falls 5 / 10 / 13 % with α (paired 111/80, 112/85, 116/85 better
/ worse), the shape median is flat within ±0.004 (paired 126/108, 123/114, 121/117), χ² 1385 → 1330, 1257 → 1264,
1339 → 1281; on PDVD `f_low` −6 / −7 / +7 %, the shape median falls 7 / 5 / 2 %, χ² 920 → 771 (α 0.25), 861 → 803,
905 → 923; `k_pop` is unchanged to 0.2 % everywhere, the Bee holes fall on every level.

**The tags** (`figs/115r2_grade_*`, reported): PDHD `is_stm` purity / efficiency 0.959 / 0.629 → 0.957 / 0.591 (α
0.25), 0.952 / 0.640 (α 0.5), 0.946 / 0.656 (α 1); Michel 0.880 / 0.702 → 0.847 / 0.692, 0.855 / 0.683, 0.884 /
0.731 — α 1 raises both efficiencies (+0.027, +0.029) and grades UNDECIDED only through the purity bound. PDVD
`is_stm` 0.982 / 0.704 → 0.963 / 0.685–0.714; Michel 0.940 / 0.725 → 0.914 / 0.678, 0.901 / 0.691, 0.885 / 0.682:
the −0.03…−0.05 Michel efficiency cost of every `prefer3` level (doc 114, round 1) is unchanged by α.

### 7.5 The recommendation: the trajectory configuration for the retune session

1. **The seed-side levers are done.** After this round the unsupported trajectory length is gaps (92–93 %, kept by
   design, every gap clause passes) plus 3–6 m of FIT — a supported seed the fit leaves — and 5–6 m of DETOUR, of
   which the blank-carried part is 69–133 stretches. A smaller α buys nothing (the S1 gain saturates at 0.25) and a
   larger α costs wiggle (α 1: +9.5 / +16.3 %). The remaining detour length is either the two-blank interiors of a
   sparse image (`D-blank-int` on PDVD) or the fit leaving a good seed, and neither is a base-graph or terminal
   question.
2. **The configuration to carry into the tagger / PR retune is `prefer3` + `tree+path` at α 0.5 (P3BWP05, arms
   `d115hp3bwp05` / `d115vp3bwp05`)**, the rule's fewest-failing level, with its four misses named: on PDHD the
   shape median (+0.004, noise); on PDVD U1b (one cluster's 1.79 m spur; −19.9 % without it), U2 (−39.8 % against
   −40) and v3 (`prefer3`'s cost at one spot). On PDHD alone α 1 passes every clause and lifts both tag efficiencies,
   so it is the alternative there; a single cross-detector configuration is α 0.5.
3. **What the retune session must know.** (a) The tags were tuned on the old trajectories; every pricing level
   costs the PDVD Michel efficiency 0.03–0.05 and this round leaves that untouched. (b) The `prefer3` / pricing
   conflict at v3 — and the PDVD U2 / U1b misses — would be settled by the one arm this round did not run, the
   `tree+path` pricing at α 0.5 *without* `prefer3` on PDVD (round 1 ran it only at α 1), if a per-detector choice is
   acceptable; (c) the painted-stripe ghost that drives the U1b miss is the doc-113 retile class, and the
   `retile_mode` knob, not α, is its lever. None of these blocks the retune, which measures the tags on the P3BWP05
   trajectories as they are.
4. **Nothing is flipped.** Both knobs stay default-OFF; adopting P3BWP05 in the production configs is the owner's
   decision, to be taken with the retuned tagger.

## 8. Files

| path | what |
|---|---|
| toolkit `clus/inc/WireCellClus/SteinerBaseWeight.h` | the pricing, the priced copy, the scope parser |
| toolkit `clus/src/SteinerGrapher.{h,cxx}` | `base_weight_*` in `Config`; the routing graph in `create_steiner_tree` and `create_enhanced_steiner_graph`; `STGW` / `STGA` dump |
| toolkit `clus/src/CreateSteinerGraph.cxx` | the two keys |
| toolkit `clus/test/doctest_steiner_base_weight.cxx` | 3 cases |
| toolkit `cfg/pgrapher/common/clus.jsonnet`, `cfg/pgrapher/experiment/{pdhd,protodunevd}/pr.jsonnet` | builder args and TLAs, key-suppressed |
| `pdhd/wct-pr-perevt.jsonnet`, `pdvd/wct-pr-perevt.jsonnet` | runner TLAs |
| `scripts/d111s_common.py` | `STGA` / `STGW` parsing (`want_base`) |
| `scripts/d115_base_replay.py`, `d115_sizing_verdict.py` | sec 2 |
| `scripts/d115_proj_resid.py`, `d115_dqdx_compare.py` | sec 3 |
| `scripts/d115_run_levels.sh`, `d115_post_arms.sh`, `d115_verdict.py` | sec 5 |
| `figs/115_gate_*`, `115_sizing_rule*.txt` (+ `.sha256`), `115_replay_*`, `115_sizing_verdict.txt`, `115_pred.txt` (+ `.sha256`), `115_steiner_*`, `115_eval_*`, `115_compare_*`, `115_support_*`, `115_resid_*`, `115_dqdx_*`, `115_grade_*`, `115_spot*`, `115_verdict.txt`, `115_arms_complete.txt` | results (round 1) |
| `scripts/d115r2_run_levels.sh`, `d115r2_post_arms.sh`, `d115r2_verdict.py` | round 2 launcher (sha + pin gate), clause scripts with the `115r2_` prefix, verdict with the round-1 anchors (sec 7.1, 7.3) |
| `scripts/d115r2_adjudicate.py`, `d115r2_cluster_fig.py`; `d115_sizing_verdict.py --replay-prefix` | FIT / `D-3live` provenance, the detector-frame cluster figure, the round-2 sizing table (sec 7.2, 7.4) |
| `figs/115r2_replay_*`, `115r2_sizing.txt`, `115r2_gate_config.txt`, `115r2_pred.txt` (+ `.sha256`), `115r2_steiner_*`, `115r2_eval_*`, `115r2_compare_*`, `115r2_support_*`, `115r2_resid_*`, `115r2_dqdx_*`, `115r2_grade_*`, `115r2_spot*`, `115r2_newstretch_*`, `115r2_case_*`, `115r2_verdict.txt`, `115r2_arms_complete.txt` | results (round 2) |
