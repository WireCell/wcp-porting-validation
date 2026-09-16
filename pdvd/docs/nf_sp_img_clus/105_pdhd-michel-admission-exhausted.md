# 105 — The Michel admission layer is exhausted: the diagnostic arm, and why the fix is upstream (PDHD)

**Status (2026-09-15, round 8 of the doc 101/102/103/104 campaign).**
* **Question.** Doc 104 sec 7 left exactly one unrefuted discriminator — *is the muon's Bragg rise at the claimed
  stop, or beyond the far end of the candidate arm?* — and proposed one PDHD arm with `survey_enable` +
  `publish_other_arms` to expose the geometry the stock output does not carry. This round runs both.
* **Answer: the admission layer is exhausted.** Nothing in the new information separates PDHD's Michel false
  positives from its true Michels.
  1. **The arm-profile / Bragg-position test fails** (sec 2). Rebuilt geometrically from the Michel object's own
     points, every feature's AUC is consistent with 0.5 — PDHD `q_far` 0.586 [0.519, 0.667], `slope` 0.519,
     `contrast` 0.481; PDVD the same. And only **3 of PDHD's 9** false positives have an attached arm long enough
     to profile at all.
  2. **`far_full` is 0.0 for all six A1-only false positives** (sec 5). This is the direct test of doc 104 sec 3's
     truncation mechanism — track length reachable beyond the arm, walked to 100 cm with the stop fenced off — and
     it is identically zero. The muon does **not** continue past the arm in the graph, so the truncation leaves no
     signature the admission gate could read.
  3. **The four latent gate inputs do not help** (sec 5). `shower` AUC 0.654, `mip` 0.633, `kink` 0.613,
     `kink5 − kink` 0.316; the best is `len` at 0.743, which is doc 104's already-refuted L2b.
  4. **No companion near-miss** (sec 4). The nearest surveyed piece to every false positive is **20–59 cm** from
     the stop, against the Michel stage's 15 cm radius. The neighbourhood is empty; nothing good was rejected.
* **Correction to doc 104 sec 7.** It called `survey_enable` and `publish_other_arms` "pure-writer knobs, no
  verdict change by construction". **That is wrong**, and `pdhd/pr.jsonnet` says so in the owner's own ruling of
  2026-09-08. Measured here (sec 3): the diagnostic arm moves `is_stm` on 1 of 333 clusters, `reject_bits` on 3,
  `michel_conn_type` on 2, and the muon profile branches on 2 % (`plateau_med`, `contrast`, `muon_len`).
  `michel_found` itself holds on all 333, and the candidate set is identical, which is why the arm is still usable
  to *characterise* the population — but no cut sized on it is quotable against the doc 103 grade without being
  re-run in the graded configuration.
* **PDHD stays at production. No flip, no config change, no toolkit change.**
* **Where the fix is.** Five levers are now refuted across docs 104 and 105, and the label question was closed in
  both directions by `own103h2`. What remains is doc 104 sec 3.3's defect: **the fitted trajectory ends short of
  the muon's own charge.** That is a trajectory campaign, not a tagger one — sec 6.

---

## 0. Repro

```bash
cd pdvd/docs/nf_sp_img_clus/scripts
IMG=/home/xqian/toolkit-dev/wcp-porting-img

# sec 2 -- the Bragg-position / arm-profile discriminator, both detectors, no new arm needed
python3 d105_arm_profile.py                                   > ../figs/105_arm_profile.txt

# sec 3 -- IS THE DIAGNOSTIC ARM THE SAME RECONSTRUCTION?  Run this before believing sec 4 or sec 5.
python3 d105_diag_neutrality.py --a d102hcs --b d105hdiag     > ../figs/105_diag_neutrality.txt
# sec 4 -- companion-piece ledger (role-6 survey rows, rej/d_stop/d_body)
python3 d105_companion_gates.py --arm d105hdiag               > ../figs/105_companion_gates.txt
# sec 5 -- the four gate inputs that exist only in the DEBUG log
python3 d105_stop_arms.py       --arm d105hdiag               > ../figs/105_stop_arms.txt
```

The diagnostic arm itself (61 PDHD events, `libpin_d102`, clus md5 `091e142b9481`, complete 61 / incomplete 0):

```bash
KF=$PWD/../figs/101_tf_prod_pdhd_kf.json
ARM=d105hdiag DET=pdhd SRC=d102hcs JOBS=12 PIN=/home/xqian/tmp/d102/libpin_d102 \
  PR_TLA="-A trackfitting_config=$KF -S retile_sampler_strategy='charge_stepped' \
          -S stm_michel_extra={survey_enable:true,survey_radius_cm:60.0,survey_max_len_cm:25.0,publish_other_arms:true}" \
  ./d102_run_arms.sh
```

**The TLA must carry no spaces inside the object literal.** `run_pr_evt.sh` expands `${PDHD_PR_TLA}` unquoted, so a
space splits the literal into separate argv words. (Typing the same thing at an interactive prompt hits *brace*
expansion instead and splits on the commas — quote it there.) Compiled-config proof: against the A1 job, the
diagnostic TLA adds exactly `publish_other_arms`, `survey_enable`, `survey_max_len_cm`, `survey_radius_cm` and
nothing else.

---

## 1. What doc 104 left

Doc 104 refuted four admission levers (energy floor, length floor, continuation veto, Bragg-at-stop), measured the
mechanism at point level, and reported one lever — a 3 cm floor on attached Michels — that clears the arithmetic
bar and was **not** adopted. Its sec 7 named the one comparison it had not tested and the arm that would expose it.

PDHD's grade after the `own103h2` symmetric check (doc 104 sec 5.3): Michel purity **−0.035**, efficiency +0.085,
`is_stm` −0.016 / −0.011, bar −0.020. One metric, one gap.

---

## 2. The Bragg-position discriminator (`figs/105_arm_profile.txt`)

Doc 104's L4 asked only *is there a Bragg rise at the claimed stop*. The position comparison it is the degenerate
case of asks where the rise is. `T_stm_michel_pts` persists `q` and `x/y/z` for the Michel object's own points
(role 3) but `rr` and `L` are **−1** there — the Michel is not on the chain's arclength — so the arm's internal
profile is rebuilt geometrically, ordering role-3 points by distance from the stop, with `q` normalised to the
muon's **own** `plateau_med` rather than to the fixed 43000 e/cm that `michel_mip` uses.

| feature | PDHD AUC [68 %] | PDVD AUC [68 %] |
|---|---|---|
| `q_near` | 0.401 [0.210, 0.586] | 0.624 [0.482, 0.770] |
| `q_far` | 0.586 [0.519, 0.667] | 0.519 [0.334, 0.710] |
| `slope` = `q_far`/`q_near` | 0.519 [0.369, 0.680] | 0.414 [0.242, 0.582] |
| `contrast` at the stop | 0.481 [0.364, 0.605] | 0.671 [0.509, 0.844] |
| `q_sup` | 0.636 [0.463, 0.802] | 0.570 [0.386, 0.758] |

Every interval contains or sits within noise of 0.5, and no feature agrees in sign across the two detectors. The
one that formally excludes 0.5 — PDHD `q_far` — points the **wrong way**: a truncated muon should carry *more*
charge at the arm's far end, not less.

**Coverage is the second problem.** Only **3 of 9** PDHD false positives (and 5 of 11 on PDVD) have an attached
arm with enough live points to profile. Even a perfect discriminator here would leave two thirds of the class.

---

## 3. The diagnostic arm is not verdict-neutral (`figs/105_diag_neutrality.txt`)

Doc 104 sec 7 asserted the two knobs were pure writers. `cfg/pgrapher/experiment/pdhd/pr.jsonnet` already said
otherwise, in the owner's ruling of 2026-09-08:

> it stays FALSE here. The survey costs a 20–25 % mover rate on the muon's OWN profile branches through
> `preload_clusters` (doc pdvd/53 sec 6.2) for a feature with no physics value

Measured, per `(event, cluster_id)` over the 199 shared branches, 333 candidates both sides:

| | movers |
|---|---|
| candidate set | **identical** (0 only-A, 0 only-B) |
| `michel_found` | **0** |
| `michel_conn_type` | 2 (0.6 %) |
| `is_stm` | 1 (0.3 %) · `reject_bits` 3 (0.9 %) |
| muon profile (`plateau_med`, `contrast`, `muon_len`, `ks_*`) | ~7 (2.1 %) |
| `n_other_published` | 147 (44 %) — the writer knob itself |

So the effect is real but an order of magnitude smaller here than the 20–25 % the ruling quotes for PDVD. **The
consequence is stated once and applies to sec 4 and sec 5:** those columns describe the *diagnostic* arm. They can
characterise the population and motivate a cut; a cut sized on them must be re-run in the graded configuration
before any number is quoted against the doc 103 grade.

---

## 4. The companion ledger (`figs/105_companion_gates.txt`)

Six of PDHD's nine false positives are `conn` 2/3 or too sparse to profile, so the attached-arm tests of sec 2
cannot reach them. For those the tagger reached out to a companion **piece**, and `survey_enable` records every
fitted-but-unclaimed piece within 60 cm with the gate that dropped it (`rej`, doc pdvd/53's table) and its
distances (`d_stop`, `d_body`).

**Two join traps, both found the hard way and both now documented in the script:** `rej`/`d_stop`/`d_body` are
populated **only** on role-6 rows — a "distance" read off a role-3 or role-7 row is the −1 sentinel — and a role-6
row's `seg_id` encodes the cluster the *piece* belongs to, while the stopping muon is in the row's own
`cluster_id` branch. Joining on `seg_id // 1000` silently finds nothing.

**The result:** the nearest surveyed piece to each false positive is **20.4, 25.0, 39.4, 40.2, 40.8 and 59.1 cm**
from the stop (three have none at all), against the Michel stage's `michel_dot_radius_cm` = 15 cm. Nothing was
rejected near the stop, because there is nothing near the stop. The hypothesis that a better companion lost to a
worse one is dead: by gate code, the false positives' 19 surveyed pieces are `{9: 15, 16: 3, 6: 1}`, and code 9 is
*"the survey reached it and neither stage was ever offered it"*.

---

## 5. The four gate inputs that exist only in the log (`figs/105_stop_arms.txt`)

`stm_michel_michel_gate` decides on six quantities. Four are persisted and doc 104 refuted cuts on all of them.
The other two — `a.shower_like` and `a.terminal` — are persisted nowhere, and the component's DEBUG line carries
two richer measurements besides:

```
CheckSTM_Michel stop-arm: cluster C seg S kind K len L cm far_len F cm mip M kink D deg
                          shower B terminal T kink5 K5 far_full FF kink_w KW
```

`kink5` is the same kink over a 5 cm window instead of `dir_window`'s 15 cm; `far_full` is the track length beyond
the arm's far vertex walked to 100 cm **with the stop fenced off**, where `michel_far_len` stops at 25 cm and
reports 0 for a terminal arm. 155 clusters logged 171 arms; 57 of 77 true Michels and 6 of 9 false positives have
an admitted arm.

| feature | AUC [68 %] |
|---|---|
| `len` | 0.743 [0.596, 0.901] — this is doc 104's refuted L2b |
| `shower` | 0.654 [0.553, 0.763] |
| `mip` | 0.633 [0.504, 0.764] |
| `kink` | 0.613 [0.488, 0.753] |
| `far_full` | 0.595 [0.538, 0.667] |
| `kink5 − kink` | 0.316 [0.175, 0.471] |

### 5.1 `far_full` settles the mechanism question

| | `far_full` |
|---|---|
| all six A1-only false positives | **0.0** |
| `terminal` flag on those six | 1 for five of six |

Doc 104 sec 3 showed the muon's charge *is* there — A1's Michel points sit a median 0.27 cm from where production
fitted muon on `029107_28/109`, whose muon is 51 cm shorter. Sec 5 now shows that in A1's own graph **nothing is
reachable beyond the arm**. The charge exists in the cluster and the trajectory simply does not traverse it.

That is why no admission-time feature can find these: at admission the arm genuinely is the end of everything the
tagger can see. The information needed to know better was destroyed upstream, in the fit.

`kink5 − kink` is the one feature formally excluding 0.5, in the inverted sense (false positives' kinks grow more
when measured closer in). On 6 items, with `len` — an already-refuted variable — the strongest thing in the table,
this is not a discriminator; it is noted so round 9 does not re-derive it.

---

## 6. The decision, and where the work goes

**PDHD is not flipped.** Five admission levers are refuted (doc 104's four plus sec 2 here), the latent gate inputs
add nothing, the companion neighbourhood is empty, and the label question was closed in both directions by
`own103h2`. The Michel admission layer is exhausted.

**The remaining defect is doc 104 sec 3.3: the fitted trajectory ends short of the muon's own charge.** It is worth
its own doc because it is not primarily a Michel problem:

* it corrupts the stopping-muon dQ/dx end region that docs pdhd/16, 17, 29 and 50 rest on, whether or not a Michel
  is tagged;
* fixing it moves `stop_x/y/z`, and therefore `is_stm`, on both detectors;
* PDVD runs both levers in production since `8fc6070e`, so any trajectory-end change needs a PDVD re-grade against
  the same A0 baseline its D1 was stated against.

**Round 9 should start by measuring the defect's size, not by fixing it:** across the 61 PDHD events, the
distribution of `muon_len` A0 → A1 and of the stop-point displacement, split by whether a Michel was tagged, with
the sec 3 caveat that the diagnostic arm is not the graded one. Doc 104 sec 3.1's five cases are the seed, not the
sample. Only then is there a basis for a trajectory-end knob and its gate.

---

## 7. Not concluded

* Whether the truncation is driven by the fit knobs or the sampler. Doc 103's per-lever Michel purity (K 0.880,
  S 0.906, A1 0.895) points at the fit knobs; no point-level comparison per lever has been done.
* Why `029107_19/111` (muon *grew* 13 cm, Michel 14 cm away) and `029107_23/41` (`conn` 3, no fitted points, not an
  A0 candidate) behave as they do. They are not truncation and remain unexplained.
* Whether the PDVD survey mover rate really is the 20–25 % the 2026-09-08 ruling quotes; this round measured 2 % on
  PDHD and did not re-measure PDVD.
* The `kink5 − kink` inversion of sec 5. Real or a 6-item fluctuation; a larger false-positive sample would settle
  it, and none is available without more hand scanning.

---

## 8. Files

| path | what |
|---|---|
| `scripts/d105_arm_profile.py` | sec 2, the Bragg-position / arm-profile discriminator, both detectors |
| `scripts/d105_diag_neutrality.py` | sec 3, the per-branch mover check that gates sec 4 and sec 5 |
| `scripts/d105_companion_gates.py` | sec 4, the role-6 survey ledger with the two join traps documented |
| `scripts/d105_stop_arms.py` | sec 5, the stop-arm DEBUG parser (`shower`, `terminal`, `kink5`, `far_full`) |
| `figs/105_arm_profile.txt` | sec 2 |
| `figs/105_diag_neutrality.txt` | sec 3 |
| `figs/105_companion_gates.txt` | sec 4 |
| `figs/105_stop_arms.txt` | sec 5 |

Arm `d105hdiag`: PDHD 028084 + 029107, 61 events, `SRC=d102hcs`, pin `/home/xqian/tmp/d102/libpin_d102`
(clus md5 `091e142b9481` before and after), complete 61 / incomplete 0, median wall 36 s, median peak RSS 1.60 GB (the graded arm d102hcs reads 49 s /
1.66 GB, but the two ran at different job counts and on a differently loaded box, so this is a sanity check that
the survey is not expensive, NOT a like-for-like resource measurement).
