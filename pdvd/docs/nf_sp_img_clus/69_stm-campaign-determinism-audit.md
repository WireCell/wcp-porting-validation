# 69 — The STM campaign's determinism claims, audited: doc 61's "legacy-tail non-determinism" was a comparator artifact

**Status: byte-identical. No code or config was changed.** No saved output of the doc 55–67
campaign shows run-to-run non-determinism. The one claim of it (doc 61 §5.1, repeated in doc
56 §9 item 1) came from a numpy comparison that cannot compare `vector<>` branches. On the
files doc 61 judged, every tree is bit-identical, so doc 61's §9-item-1 fix
(`TrackFitting.cxx:9714`) is byte-identical on **all** legacy trees, not only on
`T_rec_charge`. The three real determinism defects the campaign had are all in the hand-scan
tooling (doc 55). Two are fixed; one is worked around, with detection still open.

This started as an investigation doc. Round 2 (§8, same day) carried out §6: docs 56/61
corrected, `d51g_branch_census.py` tightened and gated, and the harness given a per-item
WebGL-loss record.

Provenance: wcp-porting-img local `73f65486` (remote `df44af7b`), toolkit `8b1374f9`.
Python: uproot 5.7.4, numpy 2.1.1, awkward 2.9.0.

---

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
S=pdvd/docs/nf_sp_img_clus/scripts/d69_full_output_compare.py   # read-only, writes nothing

# (1) doc 61's own comparator replayed beside a bit comparison, on doc 61's saved control dirs
python3 $S --replay-d61

# (2) the full-output gate on three same-config, cross-binary production-mode pairs
python3 $S --before 'pdvd/work/*_d64vleg' --after 'pdvd/work/*_d65vleg' --before-arm d64vleg --after-arm d65vleg
python3 $S --before 'pdvd/work/*_d65vleg' --after 'pdvd/work/*_d66vleg' --before-arm d65vleg --after-arm d66vleg
python3 $S --before 'pdhd/work/*_d65hleg' --after 'pdhd/work/*_d66hleg' --before-arm d65hleg --after-arm d66hleg
```

Mode (2) compares every TTree in `tracking-pr.root` **and** `tracking-stm.root`, and every
shared branch, **bit-identically**. It checks the raw bytes of the flattened values (dtype,
NaN payload and `-0.0` all count) and the per-entry counts at every nesting level. It also
compares `calib-pr-evt*.json` by md5 and `mabc-pr.zip` by `abtest/hash_archive.py`. The
per-round gate `d51g_branch_census.py` reads only `T_stm_michel` and `T_stm_michel_pts`.
Nothing is run and nothing under `work/` is written; every input is an existing saved output.

---

## 1. Summary — every determinism claim in docs 55–67

Found by grepping docs 55–67 and `pdvd/docs/scan/` for
`nondetermin|run-to-run|same-binary|twice|race|last-writer|idempot|walk order|reproducib|concurrent`.

| # | where | claim | verdict |
|---|---|---|---|
| **A** | doc 61 §5.1 (third gate row + paragraph), §6 item 2, §7 row; doc 56 §9 item 1 | the `-nu-legacy` tail (`T_tagger`/`T_kine`/`T_proj_data`) differs run-to-run on the **same** binary ("pre-existing, M4") | **Comparator artifact**: 0 real diffs on every branch of every tree (§2) |
| B | doc 55 §17.6 item 1 (commit `7a38b9cd`) | `mkspec.py` cut a spec from a scan wave still writing, and `apply` placed a pin the scanner had declined (`039349_82/60`); `apply` could add a pin but never remove one | **Real, fixed**: re-applied through the widgets, and `scan_harness.py` now clicks `unset pin` |
| C | doc 55 §17.6 item 2 (commit `7a38b9cd`) | `os.walk` + `recs[key] = r` over 517 files for 509 items: last-writer-wins in filesystem order | **Real, fixed**: `resolve.py` collapses once. It never reached the published register, which regenerates byte-identically |
| D | `scan/pdvd_stm_michel_tranche2_findings.md` §1, doc 55 §17.5 | five concurrent headless browsers exhaust the WebGL context, so degraded PNGs are written silently | **Real, environmental; worked around** (re-shot with two browsers). Per-item detection is still **open** |

Not a software claim: doc 55 §16.2 (pin placement is the least reproducible *scanner*
measurement) is variation between human and agent scanners.

---

## 2. Claim A — the legacy-tail "non-determinism"

### Symptom

Doc 61 gated its §9-item-1 edit (`TrackFitting.cxx:9714`, `0.6*units::cm` →
`m_params.low_dis_limit/2.`) on the `-nu-legacy` chain. It ran 3 PDVD events (`039252_0/10/12`)
before and after, into `pdvd/work/039252_<e>_d61item1a` (pre-fix pin
`$HOME/tmp/d61/libpin_item1_before`) and `…_d61item1b` (post-fix pin). The doc reported
`T_rec_charge` bit-identical and `mabc-pr.zip` hash identical, but `T_tagger`/`T_kine`/
`T_proj_data` **differing**. A control, the pre-fix binary run a second time on evt 0 into
`039252_0_d61item1a2` (under `setarch -R`), "produces the same kind and scope of differences".
The doc concluded this was **pre-existing non-determinism in the legacy neutrino tail**. Doc 56
§9 item 1 and the campaign memory repeat that.

### Root cause — the comparator, not the chain

**The real content differs on 0 branches.** Compared bit-for-bit (§0 mode 1), all four pairs
doc 61 judged are identical on every branch of all four trees it named. That covers item1a vs
item1a2 (same binary twice) and item1a vs item1b (before vs after) on each event. The other
four trees are identical too, as are the calib JSONs (md5 `2a61eda2…` on all three evt-0 dirs)
and the mabc content hash (`1b70d7fd…`). These are the files doc 61 looked at: their mtimes
(21:41–21:45 on 2026-09-09) are earlier than doc 61's commit `87634693` (21:52), and nothing
has rewritten them since.

**The flagged set is the jagged set.** The session transcript (`d9d688aa…jsonl`, tool calls
at L31290 and L31305) holds the inline check doc 61 used:

```python
ta = a[tree].arrays(library='np'); tb = b[tree].arrays(library='np')
...
try:
    ok = np.array_equal(va, vb, equal_nan=True) if va.dtype.kind in 'fc' else np.array_equal(va, vb)
except Exception:
    ok = False
```

With `library='np'`, a `vector<>` branch comes back as a numpy **object array whose elements
are arrays**. `np.array_equal` then compares element by element, and each element comparison
yields an array, not a bool. Either numpy raises
`ValueError: The truth value of an array with more than one element is ambiguous`, which the
`except` turns into `ok = False`, or the reduction returns False. Either way the branch is
marked "DIFFER" whatever its content. `d69_full_output_compare.py --replay-d61` runs that
check, verbatim, next to the bit comparison:

```
039252_0_d61item1a  vs  039252_0_d61item1a2          <- same binary, twice
  T_rec_charge  branches   19 | doc-61 check flagged    0 | jagged    0 | flagged within jagged: True  | REAL diffs 0
  T_tagger      branches 1229 | doc-61 check flagged  190 | jagged  199 | flagged within jagged: True  | REAL diffs 0
      jagged but not flagged: 9, longest per-entry vector among them 1  ['act_cluster_id', 'act_evaluated', ...]
      exception x13: ValueError: The truth value of an array with more than one element is am
  T_kine        branches   34 | doc-61 check flagged    4 | jagged    4 | flagged within jagged: True  | REAL diffs 0
  T_proj_data   branches    6 | doc-61 check flagged    6 | jagged    6 | flagged within jagged: True  | REAL diffs 0
```

The other three pairs look the same: 0 real diffs everywhere. On evt 10 the flag counts match
evt 0; on evt 12 T_tagger flags 199 of 199.

Each flagged branch is jagged. The escapees are the nine `act_*` branches, whose vectors all
have **length exactly 1**; a length-1 array does compare like a scalar. Branches holding
**empty** vectors are flagged. So the flag follows the shape of the data, not its content.
`T_rec_charge` has no jagged branch, which is the whole reason it "passed". A NaN ≠ NaN
explanation was tested and rejected: the only NaN-bearing branch in these files is in
`T_rec_charge`, the tree that passed.

### Why it hid

- **The control shared the defect.** The same-binary control went through the same check, so it
  reproduced the same "differences". "Same kind and scope" was read as proof that the diffs
  pre-existed. It really showed that the check flags a fixed set of branches (the jagged ones)
  whatever the input.
- **The one tree that passed was the one that mattered for the edit**, so the result looked like
  a clean split between the gated product and a noisy tail.
- **M4 was cited as prior art, and it doesn't say this.** CLAUDE.md M4 names the FFTW plan cache
  (fixed) and the DL/SCN vertex. It says nothing about the tagger tail being bit-unstable.
- **`array_equal` under `except: ok = False` fails safe for a gate, but not for a diagnosis.** It
  never produces a false PASS. It does produce a confident false DIFF, and that became a finding.

### Fix (deferred to the next session — nothing edited here)

No code changes are needed. Only the record: docs 61 and 56 and the campaign memory need
amending (anchors in §6). The comparator itself was an inline one-off. No committed script
repeats it: `fv_curved_load.py` (T_cluster) and `pdhd/docs/scripts/d16_stm_energy_scales.py`
(T_stm_michel) pair `library="np"` with `array_equal`, but only on flat branches. For any future
tree comparison, use `d69_full_output_compare.py`, or awkward flatten plus per-level counts.

### Verification

| check | result |
|---|---|
| doc 61's saved control dirs, bit comparison, all 8 trees | **0 real diffs** in all 4 pairs (1 same-binary, 3 before/after) |
| replay of doc 61's check | flagged ⊆ jagged in all 16 tree×pair cells; escapees have per-entry length 1; `T_rec_charge` 0/0 |
| calib JSON md5 / mabc content hash, evt 0 (a, a2, b) | identical / identical |
| file provenance | tracking-pr.root mtimes 21:41–21:45, before doc 61's commit (21:52) |

---

## 3. The wider saved output — no non-determinism in the production-mode chain either

Doc 61's claim was about the legacy chain, but the question was about the campaign's saved
output generally, and the per-round gates read only two trees. So §0 mode 2 was run on three
production-mode (`-nu -stm-fit`) pairs. Within each pair the config is the same:
`PR_TLA="-S stm_michel_extra={survey_enable:true,survey_radius_cm:60.0,survey_max_len_cm:25.0}"`
per docs 64/65/66 §0, with no `publish_other_arms` in any compiled log and 0 role-7 rows. The
binaries differ only by that round's default-OFF code (T6 → T7 → T8).

| pair | event pairs | trees compared (tracking-pr + tracking-stm) | differing | calib md5 | mabc content |
|---|---|---|---|---|---|
| PDVD `d64vleg` vs `d65vleg` | 120 | 15 | **0** | 0 of 119 differ | 0 of 120 differ |
| PDVD `d65vleg` vs `d66vleg` | 120 | 15 | **0** | 0 of 119 differ | 0 of 120 differ |
| PDHD `d65hleg` vs `d66hleg` | 61 | 15 | **0** | 0 of 61 differ | 0 of 61 differ |

The trees are `T_stm_michel`, `T_stm_michel_pts`, `T_rec_charge` (both files), `T_cluster`,
`T_bad_ch` (both), `T_proj_data` (both), `T_stm_pass`, `T_stm_eval` and `Trun` (both), plus
the empty `T_proj`. The only one-sided branches are the ones each round added on purpose
(`bragg_anchor_shift_cm`; T8's eight end-geometry/support fields and `q_sup`). PDVD
`039252_11` has no STM trees and no calib JSON in **all** arms (no candidate), hence 119.

**What this proves, and what it doesn't.**
- N = 301 **cross-binary** event pairs, each run by `d53_run_arms.sh` without `setarch -R`
  (ASLR on) and multi-threaded, in separate processes. Bit identity across all of them is
  evidence of code neutrality *and* of independence from address layout and thread
  scheduling, on these 181 events.
- It is not a same-binary repeat of the production chain. The only true same-binary repeat in
  the record is doc 61's single evt-0 legacy pair (§2), and doc pdhd/16 §9.7's one PDVD event
  on the two STM trees.
- Nothing was re-run for this doc: the cross-binary evidence is stronger than a new N = 2 would
  be.

**STM code check (no issues found).** The `VertexPtr`-keyed maps in the chain's graph walk use
a stable comparator (`StmMichelFunctions.cxx:31-32`, `VertexIndexCmp`). The one
`std::set<VertexPtr>` (`CheckSTM_Michel.cxx:2135`) is membership-only and never iterated.

---

## 4. Claims B–D — the real ones, all in the hand-scan tooling

- **B, a spec cut mid-write** (doc 55 §17.6 item 1). A timing defect: `mkspec.py` snapshotted
  a wave six minutes before the scanner's final rewrite of `039349_82/60`. It did reach a saved
  label: the app placed a pin, rr 4.2, moving the stop 3.86 cm. It was corrected in the same
  round by re-applying through the real widgets, and the patch proved the other 568 rows
  byte-identical. `apply` is now idempotent over a declined pin.
- **C, filesystem-order duplicate resolution** (doc 55 §17.6 item 2). This made the published
  register depend on walk order. `resolve.py` breaks ties to the record that matches the
  label the app wrote, and refuses otherwise. The pushed `pdvd_stm_michel_failures.tsv`
  regenerates byte-identically, so it never reached published output. It was "correct by walk
  order, not by construction", and is now correct by construction.
- **D, WebGL context loss** (tranche-2 findings §1). The saved PNGs from a five-browser run
  can be degraded with nothing flagging the item. Re-shooting with two browsers restores them.
  The harness still neither detects the loss per item nor retries: doc 55 §17.5 carries it as
  found-not-fixed, and it is carried again in §6 below.

---

## 5. Found on the way, not fixed (items 1–2 fixed in round 2, §8.2)

1. **`d51g_branch_census.py` `same()`** (l.85–91) returns True whenever **both** values are
   non-finite. So NaN ↔ +inf ↔ −inf all count as "identical", and a gate PASS cannot see such a
   change. This is lenient, not a false-DIFF source. None of the §3 pairs is affected: the
   bit comparison there has no such exemption.
2. **`d51g_branch_census.py` `load_pts()`** (l.80–81) rounds x/y/z/q to 6 decimals before
   comparing. So "identical point geometry" means equal to 6 decimal places, not
   bit-identical. §3's bit comparison covers the same trees without rounding, and they are
   identical.
3. **The per-round gates read two trees out of 15.** Every round's "byte-identical" was
   measured on `T_stm_michel` + `T_stm_michel_pts` only. §3 shows that for T6–T8 the rest was
   identical too. For earlier rounds it is unmeasured, though nothing suggests a problem.

---

## 6. For next session (mechanical; anchors are line numbers at `73f65486`) — round 2 status in §8

Status: items 1–3 done (§8.1, §8.2); item 4 not done; item 5 partly done (§8.3).

1. **doc 61** `61_michel-moved-stop-veto.md`:
   - l.316, the gate-table row "`T_tagger`/`T_kine`/`T_proj_data` … differ, before vs after":
     make it "identical (doc 69; the diff was a comparator artifact)";
   - l.318–327, the paragraph "The third row looked like a failure …";
   - l.393–396, §6 item 2;
   - l.416, the §7 gate row "legacy-tail noise shown to be pre-existing via same-binary-twice
     control".
   Each needs a correction pointing to this doc.
2. **doc 56** `56_stm-michel-pr-improvement-plan.md` l.441–444 (§9 item 1): the same correction.
3. Optionally tighten `d51g_branch_census.py` (§5 items 1–2), with the change itself gated:
   re-run the last round's gate before and after and show the same verdict.
4. Optionally make `d69_full_output_compare.py` a standard companion to the per-round gate.
5. Claim D: per-item WebGL-context-loss detection plus retry in the scan harness.

---

## 7. Gates

| gate | result |
|---|---|
| `--replay-d61`, 4 pairs × 4 trees | 0 real diffs; flagged ⊆ jagged in every cell |
| full-output, PDVD `d64vleg`/`d65vleg` | BIT-IDENTICAL on every shared output (rc 0) |
| full-output, PDVD `d65vleg`/`d66vleg` | BIT-IDENTICAL on every shared output (rc 0) |
| full-output, PDHD `d65hleg`/`d66hleg` | BIT-IDENTICAL on every shared output (rc 0) |
| the script's empty-comparison refusal (bogus arm `nosucharm`) | `REFUSE: no event pairs matched`, rc 2; a matched glob of 0 branches also refuses |
| positive control after that patch, one event (`039252_0`, `d65vleg`/`d66vleg`) | BIT-IDENTICAL, rc 0 |
| files touched | this doc + `scripts/d69_full_output_compare.py`; nothing under `work/`, `prep-*`, `pdvd/docs/scan/`, or the toolkit repo |

---

## 8. Round 2 (2026-09-10, same day): §6 carried out

**Status: the per-round census gives the same verdict on every past gate pair it was re-run on
(14 of 14 reports byte-identical). No reconstruction code, config or saved output was touched.**

### 8.0 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=$HOME/tmp/d69r2; mkdir -p $W
S=pdvd/docs/nf_sp_img_clus/scripts
# the census before this round: a5bed6a5 locally, bea92d8c on the remote (same file);
# check md5 0af587c12a96d83e6aeb0edf20ff8fef whichever sha you use
git show a5bed6a5:$S/d51g_branch_census.py > $W/d51g_old.py && md5sum $W/d51g_old.py

# 8.2a  the values the change is about, straight into both versions
python3 $S/d69r2_census_value_check.py $W/d51g_old.py $S/d51g_branch_census.py

# 8.2b  past gate pairs: the new report must equal the old one, byte for byte
while read det a b; do
  for v in old new; do
    py=$W/d51g_old.py; [ $v = new ] && py=$S/d51g_branch_census.py
    python3 $py --before "$det/work/*_$a" --after "$det/work/*_$b" \
        --before-arm $a --after-arm $b --pts --out $W/${v}_${a}_${b}.txt >/dev/null 2>&1
  done
  cmp -s $W/old_${a}_${b}.txt $W/new_${a}_${b}.txt && echo "$a -> $b IDENTICAL" || echo "$a -> $b DIFFERS"
done <<'PAIRS'
pdvd d61v d62vleg
pdhd d53h d62hleg
pdvd d62bc d63vleg
pdhd d53h d63hleg
pdvd d63a d64vleg
pdhd d53h d64hleg
pdvd d64vleg d65vleg
pdhd d53h d65hleg
pdvd d65vleg d66vleg
pdhd d53h d66hleg
pdvd d65vleg d65v
pdvd d66vleg d66v
pdvd d66v d67v
pdvd d67v d67ks
PAIRS

# 8.2c  the refusals: each must exit 2 (run the same lines through $W/d51g_old.py to see the old answers)
python3 $S/d51g_branch_census.py --before 'pdvd/work/*_d65vleg' --after 'pdvd/work/*_d66vleg' \
    --before-arm d65vleg --after-arm d66vlegX --out $W/x.txt
python3 $S/d51g_branch_census.py --before 'pdvd/work/*_d65vleg' --after 'pdvd/work/*_d66vleg' \
    --before-arm d65vleg --after-arm d66vle --out $W/x.txt
python3 $S/d51g_branch_census.py --before 'pdvd/work/039252_*_d65vleg' --after 'pdvd/work/039349_*_d66vleg' \
    --before-arm d65vleg --after-arm d66vleg --pts --out $W/x.txt

# 8.3  the hand-scan harness
cd pdhd/stm_michel_scan
T=$HOME/tmp/claude-25225/-home-xqian-toolkit-dev-toolkit/d9d688aa-85d3-4ebf-900f-e2f746766f40/scratchpad/scan/t2
python3 check_shots.py $T/shots_t2_degraded     # scratch copy of doc 55's degraded frames; 187 of 204
python3 check_shots.py $T/shots_t2_fix          # their re-shoot; clean
./selftest_webgl_loss.py --work $W/st                   # PASS (forced loss is caught)
./selftest_webgl_loss.py --work $W/st2 --keep-display   # CANNOT TEST, exit 3, while DISPLAY is dead
```

### 8.1 Docs 61 and 56 corrected

Corrected in place: doc 61 §5.1 (gate row and paragraph), §6 item 2 and the §7 row, and doc 56
§9 item 1. The original wording is kept, struck through, beside a pointer to §2. I grepped every
`.md` under `pdvd/`, `pdhd/`, `sbnd/`, `qlport/` and `abtest/`, plus toolkit `clus/docs/`, for
the claim's wording (`non-bit-stable`, `pre-existing non-determinism`, `legacy tail`,
`same-binary-twice`, …): it appears nowhere else. The other "not bit-stable" hits are about the
DL/SCN vertex. That is CLAUDE.md M4, a separate and real matter, and was left alone. The
campaign memory was corrected in round 1.

### 8.2 `d51g_branch_census.py`: §5 items 1–2 tightened, and three refusals

| | before | after |
|---|---|---|
| `same()`, non-finite values | any two non-finite values equal (NaN = +inf = −inf) | NaN = NaN only; an inf equals only the same inf |
| point-geometry key (x, y, z, q) | `round(x, 6)` | the value itself (`exact()`); NaN maps to a sortable sentinel, so NaN = NaN |
| arm name not the dirs' real suffix | key cut by length, giving some other event's key | **REFUSE**, exit 2 (`event_of()`) |
| arms that share no candidate (or, with `--pts`, no pts) | "BIT-IDENTICAL 0 / 0" | **REFUSE**, exit 2 |

Left as it was, on purpose: `-0.0 == 0.0` still compares equal. It is not one of §5's findings,
and changing the gate's meaning beyond them would be a separate decision.

**Gating the change.** Each past gate pair was run through the pre-change script (from
`a5bed6a5`) and the final one (md5 `a6ca2df7`), and the two text reports were compared with
`cmp`:

| pairs | what they are | reports |
|---|---|---|
| 10 | every legacy-equivalence gate of docs 62–66, PDVD and PDHD | **10 / 10 byte-identical** |
| 4 | feature arms with real movers: `d65vleg→d65v` (72 `is_stm` flips), `d66vleg→d66v` (6), `d66v→d67v` (35), `d67v→d67ks` (15) | **4 / 4 byte-identical** |

So no past verdict moves. The reason was measured, not assumed:
- `T_stm_michel` and `T_stm_michel_pts` in `d61v`, `d66v`, `d67v` and `d66hleg` hold **no** NaN,
  inf or −0.0, so the leniency was never exercised;
- identical reports mean no compared point differs only beyond the 6th decimal.

The change guards future arms. What shows it bites is the value check (8.2a), which feeds both
versions exactly what the change is about:

```
same(u, v)                            old    new   want
NaN vs NaN                           True   True   True
NaN vs +inf                          True  False  False
NaN vs -inf                          True  False  False
+inf vs -inf                         True  False  False
+inf vs +inf                         True   True   True
1.0 vs 1.0+1e-9                     False  False  False
0.0 vs -0.0 (unchanged by design)    True   True   True
pts key equal                         old    new   want
1.0000001 vs 1.0000004               True  False  False
NaN vs NaN                          False   True   True
RESULT: all cases as intended           (14 + 5 cases in the script; excerpt)
```

The old pts key also had the opposite bug: a NaN coordinate compared **unequal** to itself, so a
NaN-bearing point would have read as "geometry changed".

**The negative control found a trap the plan did not name.** Round 1's bogus arm name
(`nosucharm`) only reaches the empty-glob path, which already exited. The dangerous inputs are
near-misses, where the glob is right and the arm name is slightly wrong:

| control | old script | new script |
|---|---|---|
| `nosucharm` (glob matches nothing) | "empty arm", rc 1 | same |
| `--after-arm d66vle` (one character short) | 0 matched, "BIT-IDENTICAL 0 / 0", "NO shared branch moved", **rc 0** | REFUSE, rc 2 |
| `--after-arm d66vlegX` (one character long) | **27 matched, 15 `is_stm` flips, rc 0**: candidates of *unrelated events* paired by colliding truncated keys | REFUSE, rc 2 |
| disjoint event sets, correct suffixes | 0 matched, "BIT-IDENTICAL 0 / 0", **rc 0** | REFUSE, rc 2 |

The one-long case is the one a zero-match refusal alone could not catch, because 27 is not zero.
Its false DIFF is also the same kind of failure as §2: a confident report of differences produced
by the comparison, not by the data.

### 8.3 Claim D: the harness now names the items a lost context spoiled

The changes are additive. The CLI, the PNG names and `context.json` are unchanged.

- **`scan_harness.py`.**
  - `App` collects every context-loss signal as it happens: the `(regl) context lost`
    pageerror, and chromium's console warning.
  - `do_shots` brackets each item. From the first signal on, it names every item on stderr, in
    `OUT/_webgl_lost.txt` (the `--items-file` format), and with exit 1. It names the item before
    the first signal too, because regl throws only at the draw after the loss.
  - It prints which backend the 3-D panel got, once, and writes it into that file's header.
  - The per-item loop body moved verbatim into `_shoot_item()`.
- **`check_shots.py`, committed.** This is the tranche-2 scratch copy that doc 55's repro was
  calling. Its process test now reads `_webgl_lost.txt` instead of a map of scratch logs, and it
  refuses (exit 2) a directory that holds no item dirs.
- **`selftest_webgl_loss.py`.** The causal control, below.

Not done: **retry**. Nothing re-shoots automatically; the named items go back through
`--items-file`, with fewer browsers at a time. The item asked for detection, and a retry loop
would reshape a tool that a concurrent session (doc 68's smx3 scan) is using.

**`check_shots.py` on doc 55's own frames** (read-only; scratch copies under the doc-55 session's
scratchpad, path in §8.0):

| shots dir | items | `c_3d_stop` colours min / median | below 1000 | verdict |
|---|---:|---|---:|---|
| `shots_t2_degraded` (output of the two context-lost processes) | 204 | 381 / 722 | 187 | 187 to re-shoot, every one inside the 204 lost-process items |
| `shots_t2_fix` (their re-shoot, two browsers) | 204 | 1421 / 4069 | 0 | clean |
| `shots_t2` (the set the scan used) | 509 | 1168 / 3943 | 0 | clean |

The first two rows reproduce doc 55 §13.2's table.

**The selftest's first run found something else.** It could not force anything: `contexts seen 0`.
Headless chromium on wcgpu1 had **no WebGL at all**, in and out of the Bash sandbox, so Bokeh drew
the 3-D panel with Canvas2D and there was no context to lose. Chromium's own log says why:
- ANGLE's SwiftShader backend calls `xcb_connect()` on `$DISPLAY`;
- in these shells `$DISPLAY` names a dead forwarded X display (`localhost:14.0`);
- so `xcb_connect()` fails, the GPU process exits ("errors during initialization"), and WebGL is gone.

With `DISPLAY` removed, the same binary gets `ANGLE (… SwiftShader driver)` and
`WEBGL_lose_context`. So the selftest clears `DISPLAY` for its browser, and `--keep-display` shows
the fallback. A 3-D panel that is not on WebGL is reported CANNOT TEST (exit 3), not PASS.

| run | 3-D backend | control arm | forced arm | `check_shots` on the forced arm | result |
|---|---|---|---|---|---|
| default (`DISPLAY` cleared) | webgl | nothing named | loss forced on entering item 2; named `039252_0/77` (margin), `039252_15/77`, `039252_15/91`; the file agrees | `c_3d_stop` 374–375 on items 2–3 (control 3167–4081) | **PASS**, exit 0 |
| `--keep-display` | canvas2d | nothing named | nothing to lose | clean | **CANNOT TEST**, exit 3 |

The forced loss drops the colour count to 374–375, inside doc 55's degraded population (min 381,
median 722), so on the one controlled case the colour test and the harness record agree.

**For the owner: this bears on the scan pictures.** Whether a `shots` run gets WebGL depends on
the `DISPLAY` of the shell that launches it:
- on 09-09 it had WebGL, since two tranche-2 processes logged regl errors;
- in this session's shells it did not.

Canvas2D frames cannot suffer context loss, and their colour counts sit in the clean range (3070–4358
on the 3 items here). But they are not drawn by the renderer the owner's browser uses. The harness now
prints which one each run got. Whether `shots` should pin one backend (by clearing `DISPLAY`, or by
refusing Canvas2D) was left to the owner. **Owner, 2026-09-10: the current setup is sufficient,
and no backend is pinned.** The harness keeps launching chromium as it always has.

### 8.4 Gates (round 2)

| gate | result |
|---|---|
| `d69r2_census_value_check.py`, old vs new census | 19 cases as intended. The old version is wrong on NaN↔±inf, +inf↔−inf, sub-1e-6 points and NaN points |
| census old vs new on 14 past gate pairs (`cmp` of the reports) | **14 / 14 byte-identical** (final md5 `a6ca2df7`) |
| census refusals | `d66vle`, `d66vlegX`, disjoint event sets → REFUSE, rc 2 each; `nosucharm` → "empty arm", rc 1 (pre-existing) |
| `check_shots.py` on doc 55's frames | 187 / 204 on the degraded set; 0 on the re-shoot; 0 on the scanned set; empty dir → REFUSE, rc 2 |
| `selftest_webgl_loss.py`, default | **PASS**, exit 0 (harness md5 `ccd16724`, the same before and after the run) |
| `selftest_webgl_loss.py --keep-display` | **CANNOT TEST**, exit 3 |
| the real CLI, `./scan_harness.py … shots --items 039252_0/77`, inherited (dead) `DISPLAY` | rc 0; one line `3-D panel backend canvas2d`; `_webgl_lost.txt` with its header and no keys; all 7 PNGs + `context.json`; `check_shots.py` clean; the scratch labeldir stays empty |
| files touched | docs 55, 56, 61, 69 and the tranche-2 findings; `scripts/d51g_branch_census.py`; new `scripts/d69r2_census_value_check.py`; `pdhd/stm_michel_scan/scan_harness.py`; new `check_shots.py` and `selftest_webgl_loss.py`. Nothing under `work/`, no reconstruction code or config, nothing in the toolkit repo |
