# 69 — The STM campaign's determinism claims, audited: doc 61's "legacy-tail non-determinism" was a comparator artifact

**Status: byte-identical. No code or config was changed.** No saved output of the doc 55–67
campaign shows run-to-run non-determinism. The one claim of it (doc 61 §5.1, repeated in doc
56 §9 item 1) came from a numpy comparison that cannot compare `vector<>` branches. On the
files doc 61 judged, every tree is bit-identical, so doc 61's §9-item-1 fix
(`TrackFitting.cxx:9714`) is byte-identical on **all** legacy trees, not only on
`T_rec_charge`. The three real determinism defects the campaign had are all in the hand-scan
tooling (doc 55). Two are fixed; one is worked around, with detection still open.

This is an investigation doc. Corrections to docs 56/61 are left for the next session (§6).

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

## 5. Found on the way, not fixed

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

## 6. For next session (mechanical; anchors are line numbers at `73f65486`)

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
