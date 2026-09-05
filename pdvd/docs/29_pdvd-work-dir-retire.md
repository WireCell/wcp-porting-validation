# 29 — pdvd `work/` retire plan (2026-09-04): PLANNED, NOT EXECUTED

**Status: staged and dry-run clean. One command away from execution, held back
deliberately because a peer session is live in this tree.**

Companion to [sbnd_xin doc 100](../../sbnd/sbnd_xin/docs/100_cleanup-two-tree-retire.md),
which covers the sbnd_xin half (executed) and the `~/tmp` sweep (executed).

| | dirs | bytes |
|---|---|---|
| `work/` before | 5444 children | 86 GiB |
| **release (planned)** | **3661** | **47.19 GiB** |
| keep | 1516 | 30.20 GiB |
| out of scope (the `<run6>_light<flash>_<tag>` space) | 267 | ~0 |

Owner scope: *"We want to keep the latest production result as well as their
input."* Asked which depth, the owner chose **option A** — substrate spine +
the gate arms of the three shipped flips + the live round.

## 0. Repro block

```
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
python3 scripts/retire/census_20260904.py /tmp/pdvd_census.json    # 213 arms
python3 scripts/retire/plan_20260904.py                            # 7 interlocks -> PASS
RETIRE_JOBS=16 python3 scripts/retire/archive_records_20260904.py  # integrity 3661/3661
./scripts/retire/retire_20260904.sh                                # DRY RUN (clean)

# *** NOT RUN.  Only after the doc-39 peer round closes: ***
CONFIRM=yes ./scripts/retire/retire_20260904.sh
```

State `scripts/retire/state-20260904/plan.json`; records
`archive/records/pdvd-rounds-20260904/` (7.10 GiB raw → 0.57 GiB, integrity
**3661/3661**, 40.23 GiB of heavy classes dropped).

## 1. `work*` means something different here — do not fork the sbnd planner

In `sbnd_xin`, `work*` globs to ~200 sibling arm dirs. **Here it globs to
exactly one directory**, `work/`, and the retirable unit is the *arm-suffix
group* over `work/<run6>_<idx>_<arm>`. Pointing the sbnd planner at this tree
gives either `dirs=0` — the 2026-08-31 failure mode, where a forked driver
silently read the previous round's list and reported zero — or one catastrophic
`rm -rf work/`. `plan_20260904.py` is a new planner over the arm grammar. The
**interlocks** are carried from the sbnd rounds; the code is not.

## 2. The substrate is a chain, and it is load-bearing five levels deep

```
keep ──▶ d27fresh ──▶ d28dlfp ──▶ d34base ──▶ d37dloff ──▶ d39*  (LIVE)
 │          │
 │          └── 9793 inbound links; stage_pr_tag.sh's documented default
 │              src_tag; the v7-uvwfit baseline of doc pdvd/27
 └── 6208 inbound links, and 960 of them are d27fresh's own
     protodune-sp-dnnroi-frames-anode*.tar.bz2
```

**`keep` IS the imaging input the owner asked to preserve**, not a superseded
arm: it holds the SP+DNNROI frames and `d27fresh` borrows 960 of them. Every
level of that chain stays, including `d37dloff`, which is protected by the
chain rather than by its own name — the live d39 round resolves its pctree
through it.

### 2.1 The first census was wrong and would have inverted the plan

Matching symlink targets with a pattern anchored on `work/` scored `d27fresh`
at **15** inbound links and ranked `keep` as the substrate. Every
`../<evt>_<arm>/…` **relative** target was invisible; the corrected census over
both forms is **20269** links and puts `d27fresh` first at 9793. A wrong
substrate ranking here would have put the production input in the release pool
and left the interlock unable to see it.

**Rule:** resolve a link by walking its normalised path for the last component
that parses as `<run6>_<idx>[_<arm>]`. Never grep the raw target string, and
never `relpath` it against a hard-coded root — the sbnd planner did the latter
and [doc 100 §4](../../sbnd/sbnd_xin/docs/100_cleanup-two-tree-retire.md)
records what that cost. This planner is immune to that defect by construction,
and was tested on one relative and one `/nfs`-alias absolute link.

## 3. What is kept

**Substrate** (§2): `keep`, `d27fresh`, `d28dlfp`, `d34base`, `d37dloff`, plus
`d31r6e2e` and `perfslide`.

**The gate arms of the three shipped owner flips** — kept because they are the
evidence for a production change, not because they are recent:

| flip | shipped | gate arms |
|---|---|---|
| doc 36 — anisotropic ctpc metric | `38245d18` | `d36on` (the flipped default must reproduce it member-for-member), `d36off` |
| doc 37 — Steiner terminals at 0.5 cm | `b38c6ea0` | `d37on05`, `d37off0`, `d37off1` |
| doc 38 — gap-aware end trim at 20 cm | `20773e0b` | `d38h20`, `d38off` (the ON-but-inert control), `d38flip20` |

**Live round:** every `d39*` arm, by prefix.
**Also:** `magnify` (cited by 25 files across docs 23/24/25).

## 4. What is released

187 arms / 3661 dirs — the full sweep sets of the CLOSED rounds, each round's
doc keeping its numbers and its record tarball:

| group | dirs | | group | dirs |
|---|---|---|---|---|
| doc28-perf | 1129 | | doc23-27-early | 284 |
| doc38-endtrim | 843 | | doc32-endcover | 252 |
| doc34-35-metric | 603 | | doc37-terminals | 163 |
| doc36-aniso | 365 | | doc31-steiner | 22 |

## 5. Why it is not executed

A peer Claude session has been running in this tree since 08:16 and committed
`6e7f6350` at 18:07; it wrote `d39base` dirs at 18:06. The owner chose *plan
now, execute after they finish*.

`retire_20260904.sh`'s **interlock A re-runs the whole plan at confirm time**,
which re-derives the symlink graph and repeats the 20-second live-writer
window, so a peer that has resumed will make the driver refuse rather than
delete under them. That is a safety net, **not a licence to run it early**:
`stage_pr_tag.sh` lets the peer create a new inbound link to any arm at any
moment, and a link created between the plan and the `rm` is invisible to both.

## 6. Pre-existing damage, recorded so the post-condition means something

Interlock 4 found **6 broken symlinks already in the tree**: `b12new`, `b12old`,
`r10eager`, `r10fast` and `stm1pf` point at `*_stm1`, an arm deleted outside
this machinery some time ago. All six sit inside arms this round releases, so
after execution the correct count is **zero** and any non-zero value the driver
prints is damage that run caused. Without recording them first, a
"0 broken links" post-condition would have tripped on damage it did not cause —
and a "6 broken links" one would have passed while hiding new breakage.

Same signal doc 89 recorded when the 09-01 sweep's DROP dirs vanished without
the sweep ever running: **the release record drifts, so re-measure current
state rather than diffing against the last plan.**

## 7. `d31r6e2e`, kept conservatively — stated rather than hidden

`d31r6e2e` is listed as substrate on 22 inbound links, but every one of its
consumers (`d32base`, `d32probe`, `d32prod035`, `d32r3*`) is in the release
set, so after this round it has **no consumers at all**. It is 0.32 GiB and
three docs cite it, so it stays; a later round can drop it with one line. Said
plainly here because "it had inbound links" will not be true after execution,
and an inherited protection whose stated ground has quietly expired is exactly
what doc 91 found three times.
