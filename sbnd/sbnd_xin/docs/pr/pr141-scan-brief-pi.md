# doc pr/141 rec 2 — the π-typed split scan  (tag `pisplit-0905-owner`, port 5022)

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./split_display/serve_split_display.sh 5022 --scan-tag pisplit-0905-owner \
    --set docs/pr/pr141-pi-scanset.tsv
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -L 5022:localhost:5022 <user>@wcgpu1.phy.bnl.gov
#   then http://localhost:5022/split_viewer
```

**This one is `split_display`, not `em_display`** — the opposite of the μ-typed
scan. That one asked "is this object a photon?"; this one asks the splitter's own
question: **should this object be cut in two?** Same tool and same verdict
vocabulary you used for the 71-object splitter set.

## The question

**KEEP / SPLIT2 / SPLIT3 / TRIM / UNSURE**, and for a SPLIT, the segment →
part boundary, exactly as before.

## Why these 7

doc pr/139 §22.5 proposed restricting the splitter to **EM-typed** objects:

| candidate class | n | confirmed cuts | fires | purity |
|---|---|---|---|---|
| EM-typed (\|pdg\| = 11) | 63 | 31 | 35 | **0.857** |
| not EM-typed | 8 | 1 | 1 | **0.000** |

— but that rests on **2 non-EM fires**, and §18 and §22.5 both refused to set a
rule on that. These 7 are the whole π-typed (|pdg| = 211, ≥ 100 MeV) population
of the 239-event sample, selected *inside* `build_population` on the arm the
viewer reads, so the node ids are the viewer's own.

**What your answer buys, either way:**

- **All KEEP** → the EM-only restriction costs no cut that would ever fire, and
  it is free purity. That is a flip I could recommend on evidence.
- **Any SPLIT** → the restriction costs that cut, and the lead closes. Equally
  decisive, and it is the outcome §22.5 could not rule out on n = 2.

## The 7, in the order the viewer shows them (descending charge)

| # | event | node | E | length | nseg | conn | proxy |
|---|---|---|---|---|---|---|---|
| 1 | **396222** | `9084` | 175.3 MeV | **318.0 cm** | 25 | 3 | MERGED |
| 2 | **388** | `23028` | 722.6 MeV | 153.5 cm | 29 | 1 | SINGLE |
| 3 | **163543** | `14123` | 402.6 MeV | 84.7 cm | 10 | 1 | SINGLE |
| 4 | **278420** | `18002` | 268.3 MeV | 47.3 cm | 5 | 1 | MERGED |
| 5 | **406125** | `8059` | 144.7 MeV | 33.4 cm | 12 | 1 | SINGLE |
| 6 | **181050** | `15005` | 113.9 MeV | 49.6 cm | 11 | 1 | MERGED |
| 7 | **499577** | `13030` | 106.0 MeV | 23.9 cm | 3 | 1 | SINGLE |

`proxy` is the arm-difference proxy's guess (MERGED / SINGLE), **not** truth —
doc pr/137 §5 measured it contaminated both ways. It is shown only so a
disagreement with your call is visible; please judge the picture, not the label.

Two worth flagging before you look:

- **396222 `9084` is 318 cm long** with 25 segments and `conn_type` 3 — by far
  the most extreme object in the set, and doc pr/139's owner labels already
  carry a **SPLIT3** on it from an earlier scan. If that still reads as a
  3-part object, the EM-only restriction is **not** free and the lead dies
  here.
- **278420 `18002`** is one of the eight **track-typed** objects doc pr/138
  excluded from the splitter scan set, so it has never had a verdict.

## One thing I could not reconcile, stated rather than hidden

§22.5 names the single non-EM false fire as **318769 / 31026, "a π±-typed
object"**. On both arms I can read it, that object is **`pdg = 11`** (210.7 MeV
on the split arm, 108.2 MeV on production) — EM-typed, not π-typed. So either the
id refers to something that has since retyped, or §22.5's description of it is
wrong. It is therefore **not** in this set, and the "1 non-EM fire" half of
§22.5's table should be treated as unverified until that is chased down.
