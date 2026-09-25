| arm | store | device | plane-sets | coord mismatches | max abs d | median max abs d | median mean abs d | min cosine | gate |
|---|---|---|---|---|---|---|---|---|---|
| `_fmf4` | f4 | cpu | 69 | 0 | 9.78e-06 | 2.21e-06 | 2.02e-07 | 0.9999997 | max abs d < 1e-4, cos > 0.9999: PASS |
| `_fmgpuf4` | f4 | gpu | 69 | 0 | 5.11e-05 | 9.30e-06 | 7.80e-07 | 0.9999996 | same bar (doc 01 said < 1e-3): PASS |
| `_fm` | f16 (u2) | cpu | 69 | 0 | 1.95e-03 | 9.77e-04 | 9.62e-05 | 0.9999996 | reported: n/a (half rounding) |
| `_fmgpu` | f16 (u2) | gpu | 69 | 0 | 1.95e-03 | 9.78e-04 | 9.62e-05 | 0.9999997 | reported: n/a (half rounding) |
