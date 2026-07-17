# Long unmatched-cluster census — tag `nm3`

Target = scorer `missed` (long objective positives, >= 25 cm or >= 100 pts, tol 0.5 us).  Total missed **95**; excluding class-A anchored-elsewhere, real Bee non-matches **95**.

## Mechanism classes

| class | count | meaning |
|---|---|---|
| A_anchored | 0 | rides a matched group anchor — NOT a Bee non-match |
| B_nobundle | 0 | containment-culled at every T0 — needs new C++ |
| C_gatefail | 91 | rescue-reachable candidate exists — gate/precull lever |
| D_wrongtime | 4 | bundles exist but none near truth time — photon-model |

## Offline what-if (precull pool, additive over current matched)

recovered = adopted at a truth-positive time (real win); phantom = truth-negative time; **wrongflash** = known positive adopted at the WRONG flash (dt>tol; provably wrong, the frozen scorer counts it only as unknown); unlabeled = no verdict (rescan-only). precision = recovered / (recovered+phantom+wrongflash).

| gate set | recovered | phantom | wrongflash | unlabeled | precision |
|---|---|---|---|---|---|
| base_.25/15/.3-3 | 8 | 1 | 13 | 91 | 0.36 |
| t_.18/4/.5-2 | 4 | 0 | 9 | 43 | 0.31 |
| t_.15/3/.5-2 | 3 | 0 | 3 | 19 | 0.50 |
| t_.12/2/.6-1.7 | 2 | 0 | 0 | 5 | 1.00 |
| t_.10/2/.6-1.6 | 0 | 0 | 0 | 0 | n/a |

## Per-event missed long clusters

### evt 298567 (9 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 18 | 300.7 | 2150 | 3222.1 | gold | C_gatefail | 1 near bundle(s), 1 precull |
| 141 | 269.9 | 635 | 3112.4 | gold | C_gatefail | 1 near bundle(s), 1 precull |
| 78 | 172.2 | 1249 | -113.8 | gold | C_gatefail | 1 near bundle(s), 1 precull |
| 4000067 | 153.4 | 613 | 4001.0 | gold | C_gatefail | 1 near bundle(s), 1 precull |
| 4000079 | 140.6 | 577 | 3472.2 | gold | C_gatefail | 1 near bundle(s), 1 precull |
| 4000101 | 131.2 | 329 | 1960.7 | gold | C_gatefail | 1 near bundle(s), 1 precull |
| 80 | 115.5 | 352 | 884.2 | gold | C_gatefail | 1 near bundle(s), 1 precull |
| 64 | 114.5 | 341 | 4737.0 | gold | C_gatefail | 1 near bundle(s), 1 precull |
| 97 | 35.9 | 172 | 1946.2 | gold | C_gatefail | 1 near bundle(s), 1 precull |

### evt 298581 (6 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 4000032 | 411.5 | 3571 | 205.8 | med | D_wrongtime | 16 bundle(s), none within 0.5us |
| 4000023 | 398.3 | 2625 | 501.7 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 168 | 248.5 | 1844 | -985.8 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000348 | 181.3 | 3176 | 1334.2 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000240 | 85.5 | 300 | 2150.4 | high | C_gatefail | 1 near bundle(s), 1 precull |
| 4000127 | 36.8 | 38 | 682.0 | med | C_gatefail | 1 near bundle(s), 1 precull |

### evt 298595 (2 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 32 | 402.1 | 2858 | 1429.0 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000076 | 367.2 | 1703 | 1429.0 | med | C_gatefail | 1 near bundle(s), 1 precull |

### evt 298609 (3 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 4000144 | 397.7 | 2619 | -848.5 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 102 | 334.8 | 2665 | 2899.7 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000064 | 108.5 | 723 | -1865.9 | med | C_gatefail | 1 near bundle(s), 1 precull |

### evt 298623 (7 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 4000017 | 537.8 | 4110 | 1959.2 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 133 | 475.0 | 2614 | 1959.2 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000006 | 472.1 | 40740 | 2403.7 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 5 | 258.2 | 1766 | -614.1 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000016 | 160.8 | 587 | -614.1 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000082 | 78.3 | 501 | 2284.6 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 128 | 17.2 | 196 | -1750.0 | med | C_gatefail | 1 near bundle(s), 1 precull |

### evt 298637 (2 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 59 | 159.1 | 1534 | 1373.5 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 156 | 128.3 | 790 | 1391.7 | med | D_wrongtime | 49 bundle(s), none within 0.5us |

### evt 298651 (10 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 33 | 407.9 | 2494 | 300.1 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000067 | 397.8 | 6470 | -784.4 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000031 | 386.0 | 16354 | 3709.2 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000166 | 356.1 | 1138 | 300.1 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 67 | 188.6 | 1184 | -490.2 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000432 | 175.6 | 405 | -1195.0 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 15 | 120.8 | 412 | -1669.4 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000519 | 77.3 | 342 | 2700.8 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 37 | 68.6 | 536 | -1972.8 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 85 | 63.2 | 329 | 2875.0 | high | C_gatefail | 1 near bundle(s), 1 precull |

### evt 298665 (1 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 4000260 | 331.1 | 2484 | 1392.4 | med | C_gatefail | 1 near bundle(s), 1 precull |

### evt 298679 (6 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 4 | 657.2 | 3704 | 1971.5 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000028 | 264.6 | 1299 | 546.7 | high | C_gatefail | 1 near bundle(s), 1 precull |
| 4000167 | 171.9 | 980 | 1973.0 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 5 | 171.1 | 919 | 546.7 | high | C_gatefail | 1 near bundle(s), 1 precull |
| 120 | 171.0 | 793 | -405.9 | high | C_gatefail | 1 near bundle(s), 1 precull |
| 4000257 | 130.7 | 574 | 3532.6 | med | C_gatefail | 1 near bundle(s), 1 precull |

### evt 298693 (6 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 4000245 | 331.3 | 2000 | 3156.2 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000076 | 317.1 | 3264 | 1630.9 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000174 | 295.5 | 2017 | -55.2 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000270 | 211.2 | 666 | 101.5 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 129 | 146.3 | 748 | -1671.1 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000170 | 51.8 | 303 | -1100.4 | high | C_gatefail | 1 near bundle(s), 1 precull |

### evt 298707 (4 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 5 | 273.1 | 1842 | -892.8 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000004 | 260.0 | 1132 | -892.8 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000098 | 180.2 | 580 | 3593.9 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000205 | 166.6 | 1144 | -1404.3 | med | C_gatefail | 1 near bundle(s), 1 precull |

### evt 298721 (5 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 4000303 | 268.3 | 1971 | 2890.8 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 23 | 254.1 | 1824 | -630.0 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 63 | 199.3 | 1198 | 3697.1 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 36 | 137.5 | 495 | 3689.7 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 100 | 40.4 | 253 | -2138.6 | med | D_wrongtime | 66 bundle(s), none within 0.5us |

### evt 298735 (11 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 4000030 | 300.1 | 7528 | -1152.9 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 126 | 287.0 | 1542 | 2911.0 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 163 | 220.0 | 1388 | -1172.2 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000148 | 187.8 | 1269 | 1226.2 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000045 | 179.8 | 692 | 2420.5 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 3 | 162.2 | 723 | 81.1 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 130 | 144.8 | 737 | 278.7 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 110 | 127.2 | 1072 | 1226.2 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000314 | 126.8 | 879 | -1199.3 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 94 | 91.5 | 702 | 3833.1 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 133 | 19.5 | 119 | -2239.7 | med | D_wrongtime | 71 bundle(s), none within 0.5us |

### evt 298749 (2 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 9 | 277.6 | 2033 | 60.7 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000002 | 76.1 | 502 | 60.7 | med | C_gatefail | 1 near bundle(s), 1 precull |

### evt 298763 (3 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 4000006 | 315.7 | 2127 | 1312.7 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 63 | 154.2 | 831 | 1515.9 | high | C_gatefail | 1 near bundle(s), 1 precull |
| 4000354 | 111.5 | 2204 | 3963.9 | med | C_gatefail | 1 near bundle(s), 1 precull |

### evt 298777 (6 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 4000033 | 405.2 | 5689 | 1700.2 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000245 | 339.8 | 12159 | 523.1 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 67 | 336.6 | 2581 | 49.9 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 125 | 127.9 | 745 | -2131.6 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000250 | 109.0 | 575 | -1384.3 | high | C_gatefail | 1 near bundle(s), 1 precull |
| 77 | 82.7 | 583 | -1799.0 | med | C_gatefail | 1 near bundle(s), 1 precull |

### evt 298791 (7 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 4000021 | 581.9 | 11655 | 61.9 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 189 | 392.9 | 3255 | 1820.1 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 16 | 364.8 | 2685 | 2581.0 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 1 | 355.2 | 2430 | 2735.3 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000613 | 339.6 | 2113 | 1978.1 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 85 | 260.5 | 1848 | 746.8 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000551 | 93.6 | 166 | -1484.3 | med | C_gatefail | 1 near bundle(s), 1 precull |

### evt 298805 (5 missed)
| uid | len_cm | npts | t_us | conf | class | detail |
|---|---|---|---|---|---|---|
| 4000033 | 390.7 | 2763 | 2841.2 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000120 | 234.8 | 1765 | 4365.0 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000214 | 217.6 | 1385 | -983.2 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000286 | 191.4 | 5374 | 1119.1 | med | C_gatefail | 1 near bundle(s), 1 precull |
| 4000263 | 116.2 | 652 | 985.0 | med | C_gatefail | 1 near bundle(s), 1 precull |

