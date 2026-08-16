# doc pr/82 -- the 24 events the label carry-forward declined

Served on **port 5017**, scan tag `vtxscan-harv3-delta`, worst first.

Your old answer (`old truth`) was taken on the `-prod0813` arm at
`min_accept=4`. On the current arm (toolkit `771f075b`, pr83/85/86 on,
`min_accept=10`) the nearest vertex to that point is further than the 1 cm
tolerance, so nothing was carried automatically. What is being asked is
simply: **where is the vertex now?**

`prod` = distance from your old answer to what production currently picks;
a small `prod` means production already agrees with your old answer and the
carry-forward was merely conservative.

| # | sample | event | reanchor | old vid | nearest vid | prod | route | old truth (x,y,z) |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | mcp1k | 320865 | **16.56 cm** | None | 46014 | 39.52 | dl-rerank-reject | 43.0, 18.0, 75.0 |
| 2 | mcp1k | 347129 | **10.72 cm** | 11000 | 11000 | 62.36 | dl-rerank-reject | -194.0, -102.9, 201.7 |
| 3 | mcp1k | 172832 | **9.77 cm** | None | 81033 | 20.70 | dl-rerank-reject | -75.1, -74.3, 142.5 |
| 4 | mcp1k | 291570 | **5.47 cm** | 17001 | 17001 | 5.47 | dl-rerank-reject | -76.9, 1.2, 141.1 |
| 5 | mcp1k | 61681 | **5.14 cm** | None | 2002 | 5.14 | dl-rerank-accept | -161.0, -116.0, 200.0 |
| 6 | nueCC48 | 122660 | **4.95 cm** | 9010 | 9013 | 4.95 | dl-rerank-accept | -99.4, -76.2, 390.9 |
| 7 | mcp1k | 61579 | **4.05 cm** | 20003 | 20051 | 4.05 | dl-rerank-reject | 20.6, -119.3, 56.0 |
| 8 | mcp1k | 278046 | **4.01 cm** | 3010 | 51020 | 4.70 | dl-rerank-accept | -102.6, -172.7, 238.7 |
| 9 | mcp1k | 59899 | **3.40 cm** | 5003 | 5004 | 3.40 | dl-rerank-reject | -194.6, -9.1, 14.6 |
| 10 | mcp1k | 406796 | **2.48 cm** | 11001 | 11001 | 2.48 | dl-rerank-reject | -53.9, -154.1, 85.7 |
| 11 | mcp1k | 286353 | **2.48 cm** | 9002 | 9012 | 2.48 | dl-rerank-reject | -63.5, 43.2, 373.0 |
| 12 | nueCC48 | 268067 | **2.27 cm** | 15004 | 15079 | 2.27 | dl-rerank-accept | -80.5, -64.9, 280.7 |
| 13 | mcp1k | 390842 | **2.02 cm** | 3004 | 3001 | 2.14 | dl-rerank-accept | -76.9, -113.0, 9.5 |
| 14 | mcp1k | 72586 | **1.89 cm** | 17001 | 17001 | 2.61 | dl-veto-protected | -61.6, -197.0, 53.8 |
| 15 | mcp1k | 284637 | **1.85 cm** | 23002 | 23002 | 1.85 | dl-rerank-reject | -198.0, -93.2, 435.4 |
| 16 | nueCC48 | 423981 | **1.48 cm** | 12003 | 12099 | 1.48 | dl-rerank-accept | 48.8, 141.3, 176.3 |
| 17 | mcp1k | 166870 | **1.45 cm** | 10010 | 10060 | 4.11 | dl-rerank-accept | -141.1, 145.8, 140.4 |
| 18 | mcp1k | 283040 | **1.34 cm** | 2000 | 2001 | 2.04 | dl-rerank-accept | -5.9, 105.2, 166.2 |
| 19 | mcp1k | 52085 | **1.29 cm** | 6002 | 6002 | 1.29 | dl-rerank-reject | -1.4, 112.3, 81.8 |
| 20 | nueCC48 | 138009 | **1.23 cm** | 12002 | 12117 | 1.23 | dl-rerank-accept | -103.6, 161.0, 82.0 |
| 21 | nueCC48 | 400474 | **1.17 cm** | 18003 | 18156 | 1.17 | dl-rerank-accept | 60.5, -117.6, 259.7 |
| 22 | mcp1k | 409546 | **1.09 cm** | 41016 | 41016 | 2.50 | dl-rerank-accept | -189.5, 36.6, 483.7 |
| 23 | mcp1k | 349945 | **1.06 cm** | 18011 | 18003 | 1.06 | dl-rerank-accept | -154.7, -100.1, 102.8 |
| 24 | mcp1k | 283091 | **1.05 cm** | 24002 | 24010 | 1.05 | dl-rerank-reject | 94.5, -93.2, 44.5 |

24 events: 9 beyond 3 cm, 15 between 1 and 3 cm. NCpi0 contributes none.
