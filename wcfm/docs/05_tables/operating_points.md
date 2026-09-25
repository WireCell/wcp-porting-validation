| decision rule | precision (kept cells that are real) | recall (real cells kept) | iso recall | cosmic recall |
|---|---|---|---|---|
| legacy chain (charge solving + deghosting) | 0.684 | 0.061 | 0.045 | 0.205 |
| charge GNN (3-seed mean), P(real) > 0.5 (t = 0.500) | 0.920 | 0.971 | 0.970 | 0.983 |
| charge GNN (3-seed mean), at the legacy precision 0.684 (t = 0.003) | 0.684 | 1.000 | 1.000 | 1.000 |
| charge GNN (3-seed mean), at recall 0.90 (t = 0.820) | 0.962 | 0.900 | 0.896 | 0.933 |
| charge GNN (3-seed mean), at recall 0.95 (t = 0.650) | 0.940 | 0.950 | 0.948 | 0.969 |
| fm GNN (3-seed mean), P(real) > 0.5 (t = 0.500) | 0.954 | 0.963 | 0.963 | 0.968 |
| fm GNN (3-seed mean), at the legacy precision 0.684 (t = 0.001) | 0.684 | 1.000 | 1.000 | 1.000 |
| fm GNN (3-seed mean), at recall 0.90 (t = 0.864) | 0.983 | 0.900 | 0.899 | 0.908 |
| fm GNN (3-seed mean), at recall 0.95 (t = 0.616) | 0.964 | 0.950 | 0.950 | 0.954 |
