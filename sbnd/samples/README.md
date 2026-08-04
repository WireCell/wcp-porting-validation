# sbnd/samples — moved to the data area

The nue-CC (Lynn's) 2025-fall-production sample — both the bulk data and the
recipe (fcl / scripts / RSE & file lists / docs) — was moved on **2026-08-04**
to keep all the sample data in one place:

**`/exp/sbnd/data/users/yuhw/2025-fall-prod-sample/nuecc-lynn/`**

Contents there:
- `filter-nuecc-rse.fcl`, `find_reco1_files.sh` — the RSE→reco1 selection recipe
- `lynn-nuecc.lst`, `lynn-nuecc-rse.csv`, `lynn-nuecc-reco1-files.lst`,
  `lynn-nuecc-reco1-files.map.txt`, `lynn-nuecc-reco1-missing-rse.lst` — lists/maps
- `docs/` — `1-bnbspillinfo-version-mismatch.md`,
  `2-rse-to-reco1-and-filtereventid-port.md`,
  `3-qlmatch-idetectorvolumes-build-issue.md`, `gen2-data-frameshift.md`
- `filtered-reco1/`, `nuecc-bee/`, `pos-offset-study/` — the produced data (~562 MB)

This file is only a pointer; the recipe lives with the data at the path above.
