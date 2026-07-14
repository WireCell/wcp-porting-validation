# 1 – `sbn::BNBSpillInfo` StreamerInfo version/checksum mismatch

## Symptom

Running `eventdump.fcl` (or any `lar` job) over a decoded data file fails at
**file open** with a fatal ROOT error, then segfaults:

```
$ lar -n 1 -c eventdump.fcl -s .../decoded-raw_filtered_data_..._eventidfiltered.root
---- FileOpenError BEGIN
  ---- FatalRootError BEGIN
    Fatal Root Error: TStreamerInfo::BuildCheck
       The StreamerInfo of class sbn::BNBSpillInfo read from file ...
       has the same version (=15) as the active class but a different checksum.
       You should update the version to ClassDef(sbn::BNBSpillInfo,16).
    ROOT severity: 2000
  ---- FatalRootError END
---- FileOpenError END
Segmentation fault (core dumped)
```

The `FileOpenError` / "was not found or could not be opened" wording is
misleading — the file is present and readable (`33M`); the failure is a class
schema conflict, not a missing/corrupt file.

## Root cause

ROOT class-schema (version + checksum) mismatch:

1. The file on disk holds `sbn::BNBSpillInfo` written with member layout **A**,
   stamped `ClassDef(sbn::BNBSpillInfo, 15)`.
2. The **active** sbndcode (`v10_14_02_03`, the release set up at the time) has a
   **different** layout **B** for the same class — but *also* still labeled
   version **15**.
3. ROOT keys deserialization on the `ClassDef` version number. Same version +
   different checksum → it cannot safely stream either layout → fatal error
   (severity 2000) → art turns it into `FatalRootError` → segfault on teardown.

Underlying bug: a **schema-evolution mistake in `sbnobj`** — members of
`BNBSpillInfo` were changed without bumping the `ClassDef` version (15 → 16), so
"version 15" now means two incompatible things depending on the build. ROOT's
own advice in the message ("update to `ClassDef(...,16)`") points at this.

## Which release wrote the file

Read directly from the file's embedded FileCatalogMetadata (no ROOT/UPS needed):

```bash
strings -n 6 "$FILE" | grep applicationVersion | head
# applicationFamily:"art" applicationVersion:"v10_06_00" fileType:"data" group:"sbnd" ...
```

- `applicationVersion` is the sbndcode release SBND stamps into every output.
- **`v10_06_00`** is the *only* sbndcode version present — every process in the
  chain (`DECODE` → `SBNDBNBInfoGen` → `FilterEventID`) ran under it.

## Fix — use the release that wrote the file

Set up **sbndcode `v10_06_00`** instead of `v10_14_02_03`:

```bash
# use a CLEAN login shell — NOT the (venv)/(wct-opt) shell, which breaks ups bootstrap
source /cvmfs/sbnd.opensciencegrid.org/products/sbnd/setup_sbnd.sh
setup sbndcode v10_06_00 -q e26:prof
lar -n 1 -c eventdump.fcl -s "$FILE"
```

Notes:
- Metadata records the base tag `v10_06_00`. Patch tags `v10_06_00_01 … _10`
  exist on CVMFS but almost never touch `sbnobj`/`BNBSpillInfo`, so the base tag
  should give the matching checksum. If it still complains, try the patch tags.
- Setting up sbndcode inside the `(venv)(wct-opt)` shell fails with
  `Product 'ups' has no current chain` — that environment pollutes the ups
  bootstrap. Use a fresh shell.

## Alternative — just list run/subrun/event IDs

If you only need event IDs (not a full product dump), skip release-matching
entirely: read `EventAuxiliary` via PyROOT. That path never builds the
`BNBSpillInfo` StreamerInfo, so the checksum conflict never triggers.

## Diagnostic tricks used

- `strings -n 6 FILE | grep -oE 'v10_[0-9_]+'` — surface embedded release tags.
- `strings -n 6 FILE | grep applicationVersion` — authoritative producing release.
- `strings -n 4 FILE | grep process_name` — reconstruct the full process chain.
