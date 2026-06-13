# PDHD Q/L matching hand-scan event display

An interactive tool to examine PDHD charge–light (Q/L) matching one event at a
time, override the matcher's auto-selection by hand, and save the human-picked
flash↔cluster matches as labels for later parameter tuning. Faithful port of the
SBND tool (`sbnd_xin/ql_scan/`, `sbnd_xin/docs/ql-scan-display.md`); this page
documents only what differs for ProtoDUNE-HD.

> **Status:** machinery only. The PDHD light reconstruction and Q/L matching chain
> is still being tuned, so this is not yet ready for production hand-scanning — the
> viewer, dump plumbing, and run flag are in place so the workflow exists once the
> chain settles.

It has two parts:

1. a **`-calib` dump** in `QLMatching` that writes, per event, the full candidate
   bundle universe to a JSON file;
2. a **Bokeh server viewer** (`ql_scan/`) that renders the bundles, lets you pick
   the correct matches under the physical selection rules, and writes a labels JSON.

## What differs from SBND: two drift sides, two files

SBND runs one **joint** QLMatching node (both TPCs → one calib JSON). PDHD images
through two drift volumes either side of a central cathode, and wires **one
QLMatching node per drift side** (`pdhd/wct-clustering.jsonnet`):

| group | APAs | imaging face | drift | merged side |
|-------|------|--------------|-------|-------------|
| `group02` | 0, 2 | 0 | −x | **0** |
| `group13` | 1, 3 | 1 | +x | **1** |

Each node writes **its own** calib JSON, so an event has up to **two** per-group
files. The central cathode is **opaque to 128 nm VUV**, so the two drift sides are
physically independent — a flash on one side never sees charge on the other, and
there is no cross-side light coincidence. Today usually only one drift side carries
a light signal, so usually only **one** file is produced.

The viewer **merges the 1–2 per-group files of an event** into one two-panel
(side 0 / side 1) view: it groups files by event id (stripping the `-group02` /
`-group13` suffix), maps `group02`→side 0 and `group13`→side 1, tags every
flash/cluster/bundle with its side, id-offsets them so the two files never collide,
unions the per-side detector boxes, and ORs the OpDet active masks. A missing side
simply leaves that side's panels **blank**.

---

## 1. Produce the calib dumps

The dump is emitted by the clustering chain (`run_clus_evt.sh`) with `-calib`:

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/pdhd
./run_clus_evt.sh -calib <run> <evt|all>
# -> work/<run6>_<evt>/calib-evt<ID>-group02.json
#    work/<run6>_<evt>/calib-evt<ID>-group13.json   (only for populated sides)
```

`-calib` implies Q/L matching (it forces `do_qlmatch`). The flag is **off by
default**; with it off the matched `mabc-*.zip` is byte-for-byte identical (the
QLMatching node's `calib_dump` is the empty string, so the dump method is never
called). The dump is observation-only — it reads the matcher's finished state and
never perturbs the matching.

### Calib JSON schema

Identical to SBND (see `sbnd_xin/docs/ql-scan-display.md` §1 for the full table),
with these PDHD specifics:

| key | PDHD value / meaning |
|-----|----------------------|
| `nchan` | optical-channel count (**160**, all flat X-ARAPUCAs — no PMTs) |
| `drift_speed` | cm/µs; a bundle's T0 x-shift is `dx = sign_offset · flash_time_us · drift_speed` |
| `geometry[apa]` | fixed detector box per drift volume: `anode_x`, `cathode_x` (central cathode ≈ 0), `sign_offset`, `y_lo/y_hi`, `z_lo/z_hi` |
| `opdets[]` | per channel `{ch, x, y, z, type, apa, active}`; `type` 0 = X-ARAPUCA |
| `flashes[]`, `clusters[]`, `bundles[]` | as SBND; per-file only one drift side is populated |

Each per-group file holds only its own drift side, so within a file `apa` is the
side's representative APA ident. After the viewer merges the two files, the in-app
`apa` field is the **drift side (0 or 1)**, and the flash gid / group ids shown in
the UI are the per-side dump values (the merge offset is stripped for display).

---

## 2. Launch the viewer

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/pdhd
# default port 5015 (img_plot owns 5013, pd_plot 5014)
./ql_scan/serve_ql_scan.sh 5015 --tag data work/*/calib-evt*-*.json
```

From a laptop, forward the port and open the app:

```bash
ssh -L 5015:localhost:5015 user@wcgpu1
# browser: http://localhost:5015/ql_scan_viewer
```

### Layout

The layout is the SBND one, relabelled for PDHD — "TPC0 / TPC1" read **side 0 /
side 1**, and "PMT" reads **OpDet** (the X-ARAPUCAs). The three charge-projection
views are at the top; below them follow the controls, the bundle table + selection
checkboxes + metrics + cluster roster, the compare-cluster table, the 2×2
measured/predicted **light patterns** (side 0 + side 1, OpDet circles ∝ √PE on the
Y–Z plane), and the per-side 1-D meas-vs-pred overlays + pred/meas ratio. See the
SBND doc for the per-panel details — behaviour is identical.

The ±80 ns coincidence-group machinery is inherited from SBND. Because PDHD's
cathode is opaque, groups are single-side in practice (a group's side-1 flash time
shows "-" when only side 0 has light).

### Selection rules, persistence, save

Identical to SBND:

- **one flash per cluster** (the **Filter selected bundles** toggle, default ON,
  locks 🔒 a cluster already matched elsewhere); **many clusters per flash** (ticked
  bundles sharing a flash sum their predicted light).
- Picks **autosave** to `work/ql_labels/<tag>/.scan_state-evt<ID>.json` and restore
  on load; **Save labels** writes `work/ql_labels/<tag>/labels-evt<ID>.json`.
- `work/ql_labels/` is a sibling of the per-event `work/<run6>_<evt>/` workspace, so
  reprocessing an event cannot delete saved scan results. `--tag` subdirs separate
  displays' labels.

---

## Notes / scope

- PDHD only (two drift sides about a central opaque cathode, `semi-analytical-pdhd`
  geometry, 160 X-ARAPUCAs). The VUV efficiency / `QtoL` in the matching config are
  still placeholders pending PDHD light calibration.
- The two drift sides are independent matched volumes; the coincidence group is only
  a scan-navigation unit, never a merge (predicted light sums only within one
  flash/side).
- The labels are the deliverable; fitting updated χ²/threshold/`PE_err` parameters
  from them is downstream manual work, not part of this tool.
