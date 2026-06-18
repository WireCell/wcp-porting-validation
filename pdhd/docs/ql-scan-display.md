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
| `drift_speed` | cm/µs; a bundle's T0 x-shift is `dx = sign_offset · flash_time_us · drift_speed`. `flash_time_us` **already includes** the per-event readout-vs-trigger offset (folded in by QLMatching), so the charge lands on the raw-`x` reference the dump uses; the top-level `trigger_offset` is `0` and must **not** be re-added |
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

### Drift side is the *lit* side, not the file side

A flash's drift side is taken from **where its measured light is** — the OpDets that
carry its PE — never from which per-group file referenced it. PDHD's cathode is
opaque to 128 nm VUV, so a flash lights OpDets on exactly **one** volume: light on
the **−x** array (APAs 0,2) ⇒ **side 0**, on the **+x** array (APAs 1,3) ⇒ **side 1**.
The per-side matcher runs against one global flash list, so the *same* flash is
referenced by clusters on both sides; using the file/node side would put a flash on
the wrong panel and double-count it. So a flash whose light is on the +x OpDets is
shown on the **S1** panel even though it also appears in the `group02` file, and its
−x copy (zero measured PE on side 0) is dropped as an empty phantom.

This is the same rule the Bee **op** display now uses. The optical-flash dump in
`MultiAlgBlobClustering` previously labelled every flash `apa=0` (it derived the side
from the flash *gid*, which encodes the processing node's anode, not the lit volume).
`QLMatching::write_opflash_pc` now emits a per-flash physical side from the PE pattern
(`m_opdets[ch].center.x()` vs the cathode), carried through the per-APA→all-APA merge
and read back by `fill_bee_flashes`, so Bee tags each flash **TPC0 (−x, APAs 0,2)** or
**TPC1 (+x, APAs 1,3)** consistently with this viewer. (Older dumps without the `apa`
column fall back to the legacy gid encoding.)

### Phantom flash collapse — keeps the two tables consistent

Each per-side dump carries the **full global flash list**, so after the merge every
physical flash appears **twice**: a *canonical* copy (file side == lit side) and a
*phantom* copy (file side != lit side — the one referenced by the **other** drift
side's clusters). A cross-side bundle (a cluster matched to a flash on the opposite
volume) points at a phantom, whose coincidence group is never navigable (it carries no
measured light on its file side). Left alone, a kept **cross-side cathode-crosser** bundle
(the `cross_side_filter` survivor — see [`qlmatching-chain.md`](qlmatching-chain.md)) would
appear in the per-cluster **Compare** table yet be **unreachable** in the **navigation**
table — the exact inconsistency this collapse fixes.

On load the viewer **re-points every bundle to the canonical flash copy** — matched by
`(lit side, time, total PE)`, unique per physical flash — and **drops the phantoms**. A
cross-side crosser bundle then lands in its flash's real, navigable group, so it shows in
**both** tables. This also subsumes the older single-sided-run dark-duplicate skip (the
dark copy *is* the phantom). The navigation table still applies the cluster-**length**
cut, so short fragments stay hidden there (Compare lists them regardless, as it does for
every candidate flash of a focused cluster).

### Group order follows flash time

Coincidence groups are **ranked by ascending flash time**, so paging the scan walks the
flashes earliest-first and the group order matches Bee's time-ordered flash list. The
cross-side pairing numbers every S0-paired group before the appended S1-only ones, so
without this an early-time S1-only flash would land on a late group id (e.g. a −1452 µs
flash showing up as "grp 66"); the viewer remaps ids to time order after pairing
(paired S0+S1 flashes keep their shared id).

### Selection rules, persistence, save

Identical to SBND:

- **one flash per cluster** (the **Filter selected bundles** toggle, default ON,
  **hides** every other bundle reusing a cluster already matched elsewhere — from the
  bundle table *and* the group navigation, so an all-reused group drops out of the
  prev/next + dropdown, mirroring the length filter); **many clusters per flash**
  (ticked bundles sharing a flash sum their predicted light).
- The per-cluster **Compare** table is **pinned**: it shows the cluster from the last
  *Compare cluster's flashes* click and does not follow main-table focus changes (so it
  no longer jumps when a different-cluster bundle is clicked); re-click Compare to move
  it.
- **Cross-side bundles** (cluster on one drift side, flash measured on the other — the
  `cross_side_filter` cathode-crosser survivors) are tinted light orange and tagged
  `XSIDE` in the flags column, marking them as low-priority context rather than primary
  scan targets. (Table label only — not written to the saved labels JSON.)
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
