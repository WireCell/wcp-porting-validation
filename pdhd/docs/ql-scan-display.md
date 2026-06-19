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

## Two drift sides in one joint node, one combined file

Like SBND, PDHD now runs **one joint** QLMatching node taking **both** drift volumes
(`matching_joint`, `pdhd/wct-clustering.jsonnet`) and writes a **single** calib JSON per
event, with both sides tagged by `apa`:

| `apa` | APAs | imaging face | drift | side |
|-------|------|--------------|-------|------|
| `0` | 0, 2 | 0 | −x | **0** |
| `1` | 1, 3 | 1 | +x | **1** |

The central cathode is **opaque to 128 nm VUV**, so each drift side is matched
**independently**; the joint node adds only a cross-cathode (xTPC) consistency pass that
pairs a cathode-crosser's two halves (see [`qlmatching-chain.md`](qlmatching-chain.md)).

The viewer renders the one file as a two-panel (side 0 / side 1) view: it reads each
flash/cluster/bundle's own `apa` field (the node already tags it and makes gid/uid
globally unique via the `apa·10⁶` stride), takes each side's detector box from the
`geometry["0"]`/`geometry["1"]` entries, and ORs the OpDet active masks. A side with no
light leaves its panels **blank**. (A legacy pair of per-side
`calib-evt<ID>-group02/group13.json` files still loads — each carries one distinct
`apa` — so old dumps keep working.)

---

## 1. Produce the calib dumps

The dump is emitted by the clustering chain (`run_clus_evt.sh`) with `-calib`:

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/pdhd
./run_clus_evt.sh -calib <run> <evt|all>
# -> work/<run6>_<evt>/calib-evt<ID>.json   (one combined file, both drift sides)
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
| `flashes[]`, `clusters[]`, `bundles[]` | as SBND; both drift sides populated, each entry tagged by `apa` (0/1) |

The combined file carries both drift sides; each flash/cluster/bundle's `apa` field is
its **drift side (0 or 1)** and its `gid`/`uid` is globally unique via the `apa·10⁶`
stride (so the viewer reads them as-is, no per-side offset).

---

## 2. Launch the viewer

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/pdhd
# default port 5015 (img_plot owns 5013, pd_plot 5014)
./ql_scan/serve_ql_scan.sh 5015 --tag data work/*/calib-evt*.json
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

#### Bee op flash de-duplication (`opflash_phys_gid`)

PDHD's all-PD light reconstruction emits **one global** `opflash_pdhd-wct.tar.gz`,
and the joint `QLMatching` node feeds **both** drift-side sub-runs from it (the same
archive on each input port, `wct-clustering.jsonnet`). Each side-run's
`write_opflash_pc` therefore dumps the **full** flash list keyed by its own anode
ident, so after the per-APA→all-APA merge **every physical flash lands on the root
twice** and the Bee **op** display drew each flash as two overlapping red boxes. This
is the structural counterpart to the viewer's phantom collapse, but on the Bee side.
(SBND differs only here: its two side-runs read **per-TPC** `opflash_apa{0,1}`
archives, so each flash is dumped once — no duplication — which is why the SBND op
display already shows one box per flash.)

The fix is the `opflash_phys_gid` knob (`QLMatching`, **default off** = legacy
node-ident gid, byte-identical for SBND / single-source-per-node configs; **PDHD on**).
With it on, the Bee flash gid — and the cluster's `matched_flash_gid` — is keyed by the
flash's **physical side** (`flash_phys_side`, the same PE-pattern rule as the `apa`
tag) instead of the processing node's anode ident. Both side-runs then emit the **same
gid** for one physical flash, so `fill_bee_flashes` collapses the duplicate, **and** a
cross-side (xTPC) `matched_flash_gid` still resolves to the surviving flash. Verified
on run 29107 evt 983: op rows **278 → 141** (138 distinct flashes; the residual rows
are legitimate one-flash-multiple-cluster matches), matched-cluster set **identical**
(36 → 36, **0 lost / 0 gained**) — the dedup does not touch which clusters are matched.
The calib-dump gid path (`dump_calib`) is **unchanged** (still node-ident encoded), so
this viewer's phantom collapse and saved scans are unaffected.

### Phantom flash collapse — keeps the two tables consistent

Each drift side's run dumps the **full global flash list**, so the combined file lists
every physical flash **twice**: a *canonical* copy (file/run side == lit side) and a
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
- **All tables are non-sortable** (`sortable=False` on every column). A row click reports
  a row *index*, which the callbacks map through a build-order list (`clus_order` for the
  roster, `compare_order` / `order` for the bundle tables) back to the bundle/cluster.
  User-sorting a column reorders only the on-screen view, **not** that list, so the index
  would resolve to the wrong row — e.g. sorting the cluster roster by `clus` to find
  cluster 90, then clicking it, would actually pick the neighbouring (often cross-side,
  other-drift-side) cluster, and *Compare* would then list **that** cluster's flashes
  (symptom: the Compare table fills with a different `clus`, on the opposite side).
  Disabling sort keeps the index ↔ row mapping exact; navigate via the fixed
  longest-cluster-first order.
- **Cross-side bundles** (cluster on one drift side, flash measured on the other — the
  `cross_side_filter` cathode-crosser survivors) are tinted light orange and tagged
  `XSIDE` in the flags column, marking them as low-priority context rather than primary
  scan targets. (Table label only — not written to the saved labels JSON.)
- Picks **autosave** to `work/ql_labels/<tag>/.scan_state-evt<ID>.json` and restore
  on load; **Save labels** writes `work/ql_labels/<tag>/labels-evt<ID>.json`.
- `work/ql_labels/` is a sibling of the per-event `work/<run6>_<evt>/` workspace, so
  reprocessing an event cannot delete saved scan results. `--tag` subdirs separate
  displays' labels.

> **Migrating pre-joint-node scans.** Scans saved by the *old* two-file viewer keyed
> their `.scan_state` / `labels` ids on a per-side `SIDE_OFF = 1e9` merge offset the joint
> viewer no longer adds, so they would not restore against the new combined dump. Convert
> them once with `ql_scan/convert_scan_prejoint.py` (strips the offset, `v % 1e9`); it
> preserves the originals as `<name>.prejoint` and is idempotent. Verified on run 29107 evt
> 983: all 31 saved picks restore, including the cluster-70 cross-side cathode-crosser.
>
> **Re-keying a scan after REPROCESSING.** `flash_gid` is a positional enumeration index
> (`anode·stride + index-in-run.flashes`), so any reprocess that changes which flashes pass
> `flash_minPE` — e.g. the `measured_pe_scale` optical retune, which raises APA0 `total_PE`
> — renumbers the flashes and **breaks the saved gids** (the clusters are unchanged; only
> the flash index moves). Re-key the saved `.scan_state` to the fresh dump with
> `ql_scan/remap_scan_after_reprocess.py <scan_state> <labels> <new_calib.json>`: it matches
> each pick by the stable `(main_cluster, flash_time)` (old gid→time recovered from the
> labels file) and emits the **collapsed** gid the viewer keys on — it reproduces
> `Event.__init__`'s phantom-flash collapse so cross-side picks re-key too. Preserves the
> original as `<name>.prereprocess`, idempotent. Verified on evt 983 after the optical
> retune: all 31 gids shifted, all 31 re-key and restore via the viewer's `load_state`.

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
