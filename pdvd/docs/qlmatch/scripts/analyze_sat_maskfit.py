#!/usr/bin/env python3
"""Effect of QLMatching saturation_mask_fit=false (keep railed channels in chi2/KS).

Repro:
  python3 analyze_sat.py <baseline_dump.json> <keepsat_dump.json>
where baseline = same chain with saturation_mask_fit at the legacy drop.
"""
import json, sys
import numpy as np

base = json.load(open(sys.argv[1]))
new  = json.load(open(sys.argv[2]))

qp = new['quality_params']
print("=== quality params (new dump) ===")
for k in ('pe_err_on_pred', 'pe_err_floor', 'pe_err_frac', 'pe_err_lowpe_frac',
          'pe_err_lowpe_knee', 'chi2_sat_inflate', 'saturation_mask_fit', 'mask_ks'):
    if k in qp:
        print(f"  {k:22s} {qp[k]}")

# ---- flash-level: how many channels carry the rail flag at all? ----
fl = {f['id']: f for f in new['flashes']}
nsat_tot = sum(int(np.sum(f['sat'])) for f in new['flashes'] if f.get('sat'))
nflash_sat = sum(1 for f in new['flashes'] if f.get('sat') and np.sum(f['sat']) > 0)
print(f"\n=== rail flags in the event ===")
print(f"  flashes                 {len(new['flashes'])}")
print(f"  flashes with >=1 railed {nflash_sat}")
print(f"  railed (flash,chan) cells {nsat_tot}")

# The dump does not carry the low-PE inflation params; take the PDVD
# qlmatching.jsonnet values (pe_err_lowpe_frac 2.0 / pe_err_lowpe_knee 10.0).
LOWPE_FRAC = qp.get('pe_err_lowpe_frac', 2.0)
LOWPE_KNEE = qp.get('pe_err_lowpe_knee', 10.0)


def perr_on_pred(pred):
    """TimingTPCBundle::per_opdet_perr, pe_err_on_pred branch (PDVD settings)."""
    rel = qp['pe_err_frac'] + (LOWPE_FRAC - qp['pe_err_frac']) * np.exp(-pred / LOWPE_KNEE)
    return np.sqrt((rel * pred) ** 2 + qp['pe_err_floor'] ** 2)

# ---- what do railed channels actually contribute to chi2 now? ----
rows = []
for b in new['bundles']:
    f = fl.get(b['flash_id'])
    if not f or not f.get('sat'):
        continue
    sat = np.asarray(f['sat'])
    if sat.sum() == 0:
        continue
    pe = np.asarray(f['pe'], float)
    pred = np.asarray(b['pred_pe'], float)
    for j in np.nonzero(sat)[0]:
        if pred[j] <= 0 and pe[j] <= 0:
            continue
        perr = perr_on_pred(pred[j])
        denom = pe[j] + perr ** 2
        rows.append((pe[j], pred[j], (pred[j] - pe[j]) ** 2 / denom))

if rows:
    a = np.array(rows)
    meas, pred, chi2 = a[:, 0], a[:, 1], a[:, 2]
    print(f"\n=== railed (bundle,chan) terms now entering the chi2: n={len(a)} ===")
    print(f"  measured PE   median {np.median(meas):8.1f}  p90 {np.percentile(meas,90):8.1f}")
    print(f"  predicted PE  median {np.median(pred):8.1f}  p90 {np.percentile(pred,90):8.1f}")
    over = np.sum(pred > meas) / len(a) * 100
    print(f"  pred > meas (the clipped direction): {over:.1f}%")
    print(f"  per-channel chi2 contribution:")
    for q in (50, 90, 99):
        print(f"    p{q:<3d} {np.percentile(chi2,q):8.2f}")
    print(f"    max   {chi2.max():8.2f}   mean {chi2.mean():6.2f}")
    print(f"  effective sigma/pred at median pred: "
          f"{perr_on_pred(np.median(pred))/max(np.median(pred),1e-9):.2f}")

# ---- bundle-level before/after ----
def key(b):
    return (b['flash_id'], b['main_cluster'])

bb = {key(b): b for b in base['bundles']}
nn = {key(b): b for b in new['bundles']}
common = sorted(set(bb) & set(nn))
print(f"\n=== bundles: base {len(bb)}  new {len(nn)}  common {len(common)} ===")

dndf = np.array([nn[k]['ndf'] - bb[k]['ndf'] for k in common])
print(f"  ndf  median {np.median([bb[k]['ndf'] for k in common]):.0f} -> "
      f"{np.median([nn[k]['ndf'] for k in common]):.0f}   "
      f"(changed on {np.sum(dndf!=0)} bundles, mean d={dndf.mean():+.2f})")

def c2n(b):
    return b['chi2'] / b['ndf'] if b['ndf'] > 0 else np.nan

cb = np.array([c2n(bb[k]) for k in common], float)
cn = np.array([c2n(nn[k]) for k in common], float)
ok = ~(np.isnan(cb) | np.isnan(cn))
print(f"  chi2/ndf median {np.nanmedian(cb[ok]):.2f} -> {np.nanmedian(cn[ok]):.2f}")

for fld in ('consistent', 'contained'):
    b0 = sum(1 for k in common if bb[k].get(fld))
    n0 = sum(1 for k in common if nn[k].get(fld))
    print(f"  {fld:11s} {b0} -> {n0}  ({n0-b0:+d})")

# ---- matching decisions ----
def sel(d):
    return {b['main_cluster']: b['flash_id'] for b in d['bundles'] if b.get('auto_selected')}

sb, sn = sel(base), sel(new)
lost = sorted(set(sb) - set(sn))
gain = sorted(set(sn) - set(sb))
moved = sorted(c for c in set(sb) & set(sn) if sb[c] != sn[c])
print(f"\n=== auto-selected matches ===")
print(f"  base {len(sb)}  new {len(sn)}   lost {len(lost)}  gained {len(gain)}  moved {len(moved)}")
cl = {c['uid']: c for c in new['clusters']}
for c in (lost + gain + moved)[:12]:
    info = cl.get(c, {})
    tag = 'LOST' if c in lost else ('GAINED' if c in gain else 'MOVED')
    print(f"    {tag:6s} uid {c} npoints {info.get('npoints','?')} "
          f"flash {sb.get(c,'-')} -> {sn.get(c,'-')}")
