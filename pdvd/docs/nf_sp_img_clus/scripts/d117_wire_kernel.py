#!/usr/bin/env python3
"""doc pdvd/117 -- real-space kernel of the SP collection wire filter, training vs data,
plus the linear-inversion budget quoted in the doc.

The kernel is the inverse DFT of exactly the array HfFilter::filter_waveform builds
(sigproc/src/HfFilter.cxx:40-60, use_negative_freqs default true at HfFilter.h:28;
freq in units of Nyquist, max_freq 1; util/src/Response.cxx hf = exp(-0.5 (f/sigma)^2)).
Same method as d42_wire_filter_toy.py Part A.
"""
import numpy as np

N, PITCH_MM = 480, 5.100
print("Wire_col real-space kernel (sigma = X/sqrt(pi), power 2):")
for label, X in (("training (dune-vd/sp-filters.jsonnet:114)", 3.0),
                 ("data PDVD/PDHD (protodunevd :113-114, pdhd :84)", 10.0)):
    s = X / np.sqrt(np.pi)
    f = np.arange(N) / N * 2.0
    f = np.where(f > 1, f - 2, f)
    h = np.real(np.fft.ifft(np.exp(-0.5 * (np.abs(f) / s) ** 2)))
    h /= h.sum()
    k = np.where(np.arange(N) > N // 2, np.arange(N) - N, np.arange(N))
    rms = np.sqrt((h * k ** 2).sum())
    print(f"  X={X:4.1f} {label}: h0={h[0]:.3f} h+-1={h[1]:.3f} h+-2={h[2]:+.4f} "
          f"rms={rms:.3f} wire = {rms * PITCH_MM:.2f} mm")

print("\nLinear-inversion slope budget  k = 2 D / v (wire axis),  2 D_L / v^3 (time axis):")
v_tr, v_pd = 1.60563, 1.48073          # mm/us: training sim, PDVD production .tlas
DT_tr = 8.8                            # cm2/s configured in training
for frac in (0.70, 1.00):              # assumed SP-returned fraction in the training chain
    ktr = frac * DT_tr / v_tr
    for lab, DT in (("PDVD W all", 4.83), ("top CRP", 6.14), ("bottom CRP", 3.16)):
        print(f"  training returns {frac:.0%}: {lab:11s} D_T,eff {DT:4.2f} -> k_data/k_train = "
              f"{DT / v_pd / ktr:.2f}")
print(f"  time axis, params D_L 4.1307 vs 4.0: (4.1307/4.0)*(v_tr/v_pd)^3 = "
      f"{4.1307 / 4.0 * (v_tr / v_pd) ** 3:.2f}")
