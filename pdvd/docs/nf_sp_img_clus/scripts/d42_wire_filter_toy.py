#!/usr/bin/env python3
"""doc pdvd/42 sec 8 -- toy study of the SP wire filter and the track-fit
transverse smearing model.  Answers the owner's question (2026-09-05): is
`ind_sigma_*_T` mis-derived, or is the derivation's approximation breaking down
at PDVD's 7.65 mm pitch?

Everything here is reproduced from the toolkit source, not from a fit:

  cfg/.../sp-filters.jsonnet   wf(name, {sigma: 1/sqrt(pi) * X, power: 2, flag: false,
                                        max_freq: 1})
  sigproc/src/HfFilter.cxx     filter_waveform(N): freq = i/N * 2 * max_freq, wrapped
                               to [-1,1) -- so freq is in units of NYQUIST
  util/src/Response.cxx:441    hf_filter = exp(-0.5 * (freq/sigma)^power)
  clus/src/TrackFitting.cxx:7310  sigma_T_u = hypot(sqrt(2*DT*t_drift), ind_sigma_u_T)/pitch_u
  clus/src/TrackFitting.cxx:6272  the model integrates that Gaussian over each wire bin (erf)

Part A  the filter's true real-space kernel, by inverse DFT of exactly the array
        HfFilter::filter_waveform builds -- against the closed form the configs use.
Part B  a forward toy: point deposit at a uniform sub-pitch offset -> transverse
        diffusion -> integration over the wire pitch -> the SP wire kernel, giving
        the centre / first-neighbour / beyond charge shares that should be measured.
Part C  what extra transverse width would be needed to reproduce the measured
        shares, and whether the SP filter can supply it.

Usage:  d42_wire_filter_toy.py [--nwires 3808]
"""
import argparse, sys
import numpy as np

SQRTPI = np.sqrt(np.pi)

# (label, Wire_ind X, Wire_col X, pitch_ind mm, pitch_col mm, DT cm^2/s, drift speed mm/us,
#  median |x| cm, ind_sigma_u/v mm, col_sigma mm)
DETS = [
    dict(name="PDVD", Xind=5.0, Xcol=10.0, pind=7.65, pcol=5.10, DT=7.9135, v=1.48073,
         absx=219.0, ind_u=0.259, ind_v=0.4316, col=0.0575),
    dict(name="SBND", Xind=1.05, Xcol=3.60, pind=3.00, pcol=3.00, DT=8.8, v=1.563,
         absx=133.0, ind_u=0.48359, ind_v=0.80599, col=0.09403),
]


def filter_waveform(nbins, sigma, power=2.0, max_freq=1.0, use_negative_freqs=True):
    """Byte-for-byte the array SigProc::HfFilter::filter_waveform(nbins) returns."""
    ff = 2.0 if use_negative_freqs else 1.0
    i = np.arange(nbins)
    freq = i / float(nbins) * ff * max_freq
    freq = np.where(freq > max_freq, freq - 2 * max_freq, freq)
    return np.exp(-0.5 * np.abs(freq / sigma) ** power)


def kernel_of(nbins, X):
    """Real-space wire kernel = inverse DFT of the filter, normalised to unit sum."""
    sigma = 1.0 / SQRTPI * X
    H = filter_waveform(nbins, sigma)
    h = np.real(np.fft.ifft(H))
    h = np.roll(h, nbins // 2)                 # centre it
    x = np.arange(nbins) - nbins // 2          # wire offset
    return x, h / h.sum(), sigma, H


def rms_of(x, w):
    w = np.asarray(w, float)
    if w.sum() <= 0:
        return float("nan")
    mu = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - mu) ** 2, weights=w)))


def bin_gauss(sigma_pitch, offset, nmax=6):
    """Charge fraction per wire for a Gaussian of width sigma (in pitch units)
    centred `offset` pitches from wire 0's centre, integrated over each wire bin
    -- the same erf integral as cal_gaus_integral."""
    from math import erf
    n = np.arange(-nmax, nmax + 1)
    lo = (n - 0.5 - offset) / (np.sqrt(2) * sigma_pitch)
    hi = (n + 0.5 - offset) / (np.sqrt(2) * sigma_pitch)
    v = 0.5 * (np.vectorize(erf)(hi) - np.vectorize(erf)(lo))
    return n, v


def shares(sigma_pitch, kern=None, nu=201, nmax=12, nsigma=None):
    """Centre / first-neighbour / beyond charge shares, averaged over a uniform
    sub-pitch track position.  Offsets run over [-0.5, 0.5] pitch, so wire 0 is
    always the wire the track crosses -- the same 'centre' the data table uses."""
    tot = np.zeros(2 * nmax + 1)
    for off in np.linspace(-0.5, 0.5, nu):
        n, v = bin_gauss(sigma_pitch, off, nmax)
        if nsigma is not None:
            # cal_gaus_integral: |wbin - w_center| <= nsigma * w_sigma, else exactly 0
            v = np.where(np.abs(n - off) <= nsigma * sigma_pitch, v, 0.0)
        if kern is not None:
            v = np.convolve(v, kern, mode="same")
        tot += v
    tot /= tot.sum()
    c = nmax
    return tot[c], tot[c - 1] + tot[c + 1], tot.sum() - tot[c] - tot[c - 1] - tot[c + 1]


def make_fig(kern, N, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))
    f = np.linspace(0, 1.6, 400)
    styles = {("PDVD", "ind"): ("tab:red", "-"), ("PDVD", "col"): ("tab:red", "--"),
              ("SBND", "ind"): ("tab:blue", "-"), ("SBND", "col"): ("tab:blue", "--")}
    labels, closed, true = [], [], []
    for D in DETS:
        for pl, X in (("ind", D["Xind"]), ("col", D["Xcol"])):
            c, ls = styles[(D["name"], pl)]
            sigma = X / SQRTPI
            ax[0].plot(f, np.exp(-0.5 * (f / sigma) ** 2), ls, color=c,
                       label="%s %s  (X=%.2f)" % (D["name"], pl, X))
            x, h = kern[(D["name"], pl)]
            m = np.abs(x) <= 5
            ax[1].plot(x[m], h[m], ls, color=c, marker="o", ms=4)
            labels.append("%s\n%s" % (D["name"], pl))
            closed.append(1.0 / (SQRTPI * X)); true.append(rms_of(x[np.abs(x) <= 12], h[np.abs(x) <= 12]))
    ax[0].axvspan(1, 1.6, color="0.85", zorder=0)
    ax[0].text(1.02, 0.5, "beyond Nyquist:\nnever evaluated", fontsize=7, va="center")
    ax[0].axvline(1, color="k", lw=.8)
    ax[0].set_xlabel("wire frequency  [Nyquist = 1]"); ax[0].set_ylabel("filter")
    ax[0].set_title("only SBND's induction filter actually filters", fontsize=9)
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].set_xlabel("wire offset"); ax[1].set_ylabel("kernel weight")
    ax[1].set_title("its real-space kernel: a near-delta everywhere else", fontsize=9)
    ax[1].grid(alpha=.3); ax[1].axhline(0, color="k", lw=.5)
    i = np.arange(len(labels))
    ax[2].bar(i - 0.2, closed, 0.4, label="closed form 1/(√π·X)", color="0.6")
    ax[2].bar(i + 0.2, true, 0.4, label="true kernel rms", color="tab:green")
    ax[2].set_xticks(i); ax[2].set_xticklabels(labels, fontsize=8)
    ax[2].set_ylabel("wires"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3, axis="y")
    ax[2].set_title("the closed form overstates the filter\neverywhere but SBND induction", fontsize=9)
    fig.suptitle("The SP wire filter the track fit's ind_sigma/col_sigma are derived from", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(out, dpi=135); plt.close(fig)
    print("  -> %s" % out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwires", type=int, default=3808)
    ap.add_argument("--rings", nargs="*", help="d42_ring_frame.py --tsv outputs (for part C)")
    ap.add_argument("--fig", help="write the filter figure here (PNG)")
    a = ap.parse_args()
    N = a.nwires

    print("=" * 78)
    print("PART A -- what the SP wire filter really does")
    print("=" * 78)
    print("  filter is exp(-0.5*(f/sigma)^2), f in NYQUIST units, sigma = X/sqrt(pi).")
    print("  the closed form the configs use, sigma_x = 1/(sqrt(pi)*X) wires, is the")
    print("  CONTINUUM inverse transform: it is only valid while the frequency-domain")
    print("  Gaussian dies well inside the Nyquist band.\n")
    print("  %-10s %-6s %8s %10s %11s %11s %9s" % (
        "detector", "plane", "X", "sigma_f", "atten@Nyq", "closed form", "true rms"))
    kern = {}
    for D in DETS:
        for pl, X in (("ind", D["Xind"]), ("col", D["Xcol"])):
            x, h, sigma, H = kernel_of(N, X)
            kern[(D["name"], pl)] = (x, h)
            m = np.abs(x) <= 12
            print("  %-10s %-6s %8.2f %10.3f %11.3f %11.4f %9.4f" % (
                D["name"], pl, X, sigma, np.exp(-0.5 / sigma ** 2),
                1.0 / (SQRTPI * X), rms_of(x[m], h[m])))
    if a.fig:
        make_fig(kern, N, a.fig)
    print("\n  central-wire weight of the true kernel, and its first neighbour:")
    for k, (x, h) in kern.items():
        c = np.argmin(np.abs(x))
        print("    %-5s %-4s  h[0] = %7.4f   h[+-1] = %+7.4f   h[+-2] = %+7.4f"
              % (k[0], k[1], h[c], h[c + 1], h[c + 2]))

    print()
    print("=" * 78)
    print("PART B -- forward toy: diffusion -> wire binning -> SP filter")
    print("=" * 78)
    print("  charge shares a straight track should leave, averaged over its sub-pitch position")
    print("  %-6s %-6s %8s %8s %9s   %s" % ("det", "plane", "sig_dif", "sig_mod", "", "centre / 1st nbr / beyond"))
    for D in DETS:
        t_us = D["absx"] / (D["v"] / 10.0)                     # cm / (cm/us)
        sig_diff_mm = np.sqrt(2 * D["DT"] * 1e-7 * t_us * 1e3)  # mm  (cm^2/s = 1e-7 mm^2/ns)
        for pl, ind, pitch, X in (("U", D["ind_u"], D["pind"], D["Xind"]),
                                  ("V", D["ind_v"], D["pind"], D["Xind"]),
                                  ("W", D["col"], D["pcol"], D["Xcol"])):
            sig_mod = np.hypot(sig_diff_mm, ind) / pitch        # exactly TrackFitting
            x, h = kern[(D["name"], "ind" if pl != "W" else "col")]
            m = np.abs(x) <= 4
            k = h[m] / h[m].sum()
            c, n1, be = shares(sig_mod)                          # what the FIT predicts
            ck, nk, bk = shares(sig_diff_mm / pitch, kern=k)     # diffusion + the REAL filter
            print("  %-6s %-6s %8.3f %8.3f   model:  %.2f / %.2f / %.2f      diffusion+true filter: %.2f / %.2f / %.2f"
                  % (D["name"], pl, sig_diff_mm / pitch, sig_mod, c, n1, be, ck, nk, bk))

    print()
    print("=" * 78)
    print("PART C -- what transverse width WOULD reproduce the measured shape")
    print("=" * 78)
    print("  shape statistic r = (first neighbour)/(centre + first neighbour), measured and")
    print("  predicted in the same densified frame by d42_ring_frame.py --tsv.  r is used")
    print("  instead of the raw shares because it does not depend on how much charge a")
    print("  fused cluster puts beyond 1.5 pitches.")
    rings = {}
    for f in (a.rings or []):
        for ln in open(f):
            w = ln.split()
            if len(w) >= 8 and w[0] != "det":
                rings[(w[0].upper(), w[1])] = (float(w[6]), float(w[7]))
    if not rings:
        print("  (no --rings given; skipped)")
        return
    grid = np.linspace(0.05, 1.5, 400)
    rvals = np.array([shares(s)[1] / (shares(s)[0] + shares(s)[1]) for s in grid])
    print("  %-6s %-6s %8s %8s %9s %9s %11s %11s" % (
        "det", "plane", "r_meas", "r_pred", "sig_mod", "sig_toy", "sig_needed", "missing[mm]"))
    for D in DETS:
        t_us = D["absx"] / (D["v"] / 10.0)
        sig_diff_mm = np.sqrt(2 * D["DT"] * 1e-7 * t_us * 1e3)
        for pl, ind, pitch in (("U", D["ind_u"], D["pind"]), ("V", D["ind_v"], D["pind"]),
                               ("W", D["col"], D["pcol"])):
            key = (D["name"], pl)
            if key not in rings:
                continue
            rm, rp = rings[key]
            sig_mod = np.hypot(sig_diff_mm, ind) / pitch
            c0, n0, _ = shares(sig_mod)
            r_toy = n0 / (c0 + n0)
            s_need = float(np.interp(rm, rvals, grid))
            missing = np.sqrt(max((s_need * pitch) ** 2 - (sig_mod * pitch) ** 2, 0))
            print("  %-6s %-6s %8.3f %8.3f %9.3f %9.3f %11.3f %11.2f" % (
                D["name"], pl, rm, rp, sig_mod, r_toy, s_need, missing))
    print()
    print("  CAVEAT: sig_toy is the toy's r for the model's own sigma, and it does NOT")
    print("  reproduce r_pred closely (0.164 vs 0.203 on PDVD U, 0.345 vs 0.255 on SBND U).")
    print("  The toy is single-drift, normal-incidence and segment-free, so its r(sigma)")
    print("  inversion is ORIENTATION ONLY -- it says the model is too narrow and roughly")
    print("  by how much, and nothing finer.  The quantitative missing width comes from")
    print("  d42_transverse_moments.py, which measures the per-point width of the measured")
    print("  and predicted profiles about their OWN centroids on the same cells, so the bin")
    print("  width, the track's extent within a slice and the drift spread all cancel.")

    print()
    print("=" * 78)
    print("PART D -- two approximations inside the model itself")
    print("=" * 78)
    print("  (i) cal_gaus_integral applies a HARD window: a wire bin contributes only if")
    print("      |wbin - w_center| <= nsigma * w_sigma, with nsigma = 4 at all nine call")
    print("      sites.  That is harmless while 4*sigma > 1 wire and amputates the first")
    print("      neighbour when it is not.")
    print("  %-6s %-6s %9s %9s %11s %11s" % ("det", "plane", "sig_mod", "4*sigma", "r no cut", "r nsigma=4"))
    for D in DETS:
        t_us = D["absx"] / (D["v"] / 10.0)
        sig_diff_mm = np.sqrt(2 * D["DT"] * 1e-7 * t_us * 1e3)
        for pl, ind, pitch in (("U", D["ind_u"], D["pind"]), ("V", D["ind_v"], D["pind"]),
                               ("W", D["col"], D["pcol"])):
            sig_mod = np.hypot(sig_diff_mm, ind) / pitch
            c0, n0, _ = shares(sig_mod)
            c1, n1, _ = shares(sig_mod, nsigma=4.0)
            print("  %-6s %-6s %9.3f %9.3f %11.3f %11.3f%s" % (
                D["name"], pl, sig_mod, 4 * sig_mod, n0 / (c0 + n0), n1 / (c1 + n1),
                "   <-- cut bites" if 4 * sig_mod < 1.5 else ""))
    print("  (ii) all nine call sites pass flag = 0, the pure-Gaussian branch.  The")
    print("       flag = 1 'induction plane with bipolar response' and flag = 2 branches")
    print("       exist in cal_gaus_integral and are never reached, so U and V are")
    print("       modelled with exactly the same transverse shape as the collection plane.")


if __name__ == "__main__":
    sys.exit(main())
